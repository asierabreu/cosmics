"""
A spark parallelization of the BAM-OBS TrackObs extraction
"""
from __future__ import print_function

import os
import sys
import numpy as np
import pandas as pd
import argparse

from pyspark import SparkContext, SparkConf
from pyspark.sql import SparkSession

import matplotlib.pyplot as plt
import matplotlib.cm as cm

def parseVector(line):
    return np.array([float(x) for x in line.split(',')])

### Determine the OBMT of the first observation in a gbin
def get_obmt_gbin(path):
    root=os.environ['HOME']
    sys.path.append(root+'/spark/libs/PythonGbinReader/GbinReader/')
    import gbin_reader

    reader = gbin_reader.GbinReader(path)
    obs = reader.__next__()
    acqTime = obs.acqTime
    reader.close()

    return acqTime


def process(pathgroup,sorted_paths,write_grouping,output):
    root=os.environ['HOME']
    sys.path.append(root+'/spark/libs/BAM_Processing/')
    sys.path.append(root+'/spark/libs/astroscrappy/')
    sys.path.append(root+'/spark/libs/TrackObs/')
    sys.path.append(root+'/spark/libs/PythonGbinReader/GbinReader/')
    import process_bam as unit
    unit.process_BAM_OBS(pathgroup,sorted_paths,write_grouping,output)


if __name__ == "__main__":

    # input parsing
    parser = argparse.ArgumentParser(description='Process BamObservations in Spark framework')
    parser.add_argument('-i','--inputFile',help='input file containing MDB paths')
    parser.add_argument('-p','--previous',help='MDB path of last file handled in the previous step', default='')
    parser.add_argument('-o','--outputRoot',help='output files root directory') 
    parser.add_argument('-f','--fileGroup',help='number of gbin files per executor (excluding overlaps)')
    parser.add_argument('-w','--writeGroup',help='maximum number of TrackObs to be written in each output file (less than 512)',default=500)
    parser.add_argument('-n','--numSorters',help='number of tasks for sorting the files by time (ideally the number of cores)')
    args = parser.parse_args()

    appName  = "BamBatchProcessor"
    conf     = SparkConf().setAppName(appName)
    sc       = SparkContext(conf=conf)


    root=os.environ['HOME']
    #sc.addPyFile(root+'/spark/libs/dependencies.zip')

    infile   = os.path.abspath(args.inputFile)
    outdir   = os.path.abspath(args.outputRoot)
    prevfile = args.previous
    file_grouping = int(args.fileGroup)
    write_grouping = int(args.writeGroup)
    numSorters = int(args.numSorters)


    # create accumulator ( nr. of BamObservations processed )
    #n = sc.accumulator(0)

    # create RDD from text file
    rdd   = sc.textFile(infile,numSorters)

    # (1) Create a key-value rdd of paths and obmt
    # (2) Sort the rdd and collect the paths
    # (3) Turn the sorted paths into a broadcast variable

    # get the sorted paths
    sorted_paths = rdd \
            .map(lambda x: (get_obmt_gbin(x),x)) \
            .sortByKey(ascending=True) \
            .map(lambda x: x[1]) \
            .collect()
    
    # broadcast them
    sorted_paths_bc = sc.broadcast(sorted_paths)
    
    # group the paths
    # the path groups will be a tuple ((start,stop),prev), indicating the indices of where to start and where to stop
    # and, in the case of the first group, the last used path
    pathgroups = []
    
    for ii in range(0,len(sorted_paths), file_grouping):
        istart = ii
        if ii+file_grouping >= len(sorted_paths):
            istop = len(sorted_paths)-1
        else:
            istop = ii+file_grouping-1
        if ii == 0:
            # the first group needs the previous file
            pathgroups.append(((istart,istop),prevfile))
        else:
            pathgroups.append(((istart,istop),''))

    group_rdd = sc.parallelize(pathgroups,len(pathgroups))

    group_rdd.foreach(lambda x: process(x,sorted_paths_bc.value,write_grouping, outdir))

    sc.stop()
