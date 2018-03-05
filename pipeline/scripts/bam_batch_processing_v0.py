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


def process(pathgroup,write_grouping,output):
    root=os.environ['HOME']
    sys.path.append(root+'/spark/libs/BAM_Processing/')
    sys.path.append(root+'/spark/libs/astroscrappy/')
    sys.path.append(root+'/spark/libs/TrackObs/')
    sys.path.append(root+'/spark/libs/PythonGbinReader/GbinReader/')
    import process_bam as unit
    unit.process_BAM_OBS(pathgroup,write_grouping,output)


if __name__ == "__main__":

    # input parsing
    parser = argparse.ArgumentParser(description='Process BamObservations in Spark framework')
    parser.add_argument('-i','--inputFile',help='input file containing MDB paths')
    parser.add_argument('-o','--outputRoot',help='output files root directory') 
    parser.add_argument('-f','--fileGroup',help='number of gbin files per executor (excluding overlaps)')
    parser.add_argument('-w','--writeGroup',help='maximum number of TrackObs to be written in each output file (less than 512)')
    args = parser.parse_args()

    appName  = "BamBatchProcessor"
    conf     = SparkConf().setAppName(appName)
    sc       = SparkContext(conf=conf)


    root=os.environ['HOME']
    #sc.addPyFile(root+'/spark/libs/dependencies.zip')

    infile   = os.path.abspath(args.inputFile)
    outdir   = os.path.abspath(args.outputRoot)
    file_grouping = int(args.fileGroup)
    write_grouping = int(args.writeGroup)


    # create accumulator ( nr. of BamObservations processed )
    #n = sc.accumulator(0)

    # create RDD from text file
    rdd   = sc.textFile(infile,288)

    # (1) Create a key-value rdd of paths and obmt
    # (2) Sort the rdd and collect the paths
    # (3) Group the paths, adding a single overlap
    # (4) Process each path group via a foreach

    # get the sorted paths
    sorted_paths = rdd \
            .map(lambda x: (get_obmt_gbin(x),x)) \
            .sortByKey(ascending=True) \
            .map(lambda x: x[1]) \
            .collect()
    
    # group the paths
    # the path groups will be a tuple ([list of paths], path for overlap)
    pathgroups = []
    for ii in range(0,len(sorted_paths), file_grouping):
        p = sorted_paths[ii:ii+file_grouping]
        if ii+file_grouping < len(sorted_paths):
            pathgroups.append((p,sorted_paths[ii+file_grouping]))
        else:
            # this is the last group, the overlap is an empty string
            pathgroups.append((p,''))

    group_rdd = sc.parallelize(pathgroups,len(pathgroups))

    #group_rdd.foreach(lambda x: print(len(x[0])))
    group_rdd.foreach(lambda x: process(x,write_grouping, outdir))

    # this should give the nr of executors?
    #print("---------", sc._jsc.sc().getExecutorMemoryStatus().size())
    sc.stop()
