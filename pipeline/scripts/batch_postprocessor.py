"""
Spark parallelization of the extraction of CosimcObservations from TrackObs data files
"""

import os
import sys
import numpy as np
import argparse

from pyspark import SparkContext, SparkConf
from pyspark.sql import SparkSession

import re
#getday = lambda x: (re.search("[0-9-]+(?!SIF_PROCESSING)",x).group(0))

from astropy.io import fits

def process(trackobs_path, root_dir):
    
    """
    Processes a TrackObs fits file into a CosmicObservation
    Writes one output file into the root directory structure, creating folders if necessary.
    """
    root=os.environ['HOME']
    sys.path.append(root+'/spark/libs/TrackObs/')
    sys.path.append(root+'/spark/libs/CosmicObservation/')

    import re
    import CosmicObservation as COb

    ### Processing
    # read the trackobs-list
    trobslist = COb.read_Obslist_fits(trackobs_path)

    # convert it into a list of CosmicObservations
    cobslist = []
    for ii in range(len(trobslist)):
        trobs = trobslist[ii]
        cobs = COb.CosmicObservation.from_TrackObs(trobs)
        cobslist.append(cobs)

    ### Writing
    # path to write to
    # get the yy/mm/dd string before the file
    date_dir = re.search("/[0-9]+/[0-1][0-9]/[0-3][0-9](?=/.+\.fits)",trackobs_path).group(0)
    outpath = os.path.abspath(root_dir) + date_dir
    # make the path, if necessary 
    os.makedirs(outpath,exist_ok=True)
    
    # the filename
    # outname is essentially a copy of the input path, just in a different directory
    outname = root_dir + re.search("/[0-9]+/[0-1][0-9]/[0-3][0-9]/.+\.fits",trackobs_path).group(0) 
    
    # run the write function
    COb.write_list_to_fits(cobslist,outname)
        
if __name__ == "__main__":
    
    
    # input parsing
    parser = argparse.ArgumentParser(description='Process TrackObs fits files to CosmicObservation fits files in Spark framework')
    parser.add_argument('-i', '--inputFile',help='input file containing absolute file paths')
    parser.add_argument('-o', '--outputRoot',help='output files root directory')
    parser.add_argument('-n', '--numCores',help='number of cores (to create rdd)')
    args = parser.parse_args()

    
    appName = "CosmicsPostProcessor"
    conf = SparkConf().setAppName(appName)
    sc = SparkContext(conf=conf)
        
    pathfile = args.inputFile
    outroot  = os.path.abspath(args.outputRoot)
    ncores = int(args.numCores)

    # create an RDD from the textfile
    # apply the process to each textfile

    # create RDD from text file
    files = sc.textFile(pathfile,4*ncores)
    # then an action, running process on the grouped files
    files.foreach(lambda x: process(x,outroot))
    sc.stop()
