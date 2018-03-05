"""
Spark parallelization of the extraction of TrackObs from the BAM-SIF data
"""

import os
import sys
import numpy as np
import argparse

from pyspark import SparkContext, SparkConf
from pyspark.sql import SparkSession

import re as regex
getday = lambda x: (regex.search("[0-9-]+(?!SIF_PROCESSING)",x).group(0))

from astropy.io import fits
    
def process(datepathlist, root_dir):
    
    """
    Processes a list of BAM-SIF files given in the datepathlist ('yyyy-mm-dd', [list of absolute sm file paths]).
    Uses the calibration data given in calibdat for bias, readnoise and gain.
    Writes one output file into the root directory structure, creating folders if necessary.
    """
    root=os.environ['HOME']
    sys.path.append(root+'/spark/libs/BAM-SIF_Processing/')
    sys.path.append(root+'/spark/libs/BAM_Processing/')
    sys.path.append(root+'/spark/libs/PythonGbinReader/GbinReader')
    sys.path.append(root+'/spark/libs/astroscrappy/')
    sys.path.append(root+'/spark/libs/TrackObs/')

    import bamsif_process_funcs as pfuncs
    
    # get the date and the files
    datestring = datepathlist[0]
    filepaths = datepathlist[1]
    
    ### TrackObs extraction
    obslist = []

    N_AL = 721
    gain=3.853           # gain [e-/ADU]
    readnoise=8.365785291814616 # readout noise [e-]
    bias=2576            # bias [ADU]

    for p in filepaths:
        # read in the data
        source, acqTime = pfuncs.bam_sif_read(p,N_AL,gain,bias)

        # signal, error and background
        signal, err_sig, background = pfuncs.bam_sif_signal(source,readnoise)
        
        # the cosmics
        obs1, obs2 = pfuncs.bam_sif_cosmics(signal, err_sig, threshold=5, threshfrac=0.5, N_mask=0, gain=gain, acqTime=acqTime)
        # for now, we'll save everything in one file
        obslist.append(obs1)
        obslist.append(obs2)
      
  
    ### writing to a file
    # path to write to
    outpath = os.path.abspath(root_dir) + "/{}/{}/{}".format(datestring[:4], datestring[5:7], datestring[8:])
    # make the path, if necessary 
    os.makedirs(outpath,exist_ok=True)
    
    # the filename
    starttime = obslist[0].acqTime    
    outname = "BAM-SIF_OBMT_START_{}.fits".format(int(starttime))
    pfuncs.write_Obslist(obslist, outpath+'/'+outname)
    
    
        
if __name__ == "__main__":
    
    
    # input parsing
    parser = argparse.ArgumentParser(description='Process BAM-SIF fits files in Spark framework')
    parser.add_argument('-i', '--inputFile',help='input file containing absolute file paths')
    parser.add_argument('-o', '--outputRoot',help='output files root directory')
    parser.add_argument('-n', '--numCores',help='number of cores (to create rdd)')
    args = parser.parse_args()

    
    appName = "BAM-SIFBatchProcessor"
    conf = SparkConf().setAppName(appName)
    sc = SparkContext(conf=conf)
        
    pathfile = args.inputFile
    outroot  = os.path.abspath(args.outputRoot)
    ncores = int(args.numCores)
    
    # create RDD from text file
    files = sc.textFile(pathfile,minPartitions=ncores)
    # get the dates and group the paths by the dates
    grouped_files = files.groupBy(lambda x: getday(x))

    # get the number of days
    ndays = grouped_files.count()
    
    # for optimum use of cores, repartition grouped_files
    grouped_files.repartition(min(ndays, 4*ncores))
    
    
    # then an action, running process on the grouped files
    grouped_files.foreach(lambda x: process(x,outroot))
    sc.stop()
