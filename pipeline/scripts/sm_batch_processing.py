"""
Spark parallelization of the extraction of TrackObs from the SM-SIF data
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
    
def process(datepathlist, calibdat, root_dir):
    
    """
    Processes a list of sm files given in the datepathlist ('yyyy-mm-dd', [list of absolute sm file paths]).
    Uses the calibration data given in calibdat for bias, readnoise and gain.
    Writes one output file into the root directory structure, creating folders if necessary.
    """
    root=os.environ['HOME']
    sys.path.append(root+'/spark/libs/SM_Processing/')
    sys.path.append(root+'/spark/libs/astroscrappy/')
    sys.path.append(root+'/spark/libs/TrackObs/')

    import time 
    import sm_process_funcs as pfuncs
    
    # get the date and the files
    datestring = datepathlist[0]
    filepaths = datepathlist[1]
    
    ### TrackObs extraction
    obslist = []

    t0= time.time()    

    for p in filepaths:
        
        # get the image
        image, fov, row, acqTime = pfuncs.sm_image_data(p)
        
        # bias, gain and readnoise
        ical = 2*(row-1) + fov -1    # index of the row of interest
        bias = calibdat["BIAS"][ical][7:] # ignore the pre-scan column
        gain = calibdat["GAIN"][ical]
        readnoise = calibdat["RNOISE"][ical]
        
        # make the starmask - with an exception for row 4, fov 1
        if (fov==1 and row==4):
            starmask = pfuncs.sm_starmask(image, 2000, badcol=850)
        else:
            starmask = pfuncs.sm_starmask(image, 2000)
            
        obs = pfuncs.sm_cosmics(image, gain, bias, readnoise, starmask, sigclip=10., sigfrac=0.1, objlim=25.)
        obs.acqTime = acqTime
        obs.row = row
        obs.fov = fov
    
        obslist.append(obs)
      
    t1=time.time()

    print('extraction %02f' %(t1-t0))
  
    ### writing to a file
    # path to write to
    outpath = os.path.abspath(root_dir) + "/{}/{}/{}".format(datestring[:4], datestring[5:7], datestring[8:])
    # make the path, if necessary 
    os.makedirs(outpath,exist_ok=True)
    
    # the filename
    starttime = obslist[0].acqTime    
    outname = "SM-SIF_OBMT_START_{}.fits".format(int(starttime))
    print('writing file : %s' %(outpath+'/'+outname))
    pfuncs.write_Obslist(obslist, outpath+'/'+outname)
    t2=time.time()
    print('writing %02f' %(t2-t1))
    
    
        
if __name__ == "__main__":
    
    
    # input parsing
    parser = argparse.ArgumentParser(description='Process SM-SIF fits files in Spark framework')
    parser.add_argument('-i', '--inputFile',help='input file containing absolute file paths')
    parser.add_argument('-o', '--outputRoot',help='output files root directory')
    parser.add_argument('-c', '--calibFile',help='fits file containing the calibration data for each chip')
    parser.add_argument('-n', '--numCores',help='number of cores (to create rdd)')
    args = parser.parse_args()

    
    appName = "SMBatchProcessor"
    conf = SparkConf().setAppName(appName)
    sc = SparkContext(conf=conf)
        
    calibfile = args.calibFile
    pathfile = args.inputFile
    outroot  = os.path.abspath(args.outputRoot)
    ncores = int(args.numCores)
    
    
    # get the calibration data
    hducal = fits.open(calibfile)
    calib = hducal[1].data
    hducal.close()
    # the calibration data should be a broadcast variable
    calibdat = sc.broadcast(calib)
    
    
    # create RDD from text file
    files = sc.textFile(pathfile,minPartitions=ncores)
    # get the dates and group the paths by the dates
    grouped_files = files.groupBy(lambda x: getday(x))

    # get the number of days
    ndays = grouped_files.count()
    
    # for optimum use of cores, repartition grouped_files
    grouped_files.repartition(min(ndays, 4*ncores))
    
    
    # then an action, running process on the grouped files
    grouped_files.foreach(lambda x: process(x,calibdat.value,outroot))
    sc.stop()
