"""
Spark parallelization of the extraction of cosmic ray fluxes from CosmicObservation fits files
"""
from __future__ import print_function

import os
import sys
import numpy as np
import pandas as pd
import argparse

from pyspark import SparkContext, SparkConf
from pyspark.sql import SparkSession

# function to extract the cosmic ray flux
def fits_flux_chip(filename):
    """
    Extract fluxes from a file of CosmicObservations
    Returns lists of acqTimes [OBMT and UTC] the number of particles, fluxes [parts/cm^2/s] and the poissonian uncertainty, as well as a tuple of (source,row,fov)
    """
    from astropy.io import fits
    from astropy.time import Time
    # Create lists for OBMT and fluxes
    acqTimes = []
    acqDates = []
    nparts = []
    fluxes = []
    fluxes_err = []
    chiplocs = []
    
    # Open the file
    hdulist = fits.open(filename)

    # Iterate over all the CosmicObservations
    for ii in range(1,len(hdulist),1):
        
        head = hdulist[ii].header        
        
        nparts.append(head["NAXIS2"])
        acqTimes.append(Time(head["ACQDATE"],scale='utc',format='isot').mjd)
        acqDates.append(head["ACQDATE"])
        fluxes.append(head["FLUX"])
        fluxes_err.append(head["FLUX_ERR"])
        chiplocs.append((head["SOURCE"], head["CCD_ROW"], head["FOV"]))

    # Close the file
    hdulist.close()
    
    return list(zip(acqTimes, acqDates, nparts, fluxes, fluxes_err, chiplocs))



if __name__ == "__main__":

    # input parsing
    parser = argparse.ArgumentParser(description='Extract fluxes from CosmicObservation files in a Spark framework')
    parser.add_argument('-i','--inputPath',help='root of directory to scan - will look for <inputPath>/yyyy/mm/dd/*.fits')
    parser.add_argument('-o','--outputFolder',help='folder to save the output files to') 
    parser.add_argument('-n','--numTasks',help='number of tasks (for making the rdd)')
    args = parser.parse_args()

    appName  = "CosmicFluxExtractor"
    conf     = SparkConf().setAppName(appName)
    sc       = SparkContext(conf=conf)


    inroot   = os.path.abspath(args.inputPath)
    outfolder   = os.path.abspath(args.outputFolder)
    ntasks = int(args.numTasks)


    # create RDD of input files
    import glob
    infiles = glob.glob(inroot+"/20**/*/*/*.fits")

    rdd   = sc.parallelize(infiles,ntasks)

    # (1) For each file, collect the OBMTs of all the TrackObs in a flat map
    # (2) Group by SOURCE, ROW and FOV
    # (3) Collect - these are not many values anyway

    grouped  = rdd \
            .flatMap(lambda x: fits_flux_chip(x)) \
            .groupBy(lambda x: x[-1]) \
            .collect()

    from astropy.io import fits

    # get the catalog version
    with fits.open(infiles[0]) as hdulist:
        CATALOG_VERSION = hdulist[0].header["VERSION"]
    
    # just directly write the files here
    # if it does not exist, make the path
    os.makedirs(outfolder,exist_ok=True)

    # turn each group into an HDU
    # make a dictionary of {source:[list of hdus for that source]}
    sources = set(map(lambda x:x[0][0],grouped))
    hdudict = {s:[] for s in sources}
    
    for chip in grouped:
        source, row, fov = chip[0]
        data = [r[:-1] for r in chip[1]]
        # sort by obmt here
        data.sort(key = lambda x: x[0])
        # define the columns
        col1 = fits.Column(name='TIME', unit='MJD',
                           array=[d[0] for d in data], format='E')
        col2 = fits.Column(name='DATE', unit='UTC',
                           array=[d[1] for d in data], format='24A')
        col3 = fits.Column(name='COUNTS',
                           array=[d[2] for d in data], format='J')
        col4 = fits.Column(name='FLUX', unit='particles/cm^2/s',
                           array=[d[3] for d in data], format='E')
        col5 = fits.Column(name='FLUX_ERR', unit='particles/cm^2/s',
                           array=[d[4] for d in data], format='E')
        
        # hdu from cols
        tbhdu = fits.BinTableHDU.from_columns([col1,col2,col3,col4,col5])
        
        # header keys
        # define the extension name
        tbhdu.name = 'ROW{}_FOV{}'.format(row,fov)
        # SOURCE, ROW, FOV, what else?
        tbhdu.header["SOURCE"] = source
        tbhdu.header.comments["SOURCE"] = "Observation type"
        tbhdu.header["CCD_ROW"] = row
        tbhdu.header.comments["CCD_ROW"] = "CCD row"
        tbhdu.header["FOV"] = fov
        tbhdu.header.comments["FOV"] = "Field of View"
        
        hdudict[source].append(tbhdu)
    
    # move allhdus into the respective fits files, sorting them before
    for source in sources:
        # construct the filename
        outfile = outfolder+"/FLUX_{}.fits".format(source)
        
        # sort the hdulist for this source by (row,fov)
        hdudict[source].sort(key = lambda x:(x.header["CCD_ROW"],x.header["FOV"]))
        
        # create an hdulist
        outlist = fits.HDUList()
        
        # add my observations
        for hdu in hdudict[source]:
            outlist.append(hdu)
        
        # add some meta-information to the primary
        import datetime
        outlist[0].header["GEN_TIME"] = datetime.datetime.utcnow().isoformat()+"Z"
        outlist[0].header.comments["GEN_TIME"] = "Generation Time [UTC]"

        outlist[0].header["ORIGIN"] = "Generated at ESAC (CALTEAM)"
        outlist[0].header.comments["ORIGIN"] = "Origin of data"

        outlist[0].header["VERSION"] = CATALOG_VERSION
        outlist[0].header.comments["VERSION"] = "Catalog Version"
        
        # write the file
        outlist.writeto(outfile)
        outlist.close()
        
    sc.stop()
