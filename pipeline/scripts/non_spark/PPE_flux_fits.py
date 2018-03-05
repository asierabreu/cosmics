import argparse
import os
import numpy as np
from astropy.io import fits
import datetime

CATALOG_VERSION = 'v1.0'

def OBMT_apyTime(obmt_in):
    """Assign a given OBMT to astropy time objects"""
    from astropy.time import Time
    obmt_reset = 10454403208162998
    
    if type(obmt_in) == np.ndarray:
        unix_ref = np.zeros(obmt_in.shape)
        obmt_ref = np.zeros(obmt_in.shape)
        
        unix_ref[obmt_in >= obmt_reset] = 1393445605
        obmt_ref[obmt_in >= obmt_reset] = obmt_reset
        
        unix_ref[obmt_in < obmt_reset] = 1388534400.0
        obmt_ref[obmt_in < obmt_reset] = 5280428890394081
    else:
        if obmt_in >= obmt_reset:
            # reference time: UNIX and OBMT at 2014-02-26T20:13:25 UTC
            unix_ref = 1393445605
            obmt_ref = obmt_reset
        else:
            # reference time: UNIX and OBMT at 2014-01-01T00:00:00 UTC
            unix_ref = 1388534400.0
            obmt_ref = 5280428890394081
        
    unix_out = unix_ref + (obmt_in - obmt_ref)/1e9
    
    out = Time(unix_out, format='unix',scale='utc')
    
    return out

def load_textfile_PPE(filename):
    
    # get the times and nr of particles
    indat = np.loadtxt(filename,skiprows=1,usecols=(0,4))
    
    # sort by obmt
    indat = indat[indat[:,0].argsort()]
    
    # get the data
    obmt = indat[:,0]
    nparts = indat[:,1]
    
    exparea = 17.1 # cm^2 from Asier's TN
    exptime = 257.635 # s length of an ASD4 packet, from the TN
    errs = np.sqrt(nparts)/exparea/exptime
    fluxes = nparts/exparea/exptime
    
    apyt = OBMT_apyTime(obmt)
    t_mjd = apyt.mjd
    t_date = [t+'Z' for t in apyt.isot]
    
    # define the columns
    col1 = fits.Column(name='TIME', unit='MJD',
                       array=t_mjd, format='E')
    col2 = fits.Column(name='DATE', unit='UTC',
                       array=t_date, format='24A')
    col3 = fits.Column(name='COUNTS',
                       array=nparts, format='J')
    col4 = fits.Column(name='FLUX', unit='particles/cm^2/s',
                       array=fluxes, format='E')
    col5 = fits.Column(name='FLUX_ERR', unit='particles/cm^2/s',
                       array=errs, format='E')
    
    return fits.BinTableHDU.from_columns([col1,col2,col3,col4,col5])



if __name__ == "__main__":
    # input parsing
    parser = argparse.ArgumentParser(description='Extract PPE fluxes from the ASD4 folder')
    parser.add_argument('-i', '--inputRoot',help='Directory containing the ASD4_Counters files')
    parser.add_argument('-o', '--outputFile',help='Output file')
    args = parser.parse_args()

    outfile = os.path.abspath(args.outputFile)
    inroot = os.path.abspath(args.inputRoot)

    PPElist = [fits.PrimaryHDU()]

    for row in range(1,8):
        for fov in range(1,3):
            # load the file into a hdu
            infile = inroot+"/ASD4_Counters_FOV{}_ROW{}.dat".format(fov,row)
            newhdu = load_textfile_PPE(infile)
              
            # edit the header
            newhdu.name = "ROW_{}FOV_{}".format(row,fov)
            newhdu.header["SOURCE"] = "SM-PPE"
            newhdu.header.comments["SOURCE"] = "Observation type"
            newhdu.header["CCD_ROW"] = row
            newhdu.header.comments["CCD_ROW"] = "CCD row"
            newhdu.header["FOV"] = fov
            newhdu.header.comments["FOV"] = "Field of View"
            
            PPElist.append(newhdu)


    outlist = fits.HDUList(PPElist)
    # add header keys
    outlist[0].header["GEN_TIME"] = datetime.datetime.utcnow().isoformat()+"Z"
    outlist[0].header.comments["GEN_TIME"] = "Generation Time [UTC]"

    outlist[0].header["ORIGIN"] = "Generated at ESAC (CALTEAM)"
    outlist[0].header.comments["ORIGIN"] = "Origin of data"

    outlist[0].header["VERSION"] = CATALOG_VERSION
    outlist[0].header.comments["VERSION"] = "Catalog Version"
    
    outlist.writeto(outfile)