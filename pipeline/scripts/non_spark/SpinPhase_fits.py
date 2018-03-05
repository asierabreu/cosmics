import argparse
import os
import re
import numpy as np
from astropy.io import fits
import datetime

CATALOG_VERSION = 'v1.0'

if __name__ == "__main__":
    
    # input parsing
    parser = argparse.ArgumentParser(description='Extract Spacecraft spinphase into a FITS file')
    parser.add_argument('-i', '--inFile',help='Spinphase textfile')
    parser.add_argument('-o', '--outputFile',help='Output file')
    args = parser.parse_args()

    outfile = os.path.abspath(args.outputFile)
    spinfile = os.path.abspath(args.inFile)

    # read the raw lines from the textfile
    with open(spinfile,'r') as f:
        data = f.readlines()[14:]

    # write everything into two lists
    tspin = []
    aspin = []
    for ii in range(len(data)):
        # time and angle are separated by a tab
        t,a = re.split('\t',data[ii].strip())
        tspin.append(t.replace(' ','T'))
        aspin.append(float(a))

    del data
    
    from astropy.time import Time
    tspin = Time(list(tspin),format='isot').mjd
    aspin = np.array(aspin) * 180/np.pi

    spinhdu = fits.BinTableHDU.from_columns([fits.Column(name='TIME', unit='MJD', format='1E', array=tspin),
                                            fits.Column(name='ANGLE', unit='deg', format='1E', array=aspin)])
    spinhdu.name = "SC_ANGLE"

    
    outlist = fits.HDUList([fits.PrimaryHDU(),spinhdu])
    
    # add header keys
    outlist[0].header["GEN_TIME"] = datetime.datetime.utcnow().isoformat()+"Z"
    outlist[0].header.comments["GEN_TIME"] = "Generation Time [UTC]"

    outlist[0].header["ORIGIN"] = "Generated at ESAC (CALTEAM)"
    outlist[0].header.comments["ORIGIN"] = "Origin of data"

    outlist[0].header["VERSION"] = CATALOG_VERSION
    outlist[0].header.comments["VERSION"] = "Catalog Version"
    
    outlist.writeto(outfile)