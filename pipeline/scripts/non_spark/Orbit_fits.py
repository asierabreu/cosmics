import argparse
import os
import numpy as np
from astropy.io import fits
import datetime

CATALOG_VERSION = 'v1.0'

def loadmjd(x):
    from astropy.time import Time
    return Time(str(x)[2:-1],format='isot').mjd

if __name__ == "__main__":
    
    # input parsing
    parser = argparse.ArgumentParser(description='Extract Geocentric Gaia Spacecraft coordinates into a FITS file')
    parser.add_argument('-eg', '--ephemGaia',help='Gaia Ephemeris textfile')
    parser.add_argument('-ee', '--ephemEarth',help='Earth Ephemeris textfile')
    parser.add_argument('-o', '--outputFile',help='Output file')
    args = parser.parse_args()

    outfile = os.path.abspath(args.outputFile)
    ephFile_gaia = os.path.abspath(args.ephemGaia)
    ephFile_earth = os.path.abspath(args.ephemEarth)
    
    
    # extract the values from the textfile
    ephgaia = np.transpose(np.loadtxt(ephFile_gaia,skiprows=1, converters={0:loadmjd}, usecols=(0,2,3,4)))
    ephearth = np.transpose(np.loadtxt(ephFile_earth,skiprows=1, converters={0:loadmjd}, usecols=(0,2,3,4)))
    
    # these values are in some species of EQUATORIAL coordinate system - they're inclined by about 23 deg wrt. the ecliptic
    # transform this into an ecliptic coordinate system - i.e. rotate z onto y

    ecliptang = -23.43694*np.pi/180

    xg = ephgaia[1,:]
    yg = ephgaia[2,:]*np.cos(ecliptang)  - ephgaia[3,:]*np.sin(ecliptang)
    zg = ephgaia[2,:]*np.sin(ecliptang)  + ephgaia[3,:]*np.cos(ecliptang)

    xe = ephearth[1,:]
    ye = ephearth[2,:]*np.cos(ecliptang)  - ephearth[3,:]*np.sin(ecliptang)
    ze = ephearth[2,:]*np.sin(ecliptang)  + ephearth[3,:]*np.cos(ecliptang)

    
    # get time, x, y, z in co-rotating earth centered system
    # time and z are easy

    T = (ephgaia)[0,:]

    z = zg-ze

    # x and y need to be extracted from the remaining rotating system of reference
    # to get out the rotation: first of all, a unit vector pointing at the earth
    rearth = np.sqrt(xe**2+ye**2)
    uearth = np.array([xe,ye])/rearth

    # x is then the scalar product of gaia's (x,y) with this
    vgaia = np.array([xg,yg])

    x = np.sum(uearth*vgaia, axis=0)-rearth

    # and y is the scalar product of the unit vector perpendicular to x
    # get the cross product such that for POSITIVE y, gaia is "ahead" of earth
    uearth_p = np.array([-uearth[1,:], uearth[0,:]])
    y = np.sum(uearth_p*vgaia, axis=0)
    
    
    # prepare the fits file
    
    outlist = [fits.PrimaryHDU()]
    
    # hdu from columns
    col1 = fits.Column(name='TIME', unit='MJD',
                       array=T, format='E')
    col2 = fits.Column(name='X', unit='m',
                       array=x, format='E')
    col3 = fits.Column(name='Y', unit='m',
                       array=y, format='E')
    col4 = fits.Column(name='Z', unit='m',
                       array=z, format='E')

    
    orbit_hdu = fits.BinTableHDU.from_columns([col1,col2,col3,col4])
    
    orbit_hdu.name = "SC_ORBIT"
    
    # add header keys
    
    # make the final file
    outlist.append(orbit_hdu)
    
    outlist = fits.HDUList(outlist)
    # add header keys
    outlist[0].header["GEN_TIME"] = datetime.datetime.utcnow().isoformat()+"Z"
    outlist[0].header.comments["GEN_TIME"] = "Generation Time [UTC]"

    outlist[0].header["ORIGIN"] = "Generated at ESAC (CALTEAM)"
    outlist[0].header.comments["ORIGIN"] = "Origin of data"

    outlist[0].header["VERSION"] = CATALOG_VERSION
    outlist[0].header.comments["VERSION"] = "Catalog Version"
    
    outlist.writeto(outfile)