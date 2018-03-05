"""Class definition file for the CosmicObservation class, defining the output data model of the extraction and post-processing"""

import os
import sys
from TrackObs import *

import numpy as np
from astropy.io import fits

CATALOG_VERSION = 'v1.0'


##################
# Functions for the conversion of TrackObs into CosmicObservation and generating the final fits files

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

def extract_track(obs, ii):
    """
    Extract and reshape a track from TrackObs obs at index ii.
    Output track is in units of electrons
    """
    return np.reshape(obs.data[ii]["TRACK"],(obs.data[ii]["DIM_AL"],obs.data[ii]["DIM_AC"]))*obs.gain

def linefunc_m(pars, x):
    """
    pars is an array of [slope,y0], the equation being slope*x + y0
    """
    return pars[0]*x + pars[1]

def track_angle_linefit(intrack, pixratio=1):
    """
    Determine the angle of the track with respect to axis 0.
    In case of non-quadratic pixels, pixratio gives the ratio of pixel lengths in axis 1 to axis 0 
    (so in case of Gaia, AC/AL including binnning)
    """
    # the input track may have some rows/colums with zero values
    # remove those
    goodx = np.max(intrack,axis=1)>0
    goody = np.max(intrack,axis=0)>0
    track = intrack[np.where(goodx)[0],:]
    track = track[:,np.where(goody)[0]]
    
    dim0, dim1 = track.shape
    
    # We will form an x/y value pairs, with x being along axis 0, y being along axis 1
    # the values of the longer axis (i.e. more pixels) will just be the pixel coordinates
    # the values of the shorter axis will be the center of mass for each row/column
    
    from scipy.ndimage.measurements import center_of_mass
    
    if dim0 >= dim1:
        labels = np.zeros((dim0,dim1)) + np.transpose(np.array([range(dim0)]))
        cms = (center_of_mass(track, labels, range(dim0)))              
    else:
        labels = np.zeros((dim0,dim1)) + np.array([range(dim1)])
        cms = (center_of_mass(track, labels, range(dim1)))
        
    # set x and y from the centers of mass and subtract offset so the middle value is (0,0)
    x = (np.array([c[0] for c in cms]) - (dim0-1)/2)
    y = (np.array([c[1] for c in cms]) - (dim1-1)/2)* pixratio
    
    # Uncertainties
    dx = np.zeros(x.shape)+0.5
    dy = np.zeros(y.shape)+pixratio/2
    
    # Estimation of the angle and the length
    
    # We give special treatment to 1-dimensional tracks
    # We will also consider a dimension to be 1 if the difference between endpoints is very small.
    # This could be, for instance, because some additional noise has been picked up, 
    # which will be absorbed by the center of mass
    if dim1==1 or np.abs(y[-1]-y[0])<pixratio/2:
        ## Angle
        # 0 degrees
        theta = 0
        sd_theta = np.arctan(pixratio/dim0)*180/np.pi
        
        ## Length
        # the longest possible track goes between the most distant corners
        longlen = np.sqrt(dim0**2 + pixratio**2)
        # the shortest possible track goes between the nearest pixel borders
        shortlen = dim0-2
        # let's calculate the measured length as the mean of the two
        tracklen = (shortlen+longlen)/2
        sd_tracklen = np.abs(longlen-shortlen)/2
        
    elif dim0==1 or np.abs(x[-1]-x[0])<1/2:
        ## Angle
        # +-90
        # if x points in a particular direction, choose that one
        tsign = np.sign(x[-1]-x[0])
        if tsign == 0:
            tsign = 1
            
        theta = 90 * tsign
        sd_theta = np.abs(theta - np.arctan((dim1*pixratio)) * 180/np.pi * tsign )
        
        ## Length
        # the longest possible track goes between the most distant corners
        longlen = np.sqrt(1 + (pixratio*dim1)**2)
        # the shortest possible track goes between the nearest pixel borders
        shortlen = pixratio*(dim1-2)
        # let's calculate the measured length as the mean of the two
        tracklen = (shortlen+longlen)/2
        sd_tracklen = np.abs(longlen-shortlen)/2
              
    else:
        ## Angle
        # get the slope via a fit

        from scipy import odr

        linemodel = odr.Model(linefunc_m)

        fitdat = odr.RealData(x, y, sx=dx, sy=dy)

        # give a rough starting value for theta
        slopestart = (y[-1]-y[0])/(max([x[-1]-x[0],1]))

        myodr = odr.ODR(fitdat, linemodel, beta0=[slopestart, 0.])
        myoutput = myodr.run()
        slope = myoutput.beta[0]
        sd_slope = myoutput.sd_beta[0]
        
        theta = np.arctan(slope) * 180/np.pi
        sd_theta = sd_slope/(1+slope**2) * 180/np.pi
        
        ## Length
        # pre-compute some exponents of the slope
        slope2 = slope**2
        slope4 = slope**4
        slope6 = slope**6
        # depending on which axis was used to fit, calculate the length differently
        if dim0>=dim1:
            tracklen = np.abs(x[-1]-x[0]) * np.sqrt(slope2+1)
            sd_tracklen = np.sqrt((slope2+1) * (dx[-1]**2+dx[0]**2)
                                  + (x[-1]-x[0])**2 * (slope2/(1+slope2)) * sd_slope**2)
        else:
            tracklen = np.abs(y[-1]-y[0]) * np.sqrt(1+1/slope2)
            sd_tracklen = np.sqrt((1+1/slope2) * (dy[-1]**2+dy[0]**2)
                                  + (y[-1]-y[0])**2 * 1/(slope4 + slope6)  * sd_slope**2)

            
    return theta, sd_theta, np.abs(tracklen), sd_tracklen


class CosmicObservation:
    """
    Data of particle tracks extracted from a Gaia CCD observation

    Attributes:
        data: A structured numpy array containing all cosmic data
        source: type of source observation (BAM-OBS, BAM-SIF, SM_SIF)
        row: CCD row
        fov: source field of view
        acqTime: Acquisition Time (OBMT)
        srcAL: AL source image dimension
        srcAC: AC source image dimension
        maskpix: number of masked pixels in source
        flux: Estimated flux
        del_flux: Uncertainty on estimated flux
    """

    def __init__(self,ntracks=0):
        """Construct an empty instance of TrackObs"""
        self.data = np.empty(ntracks, dtype=[('LOC_AL','uint16'),
                                             ('LOC_AC','uint16'),
                                             ('DIM_AL','uint16'),
                                             ('DIM_AC','uint16'),
                                             ('TRACK_NPIX','uint16'),
                                             ('TRACK_EN','uint32'),
                                             ('TRACK_EN_ERR','uint32'),
                                             ('TRACK_TRUNCATED','bool'),
                                             ('GEOMETRY_VALID','bool'),
                                             ('TRACK_LEN', 'float32'),
                                             ('TRACK_LEN_ERR', 'float32'),
                                             ('TRACK_THETA', 'float32'),
                                             ('TRACK_THETA_ERR','float32')])

        self.source = None
        self.row = None
        self.fov = None
        self.acqTime = None
        self.srcAL = None
        self.srcAC = None
        self.maskpix = None
        self.flux = None
        self.flux_err = None
    
    
    # Process a TrackObs into a CosmicObservation
    @classmethod
    def from_TrackObs(cls, trobs):
        """Build a CosmicObservation from a TrackObs"""
        ntracks = len(trobs.data)
        obs = cls(ntracks)
        
        # copy keys that can be copied
        obs.source = trobs.source
        obs.row = trobs.row
        obs.fov = trobs.fov
        obs.acqTime = trobs.acqTime
        obs.srcAL = trobs.srcAL
        obs.srcAC = trobs.srcAC
        obs.maskpix = trobs.maskpix
        
        # calculate flux
        tflux, del_tflux = trobs.calculate_flux()
        obs.flux = tflux
        obs.flux_err = del_tflux
        
        # copy data that can be copied
        obs.data["DIM_AL"] = trobs.data["DIM_AL"]
        obs.data["DIM_AC"] = trobs.data["DIM_AC"]
        obs.data["LOC_AL"] = trobs.data["LOC_AL"]
        obs.data["LOC_AC"] = trobs.data["LOC_AC"]
        obs.data["TRACK_EN"] = trobs.data["TRACK_EN"]
        obs.data["TRACK_EN_ERR"] = trobs.data["DEL_EN"]
        
        # number of pixels
        obs.data["TRACK_NPIX"] = np.array([np.sum(t>0) for t in trobs.data["TRACK"]])
        
        # cut off pixels
        # get the edges of all the cosmics' bounding boxes
        beg_al = obs.data["LOC_AL"]
        beg_ac = obs.data["LOC_AC"]
        end_al = beg_al + obs.data["DIM_AL"]
        end_ac = beg_ac + obs.data["DIM_AC"]

        obs.data['TRACK_TRUNCATED'] = np.logical_or(np.logical_or(beg_al == 0, end_al == obs.srcAL), 
                                                    np.logical_or(beg_ac == 0, end_ac == obs.srcAC))
        
        
        # do track geometry
        # Get the binning
        if obs.source in ["BAM-OBS","BAM-SIF"]:
            # 1 x 4 binning
            bin_AL = 1
            bin_AC = 4
        elif obs.source == "SM-SIF":
            # 2 x 2 binning
            bin_AL = 2
            bin_AC = 2
            
        for ii in range(len(trobs.data)):
            # get the track
            track = extract_track(trobs,ii)
            # filter the tracks: The ones that are 'true' are those we do not want
            # they have either too few pixels, or a very bad shape,
            # since being only 2 in one dimension leaves a lot of uncertainty on the distance
            # covered here
            if ( np.sum(track>0) < 5 or np.sum(track>0)>1000
            or (track.shape[0]==2 and track.shape[1]<4) 
            or (track.shape[1]==2 and track.shape[0]<np.max([3*bin_AC/bin_AL, 5])) ):
                obs.data['GEOMETRY_VALID'][ii] = False
                # set everything to default 0
                obs.data['TRACK_THETA'][ii] = 0
                obs.data['TRACK_THETA_ERR'][ii] = 0
                obs.data['TRACK_LEN'][ii] = 0
                obs.data['TRACK_LEN_ERR'][ii] = 0
                continue # too few values to process or an obvious outlier
            ang,sd_ang,l,sd_l = track_angle_linefit(track,pixratio=3*bin_AC/bin_AL)
            if sd_ang>0 and sd_ang<90: 
                # the fit succeeded, save the values
                obs.data['GEOMETRY_VALID'][ii] = True
                
                obs.data['TRACK_THETA'][ii] = ang
                obs.data['TRACK_THETA_ERR'][ii] = sd_ang
                obs.data['TRACK_LEN'][ii] = l*10*bin_AL
                obs.data['TRACK_LEN_ERR'][ii] = sd_l*10*bin_AL
            else:
                # something went wrong in the fit
                obs.data['GEOMETRY_VALID'][ii] = False
                # set everything to default 0
                obs.data['TRACK_THETA'][ii] = 0
                obs.data['TRACK_THETA_ERR'][ii] = 0
                obs.data['TRACK_LEN'][ii] = 0
                obs.data['TRACK_LEN_ERR'][ii] = 0
       
        return obs
    
    
    # Retrieve a CosmicObservation from a fits file
    @classmethod
    def from_HDU(cls, hdu):
        """Build a CosmicObservation from a fits HDU"""
        ntracks = len(hdu.data)
        obs = cls(ntracks)
        
        # move over the data
        obs.data["LOC_AL"] = hdu.data["LOC_AL"]
        obs.data["LOC_AC"] = hdu.data["LOC_AC"]
        obs.data["DIM_AL"] = hdu.data["DIM_AL"]
        obs.data["DIM_AC"] = hdu.data["DIM_AC"]
        obs.data["TRACK_NPIX"] = hdu.data["TRACK_NPIX"]
        obs.data["TRACK_EN"] = hdu.data["TRACK_EN"]
        obs.data["TRACK_EN_ERR"] = hdu.data["TRACK_EN_ERR"]
        obs.data["TRACK_TRUNCATED"] = hdu.data["TRACK_TRUNCATED"]
        obs.data["GEOMETRY_VALID"] = hdu.data["GEOMETRY_VALID"]
        obs.data["TRACK_LEN"] = hdu.data["TRACK_LEN"]
        obs.data["TRACK_LEN_ERR"] = hdu.data["TRACK_LEN_ERR"]
        obs.data["TRACK_THETA"] = hdu.data["TRACK_THETA"]
        obs.data["TRACK_THETA_ERR"] = hdu.data["TRACK_THETA_ERR"]

        # populate the keys from the header
        obs.source = hdu.header["SOURCE"]
        obs.row = hdu.header["CCD_ROW"]
        obs.fov = hdu.header["FOV"]
        obs.acqTime = hdu.header["ACQTIME"]
        obs.srcAL = hdu.header["SRC_AL"]
        obs.srcAC = hdu.header["SRC_AC"]
        obs.maskpix = hdu.header["MASKPIX"]
        obs.flux = hdu.header["FLUX"]
        obs.flux_err = hdu.header["FLUX_ERR"]

        return obs
    
    
    
    # write a CosmicObservation
    def to_BinTableHDU(self,extname):
        """
        Method to turn this into a astropy.fits BinTableHDU.
        The name of the extension is set via extname
        """
        # prepare the columns
        # write everything into columns. 
        datcols = fits.ColDefs(self.data)
        
        # set the units
        datcols.change_unit("LOC_AL","pixels")
        datcols.change_unit("LOC_AC","pixels")
        datcols.change_unit("DIM_AL","pixels")
        datcols.change_unit("DIM_AC","pixels")
        datcols.change_unit("TRACK_NPIX","pixels")
        datcols.change_unit("TRACK_EN","electrons")
        datcols.change_unit("TRACK_EN_ERR","electrons")
        #datcols.change_unit("TRACK_TRUNCATED")
        #datcols.change_unit("GEOMETRY_VALID")
        datcols.change_unit("TRACK_LEN","um")
        datcols.change_unit("TRACK_LEN_ERR","um")
        datcols.change_unit("TRACK_THETA","deg")
        datcols.change_unit("TRACK_THETA_ERR","deg")

        tbhdu = fits.BinTableHDU.from_columns(datcols)

        # the header
        tbhdu.header.comments["TTYPE1"] = "AL image location of Track[0,0]"
        tbhdu.header.comments["TTYPE2"] = "AC image location of Track[0,0]"
        tbhdu.header.comments["TTYPE3"] = "AL track dimension"
        tbhdu.header.comments["TTYPE4"] = "AC track dimension"
        tbhdu.header.comments["TTYPE5"] = "Number of signal pixels in the track"
        tbhdu.header.comments["TTYPE6"] = "Total track energy"
        tbhdu.header.comments["TTYPE7"] = "Uncertainty of TRACK_EN"
        tbhdu.header.comments["TTYPE8"] = "Is the track on the edge of the image?"
        tbhdu.header.comments["TTYPE9"] = "Could the track length and angle be calculated?"
        tbhdu.header.comments["TTYPE10"] = "Calculated track length (if GEOMETRY_VALID)"
        tbhdu.header.comments["TTYPE11"] = "Uncertainty of track length (if GEOMETRY_VALID)"
        tbhdu.header.comments["TTYPE12"] = "Calculated track angle (if GEOMETRY_VALID)"
        tbhdu.header.comments["TTYPE13"] = "Uncertainty of track angle (if GEOMETRY_VALID)"
        


        # additional header values
        tbhdu.header["SOURCE"] = self.source
        tbhdu.header.comments["SOURCE"] = "Observation type"
        tbhdu.header["CCD_ROW"] = self.row
        tbhdu.header.comments["CCD_ROW"] = "CCD row"
        tbhdu.header["FOV"] = self.fov
        tbhdu.header.comments["FOV"] = "Field of View"

        tbhdu.header["ACQTIME"] = self.acqTime
        tbhdu.header.comments["ACQTIME"] = "Acquisition Time [OBMT]"
        
        tbhdu.header["ACQDATE"] = OBMT_apyTime(self.acqTime).isot+"Z"
        tbhdu.header.comments["ACQDATE"] = "Acquisition Time [UTC]"

        tbhdu.header["SRC_AL"] = self.srcAL
        tbhdu.header.comments["SRC_AL"] = "AL source image dimension"
        tbhdu.header["SRC_AC"] = self.srcAC
        tbhdu.header.comments["SRC_AC"] = "AC source image dimension"
        tbhdu.header["MASKPIX"] = self.maskpix
        tbhdu.header.comments["MASKPIX"] = "Number of masked pixels"

        tbhdu.header["FLUX"] = self.flux
        tbhdu.header.comments["FLUX"] = "Measured flux [particles/cm^2/s]"
        tbhdu.header["FLUX_ERR"] = self.flux_err
        tbhdu.header.comments["FLUX_ERR"] = "Poisson uncertainty of flux [particles/cm^2/s]"
        
        # set the extension name
        tbhdu.name = extname
        
        return tbhdu
    
    
def write_list_to_fits(cobslist, fname):
    """
    Create an element of the output catalog, writing a list of CosmicObservations to fname
    """
    
    # Create the HDUlist
    hdulist = fits.HDUList()
    for ii in range(len(cobslist)):
        hdulist.append(cobslist[ii].to_BinTableHDU(extname='OBS_{}'.format(ii+1)))
        
    # add some meta-information to the primary
    import datetime
    hdulist[0].header["GEN_TIME"] = datetime.datetime.utcnow().isoformat()+"Z"
    hdulist[0].header.comments["GEN_TIME"] = "Generation Time [UTC]"
    
    hdulist[0].header["ORIGIN"] = "Generated at ESAC (CALTEAM)"
    hdulist[0].header.comments["ORIGIN"] = "Origin of data"
    
    global CATALOG_VERSION
    hdulist[0].header["VERSION"] = CATALOG_VERSION
    hdulist[0].header.comments["VERSION"] = "Catalog Version"
    
    
    
    hdulist.writeto(fname)
    hdulist.close()
