"""Class definition file for the TrackObs class, defining the particle tracks extracted from a Gaia observation"""

import numpy as np
from astropy.io import fits

class TrackObs:
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
        gain: CCD gain
    """

    def __init__(self,ntracks=0):
        """Construct an empty instance of TrackObs"""
        self.data = np.empty(ntracks, dtype=[('TRACK','object'),
                                             ('DIM_AL','uint16'),
                                             ('DIM_AC','uint16'),
                                             ('LOC_AL','uint16'),
                                             ('LOC_AC','uint16'),
                                             ('TRACK_EN','int32'),
                                             ('DEL_EN','int32')])

        self.source = None
        self.row = None
        self.fov = None
        self.acqTime = None
        self.srcAL = None
        self.srcAC = None
        self.maskpix = None
        self.gain = None
    
    @classmethod
    def from_HDU(cls, hdu):
        """Build a TrackObs from a fits HDU"""
        ntracks = len(hdu.data)
        obs = cls(ntracks)
        
        # move over the data
        obs.data["TRACK"] = hdu.data["TRACK"]
        obs.data["DIM_AL"] = hdu.data["DIM_AL"]
        obs.data["DIM_AC"] = hdu.data["DIM_AC"]
        obs.data["LOC_AL"] = hdu.data["LOC_AL"]
        obs.data["LOC_AC"] = hdu.data["LOC_AC"]
        obs.data["TRACK_EN"] = hdu.data["TRACK_EN"]
        obs.data["DEL_EN"] = hdu.data["DEL_EN"]
        
        # populate the keys from the header
        obs.source = hdu.header["SOURCE"]
        obs.row = hdu.header["CCD_ROW"]
        obs.fov = hdu.header["FOV"]
        obs.acqTime = hdu.header["ACQTIME"]
        obs.srcAL = hdu.header["SRC_AL"]
        obs.srcAC = hdu.header["SRC_AC"]
        obs.maskpix = hdu.header["MASKPIX"]
        obs.gain = hdu.header["GAIN"]

        return obs
    
    # TODO: Method to calculate the flux measured in a TrackObs
    # Essentially: You know the source, so pixel sizes etc. are known T_TDI is also known
    # -> then, just use the known formulas to calculate your flux
    def calculate_flux(self):
        """Calculates the particle flux (in particles/cm^2/s) for this observation"""
        # Get the correct pixel dimensions (in cm), including binning
        if self.source in ["BAM-OBS","BAM-SIF"]:
            # 1 x 4 binning
            pixAL = 10e-4
            pixAC = 120e-4
            exptime = 4.5 * 0.9828 + 19
        elif self.source == "SM-SIF":
            # 2 x 2 binning
            pixAL = 20e-4
            pixAC = 60e-4
            exptime = 2.9 * 0.9828
            
        exparea = pixAL*pixAC*(self.srcAL*self.srcAC - self.maskpix)
        
        return len(self.data)/exparea/exptime
    
    def track_geometries(self):
        """VERY PRELIMINARY!
        Calculates the geometry of each track
        Returns track lengths [mum] and angles theta, alpha [radians].
        """
        # Get the correct pixel dimensions (in mum), including binning
        if self.source in ["BAM-OBS","BAM-SIF"]:
            # 1 x 4 binning
            pixAL = 10
            pixAC = 120
            pixdepth = 40
        elif self.source == "SM-SIF":
            # 2 x 2 binning
            pixAL = 20
            pixAC = 60
            pixdepth = 14
            
        thetas = np.zeros(len(self.data))
        alphas = np.zeros(len(self.data))
        
        l_al = (self.data["DIM_AL"]-1)*pixAL
        l_ac = (self.data["DIM_AC"]-1)*pixAC
        lengths = np.sqrt(l_al*l_al + l_ac*l_ac)
        
        nonzero = np.logical_and(l_al!=0, l_ac!=0)
        thetas[nonzero] = np.arctan(l_ac[nonzero]/l_al[nonzero])
        thetas[l_al==0] = np.pi/2
        thetas[l_ac==0] = 0
        thetas[lengths==0] = 10 # dummy value
        
        alphas[nonzero] = np.arctan(pixdepth/lengths[nonzero])
        alphas[lengths==0] = np.pi/2
        
        return lengths, thetas, alphas
                
    

# function to write a list of Trackobs into a fits file
# maybe I should make the thing that's in the loop into a method? i.e. TrackObs -> HDU
# I would have to supply an extension name at some point, though - either save it in the TrackObs or give it as an argument to the method. Since I'm not sure whether these names should have any meaning, we can see
def write_Obslist(obslist, outfile):
    """Write the observations from obslist into the fits file filename"""
    from astropy.io import fits

    # create an empty HDUList
    hdulist = fits.HDUList()
    
    # write into these lists one by one
    for ii in range(len(obslist)):
        obs = obslist[ii]
        # prepare the columns
        # write everything into columns. 
        # For some reason, I need to do bzero of TRACK manually
        trackcol = fits.ColDefs(input=[fits.Column(name='TRACK', format='PI', 
                   array=list(obs.data["TRACK"]+32768), unit="ADU")])
        auxcols = fits.ColDefs(obs.data[["DIM_AL","DIM_AC","LOC_AL","LOC_AC","TRACK_EN","DEL_EN"]])

        tbhdu = fits.BinTableHDU.from_columns(trackcol+auxcols)
        tbhdu.header["TZERO1"] = 32768

        tbhdu.name = "OBS_{:03d}".format(ii+1) # maybe come up with something better here...

        # the header
        tbhdu.header.comments["TTYPE1"] = "Flattened cosmic ray track"
        tbhdu.header.comments["TTYPE2"] = "AL (Dim 1) track dimension"
        tbhdu.header.comments["TTYPE3"] = "AC (Dim 2) track dimension"
        tbhdu.header.comments["TTYPE4"] = "AL image location of Track[0,0]"
        tbhdu.header.comments["TTYPE5"] = "AC image location of Track[0,0]"
        tbhdu.header.comments["TTYPE6"] = "Total track energy"
        tbhdu.header.comments["TTYPE7"] = "Uncertainty of TRACK_EN"

        # additional header values
        tbhdu.header["SOURCE"] = obs.source
        tbhdu.header.comments["SOURCE"] = "Observation type"
        tbhdu.header["CCD_ROW"] = obs.row
        tbhdu.header.comments["CCD_ROW"] = "CCD row"
        tbhdu.header["FOV"] = obs.fov
        tbhdu.header.comments["FOV"] = "Field of View"

        tbhdu.header["ACQTIME"] = obs.acqTime
        tbhdu.header.comments["ACQTIME"] = "Acquisition Time [OBMT]"

        tbhdu.header["SRC_AL"] = obs.srcAL
        tbhdu.header.comments["SRC_AL"] = "AL source image dimension"
        tbhdu.header["SRC_AC"] = obs.srcAC
        tbhdu.header.comments["SRC_AC"] = "AC source image dimension"
        tbhdu.header["MASKPIX"] = obs.maskpix
        tbhdu.header.comments["MASKPIX"] = "Number of masked pixels"

        tbhdu.header["GAIN"] = obs.gain
        tbhdu.header.comments["GAIN"] = "CCD Gain [e/ADU]"



        # save to hdulist
        hdulist.append(tbhdu)
            
    hdulist.writeto(outfile)
    hdulist.close()
    

    
# function to read a list of TrackObs from a fits file
def read_Obslist_fits(infile):
    hdulist = fits.open(infile)
    obslist = []
    
    for ii in range(1,len(hdulist)):
        obslist.append(TrackObs.from_HDU(hdulist[ii]))
    
    hdulist.close()
    return obslist