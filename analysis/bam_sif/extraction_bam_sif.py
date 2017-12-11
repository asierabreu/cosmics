import os
import sys
sys.path.insert(0, os.path.abspath('../../lib/TrackObs/'))
from TrackObs import *

sys.path.insert(0, os.path.abspath('../bam/'))
from extraction_bam import *

import numpy as np
from astropy.io import fits
import scipy.stats

# TODO:
# we might also be able to use the region 'before' the pattern

def bam_sif_read(sourcefile, N_AL, gain, bias):
    """
    Read in the data from a BAM-SIF fits file, using the last N_AL lines.
    Perform bias subtraction and gain correction
    
    Returns total data array of both FOV and acqTime.
    """
    source = fits.getdata(sourcefile)[-99-N_AL:-99,2:]
    source = (source-bias)*gain
    
    acqTime = fits.getheader(sourcefile)["OBMT_BEG"]
    
    return source, acqTime
    

def bam_sif_signal(source, readnoise):
    """
    Extract the signal from a BAM-SIF observation and perform the error estimation
    
    Returns the signal, its uncertainty and the subtracted background.
    """
    # Get the background
    bkg_src = np.copy(source[-99:]).astype("float64") # last pixels of the image area
    
    # determine the background using outlier rejection
    # throw away highest and lowest 25%
    bkg_src = scipy.stats.trimboth(bkg_src,0.25,axis=0)

    sdevs = np.std(bkg_src, axis=0)
    background = np.mean(bkg_src, axis=0)
    
    
    # signal and uncertainty
    extracted = np.copy(source)

    signal = (extracted - background)
    err_sig = np.sqrt(np.abs(extracted) + readnoise*readnoise + sdevs*sdevs) # this should be accurate?
    
    
    return signal, err_sig, background

# this also includes the AC-extent for fov 2
# rewrite the bam_cosmics routine to write a BAM-SIF header (which then implicitly assumes our readout window)

def bam_sif_cosmics(signal, err_sig, threshold, threshfrac, N_mask, gain, acqTime):
    """
    Docstring TBD. Essentially a specialized wrapper around bam_cosmics from bam extraction. Returns TWO trackobs
    """
    # Get the cosmics via BAM cosmics
    # TODO use separate N_mask - probably 2-length array
    out1 = bam_cosmics(signal[:,:80], err_sig[:,:80], threshold, threshfrac, N_mask, gain)
    # fov 2 does not use all the field - I may want to make this a parameter above
    out2 = bam_cosmics(signal[:,80:140], err_sig[:,80:140], threshold, threshfrac, N_mask, gain)
    
    # Modify the outputs
    for output in [out1,out2]:
        output.source = "BAM-SIF"
        output.row = 1
        output.acqTime = acqTime
        output.gain = gain
    
    out1.fov = 1
    out2.fov = 2
    
    return out1, out2