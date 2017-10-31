import os
import sys
sys.path.insert(0, os.path.abspath('../../lib/PythonGbinReader/GbinReader/'))
import gbin_reader
sys.path.insert(0, os.path.abspath('../../lib/TrackObs/'))
from TrackObs import *

# TODO: Function to build the boxcar
# Idea: Give the program a list of filenames for observations, sequential in time
# then, have it open two boxcars and populate them (open the file, read the fov and forward to the existing boxcar - use the update boxcar function!)
# might just do this in the code itself - won't be repeated

# Function to read a bamObs file
def bam_read_obs(reader, bias, gain):
    """
    Read the next bamObservation of reader and perform bias-subtraction and gain-correction.
    Returns the extracted pattern, its FOV and OBMT
    Returns None, None, None if the end has been reached

    filename: filename of the BAM observation
    bias: bias value
    gain: gain value
    """
    import numpy as np
    import gbin_reader  # need to include this in the import path
    try:
        tempObs = reader.__next__()
    except StopIteration:
        return None, None, None

    fov = tempObs.fov
    acqTime = tempObs.acqTime
    pattern = (np.array(tempObs.samples).reshape(1000,80))
    epattern = ((pattern-bias)*gain).astype('float64')  # pattern in electrons
    
    # in some cases, cosmics can cause an overflow in the pattern, so that some neighbouring pixels are set to 0 ADU
    # we need to mask these - set them to a fixed value - in this case, 1e10
    # (this could never be reached by the CCD (max = 65535*gain = about 260000)
    epattern[pattern==0] = 1e10
    
    return epattern, fov, acqTime


class BoxCar:
    """
    A FIFO of bam patterns extracted via bam_read_obs used to extract cosmics from BAM observations
    
    Attributes:
    boxrad: Number of patterns before and after the pattern of interest
    npatterns: Number of patterns (2*boxrad+1)
    fov: FOV of the patterns
    
    nfilled: Number of populated patterns
    patterns: Numpy array of patterns
    acqTimes: respective acquisition times for each pattern
    i_rep: index of next pattern to replace
    i_sig: index of current signal pattern
    """
    
    def __init__(self,boxrad,fov):
        """Constructor - creates an empty boxcar"""
        # basic data
        self.boxrad = boxrad
        self.npatterns = 2*boxrad+1
        self.fov = fov
        
        # empty boxcar
        self.nfilled = 0
        self.i_rep = 0
        self.i_sig = 0
        
        self.patterns = np.zeros((self.npatterns, 1000, 80))
        self.acqTimes = np.zeros((self.npatterns))
        
        
    def update(self, pattern, acqTime):
        """Updates the boxcar by replacing pattern and acqTime i_rep and updating i_rep and i_sig"""
        
        # replace the pattern and acqTime
        self.patterns[self.i_rep,:,:] = pattern
        self.acqTimes[self.i_rep] = acqTime
        
        # update i_rep and i_sig
        # may need to start from 0 if necessary
        self.i_rep += 1
        if self.i_rep >= self.npatterns:
            self.i_rep = 0

        self.i_sig += 1
        if self.i_sig >= self.npatterns:
            self.i_sig = 0        
            
            
    def get_signal(self,readnoise):
        """ Extract background-subtracted signal and noise from a collection of bam patterns. Also returns the number of masked pixels.
        Patterns should already be bias-subtracted and gain-corrected!

        readnoise: CCD readnoise
        """
        import numpy as np 
        from astropy.stats import sigma_clip
        
        # get the background
        # compute the median over time, removing outliers via sigma clipping
        # this also throws away the overflow pixels
        # TODO: Value for sigma
        # perhaps I should manually reduce iters? To throw away less photon noise
        # essentially, once sigma is roughly poisson, I don't need to discard anymore...
        bkg_src = sigma_clip(self.patterns, sigma=2, iters=2, axis=0)
        background = np.mean(bkg_src, axis=0)
        
        # extract signal
        signal = self.patterns[self.i_sig] - background
        
        # no signal in overflow pixels - they get masked
        N_mask = np.sum(signal==1e10) # number of masked pixels
        signal[signal==1e10] = 0
        # perhaps we should also mask all pixels that are saturated (i.e. 65535, in the function before?)
        
        # compute uncertainty
        (xmax, ymax) = signal.shape
        err_mean = np.zeros((xmax,ymax))
        
        # Number of elements we averaged over
        N_time = (self.npatterns)-np.sum(bkg_src.mask.astype("int"),axis=0)
        
        # variance of background, from error propagation
        var_mean = (readnoise*readnoise + np.sum(bkg_src,axis=0)/N_time)/N_time
        
        # total error (background + signal)
        # this overestimates the uncertainty on the overflow pixels, but we don't care about those
        err_mean = np.sqrt(var_mean + readnoise*readnoise + self.patterns[self.i_sig,:,:]) 

        return signal, err_mean, N_mask




# Function to extract cosmics
# Algorithm TBD, probably either laplacian detection or median clipping
# for now, just use median clipping

def bam_cosmics(signal, err_mean, threshold, threshfrac, N_mask, gain):
    """
    Docstring TBD
    """

    import numpy as np
    import astroscrappy
    import scipy.ndimage as ndimage
    
    
    (xmax,ymax) = signal.shape  # for saving, later
    # construct mask from signal/error
    SN = signal/err_mean
    mask = np.zeros(signal.shape)
    mask[SN > threshold] = 1

    # neighbours
    mask = mask.astype('bool')
    
    mask = astroscrappy.dilate3(mask)
    mask = np.logical_and(SN > threshold, mask)

    # dilation - do this a few times
    for ii in range(10):
        newmask = astroscrappy.dilate3(mask)
        newmask = np.logical_and(SN > threshfrac*threshold, newmask)
        if (newmask==mask).all():
            break
        else:
            mask = newmask

    # label cosmics
    (labels, ntracks) = ndimage.measurements.label(mask, structure=(np.ones((3,3))))

    # object extraction
    events = ndimage.measurements.find_objects(labels)

    signal *= mask
    

    # our output is a TrackObs, containing the cosmic data and several keywords
    output = TrackObs(ntracks)
    
    output.source = "BAM-OBS"
    output.srcAL = xmax
    output.srcAC = ymax
    output.maskpix = N_mask
    # aqcTime, row, gain and fov need to be retrieved externally
    
    # fill the data
    for ii in range(ntracks):
        # the location
        location = events[ii]
        loc = ((location[0].start, location[1].start))
        
        # the event (only for this label)
        cosmic = np.copy(signal[location])
        lab = labels[location]
        cosmic[lab!=ii] == 0
        
        # total energy
        Etot = np.rint((np.sum(cosmic)))
        
        # uncertainty on total energy
        err = np.copy(err_mean[location])
        err[lab!= ii] == 0
        delEtot = np.rint((np.sqrt(np.sum(err**2))))
        
        # turn the cosmic into a flattened array, but save its dimensions
        dim = cosmic.shape
        cosmic = cosmic.flatten()
        # cosmics should be gain corrected and turned into ints
        cosmic = np.rint(cosmic/gain).astype('uint16')
        
        output.data[ii] = (cosmic, dim[0], dim[1], loc[0], loc[1], Etot, delEtot)

    return output

