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
    The extracted pattern is masked for invalid pixels
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
    #epattern[pattern==0] = 1e10
    
    # let's try to do an actual masked array instead - this high energy business causes problems
    arrmask = np.zeros(pattern.shape).astype("bool")
    arrmask[pattern==0] = True
    epattern = np.ma.masked_array(epattern,mask=arrmask,fill_value=0) # may need to rethink fill value
    
    return epattern, fov, acqTime


class BoxCar:
    """
    A FIFO of bam patterns extracted via bam_read_obs used to extract cosmics from BAM observations
    
    Attributes:
    boxrad: Number of patterns before and after the pattern of interest
    npatterns: Number of patterns (2*boxrad+1)
    fov: FOV of the patterns
    
    nfilled: Number of populated patterns
    patterns: Numpy masked(!) array of patterns
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
        
        self.patterns = np.ma.masked_array(np.zeros((self.npatterns, 1000, 80)),mask="nomask",fill_value=0)
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
        bkg_src = sigma_clip(self.patterns, sigma=1, iters=2, axis=0)
        background = np.mean(bkg_src, axis=0)
        
        # extract signal
        signal = np.copy(self.patterns[self.i_sig]) - background
        
        # no signal in overflow pixels - they get masked
        N_mask = np.sum(self.patterns[self.i_sig].mask.astype("int")) # number of masked pixels
        signal[self.patterns[self.i_sig].mask] = 0
        signal.mask = np.copy(self.patterns[self.i_sig].mask)
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

        return signal, err_mean




# Function to extract cosmics
# Algorithm TBD, probably either laplacian detection or median clipping
# for now, just use median clipping

def bam_cosmics(signal, err_mean, threshold, threshfrac, gain):
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

    
    #### Discarding cosmics connected to bad pixels
    
    N_mask = np.sum(signal.mask)
    if N_mask == 0:
        pass
    else:
        # reset N_mask
        N_mask = 0
        
        badpix = signal.mask
        
        # add bad pixels to the mask and find the connected objects
        badmask = np.logical_or(mask, badpix)
        (dilabels, ndilabs) = ndimage.measurements.label(badmask, structure=(np.ones((3,3))))
        badlabels = np.unique(dilabels[badpix])
        for l in badlabels:
            mask[dilabels == l] = 0
            N_mask += np.sum(dilabels == l)
        (labels, ntracks) = ndimage.measurements.label(mask, structure=(np.ones((3,3))))
    
    
    # object extraction
    events = ndimage.measurements.find_objects(labels)

    signal *= mask
    
    
    #### Save the cosmics in a TrackObs

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
        cosmic[lab!=ii+1] = 0
        
        # total energy
        Etot = np.rint((np.sum(cosmic)))
        
        # uncertainty on total energy
        err = np.copy(err_mean[location])
        err[lab!= ii+1] = 0
        delEtot = np.rint((np.sqrt(np.sum(err**2))))
        
        # turn the cosmic into a flattened array, but save its dimensions
        dim = cosmic.shape
        cosmic = cosmic.flatten()
        # cosmics should be gain corrected and turned into ints
        cosmic = np.rint(cosmic/gain).astype('uint16')
        
        output.data[ii] = (cosmic, dim[0], dim[1], loc[0], loc[1], Etot, delEtot)

    return output

def bam_cosmics_mended(signal, err_mean, threshold, threshfrac, gain):
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
    
    
    #### Connecting separated cosmics
    
    # Try to reconnect cosmics that have been cut apart, most likely by the fringes
    # Dilate/Close the mask and make labels again. See what overlaps now, and if they are connected
    
    # just dilation
    connect_mask = ndimage.binary_dilation(mask,structure=(np.ones((3,3))))
    # erosion by + , instead of just closing
    connect_mask = ndimage.binary_erosion(connect_mask,structure=(np.array([[0,1,0],[1,1,1],[0,1,0]])))    
    # alternative: sclosing
    #connect_mask = ndimage.binary_closing(mask,structure=(np.ones((3,3))))
    
    (dilabels, ndilabs) = ndimage.measurements.label(connect_mask, structure=(np.ones((3,3))))
    
    # what are the overlaps
    if ntracks != ndilabs:
        # if they're the same, nothing changes
        mappings = []
        # find out the overlaps
        for dilab in range(1, ndilabs+1):
            overlabels = np.unique(labels[dilabels==dilab])
            mappings.append([l for l in overlabels if l!= 0])
        # where there are overlaps, try to look for overlooked stuff
        for il in range(len(mappings)):
            mp = mappings[il]
            dilab = il+1
            if len(mp) == 1:
                continue
            else:
                # mean energies for each track in this label
                # problem: this can be more than two, so I can't just compare energies and connect
                mean_e = np.array([np.mean(signal[labels == l]) for l in mp])
                region = (dilabels == dilab)
                
                for ii in range(len(mp)):
                    # the 2 here is a parameter, and can be tuned
                    mask[region] = np.logical_or(mask[region], (np.abs(signal[region] - mean_e[ii])/err_mean[region] < 2))
        
        (labels, ntracks) = ndimage.measurements.label(mask, structure=(np.ones((3,3))))
    
    
    #### Discarding cosmics connected to bad pixels
    
    N_mask = np.sum(signal.mask)
    if N_mask == 0:
        pass
    else:
        # reset N_mask
        N_mask = 0
        
        badpix = signal.mask
        
        # add bad pixels to the mask and find the connected objects
        badmask = np.logical_or(mask, badpix)
        (dilabels, ndilabs) = ndimage.measurements.label(badmask, structure=(np.ones((3,3))))
        badlabels = np.unique(dilabels[badpix])
        for l in badlabels:
            mask[dilabels == l] = 0
            N_mask += np.sum(dilabels == l)
        (labels, ntracks) = ndimage.measurements.label(mask, structure=(np.ones((3,3))))
        
    
    #### Save the cosmics in a TrackObs
    
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
        cosmic[lab!=ii+1] = 0
        
        # total energy
        Etot = np.rint((np.sum(cosmic)))
        
        # uncertainty on total energy
        err = np.copy(err_mean[location])
        err[lab!= ii+1] = 0
        delEtot = np.rint((np.sqrt(np.sum(err**2))))
        
        # turn the cosmic into a flattened array, but save its dimensions
        dim = cosmic.shape
        cosmic = cosmic.flatten()
        # cosmics should be gain corrected and turned into ints
        cosmic = np.rint(cosmic/gain).astype('uint16')
        
        output.data[ii] = (cosmic, dim[0], dim[1], loc[0], loc[1], Etot, delEtot)

    return output