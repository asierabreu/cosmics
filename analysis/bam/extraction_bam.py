import os
import sys
sys.path.insert(0, os.path.abspath('../../lib/PythonGbinReader/GbinReader/'))
import gbin_reader

# TODO: Function to build the boxcar
# Idea: Give the program a list of filenames for observations, sequential in time
# then, have it open two boxcars and populate them (open the file, read the fov and forward to the existing boxcar - use the update boxcar function!)
# might just do this in the code itself - won't be repeated

# Function to read a bamObs file
def bam_read_obs(reader, bias, gain):
    """
    Read the next bamObservation of reader and perform bias-subtraction and gain-correction.
    Returns the extracted pattern and its FOV
    Returns pattern = None and Fov = None if the end has been reached

    filename: filename of the BAM observation
    bias: bias value
    gain: gain value
    """
    import numpy as np
    import gbin_reader  # need to include this in the import path
    try:
        tempObs = reader.__next__()
    except StopIteration:
        return None, None

    fov = tempObs.fov
    pattern = (np.array(tempObs.samples).reshape(1000,80)).astype("float64")
    pattern = (pattern-bias)*gain
    return pattern, fov


# Function to "keep the boxcar moving"
# Essentially call this after each extraction step, to update the boxcar and move the index for the pattern of interest
def boxcar_update(boxcar, pattern, i_rep, i_sig):
    """ Updates the boxcar by replacing pattern number i_sig
    increases i_rep and i_sig by 1 each, wrapping around
    """
    # replace the pattern
    boxcar[i_rep,:,:] = pattern

    npatterns = boxcar.shape[0]

    # update i_rep and i_sig
    i_rep += 1
    if i_rep >= npatterns:
        i_rep = 0

    i_sig += 1
    if i_sig >= npatterns:
        i_sig = 0

    return i_rep, i_sig

    

def boxcar_signal(boxcar, i_sig, readnoise):
    """ Extract background-subtracted signal and noise from a collection of bam patterns.
    Patterns should already be bias-subtracted and gain-corrected!

    boxcar: numpy array of bam patterns
    i_sig: index of the signal pattern
    readnoise: ccd readnoise
    """
    import numpy as np 
    from astropy.stats import sigma_clip

    npatterns = boxcar.shape[0]

    # get the background
    # compute the median over time, removing outliers via sigma clipping
    # TODO: Value for sigma
    bkg_src = sigma_clip(boxcar, sigma=2, iters=None, axis=0)

    background = np.mean(bkg_src, axis=0)

    # extract signal
    signal = boxcar[i_sig] - background


    # compute uncertainty
    (xmax, ymax) = boxcar[0].shape
    err_mean = np.zeros((xmax,ymax))

    # Number of elements we averaged over
    N_time = (npatterns)-np.sum(bkg_src.mask.astype("int"),axis=0)

    # variance of background, from error propagation
    var_mean = (readnoise*readnoise + np.sum(bkg_src,axis=0)/N_time)/N_time

    # total error (background + signal
    err_mean = np.sqrt(var_mean + readnoise*readnoise + boxcar[i_sig,:,:]) 

    return signal, err_mean

# TODO function to extract cosmics
# Algorithm TBD, probably either laplacian detection or median clipping
# THEN, need to save cosmics to some format
# For now, background-subtracted slices of that cosmic, with all other cosmics excluded (so (cosmics*labels[labels==ii])[evtlocs[ii]] )

# for now, just use median clipping

def boxcar_cosmics(signal, err_mean, threshold, threshfrac):

    import numpy as np
    import astroscrappy
    import scipy.ndimage as ndimage

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

    # output is a dictionary, containing the arrays
    cosmics = []  # the cosmic ray images
    locs = []     # a list of tuples for the START of events
    Etot = []     # total energies
    delEtot = []    # uncertainties of energy
    
    signal *= mask

    for ii in range(ntracks):
        location = events[ii]
        
        # the event (only for this label)
        cosmic = signal[location]
        lab = labels[location]
        cosmic[lab!=ii] == 0
        cosmics.append(cosmic)
        
        # the location
        locs.append((location[0].start, location[1].start))
        
        # total energy
        Etot.append(np.sum(cosmic))
        
        # uncertainty on total energy
        err = err_mean[location]
        err[lab!= ii] == 0
        delEtot.append(np.sqrt(np.sum(err**2)))
        
    output = dict([('cosmics', cosmics), ('locs', locs), ('Etot', Etot), ('delEtot', delEtot)])

    return output
    # OBMT needs to be aquired externally!
