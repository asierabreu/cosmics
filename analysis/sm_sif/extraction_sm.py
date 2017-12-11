# things that are needed still

# a reasonable value for the buffer and the threshold when extracting cosmics

import os
import sys
sys.path.insert(0, os.path.abspath('../../lib/TrackObs/'))
from TrackObs import *

def sm_get_image(filename, calibfile):
    """
    Retrieve image data from filename and gain, bias and readnoise from calibfile. Also return data on the chip used and the acquisition time.
    """
    from astropy.io import fits
    hdusource = fits.open(filename)

    row = hdusource[0].header['CCD_ROW']
    fov = int((hdusource[0].header['CCD'])[2])
    tstart = hdusource[0].header['OBMT_BEG']
    tstop = hdusource[0].header['OBMT_END']
    image = hdusource[0].data[:,7:]  # ignore the pre-scan column

    hdusource.close()


    hducal = fits.open(calibfile)

    ii = 2*(row-1) + fov -1    # index of the row of interest
    bias = hducal[1].data["BIAS"][ii][7:] # ignore the pre-scan column
    gain = hducal[1].data["GAIN"][ii]
    readnoise = hducal[1].data["RNOISE"][ii]

    hducal.close()
    
    return image, gain, bias, readnoise, fov, row, tstart, tstop


def sm_starmask(image,threshold, badcol=-1):
    """
    Construct a mask for all saturated stars in the SM image.

    This constructs a map of pixels above threshold, labels all connected pixels as one object, then searches for objects that contain at least one saturated pixel (i.e. above 65535 ADU).
    All pixels connected to those saturated pixels are then masked.
    This function has also been modified to catch ghosts caused by bright stars at lower AC
    Additionally, all pixels connected to column badcol (if it is greater equal 0) are masked as well.
    """
    import numpy as np
    from scipy.ndimage import generate_binary_structure
    from scipy.ndimage.morphology import binary_dilation
    from scipy.ndimage.measurements import label, find_objects

    # occasionally, an overflow for very bright stars causes some pixels to be set to 0
    # -> set them to 65535, so they will be masked
    image[image==0] = 65535
    
    # initialize the mask to all False
    starmask = np.zeros(image.shape, dtype=bool)

    # mask of pixels above threshold
    satmap = np.logical_and(image,image>=threshold)

    # dilate it - sometimes, pixel values fluctuate around the threshold
    satmap = binary_dilation(satmap,iterations=1)

    # labelling all connected pixels
    (starlabels, nstars) = label(satmap,structure=(np.ones((3,3))))

    # we're only interested in the labels of pixels that are saturated
    # so, get all the labels that have that:
    satlabels = np.unique(starlabels[image==65535])

    for lab in satlabels:
        starcoords = np.argwhere(starlabels==lab)
        starmask[starlabels==lab] = 1
        # this star may also cause electronic ghosts at lower AC = second axis
        # they will have the same central AL, though
        # we can determine that: it's the AL-coordinate of the masked star pixel LOWEST in AC
        edgepix = starcoords[np.where(starcoords[:,1] == np.min(starcoords[:,1]))][0] # first should be accurate enough
        ghostlabels = np.unique(starlabels[edgepix[0],0:edgepix[1]])
        if len(ghostlabels) == 1:
            # Only found 0's, so nothing
            continue
        else:
            for glab in ghostlabels[1:]:
                # ghosts are above a certain size
                if np.sum(starlabels[starlabels==glab]/glab) >= 100:
                    starmask[starlabels==glab] = 1
                    
    # get rid of bad columns
    if badcol>=0:
        # get all the bad labels - we'll throw away everything connected to this
        badlabels = np.unique(starlabels[:,badcol])
        badlabels = badlabels[badlabels!=0] 
        for lab in badlabels:
            starmask[starlabels==lab] = 1

    # done!
    return starmask

def sm_starmask_old(image, threshold):
    """
    Construct a mask for all saturated stars in the SM image.

    This constructs a map of pixels above threshold, labels all connected pixels as one object, then searches for objects that contain at least one saturated pixel (i.e. above 65535 ADU).
    All pixels connected to those saturated pixels are then masked.
    """
    import numpy as np
    from scipy.ndimage import generate_binary_structure
    from scipy.ndimage.morphology import binary_dilation
    from scipy.ndimage.measurements import label, find_objects

    # get the extent of the image
    ymax, xmax = image.shape
    ymax -= 1
    xmax -= 1

    # occasionally, an overflow for very bright stars causes some pixels to be set to 0
    # -> set them to 65535, so they will be masked
    image[image==0] = 65535
    
    # initialize the mask to all False
    starmask = np.zeros(image.shape, dtype=bool)
     
    # mask of pixels above threshold
    satmap = np.logical_and(image,image>=threshold)

    # dilate it - sometimes, pixel values fluctuate around the threshold
    satmap = binary_dilation(satmap,iterations=1)

    # labelling all connected pixels
    (starlabels, nstars) = label(satmap,structure=(np.ones((3,3))))

    # we're only interested in the labels of pixels that are saturated
    # so, get all the labels that have that:
    satlabels = np.unique(starlabels[image==65535])
    
    if badcol>=0:
        # get all the bad labels - we'll throw away everything connected to this
        badlabels = np.unique(starlabels[:,badcol])
        badlabels = badlabels[badlabels!=0] 
        # add them to the satlabels
        satlabels = np.unique(list(satlabels) + list(badlabels))

    for lab in satlabels:
        starmask[starlabels==lab] = 1

        #return starmask - dilate it once (we won't be losing much here)
        return binary_dilation(starmask,iterations=1)


def sm_starmask_box(image, threshold, buffer_row=1, buffer_col=1):
    """
    [DEPRECATED] Construct a mask for all saturated stars in the SM image.

    This constructs a map of pixels above threshold, labels all connected pixels as one object and rejects all objects that do not contain at least one saturated pixel (i.e. above 65535 ADU).
    It then computes the PSF center of the given slices by analyzing the PSF and adds a rectangle with the maximum distance in x (AC) and y (AL)
    This distance is multiplied by buffer_row and buffer_col respectively and will not exceed the image dimensions.
    """
    import numpy as np
    from scipy.ndimage import generate_binary_structure
    from scipy.ndimage.morphology import binary_dilation
    from scipy.ndimage.measurements import label, find_objects
    

    # get the extent of the image
    ymax, xmax = image.shape
    ymax -= 1
    xmax -= 1

    # occasionally, an overflow for very bright stars causes some pixels to be set to 0
    # -> set them to 65535, so they will be masked
    image[image==0] = 65535
    
    # initialize the mask to all False
    starmask = np.zeros(image.shape, dtype=bool)
   
    # if nothing is saturated, return the starmask
    if np.max(image)<65535:
        return starmask
    else:
        # mask of pixels above threshold
        satmap = np.logical_and(image,image>=threshold)

        # dilate it - sometimes, pixel values fluctuate around the threshold
        satmap = binary_dilation(satmap,iterations=1)

        # labelling all connected pixels
        (starlabels, nstars) = label(satmap,structure=(np.ones((3,3))))

        # object extraction based on labels
        stars = find_objects(starlabels)

        # only use all "stars" that have at least one sample above 65535
        stars = [s for s in stars if np.max(image[s])==65535]


    # extract central points of stars and add their boxes to the mask 
    for star in stars:
        # bounding box of coordinates
        y0 = star[0].start
        y1 = np.min([star[0].stop,ymax])
        x0 = star[1].start
        x1 = np.min([star[1].stop,xmax])
        
        # center of psf, using its "wings" as a crosshair:
        cy = int((np.argmax(image[y0:y1,x0]) + np.argmax(image[y0:y1,x1]))/2.) + y0
        cx = int((np.argmax(image[y0,x0:x1]) + np.argmax(image[y1,x0:x1]))/2.) + x0
        
        # border distances from center of mass (symmetric for now)
        # these are modified by buffer_row and buffer_col
        dx = int(np.rint(max([cx-x0, x1-cx])*buffer_col))
        dy = int(np.rint(max([cy-y0, y1-cy])*buffer_row))
        
        # bounding box of mask
        xm0 = max([0, cx-dx])
        xm1 = min([xmax, cx+dx])
        ym0 = max([0, cy-dy])
        ym1 = min([ymax, cy+dy])

        # set everything in the box to 1
        starmask[ym0:ym1,xm0:xm1] = True
        
    return starmask




# extract evtlocs (the locations of events as a list of tuples) as well as the smothed background using scrappy
def sm_extract_evtlocs(image, gain, bias, readnoise, starmask):
    """
    Extract the locations of cosmic events from image.

    This uses astroscrappy to extract individual cosmics given a mask starmask. Gain, bias and readnoise are read from global parameters.
    Individual event locations are extracted using ndimage functions label and find_objects.

    Returns a list of slice-tuples for event locations, and the extracted cosmics with the background already subtracted.
"""
    import numpy as np
    import astroscrappy
    from scipy.ndimage.measurements import label, find_objects
    
    imbias = np.subtract(np.copy(image), np.copy(bias))

    # apply scrappy
    (mask,clean) = astroscrappy.detect_cosmics(imbias, gain=gain, verbose=False, inmask=starmask, satlevel=65535, readnoise=readnoise, sepmed=False, cleantype='medmask', fsmode='median')
    # labelling
    (labels, ntracks) = label(mask, structure=(np.ones((3,3))))

    # object extraction
    evtlocs = find_objects(labels)

    # calculate the cosmis
    cosmics = ((imbias)*gain - clean)*mask

    return evtlocs, cosmics


def sm_cosmics(source, gain, bias, readnoise, starmask, sigclip, sigfrac, objlim):
    """
    Docstring TBD
    """
    import numpy as np
    import astroscrappy
    import scipy.ndimage as ndimage
    from scipy.signal import convolve2d

    
    # construct mask with astroscrappy
    imbias = np.subtract(source, bias)

    # apply scrappy
    (mask,clean) = astroscrappy.detect_cosmics(imbias, gain=gain, verbose=False, inmask=starmask, 
                                           satlevel=65535, readnoise=readnoise, sepmed=False, 
                                           cleantype='meanmask', fsmode='median',
                                           sigclip=sigclip, sigfrac=sigfrac, objlim=objlim)
    
    # cosmic signal
    signal = ((imbias)*gain - clean)*mask*(1-starmask)
    
    # sometimes, the signal is actually lower than 0 - this is because the mean mask filter overestimated the background
    # the problem here is that there is most likely a star nearby, and this was either a false detection, or a (very) weak cosmic
    # So, there are two ways to solve this:
    #  a) identify non-saturated stars and add them to the starmask - not sure how feasible
    #  b) just throw them away - negative energy cosmics make no sense anyway
    mask[signal<0] = 0
    signal[signal<0] = 0
    

    # label cosmics
    (labels, ntracks) = ndimage.measurements.label(mask, structure=(np.ones((3,3))))

    # object extraction
    events = ndimage.measurements.find_objects(labels)
    
    
    
    # calculate the uncertainty of the signal (mean mask + counting noise)
    err_mean = np.zeros(source.shape)

    (xmax, ymax) = source.shape
    totmask = starmask+mask    # masked pixels, including stars
    totmask[totmask>1]=1      # there should be no overlap, but to be safe
    unmasked = (imbias)*gain * (1-totmask)

    rad = 2   # i.e. 2 for a 5x5 filter

    # We need to count the number of unmasked pixels in the filter and
    # the sum of unmasked pixel values in the filter
    # This can be done very easily using convolution!

    kernel = np.ones((2*rad+1,2*rad+1))

    N_unm = convolve2d(1-totmask, kernel, mode="same", boundary="fill", fillvalue=0)
    N_unm[N_unm==0]=1            # we'll be dividing by this later - this stops errors

    var_mean = convolve2d(unmasked, kernel, mode="same", boundary="fill", fillvalue=0) # sum up everything around
    var_mean = (readnoise*readnoise + var_mean/N_unm)/N_unm                            # from error propagation

    err_mean = np.sqrt(var_mean + readnoise*readnoise + np.abs(gain*imbias))              # total error
    # note: we only need to use errors in the MASKED region, since everything else
    #       was never replaced!
    # (Theoretically we didn't even need to calculate it there, but convolution is very fast)
    
    
    # our output is a TrackObs, containing the cosmic data and several keywords
    output = TrackObs(ntracks)
    
    output.source = "SM-SIF"
    output.srcAL = xmax
    output.srcAC = ymax
    output.maskpix = np.sum(starmask)
    output.gain = gain
    # aqcTime, row and fov need to be retrieved externally
    
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