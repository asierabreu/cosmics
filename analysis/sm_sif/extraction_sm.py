# things that are needed still

# bias, readnoise, gain 
# using BAM values for now
# a reasonable value for the buffer and the threshold when extracting cosmics

def sm_get_image(filename, calibfile):
    """
    Retrieve image data from filename and gain, bias and readnoise from calibfile.
    """
    from astropy.io import fits
    hdusource = fits.open(filename)

    row = hdusource[0].header['CCD_ROW']
    fov = int((hdusource[0].header['CCD'])[2])
    image = hdusource[0].data

    hdusource.close()


    hducal = fits.open(calibfile)

    ii = 2*(row-1) + fov -1    # index of the row of interest
    bias = hducal[1].data["BIAS"][ii]
    gain = hducal[1].data["GAIN"][ii]
    readnoise = hducal[1].data["RNOISE"][ii]

    hducal.close()
    
    return image, gain, bias, readnoise



def sm_starmask(image, threshold, buffer_row=1, buffer_col=1):
    """
    Construct a mask for all saturated stars in the SM image.

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
    
    imbias = np.subtract(image, bias)

    # apply scrappy
    (mask,clean) = astroscrappy.detect_cosmics(imbias, gain=gain, verbose=False, inmask=starmask, satlevel=65535, readnoise=readnoise, sepmed=False, cleantype='medmask', fsmode='median')
    # labelling
    (labels, ntracks) = label(mask, structure=(np.ones((3,3))))

    # object extraction
    evtlocs = find_objects(labels)

    # calculate the cosmis
    cosmics = ((imbias)*gain - clean)*mask

    return evtlocs, cosmics
