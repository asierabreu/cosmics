# TODO: Function to update a boxcar (i.e. use the boxcar as a FIFO)
# -> need to track the next pattern to replace and the current pattern of interest

def boxcar_signal(patterns, iip, readnoise):
    """ Extract background-subtracted signal and noise from a collection of bam patterns.
    Patterns should already be bias-subtracted and gain-corrected!

    patterns: numpy array of bam patterns
    iip: index of the signal pattern
    readnoise: ccd readnoise
    """
    import numpy as np 
    from astropy.stats import sigma_clip

    npatterns = patterns.shape[0]

    # get the background
    # compute the median over time, removing outliers via sigma clipping
    # TODO: Value for sigma
    bkg_src = sigma_clip(patterns, sigma=2, iters=None, axis=0)

    background = np.mean(bkg_src, axis=0)

    # extract signal
    signal = patterns[iip] - background


    # compute uncertainty
    (xmax, ymax) = patterns[0].shape
    err_mean = np.zeros((xmax,ymax))

    # Number of elements we averaged over
    N_time = (npatterns)-np.sum(bkg_src.mask.astype("int"),axis=0)

    # variance of background, from error propagation
    var_mean = (readnoise*readnoise + np.sum(no_cosm,axis=0)/N_time)/N_time

    # total error (background + signal
    err_mean = np.sqrt(var_mean + readnoise*readnoise + patterns[iip,:,:]) 

    return signal, err_mean

# TODO function to extract cosmics
# Algorithm TBD, probably either laplacian detection or median clipping
