import os
import sys
#root = os.environ['HOME']
#sys.path.insert(0, os.path.abspath(root+'/spark/libs/astroscrappy/astroscrappy/'))
from TrackObs import *
import gbin_reader
import astroscrappy
import numpy as np
import scipy.ndimage as ndimage
from astropy.stats import sigma_clip
from astropy.time import Time

### BoxCar class
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
        self.i_sig = -1
        
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
        #import numpy as np 
        #from astropy.stats import sigma_clip
        
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

### Determine the OBMT of the first observation in a gbin
def get_obmt_gbin(path):
    reader = gbin_reader.GbinReader(path)
    obs = reader.__next__()
    acqTime = obs.acqTime
    reader.close()

    return acqTime
    
### The core algorithm - extracting cosmics from a signal

def bam_cosmics_mended(signal, err_mean, threshold, threshfrac, gain):
    """
    Docstring TBD
    """

    #import numpy as np
    #import astroscrappy
    #import scipy.ndimage as ndimage
    
    
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
    
    
### BamObservation reading and extraction of TrackObs

def decode_BamObs(obs, bias, gain):
    """
    Decodes the bamobservation obs and returns the extracted pattern and OBMT
    The extracted pattern is masked for invalid pixels

    obs: a BamObservation object
    bias: bias value
    gain: gain value
    """
    #import numpy as np
    #import gbin_reader  # need to include this in the import path

    acqTime = obs.acqTime
    pattern = (np.array(obs.samples).reshape(1000,80))
    epattern = ((pattern-bias)*gain).astype('float64')  # pattern in electrons
    
    # in some cases, cosmics can cause an overflow in the pattern, so that some neighbouring pixels are set to 0 ADU
    # we need to mask these!
    # let's try to do an actual masked array
    arrmask = np.zeros(pattern.shape).astype("bool")
    arrmask[pattern==0] = True
    epattern = np.ma.masked_array(epattern,mask=arrmask,fill_value=0) # may need to rethink fill value
    
    return epattern, acqTime


def BoxCar_extract_TrackObs(boxcar,gain,readnoise):
    # get signal
    sig, err = boxcar.get_signal(readnoise)
    # get the TrackObs
    tobs = bam_cosmics_mended(sig, err, 5, 0.4, gain)
    # update the keys of output
    tobs.acqTime = boxcar.acqTimes[boxcar.i_sig]
    tobs.row=1  # this is always true for BAM
    tobs.fov=boxcar.fov
    tobs.gain=gain

    return tobs


### Writing of TrackObs
    
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


def get_obmt_gbin(path):
    reader = gbin_reader.GbinReader(path)
    obs = reader.__next__()
    acqTime = obs.acqTime
    reader.close()

    return acqTime



def write_badfile(path, root_dir):
    """Create an extra textfile and write this path into it"""
    # get obmt for a unique filename
    reader = gbin_reader.GbinReader(path)
    obs = reader.__next__()
    acqTime = obs.acqTime
    reader.close()
    
    # make an extra folder for this one
    badpath = os.path.abspath(root_dir) + "/IRREGULAR_FILES"
    os.makedirs(badpath,exist_ok=True)
    
    # write a one-line file with this path
    fname = badpath + "/" + str(int(acqTime)) + ".txt"
    with open(fname, 'w') as the_file:
        the_file.write(path)


    
def TrackObs_list_writer_BAM(obslist, root_dir):
    """
    Writes the obslist into a single fits file.
    The location of the file will be "<root_dir>/yy-mm-dd/BAM-OBS<FOV>_OBMT_START_<OBMT of first element of obslist>.fits"
    The folder of the day will be created, if necessary
    """
    #import os
    
    # get the OBMT_START and date of the first TrackObs
    starttime = obslist[0].acqTime
    apytime = OBMT_apyTime(starttime)
    
    # the fov as well
    fov = obslist[0].fov
    
    # determine the path to write to
    path = os.path.abspath(root_dir) + "/{:4d}/{:02d}/{:02d}".format(apytime.datetime.year, apytime.datetime.month, apytime.datetime.day)
    
    # make the path, if necessary 
    os.makedirs(path,exist_ok=True)
    
    # make the filename and write it
    filename = "BAM-OBS{}_OBMT_START_{}.fits".format(fov,int(starttime))
    write_Obslist(obslist, path+'/'+filename)
    print("================\nWrote", filename,"\n================")
    

def process_BAM_OBS(pathgroup, pathlist, write_grouping, root_dir):
    """
    Extracts and writes TrackObservations from a pathgroup, and writes them to fits files

    pathlist: A list of paths to BamObservation gbins, sorted in time
    pathgroup: A tuple of of two ints (start,stop) indicating which paths should be handled here (inclusively), and a single path indicating the previous path for doing a delta. If there is no previous path, it should be an empty string.
    write_grouping: Number of TrackObs to write per fits file
    root_dir: The location of the file will be "<root_dir>/yy-mm-dd/BAM-OBS<FOV>_OBMT_START_<OBMT of first element of obslist>.fits"
    The folder of the day will be created, if necessary
    """
    # chip parameters
    gain=3.853           # gain [e-/ADU]
    readnoise=8.365785291814616 # readout noise [e-]
    bias=2576            # bias [ADU]

    # get the paths
    istart = pathgroup[0][0]
    istop = pathgroup[0][1]
    prevpath = pathgroup[1]
    
    npaths = istop - istart
    npaths_total = len(pathlist)
    
        
    ### cosmic extraction

    # build two boxcars, one per fov
    boxrad = 3
    boxcar1 = BoxCar(boxrad=boxrad, fov=1)
    boxcar2 = BoxCar(boxrad=boxrad, fov=2)
    # buffers for safety
    buffer1 = []
    buffer2 = []
    
    i_path = istart

    
    print("================\nStep 1\n================")
    
    ######################### Step 1: Boxcar filling
    # fill the boxcars. What we want is to get each boxcar exactly full, 
    # so that we can immediately do one extraction
    
    # handle the previous file if necessary
    if prevpath != '':
        reader = gbin_reader.GbinReader(prevpath)
        # keep on updating the boxcars until the end of the file
        # so, essentially use them as a fifo
        while True:
            try:
                obs = reader.__next__()
            except StopIteration:
                reader.close()
                break
            if obs.fov==0:
                try:
                    pattern, acqTime = decode_BamObs(obs,bias,gain)
                    boxcar1.update(pattern, acqTime)
                    if boxcar1.nfilled < boxcar1.npatterns: 
                        boxcar1.nfilled+=1
                except ValueError:
                    continue
            elif obs.fov==1:
                try:
                    pattern, acqTime = decode_BamObs(obs,bias,gain)
                    boxcar2.update(pattern, acqTime)
                    if boxcar2.nfilled < boxcar2.npatterns:
                        boxcar2.nfilled+=1
                except ValueError:
                    continue

    # if the both boxcars are not yet full, you need to continue with the first file
    # this is true by default if there is no previous path
    reader = gbin_reader.GbinReader(pathlist[i_path])

    # this also has to happen at least once, even in the case of a previous path, for both FOVs!
    # since if we were to get the signal right now, it would be contained in the previous run
    # so, emulate a do-while loop
    first_pass1 = True
    first_pass2 = True
    while first_pass1 or first_pass2 or (boxcar1.nfilled < boxcar1.npatterns) or (boxcar2.nfilled < boxcar2.npatterns):
        try:
            obs = reader.__next__()
        except StopIteration:
            reader.close()
            i_path += 1
            if i_path <= istop:
                reader = gbin_reader.GbinReader(pathlist[i_path])
                continue
            else:
                break
                
        if obs.fov==0:
            if (boxcar1.nfilled < boxcar1.npatterns) or first_pass1:
                try:
                    pattern, acqTime = decode_BamObs(obs,bias,gain)
                    boxcar1.update(pattern, acqTime)
                    if (boxcar1.nfilled < boxcar1.npatterns):
                        boxcar1.nfilled+=1
                    first_pass1 = False
                except ValueError:
                    continue
            else:
                buffer1.append(obs)
                
        elif obs.fov==1:
            if (boxcar2.nfilled < boxcar2.npatterns) or first_pass2:
                try:
                    pattern, acqTime = decode_BamObs(obs,bias,gain)
                    boxcar2.update(pattern, acqTime)
                    if (boxcar2.nfilled < boxcar2.npatterns):
                        boxcar2.nfilled+=1
                    first_pass2 = False
                except ValueError:
                    continue
            else:
                buffer2.append(obs)

     # set the i_sig - it must be its current value minus boxrad, wrapping around
    for car in [boxcar1,boxcar2]:
        i_sig = car.i_sig
        i_sig -= car.boxrad
        if i_sig <0:
            i_sig = car.npatterns + i_sig
        car.i_sig = i_sig
           
        
    # if for some reason the boxcars are not yet full, quit out
    if (boxcar1.nfilled < boxcar1.npatterns) or (boxcar2.nfilled < boxcar2.npatterns):
        return
    
    print("================\nStep 2\n================")
    ######################### Step 2: Extraction in the given interval
    # Now the actual extraction
    # the outputs to be written
    outlist1 = []
    outlist2 = []
    
    # since the boxcar is full, we can already do one extraction
    outlist1.append(BoxCar_extract_TrackObs(boxcar1,gain,readnoise))
    outlist2.append(BoxCar_extract_TrackObs(boxcar2,gain,readnoise))

    # first empty the buffers, if there's something in there
    while len(buffer1)>0:
        try:
            pattern,acqTime = decode_BamObs(buffer1.pop(0),bias,gain)
            boxcar1.update(pattern, acqTime)
            outlist1.append(BoxCar_extract_TrackObs(boxcar1,gain,readnoise))
        except ValueError:
            continue
    while len(buffer2)>0:
        try:
            pattern,acqTime = decode_BamObs(buffer2.pop(0),bias,gain)
            boxcar2.update(pattern, acqTime)
            outlist2.append(BoxCar_extract_TrackObs(boxcar2,gain,readnoise))
        except ValueError:
            continue
           
    # then step over the pathlist
    while i_path <= istop:
        try:
            obs = reader.__next__()
        except StopIteration:
            # this file is finished
            reader.close()
            # get the next or stop
            i_path += 1
            if i_path > istop:
                break  # we're done with the interval
            else:
                # start with the next file and go back to start
                reader = gbin_reader.GbinReader(pathlist[i_path])
                continue
        # process one step of the boxcar the obs belongs to
        if obs.fov == 0:
            try:
                pattern,acqTime = decode_BamObs(obs,bias,gain)
                boxcar1.update(pattern, acqTime)
                outlist1.append(BoxCar_extract_TrackObs(boxcar1,gain,readnoise))
                # write if necessary
                if len(outlist1) >= write_grouping:
                    TrackObs_list_writer_BAM(outlist1, root_dir)
                    # empty the list after writing
                    outlist1 = []
            except ValueError:
                continue

        elif obs.fov == 1:
            try:
                pattern,acqTime = decode_BamObs(obs,bias,gain)
                boxcar2.update(pattern, acqTime)
                outlist2.append(BoxCar_extract_TrackObs(boxcar2,gain,readnoise))
                # write if necessary
                if len(outlist2) >= write_grouping:
                    TrackObs_list_writer_BAM(outlist2, root_dir)
                    # empty the list after writing
                    outlist2 = []
            except ValueError:
                continue
                
                
    print("================\nStep 3\n================")
    ######################### Step 3: Work until our next executor
    
    # process the first 2*boxrad BamObservations after our interval,
    # since these can not be processed by anyone else
    
    # if we're at the end of all paths already, don't do the next paths
    if i_path >= npaths_total:
        pass
    else:
        overlap1 = 0
        overlap2 = 0
        
        reader = gbin_reader.GbinReader(pathlist[i_path])
        while (overlap1 < 2*boxrad) or (overlap2 < 2*boxrad):
            try:
                obs = reader.__next__()
            except StopIteration:
                reader.close()
                # get the next file, if there is one
                i_path += 1
                if i_path >= npaths_total:
                    break
                else:
                    # start with the next file and go back to start
                    reader = gbin_reader.GbinReader(pathlist[i_path])
                    continue
                        
            # process one step of the boxcar the obs belongs to
            # but only for the first 2*boxrad observations per fov!
            if obs.fov == 0 and overlap1 < 2*boxrad:
                try:
                    pattern,acqTime = decode_BamObs(obs,bias,gain)
                    boxcar1.update(pattern, acqTime)
                    outlist1.append(BoxCar_extract_TrackObs(boxcar1,gain,readnoise))
                    # write if necessary
                    if len(outlist1) >= write_grouping:
                        TrackObs_list_writer_BAM(outlist1, root_dir)
                        # empty the list after writing
                        outlist1 = []
                    overlap1 += 1
                except ValueError:
                    continue
            elif obs.fov == 1 and overlap2 < 2*boxrad:
                try:
                    pattern,acqTime = decode_BamObs(obs,bias,gain)
                    boxcar2.update(pattern, acqTime)
                    outlist2.append(BoxCar_extract_TrackObs(boxcar2,gain,readnoise))
                    # write if necessary
                    if len(outlist2) >= write_grouping:
                        TrackObs_list_writer_BAM(outlist2, root_dir)
                        # empty the list after writing
                        outlist2 = []
                    overlap2 += 1
                except ValueError:
                    continue            

    # write one last time if necessary
    if len(outlist1) > 0:
        TrackObs_list_writer_BAM(outlist1, root_dir)
    if len(outlist2) > 0:
        TrackObs_list_writer_BAM(outlist2, root_dir)

    # and we're done!
