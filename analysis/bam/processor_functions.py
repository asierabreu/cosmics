from extraction_bam import *


### BamObservation reading and extraction of TrackObs

def decode_BamObs(obs, bias, gain):
    """
    Decodes the bamobservation obs and returns the extracted pattern and OBMT
    The extracted pattern is masked for invalid pixels

    obs: a BamObservation object
    bias: bias value
    gain: gain value
    """
    import numpy as np
    import gbin_reader  # need to include this in the import path

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

def BamObs_extractor(obslist):
    """
    Extracts TrackObservations out of a list of BamObservations
    
    obslist: Time ordered list of BamObservation Objects, all having the same FOV
    returns: A list of TrackObs, corresponding to oblist[boxrad:len(obslist-boxrad)]
    """
    
    # chip parameters
    gain=3.853           # gain [e-/ADU]
    readnoise=8.365785291814616 # readout noise [e-]
    bias=2576            # bias [ADU]
    
    n_input = len(obslist)
    fov = obslist[0].fov+1 # the gbins go 0 to 1
    
    
    # set up the boxcar for this extraction
    boxrad = 3
    boxcar = BoxCar(boxrad=boxrad, fov=fov)
    boxlen = boxcar.npatterns
    
    # fill the boxcar with the first samples
    for ii in range(boxlen):
        (pattern, acqTime) = decode_BamObs(obslist[ii], bias, gain)
        boxcar.update(pattern, acqTime)
        boxcar.nfilled+=1
    
    # set the i_sig
    boxcar.i_sig = boxcar.boxrad
    
    # remember the index of the next BamObservation to take
    i_next = boxlen
    
    
    # start the extraction
    outlist = [0 for ii in range(n_input - 2*boxrad)]  # the output list of TrackObs
    i_out = 0    
    # end condition: there is no next BamObservation to take
    while True:
        ## get the TrackObs and save it
        # get signal
        sig, err = boxcar.get_signal(readnoise)
        # get the TrackObs
        tobs = bam_cosmics_mended(sig, err, 5, 0.4, gain)
        # update the keys of output
        tobs.acqTime = boxcar.acqTimes[boxcar.i_sig]
        tobs.row=1  # this is always true for BAM
        tobs.fov=boxcar.fov
        tobs.gain=gain
        # put it into our output
        outlist[i_out] = tobs
        i_out += 1
        
        ## update the boxcar for the next iteration, if there is more to get
        if i_next < n_input:
            (pattern, acqTime) = decode_BamObs(obslist[i_next], bias, gain)
            boxcar.update(pattern, acqTime)

            # update i_next
            i_next += 1
        else:
            break # we're done
    
    return outlist
    
    
# Writing of TrackObs
    
def OBMT_apyTime(obmt_in):
    """Assign a given OBMT to astropy time objects"""
    from astropy.time import Time
    # reference time: after first reset
    unix_ref = 1393445605
    obmt_ref = 10454400000000000
    
    unix_out = unix_ref + (obmt_in - obmt_ref)/1e9
    
    out = Time(unix_out, format='unix')
    out.format = 'isot' # may not matter
    
    return out


    
def TrackObs_list_writer(obslist, root_dir):
    """
    Writes the obslist into a single fits file.
    The location of the file will be "<root_dir>/yy-mm-dd/BAM-OBS<FOV>_OBMT_START_<OBMT of first element of obslist>.fits"
    The folder of the day will be created, if necessary
    """
    import os
    
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