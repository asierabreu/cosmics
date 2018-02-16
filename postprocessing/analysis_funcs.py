import os
import sys
sys.path.insert(0, os.path.abspath('../lib/TrackObs/'))
from TrackObs import *

from astropy.io import fits

### energies


def fits_energies_edgeless(filenames, row=-1,fov=-1):
    """
    Extract deposited energies from a list of fits files of TrackObs
    Returns:
    - An array of acqTimes [OBMT] 
    - A list of energy arrays [eV] - one array per acqTime
    - A list boolean arrays for each acqTime, which are TRUE when a cosmic is at the edge of the image.
    
    Setting row and/or fov to a value greater than 0 filters the input arrays
    """
    # Create output lists
    acqTimes = []
    energies = []
    rejected = []
    
    # if the input is not a list, it has to be a single filename
    if type(filenames) != type([]):
        filenames = [filenames]
    
    for filename in filenames:
        # Open the file
        hdulist = fits.open(filename)
        
        # Iterate over all the TrackObs
        for ii in range(1,len(hdulist),1):
            head = hdulist[ii].header
            
            # filter if necessary
            if fov > 0:
                if head["FOV"] != fov: continue
            if row > 0:
                if head["CCD_ROW"] != row: continue
                    
            # write acquisition time
            acqTimes.append(head["ACQTIME"])

            # get size of the image
            al_max = head["SRC_AL"]
            ac_max = head["SRC_AC"]

            # get the edges of all the cosmics
            beg_al = hdulist[ii].data["LOC_AL"]
            beg_ac = hdulist[ii].data["LOC_AC"]
            end_al = beg_al + hdulist[ii].data["DIM_AL"]
            end_ac = beg_ac + hdulist[ii].data["DIM_AC"]

            edgecosmics = np.logical_or(np.logical_or(beg_al == 0, end_al == al_max), 
                                        np.logical_or(beg_ac == 0, end_ac == ac_max))
            # write energies and rejected
            energies.append(hdulist[ii].data["TRACK_EN"])
            rejected.append(edgecosmics)

        # Close the file
        hdulist.close()
    return np.array(acqTimes), energies, rejected



def concatenate_energies(e_in, indices):
    """
    Turn the entries of e_in at indices into a flat list.
    """
    e_out = []
    for ii in indices:
        e_out += list(e_in[ii])
    return e_out

def concatenate_reject_energies(e_in, rej, indices):
    """
    Turn the entries of e_in at indices into a flat list, rejecting indicated in rej.
    """
    e_out = []
    for ii in indices:
        e_out += list((e_in[ii])[(np.logical_not(rej[ii]))])
    return e_out



### fluxes

def fits_flux(filenames, row=-1, fov=-1):
    """
    Extract fluxes from a list of fits files of TrackObs
    Returns an array of acqTimes [OBMT] the number of particles, fluxes [parts/cm^2/s] and the poissonian uncertainty
    
    Setting row and/or fov to a value greater than 0 filters the input arrays
    """
    # Create lists for OBMT and fluxes
    acqTimes = []
    nparts = []
    fluxes = []
    errs = []
    
    # if the input is not a list, it has to be a single filename
    if type(filenames) != type([]):
        filenames = [filenames]
    
    # iterate over files
    for filename in filenames:
        # Open the file
        hdulist = fits.open(filename)

        # Iterate over all the track-observations
        for ii in range(1,len(hdulist),1):
            head = hdulist[ii].header        
            
            # filter if necessary
            if fov > 0:
                if head["FOV"] != fov: continue
            if row > 0:
                if head["CCD_ROW"] != row: continue
                    
            maskpix = head["MASKPIX"]
            srcAL = head["SRC_AL"]
            srcAC = head["SRC_AC"]

            source = head["SOURCE"]
            # this bit could be put outside the loop if we are SURE that a fits file only has one source
            # I think it should, but this should be fairly quick anyhow
            if source in ["BAM-OBS","BAM-SIF"]:
                # 1 x 4 binning
                pixAL = 10e-4
                pixAC = 120e-4
                exptime = 4.5 * 0.9828 + 19
            elif source == "SM-SIF":
                # 2 x 2 binning
                pixAL = 20e-4
                pixAC = 60e-4
                exptime = 2.9 * 0.9828

            exparea = pixAL*pixAC*(srcAL*srcAC - maskpix)
            
            if exparea == 0:
                fluxes.append(0)
                errs.append(0)
            else:
                fluxes.append(head["NAXIS2"]/exparea/exptime)
                errs.append(np.sqrt(head["NAXIS2"])/exparea/exptime)
                
            nparts.append(head["NAXIS2"])
            acqTimes.append(head["ACQTIME"])

        # Close the file
        hdulist.close()
    
    # Convert output to arrays and return
    acqTimes = np.array(acqTimes,dtype=int)
    nparts = np.array(nparts,dtype=int)
    fluxes = np.array(fluxes,dtype=float)
    errs = np.array(errs,dtype=float)
    
    return acqTimes, nparts, fluxes, errs

def PPE_flux(filename):
    """
    Extract fluxes from a PPE file.
    Returns an array of acqTimes [OBMT] and fluxes [parts/cm^2/s] and poissonian errors
    """
    # get the times and nr of particles
    acqTimes = []
    fluxes = []
    
    ii = 0
    with open(filename,'r') as pf:
        for line in pf:
            # skip the first line
            if ii == 0:
                ii += 1
                continue
            line = line.split()
            acqTimes.append(int(line[0]))
            fluxes.append(float(line[4]))
    
    # sort everything by OBMT
    zipped = zip(acqTimes, fluxes)
    zipped = sorted(zipped, key=lambda t: t[0])
    
    # convert the nr. of particles to a flux
    fluxes = np.array([z[1] for z in zipped])
    
    exparea = 17.1 # cm^2 from Asier's TN
    exptime = 257.635 # s length of an ASD4 packet, from the TN
    errs = np.sqrt(fluxes)/exparea/exptime
    fluxes = fluxes/exparea/exptime
    
    return np.array([z[0] for z in zipped]), fluxes, errs