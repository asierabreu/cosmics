"""This file includes a collection of useful functions for analyzing the output
of the Gaia Cosmics analysis"""

import os
import sys

sys.path.append("../lib/TrackObs/")
sys.path.append("../lib/CosmicObservation/")

from CosmicObservation import *
from external_flux import *
from astropy.io import fits

import re


"""File selection functions"""

getobmt = lambda x: int(re.search("(?<=OBMT_START_)[0-9]+",x).group(0))

def psel(allpaths,start,stop, unit):
    """Select the paths in allpaths that have been obtained between start and stop
    The unit can be set to "OBMT", "REV", or "MJD" """
    pathtimes = np.array([getobmt(p) for p in allpaths])
    if unit == "OBMT":
        pass
    elif unit == "REV":
        pathtimes /= 1e9*3600*6
    elif unit == "MJD":
        pathtimes = OBMT_apyTime(pathtimes).mjd
    else:
        raise TypeError("Unit not defined")
        
    indices = np.where(np.logical_and(pathtimes>=start, pathtimes<=stop))[0]
    return [allpaths[ii] for ii in indices]



"""Energy Analysis"""

def CosObs_get_energies(filenames,edgeless=False,row=-1,fov=-1):
    """Return a dictionary 
        EN: [energies per observation (one list per obs)]
        FLUX: [list of flux per observation]
    """
    
    # Create output dict
    outdict = {'EN':[], 'FLUX':[]}
    
    # if the input is not a list, it has to be a single filename
    if type(filenames) != type([]):
        filenames = [filenames]
    
    for filename in filenames:
        # Open the file
        hdulist = fits.open(filename)
        
        # Iterate over all the CosmicObservations
        for ii in range(1,len(hdulist),1):
            head = hdulist[ii].header
            
            # filter if necessary
            if fov > 0:
                if head["FOV"] != fov: continue
            if row > 0:
                if head["CCD_ROW"] != row: continue
                    
            # write flux
            outdict['FLUX'].append(head["FLUX"])
            
            if edgeless:
                edgecosmics = hdulist[ii].data["TRACK_TRUNCATED"]
                # write energies and rejected
                outdict['EN'].append( list((hdulist[ii].data["TRACK_EN"])[np.logical_not(edgecosmics)]) )
            else:
                # write energies and rejected
                outdict['EN'].append( list(hdulist[ii].data["TRACK_EN"]) )

        # Close the file
        hdulist.close()
    return outdict

def energyHist_fluxnormed(obsEn, obsFluxes, bins=None, enrange=None):
    """
    Given a list of energy lists and a list of the fluxes of the corresponding observations,
    make a normed energy histogram - each observation is normed by its geometry factor and then averaged over
    The output histogram is counts/cm^2/s/eV
    Returns bin heights, lower bin edges and the step size
    """
    nobs = len(obsFluxes)
    Eflat = sum(obsEn, [])
    normflat = np.empty(len(Eflat))
    
    # determine the weights
    start = 0
    for ii in range(nobs):
        nParticles = len(obsEn[ii])
        normflat[start:start+nParticles] = obsFluxes[ii]/nParticles
        start += +nParticles
    
    normflat /= nobs
    # compute the weighted histogram
    histout = np.histogram(Eflat,bins=bins,range=enrange, weights=normflat)
    
    # divide by bin size
    step = histout[1][1] - histout[1][0]
    #step=1
    
    return histout[0]/step, histout[1][:-1], step



"""Geometry Analysis"""

def CosObs_get_geometries(filenames,edgeless=False,row=-1,fov=-1):
    """Return a dictionary 
        THETA: [theta angles, for valid tracks (one list per obs)]
        THETA_ERR: [theta errors, for valid tracks (one list per obs)]
        LEN: [track lengths, for valid tracks (one list per obs)]
        LEN_ERR: [length errors, for valid tracks (one list per obs)]
        EN: [energies per observation, for valid tracks (one list per obs)]
        FLUX: [list of flux per observation]
        VALFRAC: [list of fraction of used valid tracks, per observation]
    """
    
    # Create output dict
    outdict = {'EN':[], 'THETA':[], 'THETA_ERR':[], 'LEN':[], 'LEN_ERR':[], 'FLUX':[], 'VALFRAC':[]}
    
    # if the input is not a list, it has to be a single filename
    if type(filenames) != type([]):
        filenames = [filenames]
    
    for filename in filenames:
        # Open the file
        hdulist = fits.open(filename)
        
        # Iterate over all the CosmicObservations
        for ii in range(1,len(hdulist),1):
            head = hdulist[ii].header
            
            # filter if necessary
            if fov > 0:
                if head["FOV"] != fov: continue
            if row > 0:
                if head["CCD_ROW"] != row: continue
                    
            # write flux
            outdict['FLUX'].append(head["FLUX"])
            
            # which cosmics to take
            valids = hdulist[ii].data["GEOMETRY_VALID"]
            if edgeless:
                edgecosmics = hdulist[ii].data["TRACK_TRUNCATED"]
                valids = np.logical_and(valids, np.logical_not(egecosmics))

            # get data
            outdict['THETA'].append( list(hdulist[ii].data["TRACK_THETA"][valids]) )
            outdict['THETA_ERR'].append( list(hdulist[ii].data["TRACK_THETA_ERR"][valids]) )
            outdict['LEN'].append( list(hdulist[ii].data["TRACK_LEN"][valids]) )
            outdict['LEN_ERR'].append( list(hdulist[ii].data["TRACK_LEN_ERR"][valids]) )
            outdict['EN'].append( list(hdulist[ii].data["TRACK_EN"][valids]) )
            outdict['VALFRAC'].append( np.sum(valids)/head["NAXIS2"])

        # Close the file
        hdulist.close()
    return outdict

def spread_thetahist(GeoData, nbins):
    """Constructs an angle histogram from -90 to 90 degrees,
    which spreads each angle out according to its error,
    and normed by the utilized flux
    The units of heights will be in Counts/deg/cm^2/s"""
    
    heights = np.zeros(nbins)
    bin_lo = np.linspace(-90,90,nbins,endpoint=False)
    bin_width = bin_lo[1]-bin_lo[0]
    nobs = len(GeoData["FLUX"])
    
    for iobs in range(nobs):
        theta = np.array(GeoData["THETA"][iobs])
        sd_theta = np.array(GeoData["THETA_ERR"][iobs])
                
        # the histogram bin heights for this observation
        subheights = np.zeros(nbins)
        
        # each particle is spread over every bin that is within theta +- sd_theta
        minvals = theta-sd_theta
        maxvals = theta+sd_theta
        
        # iterate over the particles
        nparts = len(theta)
        
        for ii in range(nparts):
            add_range = np.logical_and(bin_lo+bin_width >= minvals[ii], bin_lo <= maxvals[ii])
            if maxvals[ii] > 90:
                add_range += bin_lo <= maxvals[ii]-180
            if minvals[ii] < -90:
                add_range += bin_lo+bin_width >= minvals[ii]+180

            subheights[add_range] += 1/np.sum(add_range)

        # per observation: The sum of subheights should equal the USED flux
        norm = GeoData["FLUX"][iobs] * GeoData["VALFRAC"][iobs] / nparts
        subheights *= norm
        
        heights += subheights
        
    # heights is averaged over all observations, and bin widths need to enter the norm
    heights /= nobs * bin_width
    
    return heights, bin_lo, bin_width


def dEhist_fluxnormed(GeoData, bins, histrange, selectors):
    """Constructs a histogram for dE/dx
    with a specified number of bins and range of values (histrange=(start,stop))
    and normed by the utilized flux.
    
    Selectors is a list of numpy boolean numpy arrays, specifiying for
    each observation in GeoData which cosmics are to be used - for example, for selecting angles.
    
    The units of heights will be in Counts/deg/cm^2/s/[MeV cm^2 g^-1]
    """
    nobs = len(GeoData['FLUX'])
    Eflat = sum(GeoData["EN"],[])
    lenflat = sum(GeoData["LEN"],[])
    thetaflat = sum(GeoData["LEN"],[])
    
    dEdx = []
    normflat = []
    
    # determine the weights
    for ii in range(nobs):
        selector = selectors[ii]
        if selector.all == False:
            continue
        
        nParticles = len(GeoData["THETA"][ii])
        nSelected = np.sum(selector)
        dEdx += list(np.array(GeoData["EN"][ii])[selector] / np.array(GeoData["LEN"][ii])[selector] *3.68/100/2.32)
        # the norm should be such that if we sum over the histogram, we get the flux utilized by our algorithm,
        # i.e. after both the valid geometries AND the selector, and averaged over observations
        normflat += list(np.ones(nSelected) *  GeoData["FLUX"][ii] * GeoData["VALFRAC"][ii] * nSelected/nParticles
                    / nParticles / nobs)
        
    # compute the weighted histogram
    histout = np.histogram(dEdx,bins=bins, range=histrange, weights=normflat)
    
    # divide by bin size
    step = histout[1][1] - histout[1][0]
    
    return histout[0]/step, histout[1][:-1], step


"""Flux Analysis"""

def rebin_fluxes(arrs,binning):
    """
    Rebin time,fluxes and flux errors by averaging over 'binning' samples
    flux errors (the third tuple element) uses error propagation
    """
    
    if binning==1:
        return arrs
    
    out = []
    
    for ii in range(len(arrs)):
        len_rebin = len(arrs[ii])//binning  # integer division! 
                                            # we want to round down, so we ignore the last up to binning-1 samples
        a_rebin = np.reshape(arrs[ii][:len_rebin*binning], (len_rebin,binning))
        
        if ii==2:
            a_rebin = np.sqrt(np.sum(a_rebin**2,axis=1))/binning
        else:
            a_rebin = np.mean(a_rebin,axis=1)
        
        out.append(a_rebin)

    return tuple(out)

def MJD_cutout(t,f,err,mjd_start,mjd_stop,stepping=1):
    """
    Return time [mjd], flux and error within a certain inclusive range of mjd.
    By setting stepping=n, this will only return every nth sample
    """
    retrange = np.logical_and(t>=mjd_start, t<=mjd_stop)
    
    if stepping < 1:
        stepping = 1
    
    return t[retrange][::stepping], f[retrange][::stepping], err[retrange][::stepping]