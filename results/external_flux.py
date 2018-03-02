def read_flux_external(filenames):
    """
    Takes a list of external flux filenames and returns the acquisition times and fluxes in different bands.
    """
    import numpy as np
    import csv
    from astropy.time import Time
    
    # read in the raw data - all strings
    rawdat = []
    for fname in filenames:
        with open(fname, 'r') as csvfile:
            reader = csv.reader(csvfile, delimiter=' ')
            for row in reader:
                rawdat.append(row)
                
    # convert the first columns (i.e. the timestamp) into an astropy time object
    times = [rawdat[ii][0] for ii in range(len(rawdat))]
    times = np.array([t[:-2] + ':' + t[-2:] + ':00' for t in times])
    times = Time(times, format='isot', scale='utc')
    
    # turn all the fluxes into one array
    ncols = len(times)
    fluxes = np.transpose(np.array([rawdat[ii][1:ncols] for ii in range(len(rawdat))]).astype('float'))
    
    # return only valid fluxes
    return times[fluxes[0]>=0], fluxes[:,fluxes[0]>=0]

def OBMT_apyTime(obmt_in):
    """Assign a given OBMT to astropy time objects"""
    import numpy as np
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