# Functions to get all the SM-SIF files outside the calibration activities, with absolute paths and not separated by FOV.

# input
import argparse
import os

# input parsing
parser = argparse.ArgumentParser(description='Retrieve filenames of SM-SIF observations, excluding OBMTs specified in script')
parser.add_argument('-d', '--dataRoot', help='Root path to search')
parser.add_argument('-o', '--outputFile', help='File to write to')
args = parser.parse_args()

dataroot = os.path.abspath(args.dataRoot)
outfilename = args.outputFile
    
# Here, we have all the times we need to exclude due to calibration activities

# CI_IN_SM - this applies to ALL rows
OBMT_CI = [(6.81042889094578E+015,6.81522889094786E+015),
           (1.78191982102836E+016,1.78233982102849E+016),
           (4.29615982403429E+016,4.29657982403473E+016),
           (5.44455992558824E+016,5.44539992558934E+016),
           (6.01479992631641E+016,6.01563992631747E+016),
           (7.0170399276483E+016,7.01787992764946E+016),
           (8.07111992907961E+016,8.0719599290808E+016),
           (8.78175993010565E+016,8.78259993010686E+016),
           (9.85959993166397E+016,9.86001993166458E+016),
           (1.10173600334343E+017,1.10178400334351E+017),
           (1.2062800035135E+017,1.20632200351357E+017)
          ]

# FVBS - Forward bias, row-wise
FVBS1 = [(6.70902889090192E+015,6.71532889090465E+015),
         (1.36233982073171E+016,1.36305982073196E+016),
         (5.11626982515722E+016,5.11634182515731E+016),
         (9.24618993078481E+016,9.2467299307856E+016),
         (1.1967790034981E+017,1.19683300348819E+017)
        ]

FVBS2 = [(6.70902889090192E+015,6.71532889090465E+015),
         (1.36233982073171E+016,1.36305982073196E+016),
         (5.13354982517898E+016,5.13362182517907E+016),
         (9.24618993078481E+016,9.2467299307856E+016),
         (1.1967790034981E+017,1.19683300348819E+017)
        ]

FVBS3 = [(6.70902889090192E+015,6.71532889090465E+015),
         (1.36233982073171E+016,1.36305982073196E+016),
         (5.15082982520074E+016,5.15090182520083E+016),
         (9.24618993078481E+016,9.2467299307856E+016),
         (1.1967790034981E+017,1.19683300348819E+017)
        ]

FVBS4 = [(6.70902889090192E+015,6.71532889090465E+015),
         (1.36233982073171E+016,1.36305982073196E+016),
         (5.1681098252225E+016,5.16818182522259E+016),
         (9.27210993082233E+016,9.27282993082337E+016),
         (1.19429200349407E+017,1.19436400349418E+017)
        ]

FVBS5 = [(6.70902889090192E+015,6.71532889090465E+015),
         (1.36233982073171E+016,1.36305982073196E+016),
         (5.18538982524426E+016,5.18546182524435E+016),
         (9.27210993082233E+016,9.27282993082337E+016),
         (1.19429200349407E+017,1.19436400349418E+017)
        ]

FVBS6 = [(6.70902889090192E+015,6.71532889090465E+015),
         (1.36233982073171E+016,1.36305982073196E+016),
         (5.20266982526602E+016,5.20274182526611E+016),
         (9.27210993082233E+016,9.27282993082337E+016),
         (1.19429200349407E+017,1.19436400349418E+017)
        ]

FVBS7 = [(6.70902889090192E+015,6.71532889090465E+015),
         (1.36233982073171E+016,1.36305982073196E+016),
         (5.21994982528778E+016,5.22002182528788E+016),
         (9.27210993082233E+016,9.27282993082337E+016),
         (1.19429200349407E+017,1.19436400349418E+017)
        ]

FVBS = [FVBS1,FVBS2,FVBS3,FVBS4,FVBS5,FVBS6,FVBS7]


#### define functions

import glob
import re as regex


getobmt = lambda x: int(regex.search("(?<=OBMT_START_)[0-9]+",x).group(0))
getrow = lambda x: int(regex.search("(?<=CCD_ROW_)[0-9]+",x).group(0))
getfov = lambda x: int(regex.search("(?<=_SM)[0-9]+",x).group(0))


def rowgroup(paths, nrows):
    """
    Group file paths per FOV (fov goes from 1 up to nfov)
    """
    # set up the output list
    out = []
    for ii in range(nrows):
        out.append([])
    # fill it
    for p in paths:
        row = getrow(p)
        out[row-1].append(p)
    
    return out


def clean_paths(paths, periods):
    # do the cleaning - get rid of all files within the listed time periods
    to_remove = []  # files to remove later
    for period in periods:
        tstart = period[0]
        tstop = period[1]
        for p in paths:
            time = getobmt(p)
            if (time>=tstart) and (time<=tstop):
                to_remove.append(p)
    for p in to_remove:
        paths.remove(p)


#### run the code

# get all paths
allpaths = glob.glob(dataroot+"/*/*SM*CDP_NONE.fits")

#group them
grouped_paths = rowgroup(allpaths, 7)

# do the cleaning
for irow in range(7):
    clean_paths(grouped_paths[irow],OBMT_CI+FVBS[irow])

# output: just all the paths
# sum them up and sort them by obmt, why not

out = sorted(sum(grouped_paths,[]) ,key=getobmt)


with open(outfilename, 'w') as f:
    for p in out:
        f.write(p+'\n')
