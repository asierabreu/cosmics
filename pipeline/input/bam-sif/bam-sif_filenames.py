# Functions to get all the BAM-SIF files outside the calibration activities, with absolute paths.

# input
import argparse
import os

# input parsing
parser = argparse.ArgumentParser(description='Retrieve filenames of BAM-SIF observations that can be used for cosmic extractions')
parser.add_argument('-d', '--dataRoot', help='Root path to search')
parser.add_argument('-o', '--outputFile', help='File to write to')
args = parser.parse_args()

dataroot = os.path.abspath(args.dataRoot)
outfilename = args.outputFile

#### define functions

import glob
import re as regex


getobmt = lambda x: int(regex.search("(?<=OBMT_START_)[0-9]+",x).group(0))
getrow = lambda x: int(regex.search("(?<=CCD_ROW_)[0-9]+",x).group(0))
getfov = lambda x: int(regex.search("(?<=_SM)[0-9]+",x).group(0))

#### run the code

# get all paths
allpaths = glob.glob(dataroot+"/*/*ROW_1_BAM*CLOCKING*")

# take only the paths that have the correct length in AC and AL
from astropy.io import fits

allens = []
aclens = []
times = []
bampaths = []
for ii in range(len(allpaths)):
    head = fits.getheader(allpaths[ii])
    times.append(getobmt(allpaths[ii]))
    allens.append(head['NAXIS2'])
    aclens.append(head['NAXIS1'])
    if (allens[-1] == 4599 and aclens[-1] == 162):
        bampaths.append(allpaths[ii])

out = sorted(bampaths, key=getobmt)

with open(outfilename, 'w') as f:
    for p in out:
        f.write(p+'\n')
