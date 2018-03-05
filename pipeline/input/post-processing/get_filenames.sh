#/bin/bash

# BAM-OBS
find /calteam/data/COSMICS/TrackObs/BAM/ -name *.fits* > fnames_bam-obs.txt
sort fnames_bam-obs.txt -o fnames_bam-obs.txt

# BAM-SIF
find /calteam/data/COSMICS/TrackObs/BAM-SIF/ -name *.fits* > fnames_bam-sif.txt
sort fnames_bam-sif.txt -o fnames_bam-sif.txt

# SM-SIF
find /calteam/data/COSMICS/TrackObs/SM/ -name *.fits* > fnames_sm-sif.txt
sort fnames_sm-sif.txt -o fnames_sm-sif.txt
