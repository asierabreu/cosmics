import os
import sys
sys.path.insert(0, os.path.abspath('../../lib/TrackObs/'))
from TrackObs import *

sys.path.insert(0, os.path.abspath('../bam/'))
from extraction_bam import *


# TODO:
# write a routine for signal extraction
# agree on a readout window - or maybe 2? (honestly, is this worth the measly gain in counts?)
# this also includes the AC-extent for fov 2
# rewrite the bam_cosmics routine to write a BAM-SIF header (which then implicitly assumes our readout window)