root = os.environ['HOME']
sys.path.append(root+'/spark/libs/astroscrappy/astroscrappy/')
sys.path.append(root+'/spark/libs/TrackObs/')
sys.path.append(root+'/spark/libs/PythonGbinReader/')
sys.path.append(root+'/spark/libs/BAM_Processing')

import TrackObs
import astroscrappy
import gbin_reader
import process_bam
