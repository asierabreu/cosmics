root = os.environ['HOME']
sys.path.append(root+'/spark/libs/astroscrappy/astroscrappy/')
sys.path.append(root+'/spark/libs/TrackObs/')
sys.path.append(root+'/spark/libs/PythonGbinReader/GbinReader/')

import gbin_reader
import TrackObs
import astroscrappy
