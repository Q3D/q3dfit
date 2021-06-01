import sys
sys.path.append('/jwst0nb/lwz/')

from q3dfit.common import makeqsotemplate
import os


outxdr = '/jwst0nb/lwz/jwst_q3d_data/irtest_qsotemplate'
infits = '/jwst0nb/lwz/jwst_q3d_data/NRS00001-QG-F100LP-G140H_comb_1234_g140h-f100lp_s3d.fits'

data = makeqsotemplate.makeqsotemplate(infits,outxdr,dataext=None,dqext=None,waveext=None)
#os.system('rm nucleartemplate.npy')
