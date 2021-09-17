import sys
sys.path.append('/jwst0nb/lwz/')

from q3dfit.common import makeqsotemplate
import os


#outxdr = '/jwst0nb/lwz/jwst_q3d_data/irtest_qsotemplate'
#infits = '/jwst0nb/lwz/jwst_q3d_data/NRS00001-QG-F100LP-G140H_comb_1234_g140h-f100lp_s3d.fits'
outxdr = '/Users/caroline/Documents/ARI-Heidelberg/Q3D/DATA/NIRSPEC_NIRSIM/May2021/irtest_qsotemplate'
infits = '/Users/caroline/Documents/ARI-Heidelberg/Q3D/DATA/NIRSPEC_NIRSIM/May2021/NRS00001-QG-F100LP-G140H_comb_1234_g140h-f100lp_s3d.fits'



data = makeqsotemplate.makeqsotemplate(infits,outxdr,dataext=1,dqext=3,waveext=None, varext=2)
#os.system('rm nucleartemplate.npy')
