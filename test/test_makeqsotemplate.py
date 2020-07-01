from q3dfit.common import makeqsotemplate
import os


outxdr = ''
infits = '../../pyfsfit/pg1411rb3.fits'

data = makeqsotemplate.makeqsotemplate(infits,outxdr,dataext=None,dqext=None,waveext=None)
os.system('rm nucleartemplate.npy')
