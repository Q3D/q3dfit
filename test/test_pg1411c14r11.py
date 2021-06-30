# Make quasar template
# from q3dfit.common.makeqsotemplate import makeqsotemplate
# volume = '/Users/drupke/Box Sync/q3d/pg1411/'
# outpy = volume + 'pg1411qsotemplate.npy'
# infits = volume + 'pg1411rb1.fits'
# makeqsotemplate(infits, outpy, dataext=None, dqext=None, waveext=None)

#<<<<<<< HEAD
#from q3dfit.common.q3df import q3df
#q3df('pg1411', cols=14, rows=11, quiet=False)

import os
cwd = os.getcwd()
print("Current working directory: {0}".format(cwd))

from q3dfit.common.q3da import q3da
q3da('pg1411', cols=14, rows=11, quiet=False)
#%% To test a simulated PG1411 spaxel with continuum fitting by questfit:
import os
cwd = os.getcwd()
print("Current working directory: {0}".format(cwd))

#from q3dfit.common.q3df import q3df
#q3df('pg1411_and_Spitzer', cols=0, rows=0, quiet=False)
#=======
import numpy as np
#import pdb
from q3dfit.common.q3df import q3df
from q3dfit.common.q3da import q3da

initproc = np.load('/Users/annamurphree/Docs/Rupke Research/q3d/pg1411/initproc.npy', 
                   allow_pickle=True)

#q3df(initproc[()], cols=14, rows=11, quiet=False)
q3da(initproc[()], cols=14, rows=11, quiet=False)\

#from q3dfit.common.q3da import q3da
#q3da('pg1411_and_Spitzer', cols=1, rows=1, quiet=False)
