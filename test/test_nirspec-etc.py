# # Make quasar template
# from q3dfit.common.makeqsotemplate import makeqsotemplate

# # Dave
# volume = '/Users/drupke/Box Sync/q3d/testing/pg1411/'
# outpy = volume + 'pg1411qsotemplate.npy'
# infits = volume + 'pg1411rb1.fits'

# makeqsotemplate(infits, outpy, wmapext=None, waveunit_in='Angstrom')

import numpy as np
from q3dfit.common.q3df import q3df
from q3dfit.common.q3da import q3da
#from q3dfit.common.makemaps import makemaps

# Dave:
#initproc = np.load(
#    '/Users/drupke/specfits/q3dfit/testing/nirspec-etc/initproc.npy',
#    allow_pickle=True)
initproc = "/Users/drupke/specfits/q3dfit/testing/nirspec-etc/initproc.npy"

#q3df(initproc, ncores=12)
#q3da(initproc, noplots=True)
#makemaps(initproc)
