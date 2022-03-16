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

# Dave:
q3di = '/Users/drupke/specfits/gmos/pg1411/rb3/initproc.npy'

q3df(q3di, cols=14, rows=11) #, quiet=False)
q3da(q3di, cols=14, rows=11) #, quiet=False)
