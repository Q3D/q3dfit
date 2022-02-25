import numpy as np
from q3dfit.common.q3df import q3df
from q3dfit.common.q3da import q3da

# Dave
ippath = '/Users/drupke/specfits/q3dfit/testing/22128896/'

# Caroline
#ippath = '../test/test_questfit/'
#initproc_npy_exists = False
#if not initproc_npy_exists:
#    from q3dfit.init.f22128896mir import f22128896mir
#    initproc = f22128896mir()
#    np.save(ippath+'initproc.npy', initproc)

initproc = ippath+'initproc.npy'

q3df(initproc, cols=1, rows=1, quiet=False)
q3da(initproc, cols=1, rows=1, quiet=False)
