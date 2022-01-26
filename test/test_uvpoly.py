import numpy as np
from q3dfit.common.q3df import q3df
from q3dfit.common.q3da import q3da

# Dave
initproc = \
    np.load('/Users/drupke/Box Sync/f05189stis/q3dfit/nuc/initproc.npy',
            allow_pickle=True)
q3df(initproc[()], quiet=False)
q3da(initproc[()])
