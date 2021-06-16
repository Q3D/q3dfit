import numpy as np
from q3dfit.common.q3df import q3df
from q3dfit.common.q3da import q3da

initproc = \
    np.load('/Users/drupke/specfits/q3dfit/testing/makanisdss/initproc.npy',
            allow_pickle=True)
q3df(initproc[()], cols=1, oned=True)
q3da(initproc[()], cols=1, oned=True)
