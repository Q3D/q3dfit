import numpy as np
from q3dfit.common.q3df import q3df
from q3dfit.common.q3da import q3da

ippath = '/Users/drupke/specfits/q3dfit/testing/22128896/'  # Dave

initproc = np.load(ippath+'initproc.npy', allow_pickle=True)

#q3df(initproc[()], cols=1, rows=1, quiet=False)
q3da(initproc[()], cols=1, rows=1, quiet=False)
