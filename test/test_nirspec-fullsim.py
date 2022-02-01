import numpy as np
from q3dfit.common.q3df import q3df
from q3dfit.common.q3da import q3da

# Dave:
initproc = "/Users/drupke/specfits/q3dfit/testing/nirspec-fullsim/initproc.npy"

q3df(initproc, cols=22, rows=20, quiet=False)
q3da(initproc, cols=22, rows=20)
