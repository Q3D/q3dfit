from q3dfit.common.q3df import q3df
import numpy as np

initproc_npy = np.load('../test/test_questfit/initproc.npy', allow_pickle=True)
q3df(initproc_npy, cols=1, rows=1, quiet=False)

from q3dfit.common.q3da import q3da
# q3da(initproc_npy, cols=1, rows=1, quiet=False)
