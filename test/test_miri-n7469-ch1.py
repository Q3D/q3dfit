from os import chdir
from q3dfit.q3df import q3df
from q3dfit.q3da import q3da

chdir('../jnb/')

q3di = 'miri-n7469-ch1/q3di.npy'

cols = 16
rows = 31

q3df(q3di, cols=cols, rows=rows, quiet=False)
q3da(q3di, cols=cols, rows=rows)
