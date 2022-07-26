from os import chdir
from q3dfit.q3df import q3df
from q3dfit.q3da import q3da

chdir('../jnb/')
q3di = 'nirspec-etc/q3di.npy'

q3df(q3di, cols=13, rows=20, quiet=False)
q3da(q3di, cols=13, rows=20)
