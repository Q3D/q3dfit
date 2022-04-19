from os import chdir
chdir('../jnb/')
from q3dfit.q3df import q3df
from q3dfit.q3da import q3da

q3di = 'pg1411/q3di.npy'

q3df(q3di, cols=14, rows=11, quiet=False)
q3da(q3di, cols=14, rows=11) #, quiet=False)
