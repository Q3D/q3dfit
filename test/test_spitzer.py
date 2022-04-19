from os import chdir
from q3dfit.q3df import q3df
from q3dfit.q3da import q3da

chdir('../jnb/')

q3di = '22128896/q3di.npy'

q3df(q3di, quiet=False)
q3da(q3di)
