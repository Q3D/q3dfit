from os import chdir
from q3dfit.q3df import q3dfit

chdir('../jnb/')

q3di = 'Spitzer-example/q3di.npy'

q3dfit(q3di, quiet=False)

cols = 1
rows = 1
from q3dfit.q3dout import load_q3dout
q3do = load_q3dout(q3di, cols, rows)

argsplotline = dict()
argsplotline['nx'] = 3
argsplotline['ny'] = 2
argsplotline['line'] = ['[ArII]6.99', '[ArIII]8.99', '[NeII]12.81',
                        '[NeIII]15.56', 'H2_00_S1', '[SIII]18.71']
argsplotline['size'] = [3., 3., 3., 3., 3., 3.]
q3do.plot_line(q3di,plotargs=argsplotline)
