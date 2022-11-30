q3di = 'pg1411/q3di.npy'

from os import chdir
chdir('../jnb/')

# Fitting
from q3dfit.q3df import q3dfit
q3dfit(q3di, cols=14, rows=11, quiet=False)
#q3dfit(q3di, cols=[1,17], rows=[1,26], ncores=15)

# Plotting
# from q3dfit.q3dout import load_q3dout
# q3do = load_q3dout(q3di, 6, 7, cubedim=3)

# argsplotline = dict()
# argsplotline['nx'] = 3
# argsplotline['ny'] = 2
# argsplotline['line'] = ['Hbeta', '[OIII]4959', '[NI]5200', '[OI]6300', 'Halpha', '[SII]6716']
# argsplotline['size'] = [0.01, 0.0125, 0.01, 0.0125, 0.0100, 0.0100]

# q3do.plot_line(plotargs=argsplotline)

# argscontplot = dict()
# argscontplot['xstyle'] = 'lin'
# argscontplot['ystyle'] = 'lin'
# argscontplot['waveunit_out'] = 'Angstrom'
# argscontplot['fluxunit_out'] = 'flambda'
# argscontplot['mode'] = 'dark'

# q3do.sepcontpars(q3di)
# q3do.plot_cont(q3di, plotargs=argscontplot)

# Collating
#from q3dfit.q3dcollect import q3dcollect
#q3dcollect(q3di) #, cols=[5,10], rows=[5,10])
