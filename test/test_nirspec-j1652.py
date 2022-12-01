#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jul 26 12:05:26 2022

@author: drupke
"""
from os import chdir
chdir('../jnb/')

q3di = 'nirspec-j1652/q3di.npy'

# Single spaxel
#cols=45
#rows=29

# Run the fit
#from q3dfit.q3df import q3dfit
#q3dfit(q3di, cols=cols, rows=rows, quiet=False)

#
# Plot an example
#
# from q3dfit.q3dout import load_q3dout
# q3do = load_q3dout(q3di, cols, rows)

# argsplotline = dict()
# argsplotline['nx'] = 2
# argsplotline['ny'] = 1
# argsplotline['line'] = ['Hbeta', '[OIII]5007']
# argsplotline['size'] = [0.05, 0.07]
# q3do.plot_line(plotargs=argsplotline)

# argscontplot = dict()
# argscontplot['xstyle'] = 'lin'
# argscontplot['ystyle'] = 'lin'
# argscontplot['fluxunit_out'] = 'flambda'
# argscontplot['mode'] = 'dark'
# q3do.sepcontpars(q3di)
# q3do.plot_cont(q3di, plotargs=argscontplot)

# Most of cube
#from q3dfit.q3df import q3dfit
#q3dfit(q3di, cols=[20,75], rows=[15,70], ncores=10)

# Collate data
#cols = [20, 75]
#rows = [15, 70]
#from q3dfit.q3dcollect import q3dcollect
#q3dcollect(q3di, cols=cols, rows=rows)

# Make maps
#
#from q3dfit.q3dpro import Q3Dpro, OneLineData
#from q3dfit.q3dpro import OneLineData
#qpro = Q3Dpro(q3di, PLATESCALE=0.05, NOCONT=True)

#o3data = OneLineData(qpro.linedat, '[OIII]5007')

# [OIII] cvdf
#o3data.calc_cvdf(2.9489, [-5e3, 5e3], vstep=5)

# test plot of CVDF calc
#import matplotlib.pyplot as plt
#fig, ax = plt.subplots()
#ax.step(o3data.cvdf_vel, o3data.vdf[49, 39, :])

# compute v50 for [OIII] test
#v50 = o3data.calc_cvdf_vel(68.)
#import matplotlib.pyplot as plt
#from matplotlib import cm
#norm = cm.colors.Normalize(vmax=2e3, vmin=-2e3)
#fig, ax = plt.subplots()
#ax.imshow(v50, norm=norm, cmap='RdYlBu')
#plt.show()

# map of v50 for [OIII] test
# o3data.make_cvdf_map(50., velran=[-1e3, 1e3], markcenter=[0., 0.],
#                      center=[47., 47.])
#o3data.make_cvdf_map(50., velran=[-1e3, 1e3], markcenter=[47., 47.],
#                     outfile=True)


# do_kpc = False
# saveFile = False
# flx = [1e-5,2e-2]
# qsocenter = None
# pltarg = {'Ftot':flx,
#           'Fci':flx,
#           'Sig':[100,850],
#           'v50':[-600,600],
#           'fluxlog':True}

# qpro.make_linemap('[OIII]5007', XYSTYLE=do_kpc, xyCenter=qsocenter,
#                   LINEVAC=False, SAVEDATA=saveFile, VMINMAX=pltarg, PLTNUM=1,
#                   CMAP='inferno')
