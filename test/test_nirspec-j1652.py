#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jul 26 12:05:26 2022

@author: drupke
"""

import pdb

from os import chdir
from q3dfit.q3df import q3df
from q3dfit.q3da import q3da
from q3dfit.q3dpro import Q3Dpro, OneLineData
from q3dfit.q3dpro import OneLineData


chdir('../jnb/')
q3di = 'nirspec-j1652-conv/q3di.npy'

# Single spaxel
#q3df(q3di, cols=50, rows=40, quiet=False)
#q3da(q3di, cols=40, rows=50, quiet=False)

# Most of cube
#q3df(q3di, cols=[20,75], rows=[15,70], ncores=10)
#q3da(q3di, cols=[20,75], rows=[15,70], noplots=True)

# Make maps
#
qpro = Q3Dpro(q3di, PLATESCALE=0.05, NOCONT=True)

o3data = OneLineData(qpro.linedat, '[OIII]5007')

o3data.calc_cvdf(2.9489, [-5e3, 5e3], vstep=5)

import matplotlib.pyplot as plt
fig, ax = plt.subplots()
ax.step(o3data.cvdfvel, o3data.cvdfflux[49, 39, :])
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
