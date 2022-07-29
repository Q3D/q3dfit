#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jul 26 12:05:26 2022

@author: drupke
"""

from os import chdir
from q3dfit.q3df import q3df
from q3dfit.q3da import q3da

chdir('../jnb/')
q3di = 'nirspec-j1652/q3di.npy'

# Single spaxel
q3df(q3di, cols=51, rows=43, quiet=False)
q3da(q3di, cols=51, rows=43, quiet=False)

# Most of cube
#q3df(q3di, cols=[20,75], rows=[15,70], ncores=10)
#q3da(q3di, cols=[20,75], rows=[15,70], noplots=True)


# Make maps
#
# import q3dfit.q3dpro as q3dpro
# qpro = q3dpro.Q3Dpro(q3di, PLATESCALE=0.05, NOCONT=True)

# do_kpc = False
# saveFile = False
# flx = [1e-4,1.]
# qsocenter = None
# pltarg = {'Ftot':flx,
#           'Fci':flx,
#           'Sig':[100,850],
#           'v50':[-600,600],
#           'fluxlog':True}

# qpro.make_linemap('[OIII]5007', XYSTYLE=do_kpc, xyCenter=qsocenter,
#                   LINEVAC=False, SAVEDATA=saveFile, VMINMAX=pltarg, PLTNUM=1,
#                   CMAP='inferno')
