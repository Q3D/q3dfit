#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jun  1 15:13:58 2021

@author: drupke
"""
import os.path
import numpy as np


def makanisdss():

    gal = 'makanisdss'
    ncols = 1
    nrows = 1
    fitrange = [3814, 9201]

#   These are unique to the user
    volume = '/Users/drupke/'
    infile = volume+'Box Sync/q3d/testing/makani/makanisdss.fits'
    outdir = volume+'specfits/q3dfit/testing/makanisdss/'
    stellartemplates = volume+'Box Sync/q3d/testing/makani/makani_stelmod.npy'
    logfile = outdir+'makanisdss_fitlog.txt'
#
# Required pars
#

    if not os.path.isfile(infile):
        print('Data cube not found.')

# Lines to fit.
    lines0 = ['[NeV]3345', '[NeV]3426', 'HeII4686']
    lines1 = ['[OII]3726', '[OII]3729',
              '[NeIII]3869', '[NeIII]3967',
              '[OIII]4959', '[OIII]5007',
              'Hbeta', 'Hgamma', 'Hdelta', 'Hepsilon']
    lines = [lines0, lines1]

# Max no. of components.
    maxncomp = 2

# Initialize line ties, n_comps, z_inits, and sig_inits.
    linetie = dict()
    ncomp = dict()
    zinit_gas = dict()
    siginit_gas = dict()
    for i in lines0:
        linetie[i] = '[NeV]3426'
        ncomp[i] = np.full((ncols, nrows), maxncomp)
        zinit_gas[i] = np.full((ncols, nrows,  maxncomp), 0.45666)
        siginit_gas[i] = np.full(maxncomp, 100)
    for i in lines1:
        linetie[i] = '[OII]3729'
        ncomp[i] = np.full((ncols, nrows), maxncomp)
        zinit_gas[i] = np.full((ncols, nrows, maxncomp), 0.45915)
        zinit_gas[i][:, :, 1] = 0.45737
        siginit_gas[i] = np.full(maxncomp, 100)

    zinit_stars = np.full((ncols, nrows), 0.459)

    linoth = np.full((2, 6), '', dtype=object)
    linoth[0, 1] = '[OII]3729'
    linoth[0, 2] = 'Hepsilon'
    linoth[:, 5] = ['Hbeta', '[OIII]4959']
    argspltlin1 = {'nx': 3,
                   'ny': 2,
                   'label': ['[NeV]3426', '[OII]3726', '[NeIII]3869',
                             'Hgamma', 'HeII4686', '[OIII]5007'],
                   'wave': [3426, 3728, 3869, 4340, 4686, 4925],
                   'off': [[-100, 100], [-100, 80], [-40, 150],
                           [-100, 100], [-100, 100], [-150, 150]],
                   'linoth': linoth}

    siglim_gas = np.ndarray(2)
    siglim_gas[:] = [5, 1500]

    init = { \
            # Required pars
            'fcninitpar': 'gmos',
            'fitran': fitrange,
            'fluxunits': 1e-17,  # erg/s/cm^2/arcsec^2
            'infile': infile,
            'label': gal,
            'lines': lines,
            'linetie': linetie,
            'maxncomp': maxncomp,
            'name': 'Makani',
            'ncomp': ncomp,
            'outdir': outdir,
            'zinit_stars': zinit_stars,
            'zinit_gas': zinit_gas,
            'zsys_gas': 0.459,
            # Optional pars
            'datext': -1,
            'varext': 1,
            'dqext': 2,
            'argscheckcomp': {'sigcut': 2},
            'argscontfit': {'siginit_stars': 50,
                            'uselog': 1,
                            'refit': 1},
            'argslinelist': {'vacuum': False},
            'argsinitpar': {'siglim': siglim_gas},
            'startempfile': stellartemplates,
            'argspltlin1': argspltlin1,
            'fcncheckcomp': 'checkcomp',
            'fcncontfit': 'ppxf',
            'maskwidths_def': 2000,
            'emlsigcut': 2,
            'logfile': logfile,
            'siglim_gas': siglim_gas,
            'siginit_gas': siginit_gas,
            'siginit_stars': 50,
            'nocvdf': 1
        }

    return(init)
