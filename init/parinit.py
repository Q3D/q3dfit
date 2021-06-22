#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri May 28 16:40:03 2021

Initialize parameters for fitting.

@author: drupke
"""

from lmfit import Model
import numpy as np
import pdb


def parinit(linelist, linelistz, linetie, initflux, initsig, maxncomp, ncomp,
            lratfix=None, siglim=None, sigfix=None, blrcomp=None,
            blrlines=None, specres=None):

    dblt_pairs = {'[NII]6548': '[NII]6583',
                  '[NeIII]3967': '[NeIII]3689',
                  '[NeV]2422': '[NeV]3426',
                  '[OI]6364': '[OI]6300',
                  '[OIII]4959': '[OIII]4959'}

    if not specres:
        specres = 0.
    # A reasonable lower limit of 5d for physicality ...
    # Assume line is resolved.
    if not siglim.all():
        siglim = [5., 2000.]

    # converts the astropy.Table structure of linelist into a Python
    # dictionary that is compatible with the code downstream
    lines_arr = {name: linelist['lines'][idx] for idx, name
                 in enumerate(linelist['name'])}

    # the total LMFIT Model
    # size = # model instances
    totmod = []

    # cycle through lines
    for line in lines_arr:
        # cycle through velocity components
        for i in range(0, ncomp[line]):
            # LMFIT parameters can only consist of letters,  numbers, or _
            line = line.replace('[', 'lb').replace(']', 'rb').replace('.', 'pt')
            mName = '{0}_{1}_'.format(line,i)
            imodel = Model(manygauss, prefix=mName)
            if isinstance(totmod, Model):
                totmod += imodel
            else:
                totmod = imodel

    # Create parameter dictionary
    fit_params = totmod.make_params()

    # Cycle through parameters
    for i, parname in enumerate(fit_params.keys()):
        # split parameter name string into line, component #, and parameter
        psplit = parname.split('_')
        lmline = psplit[0]  # string for line label
        line = lmline.replace('lb', '[').replace('rb', ']').replace('pt','.')
        comp = int(psplit[1])  # string for line component
        gpar = psplit[2]  # parameter name in manygauss
        # Process input values
        vary = 'True'
        if gpar == 'flx':
            value = initflux[line][comp]
            limited = np.array([1, 0], dtype='uint8')
            limits = np.array([0., 0.], dtype='float64')
            # Check if it's a doublet; this will break if weaker line
            # is in list, but stronger line is not
            if lmline in dblt_pairs.keys():
                tied = '{0}_{1}_flx / 3.'.format(dblt_pairs[line],comp)
                tied = tied.replace('[', 'lb').replace(']', 'rb').replace('.', 'pt')
            else:
                tied = ''
        elif gpar == 'cwv':
            value = linelistz[line][comp]
            limited = np.array([1, 1], dtype='uint8')
            limits = np.array([linelistz[line][comp]*0.997,
                               linelistz[line][comp]*1.003], dtype='float64')
            # Check if line is tied to something else
            if linetie[line] != line:
                linetie_tmp = linetie[line].replace('[', 'lb').\
                    replace(']', 'rb').replace('.', 'pt')
                tied = '{0:0.6e} / {1:0.6e} * {2}_{3}_cwv'.\
                    format(lines_arr[line],lines_arr[linetie[line]],
                           linetie_tmp,comp)
            else:
                tied = ''
        elif gpar == 'sig':
            value = initsig[line][comp]
            limited = np.array([1, 1], dtype='uint8')
            limits = np.array(siglim, dtype='float64')
            if linetie[line] != line:
                linetie_tmp = linetie[line].replace('[', 'lb').\
                    replace(']', 'rb').replace('.', 'pt')
                tied = '{0}_{1}_sig'.format(linetie_tmp,comp)
            else:
                tied = ''
        else:
            value = specres
            limited = None
            limits = None
            vary = False
            tied = ''

        fit_params = \
            set_params(fit_params, parname, VALUE=value,
                       VARY=vary, LIMITED=limited, TIED=tied,
                       LIMITS=limits)

    return totmod, fit_params


def set_params(fit_params, NAME, VALUE=None, VARY=True, LIMITED=None,
               TIED=None, LIMITS=None):
    if VALUE is not None:
        fit_params[NAME].set(value=VALUE)
    fit_params[NAME].set(vary=VARY)
    if TIED is not None:
        fit_params[NAME].expr = TIED
    if LIMITED is not None and LIMITS is not None:
        if LIMITED[0] == 1:
            fit_params[NAME].min = LIMITS[0]
        if LIMITED[1] == 1:
            fit_params[NAME].max = LIMITS[1]
    return fit_params


def manygauss(x, flx, cwv, sig, srsigslam):
    # param 0 flux
    # param 1 central wavelength
    # param 2 sigma
    c = 299792.458
    sigs = np.sqrt(np.power((sig/c)*cwv, 2.) + np.power(srsigslam, 2.))
    gaussian = flx*np.exp(-np.power((x-cwv) / sigs, 2.)/2.)
    return gaussian
