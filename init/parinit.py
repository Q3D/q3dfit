#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri May 28 16:40:03 2021

Initialize parameters for fitting.

@author: drupke
"""

from lmfit import Model
import numpy as np


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
    manyGauss = []
    fit_params = manyGauss.make_params()

    # cycle through lines
    for line in lines_arr:
        # cycle through velocity components
        for i in range(0, ncomp):
            mName = '%s_'+str(i)+'_' % (line)
            imodel = Model(manyGauss, independent_vars=['x'], prefix=mName)
            if isinstance(manyGauss, Model):
                manyGauss += imodel
            else:
                manyGauss = imodel

    # Create parameter dictionary
    fit_params = manyGauss.make_params()

    # Cycle through parameters
    for i, parname in enumerate(fit_params.keys()):
        # split parameter name string into line, component #, and parameter
        psplit = parname.split(parname)
        line = psplit[0]  # string for line label
        comp = int(psplit[1])  # string for line component
        gpar = psplit[2]  # parameter name in manyGauss
        # Process input values
        vary = 'True'
        if gpar == 'flx':
            value = initflux[line][comp]
            limited = np.array([1, 0], dtype='uint8')
            limits = np.array([0., 0.], dtype='float64')
            # Check if it's a doublet; this will break if weaker line
            # is in list, but stronger line is not
            if line in dblt_pairs.keys():
                tied = dblt_pairs[line]+'_'+str(comp)+'_flx / 3.'
            else:
                tied = ''
        elif gpar == 'cwv':
            value = linelistz[line][comp]
            limited = np.array([1, 1], dtype='uint8')
            limits = np.array([linelistz[line][comp]*0.997,
                               linelistz[line][i]*1.003], dtype='float64')
            # Check if line is tied to something else
            if linetie[line] != line:
                tied = '{0:0.6e}{1:1}{2:0.6e}{3:1}{4:1}'.\
                    format(lines_arr[line], ' / ',
                           lines_arr[linetie[line]], ' * ',
                           linetie[line]+'_'+str(comp)+'_'+'cwv')
            else:
                tied = ''
        elif gpar == 'sig':
            value = initsig[line][comp]
            limited = np.array([1, 1], dtype='uint8')
            limits = np.array(siglim, dtype='float64')
            if linetie[line] != line:
                tied = linetie[line]+'_'+str(comp)+'_'+'sig'
            else:
                tied = ''
        else:
            value = specres
            limited = np.array([1, 1], dtype='uint8')
            limits = np.array([specres, specres], dtype='float64')
            vary = False
            tied = ''
        fit_params = \
            set_params(fit_params, parname, VALUE=value,
                       VARY=vary, LIMITED=limited, TIED=tied,
                       LIMITS=limits)


def set_params(fit_params, NAME, VALUE=None, VARY=True, LIMITED=None,
               TIED=None, LIMITS=None):
    if VALUE is not None:
        fit_params[NAME].set(value=VALUE)
    fit_params[NAME].set(vary=VARY)
    if TIED is not None:
        fit_params[NAME].expr = TIED
    if LIMITED is not None and LIMITS is not None:
        for li in [0, 1]:
            if LIMITED[li] == 1:
                fit_params[NAME].min = LIMITS[li]
    return fit_params


def manygauss(x, flx, cwv, sig, srsigslam):
    # param 0 flux
    # param 1 central wavelength
    # param 2 sigma
    c = 299792.458
    sigs = np.sqrt(np.power((sig/c)*cwv, 2.) + np.power(srsigslam, 2.))
    gaussian = flx*np.exp(-np.power((x-cwv) / sigs, 2.)/2.)
    return gaussian
