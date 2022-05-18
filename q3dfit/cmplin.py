#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jun  9 12:35:25 2020

"""

import numpy as np
import pdb
from astropy.constants import c
from q3dfit.lmlabel import lmlabel


def cmplin(instr, line, comp, velsig=False):
    '''
    Function takes four parameters and returns specified flux

    Parameters:
    instr : dict
    line : str
    comp : int
    velsig : bool

    Returns
    -------
    flux : float
    '''


    lmline = lmlabel(line)
    mName = '{0}_{1}_'.format(lmline.lmlabel, comp)
    gausspar = np.zeros(3)
    gausspar[0] = instr['param'][mName+'flx']
    gausspar[1] = instr['param'][mName+'cwv']
    gausspar[2] = instr['param'][mName+'sig']
    if velsig:
        gausspar[2] = gausspar[2] * gausspar[1]/c.to('km/s').value

    flux = gaussian(instr['wave'], gausspar)

    return flux


def gaussian(xi, parms):

    a = parms[0]  # amp
    b = parms[1]  # mean
    c = parms[2]  # standard dev

    # Anything higher-precision than this (e.g., float64) slows things down
    # a bunch. longdouble completely chokes on lack of memory.
    arg = np.array(-0.5 * ((xi - b)/c)**2, dtype=np.float32)
    g = a * np.exp(arg)

    return g
