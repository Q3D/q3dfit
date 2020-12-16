#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jun  9 12:35:25 2020

Returns an array of Decimals (fluxes representing emission line profile)
"""

import numpy as np
# from astropy import modeling
from astropy.table import Table
# import math
# import pdb
from decimal import *


def cmplin(instr, line, comp, velsig=None):

    # From sepfitpars.py
    parinfo_new = {}
    for k, v in [(key, d[key]) for d in instr['parinfo'] for key in d]:
        if k not in parinfo_new:
            parinfo_new[k] = [v]
        else:
            parinfo_new[k].append(v)
    parinfo_new = Table(parinfo_new)

    c = 299792.458

    iline = [None]
    iline.pop(0)

    for i in range(0, len(instr['linelabel'])):
        if instr['linelabel'][i] == line:
            iline.append(i)
    # ct = len(iline)
    # ppoff = instr['param'][0]
    # ncomp = instr['param'][1]
    specres = instr['param'][2]

    indices = [None]
    indices.pop(0)
    for i in range(0, len(parinfo_new['line'])):
        if parinfo_new['line'][i] == line and \
                parinfo_new['comp'][i] == comp:
            indices.append(i)

    if indices[0] != -1:
        gausspar = []
        for i in indices:
            gausspar.append(instr['param'][i])
        if velsig is not None:
            gausspar[2] = np.sqrt((gausspar[2] * gausspar[1]/c)**2.0
                                  + specres ** 2.0)
        else:
            gausspar[2] = np.sqrt(gausspar[2]**2.0 + specres**2.0)

        if gausspar[2] == 0.0:
            flux = 0.0
        else:
            flux = gaussian(instr['wave'], gausspar)

    else:
        flux = 0.0
#    print(type(flux[0]))
    return flux


def gaussian(xi, parms):
    getcontext().prec = 40
    a = parms[0]  # amp
    b = parms[1]  # mean
    c = parms[2]  # standard dev
    g = [0.0]  # gaussian

    for x in xi:
        hl = Decimal(a) * Decimal(-0.5 * ((x - b) / c)**2).exp()
        g.append(hl)
    g = g[1:]

    return g
