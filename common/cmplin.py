#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jun  9 12:35:25 2020

@author: hadley
"""

import numpy as np
from astropy import modeling
import math
import pdb
from decimal import *


def cmplin(instr, line, comp, velsig = None):
    instr = np.load("dict.npy", allow_pickle='TRUE').item()

    c = 299792.458
        
    iline = [None]
    iline.pop(0)

    for i in range (0, len(instr['linelabel'])):
        if instr['linelabel'][i] == line:
            iline.append(i)
    ct = len(iline)
    ppoff = instr['param'][0]
    ncomp = instr['param'][1]
    specres = instr['param'][2]

    if 'parinfo' in instr:                
        indices = [None]
        indices.pop(0)
        for i in range (0, len(instr['parinfo']['line'])):
            if instr['parinfo']['line'][i] == line and \
                instr['parinfo']['comp'][i] == comp:
                indices.append(i)
    else:
        nline = len(instr['linelabel'])
        #not sure why it did where inelabel == line a Second time here? took it out
        indices = instr['param'][0] + (comp - 1) * nline * 3 + iline * 3
        indices = indices = indices[0] + np.arange(0, 3, dtype = float)

    if indices[0] != -1:
        gausspar = [None]
        for i in indices:
            gausspar.append(instr['param'][i])
        gausspar = gausspar[1:] #haha
        if velsig != None:
            gausspar[2] = np.sqrt((gausspar[2] * gausspar[1]/c)**2.0 \
                    + specres ** 2.0)
        else: gausspar[2] = np.sqrt(gausspar[2]**2.0 + specres**2.0)
        
        if gausspar[2] == 0.0: flux = 0.0
        else: #flux = modeling.models.Gaussian1D(gausspar[0], gausspar[1], gausspar[2]) 
            flux = gaussian(instr['wave'], gausspar)
        #gaussian(a, b) is in idlastro, should return a double array
    else: 
        flux = 0.0

    return flux

def gaussian(xi, parms, pderiv = None, DOUBLE = None):
    getcontext().prec = 40
    a = parms[0] #amp
    b = parms[1] #mean
    c = parms[2] #standard dev
    g = [0.0] #gaussian
#    g = g[1:] #lol
    d = [0.0] #derivative
        
    for x in xi:
        hl = Decimal(a) * Decimal(-0.5 * ((x - b) / c)**2).exp()
#        if hl > 10**-41: print('{:.5E}'.format(hl))
        g.append(hl)

    if pderiv != None:
        for i in range (0, len(xi)):
            np.append(d, (((-a * (x - b)) / c**2) \
                          / (math.exp((c**2 * (x - b)**2)/(2 * c**2)))))
        return d
    return g
