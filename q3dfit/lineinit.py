#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from astropy.constants import c
from astropy.table import QTable, Table
from lmfit import Model
from q3dfit.q3dutil import lmlabel
from q3dfit.exceptions import InitializationError
import copy
import numpy as np
import q3dfit.data
import os

def lineinit(linelist, linelistz, linetie, initflux, initsig, maxncomp, ncomp, specConv,
             lineratio=None, siglim=None, sigfix=None, blrcomp=None,
             blrlines=None, specres=None, waves=None):
    '''
    Initialize parameters for emission-line fitting.
    '''
    # Get fixed-ratio doublet pairs for tying intensities
    data_path = os.path.abspath(q3dfit.data.__file__)[:-11]
    doublets64 = Table.read(data_path+'linelists/doublets.tbl', format='ipac')
    doublets = \
        Table(doublets64,
              dtype=['str', 'str', 'int', 'float32', 'float32', 'float32'])

    dblt_pairs = dict()
    for idx, name in enumerate(doublets['line1']):
        if doublets['fixed_ratio'][idx] == 1:
            dblt_pairs[doublets['line2'][idx]] = doublets['line1'][idx]

    # pre-SPECRES object
    #if not specres:
    #    specres = np.float32(0.)
    #else:
    #    specres = np.float32(specres)
    # A reasonable lower limit of 5d for physicality
    if siglim is None:
        siglim = np.array([5., 2000.], dtype='float32')
    else:
        siglim = np.array(siglim, dtype='float32')

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
            lmline = lmlabel(line)
            mName = f'{lmline.lmlabel}_{i}_'
            imodel = Model(manygauss, prefix=mName, SPECRES=specConv)
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
        lmline = ''
        # this bit is for the case where the line label has underscores in it
        for i in range(0, len(psplit)-2):
            lmline += psplit[i]  # string for line label
            if i != len(psplit)-3:
                lmline += '_'
        line = lmlabel(lmline, reverse=True)
        inrange = True
        if waves is not None:
            if linelistz[line.label][0] > max(waves) or \
                    linelistz[line.label][0] < min(waves):
                inrange = False
        # ... the final two underscores separate the line label from the comp
        # and gaussian parname
        comp = int(psplit[len(psplit)-2])  # string for line component
        gpar = psplit[len(psplit)-1]  # parameter name in manygauss
        # Process input values
        vary = 'True'
        if gpar == 'flx' and inrange:
            value = initflux[line.label][comp]
            limited = np.array([1, 0], dtype='uint8')
            limits = np.array([0., 0.], dtype='float32')
            # Check if it's a doublet; this will break if weaker line
            # is in list, but stronger line is not
            if line.label in dblt_pairs.keys():
                dblt_lmline = lmlabel(dblt_pairs[line.label])
                tied = f'{dblt_lmline.lmlabel}_{comp}_flx/3.'
            else:
                tied = ''
        elif gpar == 'cwv':
            value = linelistz[line.label][comp]
            limited = np.array([1, 1], dtype='uint8')
            limits = np.array([linelistz[line.label][comp]*0.997,
                               linelistz[line.label][comp]*1.003],
                              dtype='float32')
            # Check if line is tied to something else
            if linetie[line.label] != line.label:
                linetie_tmp = lmlabel(linetie[line.label])
                tied = '{0:0.6e} / {1:0.6e} * {2}_{3}_cwv'.\
                    format(lines_arr[line.label],
                           lines_arr[linetie[line.label]],
                           linetie_tmp.lmlabel, comp)
            else:
                tied = ''
        elif gpar == 'sig':
            value = initsig[line.label][comp]
            limited = np.array([1, 1], dtype='uint8')
            limits = np.array(siglim, dtype='float32')
            if linetie[line.label] != line.label:
                linetie_tmp = lmlabel(linetie[line.label])
                tied = f'{linetie_tmp.lmlabel}_{comp}_sig'
            else:
                tied = ''
#            if not inrange:
#                vary = False
        else:
            value = np.float32(0.)
            limited = None
            limits = None
            vary = False
            tied = ''

        fit_params = \
            set_params(fit_params, parname, VALUE=value,
                       VARY=vary, LIMITED=limited, TIED=tied,
                       LIMITS=limits)

    # logic for bounding or fixing line ratios
    if lineratio is not None:
        if not isinstance(lineratio, QTable) and \
            not isinstance(lineratio, Table):
            raise InitializationError('The lineratio key must be' +
                                      ' an astropy Table or QTable')
        elif 'line1' not in lineratio.colnames or \
            'line2' not in lineratio.colnames or \
            'comp' not in lineratio.colnames:
            raise InitializationError('The lineratio table must contain' +
                                      ' the line1, line2, and comp columns')
        for ilinrat in range(0, len(lineratio)):
            line1 = lineratio['line1'][ilinrat]
            line2 = lineratio['line2'][ilinrat]
            comps = lineratio['comp'][ilinrat]
            lmline1 = lmlabel(line1)
            lmline2 = lmlabel(line2)
            for comp in comps:
                if f'{lmline1.lmlabel}_{comp}_flx' in fit_params.keys() and \
                    f'{lmline2.lmlabel}_{comp}_flx' in fit_params.keys():
                    # set initial value
                    if 'value' in lineratio.colnames:
                        initval = lineratio['value'][ilinrat]
                    else:
                        initval = \
                            np.divide(
                                fit_params[f'{lmline1.lmlabel}_{comp}_flx'],
                                fit_params[f'{lmline2.lmlabel}_{comp}_flx'])
                    lmrat = f'{lmline1.lmlabel}_div_{lmline2.lmlabel}_{comp}'
                    fit_params.add(lmrat, value=initval.astype('float32'))
                    # tie second line to first line divided by the ratio
                    fit_params[f'{lmline2.lmlabel}_{comp}_flx'].expr = \
                        f'{lmline1.lmlabel}_{comp}_flx'+'/'+lmrat
                    # fixed or free
                    if 'fixed' in lineratio.colnames:
                        if lineratio['fixed'][ilinrat]:
                            fit_params[lmrat].vary = False
                    # apply lower limit?
                    if 'lower' in lineratio.colnames:
                        lower = lineratio['lower'][ilinrat]
                        fit_params[lmrat].min = lower.astype('float32')
                    # logic to apply doublet lower limits if in doublets table
                    elif line1 in doublets['line1']:
                        iline1 = np.where(doublets['line1'] == line1)
                        if doublets['line2'][iline1] == line2:
                            lower = doublets['lower'][iline1][0]
                        fit_params[lmrat].min = lower.astype('float32')
                    # doublet can be specified in init file in either order
                    # relative to doublets table ...
                    elif line1 in doublets['line2']:
                        iline1 = np.where(doublets['line2'] == line1)
                        if doublets['line1'][iline1] == line2:
                            upper = 1. / doublets['lower'][iline1][0]
                        fit_params[lmrat].max = upper.astype('float32')
                    # apply upper limit?
                    if 'upper' in lineratio.colnames:
                        upper = lineratio['upper'][ilinrat]
                        fit_params[lmrat].max = upper.astype('float32')
                    elif line1 in doublets['line1']:
                        iline1 = np.where(doublets['line1'] == line1)
                        if doublets['line2'][iline1] == line2:
                            upper = doublets['upper'][iline1][0]
                        fit_params[lmrat].max = upper.astype('float32')
                    elif line1 in doublets['line2']:
                        iline1 = np.where(doublets['line2'] == line1)
                        if doublets['line1'][iline1] == line2:
                            lower = 1. / doublets['upper'][iline1][0]
                        fit_params[lmrat].min = lower.astype('float32')

    # pass siglim_gas back because the default is set here, and it's needed
    # downstream
    return totmod, fit_params, siglim


def set_params(fit_params, NAME, VALUE=None, VARY=True, LIMITED=None,
               TIED=None, LIMITS=None):
    # we can force the input to float32, but lmfit has values as float64 and
    # doesn't seem like we can change it. These astypes assume that the
    # VALUE is a numpy object
    if VALUE is not None:
        fit_params[NAME].set(value=VALUE.astype('float32'))
    fit_params[NAME].set(vary=VARY)
    if TIED is not None:
        fit_params[NAME].expr = TIED
    if LIMITED is not None and LIMITS is not None:
        if LIMITED[0] == 1:
            fit_params[NAME].min = LIMITS[0].astype('float32')
        if LIMITED[1] == 1:
            fit_params[NAME].max = LIMITS[1].astype('float32')
    return fit_params


def manygauss(x, flx, cwv, sig, SPECRES=None):
    # param 0 flux
    # param 1 central wavelength
    # param 2 sigma
    # sigs = np.sqrt(np.power((sig/c)*cwv, 2.) + np.power(srsigslam, 2.))
    sigs = sig / c.to('km/s').value * cwv
    gaussian = flx * np.exp(-np.power((x - cwv) / sigs, 2.)/2.)
    if SPECRES is not None:
        datconv = SPECRES.spect_convolver(x, gaussian, cwv)
        return datconv
    #maskval = np.float64(1e-4*max(gaussian))
    #maskind = np.asarray(gaussian < maskval).nonzero()[0]
    #gaussian[maskind] = np.float64(0.)
    else:
        return gaussian
