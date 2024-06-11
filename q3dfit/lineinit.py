#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from astropy.constants import c
from astropy.table import QTable, Table
from lmfit import Model
from q3dfit.q3dutil import lmlabel
from q3dfit.exceptions import InitializationError
import numpy as np
import q3dfit.data
import os


def lineinit(linelist, linelistz, linetie, initflux, initsig, maxncomp, ncomp,
             specConv, lineratio=None, siglim=None, blrcomp=None, 
             linevary=None, blrlines=None, blrsiglim=None, waves=None):
    '''

    Initialize parameters for emission-line fitting.

    Parameters
    ----------
    linelist : astropy Table
        Table of emission lines.
    linelistz : dict
        Dictionary of line redshifts.
    linetie : dict
        Dictionary of line ties.
    initflux : dict
        Dictionary of initial fluxes.
    initsig : dict
        Dictionary of initial sigmas.
    maxncomp : int
        Maximum number of components.
    ncomp : dict
        Dictionary of number of components.
    specConv : object
        Spectral resolution object.
    lineratio : astropy Table, optional
        Table of line ratio constraints, beyond those in the doublets table.
    siglim : list, optional
        Sigma limits. The default is to set a lower limit of 5 and an upper
        limit of 2000 km/s.
    blrcomp : list, optional
        List of components for broad-line region scattered component fit in fitqsohost. 
        The default is None.
    linevary : dict, optional
        Dictionary of line parameter vary flags (fix/free). The default is to set
        all parameters to free.
    blrlines : list, optional
        List of broad-line region scattered lines in fitqsohost. The default is None.
    blrsiglim : list, optional
        Sigma limits for broad-line region scattered component fit in fitqsohost.
        The default is None.
    waves : array, optional
        Wavelength array for determining if a line is in the fit range. The default
        is None.
    
    Returns
    -------
    totmod : lmfit Model
        Total model object for fitting.
    fit_params : lmfit Parameters
        Fit parameter object
    siglim : list
        Sigma limits.

    '''
    # Get fixed-ratio doublet pairs for tying intensities
    data_path = os.path.abspath(q3dfit.data.__file__)[:-11]
    doublets64 = Table.read(data_path+'linelists/doublets.tbl', format='ipac')
    doublets = \
        Table(doublets64,
              dtype=['str', 'str', 'int', 'float64', 'float64', 'float64'])

    dblt_pairs = dict()
    for idx, name in enumerate(doublets['line1']):
        if doublets['fixed_ratio'][idx] == 1:
            dblt_pairs[doublets['line2'][idx]] = doublets['line1'][idx]

    # A reasonable lower limit of 5d for physicality
    if siglim is None:
        siglim = np.array([5., 2000.], dtype='float64')
    else:
        siglim = np.array(siglim, dtype='float64')

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
        # If line out of range or something else
        value = np.float64(np.finfo(float).eps)
        limited = None
        limits = None
        vary = False
        tied = ''
        # Process input values
        if gpar == 'flx' and inrange:
            value = initflux[line.label][comp]
            limited = np.array([1, 0], dtype='uint8')
            limits = np.array([np.finfo(float).eps, np.finfo(float).eps],
                              dtype='float64')
            # Check if it's a doublet; this will break if weaker line
            # is in list, but stronger line is not
            if line.label in dblt_pairs.keys():
                dblt_lmline = lmlabel(dblt_pairs[line.label])
                idx_line = np.where(doublets['line1']==dblt_lmline.lmlabel.replace('lb', '[').replace('rb', ']'))[0]
                ratio = doublets['ratio'][idx_line].value[0]
                tied = f'{dblt_lmline.lmlabel}_{comp}_flx/(1.*{ratio})'
            else:
                tied = ''
            if linevary is None:
                vary = True
            else:
                try:
                    vary = linevary[line.label][gpar][comp]
                except:
                    print('lineinit: dict vary missing information')

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
            # This is pretty odd; it's a very special case
            # elif force_cwv_lines is not None:
            #     if line.label in force_cwv_lines and comp > 0:
            #         linetie_tmp = lmlabel(linetie[line.label])
            #         tied = '{}_0_cwv'.\
            #                 format(linetie_tmp.lmlabel)
            else:
                tied = ''
            if linevary is None:
                vary = True
            else:
                try:
                    vary = linevary[line.label][gpar][comp]
                except:
                    print('lineinit: dict vary missing information')

        elif gpar == 'sig':
            value = initsig[line.label][comp]
            limited = np.array([1, 1], dtype='uint8')

            limits = np.array(siglim, dtype='float64')
            if blrlines is not None and blrcomp is not None and blrsiglim is not None:
                if line.label in blrlines and comp in blrcomp:
                    limits = np.array(blrsiglim, dtype='float64')

            if linetie[line.label] != line.label:
                linetie_tmp = lmlabel(linetie[line.label])
                tied = f'{linetie_tmp.lmlabel}_{comp}_sig'
            else:
                tied = ''
            if linevary is None:
                vary = True
            else:
                try:
                    vary = linevary[line.label][gpar][comp]
                except:
                    print('lineinit: dict vary missing information')


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
                    fit_params.add(lmrat, value=initval.astype('float64'))
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
                        fit_params[lmrat].min = lower.astype('float64')
                    # logic to apply doublet lower limits if in doublets table
                    elif line1 in doublets['line1']:
                        iline1 = np.where(doublets['line1'] == line1)
                        if doublets['line2'][iline1] == line2:
                            lower = doublets['lower'][iline1][0]
                        fit_params[lmrat].min = lower.astype('float64')
                    # doublet can be specified in init file in either order
                    # relative to doublets table ...
                    elif line1 in doublets['line2']:
                        iline1 = np.where(doublets['line2'] == line1)
                        if doublets['line1'][iline1] == line2:
                            upper = 1. / doublets['lower'][iline1][0]
                        fit_params[lmrat].max = upper.astype('float64')
                    # apply upper limit?
                    if 'upper' in lineratio.colnames:
                        upper = lineratio['upper'][ilinrat]
                        fit_params[lmrat].max = upper.astype('float64')
                    elif line1 in doublets['line1']:
                        iline1 = np.where(doublets['line1'] == line1)
                        if doublets['line2'][iline1] == line2:
                            upper = doublets['upper'][iline1][0]
                        fit_params[lmrat].max = upper.astype('float64')
                    elif line1 in doublets['line2']:
                        iline1 = np.where(doublets['line2'] == line1)
                        if doublets['line1'][iline1] == line2:
                            lower = 1. / doublets['upper'][iline1][0]
                        fit_params[lmrat].min = lower.astype('float64')

    # pass siglim_gas back because the default is set here, and it's needed
    # downstream

    return totmod, fit_params, siglim


def set_params(fit_params, NAME, VALUE=None, VARY=True, LIMITED=None,
               TIED=None, LIMITS=None):
    '''

    Set parameters for the lmfit model using the lmfit Parameters object.

    Parameters
    ----------
    fit_params : lmfit Parameters
        Fit parameter object.
    NAME : str
        Parameter name.
    VALUE : float, optional
        Initial value. The default is None.
    VARY : bool, optional
        Vary flag. The default is True.
    LIMITED : list, optional
        Whether or not to limit the parameter. The default is to set no limits.
    TIED : str, optional
        Expression for tying the parameter to another. The default is None.
    LIMITS : list, optional
        Limits for the parameter. The default is None.
    
    Returns
    -------
    fit_params : lmfit Parameters
        Modified fit parameter object.
    
    '''

    # we can force the input to float64, but lmfit has values as float64 and
    # doesn't seem like we can change it. These astypes assume that the
    # VALUE is a numpy object
    if VALUE is not None:
        fit_params[NAME].set(value=VALUE.astype('float64'))
    fit_params[NAME].set(vary=VARY)
    if TIED is not None:
        fit_params[NAME].expr = TIED
    if LIMITED is not None and LIMITS is not None:
        if LIMITED[0] == 1:
            fit_params[NAME].min = LIMITS[0].astype('float64')
        if LIMITED[1] == 1:
            fit_params[NAME].max = LIMITS[1].astype('float64')
    return fit_params


def manygauss(x, flx, cwv, sig, SPECRES=None):
    '''
    Generate a Gaussian model for a given set of parameters.

    Parameters
    ----------
    x : array
        Wavelength array.
    flx : array
        Flux array.
    cwv : float
        Central wavelength.
    sig : float
        Sigma.
    SPECRES : object, optional
        Spectral resolution object, for convolving the Gaussian with the spectral 
        resolution. The default is None.
    
    Returns
    -------
    gaussian : array
        Gaussian model.
    
    '''
    
    sigs = sig / c.to('km/s').value * cwv
    gaussian = flx * np.exp(-np.power((x - cwv) / sigs, 2.)/2.)
    if SPECRES is not None:
        # resample spectrum on smaller grid before convolution if
        # sigma less than 1 pixel. Note that this doesn't assume
        # constant dispersion, which I *think* is okay for the 
        # ppxf_util.varsmooth() algorithm. But I'm not sure.
        oversample = 1
        if np.mean(sigs) <= np.mean(x[1:-1]-x[0:-2])*1.:
            oversample = 10
        datconv = SPECRES.spect_convolver(x, gaussian, wavecen=cwv,
            oversample=oversample)
        #maskval = np.float64(1e-4*max(datconv))
        #maskind = np.asarray(datconv < maskval).nonzero()[0]
        #datconv[maskind] = np.float64(0.)
        return datconv
    else:
        #maskval = np.float64(1e-4*max(gaussian))
        #maskind = np.asarray(gaussian < maskval).nonzero()[0]
        #gaussian[maskind] = np.float64(0.)
        return gaussian
