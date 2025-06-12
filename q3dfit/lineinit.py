#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from typing import Optional
import numpy as np
import os

from astropy.constants import c
from astropy.table import QTable, Table
from lmfit import Model, Parameters

import q3dfit.data
from . import q3dutil
from q3dfit.exceptions import InitializationError
from q3dfit.spectConvol import spectConvol

def lineinit(linelist: Table,
             linelistz: dict,
             linetie: dict,
             linevary: dict[str, np.ndarray],
             initflux: dict[str, np.ndarray],
             initsig: dict[str, np.ndarray],
             siglim: dict[str, np.ndarray],
             ncomp: dict[str, np.ndarray],
             specConv: Optional[spectConvol]=None,
             lineratio: Optional[Table | QTable]=None,
             waves: Optional[np.ndarray]=None) -> tuple[Model, Parameters]:
    '''

    Initialize parameters for emission-line fitting.

    Parameters
    ----------
    linelist
        Emission lines to fit.
    linelistz
        Initial guess for line redshifts.
    linetie
        Line to which to tie each line.
    linevary
        Line parameter vary flags (fix/free).
    initflux
       Initial guess for line fluxes.
    initsig
        Initial guess for line sigmas.
    siglim
        Lower and upper limits for line sigmas.
    ncomp
        Number of components for each line.
    specConv
        Optional. Spectral resolution object. If set to None, no convolution
        will be performed. The default is None.
    lineratio
        Optional. Table of line ratio constraints, beyond those in the doublets table,
        https://github.com/Q3D/q3dfit/blob/main/q3dfit/data/linelists/doublets.tbl.
        The default is None. The table must contain the columns `line1`, `line2`,
        `comp`, and optionally `value`, `fixed`, `lower`, and `upper`.\n
        - `line1`/`line2`: the names of the two lines to which the ratio `line1`/`line2` applies. These are the labels found in the linelist Table.
        - `comp`: an array of velocity components (zero-indexed) on which to apply the constraints, one array for each pair of lines.
        - `value`: the initial value of `line1`/`line2`. Presently, if `value` is specified for one pair of lines, it must be specified for all. Otherwise, the initial value is determined from the data.
        - `fixed`: The ratio can be `fixed` to the initial value. Presently, if `fixed` is defined, it must be set to `True` or `False` for all pairs of line.
        - `lower`/`upper`: If the ratio is not `fixed`, `lower` and `upper` limits can also be specified. (If they are not, and the line pair is a doublet in the doublets.tbl file, then the lower and upper limits are set using the data in that file.) Presently, if `lower` or `upper` is defined here for one set of lines, it must be defined here for every pair of lines.
    waves
        Optional. Wavelength array for determining if a line is in the fit range. 
        The default is None.
    
    Returns
    -------
    tuple[Model, Parameters]
        A tuple containing the LMFIT Model and the fit parameters.
    '''
    # Get fixed-ratio doublet pairs for tying intensities
    data_path = os.path.abspath(q3dfit.data.__file__)[:-11]
    doublets64 = Table.read(os.path.join(data_path, 'linelists', 'doublets.tbl'), format='ipac')
    doublets = \
        Table(doublets64,
              dtype=['str', 'str', 'int', 'float64', 'float64', 'float64'])

    dblt_pairs = dict()
    for idx, name in enumerate(doublets['line1']):
        if doublets['fixed_ratio'][idx] == 1:
            dblt_pairs[doublets['line2'][idx]] = doublets['line1'][idx]

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
            lmline = q3dutil.lmlabel(line)
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
        line = q3dutil.lmlabel(lmline, reverse=True)
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
                dblt_lmline = q3dutil.lmlabel(dblt_pairs[line.label])
                idx_line = np.where(doublets['line1']==dblt_lmline.lmlabel.replace('lb', '[').replace('rb', ']'))[0]
                ratio = doublets['ratio'][idx_line].value[0]
                tied = f'{dblt_lmline.lmlabel}_{comp}_flx/(1.*{ratio})'
            else:
                tied = ''
            try:
                vary = linevary[line.label][gpar][comp]
            except:
                raise InitializationError('linevary not properly defined')

        elif gpar == 'cwv':
            value = linelistz[line.label][comp]
            limited = np.array([1, 1], dtype='uint8')
            limits = np.array([value*0.997, value*1.003], dtype='float32')
            # Check if line is tied to something else
            if linetie[line.label] != line.label:
                linetie_tmp = q3dutil.lmlabel(linetie[line.label])
                tied = '{0:0.6e} / {1:0.6e} * {2}_{3}_cwv'.\
                    format(lines_arr[line.label],
                           lines_arr[linetie[line.label]],
                           linetie_tmp.lmlabel, comp)
            # This is pretty odd; it's a very special case
            # elif force_cwv_lines is not None:
            #     if line.label in force_cwv_lines and comp > 0:
            #         linetie_tmp = q3dutil.lmlabel(linetie[line.label])
            #         tied = '{}_0_cwv'.\
            #                 format(linetie_tmp.lmlabel)
            else:
                tied = ''
            try:
                vary = linevary[line.label][gpar][comp]
            except:
                raise InitializationError('linevary not properly defined')

        elif gpar == 'sig':
            value = initsig[line.label][comp]
            limited = np.array([1, 1], dtype='uint8')
            limits = siglim[line.label][comp]
            if linetie[line.label] != line.label:
                linetie_tmp = q3dutil.lmlabel(linetie[line.label])
                tied = f'{linetie_tmp.lmlabel}_{comp}_sig'
            else:
                tied = ''
            try:
                vary = linevary[line.label][gpar][comp]
            except:
                raise InitializationError('linevary not properly defined')


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
            lmline1 = q3dutil.lmlabel(line1)
            lmline2 = q3dutil.lmlabel(line2)
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

    return totmod, fit_params


def set_params(fit_params: Parameters,
               NAME: str,
               VALUE: Optional[float]=None,
               VARY: bool=True,
               LIMITED: Optional[np.ndarray]=None,
               TIED: Optional[str]=None,
               LIMITS: Optional[np.ndarray]=None)->Parameters:
    '''

    Set parameters for the lmfit model using the lmfit Parameters object.

    Parameters
    ----------
    fit_params
        Fit parameter object.
    NAME
        Parameter name.
    VALUE
        Optional. Initial value. The default is None, in which case no value is set.
    VARY
        Vary flag. The default is True, in which case the parameter is free to vary.
    LIMITED
        Whether or not to limit the parameter. The default, None, is to set no limits.
    TIED
        Expression for tying the parameter to another. The default is None.
    LIMITS
        Limits for the parameter. The default is None, in which case no limits are set.
    
    Returns
    -------
    Parameters
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


def manygauss(x: np.ndarray,
              flx: np.ndarray,
              cwv: float,
              sig: float,
              SPECRES: Optional[spectConvol]=None) -> np.ndarray:
    '''
    Generate a Gaussian model for a given set of parameters.

    Parameters
    ----------
    x
        Wavelength array.
    flx
        Flux array.
    cwv
        Central wavelength.
    sig
        Sigma.
    SPECRES
        Spectral resolution object, for convolving the Gaussian with the spectral 
        resolution. The default is None.
    
    Returns
    -------
    numpy.ndarray
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
