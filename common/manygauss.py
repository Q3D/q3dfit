# -*- coding: utf-8 -*-
"""
; docformat = 'rst'
;
;+
; This function is input to MPFIT, which uses it to compute the
; emission line spectrum of multiple Gaussian emission lines
; simultaneously. This routine assumes constant dispersion (in A/pix),
; and uses this fact to optimize the computation by working with
; smaller sub-arrays. Use MANYGAUSS_SLOW (which works with large
; arrays) for data without constant dispersion.
;
; :Categories:
;    IFSFIT
;
; :Returns:
;    An N-element array containing an emission-line spectrum.
;
; :Params:
;    wave: in, required, type=dblarr(N)
;      Wavelengths
;    param: in, required, type=dblarr
;      Best-fit parameter array output by MPFIT.
;
; :Keywords:
;
; :Author:
;    David S. N. Rupke::
;      Rhodes College
;      Department of Physics
;      2000 N. Parkway
;      Memphis, TN 38104
;      drupke@gmail.com
;
; :History:
;    ChangeHistory::
;      2009, DSNR, copied from manygauss_slow.pro and rewritten
;      2013sep, DSNR, switch sigma from wavelength to velocity space
;      2013nov13, DSNR, documented, renamed, added license and copyright
;      2014apr10, DSNR, fixed cases of floating underflow
;      2016sep26, DSNR, switched to deconvolving resolution in situ by adding
;                       resolution in quadrature in wavelength space to each
;                       velocity component
;      2020jun01, YI, translated to Python 3 - ver2
;      2020jun22, YI, added LMFIT functions that build the parameter variables
;                     and running the fits that call many_gauss() - ver2
;      2020jul06, YI, refined the LMFIT parameter set --> able to reproduce
;                     MPFIT results, but cannot get uncertainties - ver3
;      2020jul07, DSNR, fixed indexing bug in expr
;      2020jul08, YI, complete rewrite: implemented LMFIT Composite Model
;                     instead of LMFIT Minimizer for fits; can get fit
;                     stats. major changes to manygauss - ver4
;      2021jan25, DSNR, rewrote some logic for clarity; heavy commenting;
;                       removed fixing of line ratios; tested unsuccessfully
;                       one option for varying line ratio
;
; :Copyright:
;    Copyright (C) 2013--2021 David S. N. Rupke
;
;    This program is free software: you can redistribute it and/or
;    modify it under the terms of the GNU General Public License as
;    published by the Free Software Foundation, either version 3 of
;    the License or any later version.
;
;    This program is distributed in the hope that it will be useful,
;    but WITHOUT ANY WARRANTY; without even the implied warranty of
;    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU
;    General Public License for more details.
;
;    You should have received a copy of the GNU General Public License
;    along with this program.  If not, see
;    http://www.gnu.org/licenses/.
;
;-
"""
import numpy as np
import pdb
import re
from lmfit import Model


# This loops through expressions that tie parameters together and replaces
# instances of "P[index]" with the parameter name equivalent in LMFIT
# parnames = parameter names for the LMFIT model, except for srsigslam's
# partie = tied expressions
# iparinfo = parinfo indices of parnames
def fix_expressions(parnames, partie, iparinfo):
    newExp = []
    # Loop through all fitted parameters (except srsigslam)
    for i, par in enumerate(parnames):
        exp = partie[i]
        notie = False
        # Re-write expression only if it's not an empty string
        if exp != '':
            nexp = exp
            # Search for all instances of "P[index]" in the expression
            pind = np.array(re.findall(r'P\[(.*?)\]', exp)).astype(int)
            # for each instance, replace by the correct LMFIT parameter name
            # if it exists
            for epind in pind:
                sepind = 'P['+str(epind)+']'
                tmpind = (iparinfo == epind).nonzero()[0]
                if len(tmpind) > 0:
                    nexp = nexp.replace(sepind, parnames[tmpind[0]])
                else:
                    notie = True
            if not notie:
                newExp.append(nexp)
            else:
                newExp.append('')
        else:
            newExp.append('')
    return newExp


def set_params(fit_params, NAME=None, VALUE=None, VARY=True, LIMITED=None,
               TIED='', LIMITS=None, STEP=None):
    LIMITED = np.array(LIMITED).astype(int)
    fit_params[NAME].set(value=VALUE, vary=VARY)
    if STEP != 0.:
        fit_params[NAME].brute_step = STEP
    if TIED != '':
        fit_params[NAME].expr = TIED
    for li in [0, 1]:
        if LIMITED[li] == 1:
            if li == 0:
                fit_params[NAME].min = LIMITS[0]
            elif li == 1:
                fit_params[NAME].max = LIMITS[1]
    return fit_params


def run_manygauss(wave, flux_nocnt, flux_weight, parinfo, maxiter=1000, \
                  quiet=True):
    maxncomp = parinfo[1]['value']  ### CB: To be double-checked

    ppoff = parinfo[0]['value']
    nline = (len(parinfo)-ppoff)/3  # number of emission lines
    find = ppoff+np.arange(nline)*3  # indices to fluxes in parinfo
    wind = find + 1  # indices to wavelengths in parinfo
    sind = find + 2  # indices to sigmas in parinfo

    # the total LMFIT Model
    # size = # model instances
    manyGauss = []
    # parameter names used in the LMFIT model
    # size = # fitted pars - # lines (no srsigslam)
    parnames = []
    # parinfo index for each parname
    # size = # fitted pars - # lines (no srsigslam)
    iparinfo = []
    # expressions for tying parameters together
    # size = # fitted pars - # lines (no srsigslam)
    partie = []
    # initial values set in parinfo; size = # pars in parinfo
    parvals = []
    # nLineRatios = 0

    # Start translating parinfo to parameter language of LMFIT
    for ip, par in enumerate(parinfo):
        fitpar = False  # is this a parameter to fit?
        # Initial parameter values
        ival = parinfo[ip]['value']
        if type(ival) is np.ndarray or isinstance(ival, list):
            ival = ival[0]
        parvals.append(ival)
        # Process lines
        if ip >= find[0]:
            fitpar = True
            line = parinfo[ip]['line']
            line = line.replace(' ', '_').replace('.', '').\
                replace('-', '').replace('-', '').replace('[', '').\
                replace(']', '').replace('/', '')
            # Create model instance of linel add to total model;
            # and populate parameter list
            if ip in find:
                parname = '%s_flx' % (line)
                mName = '%s_' % (line)
                imodel = Model(manygauss, independent_vars=['x'], prefix=mName)
                if isinstance(manyGauss, Model):
                    try:
                        manyGauss += imodel
                    except:
                        manyGauss += Model(manygauss, independent_vars=['x'], prefix=mName+'B')     # if a second component is used, lmfit complains if that has the same linename
                else:
                    manyGauss = imodel
            elif ip in wind:
                parname = '%s_cwv' % (line)
            elif ip in sind:
                parname = '%s_sig' % (line)
        # # Process line ratios
        # else:
        #     parname = parinfo[ip]['parname']
        #     parname = parname.replace(' ', '_').replace('.', '').\
        #         replace('-', '').replace('[', '').\
        #         replace(']', '').replace('/', '_')
        #     if len(parname.split('line_ratio')) > 1:
        #         parname = parname.replace('_line_ratio', '_')
        #         nLineRatios += 1
        #         imodel = Model(lineratio, independent_vars=['x'],
        #                         prefix=parname)
        #         parname += 'lrat'
        #         if isinstance(manyGauss, Model):
        #             manyGauss += imodel
        #         else:
        #             manyGauss = imodel
        #         fitpar = True
        if fitpar:
            parnames.append(parname)
            iparinfo.append(ip)
            ptie = parinfo[ip]['tied']
            if ptie != '':
                ptie = ptie.replace('e+', 'E+')
            partie.append(ptie)

    # size = # fitted parameters
    # paramNames = manyGauss.param_names
    fit_params = manyGauss.make_params()
    iparinfo = np.array(iparinfo)  # for logic in fix_expressions
    # replaces "P[index]" instances with LMFIT parameter names
    newExp = fix_expressions(parnames, partie, iparinfo)

    # Set parameter properties properly in LMFIT parameter objects
    ip = 0  # index counting parname values
    for i, parname in enumerate(fit_params.keys()):
        # # Line ratio parameters
        # if len(parname.split('lrat')) > 1:
        #     pidat = parinfo[iparinfo[ip]]
        #     fixed = bool(int(pidat['fixed']))
        #     ifixed = not fixed
        #     fit_params = \
        #         set_params(fit_params, NAME=parname, VALUE=pidat['value'],
        #                    VARY=ifixed, LIMITED=pidat['limited'],
        #                    TIED='', LIMITS=pidat['limits'],
        #                    STEP=pidat['step'])
        #     ip += 1
        # elif len(parname.split('srsigslam')) > 1:
        if len(parname.split('srsigslam')) > 1:
            pidat = parinfo[2]
            fit_params = \
                set_params(fit_params, NAME=parname, VALUE=pidat['value'],
                           VARY=False, LIMITED=pidat['limited'],
                           TIED='', LIMITS=pidat['limits'], STEP=pidat['step'])
        else:
            pidat = parinfo[iparinfo[ip]]
            fixed = bool(int(pidat['fixed']))
            ifixed = not fixed
            itied = newExp[ip]
            fit_params = \
                set_params(fit_params, NAME=parname, VALUE=pidat['value'],
                           VARY=ifixed, LIMITED=pidat['limited'],
                           TIED=itied, LIMITS=pidat['limits'],
                           STEP=pidat['step'])
            ip += 1

    # Actual fit
    lmout = manyGauss.fit(flux_nocnt, fit_params, x=wave,
                          method='least_squares', weights=flux_weight,
                          max_nfev=maxiter, nan_policy='omit')
    specfit = manyGauss.eval(lmout.params, x=wave)
    if not quiet:
        print(lmout.fit_report())

    # get the param array back into the same format as the IDL mpfit version
    # Output parameter values and errors; size = len(parinfo)
    parout = parvals
    perror = np.zeros(len(parinfo))
    # includes line parameters flx; cwv; and sig
    # loop size is # fitted pars - # lines (no srsigslam)
    for ip, parname in enumerate(parnames):
        parout[iparinfo[ip]] = lmout.values[parname]
        perror[iparinfo[ip]] = lmout.params[parname].stderr

    return lmout, parout, specfit, perror


def manygauss(x, flx, cwv, sig, srsigslam):
    # param 0 flux
    # param 1 central wavelength
    # param 2 sigma
    c = 299792.458
    sigs = np.sqrt(np.power((sig/c)*cwv, 2.) + np.power(srsigslam, 2.))
    gaussian = flx*np.exp(-np.power((x-cwv) / sigs, 2.)/2.)
    return gaussian


def lineratio(x, lrat):
    return lrat
