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
;      2020jun22, YI, added LMFIT functions that build the parameter variables and running the fits that call many_gauss() - ver2
;      2020jul06, YI, refined the LMFIT parameter set --> able to reproduce MPFIT results, but cannot get uncertainties - ver3
;      2020jul07, DSNR, fixed indexing bug in expr
;      2020jul08, YI, complete rewrite: implemented LMFIT Composite Model instead of LMFIT Minimizer for fits; can get fit stats. major changes to manygauss - ver4
;    
; :Copyright:
;    Copyright (C) 2013--2016 David S. N. Rupke
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
import matplotlib.pyplot as plt
from lmfit import Parameters, Minimizer,minimize,fit_report,report_fit,Model
import re

def fix_expressions(pnames,paramExp,indexs=[0],lineRat=[0]):
    # now go through each expressions....
    newExp = []
    for ex, exp in enumerate(paramExp):
        if ex in indexs:
            
            exp = re.sub('d','',exp)
            pind = np.array(re.findall(r'P\[(.*?)\]', exp)).astype(int)
            nexp = exp
            for epind in pind:
                sepind = 'P['+str(epind)+']'
                if epind in lineRat:
                    nexp = nexp.replace(sepind,str(lineRat[epind]))
                else:
                    nexp = nexp.replace(sepind,pnames[epind])
            newExp.append(nexp)
        else:
            newExp.append(exp)
    return newExp

def set_params(fit_params,NAME=None,VALUE=None,VARY=True,LIMITED=None,TIED='',LIMITS=None,STEP=None):
    LIMITED = np.array(LIMITED).astype(int)
    fit_params[NAME].set(value=VALUE,vary=VARY)
    if STEP != 0.:
        fit_params[NAME].brute_step=STEP
    if TIED != '' :
        fit_params[NAME].expr=TIED
    for li in [0,1]:
        if LIMITED[li] == 1:
            if li == 0:
                fit_params[NAME].min=LIMITS[0]
            elif li == 1:
                fit_params[NAME].max=LIMITS[1]
    
    return fit_params

def run_manygauss(wave,flux_nocnt,flux_err,parinfo,maxiter=1000.):
    # set-up the LMFIT parameters
    ppoff = parinfo[0]['value']
    nline = (len(parinfo)-ppoff)/3
    find = ppoff+np.arange(nline)*3
    wind = find + 1
    sind = find + 2
    
    manyGauss = []
    parTie = []
    iparTi = []
    parnames = []
    parval = []
    ilnRat = {}
    
    for ip, par in enumerate(parinfo):
        ptie = parinfo[ip]['tied']
        if isinstance(ptie,bytes):
            ptie = ptie.decode('utf-8')
        if ptie != '':
            ptie = ptie.replace(' ','')
            ptie = ptie.replace('+',' + ').replace('*',' * ').replace('-',' - ').replace('/',' / ')
            ptie = ptie.replace('e + ','E + ')
            ptie = ptie.replace('E + ','E+')
            iparTi.append(ip)
        parTie.append(ptie)
        ival = parinfo[ip]['value']
        if type(ival) is np.ndarray or isinstance(ival,list):
            ival = ival[0]
        parval.append(ival)
        if ip >= find[0]:
            line = parinfo[ip]['line']
            if isinstance(line,bytes):
                line = line.decode('utf-8')
            if line == '':
                line = 'blank_'+str(ip)
            else:
                line = line.replace(' ','_').replace('.','').replace('-','').replace('-','').replace('[','').replace(']','').replace('/','')
            if ip in find:
                fname = '%s_flx' % (line)
                mName = '%s_' % (line)
                parnames.append(fname)
                imodel = Model(manygauss,independent_vars=['x'],prefix=mName)
                if ip == find[0]:
                    manyGauss = imodel
                else:
                    manyGauss += imodel
            elif ip in wind:
                wname = '%s_cwv' % (line)
                parnames.append(wname)
            elif ip in sind:
                sname = '%s_sig' % (line)
                parnames.append(sname)
        else:
            parname = parinfo[ip]['parname']
            if isinstance(parname,bytes):
                parname = parname.decode('utf-8')
            if parname == '':
                parname = 'blank_'+str(ip)
            else:
                parname = parname.replace(' ','_').replace('.','').replace('-','').replace('-','').replace('[','').replace(']','').replace('/','_')
                if len(parname.split('line_ratio')) > 1:
                    ilnRat[ip]=parinfo[ip]['value'][0]
            ipar = parinfo[ip]
            parnames.append(parname)
    paramNames = manyGauss.param_names
    fit_params = manyGauss.make_params()
    newExp = fix_expressions(parnames,parTie,indexs=iparTi,lineRat=ilnRat)
    # ppoff = int(parinfo[0]['value'])
    gind = np.arange(nline)*4.+3
    ipn = 0
    ip = 0
    while ipn < len(paramNames):
        par = paramNames[ipn]
        if ipn in gind:
            ipar = parinfo[2]
            fit_params = set_params(fit_params,NAME=par,VALUE=ipar['value'],VARY=False,LIMITED=ipar['limited'],
                                        TIED='',LIMITS=ipar['limits'],STEP=ipar['step'])
        else:
            par = paramNames[ipn]
            ipar = parinfo[ip+ppoff]
            fixed = bool(int(ipar['fixed']))
            ifixed = not fixed
            itied = newExp[ip+ppoff]
            fit_params = set_params(fit_params,NAME=par,VALUE=ipar['value'],VARY=ifixed,LIMITED=ipar['limited'],
                                            TIED=itied,LIMITS=ipar['limits'],STEP=ipar['step'])
            ip+=1
        ipn+=1
    lmout = manyGauss.fit(flux_nocnt,fit_params,x=wave,method='least_squares',weights=1/flux_err,max_nfev=maxiter)
    specfit = manyGauss.eval(lmout.params,x=wave)
    print('*******************************************************************')
    print(lmout.fit_report())
    print('*******************************************************************')
    # get the param (array in the same format as the IDL mpfit version)
    pardict = []
    parOut = parval
    for ip,par in enumerate(lmout.values):
        if ip not in gind:
            pardict.append(lmout.values[par])
    for ip in range(ppoff,len(parval)):
        parOut[ip]=pardict[ip-ppoff]
    # now extract the perror (stddev for each parameter)
    perror=list(np.zeros(ppoff))
    for ip,par in enumerate(lmout.params.items()):
        if ip not in gind:
            perror.append(par[1].stderr)
    
    return lmout,parOut,specfit,perror


def manygauss(x,flx, cwv, sig,srsigslam):
    # param 0 flux
    # param 1 central wavelength
    # param 2 sigma
    c=299792.458
    sigs = np.sqrt(np.power((sig/c)*cwv,2.) + np.power(srsigslam,2.))
    gaussian = flx*np.exp(-np.power((x-cwv) / sigs,2.)/2.)
    return gaussian

# def manygauss(wave,param,specresarr=None):
#     c=299792.458

#     ppoff = param[0].astype(int)
#     nwave = len(wave)
#     nline = ((len(param)-ppoff)/3).astype(int)

#     find = ppoff+np.arange(nline)*3
#     wind = find + 1
#     sind = find + 2

#     dispersion = wave[1] - wave[0]
    
    
#     if specresarr:
#         # flipped the (row,column) indices of specresarr
#         # np.searchsorted() works a little differently from value_locate() in IDL
#         wsr = np.searchsorted(specresarr[0,:],param[wind])
#         # switched from [wsr,1] since Python reads [row,column] and IDL reads [column, row]
#         srsigslam = param[wind]/specresarr[1,wsr]/2.35 
#     else:
#         srsigslam = np.zeros(nline)+param[2]
  
#     # resolution in wavelength space [sigma] assumed to be in third element of PARAM

#     sigs = np.sqrt(np.power((param[sind]/c)*param[wind],2.) + np.power(srsigslam,2.))
#     maxsig = np.max(sigs)
    
#     nsubwave = np.round(10. * maxsig / dispersion)
#     halfnsubwave = np.round(nsubwave / 2)
#     nsubwave = (halfnsubwave*2+1).astype(int)
    
    
#     # indsubwaves = rebin(transpose(fix(indgen(nsubwave)-halfnsubwave)),nline,nsubwave)
#     # in the Python version, the transpose happens after the np.resize rebinning
#     indsubwaves = np.transpose(np.resize((np.arange(nsubwave)-halfnsubwave).astype(int),[nline,nsubwave])) 

#     fluxes = param[find]
#     refwaves = param[wind]
#     indrefwaves_real = (refwaves - wave[0]) / dispersion
#     indrefwaves = (indrefwaves_real).astype(int)
#     indrefwaves_frac = indrefwaves_real - indrefwaves.astype(float)
#     # flipped the (row,column) indices of indrefwaves_frac rebin (not sure if it works)
#     dwaves = (indsubwaves - np.resize(indrefwaves_frac,[nsubwave,nline]))*dispersion
#     # flipped the (row,column) indices of indrefwaves rebin
#     indsubwaves += np.resize(indrefwaves,[nsubwave,nline])
    
#     yvals = np.zeros(nwave)
#     for i in range (0,nline):
#         # flipped the (row,column) indices for indsubwaves
#         # Python cannot do a where(condition 1 and condition 2)  extraction like in IDL, 
#         # so I need to treat the limits separately and merge the indices for the 2 conditions
#         gind1 = np.where(indsubwaves[:,i] >= 0)[0]    
#         gind2 = np.where(indsubwaves[:,i] <= nwave-1)[0]
#         gind=list(set(gind1) & set(gind2))
#         count = len(gind)
#         # The "mask" parameter eliminates floating underflow by removing very large
#         # negative exponents. See http://www.idlcoyote.com/math_tips/underflow.html
#         # for more details.
#         if count > 0:
#             exparg = -np.power((dwaves[gind,i]/sigs[i]),2.)/2.    # flipped the (row,column) indices 
#             # edited the masking code
#             # mask = (abs(exparg) lt 80)
#             ms = np.where(abs(exparg) < 80)[0]
#             mask = np.zeros(len(exparg))
#             mask[ms] = np.ones(len(ms))
#             # flipped the (row,column) indices for indsubwaves
#             yvals[indsubwaves[gind,i]] = np.add(yvals[indsubwaves[gind,i]],np.transpose(fluxes[i]*mask*np.exp(exparg*mask)))
#             # yvals[indsubwaves[gind,i]] = np.transpose(fluxes[i]*mask*np.exp(exparg*mask))

#     return yvals

