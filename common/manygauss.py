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
;      2020jun01, YI, translated to Python 3
;      2020jun22, YI, added LMFIT functions that build the parameter variables and running the fits that call many_gauss()
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
from lmfit import Parameters, Minimizer,minimize,fit_report
import re

def lm_resid(param,wave,flux_nocnt,flux_err):
    """Calculate total residual for fits of Gaussians to several data sets."""
    parvalue = np.zeros(len(param))
    for ip, par in enumerate(param):
        pValue = param[par].value
        parvalue[ip] = pValue
    manyGauss = manygauss(wave,parvalue)
    resid = (flux_nocnt - manyGauss )/flux_err
    # plt.clf()
    # plt.plot(wave,flux_nocnt,'-',color='grey',linewidth=0.5)
    # plt.plot(wave,manyGauss,'b-',linewidth=1)
    # plt.xlim(4000,8000)
    # print('lm_resid() finish')
    return resid

def fix_expressions(params,pnames,paramExp,indexs=[0]):
    new_exprs = []
    print('------------------')
    # now go through each expressions....
    for ex, exp in enumerate(paramExp):
        exp = re.sub('d','',exp)
        pind = np.array(re.findall(r'P\[(.*?)\]', exp)).astype(int)
        nexp = exp
        for epind in pind:
            sepind = 'P['+str(epind)+']'
            nexp = nexp.replace(sepind,pnames[epind])
        new_exprs.append(nexp)
        ni = indexs[ex]
        params[pnames[ni]].expr = nexp
    return params
    

def run_manygauss(wave,flux_nocnt,flux_err,parinfo,maxiter=1000.):
    # set-up the LMFIT parameters
    ppoff = parinfo[0]['value']
    nline = (len(parinfo)-ppoff)/3
    find = ppoff+np.arange(nline)*3
    wind = find + 1
    sind = find + 2

    fit_params = Parameters()
    pnames = []
    parTie = []
    
    for ip, par in enumerate(parinfo):
        ptie = parinfo[ip]['tied']
        if isinstance(ptie,bytes):
            ptie = ptie.decode('utf-8')
            if ptie != '':
                ptie = ptie.replace(' ','')
                ptie = ptie.replace('+',' + ').replace('*',' * ').replace('-',' - ').replace('/',' / ')
                ptie = ptie.replace('E + ','E+')
                parTie.append(ptie)
        fixed =bool(int(parinfo[ip]['fixed']))
        # print(ip,fixed)
        if ip >= find[0]:
            line = parinfo[ip]['line']
            if isinstance(line,bytes):
                line = line.decode('utf-8')
            if line == '':
                line = 'blank_'+str(ip)
            else:
                line = line.replace(' ','_').replace('.','').replace('-','').replace('-','').replace('[','').replace(']','').replace('/','')
            # print(ip,line)
            if ip in find:
                fpar = parinfo[ip]
                fname = 'flx_%s' % (line)
                pnames.append(fname)
                if fpar['limits'][0] != fpar['limits'][1]:
                    
                    fit_params.add(fname, value=fpar['value'],vary = fixed,
                                    min=fpar['limits'][0], max=fpar['limits'][1],
                                    brute_step= fpar['step'])
                else:
                    fit_params.add(fname, value=fpar['value'],vary = fixed,
                                    brute_step= fpar['step'])
            elif ip in wind:
                wpar = parinfo[ip]
                wname = 'wav_%s' % (line)
                pnames.append(wname)
                if wpar['limits'][0] != wpar['limits'][1]:
                    fit_params.add(wname, value=wpar['value'],vary = fixed,
                                min=wpar['limits'][0], max=wpar['limits'][1],
                                brute_step= wpar['step'])
                else:
                    fit_params.add(wname, value=wpar['value'], vary = fixed,
                                brute_step= wpar['step'])
            elif ip in sind:
                spar = parinfo[ip]
                sname = 'sig_%s' % (line)
                pnames.append(sname)
                if spar['limits'][0] != spar['limits'][1]:
                    fit_params.add(sname, value=spar['value'],vary = fixed,
                                min=spar['limits'][0], max=spar['limits'][1],
                                brute_step=spar['step'])
                else:
                    fit_params.add(sname, value=spar['value'], vary = fixed,
                                brute_step=spar['step'])
        else:
            parname = parinfo[ip]['parname']
            if isinstance(parname,bytes):
                parname = parname.decode('utf-8')
            if parname == '':
                parname = 'blank_'+str(ip)
            else:
                parname = parname.replace(' ','_').replace('.','').replace('-','').replace('-','').replace('[','').replace(']','').replace('/','_')
            pnames.append(parname)
            if parinfo[ip]['limits'][0] == parinfo[ip]['limits'][1]:
                fit_params.add(parname, value=parinfo[ip]['value'],vary = fixed,
                                brute_step= parinfo[ip]['step'])
            else:
                fit_params.add(parname, value=parinfo[ip]['value'],vary = fixed,
                                min=parinfo[ip]['limits'][0], max=parinfo[ip]['limits'][1],
                                brute_step= parinfo[ip]['step'])
    # now check for the EXPRESSIONS
    pnames = np.array(pnames)
    expr = np.where(parinfo[:]['tied'] != '')[0]
    print(expr)
    if len(parTie) != 0:
        fit_params = fix_expressions(fit_params,pnames,parTie,indexs=expr)
        
    lmout = minimize(lm_resid, fit_params, args=(wave,flux_nocnt,flux_err),
                     method='leastsq',max_nfev=maxiter)
    
    parout = lmout.params
    parval = np.zeros(len(parinfo))
    for ip, par in enumerate(parout):
        parval[ip] = parout[par].value
    specfit = manygauss(wave,parval)
    print(fit_report(lmout.params))
    return lmout,parval,specfit


def manygauss(wave,param,specresarr=None):
    c=299792.458

    ppoff = param[0].astype(int)
    nwave = len(wave)
    nline = ((len(param)-ppoff)/3).astype(int)

    find = ppoff+np.arange(nline)*3
    wind = find + 1
    sind = find + 2

    dispersion = wave[1] - wave[0]
    
    
    if specresarr:
        # flipped the (row,column) indices of specresarr
        # np.searchsorted() works a little differently from value_locate() in IDL
        wsr = np.searchsorted(specresarr[0,:],param[wind])
        # switched from [wsr,1] since Python reads [row,column] and IDL reads [column, row]
        srsigslam = param[wind]/specresarr[1,wsr]/2.35 
    else:
        srsigslam = np.zeros(nline)+param[2]
  
    # resolution in wavelength space [sigma] assumed to be in third element of PARAM

    sigs = np.sqrt(np.power((param[sind]/c)*param[wind],2.) + np.power(srsigslam,2.))
    maxsig = np.max(sigs)
    
    nsubwave = np.round(10. * maxsig / dispersion)
    halfnsubwave = np.round(nsubwave / 2)
    nsubwave = (halfnsubwave*2+1).astype(int)
    
    
    # indsubwaves = rebin(transpose(fix(indgen(nsubwave)-halfnsubwave)),nline,nsubwave)
    # in the Python version, the transpose happens after the np.resize rebinning
    indsubwaves = np.transpose(np.resize((np.arange(nsubwave)-halfnsubwave).astype(int),[nline,nsubwave])) 

    fluxes = param[find]
    refwaves = param[wind]
    indrefwaves_real = (refwaves - wave[0]) / dispersion
    indrefwaves = (indrefwaves_real).astype(int)
    indrefwaves_frac = indrefwaves_real - indrefwaves.astype(float)
    # flipped the (row,column) indices of indrefwaves_frac rebin (not sure if it works)
    dwaves = (indsubwaves - np.resize(indrefwaves_frac,[nsubwave,nline]))*dispersion
    # flipped the (row,column) indices of indrefwaves rebin
    indsubwaves += np.resize(indrefwaves,[nsubwave,nline])
    
    yvals = np.zeros(nwave)
    for i in range (0,nline):
        # flipped the (row,column) indices for indsubwaves
        # Python cannot do a where(condition 1 and condition 2)  extraction like in IDL, 
        # so I need to treat the limits separately and merge the indices for the 2 conditions
        gind1 = np.where(indsubwaves[:,i] >= 0)[0]    
        gind2 = np.where(indsubwaves[:,i] <= nwave-1)[0]
        gind=list(set(gind1) & set(gind2))
        count = len(gind)
        # The "mask" parameter eliminates floating underflow by removing very large
        # negative exponents. See http://www.idlcoyote.com/math_tips/underflow.html
        # for more details.
        if count > 0:
            exparg = -np.power((dwaves[gind,i]/sigs[i]),2.)/2.    # flipped the (row,column) indices 
            # edited the masking code
            # mask = (abs(exparg) lt 80)
            ms = np.where(abs(exparg) < 80)[0]
            mask = np.zeros(len(exparg))
            mask[ms] = np.ones(len(ms))
            # flipped the (row,column) indices for indsubwaves
            yvals[indsubwaves[gind,i]] = np.add(yvals[indsubwaves[gind,i]],np.transpose(fluxes[i]*mask*np.exp(exparg*mask)))
            # yvals[indsubwaves[gind,i]] = np.transpose(fluxes[i]*mask*np.exp(exparg*mask))

    return yvals

