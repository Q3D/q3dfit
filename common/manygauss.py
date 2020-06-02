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

def manygauss(wave,param,specresarr=None):
    c=299792.458

    ppoff = param[0]
    nwave = len(wave)
    nline = (len(param)-ppoff)/3

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

    sigs = np.sqrt(np.power(param[sind]/c * param[wind],2.) + np.power(srsigslam,2.))
    maxsig = np.max(sigs)
    
    nsubwave = np.round(10. * maxsig / dispersion)
    halfnsubwave = np.round(nsubwave / 2)
    nsubwave = halfnsubwave*2+1
    
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
  
    yvals = np.arange(nwave)
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
            yvals[indsubwaves[gind,i]] += np.transpose(fluxes[i]*mask*np.exp(exparg*mask))      

    return yvals

