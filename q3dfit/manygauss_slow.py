# -*- coding: utf-8 -*-
"""
; docformat = 'rst'
;
;+
;
; This function is input to MPFIT, which uses it to compute the
; emission line spectrum of multiple Gaussian emission lines
; simultaneously. This routine assumes nothing about dispersion, and
; thus is slower than MANYGAUSS because it uses larger arrays.
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
;    specresarr: in, optional, type=dblarr(M,2)
;      Array containing spectral resolution look-up table (FWHM as R = lam/dlam).
;      First column is wavelength, second is R.
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
;      2009, DSNR, copied base code from Harus Jabran Zahid and re-wrote
;      2010nov05, DSNR, added sigma in velocity space
;      2013nov13, DSNR, switched default sigma to velocity space
;      2013nov13, DSNR, documented, renamed, added license and copyright 
;      2018feb22, DSNR, now accepts spectral resolution look-up table
;      2020may28, YI, trough translation to Python 3 
;    
; :Copyright:
;    Copyright (C) 2013--2018 David S. N. Rupke
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

def manygauss_slow (wave,param,specresarr=None):
    c=299792.458

    ppoff = param[0]
    nwave = len(wave)
    nline = (len(param)-ppoff)/3

    find = ppoff+np.arange(nline)*3
    wind = find + 1
    sind = find + 2

    # removed the REFORM(), since it's unnecessary in Python.
    # Due to differences in row/column indexing in IDL vs Python, the procedure is: 
    # (a) Rebin the array into an NxM matrix, then (b) take the tranpose to preserve the same matrix structure.
    rrwave = np.resize(wave,[nline,nwave])
    fluxes = np.transpose(np.resize(param[find],[nwave,nline]))
    refwaves = np.transpose(np.resize(param[wind],[nwave,nline]))
    sigs = np.transpose(np.resize(param[sind],[nwave,nline]))

    dwave = rrwave-refwaves
    sigslam = sigs/c*refwaves
    if specresarr :
       wsr = np.searchsorted(specresarr[0,:],param[wind])
       srsigslam = param[wind]/specresarr[1,wsr]/2.35 
       srsigslam = np.transpose(np.resize(param[wind]/specresarr[1,wsr]/2.35 ,[nwave,nline]))
       sigslam = np.sqrt(np.power(sigslam,2.) + np.power(srsigslam,2.))

    yvals = fluxes*np.exp(-dwave*dwave/sigslam/sigslam/2.)
  
    ysum = yvals.sum(axis=0) # axis 0 sums by column values
  
    return ysum

