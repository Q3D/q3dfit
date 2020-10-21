# -*- coding: utf-8 -*-
"""
; docformat = 'rst'
;
;+
;
; This function is the core routine to fit the continuum and emission
; lines of a spectrum.
;
; The function requires an initialization structure with one required
; and a bunch of optional tags, specified in INITTAGS.txt.
;
;
; :Categories:
;    IFSFIT
;
; :Returns:
;    A structure that contains the fit and much else ...
;
; :Params:
;    lambda: in, required, type=dblarr(npix)
;      Spectrum, observed-frame wavelengths.
;    flux: in, required, type=dblarr(npix)
;      Spectrum, fluxes.
;    err: in, required, type=dblarr(npix)
;      Spectrum, flux errors.
;    zstar: in, required, type=structure
;      Initial guess for stellar redshift
;    linelist: in, required, type=hash(lines)
;      Emission line rest frame wavelengths.
;    linelistz: in, required, type=hash(lines\,ncomp)
;      Emission line observed frame wavelengths.
;    ncomp: in, required, type=hash(lines)
;      Number of components fit to each line.
;    initdat: in, required, type=structure
;      Structure of initialization parameters, with tags specified in
;      INITTAGS.txt.
;
; :Keywords:
;    maskwidths: in, optional, type=hash(lines\,maxncomp)
;      Widths, in km/s, of regions to mask from continuum fit. If not
;      set, routine defaults to +/- 500 km/s. Can also be set in INITDAT. 
;      Routine prioritizes the keyword definition.
;    peakinit: in, optional, type=hash(lines\,maxncomp)
;      Initial guesses for peak emission-line flux densities. If not
;      set, routine guesses from spectrum. Can also be set in INITDAT.
;      Routine prioritizes the keyword definition.
;    siginit_gas: in, optional, type=hash(lines\,maxncomp)
;      Initial guess for emission line widths for fitting.
;    siglim_gas: in, optional, type=dblarr(2)
;      Sigma limits for line fitting.
;    tweakcntfit: in, optional, type=dblarr(3\,nregions)
;      Parameters for tweaking continuum fit with localized polynomials. For 
;      each of nregions regions, array contains lower limit, upper limit, and 
;      polynomial degree.
;    quiet: in, optional, type=byte
;      Use to prevent detailed output to screen. Default is to print
;      detailed output.
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
;      2009, DSNR, copied base code from Harus Jabran Zahid
;      2009may, DSNR, tweaked for LRIS data
;      2009jun/jul, DSNR, rewritten
;      2010jan28, DSNR, fitting now done in observed frame, not rest frame
;      2010mar18, DSNR, added ct_coeff output to continuum fit
;      2013sep, DSNR, complete re-write
;      2013nov13, DSNR, renamed, added license and copyright 
;      2013nov25, DSNR, changed structure tags of output spectra for clarity
;      2013dec09, DSNR, removed stellar z and sig optimization;
;                       added PPXF option
;      2013dec10, DSNR, removed docs of initdat tags, since it's
;                       repeated in INITTAGS.txt; removed linelabel
;                       parameter, since it's in initdat; changed
;                       'initstr' parameter to 'initdat', for
;                       consistency with IFSF; testing and bug fixes
;      2013dec11, DSNR, added MASK_HALFWIDTH variable; changed value
;                       from 500 to 1000 km/s
;      2013dec12, DSNR, added SIGINIT_GAS_DEFAULT variable
;      2013dec17, DSNR, started propagation of hashes through code and 
;                       implementation of new calling sequence rubric
;      2014jan13, DSNR, propagated use of hashes
;      2014jan16, DSNR, updated treatment of redshifts; bugfixes
;      2014jan17, DSNR, bugfixes; implemented SIGINIT_GAS, TWEAKCNTFIT keywords
;      2014feb17, DSNR, removed code that added "treated" templates
;                       prior to running a generic continuum fitting
;                       routine (rebinning, adding polynomials, etc.);
;                       i.e., generic continuum fitting routine is now
;                       completely generic
;      2014feb26, DSNR, replaced ordered hashes with hashes
;      2014apr23, DSNR, changed MAXITER from 1000 to 100 in call to MPFIT
;      2016jan06, DSNR, allow no emission line fit with initdat.noemlinfit
;      2016feb02, DSNR, handle cases with QSO+stellar PPXF continuum fits
;      2016feb12, DSNR, changed treatment of sigma limits for emission lines
;                       so that they can be specified on a pixel-by-pixel basis
;      2016aug31, DSNR, added option to mask continuum range(s) by hand with
;                       INITDAT tag MASKCTRAN
;      2016sep13, DSNR, added internal logic to check if emission-line fit present
;      2016sep16, DSNR, allowed MASKWIDTHS_DEF to come in through INITDAT
;      2016sep22, DSNR, tweaked continuum function call to allow new continuum
;                       fitting capabilities; moved logging of things earlier
;                       instead of ensconcing in PPXF loop, for use of PPXF 
;                       elsewhere; new output tag CONT_FIT_PRETWEAK
;      2016oct03, DSNR, multiply PERROR by reduced chi-squared, per prescription
;                       in MPFIT documentation
;      2016oct11, DSNR, added calculation of fit residual
;      2016nov17, DSNR, changed FTOL in MPFITFUN call from 1d-6 to 
;                       default (1d-10)
;      2018mar05, DSNR, added option to convolve template with spectral resolution
;                       profile
;      2018may30, DSNR, added option to adjust XTOL and FTOL for line fitting
;      2018jun25, DSNR, added NOEMLINMASK switch, distinct from NOEMLINFIT
;      2020jun16, YI, rough translation to Python 3; changed all "lambda" variables to "wlambda" since it is a Python keyword
;      2020jun22, YI, replaced emission line MPFIT with LMFIT (testing separately)
;      2020jun22, YI, added scipy modules to extract XDR data (replace the IDL restore function)
;      2020jun23, YI, importing Nadia's airtovac() and the pPPXF log_rebin() functions
;      2020jun24, YI, importing the copy module and creating duplicate flux and err variables. I keep getting "ValueError: assignment destination is read-only"
;      2020jun26, YI, fixed bugs. tested the manygauss() emission line fit call. skipped the continuum fits
;      2020jun28, YI, tested the gmos.py line initialization calls for parameter set-up. minor changes
;      2020jul01, DSNR, bug fixes
;      2020jul07, DSNR, bug fixes; it runs all the way through now.
;      2020jul08, YI, cleaned up the emission line fit section; variables for new many_gauss.py
;      2020jul10, YI, bug fixes in perror_resid bloc; ran successfully
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

import importlib
import numpy as np
import time
from scipy import interpolate
import scipy.io as sio
from scipy.io import readsav
from q3dfit.common.airtovac import airtovac
from q3dfit.common.masklin import masklin
from q3dfit.common import fitqsohost
from ppxf.ppxf_util import log_rebin
import copy
import pdb

def fitspec(wlambda,flux,err,dq,zstar,linelist,linelistz,ncomp,initdat,
            maskwidths=None,peakinit=None,quiet=None,siginit_gas=None,
            siglim_gas=None,tweakcntfit=None,col=None,row=None):

    flux_out = flux
    err_out = err
    
    flux = copy.deepcopy(flux)
    err = copy.deepcopy(err)
    
    # flux.setflags(write=True)
    # err.setflags(write=True)
    
    c = 299792.458         # speed of light, km/s
    siginit_gas_def = 100.  # default sigma for initial guess 
                            # for emission line widths
    
    
    if 'lines' in initdat :
        nlines = len(initdat['lines'])
        linelabel = initdat['lines']
    else :
        linelabel = b'0'
        
# converts the astropy.Table structure of linelist into a Python dictionary that is compatible with the code downstream
    lines_arr = {name:linelist['lines'][idx] for idx,name in enumerate(linelist['name']) }

        
    if quiet :
        quiet=b'1'
    else:
        quiet=b'0'
    if siglim_gas.all():
        siglim_gas=siglim_gas
    else:
        siglim_gas=b'0'
    
    if 'fcnlinefit' in initdat :
        fcnlinefit=initdat['fcnlinefit']
    else:
        fcnlinefit='manygauss'
    if 'argslinefit' in initdat :
        argslinefit=initdat['argslinefit']
    if 'nomaskran' in initdat :
        nomaskran=initdat['nomaskran']
    else:
        nomaskran=b'0'
        
    if 'STARTEMPFILE' in initdat :
        istemp = b'1'
    else:
        istemp=b'0'
        
    if 'loglam' in initdat :
        loglam=b'1'
    else:
        loglam=b'0'
    if 'vacuum' in initdat :
        vacuum=b'1'
    else:
        vacuum=b'0'
    if 'ebv_star' in initdat :
        ebv_star=initdat['ebv_star']
    else:
        ebv_star=[]
    if 'maskwidths_def' in initdat :
        maskwidths_def = initdat['maskwidths_def']
    else:
        maskwidths_def = 1000. # default half-width in km/s for emission line masking
    if 'mpfit_xtol' in initdat :
        mpfit_xtol=initdat['mpfit_xtol']
    else:
        mpfit_xtol=1.-10
    if 'mpfit_ftol' in initdat :
        mpfit_ftol=initdat['mpfit_ftol']
    else:
        mpfit_ftol=1.-10
    
    noemlinfit = b'0'
    if 'noemlinfit' in initdat :
        ct_comp_emlist = 0
    else:
        nocomp_emlist = np.where(np.array(list(ncomp.values())) == 0)[0]
        comp_emlist = np.where(np.array(list(ncomp.values())) != 0)[0]
        ct_comp_emlist = len(comp_emlist)
    if ct_comp_emlist == 0 :
        noemlinfit=b'1'
    
    noemlinmask = b'0'
    if noemlinfit == b'1' and 'doemlinmask' not in initdat :
        noemlinmask = b'1'
    
    if bool(int(istemp)):
    # Get stellar templates
        startempfile = initdat['STARTEMPFILE']
        if isinstance(startempfile,bytes):
            startempfile = startempfile.decode('utf-8')
        
        sav_data = readsav(startempfile)
        template = sav_data['template']
        # restore,initdat.startempfile
    # Redshift stellar templates
        templatelambdaz = np.copy(template['lambda'][0])
        if 'keepstarz' not in initdat :
            templatelambdaz *= 1. + zstar
        if vacuum == b'1' :
            templatelambdaz = airtovac(templatelambdaz)
        if 'waveunit' in initdat :
            templatelambdaz *= initdat['waveunit']
        if 'fcnconvtemp' in initdat :
            impModule = __import__(initdat.fcnconvtemp)
            fcnconvtemp = getattr(impModule,initdat['fcnconvtemp'])
            if 'argsconvtemp' in initdat :
                newtemplate = fcnconvtemp(templatelambdaz,template,*initdat.argsconvtemp)
            else:
                newtemplate = fcnconvtemp(templatelambdaz,template)
    else:
        templatelambdaz = wlambda
    # Set up error in zstar
    zstar_err = 0.
    
    
# ;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;
# # Pick out regions to fit
# ;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;

    flux_raw = flux
    err_raw = err
    if 'fitran' in initdat :
        fitran_tmp = initdat['fitran']
    else:
        fitran_tmp = [wlambda[0],wlambda[len(wlambda)-1]]
    # indices locating good data and data within fit range
    gd_indx_1 = set(np.where(flux != 0)[0])
    gd_indx_2 = set(np.where(err > 0)[0])
    gd_indx_4 = set(np.where(np.isfinite(flux) == True)[0])
    gd_indx_6 = set(np.where(np.isfinite(err) == True)[0])
    gd_indx_7 = set(np.where(dq == 0 )[0])
    gd_indx_8 = set(np.where(wlambda >= min(templatelambdaz))[0])
    gd_indx_9 = set(np.where(wlambda <= max(templatelambdaz))[0])
    gd_indx_10 = set(np.where(wlambda >= fitran_tmp[0])[0])
    gd_indx_11 = set(np.where(wlambda <= fitran_tmp[1])[0])
    
    gd_indx_full = gd_indx_1.intersection(gd_indx_2,gd_indx_4,gd_indx_6,gd_indx_7,gd_indx_8,gd_indx_9,gd_indx_10,gd_indx_11)
    gd_indx_full = list(gd_indx_full)
    fitran = [min(wlambda[gd_indx_full]),max(wlambda[gd_indx_full])]
    
    # Find where flux is <= 0 or error is <= 0 or infinite or NaN
    # (Otherwise MPFIT chokes.)
    neg_indx = np.where(flux > 0)[0]
    ctneg = len(neg_indx)
    
    zerinf_indx_1 = np.where(flux == 0)[0]
    zerinf_indx_2 = np.where(err <= 0)[0]
    zerinf_indx_3 = np.where(np.isfinite(flux) == False)[0]
    zerinf_indx_5 = np.where(np.isfinite(err) == False)[0]
    zerinf_indx = np.unique(np.hstack([zerinf_indx_1,zerinf_indx_2,zerinf_indx_3,zerinf_indx_5]))
    
    ctzerinf = len(zerinf_indx)
    maxerr = max(err[gd_indx_full])
    
    
    # if ctneg gt 0 then begin
    #   flux[neg_indx]=-1d*flux[neg_indx]
    #   err[neg_indx]=maxerr*100d
    #   if not quiet then print,'Setting ',ctneg,' points from neg. flux to pos. '+$
    #       'and max(err)x100.',format='(A,I0,A)'
    if ctzerinf > 0 :
        flux[zerinf_indx]= np.median(flux[gd_indx_full])
        err[zerinf_indx]=maxerr*100.
        if quiet != None :
            print('{:s}{:0.1f}{:s}'.format('Setting ',ctzerinf,' points from zero/inf./NaN flux or '+'neg./zero/inf./NaN error to med(flux) and max(err)x100.'))

    # indices locating data within actual fit range
    fitran_indx1 = np.where(wlambda >= fitran[0])[0]
    fitran_indx2 = np.where(wlambda <= fitran[1])[0]
    fitran_indx = np.intersect1d(fitran_indx1,fitran_indx2)
    ctfitran = len(fitran_indx)
    
    # indices locating good regions within wlambda[fitran_indx]
    gd_indx_full_rezero = gd_indx_full - fitran_indx[0]
    max_gd_indx_full_rezero = max(fitran_indx) - fitran_indx[0]
    
    igdfz1 = np.where(gd_indx_full_rezero >= 0)[0]
    igdfz2 = np.where(gd_indx_full_rezero <= max_gd_indx_full_rezero)[0]
    i_gd_indx_full_rezero = np.intersect1d(igdfz1,igdfz2)
    ctgd = len(i_gd_indx_full_rezero)
    
    gd_indx = gd_indx_full_rezero[i_gd_indx_full_rezero]
    
    # Limit data to fitrange
    npix     = len(fitran_indx)
    gdflux   = flux[fitran_indx]
    gdlambda = wlambda[fitran_indx]
    gderr    = err[fitran_indx]
    
    # Weight
    gdweight = 1./np.power(gderr,2.)
    
    # Log rebin galaxy spectrum for finding bad regions in log space
    gdflux_log,_,_  = log_rebin(fitran,flux_raw[fitran_indx])
    gderrsq_log,_,_ = log_rebin(fitran,np.power(err_raw[fitran_indx],2.))
    gderr_log = np.sqrt(gderrsq_log)
    #  neg_indx_log = where(gdflux_log lt 0,ctneg_log)
    
    zil1 = set(np.where(gdflux_log == 0)[0])
    zil2 = set(np.where(gdflux_log <= 0)[0])
    zil3 = set(np.where(np.isfinite(gderr_log) == False)[0])
    zerinf_indx_log = list(zil1.intersection(zil2,zil3))
    ctzerinf_log = len(zerinf_indx_log)
    gd_indx_log = np.arange(ctfitran)
    if ctzerinf_log > 0:
        gd_indx_log = np.setdiff1d(gd_indx_log,zerinf_indx_log)

    # Log rebin galaxy spectrum for use with PPXF, this time with 
    # errors corrected before rebinning
    gdflux_log,gdlambda_log,velscale = log_rebin(fitran,gdflux)
    gderrsq_log,_,_ = log_rebin(fitran,np.power(gderr,2.))
    
    gderr_log = np.sqrt(gderrsq_log)

    # timer
    fit_time0 = time.time()

# ;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;
# # Fit continuum
# ;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;
    testing = 1
    if 'fcncontfit' in initdat and testing !=1:
        print('this will do the continuum fits...')
        print('----------------------------------------\nfitspec() debug test; STOP\n----------------------------------------')
        return
        
        qsoxdr=argscontfit_use[‘qsoxdr’]
        
        continuum = fitqsohost.fitqsohost(wlambda,flux,err,0,0,gd_indx_full,qsoxdr='nucleartemplate.npy',qsoonly=1,qsoord=1,hostonly=1)
        
        
        # Mask emission lines
        if noemlinfit != b'1':
            pass # this is just a placeholder for now
            if maskwidths == None:
                if 'maskwidths' in initdat:
                    maskwidths = initdat['maskwidths']
                else:
                    maskwidths = initdat['lines']
                    for line in initdat['lines']:
                        maskwidths[line] = np.zeros(initdat['maxncomp'])+maskwidths_def
            pass # this is just a placeholder for now
            ct_indx  = masklin(gdlambda, linelistz, maskwidths, nomaskran=nomaskran)
            # Mask emission lines in log space
            ct_indx_log = masklin(np.exp(gdlambda_log), linelistz, maskwidths, nomaskran=nomaskran)
        else:   
            ct_indx = np.arange(len(gdlambda))
            ct_indx_log = np.arange(len(gdlambda_log))
            
        ct_indx = np.intersect1d(ct_indx,gd_indx)
        ct_indx_log = np.intersect1d(ct_indx_log,gd_indx_log)
        
        ###############################
        # NOT USED IN IFSF_FITSPEC.PRO
        ###############################
        # Mask other regions
        # Now doing this in IFSF_FITLOOP with CUTRANGE tag
        # if tag_exist(initdat,'maskctran') then begin
        # mrsize = size(initdat.maskctran)
        # nreg = 1
        # if mrsize[0] gt 1 then nreg = mrsize[2]
        #     for k=0,nreg-1 do begin
        #         indx_mask = where(gdlambda ge initdat.maskctran[0,k] AND $
        #                           gdlambda le initdat.maskctran[1,k],ct)
        #         indx_mask_log = where(exp(gdlambda) ge initdat.maskctran[0,k] AND $
        #                               exp(gdlambda) le initdat.maskctran[1,k],ct)
        #         if ct gt 0 then begin
        #            ct_indx = cgsetdifference(ct_indx,indx_mask)
        #            ct_indx_log = cgsetdifference(ct_indx_log,indx_mask_log)
        #         endif
        #     endfor
        #  endif

    # ;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;
    # # Option 1: Input function
    # ;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;
        if initdat['fcncontfit'] != 'ppxf':
            if istemp:
                templatelambdaz_tmp = templatelambdaz
                templateflux_tmp = template['flux']
            else:
                templatelambdaz_tmp = b'0'
                templateflux_tmp = b'0'
                
            if 'argscontfit' in initdat:
                argscontfit_use = initdat['argscontfit']
                if initdat['fcncontfit'] == 'ifsf_fitqsohost' :
                    pass # this is just a placeholder for now
                    # argscontfit_use = create_struct(argscontfit_use,'fitran',fitran)
                if 'uselog' in initdat['argscontfit']:
                    pass # this is just a placeholder for now
                    # argscontfit_use = create_struct(argscontfit_use,'index_log',ct_indx_log)
                if 'usecolrow' in initdat['argscontfit'] and col and row:
                    pass # this is just a placeholder for now
                    # argscontfit_use = create_struct(argscontfit_use,'colrow',[col,row])
                # continuum = call_function(initdat.fcncontfit,gdlambda,gdflux,
                #                           gdweight,templatelambdaz_tmp,templateflux_tmp,
                #                           ct_indx,ct_coeff,zstar,
                #                           quiet=quiet,_extra=argscontfit_use)
                ppxf_sigma=0.
                if initdat['fcncontfit'] == 'ifsf_fitqsohost' and 'refit' in initdat['argscontfit']:
                    pass # this is just a placeholder for now
                    # ppxf_sigma=ct_coeff.ppxf_sigma
            else:
                pass # this is just a placeholder for now
                # continuum = call_function(initdat.fcncontfit,gdlambda,gdflux,
                #                           gdweight,templatelambdaz_tmp,templateflux_tmp,
                #                           ct_indx,ct_coeff,zstar,quiet=quiet)
                ppxf_sigma=0.
            add_poly_weights=0.
            ct_rchisq=0.
            ppxf_sigma_err=0.

        # ;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;
        # # Option 2: PPXF
        # ;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;
        elif (istemp == b'1' and 'siginit_stars' in initdat):
            # Interpolate template to same grid as data
            pass # this is just a placeholder for now
            # temp_log = ifsf_interptemp(gdlambda_log,alog(templatelambdaz),$
            #                             template.flux)
    
            # Check polynomial degree
            add_poly_degree = 4
            if 'argscontfit' in initdat:
                # if tag_exist(initdat.argscontfit,'add_poly_degree'):
                pass # this is just a placeholder for 
                    # add_poly_degree = initdat.argscontfit.add_poly_degree
    
            # This ensures PPXF doesn't look for wlambda if no reddening is done
            if len(ebv_star) == 0:
                redlambda = []
            else:
                redlambda=np.exp(gdlambda_log)
    
            # Add QSO template as sky spectrum so that it doesn't get convolved with anything.
            if 'qsotempfile' in initdat:
                pass # this is just a placeholder for 
            #     restore,initdat.qsotempfile
            #     log_rebin,[gdlambda[0],gdlambda[n_elements(gdlambda)-1]],$
            #               struct.cont_fit,$
            #               gdqsotemp_log,gdlambda_log_tmp
            #     sky=gdqsotemp_log       
            else:
                sky=0.
    
            # ppxf,temp_log,gdflux_log,gderr_log,velscale,$
            #       [0,initdat.siginit_stars],sol,$
            #       goodpixels=ct_indx_log,bestfit=continuum_log,moments=2,$
            #       degree=add_poly_degree,polyweights=add_poly_weights,quiet=quiet,$
            #       weights=ct_coeff,reddening=ebv_star,lambda=redlambda,sky=sky,$
            #       error=solerr
    
            # Resample the best fit into linear space
            continuum = []
            # continuum = interpol(continuum_log,gdlambda_log,ALOG(gdlambda))
    
            # Adjust stellar redshift based on fit
            sol=[]
            zstar += sol[0]/c
            ppxf_sigma=sol[1]
    
            # From PPXF docs:
            # - These errors are meaningless unless Chi^2/DOF~1 (see parameter SOL below).
            # However if one *assume* that the fit is good, a corrected estimate of the
            # errors is: errorCorr = error*sqrt(chi^2/DOF) = error*sqrt(sol[6]).
            ct_rchisq = sol[6]
            solerr = []
            solerr *= np.sqrt(sol[6])
            zstar_err = np.sqrt(np.power(zstar_err,2.) + np.power((solerr[0]/c),2.))
            ppxf_sigma_err=solerr[1]

        else:
            add_poly_weights=0.
            ct_rchisq=0.
            ppxf_sigma=0.
            ppxf_sigma_err=0.

# ;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;
# # Option to tweak cont. fit with local polynomial fits
# ;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;
        if not tweakcntfit:
            continuum_pretweak=continuum
        # Arrays holding emission-line-masked data
            ct_lambda=gdlambda[ct_indx]
            ct_flux=gdflux[ct_indx]
            ct_err=gderr[ct_indx]
            ct_cont = continuum[ct_indx]
            for i in range(len(tweakcntfit[0,:])):
            # Indices into full data
                tmp_ind1 = np.where(gdlambda >= tweakcntfit[i,0])[0]
                tmp_ind2 = np.where(gdlambda <= tweakcntfit[i,1])[0]
                tmp_ind = np.intersect1d(tmp_ind1,tmp_ind2)
                ct_ind = len(tmp_ind)
            # Indices into masked data
                tmp_ctind1 = np.where(ct_lambda >= tweakcntfit[i,0])[0]
                tmp_ctind2 = np.where(ct_lambda <= tweakcntfit[i,1])[0]
                tmp_ctind = np.intersect1d(tmp_ctind1,tmp_ctind2)
                ct_ctind = len(tmp_ctind)
                
                if ct_ind > 0 and ct_ctind > 0 :
                    parinfo =  list(np.repeat({'value':0.},tweakcntfit[2,i]+1))
                    # parinfo = replicate({value:0d},tweakcntfit[2,i]+1)
                    pass # this is just a placeholder for now
                    # tmp_pars = mpfitfun('poly',ct_lambda[tmp_ctind],$
                    #                     ct_flux[tmp_ctind] - ct_cont[tmp_ctind],$
                    #                     ct_err[tmp_ctind],parinfo=parinfo,/quiet)
                    # continuum[tmp_ind] += poly(gdlambda[tmp_ind],tmp_pars)
        else:
            continuum_pretweak=continuum
         
            if 'dividecont' in initdat :
                gdflux_nocnt = gdflux / continuum - 1
                gdweight_nocnt = gdweight * np.power(continuum,2.)
                gderr_nocnt = gderr / continuum
                method   = 'CONTINUUM DIVIDED'
            else:
                gdflux_nocnt = gdflux - continuum
                gdweight_nocnt = gdweight
                gderr_nocnt = gderr
                method   = 'CONTINUUM SUBTRACTED'
    else:
        add_poly_weights=0.
        gdflux_nocnt = gdflux
        gderr_nocnt = gderr
        method   = 'NO CONTINUUM FIT'
        continuum = 0.
        continuum_pretweak = 0.
        ct_coeff = 0.
        ct_indx = 0.
        ct_rchisq=0.
        ppxf_sigma=0.
        ppxf_sigma_err=0.

    fit_time1 = time.time()
    if quiet != None :
        print('{:s}{:0.1f}{:s}'.format('FITSPEC: Continuum fit took ',fit_time1-fit_time0,' s.'))
    
# ;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;
# Fit emission lines
# ;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;
    if noemlinfit != b'1':
    # Initial guesses for emission line peak fluxes (above continuum)
    # If initial guess is negative, set to 0 to prevent MPFITFUN from choking 
    #  (since we limit peak to be >= 0).
        if peakinit == None:
            if 'peakinit' in initdat :
                peakinit = initdat['peakinit']
            else:
                peakinit = {line:None for line in initdat['lines']}
                for line in initdat['lines'] :
                    fline = interpolate.interp1d(gdlambda,gdflux_nocnt, kind='linear')
                    peakinit[line] = np.array([fline(linelistz[line][0]).item()])
                    neg = np.where(peakinit[line] < 0)[0]
                    ct = len(neg)
                    if ct > 0 :
                        peakinit[line] = [0]
        
        # Initial guesses for emission line widths
        if siginit_gas == None:
            siginit_gas = {k:None for k in initdat['lines']}
            for line in initdat['lines']:
                siginit_gas[line] = np.zeros(initdat['maxncomp'])+siginit_gas_def
            
        ## Normalize data so it's near 1. Use 95th percentile of flux. If it's far from
        ## 1, results are different, and probably less correct b/c of issues of numerical
        ## precision in calculating line properties.
        #  ifsort = sort(gdflux_nocnt)
        #  fsort = gdflux_nocnt[ifsort]
        #  i95 = fix(n_elements(gdflux_nocnt)*0.95d)
        #  fnorm = fsort[i95]
        #  gdflux_nocnt /= fnorm
        #  gderr_nocnt /= fnorm
        #  foreach line,initdat.lines do peakinit[line] /= fnorm
        
        # Fill out parameter structure with initial guesses and constraints
        impModule = importlib.import_module('q3dfit.init.'+initdat['fcninitpar'])
        fcninitpar = getattr(impModule,initdat['fcninitpar'])
        
        # running test functions for now.....
        # I need to fix the manygauss() fits
        if 'argsinitpar' in initdat:
            # need to fix the _extra keywords
            parinit = fcninitpar(linelist,linelistz,initdat['linetie'],peakinit,siginit_gas,
                                  initdat['maxncomp'],ncomp,siglim=siglim_gas,_extra=initdat['argsinitpar']) 
        else:
            parinit = fcninitpar(linelist,linelistz,initdat['linetie'],peakinit,siginit_gas,
                                  initdat['maxncomp'],ncomp,siglim=siglim_gas)
            
        testsize = len(parinit)
        if testsize == 0 :
            raise Exception('Bad initial parameter guesses.')
        
        efitModule = importlib.import_module('q3dfit.common.'+fcnlinefit)
        elin_lmfit = getattr(efitModule,'run_'+fcnlinefit)
        lmout,parout,specfit,perror = elin_lmfit(gdlambda,gdflux_nocnt,gderr_nocnt,parinfo=parinit,maxiter=1000)

        param  = parout
        covar  = lmout.covar
        dof    =lmout.nfree
        nfev   = lmout.nfev
        rchisq  = lmout.chisqr
        errmsg = lmout.message
        status = lmout.success
        # the following MPFIT variables that were not compatible with LMFIT,
        # I have deleted these from the final output structure where applicable
        # niter=niter 
        # quiet=quiet
        # npegged=npegged
        # functargs=argslinefit
        # xtol=mpfit_xtol
        # ftol=mpfit_ftol

        # Un-normalize fit. (NOT USED)
        #  specfit *= fnorm
        #  gdflux_nocnt *= fnorm
        #  gderr_nocnt *= fnorm
        #  foreach line,linelist.keys() do begin
        #     iline = where(parinit.line eq line)
        #     ifluxpk = cgsetintersection(iline,where(parinit.parname eq 'flux_peak'),$
        #                                 count=ctfluxpk)
        #     param[ifluxpk] *= fnorm
        #     perror[ifluxpk] *= fnorm
        #  endforeach
        
        # error messages corresponding to LMFIT, documentation was not very helpful with the error messages...
        if status == False  :
            raise Exception('LMFIT: '+errmsg)
        # #     outstr = 0
        # #     goto,finish
        # if status == 5 :
        #     print('LMFIT: Max. iterations reached.')

        # Errors from covariance matrix ...
#        rchisq = chisq/dof
        perror = np.multiply(perror,np.sqrt(rchisq))
        # perror =  np.sqrt(rchisq)
        # ... and from fit residual.
        resid=gdflux-continuum-specfit
        perror_resid = perror
        sigrange = 20.
        for line in lines_arr:
            # iline = next(idx for idx,item in enumerate(parinit) if item['line'] == line)
            iline = np.array([ip for ip,item in enumerate(parinit) if item['line'] == line])
            ifluxpk = np.intersect1d(iline,np.array([ip for ip,item in enumerate(parinit) if item['parname'] == 'flux_peak']))
            ctfluxpk = len(ifluxpk)
            isigma = np.intersect1d(iline,np.array([ip for ip,item in enumerate(parinit) if item['parname'] == 'sigma']))
            iwave = np.intersect1d(iline,np.array([ip for ip,item in enumerate(parinit) if item['parname'] == 'wavelength']))
            for i in range(0,ctfluxpk):
                waverange = sigrange*np.sqrt(np.power((param[isigma[i]]/c*param[iwave[i]]),2.) + np.power(param[2],2.))
                wlo = np.searchsorted(gdlambda,param[iwave[i]]-waverange/2.)
                whi = np.searchsorted(gdlambda,param[iwave[i]]+waverange/2.)
                if gdlambda[wlo] < gdlambda[0] or wlo == -1:
                    wlo=0
                if gdlambda[whi] > gdlambda[len(gdlambda)-1] or whi == -1 :
                    whi=len(gdlambda)-1
                if param[ifluxpk[i]] > 0 :
                    perror_resid[ifluxpk[i]] = np.sqrt(np.mean(np.power(resid[wlo:whi],2.)))
         
        outlinelist = linelist # this bit of logic prevents overwriting of linelist
        cont_dat = gdflux - specfit
    else:
        cont_dat = gdflux
        specfit = 0
        chisqr = 0
        dof = 1
        niter = 0
        status = 0
        outlinelist = 0
        parinit = 0
        param = 0
        perror = 0
        perror_resid = 0
        covar = 0
    # This sets the output reddening to a numerical 0 instead of NULL
    if len(ebv_star) == 0 :
        ebv_star=0.
        fit_time2 = time.time()
        if quiet != None :
            print('{:s}{:0.1f}{:s}'.format('FITSPEC: Line fit took ',fit_time2-fit_time1,' s.'))
                
                
#;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;
# Output structure
#;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;

# restore initial values
    flux = flux_out
    err = err_out
    
    # need to adjust the output values here...
    outstr = {'fitran': fitran,
              # Continuum fit parameters
              'ct_method': method, 
              'ct_coeff': ct_coeff, 
              'ct_ebv': ebv_star, 
              'zstar': zstar, 
              'zstar_err': zstar_err,
              'ct_add_poly_weights': add_poly_weights,
              'ct_ppxf_sigma': ppxf_sigma,
              'ct_ppxf_sigma_err': ppxf_sigma_err,
              'ct_rchisq': ct_rchisq,
              # Spectrum in various forms
              'wave': gdlambda, 
              'spec': gdflux,       # data
              'spec_err': gderr, 
              'cont_dat': cont_dat,  # cont. data (all data - em. line fit)
              'cont_fit': continuum,      # cont. fit
              'cont_fit_pretweak': continuum_pretweak,  # cont. fit before tweaking
              'emlin_dat': gdflux_nocnt, # em. line data (all data - cont. fit)
              'emlin_fit': specfit,       # em. line fit
              # gd_indx is applied, and then ct_indx
              'gd_indx': gd_indx,         # cuts on various criteria
              'fitran_indx': fitran_indx, # cuts on various criteria
              'ct_indx': ct_indx,         # where emission is not masked
              # Line fit parameters
              'noemlinfit': noemlinfit,   # was emission line fit done?
              'noemlinmask': noemlinmask, # were emission lines masked?
              'redchisq': rchisq, 
              #'niter': niter, (DOES NOT EXIST)
              'fitstatus': status, 
              'linelist': outlinelist, 
              'linelabel': linelabel, 
              'parinfo': parinit, 
              'param': param, 
              'perror': perror, 
              'perror_resid': perror_resid,  # error from fit residual
#              'covar': covar, 
              'siglim': siglim_gas}
    # finish:
    return outstr
  
