# -*- coding: utf-8 -*-
"""

  This function is the core routine to fit the continuum and emission
  lines of a spectrum.

  The function requires an initialization structure with one required
  and a bunch of optional tags, specified in INITTAGS.txt.


 :Categories:
     IFSFIT

 :Returns:
     A structure that contains the fit and much else ...

 :Params:
     lambda: in, required, type=dblarr(npix)
       Spectrum, observed-frame wavelengths.
     flux: in, required, type=dblarr(npix)
       Spectrum, fluxes.
     err: in, required, type=dblarr(npix)
       Spectrum, flux errors.
     zstar: in, required, type=structure
       Initial guess for stellar redshift
     listlines: in, required, type=hash(lines)
       Emission line rest frame wavelengths.
     listlinesz: in, required, type=hash(lines\,ncomp)
       Emission line observed frame wavelengths.
     ncomp: in, required, type=hash(lines)
       Number of components fit to each line.
     initdat: in, required, type=structure
       Structure of initialization parameters, with tags specified in
       INITTAGS.txt.

 :Keywords:
     maskwidths: in, optional, type=hash(lines\,maxncomp)
       Widths, in km/s, of regions to mask from continuum fit. If not
       set, routine defaults to +/- 500 km/s. Can also be set in INITDAT.
       Routine prioritizes the keyword definition.
     peakinit: in, optional, type=hash(lines\,maxncomp)
       Initial guesses for peak emission-line flux densities. If not
       set, routine guesses from spectrum. Can also be set in INITDAT.
       Routine prioritizes the keyword definition.
     siginit_gas: in, optional, type=hash(lines\,maxncomp)
       Initial guess for emission line widths for fitting.
     siglim_gas: in, optional, type=dblarr(2)
       Sigma limits for line fitting.
     tweakcntfit: in, optional, type=dblarr(3\,nregions)
       Parameters for tweaking continuum fit with localized polynomials. For
       each of nregions regions, array contains lower limit, upper limit, and
       polynomial degree.
     quiet: in, optional, type=byte
       Use to prevent detailed output to screen. Default is to print
       detailed output.

 :History:
     ChangeHistory::
       2009, DSNR, copied base code from Harus Jabran Zahid
       2009may, DSNR, tweaked for LRIS data
       2009jun/jul, DSNR, rewritten
       2010jan28, DSNR, fitting now done in observed frame, not rest frame
       2010mar18, DSNR, added ct_coeff output to continuum fit
       2013sep, DSNR, complete re-write
       2013nov13, DSNR, renamed, added license and copyright
       2013nov25, DSNR, changed structure tags of output spectra for clarity
       2013dec09, DSNR, removed stellar z and sig optimization;
                        added PPXF option
       2013dec10, DSNR, removed docs of initdat tags, since it's
                        repeated in INITTAGS.txt  removed linelabel
                        parameter, since it's in initdat; changed
                        'initstr' parameter to 'initdat', for
                        consistency with IFSF; testing and bug fixes
       2013dec11, DSNR, added MASK_HALFWIDTH variable; changed value
                        from 500 to 1000 km/s
       2013dec12, DSNR, added SIGINIT_GAS_DEFAULT variable
       2013dec17, DSNR, started propagation of hashes through code and
                        implementation of new calling sequence rubric
       2014jan13, DSNR, propagated use of hashes
       2014jan16, DSNR, updated treatment of redshifts; bugfixes
       2014jan17, DSNR, bugfixes; implemented SIGINIT_GAS, TWEAKCNTFIT keywords
       2014feb17, DSNR, removed code that added "treated" templates
                        prior to running a generic continuum fitting
                        routine (rebinning, adding polynomials, etc.);
                        i.e., generic continuum fitting routine is now
                        completely generic
       2014feb26, DSNR, replaced ordered hashes with hashes
       2014apr23, DSNR, changed MAXITER from 1000 to 100 in call to MPFIT
       2016jan06, DSNR, allow no emission line fit with initdat.noemlinfit
       2016feb02, DSNR, handle cases with QSO+stellar PPXF continuum fits
       2016feb12, DSNR, changed treatment of sigma limits for emission lines
                        so that they can be specified on a pixel-by-pixel basis
       2016aug31, DSNR, added option to mask continuum range(s) by hand with
                        INITDAT tag MASKCTRAN
       2016sep13, DSNR, added internal logic to check if emission-line fit present
       2016sep16, DSNR, allowed MASKWIDTHS_DEF to come in through INITDAT
       2016sep22, DSNR, tweaked continuum function call to allow new continuum
                        fitting capabilities; moved logging of things earlier
                        instead of ensconcing in PPXF loop, for use of PPXF
                        elsewhere; new output tag CONT_FIT_PRETWEAK
       2016oct03, DSNR, multiply PERROR by reduced chi-squared, per prescription
                        in MPFIT documentation
       2016oct11, DSNR, added calculation of fit residual
       2016nov17, DSNR, changed FTOL in MPFITFUN call from 1d-6 to
                        default (1d-10)
       2018mar05, DSNR, added option to convolve template with spectral resolution
                        profile
       2018may30, DSNR, added option to adjust XTOL and FTOL for line fitting
       2018jun25, DSNR, added NOEMLINMASK switch, distinct from NOEMLINFIT
       2020jun16, YI, rough translation to Python 3; changed all "lambda" variables to "wlambda" since it is a Python keyword
       2020jun22, YI, replaced emission line MPFIT with LMFIT (testing separately)
       2020jun22, YI, added scipy modules to extract XDR data (replace the IDL restore function)
       2020jun23, YI, importing Nadia's airtovac() and the pPPXF log_rebin() functions
       2020jun24, YI, importing the copy module and creating duplicate flux and err variables. I keep getting "ValueError: assignment destination is read-only"
       2020jun26, YI, fixed bugs. tested the manygauss() emission line fit call. skipped the continuum fits
       2020jun28, YI, tested the gmos.py line initialization calls for parameter set-up. minor changes
       2020jul01, DSNR, bug fixes
       2020jul07, DSNR, bug fixes; it runs all the way through now.
       2020jul08, YI, cleaned up the emission line fit section; variables for new many_gauss.py
       2020jul10, YI, bug fixes in perror_resid bloc; ran successfully
       2021jan, AV and DSNR, major changes in continuum implementation
       2021jan25, DSNR, fixed bug in errors passing into elin_lmfit
"""

import copy
import numpy as np
import pdb
import time
import pickle
from astropy.table import Table
from importlib import import_module
from lmfit import Model
from ppxf.ppxf import ppxf
from ppxf.ppxf_util import log_rebin
from q3dfit.common.airtovac import airtovac
from q3dfit.common.masklin import masklin
from q3dfit.common.interptemp import interptemp
from scipy.interpolate import interp1d
from q3dfit.common.questfit import questfit
from q3dfit.common.plot_quest import plot_quest
from q3dfit.common.plot_cont import plot_cont


def fitspec(wlambda, flux, err, dq, zstar, listlines, listlinesz, ncomp,
            initdat, maskwidths=None, peakinit=None, quiet=True,
            siginit_gas=None, siglim_gas=None, tweakcntfit=None,
            col=None, row=None):

    flux_out = flux
    err_out = err

    flux = copy.deepcopy(flux)
    err = copy.deepcopy(err)

    # flux.setflags(write=True)
    # err.setflags(write=True)

    c = 299792.458         # speed of light, km/s
    siginit_gas_def = 100.  # default sigma for initial guess
                            # for emission line widths

    if 'ebv_star' in initdat:
        ebv_star=initdat['ebv_star']
    else:
        ebv_star=None
    if 'fcninitpar' in initdat:
        fcninitpar = initdat['fcninitpar']
    else:
        fcninitpar = 'parinit'
    if 'lines' in initdat:
        # nlines = len(initdat['lines'])
        linelabel = initdat['lines']
    else:
        linelabel = b'0'
    if 'maskwidths_def' in initdat:
        maskwidths_def = initdat['maskwidths_def']
    else:
        maskwidths_def = 1000.  # default half-width in km/s for emission line masking
    if 'nomaskran' in initdat:
        nomaskran=initdat['nomaskran']
    else:
        nomaskran=''
    if siglim_gas.all():
        siglim_gas = siglim_gas
    else:
        siglim_gas = b'0'
    if 'startempfile' in initdat:
        istemp = b'1'
    else:
        istemp=b'0'
    if 'vacuum' in initdat:
        vacuum=b'1'
    else:
        vacuum=b'0'

    noemlinfit = b'0'
    if 'noemlinfit' in initdat:
        ct_comp_emlist = 0
    else:
        comp_emlist = np.where(np.array(list(ncomp.values())) != 0)[0]
        ct_comp_emlist = len(comp_emlist)
    if ct_comp_emlist == 0:
        noemlinfit = b'1'

    noemlinmask = b'0'
    if noemlinfit == b'1' and 'doemlinmask' not in initdat:
        noemlinmask = b'1'

    if bool(int(istemp)) and initdat['fcncontfit']!='questfit':

        # Get stellar templates
        startempfile = initdat['startempfile']
        if isinstance(startempfile, bytes):

            startempfile = startempfile.decode('utf-8')

        sav_data = np.load(startempfile, allow_pickle=True).item()
        template = sav_data
        # Redshift stellar templates
        templatelambdaz = np.copy(template['lambda'])
        if 'keepstarz' not in initdat:
            templatelambdaz *= 1. + zstar
        if vacuum == b'1':
            templatelambdaz = airtovac(templatelambdaz)
        if 'waveunit' in initdat:
            templatelambdaz *= initdat['waveunit']
        if 'fcnconvtemp' in initdat:
            impModule = import_module('q3dfit.common.'+initdat['fcnconvtemp'])
            fcnconvtemp = getattr(impModule, initdat['fcnconvtemp'])
            if 'argsconvtemp' in initdat:
                newtemplate = fcnconvtemp(templatelambdaz, template,
                                          **initdat['argsconvtemp'])
            else:
                newtemplate = fcnconvtemp(templatelambdaz, template)
    else:
        templatelambdaz = wlambda
    # Set up error in zstar
    zstar_err = 0.


# ;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;
# # Pick out regions to fit
# ;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;


    flux_raw = flux
    err_raw = err
    if 'fitran' in initdat:
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

    gd_indx_full = gd_indx_1.intersection(gd_indx_2, gd_indx_4, gd_indx_6,
                                          gd_indx_7, gd_indx_8, gd_indx_9,
                                          gd_indx_10, gd_indx_11)
    gd_indx_full = list(gd_indx_full)
    fitran = [min(wlambda[gd_indx_full]), max(wlambda[gd_indx_full])]

    # Find where flux is <= 0 or error is <= 0 or infinite or NaN
    # (Otherwise MPFIT chokes.)
    neg_indx = np.where(flux > 0)[0]
    ctneg = len(neg_indx)

    zerinf_indx_1 = np.where(flux == 0)[0]
    zerinf_indx_2 = np.where(err <= 0)[0]
    zerinf_indx_3 = np.where(np.isfinite(flux) == False)[0]
    zerinf_indx_5 = np.where(np.isfinite(err) == False)[0]
    zerinf_indx = np.unique(np.hstack([zerinf_indx_1, zerinf_indx_2,
                                       zerinf_indx_3, zerinf_indx_5]))

    ctzerinf = len(zerinf_indx)
    maxerr = max(err[gd_indx_full])


    # if ctneg gt 0 then begin
    #   flux[neg_indx]=-1d*flux[neg_indx]
    #   err[neg_indx]=maxerr*100d
    #   if not quiet then print,'Setting ',ctneg,' points from neg. flux to pos. '+$
    #       'and max(err)x100.',format='(A,I0,A)'
    if ctzerinf > 0:
        flux[zerinf_indx]= np.median(flux[gd_indx_full])
        err[zerinf_indx]=maxerr*100.
        if not quiet:
            print('{:s}{:0.1f}{:s}'.
                  format('FITLOOP: Setting ', ctzerinf,
                         ' points from zero/inf./NaN flux or ' +
                         'neg./zero/inf./NaN error to med(flux) ' +
                         'and max(err)x100.'))

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
    testing = 0
    if 'fcncontfit' in initdat and testing != 1:

        # Some defaults. These only apply in case of fitting with stellar model
        # + additive polynomial.
        stel_mod = 0.
        poly_mod = 0.

        # Mask emission lines
        # Note that maskwidths is now an astropy Table
        # Column names are line labels, rows are components
        if noemlinmask != b'1':
            if maskwidths is None:
                if 'maskwidths' in initdat:
                    maskwidths = initdat['maskwidths']
                else:
                    maskwidths = Table(np.full([initdat['maxncomp'],
                                                listlines['name'].size],
                                               maskwidths_def, dtype='float'),
                                       names=listlines['name'])

            ct_indx = masklin(gdlambda/(1+zstar), listlines, maskwidths,
                              nomaskran=nomaskran)
            # Mask emission lines in log space
            ct_indx_log = masklin(np.exp(gdlambda_log)/(1+zstar), listlines,
                                  maskwidths, nomaskran=nomaskran)
        else:
            ct_indx = np.arange(len(gdlambda))
            ct_indx_log = np.arange(len(gdlambda_log))

        ct_indx = np.intersect1d(ct_indx, gd_indx)
        ct_indx_log = np.intersect1d(ct_indx_log, gd_indx_log)

    # ;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;
    # # Option 1: Input function
    # ;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;
        if initdat['fcncontfit'] != 'ppxf':

            module = import_module('q3dfit.common.' + initdat['fcncontfit'])
            fcncontfit = getattr(module, initdat['fcncontfit'])

            if initdat['fcncontfit']=='questfit' or istemp==b'0':
              istemp=None


            if istemp:
                templatelambdaz_tmp = templatelambdaz
                templateflux_tmp = template['flux']
            else:
                templatelambdaz_tmp = b'0'
                templateflux_tmp = b'0'

            if 'argscontfit' in initdat:
                argscontfit_use = initdat['argscontfit']
                if initdat['fcncontfit'] == 'fitqsohost':
                    argscontfit_use['fitran'] = fitran
                if 'uselog' in initdat['argscontfit']:
                    argscontfit_use['index_log'] = ct_indx_log
                if 'usecolrow' in initdat['argscontfit'] and col and row:
                    argscontfit_use['colrow'] = [col, row]

                if initdat['fcncontfit'] == 'fitpoly':
                    continuum = fcncontfit(gdlambda, gdflux, gdweight, templatelambdaz_tmp,
                               templateflux_tmp, ct_indx, zstar, fitord=3,
                               quiet=quiet)[0]
                    ct_coeff=fcncontfit(gdlambda, gdflux, gdweight, templatelambdaz_tmp,
                               templateflux_tmp, ct_indx, zstar, fitord=3,
                               quiet=quiet)[1]
                    zstar=fcncontfit(gdlambda, gdflux, gdweight, templatelambdaz_tmp,
                               templateflux_tmp, ct_indx, zstar, fitord=3,
                               quiet=quiet)[2]
                else:
                  continuum, ct_coeff, zstar = \
                    fcncontfit(gdlambda, gdflux, gdweight, templatelambdaz_tmp,
                               templateflux_tmp, ct_indx, zstar,
                               quiet=quiet, **argscontfit_use)

                ppxf_sigma = 0.
                if initdat['fcncontfit'] == 'ifsf_fitqsohost' and \
                    'refit' in initdat['argscontfit']:
                    ppxf_sigma = ct_coeff['ppxf_sigma']

            else:
                if initdat['fcncontfit'] == 'fitpoly':
                    continuum = fcncontfit(gdlambda, gdflux, gdweight, templatelambdaz_tmp,
                               templateflux_tmp, ct_indx, zstar, fitord=3,
                               quiet=quiet)[0]
                    ct_coeff=fcncontfit(gdlambda, gdflux, gdweight, templatelambdaz_tmp,
                               templateflux_tmp, ct_indx, zstar, fitord=3,
                               quiet=quiet)[1]
                    zstar=fcncontfit(gdlambda, gdflux, gdweight, templatelambdaz_tmp,
                               templateflux_tmp, ct_indx, zstar, fitord=3,
                               quiet=quiet)[2]
                else:
                  continuum = \
                    fcncontfit(gdlambda, gdflux, gderr, templatelambdaz_tmp,
                               templateflux_tmp, ct_indx, ct_coeff, zstar,
                               quiet=quiet)
                ppxf_sigma = 0.

            add_poly_weights = 0.
            ct_rchisq = 0.
            ppxf_sigma_err = 0.

        # ;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;
        # # Option 2: PPXF
        # ;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;
        elif (istemp == b'1' and 'siginit_stars' in initdat):

            # Interpolate template to same grid as data
            temp_log = interptemp.interptemp(gdlambda_log, np.log(templatelambdaz.T[0]),
                                  template['flux'])

            # Check polynomial degree
            add_poly_degree = 4
            if 'argscontfit' in initdat:
                if 'add_poly_degree' in initdat['argscontfit']:
                    add_poly_degree = initdat['argscontfit']['add_poly_degree']

            # run ppxf
            pp = \
                ppxf(temp_log, gdflux_log, gderr_log, velscale,
                     [0, initdat['siginit_stars']], goodpixels=ct_indx_log,
                     degree=add_poly_degree, quiet=quiet, reddening=ebv_star)
            poly_mod = pp.apoly
            continuum_log = pp.bestfit
            add_poly_weights = pp.polyweights
            ct_coeff = pp.weights
            ebv_star = pp.reddening
            sol = pp.sol
            error = pp.error

            # Resample the best fit into linear space
            cinterp = interp1d(gdlambda_log, continuum_log,
                               kind='cubic', fill_value="extrapolate")
            continuum = cinterp(np.log(gdlambda))

            # Adjust stellar redshift based on fit
            zstar += sol[0]/c
            ppxf_sigma = sol[1]

            # From PPXF docs:
            # - These errors are meaningless unless Chi^2/DOF~1 (see parameter SOL below).
            # However if one *assume* that the fit is good, a corrected estimate of the
            # errors is: errorCorr = error*sqrt(chi^2/DOF) = error*sqrt(sol[6]).
            # ct_rchisq = sol[6]
            # solerr = []
            # solerr *= np.sqrt(sol[6])
            # zstar_err = np.sqrt(np.power(zstar_err,2.) + np.power((solerr[0]/c),2.))
            # ppxf_sigma_err=solerr[1]
            ppxf_sigma_err = 0. # for now; correct this later
            ct_rchisq = 0.

        else:
            add_poly_weights = 0.
            ct_rchisq = 0.
            ppxf_sigma = 0.
            ppxf_sigma_err = 0.


# ;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;
# # Option to tweak cont. fit with local polynomial fits
# ;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;
        if 'tweakcntfit' in initdat:
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

                if ct_ind > 0 and ct_ctind > 0:
                    parinfo =  list(np.repeat({'value':0.},tweakcntfit[2,i]+1))
                    # parinfo = replicate({value:0d},tweakcntfit[2,i]+1)
                    pass # this is just a placeholder for now
                    # tmp_pars = mpfitfun('poly',ct_lambda[tmp_ctind],$
                    #                     ct_flux[tmp_ctind] - ct_cont[tmp_ctind],$
                    #                     ct_err[tmp_ctind],parinfo=parinfo,/quiet)
                    # continuum[tmp_ind] += poly(gdlambda[tmp_ind],tmp_pars)
        else:
            continuum_pretweak=continuum

            if 'dividecont' in initdat:
                gdflux_nocnt = gdflux / continuum - 1
                gdweight_nocnt = gdweight * np.power(continuum, 2.)
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
    if not quiet:
        print('{:s}{:0.1f}{:s}'.format('FITSPEC: Continuum fit took ',fit_time1-fit_time0,' s.'))


    #
    # Fit emission lines
    #

    fit_params = []
    if noemlinfit != b'1':

        # Initial guesses for emission line peak fluxes (above continuum)
        if peakinit is None:
            if 'peakinit' in initdat:
                peakinit = initdat['peakinit']
            else:
                peakinit = {line: None for line in initdat['lines']}
                for line in initdat['lines']:
                    fline = interp1d(gdlambda, gdflux_nocnt, kind='linear')
                    # Check that line wavelength is in data range
                    # Use first component as a proxy for all components
                    if listlinesz[line][0] >= min(gdlambda) and \
                        listlinesz[line][0] <= max(gdlambda):
                        peakinit[line] = fline(listlinesz[line])
                        # If initial guess is negative, set to 0 to prevent
                        # fitter from choking (since we limit peak to be >= 0)
                        peakinit[line] = \
                            np.where(peakinit[line] < 0., 0., peakinit[line])
                    else:
                        peakinit[line] = np.zeros(initdat['maxncomp'])

        # Initial guesses for emission line widths
        if siginit_gas is None:
            siginit_gas = {k: None for k in initdat['lines']}
            for line in initdat['lines']:
                siginit_gas[line] = \
                    np.zeros(initdat['maxncomp']) + siginit_gas_def

        # Fill out parameter structure with initial guesses and constraints
        impModule = import_module('q3dfit.init.' + fcninitpar)
        run_fcninitpar = getattr(impModule, fcninitpar)
        emlmod, fit_params = \
            run_fcninitpar(listlines, listlinesz, initdat['linetie'], peakinit,
                           siginit_gas, initdat['maxncomp'], ncomp,
                           siglim=siglim_gas)

        # testsize = len(parinit)
        # if testsize == 0:
            # raise Exception('Bad initial parameter guesses.')

        # Actual fit
        lmout = emlmod.fit(gdflux_nocnt, fit_params, x=gdlambda,
                           method='least_squares', weights=gdweight_nocnt,
                           max_nfev=1000, nan_policy='omit')
        specfit = lmout.best_fit
        if not quiet:
            print(lmout.fit_report(show_correl=False))

        param = lmout.best_values
        if 'plotMIR' in initdat.keys():    # Test plot here - need to transfer this to q3dfa later
          print('Plotting')
          from matplotlib import pyplot as plt
          plot_quest(gdlambda, gdflux, continuum+specfit, ct_coeff, initdat, lines=[12.8], linespec=specfit)
        covar = lmout.covar
        dof = lmout.nfree
        rchisq = lmout.redchi
        errmsg = lmout.message
        status = lmout.success

        # error messages corresponding to LMFIT,plt
        # documentation was not very helpful with the error messages...
        if status == False :
            raise Exception('LMFIT: '+errmsg)
        # #     outstr = 0
        # #     goto,finish
        # if status == 5:
        #     print('LMFIT: Max. iterations reached.')

        # Errors from covariance matrix and from fit residual.
        resid = gdflux-continuum-specfit
        perror = dict()
        for p in lmout.params:
            perror[p] = lmout.params[p].stderr
        perror_resid = perror
        # sigrange = 20.
        # for line in lines_arr:
        #     iline = np.array([ip for ip, item in enumerate(parinit)
        #                       if item['line'] == line])
        #     ifluxpk = \
        #         np.intersect1d(iline,
        #                        np.array([ip for ip, item in enumerate(parinit)
        #                                  if item['parname'] == 'flux_peak']))
        #     ctfluxpk = len(ifluxpk)
        #     isigma = \
        #         np.intersect1d(iline,
        #                        np.array([ip for ip, item in enumerate(parinit)
        #                                  if item['parname'] == 'sigma']))
        #     iwave = \
        #         np.intersect1d(iline,
        #                        np.array([ip for ip, item in enumerate(parinit)
        #                                  if item['parname'] == 'wavelength']))
        #     for i in range(0, ctfluxpk):
        #         waverange = \
        #             sigrange * np.sqrt(np.power((param[isigma[i]] /
        #                                          c*param[iwave[i]]), 2.) +
        #                                np.power(param[2], 2.))
        #         wlo = np.searchsorted(gdlambda, param[iwave[i]] - waverange/2.)
        #         whi = np.searchsorted(gdlambda, param[iwave[i]] + waverange/2.)
        #         if whi == len(gdlambda)+1:
        #             whi = len(gdlambda)-1
        #         if param[ifluxpk[i]] > 0:
        #             perror_resid[ifluxpk[i]] = \
        #                 np.sqrt(np.mean(np.power(resid[wlo:whi], 2.)))

        outlistlines = listlines # this bit of logic prevents overwriting of listlines
        cont_dat = gdflux - specfit
    else:
        cont_dat = gdflux
        specfit = 0
        rchisq = 0
        dof = 1
        niter = 0
        status = 0
        outlistlines = 0
        parinit = 0
        param = 0
        perror = 0
        perror_resid = 0
        covar = 0
    # This sets the output reddening to a numerical 0 instead of NULL
    if ebv_star is None:
        ebv_star = 0.
        fit_time2 = time.time()
        if not quiet:
            print('{:s}{:0.1f}{:s}'.format('FITSPEC: Line fit took ',
                                           fit_time2-fit_time1, ' s.'))

# ;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;
# Output structure
# ;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;

# restore initial values
    flux = flux_out
    err = err_out

    # need to adjust the output values here...
    outstr = {'fitran': fitran,
              # Continuum fit parameters
              'ct_method': method,
              'ct_coeff': ct_coeff,
              'ct_ebv': ebv_star,
              'ct_indx': ct_indx,
              'zstar': zstar,
              'zstar_err': zstar_err,
              'ct_add_poly_weights': add_poly_weights,
              'ct_ppxf_sigma': ppxf_sigma,
              'ct_ppxf_sigma_err': ppxf_sigma_err,
              'ct_rchisq': ct_rchisq,
              # Spectrum in various forms
              'wave': gdlambda,
              'spec': gdflux,  # data
              'spec_err': gderr,
              'cont_dat': cont_dat,  # cont. data (all data - em. line fit)
              'cont_fit': continuum,  # cont. fit
              'cont_fit_pretweak': continuum_pretweak,  # cont. fit before tweaking
              'emlin_dat': gdflux_nocnt,  # em. line data (all data - cont. fit)
              'emlin_fit': specfit,  # em. line fit
              # gd_indx is applied, and then ct_indx
              'gd_indx': gd_indx,  # cuts on various criteria
              'fitran_indx': fitran_indx,  # cuts on various criteria
              #              'ct_indx': ct_indx,         # where emission is not masked, masking not in yet.
              # Line fit parameters
              'noemlinfit': noemlinfit,  # was emission line fit done?
              'noemlinmask': noemlinmask,  # were emission lines masked?
              'redchisq': rchisq,
              # 'niter': niter, (DOES NOT EXIST)
              'fitstatus': status,
              'linelist': outlistlines,
              'linelabel': linelabel,
              'maxncomp': initdat['maxncomp'],
              'parinfo': fit_params,
              'param': param,
              'perror': perror,
              'perror_resid': perror_resid,  # error from fit residual
              # 'covar': covar,
              'siglim': siglim_gas}

    # finish:
    return outstr
