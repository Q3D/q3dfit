# -*- coding: utf-8 -*-
import copy
import numpy as np
import q3dfit.q3dout as q3dout
import time

from astropy.constants import c
from astropy.table import Table
from importlib import import_module
from ppxf.ppxf import ppxf
from ppxf.ppxf_util import log_rebin
from q3dfit.q3dmath import airtovac
from q3dfit.q3dutil import lmlabel
from q3dfit.interptemp import interptemp
from q3dfit.masklin import masklin
from scipy.interpolate import interp1d


def fitspec(wlambda, flux, err, dq, zstar, listlines, listlinesz, ncomp,
            specConv, q3di, linevary=None, maskwidths=None, peakinit=None,
            quiet=True, siginit_gas=None, siginit_stars=None, siglim_gas=None,
            tweakcntfit=None, logfile=None):

    """
    This function is the core routine to fit the continuum and emission
    lines of a spectrum.

    Parameters
    ----------

    wlambda : array
        Spectrum, observed-frame wavelengths.
    flux : array
        Spectrum, fluxes.
    err : array
        Spectrum, flux errors.
    zstar : float
        Initial guess for stellar redshift
    listlines :
        Emission line rest frame wavelengths
    listlinesz :
        Emission line observed frame wavelengths.
    logfile : str
    ncomp : dict
        Number of components fit to each line.
    q3di : object
        initialization parameters
    maskwidths : dict(lines,maxncomp), optional, default=None
        Widths, in km/s, of regions to mask from continuum fit. If not
        set, routine defaults to +/- 500 km/s. Can also be set in q3di.
        Routine prioritizes the keyword definition.
    peakinit : dict(lines,maxncomp), optional, default=None
        Initial guesses for peak emission-line flux densities. If not
        set, routine guesses from spectrum. Can also be set in q3di.
        Routine prioritizes the keyword definition.
    quiet : bool, optional, default=True
        Use to prevent detailed output to screen. Default is to print
        detailed output.
    siginit_gas : dict(lines, maxncomp), optional, default=None
        Initial guess for emission line widths for fitting.
    siginit_stars : float, optional, default=None
        Initial guess for stellar line widths for fitting.
    siglim_gas: array, optional, type=dblarr(2)
        Sigma limits for line fitting.
    tweakcntfit : array, optional, default=None
        Parameters for tweaking continuum fit with localized polynomials. For
        each of nregions regions, array contains lower limit, upper limit, and
        polynomial degree.

    Returns
    -------
    A dictionary that contains the fit and much else ...

    """

    usetype='float64'

    flux = copy.deepcopy(flux)
    err = copy.deepcopy(err)
    flux_out = copy.deepcopy(flux)
    err_out = copy.deepcopy(err)

    # default sigma for initial guess for emission line widths
    siginit_gas_def = np.float64(100.)

    if logfile is None:
        from sys import stdout
        logfile = stdout

    if q3di.startempfile is not None and \
        zstar is not None and \
        q3di.fcncontfit != 'questfit':

        # Get stellar templates
        startempfile = q3di.startempfile
        if isinstance(startempfile, bytes):
            startempfile = startempfile.decode('utf-8')
        sav_data = np.load(startempfile, allow_pickle=True).item()
        template = sav_data
        # Redshift stellar templates
        templatelambdaz = np.copy(template['lambda'])
        if not q3di.keepstarz:
            templatelambdaz *= 1. + zstar
        # Need option for when template is in vac and data in air ...
        if q3di.vacuum and not q3di.startempvac:
            templatelambdaz = airtovac(templatelambdaz)
        # if 'waveunit' in q3di:
        #     templatelambdaz *= q3di['waveunit']
        if q3di.fcnconvtemp is not None:
            impModule = import_module('q3dfit.'+q3di.fcnconvtemp)
            fcnconvtemp = getattr(impModule, q3di.fcnconvtemp)
            newtemplate = fcnconvtemp(templatelambdaz, template,
                                      **q3di.argsconvtemp)
    else:
        templatelambdaz = wlambda

# ;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;
# # Pick out regions to fit
# ;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;

    if q3di.fitrange is not None:
        fitran_tmp = list(map(np.float64, q3di.fitrange))
    else:
        fitran_tmp = [wlambda[0], wlambda[len(wlambda)-1]]
    # indices locating good data and data within fit range
    # these index the full data range.
    gd_indx_1 = set(np.where(flux != 0.)[0])
    gd_indx_2 = set(np.where(err > 0.)[0])
    gd_indx_3 = set(np.where(np.isfinite(flux))[0])
    gd_indx_4 = set(np.where(np.isfinite(err))[0])
    gd_indx_5 = set(np.where(dq == 0)[0])
    gd_indx_6 = set(np.where(wlambda >= min(templatelambdaz))[0])
    gd_indx_7 = set(np.where(wlambda <= max(templatelambdaz))[0])
    gd_indx_8 = set(np.where(wlambda >= fitran_tmp[0])[0])
    gd_indx_9 = set(np.where(wlambda <= fitran_tmp[1])[0])
    gd_indx_full = gd_indx_1.intersection(gd_indx_2, gd_indx_3, gd_indx_4,
                                          gd_indx_5, gd_indx_6, gd_indx_7,
                                          gd_indx_8, gd_indx_9)
    gd_indx_full = list(gd_indx_full)

    # Check that gd_indx_full is not empty, and has more than one point
    if len(gd_indx_full) > 1:

        # limit actual fit range to good data
        fitran = [np.min(wlambda[gd_indx_full]).astype(usetype),
                  np.max(wlambda[gd_indx_full]).astype(usetype)]

        # indices locating data within actual fit range
        fitran_indx1 = np.where(wlambda >= fitran[0])[0]
        fitran_indx2 = np.where(wlambda <= fitran[1])[0]
        fitran_indx = np.intersect1d(fitran_indx1, fitran_indx2)
        # indices locating good regions within wlambda[fitran_indx]
        gd_indx_full_rezero = gd_indx_full - fitran_indx[0]
        max_gd_indx_full_rezero = max(fitran_indx) - fitran_indx[0]
        igdfz1 = np.where(gd_indx_full_rezero >= 0)[0]
        igdfz2 = np.where(gd_indx_full_rezero <= max_gd_indx_full_rezero)[0]
        i_gd_indx_full_rezero = np.intersect1d(igdfz1, igdfz2)
        # Final index for addressing ALL "good" pixels
        # these address only the fitted data range; i.e., they address gdflux,
        # etc.
        gd_indx = gd_indx_full_rezero[i_gd_indx_full_rezero]

        # Limit data to fit range
        gdflux = flux[fitran_indx]
        gdlambda = wlambda[fitran_indx]
        gderr = err[fitran_indx]
        gddq = dq[fitran_indx]
        gdinvvar = 1./np.power(gderr, 2.)  # inverse variance

        # Log rebin galaxy spectrum for PPXF
        gdflux_log, gdlambda_log, velscale = log_rebin(fitran, gdflux)
        gderrsq_log, _, _ = log_rebin(fitran, np.power(gderr, 2.))
        gderr_log = np.sqrt(gderrsq_log)
        # gdinvvar_log = 1./np.power(gderr_log, 2.)

        # Find where flux is <= 0 or error is <= 0 or infinite or NaN or dq != 0
        # these index the fitted data range
        zerinf_indx_1 = np.where(gdflux == 0.)[0]
        zerinf_indx_2 = np.where(gderr <= 0.)[0]
        zerinf_indx_3 = np.where(np.isinf(gdflux))[0]
        zerinf_indx_4 = np.where(np.isinf(gderr))[0]
        zerinf_indx_5 = np.where(gddq != 0)[0]
        zerinf_indx = np.unique(np.hstack([zerinf_indx_1, zerinf_indx_2,
                                           zerinf_indx_3, zerinf_indx_4,
                                           zerinf_indx_5]))

        zerinf_indx_1 = np.where(gdflux_log == 0.)[0]
        zerinf_indx_2 = np.where(gderr_log <= 0.)[0]
        zerinf_indx_3 = np.where(np.isinf(gdflux_log))[0]
        zerinf_indx_4 = np.where(np.isinf(gderr_log))[0]
        # to-do: log rebin dq and apply here?
        zerinf_indx_log = np.unique(np.hstack([zerinf_indx_1, zerinf_indx_2,
                                               zerinf_indx_3, zerinf_indx_4]))
        # good indices for log arrays
        ctfitran = len(gdflux_log)
        gd_indx_log = np.arange(ctfitran)
        ctzerinf_log = len(zerinf_indx_log)
        if ctzerinf_log > 0:
            gd_indx_log = np.setdiff1d(gd_indx_log, zerinf_indx_log)

        # Set bad points to nan so lmfit will ignore
        ctzerinf = len(zerinf_indx)
        if ctzerinf > 0:
            gdflux[zerinf_indx] = np.nan
            gderr[zerinf_indx] = np.nan
            gdinvvar[zerinf_indx] = np.nan
            if not quiet:
                print('{:s}{:0f}{:s}'.
                      format('FITLOOP: Setting ', int(ctzerinf),
                             ' points from zero/inf flux or ' +
                             'neg/zero/inf error to np.nan'))

        if ctzerinf_log > 0:
            # can't just use np.nan because ppxf will choke on it
            gdflux_log[zerinf_indx_log] = 0.
            gderr_log[zerinf_indx_log] = 100.*max(gderr_log)
            # gdinvvar_log[zerinf_indx_log] = np.nan


# ;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;
# Initialize fit
# ;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;

        q3do = q3dout.q3dout(gdlambda, gdflux, gderr, fitrange=fitran,
                             gd_indx=gd_indx, fitran_indx=fitran_indx)

    else:

        gdflux = np.array((0.))
        q3do = q3dout.q3dout(0., 0., 0.,  nogood=True)

    # timer
    fit_time0 = time.time()

# ;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;
# Fit continuum
# ;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;

    if q3di.docontfit and not q3do.nogood:

        q3do.init_contfit(zstar=zstar)

        # Some defaults. These only apply in case of fitting with stellar model
        # + additive polynomial.
        # stel_mod = 0.
        # poly_mod = 0.

        # Mask emission lines
        # Note that maskwidths is now an astropy Table
        # Column names are line labels, rows are components
        if not q3di.nolinemask:
            if maskwidths is None:
                if q3di.maskwidths is not None:
                    maskwidths = q3di.maskwidths
                else:
                    maskwidths = \
                        Table(np.full([q3di.maxncomp, listlines['name'].size],
                                      q3di.maskwidths_def, dtype=usetype),
                              names=listlines['name'])
            # This loop overwrites nans in the case that ncomp gets lowered
            # by checkcomp; these nans cause masklin to choke
            for line in listlines['name']:
                for comp in range(ncomp[line], q3di.maxncomp):
                    maskwidths[line][comp] = 0.
                    listlinesz[line][comp] = 0.
            q3do.ct_indx = masklin(gdlambda, listlinesz, maskwidths,
                                    nomaskran=q3di.nomaskran)
            # Mask emission lines in log space
            ct_indx_log = masklin(np.exp(gdlambda_log), listlinesz,
                                  maskwidths, nomaskran=q3di.nomaskran)
        else:
            q3do.ct_indx = np.arange(len(gdlambda))
            ct_indx_log = np.arange(len(gdlambda_log))

        q3do.ct_indx = np.intersect1d(q3do.ct_indx, gd_indx)
        ct_indx_log = np.intersect1d(ct_indx_log, gd_indx_log)

    # ;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;
    # # Option 1: Input function that is not ppxf
    # ;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;

        if q3di.fcncontfit != 'ppxf':

            # get fitting function
            module = import_module('q3dfit.contfit')
            fcncontfit = getattr(module, q3di.fcncontfit)

            if q3di.fcncontfit == 'questfit' or q3di.startempfile is None:
                templatelambdaz_tmp = None
                templateflux_tmp = None
            else:
                templatelambdaz_tmp = templatelambdaz
                templateflux_tmp = template['flux']

            if q3di.argscontfit is not None:
                argscontfit = q3di.argscontfit
            else:
                argscontfit = dict()
            if q3di.fcncontfit == 'fitqsohost':
                argscontfit['fitran'] = fitran
            if 'refit' in argscontfit.keys():
                if argscontfit['refit'] == 'ppxf':
                    argscontfit['index_log'] = ct_indx_log
                    argscontfit['flux_log'] = gdflux_log
                    argscontfit['err_log'] = gderr_log
                    argscontfit['siginit_stars'] = siginit_stars

            if q3di.fcncontfit == 'linfit_plus_FeII':
                argscontfit['specConv'] = specConv

            if zstar is None:
                zstarin = np.nan
            else:
                zstarin = copy.copy(zstar)
            q3do.cont_fit, q3do.ct_coeff, zstarout = \
                fcncontfit(gdlambda.astype(usetype),
                           gdflux.astype(usetype),
                           gdinvvar.astype(usetype),
                           templatelambdaz_tmp,
                           templateflux_tmp, q3do.ct_indx,
                           zstarin.astype(usetype),
                           quiet=quiet, **argscontfit)
            if zstarout is not None:
                q3do.zstar = copy.copy(zstarout)

            if 'refit' in argscontfit.keys():
                if argscontfit['refit'] == 'ppxf':
                    q3do.ct_ppxf_sigma = q3do.ct_coeff['ppxf_sigma']

        # ;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;
        # # Option 2: PPXF
        # ;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;

        elif q3di.startempfile is not None:

            # Interpolate template to same grid as data
            temp_log = interptemp(gdlambda_log, np.log(templatelambdaz),
                                  template['flux'])

            # Check polynomial degree
            add_poly_degree = 4
            if q3di.argscontfit is not None:
                if 'add_poly_degree' in q3di.argscontfit:
                    add_poly_degree = q3di.argscontfit['add_poly_degree']

            # run ppxf
            pp = ppxf(temp_log, gdflux_log, gderr_log, velscale[0],
                      [0, siginit_stars], goodpixels=ct_indx_log,
                      degree=add_poly_degree, quiet=quiet,
                      reddening=q3di.ebv_star)
            # poly_mod = pp.apoly
            continuum_log = pp.bestfit
            q3do.ct_add_poly_weights = pp.polyweights
            q3do.ct_coeff = pp.weights
            q3do.ct_ebv = pp.reddening
            sol = pp.sol
            solerr = pp.error

            # Resample the best fit into linear space
            cinterp = interp1d(gdlambda_log, continuum_log,
                               kind='cubic', fill_value="extrapolate")
            q3do.cont_fit = cinterp(np.log(gdlambda))

            # Adjust stellar redshift based on fit
            # From ppxf docs:
            # IMPORTANT: The precise relation between the output pPXF velocity
            # and redshift is Vel = c*np.log(1 + z).
            # See Section 2.3 of Cappellari (2017) for a detailed explanation.
            q3do.zstar += np.exp(sol[0]/c.to('km/s').value)-1.
            q3do.ct_ppxf_sigma = sol[1]

            # From PPXF docs:
            # These errors are meaningless unless Chi^2/DOF~1.
            # However if one *assumes* that the fit is good ...
            q3do.ct_rchisq = pp.chi2
            solerr *= np.sqrt(pp.chi2)
            q3do.zstar_err = (np.exp(solerr[0]/c.to('km/s').value))-1.
            q3do.ct_ppxf_sigma_err = solerr[1]

# ;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;
# # Option to tweak cont. fit with local polynomial fits
# ;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;
        # if q3di.tweakcntfit is not None:
        #     q3do.cont_fit_pretweak = q3do.cont_fit.copy()
        # # Arrays holding emission-line-masked data
        #     ct_lambda = gdlambda[q3do.ct_indx]
        #     ct_flux = gdflux[q3do.ct_indx]
        #     ct_err = gderr[q3do.ct_indx]
        #     ct_cont = q3do.cont_fit[q3do.ct_indx]
        #     for i in range(len(tweakcntfit[0,:])):
        #     # Indices into full data
        #         tmp_ind1 = np.where(gdlambda >= tweakcntfit[i,0])[0]
        #         tmp_ind2 = np.where(gdlambda <= tweakcntfit[i,1])[0]
        #         tmp_ind = np.intersect1d(tmp_ind1,tmp_ind2)
        #         ct_ind = len(tmp_ind)
        #     # Indices into masked data
        #         tmp_ctind1 = np.where(ct_lambda >= tweakcntfit[i,0])[0]
        #         tmp_ctind2 = np.where(ct_lambda <= tweakcntfit[i,1])[0]
        #         tmp_ctind = np.intersect1d(tmp_ctind1,tmp_ctind2)
        #         ct_ctind = len(tmp_ctind)

        #         if ct_ind > 0 and ct_ctind > 0:
        #             parinfo =  list(np.repeat({'value':0.},tweakcntfit[2,i]+1))
        #             # parinfo = replicate({value:0d},tweakcntfit[2,i]+1)
        #             pass # this is just a placeholder for now
        #             # tmp_pars = mpfitfun('poly',ct_lambda[tmp_ctind],$
        #             #                     ct_flux[tmp_ctind] - ct_cont[tmp_ctind],$
        #             #                     ct_err[tmp_ctind],parinfo=parinfo,/quiet)
        #             # continuum[tmp_ind] += poly(gdlambda[tmp_ind],tmp_pars)
        # else:
        #     continuum_pretweak = continuum.copy()

        if q3di.dividecont:
            continuum = gdflux.copy() / q3do.cont_fit - 1
            gdinvvar_nocnt = gdinvvar.copy() * np.power(q3do.cont_fit, 2.)
            # gderr_nocnt = gderr / continuum
            q3do.ct_method = 'CONTINUUM DIVIDED'
        else:
            continuum = gdflux.copy() - q3do.cont_fit
            gdinvvar_nocnt = gdinvvar.copy()

        fit_time1 = time.time()
        if not quiet:
            print('{:s}{:0.1f}{:s}'.format('FITSPEC: Continuum fit took ',
                                           fit_time1-fit_time0, ' s.'))

    else:

        continuum = gdflux.copy()


# ;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;
# Fit emission lines
# ;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;

    if q3di.dolinefit and not q3do.nogood:

        q3do.init_linefit(listlines, q3di.lines, q3di.maxncomp,
                          line_dat=continuum.astype(usetype))

        # Check that # components being fit to at least one line is > 0
        nonzerocomp = np.where(np.array(list(ncomp.values())) != 0)[0]

        # Make sure line within fitrange
        # To deal with case of truncated data where continuum can be fit but
        # line not within good data range
        line_in_good_range = False
        for line in q3di.lines:
            if line_in_good_range:
                break
            for icomp in range(0, ncomp[line]):
                if listlinesz[line][icomp] >= min(gdlambda) and \
                    listlinesz[line][icomp] <= max(gdlambda):
                    line_in_good_range = True
                    break

        if len(nonzerocomp) > 0 and line_in_good_range:

            # Initial guesses for emission line peak fluxes (above continuum)
            if peakinit is None:
                if q3di.peakinit is not None and isinstance(q3di.peakinit, dict):
                    peakinit = q3di.peakinit
                else:
                    # initialize peakinit
                    peakinit = {line: np.zeros(q3di.maxncomp) for line in q3di.lines}
                    # apply some light filtering
                    # https://stackoverflow.com/questions/20618804/how-to-smooth-a-curve-in-the-right-way/20642478
                    # https://docs.scipy.org/doc/scipy/reference/generated/scipy.signal.savgol_filter.html
                    from scipy.signal import savgol_filter
                    # from scipy.ndimage import gaussian_filter
                    line_dat_sm = savgol_filter(q3do.line_dat, 11, 3)
                    # gaussian_filter(q3do.line_dat,1)
                    # fline = interp1d(gdlambda, line_dat_sm, kind='linear')
                    for line in q3di.lines:
                        # Check that line wavelength is in data range
                        # Use first component as a proxy for all components
                        if listlinesz[line][0] >= min(gdlambda) and \
                            listlinesz[line][0] <= max(gdlambda):
                            # peakinit[line] = fline(listlinesz[line][0:ncomp[line]])
                            # look for peak within dz = +/-0.001
                            for icomp in range(0, ncomp[line]):
                                # https://stackoverflow.com/questions/12141150/from-list-of-integers-get-number-closest-to-a-given-value
                                lamwin = \
                                    [min(gdlambda, key=lambda x:
                                        abs(x - listlinesz[line][icomp]*0.999)),
                                     min(gdlambda, key=lambda x:
                                         abs(x - listlinesz[line][icomp]*1.001))]
                                ilamwin = [np.where(gdlambda == lamwin[0])[0][0],
                                           np.where(gdlambda == lamwin[1])[0][0]]
                                # e.g. in the MIR, the wavelength range spanned by
                                # (listlinesz[line][icomp]*0.999,
                                # listlinesz[line][icomp]*1.001) can be smaller
                                # than one spectral element
                                if ilamwin[1]-ilamwin[0] < 10:
                                    dlam = \
                                        gdlambda[int(0.5*(ilamwin[0] +
                                                          ilamwin[1]))]\
                                        - gdlambda[int(0.5*(ilamwin[0] +
                                                            ilamwin[1]))-1]
                                    lamwin = \
                                        [min(gdlambda, key=lambda x:
                                             abs(x - (listlinesz[line][icomp] -
                                                      5.*dlam))),
                                         min(gdlambda, key=lambda x:
                                             abs(x - (listlinesz[line][icomp] +
                                                      5.*dlam)))]
                                    ilamwin = \
                                        [np.where(gdlambda == lamwin[0])[0][0],
                                         np.where(gdlambda == lamwin[1])[0][0]]

                                peakinit[line][icomp] = \
                                    np.nanmax(line_dat_sm
                                              [ilamwin[0]:ilamwin[1]])

                                # If the smoothed version gives all nans, try
                                # unsmoothed
                                if np.isnan(peakinit[line][icomp]) and \
                                    q3do.line_dat is not None:
                                    peakinit[line][icomp] = \
                                    np.nanmax(q3do.line_dat[ilamwin[0]:ilamwin[1]])
                                # if it's still all nans, just set to 0.
                                if np.isnan(peakinit[line][icomp]):
                                    peakinit[line][icomp] = 0.
                            # If initial guess is negative, set to 0 to prevent
                            # fitter from choking (since we limit peak to be >= 0)
                            peakinit[line] = \
                                np.where(peakinit[line] < 0., 0., peakinit[line])
                        # re-cast as float32
                        peakinit[line] = peakinit[line].astype(usetype)

            # Initial guesses for emission line widths
            if siginit_gas is None:
                siginit_gas = \
                    {k: np.zeros(q3di.maxncomp, dtype=usetype) + 
                        np.float64(siginit_gas_def) for k in q3di.lines}                        

            # Fill out parameter structure with initial guesses and constraints
            impModule = import_module('q3dfit.' + q3di.fcnlineinit)
            run_fcnlineinit = getattr(impModule, q3di.fcnlineinit)
            if q3di.argslineinit is not None:
                argslineinit = q3di.argslineinit
            else:
                argslineinit = dict()
            if q3di.fcnlineinit == 'lineinit':
                argslineinit['waves'] = gdlambda
            if siglim_gas is not None:
                argslineinit['siglim'] = siglim_gas
            if linevary is not None:
                argslineinit['linevary'] = linevary
            emlmod, q3do.parinit, q3do.siglim = \
                run_fcnlineinit(listlines, listlinesz, q3di.linetie, peakinit,
                                siginit_gas, q3di.maxncomp, ncomp, specConv,
                                **argslineinit)


            # Actual fit

            # Collect keywords to pass to the minimizer routine via lmfit
            fit_kws = {}

            # Maximum # evals cannot be specified as a keyword to the minimzer,
            # as it's a parameter of the fit method. Default for 'least_squares'
            # with 'trf' method (which is what lmfit assumes)
            # is 100*npar, but lmfit changes this to 2000*(npar+1)
            # https://github.com/lmfit/lmfit-py/blob/b930ddef320d93f984181db19fec8e9c9a41be8f/lmfit/minimizer.py#L1526
            # We'll change default to 200*(npar+1).
            # Note that lmfit method 'least_squares’ with default least_squares
            # method 'lm' counts function calls in
            # Jacobian estimation if numerical Jacobian is used (again the default)
            max_nfev = 200*(len(q3do.parinit)+1)
            iter_cb = None
            method = 'least_squares'

            # Add more using 'argslinefit' dict in init file
            if q3di.argslinefit is not None:
                for key, val in q3di.argslinefit.items():
                    # max_nfev goes in as parameter to fit method instead
                    if key == 'max_nfev':
                        max_nfev = val
                    # iter_cb goes in as parameter to fit method instead
                    elif key == 'iter_cb':
                        iter_cb = globals()[val]
                    elif key == 'method':
                        method = val
                    else:
                        fit_kws[key] = val

            if method == 'least_squares':
                # verbosity for scipy.optimize.least_squares
                if quiet:
                    lmverbose = 0
                else:
                    lmverbose = 2
                fit_kws['verbose'] = lmverbose

            if method == 'leastsq':
                # to get mesg output
                fit_kws['full_output'] = True
                # increase number of max iterations; this is the default for this algorithm
                # https://github.com/lmfit/lmfit-py/blob/7710da6d7e878ffee0dc90a85286f1ec619fc20f/lmfit/minimizer.py#L1624
                max_nfev = 2000*(len(q3do.parinit)+1)


            lmout = emlmod.fit(q3do.line_dat, q3do.parinit, x=gdlambda,
                               method=method, weights=np.sqrt(gdinvvar_nocnt),
                               nan_policy='omit', max_nfev=max_nfev,
                               fit_kws=fit_kws, iter_cb=iter_cb)

            q3do.line_fit = emlmod.eval(lmout.params, x=gdlambda)
            if not quiet:
                print(lmout.fit_report(show_correl=False))

            q3do.param = lmout.best_values
            q3do.covar = lmout.covar
            q3do.dof = lmout.nfree
            q3do.redchisq = lmout.redchi
            q3do.nfev = lmout.nfev

            '''
            error messages corresponding to LMFIT, plt
            documentation was not very helpful with the error messages...
            This can happen if, e.g., max_nfev is reached. Status message
            is in this case not set, so we'll set it by hand.
            The reason for algorithm termination (in least_squaares):
            -1 : improper input parameters status returned from MINPACK.
            0 : the maximum number of function evaluations is exceeded.
            1 : gtol termination condition is satisfied.
            2 : ftol termination condition is satisfied.
            3 : xtol termination condition is satisfied.
            4 : Both ftol and xtol termination conditions are satisfied.
            '''
            # q3do.fitstatus = 1 # default for good fit
            if method == 'least_squares':
                # https://lmfit.github.io/lmfit-py/model.html#lmfit.model.success
                if not lmout.success:
                    print('lmfit: '+lmout.message, file=logfile)
                    if not quiet:
                        print('lmfit: '+lmout.message)
                if hasattr(lmout, 'status'):
                    q3do.fitstatus = lmout.status
                elif lmout.nfev >= max_nfev:
                    q3do.fitstatus = 0

            if method == 'leastsq':
            # Return values from scipy.optimize.leastsq for fit status
            # https://github.com/scipy/scipy/blob/44e4ebaac992fde33f04638b99629d23973cb9b2/scipy/optimize/_minpack_py.py#L446
            # success = 1-4, failure=4-8
            # Possible lmfit error messages here:
            # Presently, fit is aborting with messaage "Fit aborted" if max_nfev is reached
            # setting ier=-1.
            # I don't understand why this is happening, as I can't find the code that
            # actually sets result.aborted to True in the lmfit code.
            # See here for where ier is set in this case:
            # https://github.com/lmfit/lmfit-py/blob/7710da6d7e878ffee0dc90a85286f1ec619fc20f/lmfit/minimizer.py#L1653
                if not lmout.success:
                    print('lmfit: '+lmout.message, file=logfile)
                    if not quiet:
                        print('lmfit: '+lmout.message)
                print('lmfit: '+lmout.lmdif_message, file=logfile)
                if not quiet:
                    print('lmfit: '+lmout.lmdif_message)
                q3do.fitstatus = lmout.ier

            # Errors from covariance matrix
            q3do.perror = dict()
            for p in lmout.params:
                q3do.perror[p] = lmout.params[p].stderr
            # Get flux peak errors from error spectrum and std dev of residual
            q3do.perror_errspec = copy.deepcopy(q3do.perror)
            q3do.perror_resid = copy.deepcopy(q3do.perror)
            for line in listlines['name']:
                for i in range(0, ncomp[line]):
                    lmline = lmlabel(line)
                    fluxlab = f'{lmline.lmlabel}_{i}_flx'
                    if q3do.param[fluxlab] > 0:
                        peakwave = q3do.param[f'{lmline.lmlabel}_{i}_cwv']
                        ipeakwave = (np.abs(gdlambda - peakwave)).argmin()
                        # from error spec
                        ipklo = ipeakwave - round(q3di.perror_errspecwin/2)
                        ipkhi = ipeakwave + round(q3di.perror_errspecwin/2)
                        if ipklo < 0:
                            ipklo = 0
                        if ipkhi > len(gdlambda)-1:
                            ipkhi = len(gdlambda)-1
                        q3do.perror_errspec[fluxlab] = \
                            np.median(gderr[ipklo:ipkhi+1])
                        # from residual
                        ipklo = ipeakwave - round(q3di.perror_residwin/2)
                        ipkhi = ipeakwave + round(q3di.perror_residwin/2)
                        if ipklo < 0:
                            ipklo = 0
                        if ipkhi > len(gdlambda)-1:
                            ipkhi = len(gdlambda)-1
                        q3do.perror_resid[fluxlab] = \
                            np.std((q3do.line_dat - q3do.line_fit)
                                   [ipklo:ipkhi+1])
                        # Deal with flux pegging at boundary and no error
                        # from lmfit. Check for both Nones and nans:
                        if q3do.perror[fluxlab] is None:
                            q3do.perror[fluxlab] = q3do.perror_errspec[fluxlab]
                        elif np.isnan(q3do.perror[fluxlab]):
                            q3do.perror[fluxlab] = q3do.perror_errspec[fluxlab]
                        if q3di.perror_useresid and \
                            q3do.perror_resid[fluxlab] > q3do.perror[fluxlab]:
                            q3do.perror[fluxlab] = q3do.perror_resid[fluxlab]

            # Flux peak errors from fit residual.
            # resid = gdflux - continuum - specfit
            # q3do.perror_resid = copy.deepcopy(q3do.perror)
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

            q3do.cont_dat = gdflux.copy() - q3do.line_fit

            fit_time2 = time.time()
            if not quiet:
                print('{:s}{:0.1f}{:s}'.format('FITSPEC: Line fit took ',
                                               fit_time2-fit_time1, ' s.'))

        else:

            q3do.dolinefit = False
            q3do.cont_dat = gdflux.copy()

    else:
        #q3do.fitstatus = 1
        q3do.cont_dat = gdflux.copy()

# ;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;
#  Finish
# ;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;

    return(q3do)


# For diagnosing problems, print value of each parameter every iteration
# To use, set 'iter_cb': 'per_iteration' as a keyword/value pair in 'argslinefit'
# https://lmfit.github.io/lmfit-py/examples/documentation/model_with_iter_callback.html
def per_iteration(pars, iteration, resid, *args, **kws):
    print(" ITER ", iteration, [f"{p.name} = {p.value:.5f}"
                                for p in pars.values()])
