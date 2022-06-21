import lmfit
import numpy as np
import pdb
import ppxf.ppxf_util as util
import sys

from astropy.constants import c
from ppxf.ppxf import ppxf
from q3dfit.qsohostfcn import qsohostfcn
from q3dfit.interptemp import interptemp
from scipy import interpolate


def fitqsohost(wave, flux, weight, template_wave, template_flux, index,
               zstar, quiet=True, blrpar=None, qsoxdr=None,
               qsoonly=False, index_log=None, err_log=None,
               flux_log=None, refit=None,
               add_poly_degree=30, siginit_stars=50.,
               fitran=None, fittol=None,
               qsoord=None, hostonly=False, hostord=None, blronly=False,
               blrterms=None, **kwargs):
    '''Function defined to fit the continuum

    Parameters
    -----
    wave: array
        wavelength

    flux: array
        Flux values to be fit

    weight: array
        Weights of each individual pixel to be fit

    template_wave: array
        Wavelength array of the stellar template used as model for
        stellar continuum

    template_flux: array
        Flux of the stellar template used ass model for stellar continuum

    index: array
        Pixels used in the fit

    zstar: float
        redshift of the stellar continuum




    returns
    -------
    continuum: array
        best fit continuum model

    ct_coeff: dictionary or lmfit best fit params structure
        best fit parameters

    zstar: float
        best fit stellar redshift
    '''

    if qsoxdr is None:
        sys.exit('Quasar template (qsoxdr) not specified in \
                 initialization file.')
    try:
        qsotemplate = np.load(qsoxdr, allow_pickle=True).item()
        qsowave = qsotemplate['wave']
        qsoflux_full = qsotemplate['flux']
    except:
        sys.exit('Cannot find quasar template (qsoxdr).')

    # qsoflux = interptemp(wave, qsowave, qsoflux_full)

    iqsoflux = np.where((qsowave >= fitran[0]) & (qsowave <= fitran[1]))
    qsoflux = qsoflux_full[iqsoflux]

    index = np.array(index)
    index = index.astype(dtype='int')

    # err = 1/weight**0.5
    iwave = wave[index]
    iflux = flux[index]
    iweight = weight[index]
    # ierr = err[index]

    ymod, params = \
        qsohostfcn(wave, params_fit=None, qsoxdr=qsoxdr, qsoonly=qsoonly,
                   qsoord=qsoord, hostonly=hostonly, hostord=hostord,
                   blronly=blronly, blrpar=blrpar, qsoflux=qsoflux, **kwargs)
    if quiet:
        lmverbose = 0  # verbosity for scipy.optimize.least_squares
    else:
        lmverbose = 2
    fit_kws = {'verbose': lmverbose}

    # Add additional parameter settings for scipy.optimize.least_squares
    if 'argslmfit' in kwargs:
        for key, val in kwargs['argslmfit'].items():
            fit_kws[key] = val

    result = ymod.fit(iflux, params, weights=np.sqrt(iweight),
                      qsotemplate=qsoflux[index],
                      wave=iwave, x=iwave, method='least_squares',
                      nan_policy='omit', fit_kws=fit_kws)

    if not quiet:
        lmfit.report_fit(result.params)

    # comps = result.eval_components(wave=wave, qso_model=qsoflux, x=wave)
    continuum = result.eval(wave=wave, qsotemplate=qsoflux, x=wave)
    # Test plot
    # import matplotlib.pyplot as plt
    # for i in comps.keys():
    #     plt.plot(wave, comps[i], label=i)
    # plt.plot(wave, continuum, label='best-fit')
    # plt.plot(wave, flux, label='flux')
    # plt.plot(wave, flux-continuum, label='resid')
    # plt.plot(wave, test_qsofcn, label='test')
    # plt.legend(loc='best')
    # plt.show()

    ct_coeff = result.params

    if refit == 'ppxf' and index_log is not None and \
        err_log is not None and flux_log is not None:

        # log rebin residual
        # lamRange1 = np.array([wave.min(), wave.max()])/(1+zstar)
        cont_log, lambda_log, velscale = util.log_rebin(fitran, continuum)

        resid_log = flux_log - cont_log

        # nan_indx = np.where(np.isnan(resid_log))[0]
        # if len(nan_indx) > 0:
        #    resid_log[nan_indx] = 0

        # Interpolate template to same grid as data
        temp_log = interptemp(lambda_log, np.log(template_wave.T[0]),
                              template_flux)

        # vel = c*np.log(1 + zstar)   # eq.(8) of Cappellari (2017)
        # t = clock()
        start = [0, siginit_stars]  # (km/s), starting guess for [V, sigma]
        pp = ppxf(temp_log, resid_log, err_log, velscale, start,
                  goodpixels=index_log,  quiet=quiet,  # plot=True, moments=2
                  degree=add_poly_degree)  # clean=False

        # resample additive polynomial to linear grid
        # poly_log = pp.apoly
        # pinterp = \
        #     interpolate.interp1d(residlambda_log, poly_log,
        #                          kind='cubic', fill_value="extrapolate")
        # poly = pinterp(np.log(wave))

        ct_coeff = {'qso_host': result.params,
                    'stel': pp.weights,
                    'poly': pp.polyweights,
                    'ppxf_sigma': pp.sol[1]}

        # From ppxf docs:
        # IMPORTANT: The precise relation between the output pPXF velocity
        # and redshift is Vel = c*np.log(1 + z).
        # See Section 2.3 of Cappellari (2017) for a detailed explanation.
        zstar += np.exp(pp.sol[0]/c.to('km/s').value)-1.

        # host can't be negative
        ineg = np.where(continuum < 0)
        continuum[ineg] = 0

        ppxfcontinuum_log = pp.bestfit
        cinterp = interpolate.interp1d(lambda_log, ppxfcontinuum_log,
                                       kind='cubic', fill_value="extrapolate")

        ppxfcont_resid = cinterp(np.log(wave))
        continuum += ppxfcont_resid

        return continuum, ct_coeff, zstar

    elif refit == 'questfit':

        from q3dfit.questfit import questfit
        resid = flux - continuum
        argscontfit_use = kwargs['args_questfit']
        cont_resid, ct_coeff, zstar = questfit(wave, resid, weight, b'0',
                                               b'0', index, zstar,
                                               quiet=quiet, **argscontfit_use)

        from q3dfit.plot_quest import plot_quest
        from matplotlib import pyplot as plt
        initdatdict = argscontfit_use.copy()
        initdatdict['label'] = 'miritest'
        initdatdict['plotMIR'] = True
        plot_quest(wave, resid, cont_resid, ct_coeff, initdatdict)
        plt.show()

        continuum += cont_resid
        ct_coeff['qso_host'] = result.params

    return continuum, ct_coeff, zstar
