import numpy as np
import pdb
import ppxf.ppxf_util as util
import sys

from astropy.constants import c
from ppxf.ppxf import ppxf
from q3dfit.common.qsohostfcn import qsohostfcn
from q3dfit.common.interptemp import interptemp
from scipy import interpolate


def fitqsohost(wave, flux, weight, template_wave, template_flux, index,
               zstar, quiet=True, blrpar=None, qsoxdr=None,
               qsoonly=None, index_log=None, refit=None,
               add_poly_degree=None, siginit_stars=None,
               polyspec_refit=None, fitran=None, fittol=None,
               qsoord=None, hostonly=None, hostord=None, blronly=None,
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
        Wavelength array of the stellar template used as model for stellar continuum

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
    except:
        sys.exit('Cannot find quasar template (qsoxdr).')

    qsowave = qsotemplate['wave']
    qsoflux_full = qsotemplate['flux']

    iqsoflux = np.where((qsowave >= fitran[0]) & (qsowave <= fitran[1]))
    qsoflux = qsoflux_full[iqsoflux]

    # Normalizing qsoflux template
    qsoflux = qsoflux/np.median(qsoflux)

    index = np.array(index)
    index = index.astype(dtype='int')

    #err = 1/weight**0.5
    iwave = wave[index]
    iflux = flux[index]
    iweight = weight[index]
    # ierr = err[index]

    if add_poly_degree is None:
        add_poly_degree = 30

    ymod, params = \
        qsohostfcn(wave, params_fit=None, quiet=quiet, blrpar=blrpar,
                   qsoxdr=qsoxdr, qsoonly=qsoonly, index_log=index_log,
                   refit=refit, add_poly_degree=add_poly_degree,
                   siginit_stars=siginit_stars, polyspec_refit=polyspec_refit,
                   fitran=fitran, fittol=fittol, qsoord=qsoord,
                   hostonly=hostonly, hostord=hostord,
                   blronly=blronly, qsoflux=qsoflux, **kwargs)

    result = ymod.fit(iflux, params, weights=iweight, qso_model=qsoflux[index],
                      wave=iwave, x=iwave, method='least_squares',
                      nan_policy='omit')

    # comps = result.eval_components(wave=wave, qso_model=qsoflux, x=wave)
    continuum = result.eval(wave=wave, qso_model=qsoflux, x=wave)

#    Test plot
#    import matplotlib.pyplot as plt
#    for i in comps.keys():
#        plt.plot(wave, comps[i], label=i)
#    plt.plot(wave, continuum, label='best-fit')
#    plt.plot(wave,flux,label='flux')
#    plt.plot(wave,flux-continuum,label='resid')
#    plt.plot(wave,test_qsofcn,label='test')
#    plt.legend(loc='best')
#    plt.show()

    ct_coeff = result.params

    # Fit residual with PPXF
    if refit:

        resid = flux - continuum

        # log rebin residual
        # lamRange1 = np.array([wave.min(), wave.max()])/(1+zstar)
        resid_log, residlambda_log, velscale = util.log_rebin(fitran, resid)
        residerrsq_log, _, _ = util.log_rebin(fitran, np.divide(1., weight))
        residerr_log = np.sqrt(residerrsq_log)

        # Interpolate template to same grid as data
        temp_log = interptemp(residlambda_log, np.log(template_wave.T[0]),
                              template_flux)

        # vel = c*np.log(1 + zstar)   # eq.(8) of Cappellari (2017)
        # t = clock()
        start = [0, siginit_stars]  # (km/s), starting guess for [V, sigma]
        pp = ppxf(temp_log, resid_log, residerr_log, velscale, start,
                  goodpixels=index_log,  quiet=quiet, # plot=True, moments=2
                  degree=add_poly_degree)  # clean=False

        # resample additive polynomial to linear grid
        poly_log = pp.apoly
        pinterp = interpolate.interp1d(residlambda_log, poly_log,
                                       kind='cubic', fill_value="extrapolate")
        poly = pinterp(np.log(wave))

        ct_coeff = {'qso_host': result.params,
                    'stel': pp.weights,
                    'poly': pp.polyweights,
                    'ppxf_sigma': pp.sol[1]}

        zstar += pp.sol[0]/c.to('km/s').value

        # host can't be negative
        ineg = np.where(continuum < 0)
        continuum[ineg] = 0

        continuum_log = pp.bestfit
        cinterp = interpolate.interp1d(residlambda_log, continuum_log,
                                       kind='cubic', fill_value="extrapolate")

        cont_resid = cinterp(np.log(wave))
        continuum += cont_resid

        return continuum, ct_coeff, zstar
