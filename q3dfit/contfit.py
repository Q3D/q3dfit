import copy
import importlib.resources as pkg_resources
import lmfit
import numpy as np
import ppxf.ppxf_util as util
import sys, os

from astropy import units as u
from astropy.constants import c
from astropy.modeling import models, fitting
# from lmfit.models import ExpressionModel
from matplotlib import pyplot as plt
from ppxf.ppxf import ppxf
from q3dfit import interp_temp_quest
from q3dfit import questfitfcn
from q3dfit.questfitfcn import readcf
# from q3dfit import writeout_quest
from q3dfit.data import questfit_templates
from q3dfit.data.questfit_templates import silicatemodels
from q3dfit.qsohostfcn import qsohostfcn
from q3dfit.interptemp import interptemp
from scipy import constants, interpolate

from lmfit.models import LinearModel

def fitpoly(lam, flux, weight, unused1, unused2, index, unused3,
            fitord=3, quiet=False, refit=None, **kwargs):
    '''
    This function fits the continuum as a polynomial.

    Parameters
    ----------
    lam: numpy.ndarray
        wavelength
    flux: numpy.ndarray
        Flux values to be fit
    weight: numpy.ndarray
        Weights of each individual pixel to be fit
    unused1
        Not used in this function.
    unused2
        Not used in this function.
    index: array
        Pixels used in the fit.
    unused3
        Not used in this function.
    fitord: int, optional
        Order of the polynomial fit.
    quiet: bool, optional
        Not used in this function.
    refit: NoneType or dictionary, optional
        Refit with a different polynomial order. Default is None. Dictionary
        should have keys 'ord' and 'ran' for the order and range of the
        refit, respectively.
    kwargs: dict, optional
        Additional keyword arguments.

    Returns
    -------
    continuum: array
        best fit continuum model
    ct_coeff: dictionary or lmfit best fit params structure
        best fit parameters

    '''
    ilam = np.array(lam[index])
    iflux = np.array(flux[index])
    #the fitter I used puts weights in 1/sigma so I took the square root to make the data correct
    w = np.array(weight[index])
    iweight = np.array(np.sqrt(w))

    ilam = ilam.reshape(ilam.size)
    iflux = iflux.reshape(ilam.size)
    iweight = iweight.reshape(ilam.size)

    if fitord==0:
        deg1=len(ilam)-1
        deg2=fitord
    else:
        deg1=fitord
        deg2=fitord
    
    #making astropy fitter
    fitter = fitting.LevMarLSQFitter()
    #making polynomial model
    polymod1= models.Polynomial1D(deg1)
    polymod2= models.Polynomial1D(deg2)

    #creating fluxfit
    fluxfit = fitter(polymod1, ilam, iflux, weights=iweight)
    fluxfitparam=fluxfit.parameters

    #flip for numpy.poly1d
    ct_coeff = np.flip(fluxfitparam)
    # this is a class:
    # https://numpy.org/doc/stable/reference/generated/numpy.poly1d.html
    ct_poly = np.poly1d(ct_coeff, variable='lambda')
    continuum = ct_poly(lam)
    icontinuum = continuum[index]

    if refit is not None:
        for i in range (0, np.size(refit['ord']) - 1):
            tmp_ind=np.where(lam >= refit['ran'][0,i] and
                             lam <= refit['ran'][1,i])
            tmp_iind=np.where(ilam >= refit['ran'][0,i] and
                              ilam <= refit['ran'][1,i])
            #  parinfo=np.full(refit['ord'][i]+1, {'value':0.0})

            #degree of polynomial fit defaults to len(x-variable)-1
            if deg2==0:
                deg2=len(ilam[tmp_iind])-1

            #creating tmp_pars
            tmp_pars=fitter(polymod2, ilam[tmp_iind],
                            (iflux[tmp_iind]-icontinuum[tmp_iind]),
                            z=None, weights=iweight[tmp_iind])
            tmp_parsptmp=tmp_pars.parameters
            tmp_parsparam=np.flip(tmp_parsptmp)

            # refitted continuum object
            ct_poly = np.poly1d(tmp_parsparam, variable='lambda')
            # refitted continuum over wavelength range for refitting, 
            # indexed to full wavelength array
            cont_refit = ct_poly(lam[tmp_ind])
            # now add in the refitted continuum to the full continuum
            continuum += cont_refit

    return continuum, ct_coeff, None


def Fe_flux_balmer(Fe_FWHM, zstar, specConv):
    '''
    This function loads the optical FeII template, redshifts it,
    convolves it with the intrinsic broadening, and convolves it 
    with the instrumental resolution.

    Parameters
    ----------
    Fe_FWHM: float
        Defines the smoothing of the FeII template in km/s. FWHM of the
        Gaussian smoothing that will be applied. Needs to be >900 km/s
        (FWHM of template).
    zstar: float
        Redshift of the stellar continuum.
    specConv: instance of the spectConvol class

    Returns
    -------
    F_Fe_opt_conv: array
        Convolved FeII template.
    wave_fe: array
        Wavelength array of the FeII template.

    '''
    temp_fe = '../q3dfit/data/questfit_templates/fe_optical.txt'
    data1 = np.loadtxt(temp_fe)
    wave_fe = 10**data1[:,0] * (1. + zstar) /1e4
    F_fe = data1[:,1]/np.max(data1[:,1])
    # sig_conv in km/s. 900 km/s is the FWHM of the Vestergaard & 
    # Wilkes 2001 template.
    # sig_tot_BLR = (sig_temp^2+sig_add_broadening^2)^0.5
    sig_conv = np.sqrt(Fe_FWHM ** 2 - 900.0 ** 2) / 2. / \
        np.sqrt(2. * np.log(2.))
    # 106.3 km/s is the dispersion for the BG92 FeII template 
    # used by Vestergaard & Wilkes 2001
    sig_pix = sig_conv / 106.3
    xx = np.arange(0, 2*np.round(3.5*sig_pix+1, 0), 1) - \
        np.round(3.5*sig_pix+1, 0)
    kernel = np.exp(-xx ** 2 / (2 * sig_pix ** 2))
    kernel = kernel / np.sum(kernel)
    F_Fe_conv_intermed = np.convolve(F_fe, kernel, 'same')
    F_Fe_opt_conv = F_Fe_conv_intermed
    F_Fe_opt_conv = \
        specConv.spect_convolver(wave_fe, F_Fe_conv_intermed, 
                                 0.4861*(1. + zstar))
    return F_Fe_opt_conv, wave_fe

def Fe_flux_UV(Fe_FWHM, zstar, specConv):
    '''
    This function loads the UV FeII template, redshifts it,
    convolves it with the intrinsic broadening, and convolves it
    with the instrumental resolution.

    Parameters
    ----------
    Fe_FWHM: float
        Defines the smoothing of the FeII template in km/s. FWHM of the
        Gaussian smoothing that will be applied. Needs to be >900 km/s
        (FWHM of template).
    zstar: float
        Redshift of the stellar continuum.
    specConv: instance of the spectConvol class

    Returns
    -------
    F_Fe_uv_conv: array
        Convolved FeII template.
    wave_fe: array
        Wavelength array of the FeII template. 
    '''
    temp_fe_uv = '../q3dfit/data/questfit_templates/fe_uv.txt'
    data2 = np.loadtxt(temp_fe_uv)
    wave_fe = 10**data2[:,0] * (1. + zstar) /1e4
    F_fe = data2[:,1]/np.max(data2[:,1])
    sig_conv = np.sqrt(Fe_FWHM ** 2 - 900.0 ** 2) / 2. / \
        np.sqrt(2. * np.log(2.))
    sig_pix = sig_conv / 106.3
    xx = np.arange(0, 2*np.round(3.5*sig_pix+1, 0), 1) - \
        np.round(3.5*sig_pix+1, 0)
    kernel = np.exp(-xx ** 2 / (2 * sig_pix ** 2))
    kernel = kernel / np.sum(kernel)
    F_Fe_conv_intermed = np.convolve(F_fe, kernel, 'same')
    F_Fe_uv_conv = F_Fe_conv_intermed
    F_Fe_uv_conv = \
        specConv.spect_convolver(wave_fe, F_Fe_conv_intermed, \
                                 0.2800*(1. + zstar))
    return F_Fe_uv_conv, wave_fe



def linfit_plus_FeII(lam, flux,weight, unused1, unused2, index, 
                     zstar, specConv, outdir=None, quiet=False, refit=None, 
                     rows=None, cols=None, Fe_FWHM=None, Fe_FWHM_UV=None, 
                     **kwargs):
    '''
    This function fits the continuum as a superposition of a linear fit and 
    an FeII template. 
    * Optical FeII template:  taken from Vestergaard & Wilkes (2001).
    * UV FeII template: following the PyQSOFit code (Guo et al. 2018): 
        template by Shen et al. (2019) - composite of 
        Vestergaard & Wilkes (2001), Tsuzuki et al. (2006) and 
        Salviander et al. (2007)
    The UV and optical FeII templates are scaled up/down separately 
    from each other.

    Parameters
    -----
    lam: array
        wavelength
    flux: array
        Flux values to be fit
    weight: array
        Weights of each individual pixel to be fit
    unused1:
        Not used in this function.
    unused2:
        Not used in this function.
    index: array
        Pixels used in the fit
    zstar: float
        redshift of the stellar continuum
    specConv: instance of the spectConvol class
        specifying the instrumental spectral resolution convolution
    outdir: string
        directory for saving the output log
    quiet: bool
        not used in this function
    refit: bool
        not used in this function
    rows, cols: int
        row(s) and column(s) of the input cube in which the fitting is done. 
        Used only to define the name of the output log
    Fe_FWHM: float
        Defines the smoothing of the FeII template in km/s. FWHM of the Gaussian smoothing that will be applied. Needs to be >900 km/s (FWHM of template).

    Returns
    -------
    continuum: array
        best fit continuum model
    ct_coeff: dictionary or lmfit best fit params structure
        best fit parameters
    '''

    flux_cut = flux[index]
    lam_cut = lam[index]

    models_dictionary = {'x': lam_cut}

    temp_fe = '../q3dfit/data/questfit_templates/fe_optical.txt'
    temp_fe_uv = '../q3dfit/data/questfit_templates/fe_uv.txt'
    ind_opt = (lam_cut > 0.3686*(1.+zstar))&(lam_cut < 0.7484*(1.+zstar))
    ind_UV = (lam_cut > 0.120*(1.+zstar))&(lam_cut < 0.350*(1.+zstar))

    if Fe_FWHM is None:
        print('\n\nFitting with FeII template, but no broadening applied. \
              Please supply argscontfit with Fe_FWHM... Halting.')
        import sys; sys.exit()

    model_linfit = LinearModel(independent_vars=['x'], prefix='lincont', 
                               nan_policy='raise', **kwargs)
    pars_lin = model_linfit.make_params(intercept=np.nanmedian(flux), slope=0)

    model = model_linfit
    param = pars_lin

    if np.sum(ind_opt)>0:
        F_fe_rebin = np.zeros(len(lam_cut))
        F_Fe_opt_conv, wave_fe = Fe_flux_balmer(Fe_FWHM, zstar, specConv)
        F_fe_rebin[ind_opt] = \
            interp_temp_quest.interp_lis(lam_cut[ind_opt], wave_fe, 
                                         F_Fe_opt_conv)
        models_dictionary['temp_fe_opt'] = F_fe_rebin
        p_init = [0.5]
        p_vary = [True]
        model_fe_templ_opt, param_fe_templ_opt = \
            questfitfcn.set_up_fit_model_scale(p_init, p_vary, 
            'temp_fe_opt', 'temp_fe_opt', maxamp=1.05*max(flux[index]))
        model += model_fe_templ_opt
        param += param_fe_templ_opt

    if np.sum(ind_UV)>0:
        if Fe_FWHM_UV is None:
            Fe_FWHM_UV = Fe_FWHM
        F_Fe_rebin_UV = np.zeros(len(lam_cut))
        F_Fe_UV_conv, wave_fe_UV = Fe_flux_UV(Fe_FWHM_UV, zstar, specConv)
        F_Fe_rebin_UV[ind_UV] = \
            interp_temp_quest.interp_lis(lam_cut[ind_UV], wave_fe_UV, 
                                         F_Fe_UV_conv)
        models_dictionary['temp_fe_UV'] = F_Fe_rebin_UV
        p_init = [0.5]
        p_vary = [True]
        model_fe_templ_UV, param_fe_templ_UV = \
            questfitfcn.set_up_fit_model_scale(p_init, p_vary, 'temp_fe_UV', 
            'temp_fe_UV', maxamp=1.05*max(flux[index]) ) 
        model += model_fe_templ_UV
        param += param_fe_templ_UV

    result = model.fit(flux_cut, param, **models_dictionary,
                               max_nfev=int(1e5), method='least_squares',
                               nan_policy='omit', **{'verbose': 2})
    print(result.fit_report())

    # extrapolate result to full wavelength vector
    models_dictionary_B = {'x': lam}
    if np.sum(ind_opt)>0:
        F_Fe_opt_all = interp_temp_quest.interp_lis(lam, wave_fe, F_Fe_opt_conv)
        models_dictionary_B['temp_fe_opt'] = F_Fe_opt_all
    if np.sum(ind_UV)>0:
        F_Fe_UV_all = interp_temp_quest.interp_lis(lam, wave_fe_UV, F_Fe_UV_conv)
        models_dictionary_B['temp_fe_UV'] = F_Fe_UV_all

    continuum = result.eval(**models_dictionary_B)
    comp_best_fit = result.eval_components(**models_dictionary_B)
    

    ct_coeff = {'params_best_fit': result.params,
                'comp_best_fit': comp_best_fit}

    if isinstance(rows, list) or isinstance(rows, np.ndarray):
        rows = rows[0]        
    if isinstance(cols, list) or isinstance(cols, np.ndarray):
        cols = cols[0]        
    if rows is not None and cols is not None:
        saveres = 'fit_result__{}_{}.txt'.format(cols, rows)
    else:
        saveres = 'fit_result.txt'

    if outdir is not None:
        with open(outdir+saveres, 'w') as fh:
            fh.write(result.fit_report())
            fh.write('\n')

    return continuum, ct_coeff, None


def fitqsohost(wave, flux, weight, template_wave, template_flux, index,
               zstar, quiet=True, blrpar=None, qsoxdr=None,
               qsoonly=False, index_log=None, err_log=None,
               flux_log=None, refit=None,
               add_poly_degree=30, siginit_stars=50.,
               fitran=None, qsoord=None, hostonly=False, hostord=None, 
               blronly=False, **kwargs):
    '''
    Function defined to fit the continuum using a quasar template and a
    stellar template.

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
        Flux of the stellar template used as model for stellar continuum
    index: array
        Pixels used in the fit
    zstar: float
        redshift of the stellar continuum
    quiet: bool, optional
    blrpar: array, optional
        Parameters of the broad line region scattering model.
    qsoxdr: string, optional
        Path and filename for the quasar template.
    qsoonly: bool, optional
        Fit only the quasar template.
    index_log: array, optional
        Pixels used in the fit for the pPXF fit, as log-rebinned.
    err_log: array, optional
        Error values for the log-rebinned pixels.
    flux_log: array, optional
        Flux values for the log-rebinned pixels.
    refit: string, optional
        Refit the continuum residual after subtracting the quasar. 
        Options are 'ppxf' or 'questfit'.
    add_poly_degree: int, optional
        Degree of the additive polynomial for the pPXF fit. Default is 30.
    siginit_stars: float, optional
        Initial guess for the stellar velocity dispersion in km/s, used
        in the pPXF fit. Default is 50.
    fitran: array, optional
        Wavelength range to fit. Default is None, to fit all wavelengths.
    qsoord: int, optional
        Order of the polynomial added to the quasar multiplier.
        Default is None.
    hostonly: bool, optional
        Fit only the host galaxy.
    hostord: int, optional
        Order of the polynomial model for the host galaxy. Default is None.
    blronly: bool, optional
        Fit only the broad line region scattering model.
    kwargs: dict, optional
        Additional keyword arguments.
    
    Returns
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

    if fitran is not None:
        iqsoflux = np.where((qsowave >= fitran[0]) & (qsowave <= fitran[1]))
        qsoflux = qsoflux_full[iqsoflux]
    else:
        qsoflux = qsoflux_full

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
                   blronly=blronly, blrpar=blrpar, qsoflux=qsoflux,
                   medflux=np.median(iflux), **kwargs)

    method = 'least_squares'
    if 'method' in kwargs:
        method = kwargs['method']

    fit_kws = {}
    if method == 'least_squares':
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
                      wave=iwave, x=iwave, method=method,
                      nan_policy='raise', fit_kws=fit_kws)

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
        continuum[continuum < 0] = 0

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
        cont_resid, ct_coeff, zstar = questfit(wave, resid, weight, None,
                                               None, index, zstar,
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


def questfit(wlambda, flux, weights, singletemplatelambda, singletemplateflux,
             index, z, quiet=True, config_file=None, global_ice_model='None',
             global_ext_model='None', fitran=None, convert2Flambda=True,
             outdir=None, plot_decomp=None, rows=None, cols=None, **kwargs):
    '''
    Function defined to fit the MIR continuum

    Parameters
    -----
    wlambda: array
        wavelength array in micron
    flux: array
        Flux values to be fit. Assumed input units are Jy. [Need to check 
        this.]
    weights: array
        Weights of each individual pixel to be fit
    singletemplatelambda : array
    singletemplateflux : array
        Wavelength and flux arrays for any continuum template
        separate from the simple empirical BB, power-law etc. components.
    index: array
        Pixels used in the fit
    z: float
        redshift
    quiet: bool, optional
    config_file: string, optional
        Configuration file for the fit. Default is None.
    global_ice_model: string, optional
        Global ice model. Default is 'None'. Options are ...
    global_ext_model: string, optional
        Global extinction model. Default is 'None'. Options are ...
    fitran: array, optional
        Wavelength range to fit. Default is None, to fit all wavelengths.
    convert2Flambda: bool, optional
        Convert flux to erg/s/cm2/mu [/sr] from Jy. [Need to check this.]
        Default is True.
    outdir: string, optional
        Directory for saving the output log. Default is None.
    plot_decomp: bool, optional
        Decompose components of the fit when plotting. Default is None.
    rows, cols: int, optional
        Row(s) and column(s) of the input cube in which the fitting is done.
        Used only to define the name of the output log.
    kwargs: dict, optional
        Additional keyword arguments.

    Returns
    -------
    continuum: array
        best fit continuum model
    ct_coeff: dictionary or lmfit best fit params structure
        best fit parameters
    z: float
        Same as input z.
    '''

    # models dictionary holds extinction, absorption models
    emcomps = dict()
    # template dictionary holds templates, blackbodies, powerlaws
    allcomps = dict()

    # Apply fit range to data
    if fitran:
        flux = flux[wlambda >= fitran[0] & wlambda <= fitran[1]]
        wlambda = wlambda[wlambda >= fitran[0] & wlambda <= fitran[1]]
        #flux = flux[np.logical_and(wlambda >= fitran[0]),
        #            np.logical_and(wlambda <= fitran[1])]
        #wlambda = wlambda[np.logical_and(wlambda >= fitran[0]),
        #                  np.logical_and(wlambda <= fitran[1])]

    if singletemplatelambda is not None:
        print('Trying to pass a single separate template to questfit, \
              which is not implemented ... Halting.')
        import sys; sys.exit()

    # Read input configuration file
    config_file = readcf(config_file)

    # set up global extinction. Loop through all lines, looking for global tag
    global_extinction = False
    for key in config_file:
        if len(config_file[key]) > 3:
            if 'global' in config_file[key][3]:
                global_extinction = True

    # Set name of global extinction / ice models
    if global_extinction:
        for key in config_file:
            if 'extinction' in config_file[key]:
                global_ext_model = key
            if 'absorption' in config_file[key]:
                global_ice_model = key

    # counter for PAH/silicate/quasar templates
    n_temp = 0
    # populating the components dictionaries and setting up lmfit models
    for i in config_file.keys():

        # loc_models = q3dfit.__path__[0]+'/data/questfit_templates/'
        if 'blackbody' in i:
            #starting with the blackbodies
            model_parameters = config_file[i]
            name_model = 'blackbody'+str(int(float(model_parameters[7])))
            extinction_model = config_file[i][3]
            ice_model = config_file[i][9]
            # instantiate model class and set parameters
            model_temp_BB, param_temp_BB = \
                questfitfcn.\
                set_up_fit_blackbody_model([np.float64(model_parameters[1]),
                                            np.float64(model_parameters[7])],
                                           [np.float64(model_parameters[2]),
                                            np.float64(model_parameters[8])],
                                           name_model[:])
            # instantiate individual extinction model
            if global_extinction is False and \
                config_file[i][3] != '_' and \
                config_file[i][3] != '-':
                model_temp_extinction, param_temp_extinction = \
                    questfitfcn.\
                    set_up_fit_extinction([np.float64(model_parameters[4])],
                                          [np.float64(model_parameters[5])],
                                          name_model+'_ext',
                                          extinction_model,
                                          model_parameters[6])
                # multiply the BB model by extinction and add parameters
                # temporary holders just for this component
                model_temp = model_temp_BB*model_temp_extinction
                param_temp = param_temp_BB + param_temp_extinction
                # add extinction model to dictionary
                allcomps[extinction_model] = \
                    config_file[extinction_model][0]
            else:
                # temporary holders just for this component, without extinction
                model_temp = model_temp_BB
                param_temp = param_temp_BB

            # checking if we need to add ice absorption
            if 'ice' in i and global_extinction is False:
                model_temp_ice, param_temp_ice = \
                    questfitfcn.\
                    set_up_absorption([np.float64(model_parameters[10])],
                                      [np.float64(model_parameters[11])],
                                      name_model+'_abs',
                                      model_parameters[9])
                model_temp = model_temp*model_temp_ice
                param_temp += param_temp_ice
                allcomps[model_parameters[9]] = \
                    config_file[model_parameters[9]][0]

            # Check if this is the first component; if not, set up full
            # model/pars
            if 'model' not in vars():
                model, param = model_temp, param_temp
            # otherwise, add them in
            else:
                model += model_temp
                param += param_temp

        # powerlaw model
        if 'powerlaw' in i:
            model_parameters = config_file[i]
            name_model = 'powerlaw'+str(int(float(model_parameters[7])))
            extinction_model = config_file[i][3]
            ice_model = config_file[i][9]

            #powerlaw model
            # exponent to that specified in config. file.
            model_temp_powerlaw, param_temp_powerlaw = \
                questfitfcn.\
                    set_up_fit_powerlaw_model(
                        [np.float64(1.), np.float64(model_parameters[7])],
                        [np.float64(model_parameters[2]),
                         np.float64(model_parameters[8])],
                        name_model[:])
            if global_extinction is False and \
                config_file[i][3] != '_' and \
                config_file[i][3] != '-':
                model_temp_extinction, param_temp_extinction = \
                    questfitfcn.\
                    set_up_fit_extinction([np.float64(model_parameters[4])],
                                          [np.float64(model_parameters[5])],
                                          'powerlaw' +
                                          str(int(float(model_parameters[7])))
                                          + '_ext', extinction_model,
                                          model_parameters[6])

                model_temp = model_temp_powerlaw*model_temp_extinction
                param_temp = param_temp_powerlaw + param_temp_extinction

                allcomps[extinction_model] = \
                    config_file[extinction_model][0]
            else:
                model_temp = model_temp_powerlaw
                param_temp = param_temp_powerlaw

            if 'ice' in i and global_extinction is False:
                model_temp_ice, param_temp_ice = \
                    questfitfcn.\
                    set_up_absorption([np.float64(model_parameters[10])],
                                      [np.float64(model_parameters[11])],
                                      name_model+'_abs', model_parameters[9])
                model_temp = model_temp*model_temp_ice
                param_temp += param_temp_ice
                allcomps[model_parameters[9]] = \
                    config_file[model_parameters[9]][0]

            if 'model' not in vars():
                model, param = model_temp, param_temp
            else:
                model += model_temp
                param += param_temp

        if 'template' in i: #template model
            model_parameters = config_file[i]
            name_model = 'template_'+str(n_temp)#i
            extinction_model = config_file[i][3]
            ice_model = config_file[i][9]
            # if it's not a polynomial model for PSF fitting
            if not 'poly' in i:
                model_temp_template, param_temp_template = \
                    questfitfcn.\
                        set_up_fit_model_scale([np.float64(model_parameters[1])],
                                               [np.float64(model_parameters[2])],
                                               name_model, name_model) #,
                                                #maxamp=1.05*max(flux[index]) )
            # ... and if it is!
            else:
                # constrain amplitude
                minamp = np.float64(model_parameters[1]) / 1.25
                maxamp = np.float64(model_parameters[1]) * 1.25
                model_temp_template, param_temp_template = \
                    questfitfcn.set_up_fit_model_scale_withpoly(
                        [np.float64(model_parameters[1])],
                        [np.float64(model_parameters[2])],
                        name_model, name_model, minamp=minamp, maxamp=maxamp)

                # testing = False
                # if testing:
                #     with pkg_resources.path(
                #             questfit_templates,
                #             'miri_qsotemplate_flex.npy') as p:
                #         temp_model = np.load(p, allow_pickle=True)
                #     temp_model = temp_model[()]
                #     wave_ex = temp_model['wave']
                #     flux_ex = temp_model['flux']
                #     c_scale =  constants.c * u.Unit('m').to('micron') /(wave_ex)**2 *1e-23  *1e10      # [1e-10 erg/s/cm^2/um/sr]]
                #     flux_ex = flux_ex * c_scale
                #     flux_ex = flux_ex/flux_ex.max()
                #     c2 = [1., 0., 0.]
                #     c2 = [0.116, 0.224, 147.71]
                #     c2 = [0.213, 0.024, 75.794]
                #     c2 = [1, 100., 500.71]
                #     m1 = model_temp_template.eval(template_0=flux_ex , wave=wave_ex, template_0_amp=1.,template_0_multpolyA=1.,template_0_multpolyB=2.,template_0_multpolyC = 3. )
                #     m2 = model_temp_template.eval(template_0=flux_ex , wave=wave_ex, template_0_amp=1.,template_0_multpolyA=c2[0],template_0_multpolyB=c2[1],template_0_multpolyC = c2[2] )

                #     ex_poly = lambda lam, p0,p1,p2:  (p0+p1*lam+p2*lam**2)/max(p0+p1*lam+p2*lam**2)
                #     ex_poly2 = lambda lam, p0,p1,p2:  (p0+p1*lam+p2*lam**7)/max(p0+p1*lam+p2*lam**7)
                #     ex_poly = lambda lam, p0,p1,p2:  (p0+p1*lam+p2*lam**2)/max(p0+p1*lam+p2*lam**2)
                #     lin_poly = lambda lam, p0,p1,p2:  (p0+p1*np.arange(len(lam)))/max((p0+p1*np.arange(len(lam))))
                #     plt.figure()
                #     plt.plot(wave_ex, flux_ex, label='orig', color='k', linewidth=2.5)
                #     plt.plot(wave_ex, m1, linewidth=1, label='model.eval() with  A=1, B=2, C=3 (and Amp=1)')
                #     plt.plot(wave_ex, m2, linewidth=1, label='model.eval() with  A={}, B={}, C={} (and Amp=1)'.format(c2[0], c2[1], c2[2]))
                #     plt.xlabel(r'$\lambda$')
                #     plt.ylabel('F_norm')
                #     plt.legend()
                #     plt.show()

                #     breakpoint()

            if global_extinction is False and \
                config_file[i][3] != '_' and \
                config_file[i][3] != '-':

                model_temp_extinction, param_temp_extinction = \
                    questfitfcn.\
                    set_up_fit_extinction([np.float64(model_parameters[4])],
                                          [np.float64(model_parameters[5])],
                                          name_model + '_ext',
                                          extinction_model,
                                          model_parameters[6])
                model_temp = model_temp_template*model_temp_extinction
                param_temp = param_temp_template + param_temp_extinction

                allcomps[extinction_model] = \
                    config_file[extinction_model][0]

            else:
                model_temp = model_temp_template
                param_temp = param_temp_template

            if 'ice' in i and global_extinction is False:
                model_temp_ice, param_temp_ice = \
                    questfitfcn.set_up_absorption([float(model_parameters[10])],
                                                  [float(model_parameters[11])],
                                                  name_model+'_abs',
                                                  model_parameters[9])
                model_temp = model_temp*model_temp_ice
                param_temp += param_temp_ice

                allcomps[model_parameters[9]] = \
                    config_file[model_parameters[9]]

            if 'model' not in vars():
                model,param = model_temp,param_temp
            else:
                model += model_temp
                param += param_temp
            # add numerical template to emcomps
            emcomps[name_model] = config_file[i][0]
            # increment counter for PAH/silicate/quasar templates
            n_temp+=1

        #if qsoflux is not None:
            #model_qso, param_qso = questfitfcn.set_up_fit_model_scale

    # Check to see if we are using global extinction, where the total
    # model flux is extincted by the same ice and dust model.
    if global_extinction:

        model_global_ext, param_global_ext = \
            questfitfcn.set_up_fit_extinction([np.float64(0.)],
                                              [np.float64(1.)], 'global_ext',
                                              global_ext_model, 'S')
        model = model*model_global_ext
        param += param_global_ext
        allcomps[global_ext_model] = config_file[global_ext_model][0]

        model_global_ice, param_global_ice = \
            questfitfcn.set_up_absorption([np.float64(0.)], [np.float64(1.)],
                                          'global_ice', global_ice_model)
        model = model*model_global_ice
        param += param_global_ice
        allcomps[global_ice_model] = \
            config_file[global_ice_model][0]

    # loop over abs. dictionary, load in the numerical models and resample.
    for i in allcomps.keys():
        #if qsoflux is not None:
        with pkg_resources.path(questfit_templates, allcomps[i]) as p:
            temp_model = np.load(p, allow_pickle=True)
            if not os.path.exists(p):
                temp_model = np.load(outdir+allcomps[i])

        # Check to see if we are using global extinction, where the total
        temp_wave=temp_model['WAVE']*(1.+z)
        temp_value=temp_model['FLUX']

        temp_value_rebin = \
            interp_temp_quest.interp_lis(wlambda, temp_wave, temp_value)
        allcomps[i] = temp_value_rebin

    # conversion from f_nu to f_lambda: 
    # f_lambda = f_nu x c/lambda^2
    # [1 Jy = 1e-10 erg/s/cm^2/um [/sr] ???]
    # about 1.2 for wlambda=5 [micron]
    c_scale =  constants.c * u.Unit('m').to('micron') /(wlambda)**2 * \
        1e-23 * 1e10

    # loop over emission template dictionary, load them in and resample.
    for i in emcomps.keys():
        if 'sifrom' in emcomps[i]:
            tempdir = silicatemodels
        else:
            tempdir = questfit_templates
        with pkg_resources.path(tempdir, emcomps[i]) as p:
             temp_model = np.load(p, allow_pickle=True)

        try:
            temp_wave = np.float64(temp_model['WAVE']*(1.+z))
            temp_value = np.float64(temp_model['FLUX'])
        except:
            # if a QSO template generated by makeqsotemplate() is included,
            # that is formatted slightly differently
            temp_model = temp_model[()]
            temp_wave = temp_model['wave'] #*(1.+z)
            temp_value = temp_model['flux']

        temp_value_rebin = \
            interp_temp_quest.interp_lis(wlambda, temp_wave, temp_value)
        allcomps[i] = temp_value_rebin
        if convert2Flambda:
            allcomps[i] *= c_scale

        # conversion from f_nu to f_lambda: f_lambda = f_nu x c/lambda^2
        allcomps[i] = allcomps[i]/allcomps[i].max()

    # Add in wavelength and units flag
    allcomps['wave'] = np.float64(wlambda)
    allcomps['fitFlambda'] = bool(convert2Flambda)

    #if convert2Flambda:
    #    flux *= c_scale

    # plot_ini_guess = False
    # if plot_ini_guess:
    #     plt.plot(allcomps['wave'],
    #              param['template_0_amp'].value *
    #              allcomps['template_0']/c_scale,
    #              color='c', label = 'QSO model init')
    #     with pkg_resources.path(questfit_templates,
    #                             'miri_qsotemplate_flexB.npy') as p:
    #         data1 = np.load(p, allow_pickle='TRUE').item()
    #     F1 = data1['flux'][:-1] * c_scale
    #     plt.plot(allcomps['wave'], F1/c_scale, color='b', label = 'QSO real')

    #     gal_model_comp = [el for el in allcomps if 'template' in el and 'template_0' not in el]
    #     Fgalmodel = 0
    #     for comp_i in gal_model_comp:
    #         Fgalmodel += param[comp_i+'_amp'].value * allcomps[comp_i]
    #     plt.plot(allcomps['wave'], Fgalmodel/c_scale, color='plum', label = 'host model init')

    #     with pkg_resources.path(questfit_templates,
    #                             'miri_gal_spec.npy') as p:
    #         data2 = np.load(p, allow_pickle='TRUE').item()
    #     F2 = data2['flux'][:-1] * c_scale
    #     plt.plot(allcomps['wave'], F2/c_scale, color='darkviolet', label = 'host real')
    #     plt.yscale("log")
    #     plt.xlabel(r'$\lambda \ \mathrm{[micron]}$')
    #     plt.legend()
    #     plt.show()
    #     breakpoint()

    # produce copy of components dictionary with index applied
    flux_cut = flux[index]
    allcomps_cut = copy.deepcopy(allcomps)
    for el in allcomps.keys():
        if not ('fitFlambda' in el):
            allcomps_cut[el] = allcomps_cut[el][index]

    # not sure what this is ...
    # with pkg_resources.path(questfit_templates,
    #                         'miri_qsotemplate_flex.npy') as p:
    #     data1 = np.load(p, allow_pickle=True)
    # f_orig = data1.item()['flux'][:-1]

    # from multiprocessing import Pool
    # with Pool() as pool:
    # use_emcee = False
    # if use_emcee:

    #     # -- Originally used max_nfev=int(1e5), and method='least_squares'
    #     emcee_kws = dict(steps=5000, burn=500, thin=20, is_weighted=False, progress=True) #, run_mcmc_kwargs={'skip_initial_state_check': True} )
    #     #emcee_kws = dict(nwalkers=500, steps=5000, burn=500, thin=20, workers=pool, is_weighted=False, progress=True) #, run_mcmc_kwargs={'skip_initial_state_check': True} )
    #     # emcee_kws = dict(nwalkers=256, steps=50000, burn=500, thin=5, is_weighted=False, progress=True) #, run_mcmc_kwargs={'skip_initial_state_check': True} )

    #     param.add('__lnsigma', value=np.log(0.1), min=np.log(0.001), max=np.log(2.0))
    #     import time
    #     t1 = time.time()
    #     result = model.fit(flux_cut,param,**allcomps_cut,max_nfev=int(1e5),method='emcee',nan_policy='omit', fit_kws=emcee_kws)#method='least_squares'nan_policy='omit'
    #     print('Time needed for fitting: ', time.time()-t1)

    #     import corner
    #     emcee_plot = corner.corner(result.flatchain, labels=result.var_names,truths=list(result.params.valuesdict().values()))
    #     plt.savefig(outdir+'corner')

    # else:
    #test = model.eval(param,**allcomps_cut)
    #import pdb; pdb.set_trace()

    method = 'least_squares'
    if 'method' in kwargs:
        method = kwargs['method']

    fit_kws = {}
    if method == 'least_squares':
        if quiet:
            lmverbose = 0  # verbosity for scipy.optimize.least_squares

        else:
            lmverbose = 2
        fit_kws = {'verbose': lmverbose}

    # Add additional parameter settings for scipy.optimize.least_squares
    if 'argslmfit' in kwargs:
        for key, val in kwargs['argslmfit'].items():
            fit_kws[key] = val

    # FIT!
    result = model.fit(flux_cut, param, **allcomps_cut,
                       max_nfev=int(1e5), method=method,
                       nan_policy='raise', fit_kws=fit_kws)

    lmfit.report_fit(result.params)
    if rows is not None and cols is not None:
        saveres = 'fit_result__{}_{}.txt'.format(cols, rows)
    else:
        saveres = 'fit_result.txt'
    with open(outdir+saveres, 'w') as fh:
        fh.write(result.fit_report())
        fh.write('\n')

    # use allcomps rather than allcomps_cut to evaluate
    # over all wavelengths within fitran (not just [index])
    best_fit = result.eval(**allcomps)
    comp_best_fit = result.eval_components(**allcomps)
    # print(result.best_values)

    # apply extinction and ice absorption to total best fit
    if global_extinction:
        for el in comp_best_fit.keys():
            if el != 'global_ext' and el != 'global_ice':
                comp_best_fit[el] *= comp_best_fit['global_ext']
                comp_best_fit[el] *= comp_best_fit['global_ice']

    # if convert2Flambda:
    #     flux /= c_scale
    #     best_fit /= c_scale
    #     for el in comp_best_fit.keys():
    #         if not (global_ext_model in el) and \
    #             not (global_ice_model in el) and \
    #             not ('ext' in el) and not ('ice' in el):
    #             try:
    #                 comp_best_fit[el] /= c_scale
    #             except Exception as e:
    #                 print(e)
    #                 import pdb; pdb.set_trace()

    ct_coeff = {'MIRparams': result.params,
                'comp_best_fit': comp_best_fit}

    return best_fit, ct_coeff, z


def quest_extract_QSO_contrib(ct_coeff, initdat):
    '''
    This function can be used to recover the QSO-host decomposition after running questfit

    :Params:
        ct_coeff: in, required, type=dict
            dict returned by questfit containing the continuum fitting results
        initdat: in, required, type=dict
            dict that was used to initialize the fit

        qso_out_ext: out, type=array
            spectrum of the QSO component (with dust extinction and ice absorption)
        host_out_ext:  out, type=array
            spectrum of the host component (with dust extinction and ice absorption)
    '''
    comp_best_fit = ct_coeff['comp_best_fit']
    qso_out_ext = np.array([])
    qso_out_intr = np.array([])

    #config_file = questfitfcn.readcf(initdat['argscontfit']['config_file'])
    config_file = readcf(initdat.argscontfit['config_file'])
    if not 'qso' in list(config_file.keys())[1]:    ### This function assumes that in the config file the qso temple is the first template. Rudimentary check here.
        print('\n\nWARNING during QSO-host decomposition: \nThe function assumes that in the config file the qso template is the first template, but its name does not contain \"qso\". Pausing here as a checkpoint, press c for continuing.\n')
        import pdb; pdb.set_trace()

    global_extinction = False
    for key in config_file:
        if len(config_file[key]) > 3:
            if 'global' in config_file[key][3]:
                global_extinction = True

    if global_extinction:
        str_global_ext = list(comp_best_fit.keys())[-2]
        str_global_ice = list(comp_best_fit.keys())[-1]
        if len(comp_best_fit[str_global_ext].shape) > 1:  # global_ext is a multi-dimensional array
          comp_best_fit[str_global_ext] = comp_best_fit[str_global_ext] [:,0,0]
        if len(comp_best_fit[str_global_ice].shape) > 1:  # global_ice is a multi-dimensional array
          comp_best_fit[str_global_ice] = comp_best_fit[str_global_ice] [:,0,0]
        host_out_ext = np.zeros(len(comp_best_fit[str_global_ext]))
        host_out_intr = np.zeros(len(comp_best_fit[str_global_ext]))

        for i, el in enumerate(comp_best_fit):
          if (el != str_global_ext) and (el != str_global_ice):
            if len(comp_best_fit[el].shape) > 1:              # component is a multi-dimensional array
              comp_best_fit[el] = comp_best_fit[el] [:,0,0]
            if hasattr(initdat, 'decompose_qso_fit'):
              if initdat.decompose_qso_fit and i==0:     ### NOTE on i==0: This only works is in the config file the qso temple is the first template
                qso_out_ext = comp_best_fit[el]*comp_best_fit[str_global_ext]*comp_best_fit[str_global_ice]
                qso_out_intr = comp_best_fit[el]
              else:
                host_out_ext += comp_best_fit[el]*comp_best_fit[str_global_ext]*comp_best_fit[str_global_ice]
                host_out_intr += comp_best_fit[el]
    else:
        el1 = list(comp_best_fit.keys())[0]
        host_out_ext = np.zeros(len(comp_best_fit[el1]))
        host_out_intr = np.zeros(len(comp_best_fit[el1]))

        spec_i = np.array([])
        for i, el in enumerate(comp_best_fit):
            if len(comp_best_fit[el].shape) > 1:
              comp_best_fit[el] = comp_best_fit[el] [:,0,0]

            if not ('_ext' in el or '_abs' in el):
                spec_i = comp_best_fit[el]
                intr_spec_i = comp_best_fit[el].copy()
                if el+'_ext' in comp_best_fit.keys():
                    spec_i = spec_i*comp_best_fit[el+'_ext']
                if el+'_abs' in comp_best_fit.keys():
                    spec_i = spec_i*comp_best_fit[el+'_abs']

                if hasattr(initdat, 'decompose_qso_fit'):
                    if initdat.decompose_qso_fit and i==0:
                        qso_out_ext = spec_i
                        qso_out_intr = intr_spec_i
                    else:
                        host_out_ext += spec_i
                        host_out_intr += intr_spec_i
                        #breakpoint()

    return qso_out_ext, host_out_ext, qso_out_intr, host_out_intr

