'''
Continuum fitting routines and subroutines.
'''
from __future__ import annotations

__all__ = ['fitpoly', 'fitqsohost', 'questfit', 'linfit_plus_FeII',
           'Fe_flux_balmer', 'Fe_flux_UV', 'quest_extract_QSO_contrib']

from typing import Optional, Any
from numpy.typing import ArrayLike

import copy
import importlib.resources as pkg_resources
import lmfit
import numpy as np
import ppxf.ppxf_util as util
import os

from astropy import units as u
from astropy.constants import c # type: ignore
from astropy.modeling import models, fitting
# from lmfit.models import ExpressionModel
from lmfit.models import LinearModel
from ppxf.ppxf import ppxf
from scipy import interpolate

from . import q3dmath, q3dutil, qsohostfcn, questfitfcn
from q3dfit import questfitfcn
from q3dfit.data import questfit_templates
#from q3dfit.data.questfit_templates import silicatemodels
from q3dfit.exceptions import InitializationError


def fitpoly(wave: np.ndarray, 
            flux: np.ndarray, 
            weight: np.ndarray, 
            index: np.ndarray, 
            logfile: Optional[str]=None,
            quiet: bool=True,
            fitord: int=3, 
            refit: Optional[dict[str, ArrayLike]] = None) \
                -> tuple[np.ndarray, ArrayLike, None]:
    '''
    Fit the continuum as a polynomial.

    Parameters
    ----------
    wave
        Wavelengths
    flux
        Fluxes
    weight
        Weights
    index
        Points used in the fit.
    logfile
        Optional. Name of the log file. Default is None.
    quiet
        Optional. Suppress output to stdout? Default is True.
    fitord
        Optional. Order of the polynomial. Default is 3.
    refit
        Optional. Refit with a different polynomial order? If so, set to dictionary
        with keys 'ord' and 'ran' for the order and range of the
        refit, respectively. Can specify multiple ranges and orders as
        array_like. Default is None.

    Returns
    -------
    np.ndarray
        best fit continuum model
    ArrayLike
        Polynomial coefficients.
    None
        Unused
    '''
    iwave = np.array(wave[index])
    iflux = np.array(flux[index])
    #the fitter I used puts weights in 1/sigma so I took the square root to make the data correct
    w = np.array(weight[index])
    iweight = np.array(np.sqrt(w))

    iwave = iwave.reshape(iwave.size)
    iflux = iflux.reshape(iwave.size)
    iweight = iweight.reshape(iwave.size)

    if fitord==0:
        deg1=len(iwave)-1
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
    fluxfit = fitter(polymod1, iwave, iflux, weights=iweight)
    # polynomial coefficients
    fluxfitparam = fluxfit.parameters

    #flip for numpy.poly1d
    ct_coeff = np.flip(fluxfitparam)
    # this is a class:
    # https://numpy.org/doc/stable/reference/generated/numpy.poly1d.html
    ct_poly = np.poly1d(ct_coeff, variable='lambda')
    continuum = ct_poly(wave)
    icontinuum = continuum[index]

    if refit is not None:
        for i in range (0, np.size(refit['ord']) - 1):
            tmp_ind=np.where(wave >= refit['ran'][0,i] and
                             wave <= refit['ran'][1,i])
            tmp_iind=np.where(iwave >= refit['ran'][0,i] and
                              iwave <= refit['ran'][1,i])
            #  parinfo=np.full(refit['ord'][i]+1, {'value':0.0})

            #degree of polynomial fit defaults to len(x-variable)-1
            if deg2==0:
                deg2=len(iwave[tmp_iind])-1

            #creating tmp_pars
            tmp_pars=fitter(polymod2, iwave[tmp_iind],
                            (iflux[tmp_iind]-icontinuum[tmp_iind]),
                            z=None, weights=iweight[tmp_iind])
            tmp_parsptmp=tmp_pars.parameters
            tmp_parsparam=np.flip(tmp_parsptmp)

            # refitted continuum object
            ct_poly = np.poly1d(tmp_parsparam, variable='lambda')
            # refitted continuum over wavelength range for refitting, 
            # indexed to full wavelength array
            cont_refit = ct_poly(wave[tmp_ind])
            # now add in the refitted continuum to the full continuum
            continuum += cont_refit

    return continuum, ct_coeff, None


def fitqsohost(wave: np.ndarray,
               flux: np.ndarray,
               weight: np.ndarray,
               index: np.ndarray,
               logfile: Optional[str]=None,
               quiet: bool=True,
               refit: Optional[str]=None,
               template_wave: Optional[np.ndarray]=None,
               template_flux: Optional[np.ndarray]=None, 
               zstar: float=0.,
               flux_log: Optional[np.ndarray]=None,
               err_log: Optional[np.ndarray]=None,
               index_log: Optional[np.ndarray]=None,
               siginit_stars: float=50.,
               add_poly_degree: int=30,
               fitran: Optional[ArrayLike]=None,
               qsoord: Optional[int]=None,
               hostord: Optional[int]=None, 
               blrpar: Optional[ArrayLike]=None,
               qsoonly: bool=False,
               hostonly: bool=False,
               blronly: bool=False,
               fluxunit: Optional[str]=None,
               *, qsoxdr: str,
               **kwargs) -> tuple[np.ndarray, dict[str, Any], float]:
    '''
    Fit the continuum using a quasar template and a stellar template. Calls
    :py:func:`~q3dfit.qsohostfcn.qsohostfcn` to jointly fit the quasar and host
    galaxy components. Calls :py:func:`~ppxf.ppxf` or :py:func:`~q3dfit.contfit.questfit`
    to refit the residual after subtracting the quasar component.

    Additional keywords can be passed to :py:meth:`~lmfit.model.Model.fit` by passing 
    them as keyword arguments to this function. Arguments to the :py:mod:`~scipy.optimize` 
    functions can be passed by setting the keyword argument `argslmfit` to a dictionary of 
    parameter names and values as keys and values. Arguments to 
    :py:func:`~q3dfit.contfit.questfit` for the `refit` can be passed by setting the 
    keyword argument `args_questfit` to a dictionary of parameter names and values as 
    keys and values.

    Parameters
    -----
    wave
        Wavelengths
    flux
        Fluxes
    weight
        Weights
    index
        Points used in the fit.
    logfile
        Optional. Name of the log file. Default is None.
    quiet
        Optional. Suppress output to stdout? Default is True.
    fitran
        Optional. Wavelength range to fit. Default is None, to fit all wavelengths.        
    qsoord
        Optional. Order of the polynomial added to the quasar multiplier.
        Default is None.
    hostord
        Optional. Order of the polynomial model for the host galaxy. 
        Default is None.
    blrpar
        Optional. Parameters of the broad line region scattering model. If set,
        fit this part of the model. Set to array_like Default is None.
    qsoonly
        Optional. Fit only the quasar template. Default is False.
    hostonly
        Optional. Fit only the host galaxy. Default is False.
    blronly
        Optional. Fit only the broad line region scattering model. Default is False.
    refit
        Optional. Refit the continuum residual after subtracting the quasar. 
        Options are 'ppxf' or 'questfit'.
    template_wave
        Optional, but must be set if refit='ppxf'. Wavelength array of the stellar 
        template used as model for stellar continuum.
    template_flux
        Optional, but must be set if refit='ppxf'. Wavelength array of the stellar 
        template used as model for stellar continuum.
    zstar
        Optional, but must be set if refit='ppxf'. Input guess for redshift of the 
        stellar continuum. Default is 0.
    flux_log
        Optional, but must be set if refit='ppxf'. Flux values for the log-rebinned 
        pixels.
    err_log
        Optional, but must be set if refit='ppxf'. Error values for the log-rebinned 
        pixels.
    index_log
        Optional, but must be set if refit='ppxf'. Pixels used in the fit for the 
        pPXF fit, as log-rebinned.
    siginit_stars
        Optional. Initial guess for the stellar velocity dispersion in km/s, used
        in the pPXF fit. Default is 50. km/s.
    add_poly_degree
         Optional. Degree of the additive polynomial for the pPXF fit. Default is 
         30.
    qsoxdr
        Path and filename for the quasar template.

    Returns
    -------
    numpy.ndarray
        Best fit continuum model.
    dict[str, Any]
        Best fit parameters. The exact form will depend on the value of the 
        refit parameter. If refit is None, this will be the best fit parameters 
        from the initial fit. If refit is 'ppxf', this will be a dictionary with 
        keys 'qso_host', 'stel', 'poly', and 'ppxf_sigma'. If refit is 'questfit', 
        this will be a dictionary with keys 'qso_host', 'stel', and 'poly'.
    float
        Best fit stellar redshift if refit='ppxf'. Otherwise, the input redshift.
    '''

    # Load quasar template
    try:
        qsotemplate = np.load(qsoxdr, allow_pickle=True).item()
        qsowave = qsotemplate['wave']
        qsoflux_full = qsotemplate['flux']
    except:
        raise InitializationError('Cannot locate quasar template (qsoxdr) specified ')

    # Interpolate quasar template to the data grid. By not doing this
    # we are assuming that the quasar template is already on the same
    # grid as the data.
    # qsoflux = interptemp(wave, qsowave, qsoflux_full)

    # Make sure we are fitting only the range of the quasar template
    # that matches the range of the complete fit, as specfified by fitspec
    if fitran is not None:
        iqsoflux = np.where((qsowave >= fitran[0]) & (qsowave <= fitran[1]))
        qsoflux = qsoflux_full[iqsoflux]
    else:
        qsoflux = qsoflux_full

    index = np.array(index, dtype=np.int64)

    # err = 1/weight**0.5
    iwave = wave[index]
    iflux = flux[index]
    iweight = weight[index]
    # ierr = err[index]

    # Fit the quasar and a featureless host continuum
    ymod, params = \
        qsohostfcn.qsohostfcn(wave, params_fit=None, qsoonly=qsoonly,
            qsoord=qsoord, hostonly=hostonly, hostord=hostord,
            blronly=blronly, blrpar=blrpar, qsoflux=qsoflux,
            medflux=np.median(iflux))

    # Set the fitting method
    # default is scipy.optimize.least_squares
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

    q3dutil.write_msg(result.fit_report(), file=logfile, quiet=quiet)

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

    # Refit the residual after subtracting the quasar component
    if refit == 'ppxf' and \
        flux_log is not None and \
        err_log is not None and \
        index_log is not None and \
        template_wave is not None and \
        template_flux is not None:

        # log rebin residual
        # lamRange1 = np.array([wave.min(), wave.max()])/(1+zstar)
        cont_log, lambda_log, velscale = util.log_rebin(fitran, continuum)

        resid_log = flux_log - cont_log

        # nan_indx = np.where(np.isnan(resid_log))[0]
        # if len(nan_indx) > 0:
        #    resid_log[nan_indx] = 0

        # Interpolate template to same grid as data
        temp_log = q3dmath.interptemp(lambda_log, np.log(template_wave.T[0]),
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
        #ineg = np.where(continuum < 0)
        continuum[continuum < 0] = 0

        ppxfcontinuum_log = pp.bestfit
        cinterp = interpolate.interp1d(lambda_log, ppxfcontinuum_log,
                                       kind='cubic', fill_value='extrapolate')

        ppxfcont_resid = cinterp(np.log(wave))
        continuum += ppxfcont_resid

    elif refit == 'questfit':

        resid = flux - continuum
        argscontfit_use = kwargs['args_questfit']
        cont_resid, ct_coeff, zstar = questfit(wave, resid, weight, index, 
                                               logfile=logfile, quiet=quiet, z=zstar, 
                                               fluxunit=fluxunit, **argscontfit_use)

        from . import plot
        from matplotlib import pyplot as plt
        initdatdict = argscontfit_use.copy()
        initdatdict['label'] = 'miritest'
        initdatdict['plotMIR'] = True
        plot.plotquest(wave, resid, cont_resid, ct_coeff, initdatdict)
        plt.show()

        continuum += cont_resid
        ct_coeff['qso_host'] = result.params

    return continuum, ct_coeff, zstar


def questfit(wave: np.ndarray, 
             flux: np.ndarray, 
             weight: np.ndarray, 
             index: np.ndarray, 
             logfile: Optional[str]=None,
             quiet: bool=True, 
             z: float=0.,
             fluxunit: Optional[str]=None,
             tempdir: Optional[str]=None,
             *, config_file: str) -> tuple[np.ndarray, dict[str, Any], float]:
    '''
    Fit the MIR continuum. Calls :py:func:`~q3dfit.contfit.questfitfcn` to fit the
    continuum using the templates specified in the configuration file.

    Parameters
    -----
    wave
        Wavelengths
    flux
        Fluxes
    weight
        Weights
    index
        Points used in the fit.
    quiet
        Optional. Suppress output? Default is True.
    z
        Optional. Redshift of the source, for redshifting the templates. 
        Default is 0.
    fluxunit
        Optional. Units of the flux, as defined by :py:class:`~q3dfit.readcube.Cube`.
        Default is None. The templates are assumed to be in f_nu. If this
        unit contains 'micron' or 'Angstrom', the flux is assumed to be in f_lambda.
        The templates are then converted to f_lambda. If None, the templates are not
        converted.
    tempdir
        Optional. Directory containing the templates, to search if default
        directory does not contain template. Default is None.
    config_file
        Configuration file for the fit.

    Returns
    -------
    numpy.ndarray
        Best fit continuum model.
    dict[str, Any]
        Best fit parameters. Key 'MIRparams' is a :py:class:`~lmfit.Parameters` object
        containing the best fit parameters. Key 'comp_best_fit' is a dictionary
        containing with key/value pairs of the component names and their values
        evaluated at the best fit parameters.
    z
        Same as input z.
    '''

    # models dictionary holds extinction, absorption models
    emcomps = dict()
    # template dictionary holds templates, blackbodies, powerlaws
    allcomps = dict()

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
            #if not 'poly' in i:
            model_temp_template, param_temp_template = \
                questfitfcn.\
                    set_up_fit_model_scale([np.float64(model_parameters[1])],
                                           [np.float64(model_parameters[2])],
                                           name_model, name_model) #,
                                            #maxamp=1.05*max(flux[index]) )
            # # ... and if it is!
            # Not presently implmented
            # else:
            #     # constrain amplitude
            #     minamp = np.float64(model_parameters[1]) / 1.25
            #     maxamp = np.float64(model_parameters[1]) * 1.25
            #     model_temp_template, param_temp_template = \
            #         questfitfcn.set_up_fit_model_scale_withpoly(
            #             [np.float64(model_parameters[1])],
            #             [np.float64(model_parameters[2])],
            #             name_model, name_model, minamp=minamp, maxamp=maxamp)

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
            if not os.path.exists(p) and tempdir is not None:
                temp_model = np.load(tempdir+allcomps[i])
            else:
                raise InitializationError('Cannot locate template specified in config file.')
        # Check to see if we are using global extinction, where the total
        temp_wave=temp_model['WAVE']*(1.+z)
        temp_value=temp_model['FLUX']

        temp_value_rebin = \
            q3dmath.interp_lis(wave, temp_wave, temp_value)
        allcomps[i] = temp_value_rebin

    # conversion from f_nu to f_lambda is f_lambda = f_nu x c/lambda^2
    # [1 Jy = 1e-10 erg/s/cm^2/um [/sr] ???]
    # about 1.2 for wave=5 [micron]
    #c_scale =  constants.c * u.Unit('m').to('micron') / wave**2 * \
    #    1e-23 * 1e10
    # the exact value doesn't matter, because we normalize the templates
    c_scale =  1. / wave**2

    # loop over emission template dictionary, load them in and resample.
    for i in emcomps.keys():
        if 'sifrom' in emcomps[i]:
            tempdir = questfit_templates.silicatemodels
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
            q3dmath.interp_lis(wave, temp_wave, temp_value)
        allcomps[i] = temp_value_rebin
        # conversion from f_nu to f_lambda: f_lambda = f_nu x c/lambda^2
        # assume that presence of micron or Angstrom in fluxunit means that the flux is
        # f_lambda
        if fluxunit is not None:
            if 'micron' in fluxunit or 'Angstrom' in fluxunit:
                allcomps[i] *= c_scale

        allcomps[i] = allcomps[i]/allcomps[i].max()

    # Add in wavelength and units flags
    allcomps['wave'] = np.float64(wave)
    allcomps['fluxunit'] = fluxunit

    # produce copy of components dictionary with index applied
    flux_cut = flux[index]
    allcomps_cut = copy.deepcopy(allcomps)
    for el in allcomps.keys():
        if not ('fluxunit' in el):
            allcomps_cut[el] = allcomps_cut[el][index]

    # Get ready for fit
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

    q3dutil.write_msg(result.fit_report(), file=logfile, quiet=quiet)

    # use allcomps rather than allcomps_cut to evaluate
    # over all wavelengths within fit range (not just [index])
    best_fit = result.eval(**allcomps)
    comp_best_fit = result.eval_components(**allcomps)
    # print(result.best_values)

    # apply extinction and ice absorption to total best fit
    if global_extinction:
        for el in comp_best_fit.keys():
            if el != 'global_ext' and el != 'global_ice':
                comp_best_fit[el] *= comp_best_fit['global_ext']
                comp_best_fit[el] *= comp_best_fit['global_ice']

    ct_coeff = {'MIRparams': result.params,
                'comp_best_fit': comp_best_fit}

    return best_fit, ct_coeff, z


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
            q3dmath.interp_lis(lam_cut[ind_opt], wave_fe, 
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
            q3dmath.interp_lis(lam_cut[ind_UV], wave_fe_UV, 
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
        F_Fe_opt_all = q3dmath.interp_lis(lam, wave_fe, F_Fe_opt_conv)
        models_dictionary_B['temp_fe_opt'] = F_Fe_opt_all
    if np.sum(ind_UV)>0:
        F_Fe_UV_all = q3dmath.interp_lis(lam, wave_fe_UV, F_Fe_UV_conv)
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


def Fe_flux_balmer(Fe_FWHM: float, 
                   zstar: float, 
                   specConv: object) \
                    -> tuple[np.ndarray, np.ndarray]:
    '''
    Loads the optical FeII template, redshifts it,
    convolves it with the intrinsic broadening, and convolves it 
    with the instrumental resolution.

    Parameters
    ----------
    Fe_FWHM
        Defines the smoothing of the FeII template in km/s. FWHM of the
        Gaussian smoothing that will be applied. Needs to be >900 km/s
        (FWHM of template).
    zstar
        Redshift of the stellar continuum.
    specConv
        Instance of :py:class:`~q3dfit.spectConvol.spectConvol` specifying 
        the instrumental spectral resolution convolution.

    Returns
    -------
    np.ndarray
        Convolved FeII template.
    np.ndarray
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

    #config_file = readcf(initdat['argscontfit']['config_file'])
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


def readcf(filename: str) -> dict:
    '''
    Read in the configuration file for fitting with questfit.
    
    Parameters
    ----------
    filename
        Name of, and path, to the configuration file.
    
    Returns
    -------
    dict
        Dictionary with the configuration file. Keys are the names of the
        components to be fitted, values are lists with the parameters for
        the components.
    '''
    cf = np.loadtxt(filename, dtype=np.str_, comments="#")

    # Cycle through rows
    init_questfit = dict()
    for i in cf:
        if i[0] == 'template' or i[0] == 'powerlaw' or i[0] == 'blackbody' \
        or i[0] == 'template_poly':
            # populate initilization dictionary with
            # col 0: filename (if nessesary; path hardcoded)
            # col 1: lower wavelength limit or normalization factor
            # col 2: upper wavelength limit or fix/free parameter (0 or 1) for normalization
            # col 3: name of ext. curve or ice feature
            # col 4: initial guess for Av
            # col 5: fix/free parameter (0/1) for Av
            # col 6: S,M = screen or mixed extinction
            # col 7: initial guess for BB temperature or powerlaw index
            # col 8: fix/free parameter (0/1) for BB temperature or powerlaw index
            # col 9: ice name model
            # col 10: intial guess for ice absorption tau
            # col 11: fix/free parameter (0/1) for tau
            init_questfit[i[0]+'_'+i[1]+'_'+i[4]+'_'+i[10]] = i[1:]

        if i[0] == 'absorption' or i[0] == 'extinction':
            #init_questfit[i[0]+'_'+i[1]+'_'+i[4]+'_'+i[10]] = i[1:]
            init_questfit[i[4]] = [i[1], i[0]]

        if i[0] == 'source':
            init_questfit['source'] = i[1:]

    return init_questfit
