from __future__ import annotations

from typing import Literal, Optional, Tuple

import numpy as np
from numpy.typing import ArrayLike

from astropy import units as u
from astropy import constants as const
import lmfit


def blackbody(wave: np.ndarray,
              a: float,
              T: float,
              fluxunit: Optional[str]=None) -> np.ndarray:
    '''
    Blackbody model function

    Parameters
    -----
    wave
        1-D array of the wavelength to be fit, in microns.
    a
        Scale factor for blackbody model after it is normalized to its maximum value.
    T
        Temperature of the blackbody in Kelvin.

    returns
    -------
    BB_model: array
    '''
    # Use astropy blackbody model
    # Blam = BlackBody(temperature=T*u.K, scale=1.*u.Unit('erg/cm^2/micron/s/sr'))
    # Blamval = Blam(wave*u.micron).value
 
    # hck in cgs units
    hck = const.h.to('cm2 g/s').value*const.c.to('cm/s').value/\
        const.k_B.to('cm2 g / (s2 K)').value
    # flux density in erg/cm^2/s/micron
    # convert wave from microns to cm
    BB_model = (wave*1e-4)**-4*(np.exp(hck/(wave*1e-4)/T)-1)**-1 / wave

    # if data is in f_lambda, convert to f_nu
    # exact scale doesn't matter since we normalize by the maximum value
    if fluxunit is not None:
        if 'micron' in fluxunit or 'Angstrom' in fluxunit:
            #c_scale = c * u.Unit('m').to('micron') /(wave)**2 *1e-23
            c_scale = 1. / wave**2            
            BB_model /= c_scale
    
    BB_model /= BB_model.max()

    return a*BB_model


def set_up_fit_blackbody_model(p: list,
                               p_fixfree: list,
                               name: str) -> Tuple[lmfit.Model, lmfit.Parameters]:
    '''Function defined to set up fitting blackbody_model within lmfit

        Parameters
        -----
        p
            List of initial guesses for the normalization and T of the blackbody model fit.
            Limits on the temperature are set to 50-3000 K. Lower limit on the normalization
            is the minimum positive float value.
        p_fixfree
            List of fix/free values (0/1) for the blackbody model parameters.
        name
            Name of the blackbody model, for labeling.

        Returns
        -------
        blackbody_model_model: lmfit model
        blackbody_model_paramters: lmfit model parameters
        '''


    model_name = name
    blackbody_model = \
        lmfit.Model(blackbody, independent_vars=['wave', 'fluxunit'], prefix=model_name)
    blackbody_model_parameters = blackbody_model.make_params()
    blackbody_model_parameters[model_name+'a'].\
        set(value=p[0], min=np.finfo(float).eps, vary=p_fixfree[0])
    blackbody_model_parameters[model_name+'T'].\
        set(value=p[1], min=50., max=3000., vary=p_fixfree[1])

    return blackbody_model, blackbody_model_parameters


""" def modelmultpoly(template_0: ,
                  wave, amp, multpolyA, multpolyB, multpolyC):

    wave_new = (wave-min(wave))/(max(wave)-min(wave)) * 2 - 1 # mapped onto range [-1,1]
    wave_new = 10**wave_new
    polymodel = template_0 * (multpolyA + multpolyB * wave_new) # + multpolyC * wave_new**2)
    return amp * polymodel/polymodel.max()


def set_up_fit_model_scale_withpoly(p,p_fixfree,model_name,model, minamp=0., 
                                    maxamp=None):

    '''Function defined to set up fitting model_scale within lmfit

        Parameters
        -----
        p: list
        list of initial guess for the model_scale fit



        returns
        -------
        scale_model_model: lmfit model
        scale_model_paramters: lmfit model parameters
        '''

    model_scale_model = lmfit.Model(modelmultpoly, independent_vars=[model, 'wave'], name=model_name, prefix=model_name+'_')
    model_scale_parameters = model_scale_model.make_params()

    if maxamp is not None:
        model_scale_parameters[model+'_amp'].set(value=p[0],min=minamp, max=maxamp,vary=p_fixfree[0])#,min=0.
    else:
        model_scale_parameters[model+'_amp'].set(value=p[0],min=minamp,vary=p_fixfree[0])#,min=0.

    model_scale_parameters[model+'_multpolyA'].set(value=1.,min=0., max=2., vary=p_fixfree[0])#,min=0.
    model_scale_parameters[model+'_multpolyB'].set(value=1.,min=0., max=2., vary=p_fixfree[0])#,min=0.
    model_scale_parameters[model+'_multpolyC'].set(value=0.1,min=0., max=2., vary=p_fixfree[0])#,min=0.

    return model_scale_model,model_scale_parameters
 """


def powerlaw(wave: np.ndarray,
             a: float,
             b: float) -> np.ndarray:
    '''
    Powerlaw model

    Parameters
    ----------
    wave
        1-D array of the wavelength to be fit.
    a
        Scale factor the powerlaw model.
    b
        Exponent for powerlaw.

    returns
    -------
    np.ndarray
    '''
    powerlaw_model = wave**b
    return a*powerlaw_model


def set_up_fit_powerlaw_model(p: ArrayLike,
                              p_fixfree: ArrayLike,
                              name: str) -> Tuple[lmfit.Model, lmfit.Parameters]:
    '''
    Set up fitting powerlaw_model within lmfit.

    Parameters
    ----------
    p
        List of initial guesses for the normalization and exponent of the powerlaw model fit.
        The lower limit on the normalization is the minimum positive float value.
    p_fixfree
        List of fix/free values (0/1) for the powerlaw model parameters.

    Returns
    -------
    lmfit.Model
    lmfit.Parameters
    '''

    model_name = name
    powerlaw_model = \
        lmfit.Model(powerlaw, independent_vars=['wave'],
                    prefix=model_name)
    powerlaw_model_parameters = powerlaw_model.make_params()
    powerlaw_model_parameters[model_name+'a'].\
        set(value=p[0], min=np.finfo(float).eps, vary=p_fixfree[0])
    powerlaw_model_parameters[model_name+'b'].\
        set(value=p[1], vary=p_fixfree[1])

    return powerlaw_model, powerlaw_model_parameters


def model_scale(model: np.ndarray,
                a: float) -> np.ndarray:
    '''
    Scale a model by a factor a.

    Parameters
    -----
    model
        The model to be scaled.
    a
        Scale factor.

    Returns
    -------
    np.ndarray
    '''
    model_scale = model*a
    return model_scale


def set_up_fit_model_scale(p: ArrayLike,
                           p_fixfree: ArrayLike,
                           model_name: str,
                           model: str,
                           maxamp: Optional[float]=None) \
                            -> Tuple[lmfit.Model, lmfit.Parameters]:
    '''
    Set up fitting model_scale within lmfit

    Parameters
    ----------
    p
        List containing initial guess for the scale factor. The lower limit on the scale 
        factor is the minimum positive float value. If maxamp is not None, the upper limit 
        is set to maxamp.
    p_fixfree
        List containing fix/free value (0/1) for the scale factor.
    model_name
        Name of the scaled model.
    model
        Name of the model to be scaled.
    maxamp
        Optional. Maximum value for the scale factor. If None, the maximum value is not set.

    Returns
    -------
    lmfit.Model
    lmfit.Parameters
    '''
    exp = "%s_amp*%s" % (model, model)
    model_scale_model = \
        lmfit.models.ExpressionModel(exp,independent_vars=[model],name=model_name)
    model_scale_parameters = model_scale_model.make_params()

    if maxamp is not None:
        model_scale_parameters[model+'_amp'].set(value=p[0],
                                                 min=np.finfo(float).eps,
                                                 max=maxamp,
                                                 vary=p_fixfree[0])
    else:
        model_scale_parameters[model+'_amp'].set(value=p[0],
                                                 min=np.finfo(float).eps,
                                                 vary=p_fixfree[0])

    return model_scale_model, model_scale_parameters


def set_up_absorption(p: ArrayLike,
                      p_fixfree: ArrayLike,
                      model_name: str,
                      abs_model: str) -> Tuple[lmfit.Model, lmfit.Parameters]:
    '''
    Set up fitting ice absorption model within lmfit.

    Parameters
    -----
    p
        List containing initial guess for the tau of the absorption model.
        The lower limit on tau is the minimum positive float value, and the upper limit is 10.
    p_fixfree
        List containing fix/free value (0/1) for tau.
    model_name
        Name for the output model.
    abs_model
        Name of the absorption model with values of tau.

    Returns
    -------
    lmfit.Model
    lmfit.Parameters
    '''

    exp = "exp(-1.*%s_tau*%s)" % (model_name, abs_model)
    model = lmfit.models.ExpressionModel(exp, independent_vars=[abs_model], 
                                         name = model_name)
    abs_model_parameters = model.make_params()
    abs_model_parameters[model_name+'_tau'].\
        set(value=p[0], min=np.finfo(float).eps, max=10., vary=p_fixfree[0])

    return model,abs_model_parameters


def set_up_fit_extinction(p: ArrayLike,
                          p_fixfree: ArrayLike,
                          model_name: str,
                          extinction_model: str,
                          mixed_or_screen: Literal['M', 'S']) \
                            -> Tuple[lmfit.Model, lmfit.Parameters]:
    '''
    Set up extinction fit.

    Parameters
    ----------
    p
        List containing initial guess for A_V. The lower limit on A_V is the minimum 
        positive float value, and the upper limit is 100.
    p_fixfree
        List containing fix/free value (0/1) for A_V.
    model_name
        Name for extincted model.
    extinction_model
        Name of extinction curve containing values A_lambda.
    mixed_or_screen
        'M' for mixed model, 'S' for screen model.

    Returns
    -------
    lmfit.Model
    lmfit.Parameters
    '''

    if mixed_or_screen == 'M':
        exp = "1. - exp(-0.4*%s_Av*log10(%s))/(0.4*%s_Av*log10(%s))" \
            % (model_name, extinction_model, model_name, extinction_model)

    if mixed_or_screen == 'S':
        exp = "power(10.,-0.4*%s_Av*log10(%s))" \
            % (model_name, extinction_model)

    model_extinction = \
        lmfit.models.ExpressionModel(exp, independent_vars=[extinction_model],
            name=model_name)

    model_extinction_parameters = model_extinction.make_params()
    print(p_fixfree[0])
    if mixed_or_screen == 'M':
        model_extinction_parameters[model_name+'_Av'].\
            set(value=p[0], min=np.finfo(float).eps, max=100., vary=p_fixfree[0])
    if mixed_or_screen == 'S':
        model_extinction_parameters[model_name+'_Av'].\
            set(value=p[0], min=np.finfo(float).eps, max=100., vary=p_fixfree[0])

    return model_extinction, model_extinction_parameters
