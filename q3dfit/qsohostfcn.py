from __future__ import annotations

from typing import Optional

import lmfit
import numpy as np
from numpy.typing import ArrayLike

from . import lineinit, q3dutil
from q3dfit.exceptions import InitializationError


def qso_mult_exp(wave: ArrayLike,
                 qsotemplate: np.ndarray,
                 a: float, b: float, c: float, d: float, e: float,
                 f: float, g: float, h: float, i: float) -> np.ndarray:
    '''
    Model exponentials for qso template multiplier

    Parameters
    -----
    wave
        1-D array of wavelengths
    qsotemplate
        1-D array of the quasar spectrum to be fit
    a
        scale factor for constant term
    b
        scale factor for exponential decay
    c   
        exponential decay constant
    d
        scale factor for exponential decay in reverse
    e
        exponential decay constant in reverse
    f
        scale factor for exponential rise
    g
        exponential rise constant
    h
        scale factor for exponential rise in reverse
    i
        exponential rise constant in reverse

    Returns
    -------
    np.ndarray
        Multiplicative factor times the quasar spectrum
    '''
    x = np.linspace(0., 1., len(wave))
    x2 = np.linspace(1., 0., len(wave))
    multiplier = a + b * (np.exp(-c*x)) + d*(np.exp(-e*x2)) + \
        f*(1.-np.exp(-g*x)) + h*(1.-np.exp(-i*x2))
    return multiplier*qsotemplate


def setup_qso_mult_exp(p: np.ndarray) -> tuple[lmfit.Model, lmfit.Parameters]:
    '''
    Set up model exponentials for qso template multiplier

    Parameters
    ----------
    p
        list of initial guesses for the exponential fit parameters a-i

    Returns
    -------
    lmfit.Model
        lmfit model for the qso template multiplier
    lmfit.Parameters
        lmfit parameters for the qso template multiplier
    '''

    model_name = 'qso_mult_exp_'
    qsotemplate_x_exp = \
        lmfit.Model(qso_mult_exp,
                    independent_vars=['wave', 'qsotemplate'],
                    prefix=model_name)
    qso_mult_exp_pars = qsotemplate_x_exp.make_params()
    
    for counter, i in enumerate(['a', 'b', 'c', 'd', 'e',
                                 'f', 'g', 'h', 'i']):
        if not np.isnan(p[counter]):
            qso_mult_exp_pars[model_name+i].set(value=p[counter],
                                                min=np.finfo(float).eps)
        else:
            qso_mult_exp_pars[model_name+i].set(value=np.finfo(float).eps,
                                                vary=False)

    return qsotemplate_x_exp, qso_mult_exp_pars


def qso_mult_leg(wave: ArrayLike,
                 qsotemplate: np.ndarray,
                 i: float, j: float, k: float, l: float, m: float,
                 n: float, o: float, p: float, q: float, r: float) -> np.ndarray:
    '''
    Model legendre polys for qso template multiplier

    Parameters
    ----------
    wave
        1-D array of wavelengths
    qsotemplate
        1-D array of the quasar spectrum to be fit
    i,j,k,l,m,n,o,p,q,r
        scale factors for 1-10 order legendre polynomials. 0th order
        is the constant term and is set to 0.

    Returns
    -------
    np.ndarray
        Multiplicative factor times the quasar spectrum
    '''
    x = np.linspace(-1., 1., len(wave))
    multiplier = \
        np.polynomial.legendre.legval(x, [0., i, j, k, l, m, n, o, p, q, r])
    return multiplier*qsotemplate


def setup_qso_mult_leg(p: np.ndarray) -> tuple[lmfit.Model, lmfit.Parameters]:
    '''
    Set up model legendre polys for qso template multiplier

    Parameters
    ----------
    p
        list of initial guesses for the exponential fit parameters a-i

    Returns
    -------
    lmfit.Model
        lmfit model for the qso template multiplier
    lmfit.Parameters
        lmfit parameters for the qso template multiplier
    '''
    model_name = "qso_mult_leg_"
    qsotemplate_x_leg = \
        lmfit.Model(qso_mult_leg,
                    independent_vars=['wave', 'qsotemplate'],
                    prefix=model_name)
    qso_mult_leg_pars = qsotemplate_x_leg.make_params()

    for counter, i in enumerate(['i', 'j', 'k', 'l', 'm',
                                 'n', 'o', 'p', 'q', 'r']):
        if not np.isnan(p[counter]):
            qso_mult_leg_pars[model_name+i].set(value=p[counter],
                                                min=np.finfo(float).eps)
        else:
            qso_mult_leg_pars[model_name+i].set(value=np.finfo(float).eps,
                                                vary=False)

    return qsotemplate_x_leg, qso_mult_leg_pars


def stars_add_leg(wave: ArrayLike,
                  i: float, j: float, k: float, l: float, m: float,
                  n: float, o: float, p: float, q: float, r: float) -> np.ndarray:
    '''
    Model legendre for additive starlight continuum

    Parameters
    ----------
    wave
        1-D array of wavelengths
    i,j,k,l,m,n,o,p,q,r
        scale factors for 1-10 order legendre polynomials. 0th order
        is the constant term and is set to 0.

    Returns
    -------
    np.ndarray
        Polynomial model for the additive starlight continuum
    '''

    x = np.linspace(-1., 1., len(wave))
    starlight = \
        np.polynomial.legendre.legval(x, [0., i, j, k, l, m, n, o, p, q, r])
    return starlight


def setup_stars_add_leg(p: np.ndarray) -> tuple[lmfit.Model, lmfit.Parameters]:
    '''
    Set up model legendre for additive starlight continuum

    Parameters
    ----------
    p
        list of initial guess for the legendre polynomial

    Returns
    -------
    lmfit.Model
        lmfit model for the additive starlight continuum
    lmfit.Parameters
        lmfit parameters for the additive starlight continuum    
    '''
    model_name = "stars_add_leg_"
    stars = lmfit.Model(stars_add_leg, independent_vars=['wave'],
                        prefix=model_name)
    stars_add_leg_pars = stars.make_params()

    for counter, i in enumerate(['i', 'j', 'k', 'l', 'm',
                                 'n', 'o', 'p', 'q', 'r']):
        if not np.isnan(p[counter]):
            stars_add_leg_pars[model_name+i].set(value=p[counter],
                                                 min=np.finfo(float).eps)
        else:
            stars_add_leg_pars[model_name+i].set(value=np.finfo(float).eps,
                                                 vary=False)

    return stars, stars_add_leg_pars


def stars_add_exp(wave: ArrayLike,
                  a: float, b: float, c: float, d: float, e: float,
                  f: float, g: float, h: float, i: float) -> np.ndarray:
    '''
    Model exponentials for additive starlight continuum

    Parameters
    ----------
    wave
        1-D array of the wavelength to be fit
    a
        scale factor for constant term
    b
        scale factor for exponential decay
    c   
        exponential decay constant
    d
        scale factor for exponential decay in reverse
    e
        exponential decay constant in reverse
    f
        scale factor for exponential rise
    g
        exponential rise constant
    h
        scale factor for exponential rise in reverse
    i
        exponential rise constant in reverse

    Returns
    -------
    np.ndarray
        Exponential model for the additive starlight continuum
    '''
    x = np.linspace(0., 1., len(wave))
    x2 = np.linspace(1., 0., len(wave))
    starlight = a + b*(np.exp(-c*x)) + d*(np.exp(-e*x2)) + f*(1.-np.exp(-g*x)) + \
        h*(1.-np.exp(-i*x2))
    return starlight


def setup_stars_add_exp(p: np.ndarray) -> tuple[lmfit.Model, lmfit.Parameters]:
    '''
    Set up model exponentials for additive starlight continuum

    Parameters
    ----------
    p
        list of initial guess for the legendre polynomial fit

    Returns
    -------
    lmfit.Model
        lmfit model for the additive starlight continuum
    lmfit.Parameters
        lmfit parameters for the additive starlight continuum
    '''
    model_name = 'stars_add_exp_'
    stars = lmfit.Model(stars_add_exp, independent_vars=['wave'],
                        prefix=model_name)
    stars_add_exp_pars = stars.make_params()
    stars_add_exp_pars[model_name+'a'].set(value=p[0], min=np.finfo(float).eps)
    stars_add_exp_pars[model_name+'b'].set(value=p[1], min=np.finfo(float).eps)
    stars_add_exp_pars[model_name+'c'].set(value=p[2], min=np.finfo(float).eps)
    stars_add_exp_pars[model_name+'d'].set(value=p[3], min=np.finfo(float).eps)
    stars_add_exp_pars[model_name+'e'].set(value=p[4], min=np.finfo(float).eps)
    stars_add_exp_pars[model_name+'f'].set(value=p[5], min=np.finfo(float).eps)
    stars_add_exp_pars[model_name+'g'].set(value=p[6], min=np.finfo(float).eps)
    stars_add_exp_pars[model_name+'h'].set(value=p[7], min=np.finfo(float).eps)
    stars_add_exp_pars[model_name+'i'].set(value=p[8], min=np.finfo(float).eps)

    return stars, stars_add_exp_pars


def qsohostfcn(wave: np.ndarray,
               params_fit: Optional[dict]=None,
               qsoflux: Optional[np.ndarray]=None,
               qsoonly: bool=False,
               qsoord: Optional[int]=None,
               hostonly: bool=False,
               hostord: Optional[int]=None,
               blronly: bool=False,
               blrpar: Optional[ArrayLike]=None,
               medflux: Optional[float]=None) -> tuple[Optional[np.ndarray], 
                                                       Optional[lmfit.Model], 
                                                       Optional[lmfit.Parameters]]:
    '''
    Set up or evaluate the model for the QSO and host galaxy featureless 
    continuum fit.

    Parameters
    ----------
    wave
        1-D array of wavelengths
    params_fit
        Optional. Dictionary of parameters to evaluate the model. If None,
        the function returns the lmfit model object and parameter object for fitting.
        If not None, the function returns the model evaluated at the
        parameters in params_fit.
    qsoflux
        Optional. 1-D array of the quasar template flux. Only needed if
        params_fit is not None.
    qsoonly
        Optional. Fit/evaluate only the QSO component. Default is False.
    qsoord
        Optional. Order of the Legendre polynomial for the QSO template multiplier.
        Default is None, which means no Legendre polynomial is used. The maximum
        order is 10.
    hostonly
        Optional. Fit/evaluate only the host galaxy component. Default is False.
    hostord
        Optional. Order of the Legendre polynomial for the host galaxy template
        multiplier. Default is None, which means no Legendre polynomial is used.
        The maximum order is 10.
    blronly
        Optional. Fit/evaluate only the scattered-light broad line region component. 
        Default is False.
    blrpar
        Optional. Array of parameters for the broad line region model. Only needed
        if scattered light broad line region component is included. Default is None.
        Must be in the form [amplitude, center, sigma, amplitude, center, sigma, ...],
        where each set of three parameters is for a single Gaussian component.
        The center is not varied in the fit, and the bounds on the sigma are set to
        2000 and 6000 km/s.
    medflux
        Optional. Estimate for the continuum level for setting initial guesses.
        Default is None, which sets the continuum level to 1.

    Returns
    -------
    tuple
        If params_fit is not None, returns a tuple of the evaluated model,
        None, None. If params_fit is None, returns a tuple of None, the lmfit
        model object, and the lmfit parameters object.
    '''
    # maximum model legendre polynomial order
    legordmax = 10
    # estimate for continuum level
    if medflux is None:
        medflux=1.

    # Additive starlight component:
    if not qsoonly and not blronly:
        # Terms with exponentials
        initvals = np.concatenate((np.array([medflux/2.]),np.zeros(8)))
        stars_add = setup_stars_add_exp(initvals)
        ymod = stars_add[0]
        params = stars_add[1]
        # optional legendre polynomials up to order ordmax
        if hostord is not None:
            if hostord <= legordmax and hostord > 0:
                initvals = np.zeros(legordmax)
                for ind in np.arange(legordmax, hostord, -1):
                    initvals[ind-1] = np.nan
                stars_add = setup_stars_add_leg(initvals)
                ymod += stars_add[0]
                params += stars_add[1]
            else:
                raise InitializationError('Order of starlight additive ' +
                                          'Legendre polynomial [hostord] ' +
                                          'must be 0 < hostord <= {legordmax}')

    # Scaled QSO component
    if not hostonly and not blronly:
        # Terms with exponentials
        medfluxuse = medflux/2.
        if qsoonly:
            medfluxuse *= 2.
        initvals = np.concatenate((np.array([medfluxuse]),np.zeros(8)))
        qso_mult = setup_qso_mult_exp(initvals)
        if 'ymod' not in vars():
            ymod = qso_mult[0] 
            params = qso_mult[1]
        else:
            ymod += qso_mult[0] # type: ignore
            params += qso_mult[1] # type: ignore
        # optional legendre polynomials
        if qsoord is not None:
            if qsoord <= legordmax and qsoord > 0:
                initvals = np.zeros(legordmax)
                for ind in np.arange(legordmax, qsoord, -1):
                    initvals[ind-1] = np.nan
                qso_mult = setup_qso_mult_leg(initvals)
                ymod += qso_mult[0]
                params += qso_mult[1]
            else:
                raise InitializationError('Order of multiplicative ' +
                                          'Legendre polynomial for scaling ' +
                                          'QSO template [qsoord] ' +
                                          'must be 0 < qsoord <= {legordmax}')

    # BLR model
    if not hostonly and blrpar is not None:

        counter = 0
        for i in np.arange(0, len(blrpar)/3.):
            lmline = q3dutil.lmlabel(f'{blrpar[counter+1]:g}')
            gaussian_name = f'g_{lmline.lmlabel}'
            #gaussian_model = lmfit.models.GaussianModel(prefix=gaussian_name)
            #gaussian_model_parameters = gaussian_model.make_params()
            #gaussian_model_parameters\
            #    [gaussian_name+'amplitude'].set(value=blrpar[counter],
            #                                    min=0.)
            #gaussian_model_parameters\
            #    [gaussian_name+'center'].set(value=blrpar[counter + 1],
            #                                 vary=False)
            #gaussian_model_parameters\
            #    [gaussian_name+'sigma'].set(value=blrpar[counter + 2],
            #                                min=2000. / c.to('km/s').value *
            #                                blrpar[counter + 1],
            #                                max=6000. / c.to('km/s').value *
            #                                blrpar[counter + 1])

            gaussian_model = lmfit.Model(lineinit.manygauss, prefix=gaussian_name, 
                                         SPECRES=None)
            gaussian_model_parameters = gaussian_model.make_params()
            gaussian_model_parameters\
                [gaussian_name+'flx'].set(value=blrpar[counter],
                                          min=np.finfo(float).eps)
            gaussian_model_parameters\
                [gaussian_name+'cwv'].set(value=blrpar[counter + 1],
                                          vary=False)
            gaussian_model_parameters\
                [gaussian_name+'sig'].set(value=blrpar[counter + 2],
                                          min=2000., max=6000.)
            if blronly and i == 0:
                ymod = gaussian_model
                params = gaussian_model_parameters
            else:
                ymod += gaussian_model # type: ignore
                params += gaussian_model_parameters # type: ignore
            counter += 3

    # Option to evaulate model for plotting, else return the lmfit model
    # and parameters for fitting in fitqsohost
    if params_fit is not None:
        continuum = ymod.eval(params_fit, wave=wave, qsotemplate=qsoflux,
                              x=wave)
        return continuum, None, None
    else:
        return None, ymod, params
