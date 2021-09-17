import numpy as np
import lmfit
# import scipy
from astropy.constants import c


def qsotemplate_model_exponential(wave, qso_model, a, b, c, d, e, f, g, h):

    '''Function defined for fitting a qsotemplate

        Parameters
        -----
        qsotemplate: array
        1-D array of the quasar spectrum to be fit
        a,c,c,d,e,f,g,h: floats
        scale factors for the qsotemplate

        returns
        -------
        qsotemplate_model: array
        '''

    x = np.linspace(0, 1, len(wave))
    x2 = np.linspace(1, 0, len(wave))
    qsotemplate_for_fit = b * (np.exp(-a*x)) + d*(np.exp(-c*x2)) + \
        f*(1-np.exp(-e*x)) + h*(1-np.exp(-g*x2))
    return qsotemplate_for_fit*qso_model


def set_up_fit_qso_exponential_scale_model(p):
    '''Function defined to set up fitting legendre polynomial within lmfit

        Parameters
        -----
        p: list
        list of initial guess for the legendre polynomial fit



        returns
        -------
        qso_continuum_exponential_scale_model: lmfit model
        qso_continuum_exponential_scale_paramters: lmfit model parameters
        '''


    model_name = 'exp_scale_'
    qso_exponential_scale_model = \
        lmfit.Model(qsotemplate_model_exponential,
                    independent_vars=['wave', 'qso_model'], prefix=model_name)
    exponential_model_parameters = qso_exponential_scale_model.make_params()
    exponential_model_parameters\
        [model_name+'a'].set(value=p[0], min=0.)
    exponential_model_parameters\
        [model_name+'b'].set(value=p[1], min=0.)
    exponential_model_parameters\
        [model_name+'c'].set(value=p[2], min=0.)
    exponential_model_parameters\
        [model_name+'d'].set(value=p[3], min=0.)
    exponential_model_parameters\
        [model_name+'e'].set(value=p[4], min=0.)
    exponential_model_parameters\
        [model_name+'f'].set(value=p[5], min=0.)
    exponential_model_parameters\
        [model_name+'g'].set(value=p[6], min=0.)
    exponential_model_parameters\
        [model_name+'h'].set(value=p[7], min=0.)

    return qso_exponential_scale_model, exponential_model_parameters


def qsotemplate_scale_legendre(wave, qso_model, i, j, k):

    '''Function defined for fitting a legendre polynomial to continuum

        Parameters
        -----
        wave: array
        1-D array of wavelengths
        i,j,k: floats
        scale factors for first, second and third order legendre polynomials

        returns
        -------
        legendre_poly: array
        '''

    x = np.linspace(0, 1, len(wave))
    legendre_poly = np.polynomial.legendre.legval(x, [i, j, k])
    return qso_model*legendre_poly


def set_up_fit_qso_scale_legendre(p):
    '''Function defined to set up fitting legendre polynomial within lmfit

        Parameters
        -----
        p: list
        list of initial guess for the legendre polynomial fit



        returns
        -------
        qso_continuum_legendre_model: lmfit model
        qso_continuum_legendre_model_paramters: lmfit model parameters
        '''

    model_name = "legendre_scale_"
    qso_continuum_legendre_model = \
        lmfit.Model(qsotemplate_scale_legendre,
                    independent_vars=['wave', 'qso_model'],
                    prefix=model_name)
    qso_continuum_legendre_model_paramters = \
        qso_continuum_legendre_model.make_params()
    qso_continuum_legendre_model_paramters\
        [model_name+'i'].set(value=p[0], min=0.)
    qso_continuum_legendre_model_paramters\
        [model_name+'j'].set(value=p[1], min=0.)
    qso_continuum_legendre_model_paramters\
        [model_name+'k'].set(value=p[2], min=0.)

    return qso_continuum_legendre_model, qso_continuum_legendre_model_paramters


def qso_continuum_legendre(wave, i, j, k, l, m):

    '''Function defined for fitting a legendre polynomial to continuum

        Parameters
        -----
        wave: array
        1-D array of wavelengths
        i,j,k: floats
        scale factors for first, second and third order legendre polynomials

        returns
        -------
        legendre_poly: array
        '''

    x = np.linspace(-1, 1, len(wave))
    legendre_poly = np.polynomial.legendre.legval(x, [i, j, k, l, m])
    return legendre_poly


def set_up_fit_qso_continuum_legendre(p):
    '''Function defined to set up fitting legendre polynomial within lmfit

        Parameters
        -----
        p: list
        list of initial guess for the legendre polynomial fit



        returns
        -------
        qso_continuum_legendre_model: lmfit model
        qso_continuum_legendre_model_paramters: lmfit model parameters
        '''

    model_name = "legendre_continuum_"
    qso_continuum_legendre_model = \
        lmfit.Model(qso_continuum_legendre, independent_vars=['wave'],
                    prefix=model_name)
    qso_continuum_legendre_model_paramters = \
        qso_continuum_legendre_model.make_params()

    counter = 0
    for i in ['i', 'j', 'k', 'l', 'm']:
        if p[counter] != 'nan':
            qso_continuum_legendre_model_paramters\
                [model_name+i].set(value=p[counter], min=0.)
        else:
            qso_continuum_legendre_model_paramters\
                [model_name+i].set(value=0, vary=False)
        counter += 1

    return qso_continuum_legendre_model, \
        qso_continuum_legendre_model_paramters


def continuum_additive_polynomial(wave, a, b, c, d, e, f, g, h):
    '''Function defined for fitting a continuum with additive polynomials

        Parameters
        -----
        wave: array
        1-D array of the wavelength to be fit
        a,b,c,d,e,f,g,h: floats
        scale factors

        returns
        -------
        ymod: array
        '''

    x = np.linspace(0, 1, len(wave))
    x2 = np.linspace(1, 0, len(wave))
    ymod = b*(np.exp(-a*x)) + d*(np.exp(-c*x2)) + f*(1-np.exp(-e*x)) + \
        h*(1-np.exp(-g*x2))
    return ymod


def set_up_fit_continuum_additive_polynomial_model(p):
    '''Function defined to set up fitting legendre polynomial within lmfit

        Parameters
        -----
        p: list
        list of initial guess for the legendre polynomial fit



        returns
        -------
        qso_continuum_exponential_scale_model: lmfit model
        qso_continuum_exponential_scale_paramters: lmfit model parameters
        '''

    model_name = 'continuum_additive_polynomial'
    continuum_additive_polynomial_model = \
        lmfit.Model(continuum_additive_polynomial,
                    independent_vars=['wave'], prefix=model_name)
    continuum_additive_polynomial_model_parameters = \
        continuum_additive_polynomial_model.make_params()
    continuum_additive_polynomial_model_parameters\
        [model_name+'a'].set(value=p[0])
    continuum_additive_polynomial_model_parameters\
        [model_name+'b'].set(value=p[1])
    continuum_additive_polynomial_model_parameters\
        [model_name+'c'].set(value=p[2])
    continuum_additive_polynomial_model_parameters\
        [model_name+'d'].set(value=p[3])
    continuum_additive_polynomial_model_parameters\
        [model_name+'e'].set(value=p[4])
    continuum_additive_polynomial_model_parameters\
        [model_name+'f'].set(value=p[5])
    continuum_additive_polynomial_model_parameters\
        [model_name+'g'].set(value=p[6])
    continuum_additive_polynomial_model_parameters\
        [model_name+'h'].set(value=p[7])

    return continuum_additive_polynomial_model, \
        continuum_additive_polynomial_model_parameters


def qsohostfcn(wave, params_fit=None, quiet=True, blrpar=None, qsoxdr=None,
               qsoonly=False, index_log=None, refit=None,
               add_poly_degree=None, siginit_stars=None,
               polyspec_refit=None, fitran=None, fittol=None,
               qsoord=None, hostonly=False, hostord=None,
               blronly=False, qsoflux=None, blrterms=None, **kwargs):

    if blrterms is None:
        blrterms = 0
    if hostord is None:
        hostord = 0
    if qsoord is None:
        qsoord = 0

    # Additive polynomial:
    if not qsoonly and not blronly:
        continuum_model = \
            set_up_fit_continuum_additive_polynomial_model(np.zeros(8))
        ymod = continuum_model[0]
        params = continuum_model[1]

        if hostord > 0:
            continuum_legendre = \
                set_up_fit_qso_continuum_legendre([0., 0., 0, 'nan', 'nan'])
            ymod += continuum_legendre[0]
            params += continuum_legendre[1]

    # QSO continuum
    if not hostonly and not blronly:
        qso_scale_model = set_up_fit_qso_exponential_scale_model(np.zeros(8))

        if 'ymod' not in vars():
            ymod = qso_scale_model[0]
            params = qso_scale_model[1]
        else:
            ymod += qso_scale_model[0]
            params += qso_scale_model[1]

        if qsoord > 0:
            qso_scale_legendre = set_up_fit_qso_scale_legendre(np.zeros(3))
            ymod += qso_scale_legendre[0]
            params += qso_scale_legendre[1]

    # BLR model
    if not hostonly and blrpar is not None:

        counter = 0
        for i in np.arange(0, len(blrpar)/3.):
            gaussian_name = 'g_' + str(int(blrpar[counter+1]))
            gaussian_model = lmfit.models.GaussianModel(prefix=gaussian_name)
            gaussian_model_parameters = gaussian_model.make_params()
            gaussian_model_parameters\
                [gaussian_name+'amplitude'].set(value=blrpar[counter],
                                                min=0.)
            gaussian_model_parameters\
                [gaussian_name+'center'].set(value=blrpar[counter + 1],
                                             vary=False)
            gaussian_model_parameters\
                [gaussian_name+'sigma'].set(value=blrpar[counter + 2],
                                            min=2000. / c.to('km/s').value *
                                            blrpar[counter + 1],
                                            max=6000. / c.to('km/s').value *
                                            blrpar[counter + 1])

            if 'ymod' not in vars():
                ymod = gaussian_model
                params = gaussian_model_parameters
            else:
                ymod += gaussian_model
                params += gaussian_model_parameters
            counter += 3

    # Option to evaulate model for plotting, else return the lmfit model
    # and parameters for fitting in fitqsohost
    if params_fit is not None:
        # comps = ymod.eval_components(params=params_fit, wave=wave,
        #                             qso_model=qsoflux, x=wave)
        continuum = ymod.eval(params_fit, wave=wave, qso_model=qsoflux, x=wave)
        return continuum
    else:
        return ymod, params
