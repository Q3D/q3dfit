import lmfit
import numpy as np

from q3dfit.q3dutil import lmlabel
from q3dfit.exceptions import InitializationError
from q3dfit.lineinit import manygauss


def qso_mult_exp(wave, qsotemplate, a, b, c, d, e, f, g, h, i):
    ''' Model exponentials for qso template multiplier

        Parameters
        -----
        qsotemplate: array
            1-D array of the quasar spectrum to be fit
        a,c,c,d,e,f,g,h: floats, > 0
            Model parameters
        returns
        -------
        multiplier*qsotemplate: array
        '''

    x = np.linspace(0., 1., len(wave))
    x2 = np.linspace(1., 0., len(wave))
    multiplier = a + b * (np.exp(-c*x)) + d*(np.exp(-e*x2)) + \
        f*(1.-np.exp(-g*x)) + h*(1.-np.exp(-i*x2))
    return multiplier*qsotemplate


def setup_qso_mult_exp(p):
    '''Set up model exponentials for qso template multiplier

        Parameters
        -----
        p: list
        list of initial guess

        returns
        -------
        qsotemplate_x_exp: lmfit model
        qso_mult_exp_pars: lmfit model parameters
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


def qso_mult_leg(wave, qsotemplate, i, j, k, l, m, n, o, p, q, r):
    '''Model legendre polys for qso template multiplier

        Parameters
        -----
        wave: array
            1-D array of wavelengths
        i,j,k,l,m, ...: floats
            scale factors for legendre polynomials

        returns
        -------
        multiplier*qsotemplate: array
        '''

    x = np.linspace(-1., 1., len(wave))
    multiplier = \
        np.polynomial.legendre.legval(x, [0., i, j, k, l, m, n, o, p, q, r])
    return multiplier*qsotemplate


def setup_qso_mult_leg(p):
    '''Set up model legendre polys for qso template multiplier

        Parameters
        -----
        p: list
            list of initial guess for the legendre polynomial fit

        returns
        -------
        qsotemplate_x_leg: lmfit model
        qso_mult_leg_pars: lmfit model parameters
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


def stars_add_leg(wave, i, j, k, l, m, n, o, p, q, r):
    '''Model legendre for additive starlight continuum

        Parameters
        -----
        wave: array
            1-D array of wavelengths
        i,j,k,l,m: floats
            scale factors for 0-4 order legendre polynomials

        returns
        -------
        starlight: array
        '''

    x = np.linspace(-1., 1., len(wave))
    starlight = \
        np.polynomial.legendre.legval(x, [0., i, j, k, l, m, n, o, p, q, r])
    return starlight


def setup_stars_add_leg(p):
    '''Set up model legendre for additive starlight continuum

        Parameters
        -----
        p: list
            list of initial guess for the legendre polynomial

        returns
        -------
        stars: lmfit model
        stars_add_leg_stars: lmfit model parameters
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


def stars_add_exp(wave, a, b, c, d, e, f, g, h, i):
    ''' Model exponentials for additive starlight continuum


        Parameters
        -----
        wave: array
            1-D array of the wavelength to be fit
        a,b,c,d,e,f,g,h: floats

        returns
        -------
        starlight: array
        '''

    x = np.linspace(0., 1., len(wave))
    x2 = np.linspace(1., 0., len(wave))
    starlight = a + b*(np.exp(-c*x)) + d*(np.exp(-e*x2)) + f*(1.-np.exp(-g*x)) + \
        h*(1.-np.exp(-i*x2))
    return starlight


def setup_stars_add_exp(p):
    '''Set up model exponentials for additive starlight continuum

        Parameters
        -----
        p: list
        list of initial guess for the legendre polynomial fit

        returns
        -------
        stars: lmfit model
        stars_add_exp_pars: lmfit model parameters
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


def qsohostfcn(wave, params_fit=None, qsoflux=None,
               qsoonly=False, qsoord=None, hostonly=False, hostord=None,
               blronly=False, blrpar=None, medflux=None, **kwargs):

    # maximum model legendre polynomial order
    legordmax = 10
    # estimate for continuum level
    if medflux is None:
        medflux=1.

    # Additive starlight component:
    if not qsoonly and not blronly:
        # Terms with exponentials
        initvals = np.concatenate(([np.array(medflux/2.)],np.zeros(8)))
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
        initvals = np.concatenate(([np.array(medfluxuse)],np.zeros(8)))
        qso_mult = setup_qso_mult_exp(initvals)
        if 'ymod' not in vars():
            ymod = qso_mult[0]
            params = qso_mult[1]
        else:
            ymod += qso_mult[0]
            params += qso_mult[1]
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
            lmline = lmlabel(f'{blrpar[counter+1]:g}')
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

            gaussian_model = lmfit.Model(manygauss, prefix=gaussian_name, 
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
                ymod += gaussian_model
                params += gaussian_model_parameters
            counter += 3

    # Option to evaulate model for plotting, else return the lmfit model
    # and parameters for fitting in fitqsohost
    if params_fit is not None:
        continuum = ymod.eval(params_fit, wave=wave, qsotemplate=qsoflux,
                              x=wave)
        return continuum
    else:
        return ymod, params
