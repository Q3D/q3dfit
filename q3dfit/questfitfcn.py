from astropy import units as u
from lmfit.models import ExpressionModel

import lmfit
import numpy as np


def readcf(filename):
    ''' Function defnied to read in the configuration file for
        fitting with questfit
        in:
        filename: str
        out:
        '''

    cf = np.loadtxt(filename, dtype='str', comments="#")

    # Leftover from IDL QUESTFIT
    # numberoftemplates = list(cf.T[0]).count('template')
    # numberofBB = list(cf.T[0]).count('blackbody')
    # numberofpl = list(cf.T[0]).count('powerlaw')
    # numberofext = list(cf.T[0]).count('extinction')
    # numberofabs = list(cf.T[0]).count('absorption')
    # numberofsources = list(cf.T[0]).count('source')

    # Cycle throughrows
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


def blackbody(wave, a, T, fitFlambda=True):
    '''
    Function defined for fitting a blackbody model

    Parameters
    -----
    wave: array
        1-D array of the wavelength to be fit
    a: float
        scale factor for model
    T: float
        temperature of blackbody

    returns
    -------
    BB_model: array
    '''
    # Blam = BlackBody(temperature=T*u.K, scale=1.*u.Unit('erg/cm^2/micron/s/sr'))
    # Blamval = Blam(wave*u.micron).value

    # if not fitFlambda:
    #     c_scale = c * u.Unit('m').to('micron') /(wave)**2 *1e-23
    #     BB_model /= c_scale

    # return a*Blamval/Blamval.max()

    h=6.6261e-27 # cm^2 g / s
    c=2.99792e10 # cm/s
    k=1.3807e-16 # cm^2 g s-2 K-1

    hck = h*c/k
    wave = wave *1e-4   # cm
    BB_model = wave**-5*(np.exp(hck/wave/T)-1)**-1

    if not fitFlambda:
        c_scale = c * u.Unit('m').to('micron') /(wave)**2 *1e-23
        BB_model /= c_scale

    return a*BB_model/BB_model.max()


def set_up_fit_blackbody_model(p,p_fixfree,name):
    '''Function defined to set up fitting blackbody_model within lmfit

        Parameters
        -----
        p: list
        list of initial guess for the blackbody_model fit



        returns
        -------
        blackbody_model_model: lmfit model
        blackbody_model_paramters: lmfit model parameters
        '''


    model_name = name
    blackbody_model = lmfit.Model(blackbody,independent_vars=['wave', 'fitFlambda'],prefix=model_name)
    blackbody_model_parameters = blackbody_model.make_params()
    print(p_fixfree[0])
    blackbody_model_parameters[model_name+'a'].\
        set(value=p[0], min=np.finfo(float).eps, vary=p_fixfree[0])
    blackbody_model_parameters[model_name+'T'].\
        set(value=p[1], min=50., max=3000., vary=p_fixfree[1])

    return blackbody_model,blackbody_model_parameters


def modelmultpoly(template_0, wave, amp, multpolyA, multpolyB, multpolyC):

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



def powerlaw(wave, a, b, fitFlambda):

    '''Function defined for fitting a powerlaw model

    Parameters
    -----
    wave: array
        1-D array of the wavelength to be fit
    a: float
        scale factor for powerlaw
    b: float
        exponent for powerlaw

    returns
    -------
    powerlaw_model: array
    '''

    powerlaw_model = wave**b

    # if not fitFlambda:
    #     c_scale = c * u.Unit('m').to('micron') /(wave)**2 *1e-23
    #     powerlaw_model /= c_scale

    return a*powerlaw_model #/(powerlaw_model).max()


def set_up_fit_powerlaw_model(p, p_fixfree, name):
    '''Function defined to set up fitting powerlaw_model within lmfit

        Parameters
        -----
        p: list
            list of initial guess for the powerlaw_model fit

        returns
        -------
        powerlaw_model_model: lmfit model
        powerlaw_model_paramters: lmfit model parameters
        '''

    model_name = name
    powerlaw_model = \
        lmfit.Model(powerlaw, independent_vars=['wave', 'fitFlambda'],
                    prefix=model_name)
    powerlaw_model_parameters = powerlaw_model.make_params()
    powerlaw_model_parameters[model_name+'a'].\
        set(value=p[0], min=np.finfo(float).eps, vary=p_fixfree[0])
    powerlaw_model_parameters[model_name+'b'].\
        set(value=p[1], vary=p_fixfree[1])

    return powerlaw_model, powerlaw_model_parameters


def model_scale(model,a):

    '''Function defined for fitting a continuum model

    Parameters
    -----
    wave: array
    1-D array of the wavelength to be fit
    a: float
    scale factor for model_scale

    returns
    -------
    powerlaw_model: array
    '''

    model_scale = model*a
    return model_scale




#def screen_exctinction_model(extinction_curve,Av):
#
#    '''Function defined for fitting a screen_exctinction_model
#
#        Parameters
#        -----
#        wave: array
#        1-D array of the wavelength to be fit
#        Av: float
#        scale factor for exctinction_model
#
#
#        returns
#        -------
#        powerlaw_model: array
#        '''
#
#
#
#    model_extinction = 10**-0.4*Av*np.log10(extinction_curve)
#
#
#    return model_extinction


#def mixed_exctinction_model(extinction_curve,Av):
#
#    '''Function defined for fitting a screen_exctinction_model
#
#        Parameters
#        -----
#        wave: array
#        1-D array of the wavelength to be fit
#        Av: float
#        scale factor for exctinction_model
#
#
#        returns
#        -------
#        powerlaw_model: array
#        '''
#
#
#    tau = 0.4*Av*np.log10(extinction_curve)
#
#    model_extinction = 1 - np.exp(-tau)/(tau)
#
#    return model_extinction

def set_up_fit_extinction(p, p_fixfree, model_name, extinction_model,
                          mixed_or_screen):
    '''
    Function defined to set up fitting model_scale within lmfit

        Parameters
        -----
        p: list
            list of initial guess ,,.

        returns
        -------
        model_extinction: lmfit model
        model_extinction_paramters: lmfit model parameters
    '''

    if mixed_or_screen == 'M':
        exp = "1. - exp(-0.4*%s_Av*log10(%s))/(0.4*%s_Av*log10(%s))" \
            % (model_name, extinction_model, model_name, extinction_model)

    if mixed_or_screen == 'S':
        exp = "power(10.,-0.4*%s_Av*log10(%s))" \
            % (model_name, extinction_model)

    model_extinction = \
        ExpressionModel(exp, independent_vars=[extinction_model],
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


def set_up_fit_model_scale(p, p_fixfree, model_name,model, fitFlambda=True,
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


    exp = "1*%s_amp*1*%s" % (model,model)
    model_scale_model = ExpressionModel(exp,independent_vars=[model],name=model_name)
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


    return model_scale_model,model_scale_parameters



def set_up_absorption(p, p_fixfree, model_name, abs_model):
    '''Function defined to set up fitting model_scale within lmfit

    Parameters
    -----
    p: list
        list of initial guess for the absorption model
    p_fixfree: list
        list of fix/free values (0/1) for the absorption model

    Returns
    -------
    absorption_model: lmfit model
    absorption_paramters: lmfit model parameters
    '''

    exp = "exp(-1.*%s_tau*%s)" % (model_name, abs_model)
    model = ExpressionModel(exp,independent_vars=[abs_model], name = model_name)
    abs_model_parameters = model.make_params()
    abs_model_parameters[model_name+'_tau'].\
        set(value=p[0], min=np.finfo(float).eps, max=10., vary=p_fixfree[0])

    return model,abs_model_parameters
