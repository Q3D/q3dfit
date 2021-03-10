import numpy as np


def blackbody(wave,a,T):

    '''Function defined for fitting a blackbody model
    
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
    
    h=6.6261e-27

    c=3e10

    k=1.3807e-16

    hck = h*c/k

    BB_model = wave**-5*(np.exp(hck/wave*T)-1)**-1


    return BB_model

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
    blackbody_model = lmfit.Model(blackbody,independent_vars=['wave'],prefix=model_name)
    blackbody_model_parameters = blackbody_model.make_params()
    blackbody_model_parameters[model_name+'a'].set(value=p[0],min=0,fix=p_fixfree[0])
    blackbody_model_parameters[model_name+'a'].set(value=p[1],min=1,fix=p_fixfree[1])

    return blackbody_model,blackbody_model_parameters

def powerlaw(wave,a,b):

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
    


    powerlaw_model = a*wave**b


    return powerlaw_model

def set_up_fit_powerlaw_model(p,p_fixfree,name):
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
    powerlaw_model = lmfit.Model(powerlaw,independent_vars=['wave'],prefix=model_name)
    powerlaw_model_parameters = powerlaw_model.make_params()
    powerlaw_model_parameters[model_name+'a'].set(value=p[0],min=0,fix=p_fixfree[0])
    powerlaw_model_parameters[model_name+'b'].set(value=p[1],min=1,fix=p_fixfree[1])

    return powerlaw_model,powerlaw_model_parameters


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
        
        
        
    model_scale = a*model


    return model_scale

def set_up_fit_model_scale(p,p_fixfree,name):
    '''Function defined to set up fitting model_scale within lmfit
        
        Parameters
        -----
        p: list
        list of initial guess for the model_scale fit
        
        
        
        returns
        -------
        powerlaw_model_model: lmfit model
        powerlaw_model_paramters: lmfit model parameters
        '''
    
    
    model_name = name
    model_scale_model = lmfit.Model(powerlaw,independent_vars=['model'],prefix=model_name)
    model_scale_parameters = model_scale.make_params()
    model_scale_parameters[model_name+'a'].set(value=p[0],min=0,fix=p_fixfree[0])
    
    return model_scale_model,model_scale_parameters



