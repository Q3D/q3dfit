import numpy
import scipy
from q3dfit.common.fitqsohost import *


def qsotemplate_model_exponential(wave,qso_model,a,b,c,d,e,f,g,h):
    
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
    
    #x = numpy.arange(0,1.,len(qsotemplate))
    #x=numpy.arange(0,len(qsotemplate))
    #x2=x
    x=numpy.linspace(0,1,len(wave))
    x2=numpy.linspace(1,0,len(wave))
    #x=wave
    #x2=wave
    #x2=wave[::-1]
    qsotemplate_for_fit = b*(numpy.exp(-a*x))+d*(numpy.exp(-c*x2))+f*(1-numpy.exp(-e*x))+h*(1-numpy.exp(-g*x2))
    
    return qsotemplate_for_fit*qso_model


def qso_continuum_legendre(wave,i,j,k,l,m):
    
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
    
    
    
    #print(kwargs.items())
    #return numpy.polynomial.legendre.legval(wave,list(kwargs.values()))
    x=numpy.linspace(-1,1,len(wave))
    legendre_poly = numpy.polynomial.legendre.legval(x,[i,j,k,l,m])
    #legendre_poly=scipy.special.eval_legendre(n,x)
    return legendre_poly

def qsotemplate_scale_legendre(wave,qso_model,i,j,k):
    
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
    
    
    
    #print(kwargs.items())
    #return numpy.polynomial.legendre.legval(wave,list(kwargs.values()))
    x=numpy.linspace(0,1,len(wave))
    legendre_poly = numpy.polynomial.legendre.legval(x,[i,j,k])
    return qso_model*legendre_poly


#def qsotemplate_model_legendre(wave,qso_model,degree,a):
#    x=numpy.linspace(0,1,len(wave))
#    return qso_model*numpy.polynomial.legendre.legval(x,[1]*degree)



def continuum_additive_polynomial(wave,a,b,c,d,e,f,g,h):
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
    
    #x = numpy.arange(0,1.,len(qsotemplate))
    #x=numpy.arange(0,len(qsotemplate))
    #x2=x
    x=numpy.linspace(0,1,len(wave))
    x2=numpy.linspace(1,0,len(wave))
    #x=wave
    #x2=wave
    #x2=wave[::-1]
    ymod = b*(numpy.exp(-a*x))+d*(numpy.exp(-c*x2))+f*(1-numpy.exp(-e*x))+h*(1-numpy.exp(-g*x2))




    return ymod

def qsohostfcn(wave, params_fit=None, quiet=None, blrpar=None, qsoxdr=None,
              qsoonly=None, index_log=None, refit=None,
              add_poly_degree=None, siginit_stars=None,
              polyspec_refit=None, fitran=None, fittol=None,
              qsoord=None, hostonly=None, hostord=None,
              blronly=None,qsoflux=None,blrterms=None, **kwargs):
    
    if blrterms == None:
        blrterms = 0
    if hostord == None:
        hostord = 0
    if qsoord == None:
        qsoord = 0



    #Additive polynomial:
    if qsoonly == None and blronly == None:
        continuum_model = set_up_fit_continuum_additive_polynomial_model([0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0])#[1e-2,0.5,1e-2,0.5,1e-3,0.5,1e-3,0.5]
        ymod = continuum_model[0]
        params = continuum_model[1]
        
        if hostord > 0:
            continuum_legendre = set_up_fit_qso_continuum_legendre([0.,0.,0,'nan','nan'])
            ymod +=  continuum_legendre[0]
            params += continuum_legendre[1]


    #QSO continuum

    if hostonly == None and blronly == None:
        qso_scale_model = set_up_fit_qso_exponential_scale_model([0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0])#[0.0,0.5,0.0,0.5,0.0,0.5,0.0,0.5]

        if 'ymod' not in vars():
            ymod = qso_scale_model[0]
            params = qso_scale_model[1]
        else:
            ymod += qso_scale_model[0]
            params += qso_scale_model[1]

        if qsoord > 0:
            qso_scale_legendre = set_up_fit_qso_scale_legendre([0.0,0.0,0.0])#[1e-1,1e-3,1e-4]
            ymod += qso_scale_legendre[0]
            params += qso_scale_legendre[1]



    #BLR model
    if hostonly == None and blrpar:
        
        counter = 0
        for i in np.arange(0,len(blrpar)/3.):
            gaussian_name = 'g_' + str(int(blrpar[counter+1]))
            gaussian_model = lmfit.models.GaussianModel(prefix=gaussian_name)
            gaussian_model_parameters = gaussian_model.make_params()
            gaussian_model_parameters[gaussian_name+'amplitude'].set(value=blrpar[counter])
            gaussian_model_parameters[gaussian_name+'center'].set(value=blrpar[counter+1],min=blrpar[counter+1]-blrpar[counter+1]*0.001,max=blrpar[counter+1]+blrpar[counter+1]*0.001)
            gaussian_model_parameters[gaussian_name+'sigma'].set(value=blrpar[counter+2],min=10,max=200)

            if 'ymod' not in vars():
                ymod = gaussian_model
                params = gaussian_model_parameters
            else:
                ymod += gaussian_model
                params += gaussian_model_parameters
            counter += 3
            


    #Option to evaulate model for plotting, else return the lmfit model and parameters for fitting in fitqsohost
    if params_fit != None:
        comps = ymod.eval_components(params=params_fit,wave=wave, qso_model=qsoflux, x=wave)
        continuum = ymod.eval(params_fit,wave=wave, qso_model=qsoflux, x=wave)
        return continuum
    else:
        return ymod,params




