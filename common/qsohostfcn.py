import numpy
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


def qso_continuum_legendre(wave,i,j,k):
    
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


def qsohostfcn(wave, params_fit, quiet=None, blrpar=None, qsoxdr=None,
                         qsoonly=None, index_log=None, refit=None,
                         add_poly_degree=None, siginit_stars=None,
                         polyspec_refit=None, fitran=None, fittol=None,
                         qsoord=None, hostonly=None, hostord=None,
                         blronly=None,qsoflux=None, **kwargs):
    pass


    '''Function defined to recreate the best fit continuum model.
    
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
    
    ct_coeff: dictionary or lmfit best fit params structure
        best fit parameters
    
    
    returns
    -------
    continuum: array
        best fit continuum model
    
    comps: lmfit individual components structure
        best fit continuum model for each component
    '''



#    if qsoxdr is None:
#        sys.exit('Quasar template (qsoxdr) not specified in \
#                 initialization file.')
#    try:
#        qsotemplate = np.load(qsoxdr, allow_pickle=True).item()
#    except:
#        sys.exit('Cannot find quasar template (qsoxdr).')
#
#    qsowave = qsotemplate['wave']
#    qsoflux_full = qsotemplate['flux']
#
#    iqsoflux = np.where((qsowave >= fitran[0]) & (qsowave <= fitran[1]))
#    qsoflux = qsoflux_full[iqsoflux]
#
#    #Normalizing qsoflux template
#    qsoflux = qsoflux/np.median(qsoflux)
#
#    index = np.array(index)
#    index = index.astype(dtype='int')

#    if type(ct_coeff) == dict:
#        params = ct_coeff['qso_host']
#
#    else:
#        params = ct_coeff


    if add_poly_degree ==  None:
        add_poly_degree = 30

    #Default is QSO exponential scale + Host exponential scale model
    if hostonly == None and hostord == None and qsoonly == None and qsoord == None:

        qso_scale_model = set_up_fit_qso_exponential_scale_model([0.0,0.5,0.0,0.5,0.0,0.5,0.0,0.5])
        ymod = qso_scale_model[0]
        params = qso_scale_model[1]

        continuum_model = set_up_fit_continuum_additive_polynomial_model([1e-2,0.5,1e-2,0.5,1e-3,0.5,1e-3,0.5])
        ymod += continuum_model[0]
        params += continuum_model[1]

    #Additional options for fitting only QSO or HOST
    if hostonly:
        continuum_model = set_up_fit_continuum_additive_polynomial_model([1e-2,0.5,1e-2,0.5,1-3,0.5,1e-3,0.5])
        ymod = continuum_model[0]
        params = continuum_model[1]

    if hostonly and hostord:

        additive_polynomial_model = set_up_fit_continuum_additive_polynomial_model([1e-2,0.5,1e-2,0.5,1-3,0.5,1e-3,0.5])
        continuum_legendre = set_up_fit_qso_continuum_legendre([1e-1,1e-3,0.])
        ymod += additive_polynomial_model[0] + continuum_legendre[0]
        params += additive_polynomial_model[1] + continuum_legendre[1]

    if qsoonly:
        qso_scale_model = set_up_fit_qso_exponential_scale_model([0.0,0.5,0.0,0.5,0.0,0.5,0.0,0.5])
        ymod = qso_scale_model[0]
        params = qso_scale_model[1]

    if qsoonly and qsoord:
        P_L = [1e-1,1e-3,1e-4]
        qso_scale_legendre = set_up_fit_qso_scale_legendre([1e-1,1e-3,1e-4])
        ymod += qso_scale_legendre[0]
        params += qso_scale_legendre[1]

    if blrpar:
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


    comps = ymod.eval_components(params=params_fit,wave=wave, qso_model=qsoflux, x=wave)
    continuum = ymod.eval(params_fit,wave=wave, qso_model=qsoflux, x=wave)

    return continuum

