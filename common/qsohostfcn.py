import numpy


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


