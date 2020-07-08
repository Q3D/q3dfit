import numpy 
from q3dfit.common import qsohostfcn
import lmfit

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
    qso_continuum_legendre_model = lmfit.Model(qsohostfcn.qso_continuum_legendre,independent_vars=['wave'],prefix=model_name)
    qso_continuum_legendre_model_paramters=qso_continuum_legendre_model.make_params()
    qso_continuum_legendre_model_paramters[model_name+'i'].set(value=p[0])
    qso_continuum_legendre_model_paramters[model_name+'j'].set(value=p[1])
    qso_continuum_legendre_model_paramters[model_name+'k'].set(value=p[2])
    
    
    return qso_continuum_legendre_model, qso_continuum_legendre_model_paramters


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
    qso_exponential_scale_model = lmfit.Model(qsotemplate_model_exponential,independent_vars=['wave','qso_model'],prefix=model_name)
    exponential_model_parameters = qso_exponential_scale_model.make_params()
    exponential_model_parameters[model_name+'a'].set(value=p[0])#,min=-0.001,max=0.001)
    exponential_model_parameters[model_name+'b'].set(value = p[1])#,min=0,max=1.)
    exponential_model_parameters[model_name+'c'].set(value = p[2])#,min=-0.001,max=0.001)
    exponential_model_parameters[model_name+'d'].set(value = p[3])#,max=1)
    exponential_model_parameters[model_name+'e'].set(value = p[4])#,min=-0.001,max=0.001)
    exponential_model_parameters[model_name+'f'].set(value = p[5])#,min=0,max=1)
    exponential_model_parameters[model_name+'g'].set(value = p[6])#,min=-0.001,max=0.001)
    exponential_model_parameters[model_name+'h'].set(value = p[7])#,min=0,max=1)
    
    return qso_exponential_scale_model,exponential_model_parameters


#def set_up_fit_qso_legendre_scale_model(p,degree)
#
#
#    model_name = 'legendre_scale_'+str(degree)
#    qso_legendre_scale_model = lmfit.Model(qsohostfcn.qsotemplate_model_legendre,independent_vars=['wave','qso_model','degree'],prefix =model_name)
#    qso_legendre_scale_model_parameters = qso_legendre_scale_model.make_params()
#    qso_legendre_scale_model_parameters[model_name+'a'].set(value=p)
#
#
#    return qso_legendre_scale_model,qso_legendre_scale_model_parameters


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
    qso_continuum_legendre_model = lmfit.Model(qsohostfcn.qsotemplate_scale_legendre,independent_vars=['wave','qso_model'],prefix=model_name)
    qso_continuum_legendre_model_paramters=qso_continuum_legendre_model.make_params()
    qso_continuum_legendre_model_paramters[model_name+'i'].set(value=p[0],min=0)
    qso_continuum_legendre_model_paramters[model_name+'j'].set(value=p[1],min=0)
    qso_continuum_legendre_model_paramters[model_name+'k'].set(value=p[2],min=0)
    
    
    return qso_continuum_legendre_model, qso_continuum_legendre_model_paramters



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
    continuum_additive_polynomial_model = lmfit.Model(qsohostfcn.continuum_additive_polynomial,independent_vars=['wave'],prefix=model_name)
    continuum_additive_polynomial_model_parameters = continuum_additive_polynomial_model.make_params()
    continuum_additive_polynomial_model_parameters[model_name+'a'].set(value=p[0])#,min=-0.001,max=0.001)
    continuum_additive_polynomial_model_parameters[model_name+'b'].set(value = p[1])#,min=0,max=1.)
    continuum_additive_polynomial_model_parameters[model_name+'c'].set(value = p[2])#,min=-0.001,max=0.001)
    continuum_additive_polynomial_model_parameters[model_name+'d'].set(value = p[3])#,max=1)
    continuum_additive_polynomial_model_parameters[model_name+'e'].set(value = p[4])#,min=-0.001,max=0.001)
    continuum_additive_polynomial_model_parameters[model_name+'f'].set(value = p[5])#,min=0,max=1)
    continuum_additive_polynomial_model_parameters[model_name+'g'].set(value = p[6])#,min=-0.001,max=0.001)
    continuum_additive_polynomial_model_parameters[model_name+'h'].set(value = p[7])#,min=0,max=1)
    
    return continuum_additive_polynomial_model,continuum_additive_polynomial_model_parameters


def fitqsohost(wave,flux,weight,template_wave,template_flux,index,ct_coeff=None,zstar=None,quiet=None,blrpar=None,qsoxdr=None,qsoonly=None,index_log=None,refit=None,add_poly_degree=None,sigint_stars=None,polyspec_refit=None,fitran=None,fittol=None,qsoord=None,hostonly=None,hostord=None,**kwargs):


    qsotemplate = numpy.load(qsoxdr).item()
    
    qsowave = qsotemplate['wave']
    qsoflux_full = qsotemplate['flux']


    qsoflux = qsoflux_full

    #Normalizing qsoflux template
    qsoflux = qsoflux/numpy.median(qsoflux)

    index = index.astype(dtype='int')

    err = 1/weight**0.5
    iwave = wave[index]
    iflux = flux[index]
    iweight = weight[index]
    ierr = err[index]




    if hostonly:
        print('here')
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
        ymod += qso_scale_model[0]
        params += qso_scale_model[1]

    if qsoord:
        P_L = [1e-1,1e-3,1e-4]
        qso_scale_legendre = set_up_fit_qso_scale_legendre([1e-1,1e-3,1e-4])
        ymod += qso_scale_legendre[0]
        params += qso_scale_legendre[1]



    if blrpar:
        counter = 0
        for i in numpy.arange(0,len(blrpar)/3.):
            gaussian_name = 'g_' + str(blrpar[counter+1])
            gaussian_model = lmfit.models.GaussianModel(prefix=gaussian_name)
            gaussian_model_parameters = gaussian_model.make_params()
            gaussian_model_parameters[gaussian_name+'amplitude'].set(value=blrpar[counter])
            gaussian_model_parameters[gaussian_name+'center'].set(value=blrpar[counter+1])
            gaussian_model_parameters[gaussian_name+'sigma'].set(value=blrpar[counter+2],min=10,max=400)

            ymod += gaussian_model
            params = gaussian_model_parameters
            counter += 3




    result = ymod.fit(iflux,params,qso_model=qsoflux[index],wave=iwave)#,x=x_to_fit) #

    comps = result.eval_components(wave=wave,qso_model=qsoflux,x=wave)
    y_final = result.eval(wave=wave,qso_model=qsoflux)


    if 'fcn_test' in kwargs.keys():
    
        return result,comps,y_final

    else:
        return y_final

