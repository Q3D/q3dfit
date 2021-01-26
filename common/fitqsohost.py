import lmfit
import numpy as np
import pdb
import ppxf.ppxf_util as util
import sys

from astropy.constants import c
from ppxf.ppxf import ppxf
from q3dfit.common import qsohostfcn
from q3dfit.common.interptemp import interptemp
from scipy import interpolate
from time import perf_counter as clock


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

    # x = np.arange(0,1.,len(qsotemplate))
    # x=np.arange(0,len(qsotemplate))
    # x2=x
    x = np.linspace(0, 1, len(wave))
    x2 = np.linspace(1, 0, len(wave))
    # x=wave
    # x2=wave
    # x2=wave[::-1]
    qsotemplate_for_fit = b*(np.exp(-a*x))+d*(np.exp(-c*x2))+\
        f*(1-np.exp(-e*x))+h*(1-np.exp(-g*x2))

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


def fitqsohost(wave, flux, weight, template_wave, template_flux, index,
               zstar, quiet=None, blrpar=None, qsoxdr=None,
               qsoonly=None, index_log=None, refit=None,
               add_poly_degree=None, siginit_stars=None,
               polyspec_refit=None, fitran=None, fittol=None,
               qsoord=None, hostonly=None, hostord=None, **kwargs):

    '''Function defined to fit the continuum
    
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
    
    
    
    
    returns
    -------
    continuum: array
        best fit continuum model
    
    ct_coeff: dictionary or lmfit best fit params structure
        best fit parameters
    
    zstar: float
        best fit stellar redshift
    '''
        
        
    if qsoxdr is None:
        sys.exit('Quasar template (qsoxdr) not specified in \
                 initialization file.')
    try:
        qsotemplate = np.load(qsoxdr, allow_pickle=True).item()
    except:
        sys.exit('Cannot find quasar template (qsoxdr).')

    qsowave = qsotemplate['wave']
    qsoflux_full = qsotemplate['flux']

    iqsoflux = np.where((qsowave >= fitran[0]) & (qsowave <= fitran[1]))
    qsoflux = qsoflux_full[iqsoflux]

    #Normalizing qsoflux template
    qsoflux = qsoflux/np.median(qsoflux)

    index = np.array(index)
    index = index.astype(dtype='int')

    err = 1/weight**0.5
    iwave = wave[index]
    iflux = flux[index]
    iweight = weight[index]
    ierr = err[index]

    if add_poly_degree ==  None:
        add_poly_degree = 30

    #Default is QSO exponential scale + Host exponential scale model
    if hostonly == None and hostord == None and qsoonly == None and qsoord == None:

        qso_scale_model = set_up_fit_qso_exponential_scale_model([0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0])#[0.0,0.5,0.0,0.5,0.0,0.5,0.0,0.5]
        ymod = qso_scale_model[0]
        params = qso_scale_model[1]

        continuum_model = set_up_fit_continuum_additive_polynomial_model([0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0])#[1e-2,0.5,1e-2,0.5,1e-3,0.5,1e-3,0.5]
        ymod += continuum_model[0]
        params += continuum_model[1]

    #Additional options for fitting only QSO or HOST
    if hostonly:
        continuum_model = set_up_fit_continuum_additive_polynomial_model([0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0])#[1e-2,0.5,1e-2,0.5,1e-3,0.5,1e-3,0.5]
        ymod = continuum_model[0]
        params = continuum_model[1]

    if hostonly and hostord:

        additive_polynomial_model = set_up_fit_continuum_additive_polynomial_model([0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0])#[1e-2,0.5,1e-2,0.5,1-3,0.5,1e-3,0.5]
        continuum_legendre = set_up_fit_qso_continuum_legendre([1e-1,1e-3,0.])
        ymod += additive_polynomial_model[0] + continuum_legendre[0]
        params += additive_polynomial_model[1] + continuum_legendre[1]

    if qsoonly:
        qso_scale_model = set_up_fit_qso_exponential_scale_model([0.0,0.5,0.0,0.0,0.0,0.5,0.0,0.0])#[0.0,0.5,0.0,0.5,0.0,0.5,0.0,0.5]
        ymod += qso_scale_model[0]
        params += qso_scale_model[1]

    if qsoonly and qsoord:
        P_L = [1e-1,1e-3,1e-4]
        qso_scale_legendre = set_up_fit_qso_scale_legendre([0.0,0.0,0.0])#[1e-1,1e-3,1e-4]
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

            ymod += gaussian_model
            params += gaussian_model_parameters
            counter += 3

#testing with resid function
#    def resid(params,wave,qsoflux=None,ymod=None,flux=None):
#
#        return ymod.eval(params,wave=wave, qso_model=qsoflux, x=wave) - flux
#    out = lmfit.minimize(resid, params, args=(iwave,), kws={'qsoflux': qsoflux[index],'ymod':ymod,'flux':iflux},method='least_squares',nan_policy='omit)

    result = ymod.fit(iflux, params, weights=iweight, qso_model=qsoflux[index],wave=iwave, x=iwave,method='least_squares',nan_policy='omit')

    comps = result.eval_components(wave=wave, qso_model=qsoflux, x=wave)
    continuum = result.eval(wave=wave, qso_model=qsoflux, x=wave)

#    Test plot
#    import matplotlib.pyplot as plt
#    for i in comps.keys():
#        plt.plot(wave, comps[i], label=i)
#    plt.plot(wave, continuum, label='best-fit')
#    plt.plot(wave,flux,label='flux')
#    plt.plot(wave,flux-continuum,label='resid')
#    plt.legend(loc='best')
#    plt.show()

    ct_coeff = result.params

    # Fit residual with PPXF
    if refit:

        resid = flux - continuum

        # log rebin residual
        # lamRange1 = np.array([wave.min(), wave.max()])/(1+zstar)
        resid_log, residlambda_log, velscale = util.log_rebin(fitran, resid)
        residerrsq_log, _, _ = util.log_rebin(fitran, np.divide(1., weight))
        residerr_log = np.sqrt(residerrsq_log)

        # Interpolate template to same grid as data
        temp_log = interptemp(residlambda_log, np.log(template_wave.T[0]),
                              template_flux)

        # vel = c*np.log(1 + zstar)   # eq.(8) of Cappellari (2017)
        # t = clock()
        start = [0, siginit_stars]  # (km/s), starting guess for [V, sigma]
        pp = ppxf(temp_log, resid_log, residerr_log, velscale, start,
                  goodpixels=index_log, moments=2,  # plot=True, moments=2
                  degree=add_poly_degree)  # clean=False

        # resample additive polynomial to linear grid
        poly_log = pp.apoly
        pinterp = interpolate.interp1d(residlambda_log, poly_log,
                                       kind='cubic', fill_value="extrapolate")
        poly = pinterp(np.log(wave))

        ct_coeff = {'qso_host': result.params,
                    'stel': pp.weights,
                    'poly': poly,
                    'ppxf_sigma': pp.sol[1]}

        zstar += pp.sol[0]/c.to('km/s').value

        # host can't be negative
        ineg = np.where(continuum < 0)
        continuum[ineg] = 0

        continuum_log = pp.bestfit
        cinterp = interpolate.interp1d(residlambda_log, continuum_log,
                                       kind='cubic', fill_value="extrapolate")

        cont_resid = cinterp(np.log(wave))
        continuum += cont_resid

    # if 'fcn_test' in kwargs.keys():
    #     return result, comps, y_final
    # else:
        return continuum, ct_coeff, zstar





# def fit_cont_ppxf(wave,flux,z_stars,index,template):

#     z = z_stars
#     mask = ((wave > 3540) & (wave < 7450))
#     flux_to_fit = flux[index.astype(int)]
#     galaxy = flux_to_fit/np.abs(np.median(flux_to_fit))
#     lam_gal = wave[index.astype(int)]
#     loglam_gal = np.log10(wave[index.astype(int)])
#     noise = np.full_like(galaxy,np.std(galaxy))


#     #Setting instrument properties:
#     frac = wave[1]/wave[0]  # Constant lambda fraction per pixel
#     dlam_gal = (frac - 1)*wave[index.astype(int)]  # Size of every pixel in Angstrom
#     wdisp = np.full_like(galaxy,1)  # Intrinsic dispersion of every pixel, in pixels units
#     fwhm_gal = 2.355*wdisp*dlam_gal  # Resolution FWHM of every pixel, in Angstroms
#     velscale = np.log(frac)*c.to('km/s')  # Velocity scale in km/s per pixel (eq.8 of Cappellari 2017)


#     #Setting up models to fit:

#     if template == 'Vazdekis':
#         ppxf_dir = path.dirname(path.realpath(ppxf_package.__file__))
#         # Read the list of filenames from the Single Stellar Population library
#         # by Vazdekis (2010, MNRAS, 404, 1639) http://miles.iac.es/. A subset
#         # of the library is included for this example with permission
#         vazdekis = glob.glob(ppxf_dir + '/miles_models/Mun1.30Z*.fits')
#         fwhm_tem = 2.51 # Vazdekis+10 spectra have a constant resolution FWHM of 2.51A.

#         # Extract the wavelength range and logarithmically rebin one spectrum
#         # to the same velocity scale of the galaxy spectrum, to determine
#         # the size needed for the array which will contain the template spectra.
#         #
#         hdu = fits.open(vazdekis[0])
#         ssp = hdu[0].data
#         h2 = hdu[0].header
#         lam_temp = h2['CRVAL1'] + h2['CDELT1']*np.arange(h2['NAXIS1'])
#         lamRange_temp = [np.min(lam_temp), np.max(lam_temp)]
#         sspNew = util.log_rebin(lamRange_temp, ssp, velscale=velscale)[0]
#         templates = np.empty((sspNew.size, len(vazdekis)))

#         #Interpolates the galaxy spectral resolution at the location of every pixel
#         #of the templates. Outside the range of the galaxy spectrum the resolution
#         # will be extrapolated, but this is irrelevant as those pixels cannot be
#         # used in the fit anyway.
#         fwhm_gal = np.interp(lam_temp, lam_gal, fwhm_gal)

#         fwhm_dif = np.sqrt((fwhm_gal**2 - fwhm_tem**2).clip(0))
#         sigma = fwhm_dif/2.355/h2['CDELT1'] # Sigma difference in pixels

#         for j, fname in enumerate(vazdekis):
#             hdu = fits.open(fname)
#             ssp = hdu[0].data
#             ssp = util.gaussian_filter1d(ssp, sigma)  # perform convolution with variable sigma
#             sspNew = util.log_rebin(lamRange_temp, ssp, velscale=velscale)[0]
#             templates[:, j] = sspNew/np.median(sspNew) # Normalizes templates

#     if template == 'PG1411':
#         fwhm_tem = 1.35 #1A?
#         template_PG1411 = np.load('data/pg1411hosttemplate.npy',allow_pickle=True).item()
#         lam_temp = template_PG1411['lambda']
#         lamRange_temp = [np.min(lam_temp), np.max(lam_temp)]
#         ssp_PG1411 = template_PG1411['flux']
#         sspNew_PG1411 = util.log_rebin(lamRange_temp, ssp_PG1411, velscale=velscale)[0]
#         sspNew_PG1411 = sspNew_PG1411/np.median(sspNew_PG1411)
#         templates = sspNew_PG1411

#     dv = c*np.log(lam_temp[0]/lam_gal[0])    # eq.(8) of Cappellari (2017)
#     goodpixels = util.determine_goodpixels(np.log(lam_gal), lamRange_temp, z=z)

#     vel = c*np.log(1 + z)   # eq.(8) of Cappellari (2017)
#     start = [vel, 200.]  # (km/s), starting guess for [V, sigma]
#     t = clock()

#     pp = ppxf(templates, galaxy, noise, velscale, start,
#             goodpixels=goodpixels, plot=True, moments=4,
#               degree=4, vsyst=dv, clean=False, lam=lam_gal)


#     print("Formal errors:")
#     print("     dV    dsigma   dh3      dh4")
#     print("".join("%8.2g" % f for f in pp.error*np.sqrt(pp.chi2)))
#     print('Elapsed time in PPXF: %.2f s' % (clock() - t))


#     return pp
