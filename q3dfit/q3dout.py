# -*- coding: utf-8 -*-
"""
@author: Caroline Bertemes, based on q3da (now q3di) by Hadley Lim
"""

import copy
import numpy as np
import q3dfit.q3dutil as q3dutil

from astropy.constants import c
from astropy.stats import gaussian_sigma_to_fwhm
from astropy.table import Table
from importlib import import_module
from ppxf.ppxf_util import log_rebin
from q3dfit.q3dmath import gaussflux
from q3dfit.q3dutil import lmlabel
from q3dfit.qsohostfcn import qsohostfcn
from scipy import constants
from scipy.interpolate import interp1d


def load_q3dout(q3di, col, row, cubedim=None):
    """
    Load object after it's been saved to a file.

    Parameters
    ----------
    q3di : object
        q3dinit object.
    col : int
        Column index.
    row : int
        Row index.
    cubedim : int, optional
        Dimension of cube (1, 2, or 3). If None, will try to get from q3di or 
        cube itself.

    Returns
    -------
    q3dout : object
        q3dout object.

    """

    # convert from string to object if necessary.
    q3dii = q3dutil.get_q3dio(q3di)

    filelab = '{0.outdir}{0.label}'.format(q3dii)
    if cubedim is None:
        if hasattr(q3dii, 'cubedim'):
            cubedim = q3dii.cubedim
        else:
            print('load_q3dout: q3di has no attribute cubedim, loading cube')
            cube, vormap = q3dutil.get_Cube(q3dii)
            cubedim = cube.dat.ndim
    if cubedim > 1:
        filelab += '_{:04d}'.format(col)
    if cubedim > 2:
        filelab += '_{:04d}'.format(row)
    infile = filelab + '.npy'

    # this should work for both q3di and q3do
    q3dout = q3dutil.get_q3dio(infile)
    # add file label to object
    q3dout.filelab = filelab
    return q3dout


class q3dout:
    '''
    This class defines a q3dout object, which is created by q3df when
    running on any single spaxel. It collects all the output of
    q3df/fitspec and contains functions to generate plots for a single
    spaxel.

    (For multi-spaxel processing instead, please see q3dpro.)

    Attributes
    ----------
    wave : ndarray
        Wavelength array.
    spec : ndarray
        Flux array.
    spec_err : ndarray
        Error array.
    fitrange : ndarray
        Wavelength range for fitting.
    col : int
        Column index.
    row : int
        Row index.
    gd_indx : ndarray
        Good data indices.
    fitran_indx : ndarray
        Fit range indices.
    fluxunit : str
        Flux unit.
    waveunit : str
        Wavelength unit.
    fluxnorm : float
        Flux normalization.
    pixarea_sqas : float
        Pixel area in square arcseconds.
    nogood : bool
        No good data present.
    docontfit : bool
        Do continuum fit.
    dolinefit : bool
        Do line fit.
    linelist : dict
        List of emission line rest-frame wavelengths.
    linelabel : dict
        Line labels.
    fitstatus : dict
        Fit status.
    maxncomp : int
        Maximum number of components.
    nfev : int
        Number of function evaluations.
    noemlinmask : dict
        Mask for regions with emission lines.
    redchisq : float
        Reduced chi-squared.
    param : dict
        Best-fit parameters.
    parinit : dict
        Initial parameters.
    perror : dict
        Parameter errors.
    perror_resid : dict
        Parameter errors from residuals.
    perror_errspec : dict
        Parameter errors from error spectrum.
    siglim : float
        Sigma limits for emission lines.
    covar : ndarray
        Covariance matrix. Added/updated by init_linefit().
    dof : int
        Degrees of freedom. Added/updated by init_linefit().
    line_dat : dict
        Line data. Added/updated by init_linefit().
    line_fit : dict
        Line fit. Added/updated by init_linefit().
    cont_dat : dict
        Continuum data.  Added/updated by init_contfit().
    cont_fit : dict
        Continuum fit. Added/updated by init_contfit().
    ct_method : str
        Continuum fit method. Added/updated by init_contfit().
    ct_coeff : dict
        Continuum fit coefficients. Added/updated by init_contfit().
    ct_ebv : float
        E(B-V) value from continuum fit. Added/updated by init_contfit().
    ct_add_poly_weights : dict
        Additive polynomial weights for continuum fit. Added/updated by
    ct_ppxf_sigma : float
        Best-fit sigma from ppxf. Added/updated by init_contfit().
    ct_ppxf_sigma_err : float
        Error in best-fit sigma from ppxf. Added/updated by init_contfit().    
    ct_rchisq : float
        Reduced chi-squared from continuum fit. Added/updated by init_contfit().
    ct_indx : ndarray
        Continuum fit indices. Added/updated by init_contfit().
    zstar : float
        Best-fit zstar. Added/updated by init_contfit().
    zstar_err : float
        Error in best-fit zstar. Added/updated by init_contfit().
    add_poly_degree : int
        Degree of additive polynomial for continuum fit. Added/updated by
    cont_fit_poly : ndarray
        Polynomial component of continuum fit. Added/updated by sepcontpars().
    cont_fit_stel : ndarray
        Stellar component of continuum fit. Added/updated by sepcontpars().
    qsomod : float
        QSO component of continuum fit. Added/updated by sepcontpars().
    hostmod : float
        Host component of continuum fit. Added/updated by sepcontpars().
    polymod_refit : ndarray
        Polynomial component of continuum fit, if refit with fitqsohost. 
        Added/updated by sepcontpars().
    blrpar : dict
        BLR fit parameters, if using fitqsohost. Added/updated by sepcontpars().
    line_fitpars : dict
        Line fit parameters. Added/updated by sepfitpars().
    tflux : dict
        Total fluxes of emission lines. Added/updated by sepfitpars().
    tfluxerr : dict
        Total flux errors of emission lines. Added/updated by sepfitpars().
    filelab : str
        File label for plotting methods. Added/updated by load_q3dout().
    
    Methods
    -------
    init_linefit(linelist, linelabel, maxncomp, covar=None, dof=None,
                fitstatus=None, line_dat=None, line_fit=None, nfev=None,
                noemlinmask=None, redchisq=None, param=None, parinit=None,
                perror=None, perror_resid=None, perror_errspec=None, siglim=None)
        Initialize line fit parameters.

    init_contfit(ct_method='CONTINUUM SUBTRACTED', ct_coeff=None, ct_ebv=None,
                ct_add_poly_weights=None, ct_ppxf_sigma=None, ct_ppxf_sigma_err=None,
                ct_rchisq=None, cont_dat=None, cont_fit=None, ct_indx=None,
                zstar=None, zstar_err=None)
        Initialize continuum fit parameters.
    
    sepfitpars(waveran=None, doublets=None, ignoreres=False)
        Convert output of LMFIT, with best-fit line parameters in a single
        array, into a dictionary with separate arrays for different line
        parameters. Compute total line fluxes from the best-fit line
        parameters.
        :no-index:

    sepcontpars(q3di)
        Compute PPXF components: additive polynomial and stellar fit.

    plot_linefit(q3di, q3dout, line, comp, ax=None, fig=None, show=True,
                save=False, savepath=None, saveformat='pdf', **kwargs)
        Plot line fit.

    plot_contfit(q3di, q3dout, ax=None, fig=None, show=True, save=False,
                savepath=None, saveformat='pdf', **kwargs)
        Plot continuum fit.
        
    '''

    def __init__(self, wave, spec, spec_err, fitrange=None, col=None, row=None,
                 gd_indx=None, fitran_indx=None, fluxunit='erg/s/cm^2/micron',
                 waveunit='micron', fluxnorm=1., pixarea_sqas=None,
                 nogood=False):
        '''
        Initialize q3dout object.
        '''
        self.fitrange = fitrange
        self.wave = np.float64(wave)
        self.spec = np.float64(spec)
        self.spec_err = np.float64(spec_err)

        self.fluxunit = fluxunit
        self.waveunit = waveunit
        self.fluxnorm = fluxnorm
        self.pixarea_sqas = pixarea_sqas

        self.fitran_indx = fitran_indx
        self.gd_indx = gd_indx

        self.docontfit = False
        self.dolinefit = False

        self.nogood = nogood # no good data present

        self.col = col
        self.row = row

    def init_linefit(self, linelist, linelabel, maxncomp, covar=None,
                     dof=None, fitstatus=None,
                     line_dat=None, line_fit=None, nfev=None,
                     noemlinmask=None, redchisq=None, param=None,
                     parinit=None, perror=None, perror_resid=None,
                     perror_errspec=None, siglim=None):

        self.dolinefit = True

        # Line fit parameters
        self.covar = covar
        self.dof = dof
        self.line_dat = line_dat
        self.line_fit = line_fit
        self.linelist = linelist
        self.linelabel = linelabel
        self.fitstatus = fitstatus
        self.maxncomp = maxncomp
        self.nfev = nfev
        self.noemlinmask = noemlinmask
        self.parinit = parinit
        self.param = param
        self.perror = perror
        self.perror_errspec = perror_errspec
        self.perror_resid = perror_resid
        self.redchisq = redchisq
        self.siglim = siglim

    def init_contfit(self, ct_method='CONTINUUM SUBTRACTED',
                     ct_coeff=None, ct_ebv=None, ct_add_poly_weights=None,
                     ct_ppxf_sigma=None, ct_ppxf_sigma_err=None,
                     ct_rchisq=None, cont_dat=None, cont_fit=None,
                     ct_indx=None, zstar=None, zstar_err=None):
                     # cont_fit_pretweak=None

        self.docontfit = True

        # Continuum fit parameters
        self.ct_method = ct_method
        self.ct_coeff = ct_coeff
        self.ct_ebv = ct_ebv
        self.zstar = zstar
        self.zstar_err = zstar_err
        self.ct_add_poly_weights = ct_add_poly_weights
        self.ct_ppxf_sigma = ct_ppxf_sigma
        self.ct_ppxf_sigma_err = ct_ppxf_sigma_err
        self.ct_rchisq = ct_rchisq

        self.cont_dat = cont_dat
        self.cont_fit = cont_fit
        # self.cont_fit_pretweak = cont_fit_pretweak

        # gd_indx is applied, and then ct_indx
        self.ct_indx = ct_indx

    def sepfitpars(self, waveran=None, doublets=None, ignoreres=False):
        """
        Convert output of LMFIT, with best-fit line parameters in a single
        array, into a dictionary with separate arrays for different line
        parameters. Compute total line fluxes from the best-fit line
        parameters.

        Parameters
        ----------
        waveran: ndarray, optional
            Set to upper and lower limits to return line parameters only
            for lines within the given wavelength range. Lines outside this
            range have fluxes set to 0.
        doublets: dict, optional
            Dictionary of doublet lines for which to combine fluxes.
        ignoreres: bool, optional
            Ignore spectral resolution in computing observed sigmas and peak 
            fluxes. This is mainly for backward compatibility with old versions,
            which did not store the spectral resolution in an easily accessible
            way in the specConv object.

        """

        # pass on if no lines were fit
        if not self.dolinefit:

            pass

        else:

            basearr = np.full(self.linelist['name'].T.data.shape, np.nan)
            basearr_1comp = basearr
            if self.maxncomp > 1:
                basearr = np.tile(basearr, (self.maxncomp, 1))
            flux = Table(basearr, names=self.linelist['name'])
            fluxerr = Table(basearr, names=self.linelist['name'])
            fluxpk = Table(basearr, names=self.linelist['name'])
            fluxpkerr = Table(basearr, names=self.linelist['name'])
            fluxpk_obs = Table(basearr, names=self.linelist['name'])
            fluxpkerr_obs = Table(basearr, names=self.linelist['name'])
            sigma = Table(basearr, names=self.linelist['name'])
            sigmaerr = Table(basearr, names=self.linelist['name'])
            sigma_obs = Table(basearr, names=self.linelist['name'])
            sigmaerr_obs = Table(basearr, names=self.linelist['name'])
            wave = Table(basearr, names=self.linelist['name'])
            waveerr = Table(basearr, names=self.linelist['name'])

            tf = Table(basearr_1comp, names=self.linelist['name'])
            tfe = Table(basearr_1comp, names=self.linelist['name'])

            #   Populate Tables

            for line in self.linelist['name']:

                for i in range(0, self.maxncomp):

                    # indices
                    lmline = lmlabel(line)
                    ifluxpk = f'{lmline.lmlabel}_{i}_flx'
                    isigma = f'{lmline.lmlabel}_{i}_sig'
                    iwave = f'{lmline.lmlabel}_{i}_cwv'
                    ispecres = f'{lmline.lmlabel}_{i}_SPECRES'

                    # make sure the line was fit -- necessary if, e.g., #
                    # components reset to 0 by checkcomp
                    if iwave in self.param.keys():

                        wave[line][i] = self.param[iwave]
                        sigma[line][i] = self.param[isigma]
                        fluxpk[line][i] = self.param[ifluxpk]

                        waveerr[line][i] = self.perror[iwave]
                        sigmaerr[line][i] = self.perror[isigma]
                        fluxpkerr[line][i] = self.perror[ifluxpk]

                        specres = self.param[ispecres]

                        # Check to see if line went to 0 boundary; possibly
                        # relevant only for leastsq. When this happens with 
                        # leastsq, uncertainties don't get calculated and code
                        # to calculate observed sigma and fluxes will fail.
                        # Setting flux to 0 and fluxerr to 0 will allow 
                        # checkcomp to ignore this line.
                        zeroflux = False
                        if np.allclose(fluxpk[line][i], self.parinit[ifluxpk].min):
                            zeroflux = True
                            fluxpk[line][i] = 0.
                            fluxpkerr[line][i] = 0.

                        # Check to see if uncertainties estimated. If not, there
                        # was a problem in leastsq (or perhaps another fitting 
                        # routine) and we should set flux to 0 and fluxerr to 0.
                        # This will allow checkcomp to ignore this line.
                        nouncert = False
                        if np.isnan([waveerr[line][i], 
                                     sigmaerr[line][i], 
                                     fluxpkerr[line][i]]).any():
                            nouncert = True
                            fluxpk[line][i] = 0.
                            fluxpkerr[line][i] = 0.

                        # Compute observed sigma and fluxpk, taking into account
                        # spectral resolution.
                        if specres is not None and ignoreres is False and \
                            zeroflux is False and nouncert is False:    
                            # Get spectral resolution
                            Ruse = specres.get_R(wave[line][i])
                            sigma_obs[line][i] = \
                                np.sqrt(self.param[isigma]**2 +
                                        (c.to('km/s').value/Ruse/
                                         gaussian_sigma_to_fwhm)**2)
                            sigmaerr_obs[line][i] = self.perror[isigma] * \
                                sigma_obs[line][i]/sigma[line][i]
                            fluxpk_obs[line][i] = self.param[ifluxpk] * \
                                sigma[line][i]/sigma_obs[line][i]
                            fluxpkerr_obs[line][i] = self.perror[ifluxpk]
                        else:
                            sigma_obs[line][i] = sigma[line][i]
                            sigmaerr_obs[line][i] = sigmaerr[line][i]
                            fluxpk_obs[line][i] = fluxpk[line][i]
                            fluxpkerr_obs[line][i] = fluxpkerr[line][i]

                # Because of the way these lines are tied to others (with a division!) they
                # can yield NaNs in components that aren't fit. Correct this.
                # if line eq '[SII]6731' OR line eq 'Hbeta' OR line eq '[NI]5189' then begin

                #if (line == '[SII]6731') or (line == '[NI]5189'):
                #    inan = np.where(np.isfinite(fluxpk[line]) == False)
                #    ctnan = np.count_nonzero(np.isfinite(fluxpk[line]) == False)
                #    if ctnan > 0:
                #        fluxpk[line][inan] = 0.
                #        fluxpkerr[line,inan] = 0.
                #        fluxpk_obs[line,inan] = 0.
                #        fluxpkerr_obs[line,inan] = 0.

            #for line in self.linelist['name']:

            # Fix flux errors associated with line ratios. E.g., [NII]/Halpha is a fitted
            # parameter and [NII]6583 is tied to it, so the formal error in [NII]6583
            # flux is 0. Add errors in Halpha and [NII]/Halpha in quadrature to get
            #  error in [NII]6583.

                # if (line == "[NII]6583") and (ctn2ha > 0):
                #     fluxpkerr_obs[line][0:ctn2ha] = fluxpk_obs[line][0:ctn2ha] * \
                #     np.sqrt((perror[in2ha]/param[in2ha])**2. + \
                #     (fluxpkerr_obs['Halpha'][0:ctn2ha]/ fluxpk_obs['Halpha'][0:ctn2ha])**2.)
                #     # In pegged case, set errors equal to each other
                #     ipegged = np.where((perror[in2ha] == 0.) and (param[in2ha] != 0.))
                #     ctpegged = np.count_nonzero((perror[in2ha] == 0.) and (param[in2ha] != 0.))
                #     if ctpegged > 0:
                #         fluxpkerr_obs['[NII]6583'][ipegged] = \
                #         fluxpkerr_obs['Halpha'][ipegged]
                #     fluxpkerr[line] = fluxpkerr_obs[line]

                # if (line == '[SII]6731') and (cts2rat > 0):
                #     fluxpkerr_obs[line][0:cts2rat] = \
                #     fluxpk_obs[line][0:cts2rat] * \
                #     np.sqrt((perror[is2rat]/param[is2rat])**2. + \
                #     (fluxpkerr_obs['[SII]6716'][0:cts2rat] / \
                #     fluxpk_obs['[SII]6716'][0:cts2rat])**2.)
                #     # In pegged case, set errors equal to each other
                #     ipegged = np.where((perror[is2rat] == 0.) and (param[is2rat] != 0.))
                #     ctpegged = np.count_nonzero((perror[in2ha] == 0.) and (param[in2ha] != 0.))
                #     if ctpegged > 0:
                #         fluxpkerr_obs['[SII]6731'][ipegged] = \
                #         fluxpkerr_obs['[SII]6716'][ipegged]
                #     fluxpkerr[line] = fluxpkerr_obs[line]

                # if (line == '[NI]5198') and (ctn1rat > 0):
                #     fluxpkerr_obs[line][0:ctn1rat] = \
                #     fluxpk_obs[line][0:ctn1rat] * \
                #     np.sqrt((perror[in1rat]/param[in1rat])**2. + \
                #     (fluxpkerr_obs['[NI]5200'][0:ctn1rat]/ \
                #     fluxpk_obs['[NI]5200'][0:ctn1rat])**2.)

                #     fluxpkerr[line] = fluxpkerr_obs[line]

                #     # In pegged case, set errors equal to each other
                #     ipegged = np.where((perror[in1rat] == 0.) and (param[in1rat] != 0.))
                #     ctpegged = np.count_nonzero((perror[in1rat] == 0.) and (param[in1rat] != 0.))
                #     if ctpegged > 0:
                #         fluxpkerr_obs['[NI]5198'][ipegged] = fluxpkerr_obs['[NI]5200'][ipegged]

                #     fluxpkerr[line] = fluxpkerr_obs[line]

                # if (line == 'Hbeta') and (np.count_nonzero(self.linelist['name'] == 'Halpha') == 1):
                #     # If Halpha/Hbeta goes belowlower limit, then we re-calculate the errors
                #     # add discrepancy in quadrature to currently calculated error. Assume
                #     # error in fitting is in Hbeta and adjust accordingly.
                #     fha = fluxpk['Halpha']
                #     ihahb = np.where((fluxpk['Halpha'] > 0.) and (fluxpk['Hbeta'] > 0.))
                #     cthahb = np.count_nonzero((fluxpk['Halpha'] > 0.) and (fluxpk['Hbeta'] > 0.))
                #     if cthahb > 0.:
                #         itoolow = np.where(fluxpk['Halpha'][ihahb]/fluxpk['Hbeta'] < 2.86)
                #         cttoolow = np.count_nonzero(fluxpk['Halpha'][ihahb]/fluxpk['Hbeta'] < 2.86)
                #         if cttoolow > 0:
                #             fluxpkdiff = fluxpk[line][itoolow] - fluxpk['Halpha'][itoolow]/2.86
                #             fluxpk[line][itoolow] -= fluxpkdiff
                #             fluxpk_obs[line][itoolow] -= fluxpkdiff
                #             fluxpkerr[line][itoolow] = np.sqrt(fluxpkerr[line][itoolow]**2. + fluxpkdiff**2.)
                #             fluxpkerr_obs[line][itoolow] = np.sqrt(fluxpkerr_obs[line][itoolow]**2. + fluxpkdiff**2.)

                # if (line == '[OII]3729') and (cto2rat > 0.):
                #     fluxpkerr_obs[line][0:cto2rat] = \
                #     fluxpk_obs[line][0:cto2rat]*np.sqrt( \
                #     (perror[io2rat]/param[io2rat])**2. + \
                #     (fluxpkerr_obs['[OII]3726'][0:cto2rat]/ \
                #     fluxpk_obs['[OII]3726'][0:cto2rat])**2.)
                #     # In pegged case, set errors equal to each other
                #     ipegged = np.where((perror[io2rat] == 0.) and (param[io2rat] != 0.))
                #     ctpegged = np.count_nonzero((perror[io2rat] == 0.) and (param[io2rat] != 0.))
                #     if ctpegged > 0:
                #         fluxpkerr_obs['[OII]3729'][ipegged] = fluxpkerr_obs['[OII]3726'][ipegged]
                #     fluxpkerr[line] = fluxpkerr_obs[line]

                        # Add back in spectral resolution

                        # Make sure we're not adding something to 0 --
                        # i.e. the component wasn't fit.
                        # Can't use sigma = 0 as criterion since the line could be
                        # fitted but unresolved.
                        if fluxpk[line][i] > 0:
                            #sigmatmp = \
                            #    sigma[line][i]/(constants.c/1.e3)*wave[line][i]
                            # in km/s
                            #sigma_obs[line][i] = \
                            #    sigmatmp/wave[line][i]*(constants.c/1.e3)
                            # error propagation for adding in quadrature
                            #sigmaerr_obs[line][i] *= \
                            #    sigma[line][i]/(constants.c/1.e3)*wave[line][i] /\
                            #    sigmatmp
                            # Correct peak flux and error for deconvolution
                            #fluxpk[line][i] *= sigma_obs[line][i]/sigma[line][i]
                            #fluxpkerr[line][i] *= sigma_obs[line][i]/sigma[line][i]

                            # Compute total Gaussian flux
                            # sigma and error need to be in wavelength space
                            gflux = \
                                gaussflux(fluxpk[line][i],
                                          sigma[line][i] /
                                          (constants.c / 1.e3) * wave[line][i],
                                          normerr=fluxpkerr[line][i],
                                          sigerr=sigmaerr[line][i] /
                                          (constants.c / 1.e3) * wave[line][i])
                        else:
                            gflux = {'flux': 0., 'flux_err': 0.}
                        flux[line][i] = gflux['flux']
                        fluxerr[line][i] = gflux['flux_err']

                        # Set fluxes to 0 outside of wavelength range,
                        # or if NaNs or infinite errors

                        inoflux_wr = False
                        if waveran:
                            inoflux_wr = \
                                ((waveran[0] > wave[line][i] *
                                  (1. - 3.*sigma[line][i] /
                                   (constants.c/1.e3))) or
                                 (waveran[1] < wave[line][i] *
                                  (1. + 3.*sigma[line][i] /
                                   (constants.c/1.e3))))

                        inoflux = \
                            ((np.isfinite(fluxerr[line][i]) is False) or
                             (np.isfinite(fluxpkerr[line][i]) is False))

                        if inoflux_wr or inoflux:
                            flux[line][i] = 0.
                            fluxerr[line][i] = 0.
                            fluxpk[line][i] = 0.
                            fluxpkerr[line][i] = 0.
                            fluxpk_obs[line][i] = 0.
                            fluxpkerr_obs[line][i] = 0.

                # Compute total fluxes summed over components
                igd = np.where(flux[line] > 0.)
                ctgd = np.count_nonzero(flux[line] > 0.)
                if ctgd > 0:
                    tf[line][0] = np.sum(flux[line][igd])
                    tfe[line][0] = np.sqrt(np.sum(fluxerr[line][igd]**2.))
                else:
                    tf[line][0] = 0.
                    tfe[line][0] = 0.

            # Special doublet cases: combine fluxes from each line
            if doublets is not None:

                for (name1, name2) in zip(doublets['line1'],
                                          doublets['line2']):
                    if name1 in self.linelist['name'] and \
                        name2 in self.linelist['name']:
                        # new line label
                        dkey = name1+'+'+name2
                        # add fluxes
                        tf[dkey] = tf[name1]+tf[name2]
                        flux[dkey] = flux[name1]+flux[name2]
                        fluxpk[dkey] = fluxpk[name1]+fluxpk[name2]
                        fluxpk_obs[dkey] = fluxpk_obs[name1]+fluxpk_obs[name2]
                        # add flux errors in quadrature
                        tfe[dkey] = np.sqrt(tfe[name1]**2. + tfe[name2]**2.)
                        fluxerr[dkey] = np.sqrt(fluxerr[name1]**2. +
                                                fluxerr[name2]**2.)
                        fluxpkerr[dkey] = np.sqrt(fluxpkerr[name1]**2. +
                                                  fluxpkerr[name2]**2.)
                        fluxpkerr_obs[dkey] = np.sqrt(fluxpkerr_obs[name1]**2. +
                                                      fluxpkerr_obs[name2]**2.)
                        # average waves and sigmas and errors
                        wave[dkey] = (wave[name1]+wave[name2])/2.
                        waveerr[dkey] = (waveerr[name1]+waveerr[name2])/2.
                        sigma[dkey] = (sigma[name1]+sigma[name2])/2.
                        sigmaerr[dkey] = (sigmaerr[name1]+sigmaerr[name2])/2.
                        sigma_obs[dkey] = (sigma_obs[name1]+sigma_obs[name2])/2.
                        sigmaerr_obs[dkey] = (sigmaerr_obs[name1] +
                                              sigmaerr_obs[name2])/2.

            self.line_fitpars = {'flux': flux, 'fluxerr': fluxerr,
                                 'fluxpk': fluxpk, 'fluxpkerr': fluxpkerr,
                                 'wave': wave, 'waveerr': waveerr,
                                 'sigma': sigma, 'sigmaerr': sigmaerr,
                                 'sigma_obs': sigma_obs,
                                 'sigmaerr_obs': sigmaerr_obs,
                                 'fluxpk_obs': fluxpk_obs,
                                 'fluxpkerr_obs': fluxpkerr_obs,
                                 'tflux': tf, 'tfluxerr': tfe}

    def sepcontpars(self, q3di):
        '''
        '''
        q3dii = q3dutil.get_q3dio(q3di)

        # Compute PPXF components: additive polynomial and stellar fit
        if q3dii.decompose_ppxf_fit:
            self.add_poly_degree = 4  # should match fitspec
            if q3dii.argscontfit is not None:
                if 'add_poly_degree' in q3dii.argscontfit:
                    add_poly_degree = \
                        q3dii.argscontfit['add_poly_degree']
            # Compute polynomial
            dumy_log, wave_log, _ = \
                log_rebin([self.wave[0],
                           self.wave[len(self.wave)-1]],
                          self.spec)
            xnorm = np.linspace(-1., 1., len(wave_log))
            cont_fit_poly_log = 0.0
            for k in range(0, add_poly_degree):
                cfpllegfun = np.polynomial.legendre(k)
                cont_fit_poly_log += cfpllegfun(xnorm) * \
                    self.ct_add_poly_weights[k]
            interpfunction = \
                interp1d(cont_fit_poly_log, wave_log, kind='linear',
                         fill_value="extrapolate")
            self.cont_fit_poly = interpfunction(np.log(self.wave))
            # Compute stellar continuum
            self.cont_fit_stel = np.subtract(self.cont_fit,
                                             self.cont_fit_poly)

        # Compute FITQSOHOST components
        elif q3dii.decompose_qso_fit:

            self.qsomod = 0.
            self.hostmod = 0.
            self.polymod_refit = np.zeros(len(self.wave),
                                          dtype='float64')

            if q3dii.fcncontfit == 'fitqsohost':
                if 'qsoord' in q3dii.argscontfit:
                    qsoord = q3dii.argscontfit['qsoord']
                else:
                    qsoord = None

                if 'hostord' in q3dii.argscontfit:
                    hostord = q3dii.argscontfit['hostord']
                else:
                    hostord = None
                # copy of BLR fit parameters from argscontfit
                if 'blrpar' in q3dii.argscontfit:
                    self.blrpar = q3dii.argscontfit['blrpar']
                else:
                    self.blrpar = None
                # default here must be same as in IFSF_FITQSOHOST
                if 'add_poly_degree' in q3dii.argscontfit:
                    self.add_poly_degree = \
                        q3dii.argscontfit['add_poly_degree']
                else:
                    self.add_poly_degree = 30

                # Get and renormalize template
                qsotemplate = \
                    np.load(q3dii.argscontfit['qsoxdr'],
                            allow_pickle='TRUE').item()
                qsowave = qsotemplate['wave']
                qsoflux_full = qsotemplate['flux']

                #iqsoflux = \
                #    np.where((qsowave >= q3dii.fitrange[0]) &
                #             (qsowave <= q3dii.fitrange[1]))
                qsoflux = qsoflux_full[self.fitran_indx]

                # If polynomial residual is re-fit with PPXF,
                # compute polynomial component
                if 'refit' in q3dii.argscontfit and \
                    'args_questfit' not in q3dii.argscontfit:

                    par_qsohost = self.ct_coeff['qso_host']
                    # par_stel = self.ct_coeff']['stel']
                    dumy_log, wave_rebin, _ = \
                        log_rebin([self.wave[0],
                                   self.wave
                                   [len(self.wave)-1]],
                                  self.spec)
                    xnorm = np.linspace(-1., 1., len(wave_rebin))
                    if self.add_poly_degree > 0:
                        par_poly = self.ct_coeff['poly']
                        polymod_log = \
                            np.polynomial.legendre.legval(xnorm, par_poly)
                        interpfunct = \
                            interp1d(wave_rebin, polymod_log,
                                     kind='cubic',
                                     fill_value="extrapolate")
                        self.polymod_refit = \
                            interpfunct(np.log(self.wave))

                # Refitting with questfit in the MIR
                elif 'refit' in q3dii.argscontfit and \
                    q3dii.argscontfit['refit'] == 'questfit':

                    par_qsohost = self.ct_coeff['qso_host']
                else:
                    par_qsohost = self.ct_coeff

                # produce fit with template only and with template + host. Also
                # output QSO multiplicative polynomial
                #qsomod_polynorm = 0.
                self.qsomod = \
                    qsohostfcn(self.wave, params_fit=par_qsohost,
                               qsoflux=qsoflux, qsoonly=True,
                               blrpar=self.blrpar, qsoord=qsoord,
                               hostord=hostord)
                #self.hostmod = self.cont_fit_pretweak - self.qsomod
                self.hostmod = self.cont_fit - self.qsomod

                # if continuum is tweaked in any region, subide resulting residual
                # proportionality @ each wavelength btwn qso and host components
                # qsomod_notweak = qsomod
                # if q3dii.tweakcntfit is not None:
                #     modresid = self.cont_fit - self.cont_fit_pretweak
                #     inz = np.where((qsomod != 0) & (hostmod != 0))[0]
                #     qsofrac = np.zeros(len(qsomod))
                #     for ind in inz:
                #         qsofrac[ind] = qsomod[ind] / \
                #             (qsomod[ind] + hostmod[ind])
                #     qsomod += modresid * qsofrac
                #     hostmod += modresid * (1.0 - qsofrac)
                # components of qso fit for plotting
                self.qsomod_normonly = qsoflux
                if self.blrpar is not None:
                    self.qsomod_blronly = \
                        qsohostfcn(self.wave,
                                   params_fit=par_qsohost,
                                   qsoflux=qsoflux, blronly=True,
                                   blrpar=self.blrpar, qsoord=qsoord,
                                   hostord=hostord)
                else:
                    self.qsomod_blronly = 0.

            # CB: adding option to plot decomposed QSO fit if questfit is used
            elif q3dii.fcncontfit == 'questfit':
                from q3dfit.contfit import quest_extract_QSO_contrib
                self.qsomod, self.hostmod, qsomod_intr, hostmod_intr = \
                    quest_extract_QSO_contrib(self.ct_coeff, q3dii)
                # qsomod_polynorm = 1.
                # qsomod_notweak = self.qsomod
                qsoflux = self.qsomod.copy()/np.median(self.qsomod)
                self.qsomod_normonly = qsoflux
                self.blrpar = None
                self.qsomod_blronly = 0.

        #self.host_spec = self.spec - self.qsomod
        #self.host_cont_dat = self.cont_dat - self.qsomod
        #self.host_cont_fit = self.cont_fit - self.qsomod

        #self.qso_spec = self.spec - self.hostmod
        #self.qso_cont_dat = self.cont_dat - self.hostmod
        #self.qso_cont_fit = self.cont_fit - self.hostmod

    def plot_line(self, q3di, fcn='plotline', savefig=False, outfile=None,
                  plotargs={}):
        '''
        '''

        # if inline is False:
        #    mpl.use('agg')

        q3dii = q3dutil.get_q3dio(q3di)

        if self.dolinefit:
            mod = import_module('q3dfit.plot')
            plotline = getattr(mod, fcn)

            if savefig:
                if outfile is None:
                    # use default label
                    if hasattr(self, 'filelab'):
                        outfile = self.filelab+'_lin'
                    # make sure an outfile is available if the default is not
                    # specified
                    else:
                        print('plot_line: need to specify outfile')
                else:
                    outfile = outfile + '_lin',

            if q3dii.spect_convol:
                # instantiate specConv object
                specConv = q3dutil.get_dispersion(q3dii)
            else:
                specConv = None

            plotline(self, savefig=savefig, outfile=outfile, specConv=specConv,
                     **plotargs)
        else:
            print('plot_line: no lines to plot!')

    def plot_cont(self, q3di, fcn='plotcont', savefig=False, outfile=None,
                  plotargs={}):
        '''
        Continuum plotting function
        '''

        # if inline is False:
        #    mpl.use('agg')

        if self.docontfit:

            q3dii = q3dutil.get_q3dio(q3di)

            mod = import_module('q3dfit.plot')
            plotcont = getattr(mod, fcn)

            if savefig:
                if outfile is None:
                    if hasattr(self, 'filelab'):
                        outfile = self.filelab
                    else:
                        print('plot_line: need to specify outfile')

            if outfile is not None:
                outfilecnt = outfile + '_cnt',
                outfileqso = outfile + '_cnt_qso',
                outfilehost = outfile + '_cnt_host',
            else:
                outfilecnt = None
                outfileqso = None
                outfilehost = None

            # Plot QSO and host only continuum fit
            if q3dii.decompose_qso_fit:

                q3do_host = copy.deepcopy(self)
                q3do_qso = copy.deepcopy(self)

                q3do_host.spec -= self.qsomod
                q3do_host.cont_dat -= self.qsomod
                q3do_host.cont_fit -= self.qsomod

                q3do_qso.spec -= self.hostmod
                q3do_qso.cont_dat -= self.hostmod
                q3do_qso.cont_fit -= self.hostmod

                if np.sum(self.cont_fit) != 0.0:

                    # Host only plot
                    # assume argscontfit exists if we're doing fitqsohost
                    if 'refit' in q3dii.argscontfit:
                        compspec = np.array([self.polymod_refit,
                                             self.hostmod-self.polymod_refit])
                        comptitles = ['ord. ' + str(self.add_poly_degree) +
                                      ' Leg. poly.', 'stel. temp.']
                    else:
                        compspec = [self.hostmod.copy()]
                        comptitles = ['exponential terms']
                    plotcont(q3do_host, savefig=savefig, outfile=outfilehost,
                             compspec=compspec, comptitles=comptitles,
                             title='Host', fitran=q3dii.fitrange,
                             q3di=q3dii, **plotargs)

                    # QSO only plot
                    if self.blrpar is not None and \
                        max(self.qsomod_blronly) != 0.:
                        qsomod_blrnorm = np.median(self.qsomod) / \
                            max(self.qsomod_blronly)
                        compspec = np.array([self.qsomod_normonly,
                                             self.qsomod_blronly *
                                             qsomod_blrnorm])
                        comptitles = ['raw template', 'scattered*' +
                                      str(qsomod_blrnorm)]
                    else:
                        compspec = [self.qsomod_normonly.copy()]
                        comptitles = ['raw template']

                    if q3dii.fcncontfit != 'questfit':
                        plotcont(q3do_qso, savefig=savefig, outfile=outfileqso,
                                 compspec=compspec, comptitles=comptitles,
                                 title='QSO', fitran=q3dii.fitrange,
                                 q3di=q3dii, **plotargs)
                    else:
                        plotcont(q3do_qso, savefig=savefig, outfile=outfileqso,
                                 compspec=[q3do_qso.cont_fit],
                                 title='QSO', fitran=q3dii.fitrange,
                                 comptitles=['QSO'], q3di=q3dii,
                                 **plotargs)

            if sum(self.cont_fit) != 0.0:

                if q3dii.decompose_qso_fit:
                    plotcont(self, savefig=savefig, outfile=outfilecnt,
                             compspec=np.array([self.qsomod, self.hostmod]),
                             title='Total', comptitles=['QSO', 'host'],
                             fitran=q3dii.fitrange, q3di=q3dii,
                             **plotargs)

                elif q3dii.decompose_ppxf_fit:
                    plotcont(self, savefig=savefig, outfile=outfilecnt,
                             compspec=np.array([self.cont_fit_stel,
                                                self.cont_fit_poly]),
                             title='Total',
                             comptitles=['stel. temp.', 'ord. ' +
                                         str(self.add_poly_degree) +
                                         'Leg.poly'],
                             fitran=q3dii.fitrange, q3di=q3dii,
                             **plotargs)
                else:
                    plotcont(self, savefig=savefig, outfile=outfilecnt,
                             fitran=q3dii.fitrange,
                             q3di=q3dii,
                             ct_coeff=self.ct_coeff,
                             title='Total', **plotargs)

                # Plot continuum
                # Make sure fit doesn't indicate no continuum; avoids
                # plot range error in continuum fitting routine,
                # as well as a blank plot!
                # if not noplots and q3dii.argscontfit is not None:
                #     if 'plot_decomp' in q3dii.argscontfit'].keys():
                #         if q3dii.argscontfit']['plot_decomp']:
                #             from q3dfit.plot_quest import plot_quest
                #             if not q3do.noemlinfit']:
                #                 lam_lines = \
                #                     q3do.linelist']['lines'].tolist()
                #             else:
                #                 lam_lines = []
                #             plot_quest(q3do.wave'],
                #                        q3do.cont_dat']+q3do.emlin_dat'],
                #                        q3do.cont_fit']+q3do.emlin_fit'],
                #                        q3do.ct_coeff'], q3dii,
                #                        lines=lam_lines,
                #                        linespec=q3do.emlin_fit'])

        else:
            print('plot_cont: no continuum to plot!')
