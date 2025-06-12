# -*- coding: utf-8 -*-
from __future__ import annotations

import copy
import numpy as np
from typing import Literal, Optional

from numpy.typing import ArrayLike

from astropy.constants import c
from astropy.stats import gaussian_sigma_to_fwhm
from astropy.table import Table
from importlib import import_module
from lmfit import Parameters
from ppxf.ppxf_util import log_rebin
from scipy import constants
from scipy.interpolate import interp1d

from . import q3dutil, q3din
from q3dfit.qsohostfcn import qsohostfcn
from q3dfit.contfit import readcf
from q3dfit.exceptions import InitializationError

class q3dout:
    '''
    This object is created by :py:mod:`~q3dfit.q3df` when running on any single spaxel. 
    It collects the output and contains functions to generate plots for a single
    spaxel.

    For multi-spaxel post-processing, see :py:class:`~q3dfit.q3dpro.q3dpro`.

    Parameters
    ----------
    wave
        Wavelength array of data (limited to the fit range by 
        :py:func:`~q3dfit.fitspec.fitspec`). Also updates :py:attr:`~q3dfit.q3dout.q3dout.wave`.
    spec
        Flux array (limited to the fit range by 
        :py:func:`~q3dfit.fitspec.fitspec`). 
        Also updates :py:attr:`~q3dfit.q3dout.q3dout.spec`.
    spec_err
        Flux error array (limited to the fit range by 
        :py:func:`~q3dfit.fitspec.fitspec`). 
        Also updates :py:attr:`~q3dfit.q3dout.q3dout.spec_err`.


    Attributes
    ----------
    fitrange: np.ndarray
        Wavelength range for fitting. Added by constructor. Defaults to None.
    col : int
        Column index. Added by constructor. Defaults to None.
    row : int
        Row index. Added by constructor. Defaults to None.
    gd_indx : ndarray
        Good data indices. Added by constructor. Defaults to None.
    fitran_indx : ndarray
        Fit range indices. Added by constructor. Defaults to None.
    fluxunit : str
        Flux unit. Added by constructor. Defaults to `erg/s/cm^2/micron`.
    waveunit : str
        Wavelength unit. Added by constructor. Defaults to 'micron'.
    fluxnorm : float
        Flux normalization. Added by constructor. Defaults to 1.0.
    pixarea_sqas : float
        Pixel area in square arcseconds. Added by constructor. Defaults to None.
    nogood : bool
        No good data present. Added by constructor. Defaults to False.
    docontfit : bool
        Do continuum fit. Added by constructor. Defaults to False. Updated by
        :py:func:`~q3dfit.q3dout.q3dout.init_contfit()`.
    dolinefit : bool
        Do line fit. Added by constructor. Defaults to False. Updated by
        :py:func:`~q3dfit.q3dout.q3dout.init_linefit()`.
    add_poly_degree
        Degree of additive polynomial in the pPXF fit.
        Default is -1, which means no additive polynomial. Added by
        :py:func:`~q3dfit.q3dout.q3dout.sepcontpars()`.
    stelmod: float
        Stellar component of continuum fit. Added/updated by 
        :py:func:`~q3dfit.q3dout.q3dout.sepcontpars()`.
    qsomod : float
        QSO component of continuum fit. Added/updated by 
        :py:func:`~q3dfit.q3dout.q3dout.sepcontpars()`.
    hostmod : float
        Host component of continuum fit. Added/updated by 
        :py:func:`~q3dfit.q3dout.q3dout.sepcontpars()`.
    polymod_refit : ndarray
        Polynomial component of continuum fit, if refit with fitqsohost. 
        Added/updated by :py:func:`~q3dfit.q3dout.q3dout.sepcontpars()`.
    line_fitpars : dict
        Line fit parameters. Added/updated by 
        :py:func:`~q3dfit.q3dout.q3dout.sepfitpars()`. Contains the following keys:\n
        - `flux`: Table with total fluxes of emission lines.
        - `fluxerr`: Table with total flux errors of emission lines.
        - `fluxpk`: Table with peak fluxes of emission lines.
        - `fluxpkerr`: Table with peak flux errors of emission lines.
        - `fluxpk_obs`: Table with peak fluxes of emission lines, corrected for spectral resolution.
        - `fluxpkerr_obs`: Table with peak flux errors of emission lines, corrected for spectral resolution.
        - `sigma`: Table with line widths (sigmas) of emission lines.
        - `sigmaerr`: Table with line width errors (sigmas) of emission lines.
        - `sigma_obs`: Table with line widths (sigmas) of emission lines, corrected for spectral resolution.
        - `sigmaerr_obs`: Table with line width errors (sigmas) of emission lines, corrected for spectral resolution.
        - `wave`: Table with line central wavelengths of emission lines.
        - `waveerr`: Table with line central wavelength errors of emission lines.
        - `tflux`: Total fluxes of emission lines, summed over components.
        - `tfluxerr`: Total flux errors of emission lines, summed over components.
    filelab : str
        File label for plotting methods. Added/updated by 
        :py:func:`~q3dfit.q3dout.q3dout.load_q3dout()`.
    '''

    def __init__(self,
                 wave: ArrayLike,
                 spec: ArrayLike,
                 spec_err: ArrayLike,
                 fitrange: Optional[ArrayLike]=None,
                 col: Optional[int]=None,
                 row: Optional[int]=None,
                 gd_indx: Optional[ArrayLike]=None,
                 fitran_indx: Optional[ArrayLike]=None,
                 fluxunit: str='erg/s/cm^2/micron',
                 waveunit: str='micron',
                 fluxnorm: float=1.,
                 pixarea_sqas: Optional[float]=None,
                 nogood: bool=False):
        
        self.wave = np.array(wave, dtype=np.float64)
        self.spec = np.array(spec, dtype=np.float64)
        self.spec_err = np.array(spec_err, dtype=np.float64)

        self.fitrange = fitrange

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

    def init_linefit(self,
                     linelist: Table,
                     linelabel: list[str],
                     maxncomp: int=1,
                     line_dat: Optional[ArrayLike]=None,
                     parinit: Optional[Parameters]=None,
                     line_fit: Optional[ArrayLike]=None,
                     param: Optional[dict]=None,
                     perror: Optional[dict]=None,
                     perror_resid: Optional[dict]=None,
                     perror_errspec: Optional[dict]=None,
                     fitstatus: Optional[int]=None,
                     dof: Optional[int]=None,
                     redchisq: Optional[float]=None,
                     nfev: Optional[int]=None,
                     covar: Optional[ArrayLike]=None):
        '''
        Initialize output line fit parameters.

        Parameters
        ----------
        linelist
            Emission line labels and rest frame wavelengths, as part of an astropy Table
            output by :py:func:`~q3dfit.linelist.linelist`. Passed directly from
            the `listlines` parameter of :py:func:`~q3dfit.fitspec.fitspec` to this function.
        linelabel
            List of lines to fit. Copy of :py:attr:`~q3dfit.q3din.q3din.lines` 
            from :py:class:`~q3dfit.q3din.q3din`.
        maxncomp
            Maximum number of components. Copy of :py:attr:`~q3dfit.q3din.q3din.maxncomp`.
        line_dat
            Continuum-subtracted fluxes for emission line fit, 
            from :py:func:`~q3dfit.fitspec.fitspec`. Includes data
            within the fit range.
        parinit
            Initial parameters, output from the line initialization routine.
        param
            Dictionary with parameter names as keys, and best-fit values as values.
            Copy of :py:attr:`~lmfit.ModelResult.best_values`.
        perror
            Parameter errors. Dictionary with parameter names as keys, and
            errors as values. Values are copies of :py:attr:`~lmfit.Parameters.stderr`.
            If there is no error in the peak flux values, or it is a NaN,
            it is set to the value computed from the error spectrum (see below).
            If `~q3dfit.q3din.q3din.perror_useresid` is set to True and
            the error from the flux residual is greater than the
            error from the covariance matrix, then the error is set to the
            value computed from the residuals (see below).
        perror_resid
            Copy of perror, but with peak flux errors computed from the
            residuals of the fit, rather than from the covariance matrix.
            The error is the standard deviation of the residuals
            in a window around the line, with the window size in pixels set by
            :py:attr:`~q3dfit.q3din.q3din.perror_residwin`.
        perror_errspec
            Copy of perror, but with peak flux errors computed from the
            error spectrum, rather than from the covariance matrix.
            The error is the median of the error spectrum
            in a window around the line, with the window size in pixels set by
            :py:attr:`~q3dfit.q3din.q3din.perror_errspecwin`.
        fitstatus
            Fit status, passed from the fitting routine. If method is set
            to 'least_squares' by :py:attr:`~q3dfit.q3din.q3din.argslinefit`, then
            this is described by the status attribute of 
            :py:meth:`~scipy.optimize.least_squares`. If the method is set
            to `leastsq`, then this is the `ier` integer return code from
            :py:meth:`~scipy.optimize.leastsq`.
        dof
            Degrees of freedom. Copy of :py:attr:`~lmfit.ModelResult.nfree`.
        redchisq
            Reduced chi-squared. Copy of :py:attr:`~lmfit.ModelResult.redchi`.
        nfev
            Number of function evaluations. Copy of :py:attr:`~lmfit.ModelResult.nfev`.
        covar
            Covariance matrix. Copy of :py:attr:`~lmfit.ModelResult.covar`.
        '''
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
        self.parinit = parinit
        self.param = param
        self.perror = perror
        self.perror_errspec = perror_errspec
        self.perror_resid = perror_resid
        self.redchisq = redchisq


    def init_contfit(self,
                     ct_method: Literal['subtract','divide']='subtract',
                     cont_dat: Optional[ArrayLike]=None,
                     ct_indx: Optional[ArrayLike]=None,
                     cont_fit: Optional[ArrayLike]=None,
                     ct_coeff: Optional[dict]=None,
                     ct_rchisq: Optional[float]=None,
                     zstar: Optional[float]=None,
                     zstar_err:  Optional[float]=None):
                     # cont_fit_pretweak=None
        '''
        Initialize continuum fit parameters.

        Parameters
        ----------
        ct_method
            Method for continuum fit. Default is 'subtract'. Set on 
            initialization of :py:class:`~q3dfit.q3din.q3din` by
            :py:attr:`~q3dfit.q3din.q3din.dividecont`.
        cont_dat
            Fluxes minus any best-fit emission-line model. Includes data
            within the fit range. Set to None until assigned by
            :py:func:`~q3dfit.fitspec.fitspec`.
        ct_indx
            Indices to cont_dat that are used for the continuum fit; i.e.,
            emission lines are masked out. Set to None until assigned by
            :py:func:`~q3dfit.fitspec.fitspec`.
        cont_fit
            Best-fit continuum model, over the fit range. Set to None until 
            assigned by :py:func:`~q3dfit.fitspec.fitspec`.
        ct_coeff
            Parameters of the continuum fit. Set to None until
            assigned by :py:func:`~q3dfit.fitspec.fitspec`.  Exact form depends
            on the continuum fit function used. See module
            :py:mod:`~q3dfit.contfit` for details.
        ct_rchisq : float
            Reduced chi-squared from continuum fit. Set to None until
            assigned by :py:func:`~q3dfit.fitspec.fitspec`.
        zstar : float
            Best-fit zstar. Set to None until assigned by 
            :py:func:`~q3dfit.fitspec.fitspec`.
        zstar_err : float
            Error in best-fit zstar. Set to None until assigned by
            :py:func:`~q3dfit.fitspec.fitspec`.
        '''

        self.docontfit = True

        # Continuum fit parameters
        self.ct_method = ct_method
        self.cont_dat = cont_dat
        self.ct_indx = ct_indx # gd_indx is applied, and then ct_indx
        self.cont_fit = cont_fit
        self.ct_coeff = ct_coeff
        self.zstar = zstar
        self.zstar_err = zstar_err
        self.ct_rchisq = ct_rchisq


    def cmplin(self,
               line: str,
               comp: int) -> np.ndarray:
        '''
        Return the model flux for all fitted wavelengths in a given line and component, 
        from the best-fit parameters in :py:attr:`~q3dfit.q3dout.q3dout.param`.

        Parameters
        ----------
        line
            Emission line for which to compute flux.
        comp
            Component (0-indexed) for which to compute flux.

        Returns
        -------
        np.ndarray
            Model flux evaluated at the wavelengths in :py:attr:`~q3dfit.q3dout.q3dout.wave`.
        '''

        lmline = q3dutil.lmlabel(line)
        mName = '{0}_{1}_'.format(lmline.lmlabel, comp)
        gausspar = np.zeros(3)
        gausspar[0] = self.param[mName+'flx']
        gausspar[1] = self.param[mName+'cwv']
        gausspar[2] = self.param[mName+'sig']
        # convert sigma to wavelength space from km/s, since the model is in wavelength space
        gausspar[2] = gausspar[2] * gausspar[1]/c.to('km/s').value

        def gaussian(xi: np.ndarray,
                     parms: np.ndarray) -> np.ndarray:
            a = parms[0]  # amp
            b = parms[1]  # mean
            c = parms[2]  # standard dev

            # Anything higher-precision than this (e.g., float64) slows things down
            # a bunch. longdouble completely chokes on lack of memory.
            arg = np.array(-0.5 * ((xi - b)/c)**2, dtype=np.float32)
            g = a * np.exp(arg)
            return g

        flux = gaussian(self.wave, gausspar)

        return flux


    def sepfitpars(self,
                   waveran=None,
                   doublets=None,
                   ignoreres=False):
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

        def gaussflux(norm: float,
                      sigma: float,
                      normerr: float,
                      sigerr: float) -> dict[Literal['flux', 'flux_err'], float]:
            '''
            Compute total Gaussian flux and error from normalization
            and sigma.
            '''
            flux = norm * sigma * np.sqrt(2. * np.pi)
            fluxerr = 0.
            if normerr is not None and sigerr is not None:
                fluxerr = flux*np.sqrt((normerr/norm)**2. + (sigerr/sigma)**2.)
            outstr = {'flux': flux, 'flux_err': fluxerr}
            return outstr

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
                    lmline = q3dutil.lmlabel(line)
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
                                          fluxpkerr[line][i],
                                          sigmaerr[line][i] /
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


    def sepcontpars(self,
                    q3di,
                    decompose_qso_fit: Optional[bool]=None,
                    decompose_ppxf_fit: Optional[bool]=None):
        '''
        Separate continuum fit parameters into individual components.

        Parameters
        ----------
        q3di
            :py:class:`~q3dfit.q3din.q3din` object with input parameters.
        decompose_qso_fit
            Optional. Decompose QSO fit if continuum fit method is
            :py:func:`~q3dfit.contfit.fitqsohost`. Default is None,
            which means True if the continuum fit method is
            :py:func:`~q3dfit.contfit.fitqsohost`, and False otherwise.
        decompose_ppxf_fit
            Optional. Decompose pPXF fit if continuum fit method
            is :py:mod:`~ppxf.ppxf`. Default is None,
            which means True if the continuum fit method is
            :py:mod:`~ppxf.ppxf`, and False otherwise.
        '''
        q3dii: q3din.q3din = q3dutil.get_q3dio(q3di)

        # If not set, determine whether to decompose QSO and pPXF fits
        # based on the continuum fit method.
        # If set, use the values passed in.
        if decompose_qso_fit is None:
            if q3dii.fcncontfit == 'fitqsohost':
                decompose_qso_fit = True
            else:
                decompose_qso_fit = False
        if decompose_ppxf_fit is None:
            if q3dii.fcncontfit == 'ppxf':
                decompose_ppxf_fit = True
            else:
                decompose_ppxf_fit = False
        # record these for later use by other methods
        self.decompose_qso_fit = decompose_qso_fit
        self.decompose_ppxf_fit = decompose_ppxf_fit

        self.add_poly_degree = -1  # default, no additive polynomial

        # Compute PPXF components: additive polynomial and stellar fit
        if decompose_ppxf_fit:
            self.add_poly_degree = 4  # should match fitspec
            if q3dii.argscontfit is not None:
                if 'add_poly_degree' in q3dii.argscontfit:
                    self.add_poly_degree = \
                        q3dii.argscontfit['add_poly_degree']
            self.polymod = self.ct_coeff['polymod']
            self.stelmod = self.ct_coeff['stelmod']

        # Compute FITQSOHOST components
        elif decompose_qso_fit:

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
                    blrpar = q3dii.argscontfit['blrpar']
                else:
                    blrpar = None
                # default here must be same as in IFSF_FITQSOHOST
                if 'add_poly_degree' in q3dii.argscontfit:
                    self.add_poly_degree = \
                        q3dii.argscontfit['add_poly_degree']
                else:
                    self.add_poly_degree = 30 # needs to match default in fitqsohost

                # Get QSO template
                qsotemplate = \
                    np.load(q3dii.argscontfit['qsoxdr'],
                            allow_pickle='TRUE').item()
                qsowave = qsotemplate['wave']
                qsoflux_full = qsotemplate['flux']
                qsoflux = qsoflux_full[self.fitran_indx]

                # If polynomial residual is re-fit with PPXF,
                # get the polynomial and stellar model
                # and the QSO host parameters
                par_qsohost = self.ct_coeff['qso_host']
                if 'refit' in q3dii.argscontfit:
                    if q3dii.argscontfit['refit'] == 'ppxf':
                        self.polymod_refit = self.ct_coeff['ppxf']['polymod']
                        self.stelmod = self.ct_coeff['ppxf']['stelmod']
                else:
                    self.polymod_refit = np.zeros(len(self.wave), dtype='float64')
                    self.stelmod = np.zeros(len(self.wave), dtype='float64')

                # QSO-only model
                self.qsomod, _, _ = \
                    qsohostfcn(self.wave, params_fit=par_qsohost,
                               qsoflux=qsoflux, qsoonly=True,
                               blrpar=blrpar, qsoord=qsoord,
                               hostord=hostord)
                # host-only model
                self.hostmod = self.cont_fit - self.qsomod
                # template for QSO model, before normalization
                self.qsomod_normonly = qsoflux
                # QSO model with BLR contribution only
                if blrpar is not None:
                    self.qsomod_blronly, _, _ = \
                        qsohostfcn(self.wave,
                                   params_fit=par_qsohost,
                                   qsoflux=qsoflux, blronly=True,
                                   blrpar=blrpar, qsoord=qsoord,
                                   hostord=hostord)
                else:
                    self.qsomod_blronly = 0.

            # CB: adding option to plot decomposed QSO fit if questfit is used
            elif q3dii.fcncontfit == 'questfit':
                self.qsomod, self.hostmod, qsomod_intr, hostmod_intr = \
                    self._quest_extract_QSO_contrib(q3dii)
                # qsomod_polynorm = 1.
                # qsomod_notweak = self.qsomod
                qsoflux = self.qsomod.copy()/np.median(self.qsomod)
                self.qsomod_normonly = qsoflux
                self.qsomod_blronly = 0.


    def plot_line(self,
                  q3di: q3din.q3din,
                  fcn: str='plotline',
                  savefig: bool=False,
                  outfile: Optional[str]=None,
                  argssavefig: dict={'bbox_inches': 'tight',
                                     'dpi': 300},
                  plotargs: dict={}):
        '''
        Line plotting function.

        Parameters
        ----------
        q3di
            :py:class:`~q3dfit.q3din.q3din` object with input parameters.
        fcn
            Name of the plotting function to use. Default is 
            :py:func:`~q3dfit.plot.plotcont`.
        savefig
            If True, save the figure to a file. Default is False.
        outfile
            If savefig is True, the name of the output file to save the figure.
            Default is None, which means the output file will be named
            `<filelab>_cnt` where `<filelab>` is the path+filename set
            by :py:meth:`~q3dfit.q3dout.q3dout.load_q3dout`.
        argssavefig
            Optional. Dictionary of arguments to pass to 
            :py:meth:`~matplotlib.pyplt.savefig()`. Defaults to
            {'bbox_inches': 'tight', 'dpi': 300}.
        plotargs
            Additional keyword arguments to pass to the plotting function.
        '''

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
                specConv = q3dii.get_dispersion()
            else:
                specConv = None

            plotline(self, savefig=savefig, outfile=outfile, specConv=specConv,
                     argssavefig=argssavefig, waveunit_in=self.waveunit, **plotargs)
        else:
            print('plot_line: no lines to plot!')


    def plot_cont(self,
                  q3di: q3din.q3din,
                  fcn: str='plotcont',
                  savefig: bool=False,
                  outfile: Optional[str]=None,
                  argssavefig: dict={'bbox_inches': 'tight',
                                     'dpi': 300},
                  plotargs: dict={}):
        '''
        Continuum plotting function.

        Parameters
        ----------
        q3di
            :py:class:`~q3dfit.q3din.q3din` object with input parameters.
        fcn
            Name of the plotting function to use. Default is 
            :py:func:`~q3dfit.plot.plotcont`.
        savefig
            If True, save the figure to a file. Default is False.
        outfile
            If savefig is True, the name of the output file to save the figure.
            Default is None, which means the output file will be named
            `<filelab>_cnt` where `<filelab>` is the path+filename set
            by :py:meth:`~q3dfit.q3dout.q3dout.load_q3dout`.
        argssavefig
            Optional. Dictionary of arguments to pass to 
            :py:meth:`~matplotlib.pyplt.savefig()`. Defaults to
            {'bbox_inches': 'tight', 'dpi': 300}.
        plotargs
            Additional keyword arguments to pass to the plotting function.
        '''

        if self.docontfit:

            q3dii: q3din.q3din = q3dutil.get_q3dio(q3di)

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
            if self.decompose_qso_fit:

                # Host only plot
                # assume argscontfit exists if we're doing fitqsohost
                if 'refit' in q3dii.argscontfit:
                    compspec = np.array([self.polymod_refit, self.stelmod])
                                            #self.hostmod-self.polymod_refit])
                    complabs = [f'ord. {self.add_poly_degree}' +
                                    ' Leg. poly.', 'stel. temp.']
                else:
                    compspec = [self.hostmod.copy()]
                    complabs = ['exponential terms']

                # Create copies of the current object
                # for input to the plotcont function.
                # Remove QSO model from data, continuum data, and fit
                q3do_host: q3dout = copy.deepcopy(self)
                q3do_host.spec -= self.qsomod
                q3do_host.cont_dat -= self.qsomod
                q3do_host.cont_fit -= self.qsomod
                plotcont(q3do_host, savefig=savefig, outfile=outfilehost,
                            compspec=compspec, complabs=complabs,
                            title='Host', q3di=q3dii, waveunit_in=self.waveunit, 
                            **plotargs)

                # QSO only plot
                if 'blrpar' in q3dii.argscontfit and \
                    max(self.qsomod_blronly) != 0.:
                    qsomod_blrnorm = np.median(self.qsomod) / \
                        max(self.qsomod_blronly)
                    compspec = np.array([self.qsomod_normonly,
                                            self.qsomod_blronly *
                                            qsomod_blrnorm])
                    complabs = ['raw template', 
                                f'scattered x {qsomod_blrnorm:0.2f}']
                else:
                    compspec = [self.qsomod_normonly.copy()]
                    complabs = ['raw template']

                q3do_qso: q3dout = copy.deepcopy(self)
                # Remove host model from data, continuum data, and fit
                q3do_qso.spec -= self.hostmod
                q3do_qso.cont_dat -= self.hostmod
                q3do_qso.cont_fit -= self.hostmod
                if q3dii.fcncontfit != 'questfit':
                    plotcont(q3do_qso, savefig=savefig, outfile=outfileqso,
                                argssavefig=argssavefig,
                                compspec=compspec, complabs=complabs,
                                title='QSO', q3di=q3dii, waveunit_in=self.waveunit, 
                                **plotargs)
                else:
                    plotcont(q3do_qso, savefig=savefig, outfile=outfileqso,
                                argssavefig=argssavefig,
                                compspec=[q3do_qso.cont_fit],
                                title='QSO', complabs=['QSO'], q3di=q3dii,
                                waveunit_in=self.waveunit, **plotargs)

                # Total plot
                plotcont(self, savefig=savefig, outfile=outfilecnt,
                            argssavefig=argssavefig,
                            compspec=np.array([self.qsomod, self.hostmod]),
                            title='Total', complabs=['QSO', 'host'],
                            q3di=q3dii, waveunit_in=self.waveunit, **plotargs)

            # Plot pPXF components
            elif self.decompose_ppxf_fit and self.add_poly_degree > 0:
                plotcont(self, savefig=savefig, outfile=outfilecnt,
                            argssavefig=argssavefig,
                            compspec=np.array([self.stelmod,
                                               self.polymod]),
                            title='Total',
                            complabs=['stel. temp.',
                                        f'ord. {self.add_poly_degree} Leg.poly'],
                            q3di=q3dii, waveunit_in=self.waveunit, **plotargs)
            # Plot total continuum fit in all other cases
            else:
                plotcont(self, savefig=savefig, outfile=outfilecnt,
                            argssavefig=argssavefig,
                            q3di=q3dii, title='Total', waveunit_in=self.waveunit, 
                            **plotargs)


    def _quest_extract_QSO_contrib(self,
                                   q3di: q3din.q3din) \
                                     -> tuple[np.ndarray, np.ndarray, 
                                              np.ndarray, np.ndarray]:
        '''
        Recover the QSO-host decomposition after running questfit.

        Parameters
        ----------
        q3di
            :py:class:`~q3dfit.q3din.q3din` object.
        
        Returns
        -------
        numpy.ndarray
            QSO component of the fit, extincted.
        numpy.ndarray
            Host component of the fit, extincted.
        numpy.ndarray
            QSO component of the fit, intrinsic.
        numpy.ndarray
            Host component of the fit, intrinsic.
        '''
        comp_best_fit = self.ct_coeff['comp_best_fit']
        qso_out_ext = np.array([])
        qso_out_intr = np.array([])

        config_file = readcf(q3di.argscontfit['config_file'])
        if not 'qso' in list(config_file.keys())[1]:
            raise InitializationError(\
                'During QSO-host decomposition, the function assumes that in the '+
                'config file the qso template is the first template, but its name '+
                'does not contain \"qso\".')

        global_extinction = False
        for key in config_file:
            if len(config_file[key]) > 3:
                if 'global' in config_file[key][3]:
                    global_extinction = True

        if global_extinction:
            str_global_ext = list(comp_best_fit.keys())[-2]
            str_global_ice = list(comp_best_fit.keys())[-1]
            # global_ext is a multi-dimensional array
            if len(comp_best_fit[str_global_ext].shape) > 1:
                comp_best_fit[str_global_ext] = comp_best_fit[str_global_ext] [:,0,0]
            if len(comp_best_fit[str_global_ice].shape) > 1:
                comp_best_fit[str_global_ice] = comp_best_fit[str_global_ice] [:,0,0]
            host_out_ext = np.zeros(len(comp_best_fit[str_global_ext]))
            host_out_intr = np.zeros(len(comp_best_fit[str_global_ext]))

            for i, el in enumerate(comp_best_fit):
                if (el != str_global_ext) and (el != str_global_ice):
                    if len(comp_best_fit[el].shape) > 1:
                        comp_best_fit[el] = comp_best_fit[el] [:,0,0]
                    if i==0:
                        qso_out_ext = comp_best_fit[el]*\
                            comp_best_fit[str_global_ext]*\
                                comp_best_fit[str_global_ice]
                        qso_out_intr = comp_best_fit[el]
                    else:
                        host_out_ext += comp_best_fit[el]*\
                            comp_best_fit[str_global_ext]*\
                                comp_best_fit[str_global_ice]
                        host_out_intr += comp_best_fit[el]
        else:
            el1 = list(comp_best_fit.keys())[0]
            host_out_ext = np.zeros(len(comp_best_fit[el1]))
            host_out_intr = np.zeros(len(comp_best_fit[el1]))

            spec_i = np.array([])
            for i, el in enumerate(comp_best_fit):

                if len(comp_best_fit[el].shape) > 1:
                    comp_best_fit[el] = comp_best_fit[el] [:,0,0]

                if not ('_ext' in el or '_abs' in el):
                    spec_i = comp_best_fit[el]
                    intr_spec_i = comp_best_fit[el].copy()
                    if el+'_ext' in comp_best_fit.keys():
                        spec_i = spec_i*comp_best_fit[el+'_ext']
                    if el+'_abs' in comp_best_fit.keys():
                        spec_i = spec_i*comp_best_fit[el+'_abs']
                    if i==0:
                        qso_out_ext = spec_i
                        qso_out_intr = intr_spec_i
                    else:
                        host_out_ext += spec_i
                        host_out_intr += intr_spec_i

        return qso_out_ext, host_out_ext, qso_out_intr, host_out_intr


def load_q3dout(q3di: str | q3din.q3din,
                col: Optional[int]=None,
                row: Optional[int]=None,
                cubedim: Optional[int]=None,
                quiet: bool=False) -> q3dout:
    """
    Load :py:class:`~q3dfit.q3dout.q3dout` after it's been saved to a file. It will
    set the `<filelab>` attribute as follows, where `<outdir>` and `<label>` are the 
    attributes from :py:class:`~q3dfit.q3din.q3din`, `<col>` is the column index, 
    and `<row>`is the row index.\n
    - If the data are 1D, then `<filelab>` = `<outdir>/<label>.npy`.
    - If the data are 2D, then `<filelab>` = `<outdir>/<label>_<col>.npy`.
    - If the data are 3D, then `<filelab>` = `<outdir>/<label>_<col>_<row>.npy`.

    Parameters
    ----------
    q3di
        :py:class:`~q3dfit.q3din.q3din` object or file name.
    col
        Optional. Column index. None assumes 1D spectrum fits input.
        If None and cubedim > 1, will throw ValueError. Default is None.
    row
        Optional. Row index. None assumes 1D or 2D spectrum fits input.
        If None and cubedim > 2, will throw ValueError. Default is None.
    cubedim
        Optional. Dimension of cube (1, 2, or 3). If None, will try to get from q3di or 
        cube itself. Default is None.
    quiet
        Optional. Suppress messages. Default is False.

    Returns
    -------
    q3dout

    """

    # convert from string to object if necessary.
    q3dii = q3dutil.get_q3dio(q3di)

    filelab = '{0.outdir}{0.label}'.format(q3dii)
    if cubedim is None:
        if hasattr(q3dii, 'cubedim'):
            cubedim = q3dii.cubedim
        else:
            q3dutil.write_msg('load_q3dout: q3di has no attribute cubedim, loading cube',
                              q3dii.logfile, quiet)
            cube = q3dii.load_cube() #, vormap
            cubedim = cube.dat.ndim
    if cubedim > 1:
        if col is None:
            raise ValueError('load_q3dout: col must be set for 2D or 3D cube')
        filelab += '_{:04d}'.format(col)
        if cubedim > 2:
            if row is None:
                raise ValueError('load_q3dout: row must be set for 3D cube')
            filelab += '_{:04d}'.format(row)
    infile = filelab + '.npy'

    q3dout = q3dutil.get_q3dio(infile)
    # add file label to object
    q3dout.filelab = filelab
    return q3dout
