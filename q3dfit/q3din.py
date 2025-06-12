#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Oct 27 11:52:46 2022

@author: drupke
"""
from __future__ import annotations

from typing import Any, Iterable, Literal, Optional

from astropy.table import Table
from numpy.typing import ArrayLike
import numpy as np

from . import linelist, readcube, q3dutil, spectConvol
from q3dfit.exceptions import InitializationError


class q3din:
    '''
    Initialize fit.

    Parameters
    -----------
    infile
        Name of input FITS file. 
        Also sets the attribute :py:attr:`~q3dfit.q3din.q3din.infile`.
    label
        Shorthand label for filenames. 
        Also sets the attribute :py:attr:`~q3dfit.q3din.q3din.label`.
    argsreadcube
        Optional. Dict of parameter name/value pairs to pass to 
        :py:class:`~q3dfit.readcube.Cube`. Default is an empty dictionary.
        Also sets the attribute :py:attr:`~q3dfit.q3din.q3din.argsreadcube`.
    cutrange
        Optional. Set of wavelength limits to cut out of fit. Default is None.
        Also sets the attribute :py:attr:`~q3dfit.q3din.q3din.cutrange`.
    fitrange
        Optional. Set of wavelength limits to fit. Default is None, which
        means fit the entire range. Also sets the attribute
        :py:attr:`~q3dfit.q3din.q3din.fitrange`.
    logfile
        Optional. Filename to which to print progess messages. Default is None.
        Also sets the attribute :py:attr:`~q3dfit.q3din.q3din.logfile`.
    name
        Optional. Full name of source for plot labels, etc. Default is None.
        Also sets the attribute :py:attr:`~q3dfit.q3din.q3din.name`.
    outdir
        Optional. Output directory for files. Default is None, which means
        the current directory. Also sets the attribute
        :py:attr:`~q3dfit.q3din.q3din.outdir`.
    spect_convol
        Optional. Input dictionary to pass to :py:class:`~q3dfit.spectConvol.spectConvol`.
        Default is an empty dictionary, which means no convolution. Also sets the
        attribute :py:attr:`~q3dfit.q3din.q3din.spect_convol`.
    zsys_gas
        Optional. Systemic redshift of galaxy. Default is None. Also sets the attribute
        :py:attr:`~q3dfit.q3din.q3din.zsys_gas`. Sets the initial guess for redshift
        in the line fit and/or continuum fit if not None and not overridden by the user 
        with the zinit parameter. Also serves as the systemic redshift in
        :py:class:`~q3dfit.q3dpro.q3dpro`, if not None and not overridden by the user
        with the zsys parameter in :py:class:`~q3dfit.q3dpro.q3dpro`.
    vacuum
        Optional. Are wavelengths in vacuum? Default is True. Also sets the attribute
        :py:attr:`~q3dfit.q3din.q3din.vacuum`.
    datext
        Optional. Extension number for data array in FITS. Default is 1. Also sets the
        attribute :py:attr:`~q3dfit.q3din.q3din.datext`.
    varext
        Optional. Extension number for variance array in FITS. Default is 2. Also sets the
        attribute :py:attr:`~q3dfit.q3din.q3din.varext`.
    dqext
        Optional. Extension number for data quality array in FITS. Default is 3. Also
        sets the attribute :py:attr:`~q3dfit.q3din.q3din.dqext`.


    Attributes
    ----------
    docontfit : bool
        Is the continuum fit? Set to False by default. Added by 
        constructor and updated to True by 
        :py:meth:`~q3dfit.q3din.q3din.init_contfit`.
    dolinefit : bool
        Are emission lines fit? Set to False by default. Added by 
        constructor and updated to True by 
        :py:meth:`~q3dfit.q3din.q3din.init_linefit`.
    lines : list
        List of lines to fit. Added by :py:meth:`~q3dfit.q3din.q3din.init_linefit`.
    ncomp : dict[str, numpy.ndarray]
        # of components for each fitted line and spaxel. Added by 
        :py:meth:`~q3dfit.q3din.q3din.init_linefit`.
    siginit_gas : dict[str, numpy.ndarray]
        Initial guess for sigma for each line, spaxel, and component. Added by
        :py:meth:`~q3dfit.q3din.q3din.init_linefit`.
    siglim_gas : dict[str, numpy.ndarray]
        Lower and upper limits for sigma for each line, spaxel, and component. Added by
        :py:meth:`~q3dfit.q3din.q3din.init_linefit`.
    zinit_gas : dict[str, numpy.ndarray]
        Initial guess for redshift for each line, spaxel, and component. Added by
        :py:meth:`~q3dfit.q3din.q3din.init_linefit`.
    linetie : dict[str, str]
        Anchor line for a group of one or more lines. Each line in the group
        is tied to the others in velocity. Each group is independent. Each key corresponds
        to a line being fit, and the value is the anchor line.
        Added by :py:meth:`~q3dfit.q3din.q3din.init_linefit`.
    linevary : dict[str, dict[Literal['flx','cwv','sig'], numpy.ndarray]]
        Flags to vary flux, central wavelength, and sigma for each
        line, spaxel, and component. Added by :py:meth:`~q3dfit.q3din.q3din.init_linefit`.
    siginit_gas : dict[str, numpy.ndarray]
        Initial guess for sigma for each line, spaxel, and component. Added by
        :py:meth:`~q3dfit.q3din.q3din.init_linefit`.
    siginit_stars : numpy.ndarray
        Initial guess for stellar velocity dispersion for each spaxel, if fitting a 
        stellar template. Added by :py:meth:`~q3dfit.q3din.q3din.init_contfit`.
    zinit_stars : numpy.ndarray
        Initial guess for stellar redshift for spaxel, if fitting a stellar template. \
        Added by :py:meth:`~q3dfit.q3din.q3din.init_contfit`.
    ncols : int
        Number of columns in data cube. Added by :py:meth:`~q3dfit.q3din.q3din.load_cube`.
    nrows : int
        Number of rows in data cube. Added by :py:meth:`~q3dfit.q3din.q3din.load_cube`.
    cubedim : int
        Dimension of data cube. Added by :py:meth:`~q3dfit.q3din.q3din.load_cube`.
    '''

    def __init__(self, 
                 infile: str, 
                 label: str, 
                 argsreadcube: dict[str, Any]={}, 
                 cutrange: Optional[ArrayLike] = None,
                 fitrange: Optional[Iterable[float]]=None,
                 logfile: Optional[str]=None, 
                 name: Optional[str]=None,
                 outdir: Optional[str]=None,
                 spect_convol: dict[str, Optional[str | dict]]={},
                 zsys_gas: Optional[float]=None,
                 vacuum: bool=True,
                 #vormap = None,
                 datext: int=1, varext: int=2, dqext: int=3
                ):

        self.infile = infile
        self.label = label
        self.argsreadcube = argsreadcube
        if cutrange is not None:
            self.cutrange = np.ndarray(cutrange) # convert to numpy array for downstream use
        else:
            self.cutrange = None
        self.fitrange = fitrange
        self.logfile = logfile
        self.name = name
        self.outdir = outdir
        self.spect_convol = spect_convol
        self.vacuum = vacuum
 #       self.vormap = vormap
        self.zsys_gas = zsys_gas

        self.datext = datext
        self.varext = varext
        self.dqext = dqext

        # set up switches for continuum and line fitting
        self.docontfit: bool = False
        self.dolinefit: bool = False


    def init_linefit(self, 
                     lines: list[str], 
                     linetie: Optional[str | list[str] | dict[str,str]]=None, 
                     maxncomp: int=1, 
                     siginit: float=50.,
                     siglim_gas: ArrayLike=[5., 2000.],
                     zinit: Optional[float]=None,
                     #peakinit=None, 
                     fcnlineinit: str='lineinit',
                     argslineinit: dict={},
                     argslinelist: dict={}, 
                     argslinefit: dict={},
                     checkcomp: bool=True,
                     fcncheckcomp: str='checkcomp',
                     argscheckcomp: dict={}, 
                     perror_useresid: bool=False,
                     perror_errspecwin: int=20,
                     perror_residwin: int=200,
                     quiet: bool=False
                     ):
        '''
        Initialize line fit.

        Parameters
        ----------
        lines
            Lines to fit.
        linetie
            Optional on input. As input:
            1. Set to None to fit all lines independently.
            2. Set to a single line to tie all lines together.
            3. Set to a list of strings to tie lines in groups.
            Default is None.
        maxncomp
            Optional. Maximum possible number of velocity components in any line.
            Default is 1. Also sets the attribute :py:attr:`~q3dfit.q3din.q3din.maxncomp`.
        siginit
            Optional. Uniform initial guess for emission-line velocity dispersion in km/s.
            Default is 50. This can be overridden by setting :py:attr:`~q3dfit.q3din.q3din.siginit_gas`
            for individual lines, spaxels, and components after initialization.
        siglim_gas
            Optional. Lower and upper limits for sigma in km/s for each line, spaxel,
            and component. Default is [5., 2000.]. This can be overridden by setting
            :py:attr:`~q3dfit.q3din.q3din.siglim_gas` for individual lines, spaxels,
            and components after initialization.
        zinit
            Optional. Initial redshift in km/s to apply to each line. If zinit
            is None, try to use :py:attr:`~q3dfit.q3din.q3din.zsys_gas`. Default is None.
            This can be overridden by setting :py:attr:`~q3dfit.q3din.q3din.zinit_gas` for
            individual lines, spaxels, and components after initialization.
        fcnlineinit
            Optional. Name of routine to initialize line fit. Default is 
            :py:func:`~q3dfit.lineinit.lineinit`. Also sets the attribute 
            :py:attr:`~q3dfit.q3din.q3din.fcnlineinit`.
        argslineinit
            Optional. Arguments for the line initialization function specified by
            `fcnlineinit`. Default is an empty dict. Also sets the attribute 
            :py:attr:`~q3dfit.q3din.q3din.argslineinit`.
        argslinelist
            Optional. Arguments for :py:func:`~q3dfit.linelist.linelist`. 
            Default is an empty dict. Also sets the attribute 
            :py:attr:`~q3dfit.q3din.q3din.argslinelist`.
        argslinefit 
            Optional. Arguments for :py:func:`~lmfit.Model.fit` or
            to the fitting method selected. Default is an empty dict. 
            Also sets the attribute :py:attr:`~q3dfit.q3din.q3din.argslinefit`. 
            Available arguments for
            :py:func:`~lmfit.Model.fit` are 'method', 'max_nfev', and 'iter_cb',
            which can be passed as key/value pairs. If 'method' is not set, it
            defaults to 'least_squares'. If the key 'max_nfev' is not set, it 
            defaults to 200*(Nfitpars+1). If the key 'iter_cb' is not set, it
            defaults to None. Any other key/value pairs are passed as 
            arguments to the fitting function through the 'fit_kws' argument
            to :py:func:`~lmfit.Model.fit`.
        checkcomp
            Optional. If True, reject components using 
            :py:func:`~q3dfit.checkcomp.checkcomp`. Default is True.
            Also sets the attribute :py:attr:`~q3dfit.q3din.q3din.checkcomp`.
        fcncheckcomp
            Optional. Name of routine for component rejection. Default is
            'checkcomp'. Also sets the attribute
            :py:attr:`~q3dfit.q3din.q3din.fcncheckcomp`.
        argscheckcomp
            Optional. Arguments for the component rejection function fcncheckcomp.
            Default is an empty dict. Also sets the attribute
            :py:attr:`~q3dfit.q3din.q3din.argscheckcomp`.
        perror_errspecwin
            Optional. Parameter for calculating the error on the flux peak using
            the median of the error spectrum in a window of this width in pixels,
            centered on the best-fit peak. Default is 20 pixels.
            Also sets the attribute :py:attr:`~q3dfit.q3din.q3din.perror_errspecwin`.
        perror_residwin
            Optional. Parameter for calculating the error on the flux peak using
            the standard deviation of the fit residuals in a window of this width
            centered on the best-fit peak. Default is 200 pixels. Also sets the
            attribute :py:attr:`~q3dfit.q3din.q3din.perror_residwin`.
        perror_useresid
            Optional. If True, use the flux peak error calculated from the residuals
            as the error on the flux peak if it is greater than the flux peak 
            error calculated by :py:func:`~lmfit.Model.fit`. Default is False.
            Useful if the error spectrum is not reliable. Also sets the attribute
            :py:attr:`~q3dfit.q3din.q3din.perror_useresid`.
        quiet
            Optional. Suppress messages. Default is False.
         '''
        # check for defined zsys
        if zinit is None:
            if self.zsys_gas is not None:
                zinit = np.float64(self.zsys_gas)
            else:
                raise InitializationError(
                    'q3di.init_linefit: Both zinit and q3di.zsys_gas are None')
        else:
            zinit = np.float64(zinit)

        self.lines = lines
        self.maxncomp = maxncomp
        self.fcnlineinit = fcnlineinit
        self.argslineinit = argslineinit
        self.argslinelist = argslinelist
        self.argslinefit = argslinefit
        self.checkcomp = checkcomp
        self.fcncheckcomp = fcncheckcomp
        self.argscheckcomp = argscheckcomp
        # Presently, we don't initialize peakinit. So it is 
        # automatically set by fitspec.
        # We don't have a way to auto-initialize the peak of the line
        # on a spaxel-by-spaxel basis except in fitspec. But we need
        # to input the peak of the line on a spaxel-by-spaxel basis
        # into fitspec. To specify peakinit for all spaxels here
        # would require moving the line initialization
        # routine to fitloop, which in turn would require other changes.
        #self.peakinit = peakinit
        self.perror_errspecwin = perror_errspecwin
        self.perror_residwin = perror_residwin
        self.perror_useresid = perror_useresid

        # flip this switch
        self.dolinefit = True

        # set up linetie dictionary
        # case of no lines tied
        if linetie is None:
            linetie = lines
        # case of all lines tied -- single string
        elif isinstance(linetie, str):
            linetie = [linetie] * len(lines)
        elif len(linetie) != len(lines):
            raise InitializationError(
                'q3di.init_linefit: If you are tying lines together in different groups' +
                  ', linetie must be the same length as lines')

        # check that load_cube() has been invoked, or ncols/nrows otherwise
        # defined
        if not hasattr(self, 'ncols') or not hasattr(self, 'nrows'):
            _ = self.load_cube(quiet)
            q3dutil.write_msg('q3di.init_linefit: Loading cube to get ncols, nrows',
                              self.logfile, quiet)

        # set up dictionaries to hold initial conditions
        self.linetie = {}
        self.linevary = {}
        self.ncomp = {}
        self.siginit_gas = {}
        self.siglim_gas = {}
        self.zinit_gas = {}
        for i, line in enumerate(self.lines):
            self.linetie[line] = linetie[i]
            self.ncomp[line] = np.full((self.ncols, self.nrows),
                                       self.maxncomp, dtype=np.int8)
            self.zinit_gas[line] = np.full((self.ncols, self.nrows,
                                            self.maxncomp), zinit,
                                           dtype=np.float64)
            self.siginit_gas[line] = np.full((self.ncols, self.nrows,
                                              self.maxncomp), siginit,
                                             dtype=np.float64)
            self.siglim_gas[line] = np.full((self.ncols, self.nrows,
                                            self.maxncomp, 2), siglim_gas,
                                             dtype=np.float64)
            self.linevary[line] = dict()
            # these are the three Gaussian arguments to lineinit.manygauss()
            self.linevary[line]['flx'] = np.full((self.ncols, self.nrows,
                                                  self.maxncomp), True)
            self.linevary[line]['cwv'] = np.full((self.ncols, self.nrows,
                                                  self.maxncomp), True)
            self.linevary[line]['sig'] = np.full((self.ncols, self.nrows,
                                                  self.maxncomp), True)


    def init_contfit(self, 
                     fcncontfit: str, 
                     argscontfit: dict={},
                     siginit: float=50., 
                     zinit: Optional[float]=None,
                     dividecont: bool=False,
                     av_star: Optional[float]=None,
                     keepstarz: bool=False,
                     maskwidths: Optional[Table | dict[str, np.ndarray]]=None,
                     maskwidths_def: float=500.,
                     masksig_secondfit: float=2.,
                     nolinemask: bool=False,
                     nomaskran: ArrayLike=None,
                     startempfile: Optional[str]=None,
                     startempvac: bool=True,
                     quiet: bool=False
                     #tweakcntfit=None,
                     #fcnconvtemp: str=None,
                     #argsconvtemp: dict={},
                     ):
        '''
        Initialize continuum fit.

        Parameters
        ----------
        fcncontfit
            Function to fit continuum. Assumed to be a function in 
            :py:mod:`q3dfit.contfit`. Exception is :py:mod:`ppxf.ppxf`,
            for which one can specify 'ppxf'. Also sets the attribute
            :py:attr:`~q3dfit.q3din.q3din.fcncontfit`.
        argscontfit
            Optional. Arguments for the continuum fitting function fcncontfit.
            Default is an empty dict. Also sets the attribute
            :py:attr:`~q3dfit.q3din.q3din.argscontfit`. 
        siginit
            Optional. Uniform initial guess for stellar velocity dispersion in km/s,
            if fitting a stellar template. Default is 50. This can be overridden by 
            setting :py:attr:`~q3dfit.q3din.q3din.siginit_stars` for individual
            spaxels after initialization.
        zinit
            Optional. Initial redshift for stellar template, if fitting one. If zinit
            is None, try to use :py:attr:`~q3dfit.q3din.q3din.zsys_gas`. Default is None.
            This can be overridden by setting :py:attr:`~q3dfit.q3din.q3din.zinit_stars` for
            individual spaxels after initialization.
        dividecont
            Optional. Divide data by continuum fit. Default is to subtract.
            Also sets the attribute :py:attr:`~q3dfit.q3din.q3din.dividecont`.
        av_star
            Optional. Initial guess for A_V for stellar template fitting in 
            :py:func:`~ppxf.ppxf`. Default is None, which means no extinction
            is applied. Also sets the attribute :py:attr:`~q3dfit.q3din.q3din.av_star`.
        keepstarz
            Optional. If True, don't redshift stellar template before fitting. Default
            is False. Also sets the attribute :py:attr:`~q3dfit.q3din.q3din.keepstarz`.
        maskwidths
            Optional. Prior to the first continuum fit in :py:func:`~q3dfit.fitloop.fitloop`,
            mask these half-widths, in km/s, around the emission lines to be fit.
            To use this option, specify a value for each line and component.
            If set to None and both continuum and emission-line fits are attempted, 
            routine defaults to uniform value set by maskwidths_def. Default is None.
            Also sets the attribute :py:attr:`~q3dfit.q3din.q3din.maskwidths`.
        maskwidths_def
            Optional. Prior to the first continuum fit in :py:func:`~q3dfit.fitloop.fitloop`,
            mask this uniform half-width, in km/s, around each emission line to be fit.
            Default is 500. Also sets the attribute 
            :py:attr:`~q3dfit.q3din.q3din.maskwidths_def`.
        masksig_secondfit
            Optional. When computing half-widths for masking before the second fit
            in :py:func:`~q3dfit.fitloop.fitloop`, best-fit sigmas from the
            first fit are multiplied by this number. Default is 2.
        nolinemask
            Optional. If True, don't mask emission lines before continuum fit. Default
            is False. Also sets the attribute :py:attr:`~q3dfit.q3din.q3din.nolinemask`.
        nomaskran
            Optional. ArrayLike with 2 x nreg dimensions. Set of lower and upper 
            wavelength limits of regions not to mask. Default is None.
            Also sets the attribute :py:attr:`~q3dfit.q3din.q3din.nomaskran`.
        startempfile
            Optional. Name of numpy save file containing stellar templates. Default is None.
            Also sets the attribute :py:attr:`~q3dfit.q3din.q3din.startempfile`. Must
            be set for stellar template fitting to work. Save file must contain a
            dictionary with keys 'wave' and 'flux'. 'wave' is a 1d numpy array. 'flux'
            is a 2d array, with the second dimension being the number of templates.
            If the stellar template is not in vacuum wavelengths, set 
            :py:attr:`~q3dfit.q3din.q3din.startempvac` to False.
        startempvac
            Optional. Is the stellar template in vacuum wavelengths? If True and the
            data are in air wavelengths, the template is converted to air wavelengths
            in :py:func:`~q3dfit.fitspec.fitspec`. Default is True. Also sets the attribute
            :py:attr:`~q3dfit.q3din.q3din.startempvac`.
        quiet
            Optional. Suppress messages. Default is False.
        tweakcntfit
            Optional. Tweak the continuum fit using local polynomial fitting.
            Default is None. (Not yet implemented.)
        fcnconvtemp
            Optional. Function with which to convolve template before fitting.
            Default is None. (Not yet implemented.)
        argsconvtemp
            Optional. Arguments for the template convolution function fcnconvtemp.
            Default is an empty dict. (Not yet implemented.)

        '''

        self.fcncontfit = fcncontfit
        self.argscontfit = argscontfit
        self.dividecont = dividecont
        self.av_star = av_star
        self.keepstarz = keepstarz
        self.maskwidths = maskwidths
        self.maskwidths_def = maskwidths_def
        self.masksig_secondfit = masksig_secondfit
        self.nolinemask = nolinemask
        self.nomaskran = nomaskran
        self.startempfile = startempfile
        self.startempvac = startempvac

        # flip this switch
        self.docontfit = True

        # check for defined zsys
        if zinit is None:
            if self.zsys_gas is not None:
                zinit = np.float64(self.zsys_gas)
            else:
                raise InitializationError(
                    'q3di.init_contfit: Both zinit and q3di.zsys_gas are None')
        else:
            zinit = np.float64(zinit)

        # check that load_cube() has been invoked, or ncols/nrows otherwise
        # defined
        if not hasattr(self, 'ncols') or not hasattr(self, 'nrows'):
            _ = self.load_cube(quiet)
            q3dutil.write_msg('q3di: Loading cube to get ncols, nrows', self.logfile,
                              quiet)

        self.siginit_stars = np.full((self.ncols, self.nrows), siginit,
                                     dtype=np.float64)
        self.zinit_stars = np.full((self.ncols, self.nrows), zinit,
                                   dtype=np.float64)

        # Template convolution not yet implemented
        #self.fcnconvtemp = fcnconvtemp
        #self.argsconvtemp = argsconvtemp
        # Tweaking of continuum fit not yet implemented
        #self.tweakcntfit = tweakcntfit


    def load_cube(self, quiet: bool=False) -> readcube.Cube:
        '''
        Load data cube from file. Instantiates :py:class:`~q3dfit.readcube.Cube` object.

        Parameters
        ----------
        quiet
            Optional. Suppress Cube.about(). Default is False.

        Returns
        -------
        readcube.Cube
            :py:class:`~q3dfit.readcube.Cube` object.

        '''
        try:
            cube = readcube.Cube(self.infile, datext=self.datext, varext=self.varext,
                dqext=self.dqext, logfile=self.logfile, **self.argsreadcube)
        except FileNotFoundError:
            print('Data cube not found.')
        
        self.ncols: int = cube.ncols
        self.nrows: int = cube.nrows
        self.cubedim = cube.cubedim
        if not quiet:
            cube.about()
        return cube


    def get_dispersion(self) -> Optional[spectConvol.spectConvol]:
        '''
        Instantiate :py:class:`~q3dfit.spectConvol.spectConvol` object with dispersion 
        information for selected gratings.

        Returns
        -------
        Optional[spectConvol.spectConvol]
            :py:class:`~q3dfit.spectConvol.spectConvol` object, or None (no convolution) if 
            q3di.spect_convol is an empty dictionary.
        '''
        if not self.spect_convol:
            return None
        else:
            # check for non-default wavelength unit
            argsspecconv = dict()
            if hasattr(self, 'argsreadcube'):
                if 'waveunit_out' in self.argsreadcube:
                    argsspecconv['waveunit'] = self.argsreadcube['waveunit_out']
            return spectConvol.spectConvol(self.spect_convol, **argsspecconv)


    def get_linelist(self) -> Table | list:
        '''
        Get parameters of emission lines to be fit from internal database.

        Returns
        -------
        Table | list
            Return output of py:func:`~q3dfit.linelist.linelist` as a Table if
            q3di has attribute lines (i.e., initializes an emission-line fit), 
            or as an empty list if not.

        '''
        vacuum = self.vacuum
        if hasattr(self, 'lines'):
            if hasattr(self, 'argsreadcube'):
                if 'waveunit_out' in self.argsreadcube:
                    self.argslinelist['waveunit'] = self.argsreadcube['waveunit_out']
            listlines = linelist.linelist(self.lines, vacuum=vacuum,
                                          **self.argslinelist)
        else:
            listlines = []
        return listlines
