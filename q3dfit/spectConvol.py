#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: Yuzo Ishikawa
"""

from typing import Literal, Optional, Union

#import copy
import numpy as np
import os
import re
from astropy.constants import c
from astropy.io import fits
from astropy.stats import gaussian_sigma_to_fwhm
from ppxf.ppxf_util import varsmooth
#from scipy.ndimage import gaussian_filter1d
from scipy.interpolate import CubicSpline

import q3dfit.data.dispersion_files


class spectConvol:
    '''
    Convolve an input spectrum by grating resolution.
    
    JWST provides NIRSpec dispersion files. However, MIRI is not provided. 
    Instead, we take a Cubic Spline interpolation based on the curves in 
    the Jdocs website.
    
    For the constructor, the requested dispersion curves
    are either loaded from a file, created from a flat dispersion curve,
    or computed from a formula in the case of MRS.

    Parameters
    ----------
    spect_convol : {'ws_instrum': dict[str, str], 'dispdir': Union[str, None]}
        `ws_instrum` contains instrument/grating combinations.
        `dispdir` is the directory of dispersion files. If set to None (default), 
        the q3dfit dispersion_file directory is used.
    waveunit
        Optional. Wavelength unit, either 'micron' or 'Angstrom'. Default is 'micron'.

    Raises
    ------
    SystemExit
        If no `ws_instrum` dictionary is found in the input dictionary.
        If an error is found in the input dictionary.

    Attributes
    ----------
    dispdir 
        Full directory of dispersion files in the repository.
    InstGratObjs
        Nested dictionary whose keys are instrument/grating combinations.
        The values are dispersion objects.
    '''
    def __init__(self,
                 spect_convol: dict[str, Union[dict[str, str], str, None]],
                 waveunit: Literal['micron', 'Angstrom'] = 'micron'):
        # full directory of dispersion files in repo
        #self.dispdir = os.path.join(os.path.abspath(q3dfit.data.__file__)[:-11],
        #    'dispersion_files')

        # Check if dispersion directory is specified in the input dictionary.
        # If not, use the default q3dfit directory.
        if 'dispdir' not in spect_convol:
            self.dispdir = \
                os.path.join(os.path.abspath(q3dfit.data.__file__)[:-11],
                'dispersion_files')
        else:
            self.dispdir = spect_convol['dispdir']

        # create the nested dictionaries to hold the dispersion objects
        self.InstGratObjs = {}
        # inst is a telescope+instrument string
        # gratlist is dictionary of grating strings
        if 'ws_instrum' not in spect_convol:
            raise SystemExit('No ws_instrum dictionary found in ' +
                'q3dinit.spect_convol.')
        try:
            for inst, gratlist in spect_convol['ws_instrum'].items():
                self.InstGratObjs[inst.upper()] = {}
                for grat in gratlist:
                    self.InstGratObjs[inst.upper()][grat.upper()] = None
        except:
            raise SystemExit('Error in loading spect_convol dictionary ' +
                'in q3dinit.')
    
        # set the wavelength unit
        self.waveunit = waveunit

        # Create and populate the dispersion objects
        for inst, gratlist in self.InstGratObjs.items():
            for grat in gratlist:
                if inst != 'JWST_MIRI':
                    if inst.upper() == 'FLAT':
                        # look for convolution method and dispersion
                        # in grating name
                        flattype = ''.join(re.findall("[a-zA-Z]", grat))
                        flatdisp = '.'.join(re.findall(r"\d+", grat))
                        self.InstGratObjs[inst][grat] = \
                            FlatDispersion(float(flatdisp), flattype,
                                           waveunit=self.waveunit)
                    else:
                        self.InstGratObjs[inst][grat] = \
                            InstGratDispersion(inst, grat, dispdir=self.dispdir,
                                               waveunit=self.waveunit)
                        self.InstGratObjs[inst][grat].readInstGrat()
                else:
                    self.InstGratObjs[inst][grat] = \
                        InstGratDispersion(inst, grat, dispdir=self.dispdir,
                                           waveunit=self.waveunit)  
                    self.InstGratObjs[inst][grat].get_MRS_Rdlam()

    def spect_convolver(self,
                        wave: np.ndarray,
                        flux: np.ndarray,
                        wavecen: Optional[float]=None,
                        oversample: int=1) -> np.ndarray:
        '''
        Convolves an input spectrum by the spectral resolution.

        Parameters
        ----------
        wave
            Wavelength array of input spectrum
        flux
            Flux array of input spectrum
        wavecen
            Optional. Central wavelength of a spectral feature, to check if it falls 
            in the range of a grating. Default is None, which means no check is performed.
        oversample
            Optional. Oversample the spectrum by this factor before convolution. Default
            is 1, which means no oversampling is performed. Passed to 
            :py:func:`~ppxf.ppxf_util.varsmooth`.

        Returns
        -------
        np.ndarray
            Convolved spectrum.

        '''

        # select the instrument/grating combinations that cover the 
        # input wavelength of interest; if no wavelength is specified,
        # all instrument/grating combinations are used
        InstGratSelect = self.selectInstGrat(wavecen=wavecen)
        #
        gwave, gdlam = self.get_InstGrat_waveDlam(InstGratSelect)

        # If MRS ever has grating dispersion files, we just have to comment out
        # this block and remove the if/else statement.
        if gwave is None or gdlam is None:

            iwave = wave
            iflux = flux
            _, idlam = MRS_dispersion(wave)

        else:

            # find the indices of the input wavelengths that are within the
            # range of the grating
            w1 = np.where(wave >= min(gwave))[0]
            w2 = np.where(wave <= max(gwave))[0]
            ww = np.intersect1d(w1, w2)
            # indices of the input wavelengths that are outside the range of 
            # the grating
            w1x = np.where(wave < min(gwave))[0]
            w2x = np.where(wave > max(gwave))[0]
            # wavelengths and fluxes that are within the range of the grating
            iwave= wave[ww]
            iflux = flux[ww]

            # interpolate the dispersion curve to the input wavelengths
            dlam_interp_fcn = CubicSpline(gwave, gdlam)
            idlam = dlam_interp_fcn(iwave)

        idsig = idlam/gaussian_sigma_to_fwhm
        fluxconv = varsmooth(iwave, iflux, idsig, oversample=oversample)

        if self.is_dispobj_mrs(InstGratSelect[0]):
            return fluxconv
        else:
            return np.concatenate((flux[w1x], fluxconv, flux[w2x]))


    def selectInstGrat(self,
                       wavecen: Optional[float]=None) -> list:
        '''
        Make a list of instrument/grating combinations to use for convolution.
        If wavecen is not specified, all instrument/grating combinations are
        used. The length of the output list must be 1 or 2.
        
        If it's 2, the gratings may need to have some overlap in wavelength. 
        (See get_InstGrat_waveDlam for more details.) Should probably do a check
        for this ...
        
        Parameters
        ----------
        wavecen
            Optional. Wavelength to check. Default is None, which means all
            instrument/grating combinations are used.
        
        Returns
        -------
        list
            Inst/grat objects to use for convolution. Must have a length of 1 or 2.

        Raises
        ------
        SystemExit
        '''
        InstGratSelect = list()
        for inst, gratlist in self.InstGratObjs.items():
            for grat in gratlist:
                gwave = self.InstGratObjs[inst][grat].wave
                if wavecen is None:
                    InstGratSelect.append(self.InstGratObjs[inst][grat])
                else:
                    if (wavecen > gwave[0]) and (wavecen < gwave[-1]):
                        InstGratSelect.append(self.InstGratObjs[inst][grat])
                

        # Not sure if this deserves a SystemExit, but it's a way to
        # catch the length errors.
        if InstGratSelect.__len__() == 0:
            raise SystemExit(f'spectConvol.selectInstGrat: ' +
                f'No specified inst/grat combination covers the line ' +
                f'at {wavecen.__str__()} {self.waveunit}.')
        elif InstGratSelect.__len__() > 2:
            if wavecen is None:
                raise SystemExit('spectConvol.selectInstGrat: ' +
                    f'More than 2 inst/grat combinations are specified.')
            else:
                raise SystemExit(f'spectConvol.selectInstGrat: ' +
                    f'More than 2 inst/grat combinations cover the line ' +
                    f'at {wavecen.__str__()} {self.waveunit}.')

        return InstGratSelect
    

    def get_InstGrat_waveDlam(self,
                              InstGratSelect: list) -> tuple:
        '''
        Get wavelength and dlambda for selected grating(s). If one grating,
        return the wavelength and dlambda. If two gratings, return the
        combined wavelength and dlambda, with the overlap region split
        between the two gratings. If MRS is used, return None for both, 
        as we calculate the dispersion from a formula.

        Parameters
        ----------
        InstGratSelect
            List of instrument/grating combinations. The length of the list is 
            1 or 2. If 2 are specified, they must have overlapping wavelengths.
        
        Returns
        -------
        Optional[float, np.ndarray]
            Wavelength of the grating(s).
        Optional[float, np.ndarray]
            Delta lambda of the grating(s).
        '''

        # Case of one or two MRS gratings covering the same wavelength.
        # Then no interpolation is necessary, we just reapply the formula at the
        # input wavelength(s).
        # If MRS ever has grating dispersion files, we just have to comment out
        # this block and remove the if/else statement.
        if self.is_dispobj_mrs(InstGratSelect[0]):

            gwave, gdlam = None, None

        else:
            # Case of two gratings; must have overlapping wavelengths
            if len(InstGratSelect) == 2:
                # arrays to hold wavelength ranges, wavelengths, and dlambda for
                # each grating
                gwaveranges, gwaves, gdlams = [], [], []
                # loop through each grating and populate thse arrays
                for overlap in InstGratSelect:
                    #iinst = overlap[0]
                    #igrat = instoverlap[1]
                    gwaveranges.append([overlap.wave[0], overlap.wave[-1]])
                    gwaves.append(overlap.wave)
                    gdlams.append(overlap.dlam)
                    # these are the minima and maxima of the wavelength ranges
                    gwavemins, gwavemaxs = [gwaveranges[0][0], gwaveranges[1][0]], \
                        [gwaveranges[0][1], gwaveranges[1][1]]
                    # the max of the range mins is the overlap minimum, the min of the 
                    # range maxes is the overlap maximum. The indices of these will
                    # pick out the correct grating to use for the lower and
                    # upper halves of the overlap region
                    overlapmin, igratingslo = np.max(gwavemins), np.amax(gwavemins)
                    overlapmax, igratingshi = np.min(gwavemaxs), np.amin(gwavemaxs)
                    # the median of these is the overlap midpoint
                    overlapmid = np.median([overlapmin, overlapmax])
                    # indices of the wavelengths that are less than the overlap
                    igwaveslo = np.where(gwaves[igratingslo] < overlapmid)[0]
                    # indices of the wavelengths that are greater than the overlap
                    igwaveshi = np.where(gwaves[igratingshi] >= overlapmid)[0]
                    # combined, single array of grating wavelengths and dlambda
                    gwave = np.concatenate((gwaves[igratingslo][igwaveslo], 
                        gwaves[igratingshi][igwaveshi]))
                    gdlam = np.concatenate((gdlams[igratingslo][igwaveslo], 
                        gdlams[igratingshi][igwaveshi]))
            # Case of one grating
            else:

                gwave = InstGratSelect[0].wave
                gdlam = InstGratSelect[0].dlam

        return gwave, gdlam


    def get_R(self,
              wave: Union[float, np.ndarray]) -> Union[float, np.ndarray]:
        '''
        Interpolate a dispersion curve to a new wavelength(s). This is
        used to get the resolving power at a new wavelength.

        Parameters
        ----------
        wave
            Wavelength(s) to which to interpolate.

        Returns
        -------
        Union[float, np.ndarray]
            Resolving power at the input wavelength(s).
        '''

        instgrat = self.selectInstGrat(wavecen=wave)
        if self.is_dispobj_mrs(instgrat[0]):
            R, _ = MRS_dispersion(wave)
        else:
            gwave, gdlam = self.get_InstGrat_waveDlam(instgrat)
            dlamint = CubicSpline(gwave, gdlam)
            R = wave/dlamint(wave)
        return R


    def is_dispobj_mrs(self,
                       InstGratObj: list) -> bool:
        '''
        Convenience method to check if the dispersion object is for MRS.

        Parameters
        ----------
        InstGratObj
            Instrument/grating to check.
        Returns
        -------
        bool
            True if the instrument/grating is MRS, False otherwise.
        '''
        if isinstance(InstGratObj, InstGratDispersion):
            if InstGratObj.inst == 'JWST_MIRI':
                return True
            else:
                return False
        else:
            return False


class dispersion(object):
    '''
    Class to carry dispersion information.

    Attributes
    ----------
    dispdir : str
        Directory that holds the file of the dispersion data, if any. Added by
        constructor and populated by :py:meth:`~q3dfit.spectConvol.dispersion.setdir`.
    dlam : np.ndarray
        Delta lambda. Added by constructor and populated by
        :py:meth:`~q3dfit.spectConvol.dispersion.compute_Rdlam`.
    filename : str
        Filename of the dispersion data file. Added by constructor and populated by
        subclasses as needed.
    R : np.ndarray
        Resolving power. Added by constructor and populated by
        :py:meth:`~q3dfit.spectConvol.dispersion.compute_Rdlam`.
    wave : np.ndarray
        Wavelength. Added by constructor and populated by 
        :py:meth:`~q3dfit.spectConvol.dispersion.read` or
        :py:meth:`~q3dfit.spectConvol.dispersion.compute_Rdlam`.
    waveunit : str
        Wavelength unit. Default is 'micron'. Can be set to 'Angstrom' if needed.           
    '''

    def __init__(self):
        '''
        Instantiate the dispersion class.
        '''
        self.wave = None
        self.waveunit = 'micron'  # default wavelength unit
        self.R = None
        self.dlam = None

        self.dispdir = None
        self.filename = None

    def setdir(self,
               dispdir: Optional[str]=None):
        '''
        Set the attribute carrying the directory of the associated dispersion file.
        '''
        if dispdir is not None:
            self.dispdir = dispdir
        # Don't hardwire in the q3dfit directory.
        #else:
        #    self.dispdir = \
        #        os.path.join(os.path.abspath(q3dfit.data.__file__)[:-11],
        #        'dispersion_files')

    def read(self,
             filename: str):
        '''
        Get dispersion data from a dispersion file(s). The file must be in
        FITS format. The dispersion file must have a wavelength array and
        a dispersion array. The dispersion array can be either R, dlambda,
        or dvel (delta velocity). The method chooses from these in order
        of preference dvel, R, dlambda, if more than one is present.
        It then computes R and dlambda and adds these to the object.

        Parameters
        ----------
        filename
            Filename and path of the dispersion file to load. 

        Raises
        ------
        SystemExit
        '''

        with fits.open(filename) as hdul:
            indisp = hdul[1].data

        # Some methods here assume wavelength is in increasing order
        # Should really do a check here for that ...
        try: 
            self.wave = indisp['WAVELENGTH']  # wavelength [μm]
        except:
            raise SystemExit(f"No wavelength array found in " +
                "dispersion file {filename}.")
        
        # types of dispersion, in order of preference
        # if more than one is present, the first one is used
        types = ['DVEL', 'R', 'DLAMBDA']
        type = None # default
         # get the column names to look for the dispersion type
        cols = indisp.columns.names
        for t in types:
            if t in cols:
                disp = indisp[t]
                type = t
                break
        if type is None:
            raise SystemExit(f"No viable type found in " +
                "dispersion file {filename}. Options are 'R', "+
                "'DVEL', or 'DLAMBDA'.")

        # Compute R and dlam from the dispersion array. This will read
        # wavelength from the object.
        self.compute_Rdlam(disp, type=type)


    def write(self,
              filename: str,
              wave: Optional[np.ndarray]=None, 
              disp: Optional[np.ndarray]=None,
              type: Literal['R', 'DVEL', 'LAMBDA']='R', 
              overwrite: bool=False):
        '''
        Write dispersion file from dispersion data. Can specify the
        dispersion quantity (disp) and type (type) to write. If not
        specified, the method uses the dispersion data in the object, 
        starting with R, then dlambda.

        Parameters
        ----------
        filename
            Filename and full path of the output dispersion file.
        wave
            Optional. Wavelength array. Default is None, which 
            means take from the object.
        disp
            Optional. Dispersion array. Default is None, which means
            use the dispersion data in the object.
        type
            Optional. Type of dispersion. Options are 'R', 'DVEL', or 'DLAMBDA'.
            Default is 'R'.
        overwrite
            Optional. Overwrite existing dispersion file. Default is False.

        Raises
        ------
        SystemExit

        '''

        if wave is None:
            if self.wave is None:
                raise SystemExit("No wavelength array provided.")
            wave = self.wave
        if type is None:
            raise SystemExit("No dispersion type provided.")
        c1 = fits.Column(name='WAVELENGTH', format='E', array=wave)
        clist = [c1]
        # If no dispersion array is provided, use the one in the object.
        # Start with R, then dlambda.
        if disp is None:
            if self.R is None:
                if self.dlam is None:
                    raise SystemExit("dispersion.write: no dispersion array " +
                                 "provided.")
                else:
                    disp = self.dlam
                    type = 'DLAMBDA'
            else:
                disp = self.R
                type = 'R'
        clist.append(fits.Column(name=type, format='E', array=disp))
        cols = fits.ColDefs(clist)
        tbhdu = fits.BinTableHDU.from_columns(cols)
        tbhdu.writeto(filename, overwrite=overwrite)


    def compute_Rdlam(self,
                      disp: Literal['R','DVEL','DLAMBDA']='R',
                      type: Optional[str]=None,
                      wave=None):
        '''
        Compute R and dlam for the dispersion curve from dispersion 
        value(s). If wave is not given, and the object does not have
        a wavelength array, the method uses a default wavelength array.

        The attributes R, dlam, and wave are updated.

        Parameters
        ----------
        disp : float or array
            Dispersion values.
        type : str, optional
            Type of dispersion. Options are 'R', 'DVEL', or 'DLAMBDA'.
        wave : array, optional
            Wavelength array.
        
        Returns
        -------
        None

        '''

        if wave is None:
            # Get it from the object if it's already there
            if self.wave is None:
                # Default wavelength array
                if self.waveunit == 'micron':
                    # 0.05 to 100 micron,
                    # spacing of 0.01 micron = 100 Angstrom
                    self.wave = np.linspace(0.05, 100., int((100.-0.05)/0.01))
                elif self.waveunit == 'Angstrom':
                    # 500 to 100000 Angstrom,
                    # spacing of 10 Angstrom
                    self.wave = np.linspace(500., 100000., int((100000.-500.)/10.))
        else:
            self.wave = wave

        # Match dispersion length to wavelength length
        if isinstance(disp, (int, float)):
            disparr = np.full(self.wave.__len__(), disp)
        elif disp.__len__() != self.wave.__len__():
            raise SystemExit("Dispersion and wavelength arrays must "+
                             "have the same length.")
        else:
            disparr = disp

        if type.upper() == 'R':
            self.R = disparr
            self.dlam = self.wave/self.R
        elif type.upper() == 'DVEL':
            self.R = c.to('km/s').value/disparr
            self.dlam = self.wave/self.R
        elif type.upper() == 'DLAMBDA':
            self.dlam = disparr
            self.R = self.wave/self.dlam


class InstGratDispersion(dispersion):

    def __init__(self,
                 inst: str,
                 grat: str,
                 dispdir: Optional[str]=None,
                 waveunit: Literal['micron', 'Angstrom'] = 'micron'):
        '''
        Instantiate the InstGratDispersion class. This is a subclass of
        dispersion, specifically designed for instruments/gratings
        with dispersion files. This sets the inst/grat values and defines
        the filename of the dispersion file. It also sets the directory.

        Parameters
        ----------
        inst
            Instrument name.
        grat
            Grating name.
        dispdir
            Optional. Directory to read/write the dispersion file.
            Default is None.
        waveunit
            Optional. Wavelength unit, either 'micron' or 'Angstrom'.
            Default is 'micron'.

        Attributes
        ----------
        filename : str
            Filename of the dispersion file. This is a combination of
            the instrument and grating names, lowercased, with '_disp.fits'
            appended to it.

        '''

        super().__init__()
        self.inst = inst
        self.grat = grat
        self.waveunit = waveunit
        self.filename = '_'.join([inst, grat]).lower() + '_disp.fits'
        self.setdir(dispdir)
    
    def readInstGrat(self):
        '''
        Read the dispersion file.
        '''
        try:
            self.read(os.path.join(self.dispdir, self.filename))
        except:
            raise SystemExit('The instrument/grating ' +
                'combination ' + self.inst + '/' + self.grat + 
                ' cannot be read.')
        

    def writeInstGrat(self,
                      wave: Optional[np.ndarray]=None, 
                      disp: Optional[np.ndarray]=None,
                      type: Literal['R','DVEL','DLAMBDA']='R',
                      overwrite: bool=False):
        '''
        Write the dispersion file.

        Parameters
        ----------
        wave
            Optional. Wavelength array. Default is None, which means the object's
            wavelength array is used.
        disp
            Optional. Dispersion array. Default is None, which means the object's
            dispersion array is used.
        type
            Optional. Type of dispersion. Default is 'R'. Options are 'R', 'DVEL',
            or 'DLAMBDA'.
        overwrite
            Optional. Overwrite existing dispersion file. Default is False. 

        '''
        self.write(os.path.join(self.dispdir, self.filename), 
                   wave=wave, disp=disp, type=type, overwrite=overwrite)
    

    def get_MRS_Rdlam(self):
        '''
        Populate the dispersion object with the MRS dispersion curve.
        Analogous to readInstGrat, but for the MRS grating. Won't be
        needed if MRS dispersion files are provided.

        This updates attributes R and dlam, and wave if self.wave is None.

        Assumes waveunit is 'micron', but doesn't check for it.
        '''
        if self.wave is None:
            # Default range. This is the range of the MRS.
            # Spacing of 0.01 micron.
            self.wave = np.linspace(5.0, 28.8, int((28.8-5.0)/0.01))
        self.R, self.dlam = MRS_dispersion(self.wave)


class FlatDispersion(dispersion):

    def __init__(self,
                 disp: float,
                 type: Literal['R', 'DLAMBDA', 'DVEL'],
                 wave: Optional[np.ndarray]=None,
                 waveunit: Literal['micron', 'Angstrom'] = 'micron'):
        '''
        Instantiate the FlatDispersion class. This is a subclass of
        dispersion, specifically designed for flat dispersion curves --
        i.e. that have a constant dispersion value R, dlambda, or dvel.
        This sets the flat dispersion value and type, and computes
        R and dlambda.

        Parameters
        ----------
        disp
            Flat dispersion value. This is a float, and it is the
            dispersion value for the entire wavelength range.
        type
            Type of dispersion. Options are 'R', 'DLAMBDA', or 'DVEL'.
        wave
            Optional. Wavelength array. Default is None, which means
            the object's wavelength array is used. If the object does not
            have a wavelength array, a default wavelength array is used.
        waveunit
            Optional. Wavelength unit, either 'micron' or 'Angstrom'.
            Default is 'micron'.

        Attributes
        ----------
        flatdisp : float
            Flat dispersion value.
        flattype : str
            Type of flat dispersion. Options are 'R', 'DLAMBDA', or 'DVEL'.
        '''
        super().__init__()
        self.flatdisp = disp
        self.flattype = type
        self.waveunit = waveunit
        self.compute_Rdlam(disp, type, wave=wave)


    def writeFlat(self,
                  dispdir: Optional[str]=None,
                  wave: Optional[np.ndarray]=None,
                  overwrite: bool=False):
        '''
        Write the flat dispersion file.

        Parameters
        ----------
        dispdir
            Optional. Directory to write the dispersion file. Default is None.
        wave
            Optional. Wavelength array. Default is None, which means the object's
            wavelength array is used.
        overwrite
            Optional. Overwrite existing dispersion file. Default is False.
        '''
    
        self.filename = 'flat_'+self.flattype.lower()+ \
            str(self.flatdisp)+'_disp.fits'
        self.setdir(dispdir)
        self.write(os.path.join(self.dispdir, self.filename), 
                   wave=wave, disp=self.flatdisp, type=self.flattype, 
                   overwrite=overwrite)


def MRS_dispersion(wave: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    '''
    Calculate resolution vs. wavelength from Jones et al. 2023
    https://ui.adsabs.harvard.edu/abs/2023MNRAS.523.2519J/abstract
    https://jwst-docs.stsci.edu/jwst-mid-infrared-instrument/miri-observing-modes/miri-medium-resolution-spectroscopy#MIRIMediumResolutionSpectroscopy-wavelengthMRSwavelengthcoverageandspectralresolution

    Assumes waveunit is 'micron', but doesn't check for it.

    Parameters
    ----------
    wave
        Wavelength array.
    
    Returns
    -------
    np.ndarray
        Resolving power
    np.ndarray
        Delta lambda
    '''

    R = 4603. - 128. * wave + 10.**(-7.4 * wave)
    dlam = wave / R

    return R, dlam
