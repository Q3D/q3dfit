'''
This module contains the Cube class for reading in and containing a data cube
'''

import copy
import numpy as np
import re
import warnings
from typing import Literal, Optional
from numpy.typing import ArrayLike

from astropy import units as u
from astropy.constants import c
from astropy.io import fits
#from scipy import interpolate

from q3dfit.exceptions import CubeError
from . import q3dutil

__all__ = [
    'Cube'
    ]


class Cube:
    '''
    Read in and contain a data cube and associated information.

    Parameters
    -----------
    infile
        Name of input FITS file.
    datext
        Optional. Extension number of data. Defaults to 1.
    varext
        Optional. Extension number of variance. Defaults to 2.
    dqext
        Optional. Extension number of data quality. Defaults to 3. If DQ
        extension is not present, set to None. This will set all DQ values
        to 0.
    wmapext
        Optional. Extension number  of weight. Defaults to 4. If weight
        extension is not present, set to None.
    wavext
        Optional. Extension number of wavelength data. Defaults to None.
    zerodq
        Optional. Zero out the DQ array. Default is False. Even if dqext
        is present, this will set all DQ values to 0.
    error
        Optional. If the variance extension contains errors rather than
        variance, set to True. Defaults to False.
    invvar
        Optional. If the variance extension contains inverse variance rather
        than variance, set to True. Defaults to False.
    fluxnorm
        Optional. Factor by which to divide the data. Defaults to 1.
    fluxunit_in
        Optional. Flux unit of input data cube. Default is `MJy/sr` (JWST 
        default). Must be parseable by :py:mod:`~astropy.units`.
    fluxunit_out
        Optional. Flux unit carried by object. Default is 
        `erg/s/cm2/micron/sr`. Must be parseable by :py:mod:`~astropy.units`.
    usebunit
        Optional. If BUNIT and fluxunit_in differ, default to BUNIT. 
        Default is False.
    pixarea_sqas
        Optional. Spaxel area in square arcseconds. If present and 
        flux units are per sr or per square arcseconds, will convert surface 
        brightness units to flux per spaxel. Default is None. If set to None and
        header contains PIXAR_A2, routine will use that value.
    waveunit_in 
        Optional. Wavelength unit of input data cube. Default is `micron`.
    waveunit_out
        Optional. Wavelength unit carried by object. Default is `micron`.
    usecunit
        Optional. If CUNIT and waveunit_in differ, default to CUNIT. 
        Default is False.
    logfile
        Optional. Filename for progress messages. Default is None.
    quiet
        Optional. Send progress messages to stdout. Default is False.

    Attributes
    ----------
    dat: numpy.ndarray
        Data array (ncols, nrows, nwave).
    var : numpy.ndarray
        Variance array (ncols, nrows, nwave).
    err : numpy.ndarray
        Error array (ncols, nrows, nwave). Recorded as nan when the variance is negative.
    dq : numpy.ndarray
        Data quality array (ncols, nrows, nwave).
    wmap : numpy.ndarray
        Weight (1/variance) array (ncols, nrows, nwave). If wmap extension is not present, 
        set to None.
    wave : numpy.ndarray
        Wavelength array (nwave).
    cdelt : float
        Constant dispersion. If a wavelength extension is present, this is the difference
        between the first two wavelengths.
    crpix : int
        WCS pixel zero point, in unity-offset coordinates. If a wavelength extension is 
        present, this is the first pixel (i.e., 1).
    crval : float
        WCS wavelength zero point. If a wavelength extension is present, this is the first
        wavelength.
    header_phu : astropy.io.fits.Header
        Primary header data unit.
    header_dat : astropy.io.fits.Header
        Header for the data extension.
    header_var : astropy.io.fits.Header
        Header for the variance extension.
    header_dq : astropy.io.fits.Header
        Optional. Header for the data quality extension. If dq extension is not present, 
        set to None.
    header_wmap : astropy.io.fits.Header
       Optional. Header for the wmap extension. If wmap extension is not present, 
       set to None.
    cubedim : int
        Number of dimensions of the "cube" (1, 2, or 3).
    ncols : int
        Number of columns.  If the data are 1D, this is 1. If the data are 2D, this is the
        number of spectra.
    nrows : int
        Number of rows. If the data are 1D or 2D, this is 1.
    nwave : int
        Number of wavelength elements.

    Examples
    --------
    >>> from q3dfit import readcube
    >>> cube = readcube.Cube('infile.fits')
    >>> cube.convolve(2.5)
    >>> cube.writefits('outfile.fits')

    '''

    def __init__(self,
                 infile: str,
                 datext: int=1,
                 varext: int=2,
                 dqext: Optional[int]=3,
                 wmapext: Optional[int]=4,
                 wavext: Optional[int]=None, 
                 zerodq: bool=False,
                 error: bool=False,
                 invvar: bool=False,
                 fluxnorm: float=1.,
                 fluxunit_in: str='MJy/sr',
                 fluxunit_out: str='erg/s/cm2/micron/sr',
                 usebunit: bool=False, 
                 pixarea_sqas: Optional[float]=None, 
                 waveunit_in: str='micron', 
                 waveunit_out: str='micron',
                 usecunit: bool=False,
#                 linearize: bool=False,
#                 vormap=None,
                 quiet: bool=False,
                 logfile: Optional[str]=None):

        warnings.filterwarnings("ignore")

        self.infile = infile
        try:
            hdu = fits.open(infile, ignore_missing_end=True)
        except FileNotFoundError:
            raise CubeError(infile+' does not exist')
        # fits extensions labels
        self.datext = datext
        self.varext = varext
        self.dqext = dqext
        self.wmapext = wmapext

        # Data
        # Input assumed to be array (nwave, nrows, ncols)
        # Transpose turns this into (ncols, nrows, nwave)
        try:
            self.dat = np.array((hdu[datext].data).T, dtype='float64')
        except (IndexError or KeyError):
            raise CubeError('Data extension not properly specified or absent')

        # Uncertainty
        # This could be one of three expressions of uncertainty:
        # variance, error, or inverse variance.
        try:
            uncert = np.array((hdu[varext].data).T, dtype='float64')
        except (IndexError or KeyError):
            q3dutil.write_msg('Variance extension not properly specified ' +
                'or absent', file=logfile, quiet=quiet)
            raise CubeError('Variance extension not properly specified ' +
                'or absent')
        # convert expression of uncertainty to variance and error
        # if uncert = inverse variance:
        if invvar:
            self.var = 1./copy.copy(uncert)
            self.err = 1./copy.copy(uncert) ** 0.5
        # if uncert = error:
        elif error:
            if (uncert < 0).any():
                q3dutil.write_msg('Cube: Negative values encountered in error ' +
                    'array. Taking absolute value.', file=logfile, quiet=quiet)
            self.err = np.abs(copy.copy(uncert))
            self.var = np.abs(copy.copy(uncert)) ** 2.
        # if uncert = variance:
        else:
            if (uncert < 0).any():
                q3dutil.write_msg('Cube: Negative values encountered in variance ' + 
                    'array. Taking absolute value.', file=logfile, quiet=quiet)
            self.var = np.abs(copy.copy(uncert))
            self.err = np.abs(copy.copy(uncert)) ** 0.5
 
        # Data quality
        if dqext is not None:
            try:
                self.dq = np.array((hdu[dqext].data).T)
            except (IndexError or KeyError):
                raise CubeError('DQ extension not properly specified ' +
                                'or absent')
        else:
            zerodq=True
        # put all dq to good (0), data type bytes
        if zerodq:
            # data type mirrors JWST output DQ
            # https://jwst-pipeline.readthedocs.io/en/stable/jwst/data_products/science_products.html
            self.dq = np.zeros(np.shape(self.dat), dtype='uint32')

        # Weight
        if wmapext is not None:
            try:
                self.wmap = np.array((hdu[wmapext].data).T)
            except (IndexError or KeyError):
                raise CubeError('WMAP extension not properly specified ' +
                                'or absent')
        else:
            self.wmap = None

        # headers
        self.header_phu = hdu[0].header
        self.header_dat = hdu[datext].header
        self.header_var = hdu[varext].header
        if self.dqext is not None:
            self.header_dq = hdu[dqext].header
        else:
            self.header_dq = None
        if self.wmapext is not None:
            self.header_wmap = hdu[wmapext].header
        else:
            self.header_wmap = None
        # copy of data header
        header = copy.copy(self.header_dat)

        # Get shape of data
        datashape = np.shape(self.dat)
        # cube
        if np.size(datashape) == 3:
            ncols = (datashape)[0]
            nrows = (datashape)[1]
            nwave = (datashape)[2]
            if np.max([nrows, ncols]) > nwave:
                raise CubeError('Data cube dimensions not in ' +
                                '[nwave, nrows, ncols] format.')
            CDELT = 'CDELT3'
            CRVAL = 'CRVAL3'
            CRPIX = 'CRPIX3'
            CDELT = 'CDELT3'
            CD = 'CD3_3'
            CUNIT = 'CUNIT3'
            BUNIT = 'BUNIT'
        # 1d array of spectra
        elif np.size(datashape) == 2:
            q3dutil.write_msg('Cube: Reading 2D image. Assuming dispersion ' +
                'is along rows.', file=logfile, quiet=quiet)
            nrows = 1
            ncols = (datashape)[1]
            nwave = (datashape)[-1]
            CDELT = 'CDELT1'
            CRVAL = 'CRVAL1'
            CRPIX = 'CRPIX1'
            CDELT = 'CDELT1'
            CD = 'CD1_1'
            CUNIT = 'CUNIT1'
            BUNIT = 'BUNIT'
        # single spectrum
        elif np.size(datashape) == 1:
            nrows = 1
            ncols = 1
            nwave = (datashape)[0]
            CDELT = 'CDELT1'
            CRVAL = 'CRVAL1'
            CRPIX = 'CRPIX1'
            CDELT = 'CDELT1'
            CD = 'CD1_1'
            CUNIT = 'CUNIT1'
            BUNIT = 'BUNIT'
        self.ncols = int(ncols)
        self.nrows = int(nrows)
        self.nwave = int(nwave)
        self.cubedim = np.size(datashape)

        # check on wavelength and flux units
        self.waveunit_in = waveunit_in
        self.fluxunit_in = fluxunit_in
        self.waveunit_out = waveunit_out
        self.fluxunit_out = fluxunit_out
        # set with header if available
        try:
            cunitval = header[CUNIT]
            # switch from fits header string for microns to
            # astropy.units string for microns
            if cunitval == 'um':
                cunitval = 'micron'
            if cunitval != waveunit_in and not usecunit:
                q3dutil.write_msg('Cube: Wave units in header (CUNIT) differ from ' +
                    'waveunit_in='+waveunit_in+'; ignoring CUNIT. ' +
                    'To override, set usecunit=True.', file=logfile, quiet=quiet)
            else:
                self.waveunit_in = cunitval
        except KeyError:
            q3dutil.write_msg('Cube: No wavelength units in header; using ' +
                waveunit_in, file=logfile, quiet=quiet)
        try:
            bunitval = header[BUNIT]
            if bunitval != fluxunit_in and not usebunit:
                q3dutil.write_msg('Cube: Flux units in header (BUNIT='+bunitval+
                    ') differ from ' +
                    'fluxunit_in='+fluxunit_in+'; ignoring BUNIT. ' +
                    'To override, set usebunit=True.', file=logfile, quiet=quiet)
            else:
                self.fluxunit_in = bunitval
        except KeyError:
            q3dutil.write_msg('Cube: No flux units in header; using ' + fluxunit_in,
                file=logfile, quiet=quiet)
        # Remove whitespace from units
        self.waveunit_in.strip()
        self.fluxunit_in.strip()

        # Look for pixel area in square arcseconds
        try:
            self.pixarea_sqas = header['PIXAR_A2']
        except:
            self.pixarea_sqas = pixarea_sqas
            if pixarea_sqas is None:
                q3dutil.write_msg('Cube: No pixel area in header or specified in call; ' +
                    'no correction for surface brightness flux units.',
                    file=logfile, quiet=quiet)

        # cases of weirdo flux units
        # remove string literal '/A/'
        if self.fluxunit_in.find('/A/') != -1:
            self.fluxunit_in = self.fluxunit_in.replace('/A/', '/Angstrom/')
        # remove string literal '/Ang' if it's at the end of the string
        if self.fluxunit_in.find('/Ang') != -1 and \
            self.fluxunit_in.find('/Ang') == len(self.fluxunit_in)-4:
            self.fluxunit_in = self.fluxunit_in.replace('/Ang', '/Angstrom')
        # remove scientific notation # and trailing whitespace
        # https://stackoverflow.com/questions/18152597/extract-scientific-number-from-string
        # with some modification
        # https://developers.google.com/edu/python/regular-expressions
        match_number = re.compile(r'-?\s+[0-9]+\.?[0-9]*(?:[Ee]' +
                                  r'\s*-?\s*[0-9]+)\s*')
        self.fluxunit_in = re.sub(match_number, '', self.fluxunit_in)

        # reading wavelength from wavelength extension
        self.wavext = wavext
        if wavext is not None:
            try:
                self.wave = hdu[wavext].data
                q3dutil.write_msg("Assuming constant dispersion.", file=logfile, quiet=quiet)
            except (IndexError or KeyError):
                raise CubeError('Wave extension not properly specified ' +
                                'or absent')
            self.crval = self.wave[0]
            self.crpix = 1
            # assume constant dispersion
            self.cdelt = self.wave[1]-self.wave[0]
        else:
            try:
                self.crval = header[CRVAL]
                self.crpix = header[CRPIX]
            except KeyError:
                raise CubeError('Cannot compute wavelengths; ' +
                                'CRVAL and/or CRPIX missing')
            if CDELT in header:
                self._wav0 = header[CRVAL] - (header[CRPIX] - 1) * \
                    header[CDELT]
                self.wave = self._wav0 + np.arange(nwave)*header[CDELT]
                self.cdelt = header[CDELT]
            elif CD in header:
                self._wav0 = header[CRVAL] - (header[CRPIX] - 1) * header[CD]
                self.wave = self._wav0 + np.arange(nwave)*header[CD]
                self.cdelt = header[CD]
            else:
                raise CubeError('Cannot find or compute wavelengths')
        # explicitly cast as float64
        self.wave = self.wave.astype('float64')

        # convert wavelengths if requested
        if self.waveunit_in != self.waveunit_out:
            wave_in = self.wave * u.Unit(self.waveunit_in)
            self.wave = wave_in.to(u.Unit(self.waveunit_out)).value

        '''
        This feature is not currently implemented.
        # indexing of Voronoi-tessellated data
        if vormap:
            ncols = np.max(vormap)
            nrows = 1
            vordat = np.zeros((ncols, nrows, nwave))
            vorvar = np.zeros((ncols, nrows, nwave))
            vordq = np.zeros((ncols, nrows, nwave))
            vorcoords = np.zeros((ncols, 2), dtype=int)
            nvor = np.zeros((ncols))
            for i in np.arange(ncols):
                ivor = np.where(vormap == i+1)
                xyvor = [ivor[0][0], ivor[0][1]]
                vordat[i, 0, :] = self.dat[xyvor[0], xyvor[1], :]
                if self.var is not None:
                    vorvar[i, 0, :] = self.var[xyvor[0], xyvor[1], :]
                if self.dq is not None:
                    vordq[i, 0, :] = self.dq[xyvor[0], xyvor[1], :]
                vorcoords[i, :] = xyvor
                nvor[i] = (np.shape(ivor))[1]
            self.dat = vordat
            self.var = vorvar
            self.dq = vordq
        '''

        # Flux unit conversions
        # default working flux unit is erg/s/cm^2/um/sr or erg/s/cm^2/um
        convert_flux = np.float64(1.)
        if 'Jy' in self.fluxunit_in and 'Jy' not in self.fluxunit_out:
            # IR units: https://coolwiki.ipac.caltech.edu/index.php/Units
            # default input flux unit is MJy/sr
            # 1 Jy = 10^-26 W/m^2/Hz
            # first fac: MJy to Jy
            # second fac: 10^-26 W/m^2/Hz / Jy * 10^-4 m^2/cm^-2 * 10^7 erg/s/W
            #    = 10^-23 erg/cm^-2/s/Hz
            # third fac: c/lambda**2 Hz/micron, with lambda^2 in m*micron
            wave_out = self.wave * u.Unit(self.waveunit_out)
            convert_flux = 1e-23 * c.value / wave_out.to('m').value / \
                wave_out.to('micron').value
            if 'MJy' in self.fluxunit_in:
                convert_flux *= 1e6

        # case of input flux_lambda cgs units per Angstrom:
        elif '/Angstrom' in self.fluxunit_in:
            convert_flux = np.float64(1e4)

        # case of output flux_lambda cgs units per Angstrom:
        if '/Angstrom' in self.fluxunit_out:
            convert_flux /= np.float64(1e4)

        # remove /sr or /arcsec2
        if '/sr' in self.fluxunit_in and \
            self.pixarea_sqas is not None:
            convert_flux *= 1./(206265.*206265.)*self.pixarea_sqas
            if '/sr' in self.fluxunit_out:
                self.fluxunit_out = self.fluxunit_out.replace('/sr', '')
        if '/sr' not in self.fluxunit_in and '/sr' in self.fluxunit_out:
            self.fluxunit_out = self.fluxunit_out.replace('/sr', '')
        if '/arcsec2' in self.fluxunit_in and \
            self.pixarea_sqas is not None:
            convert_flux *= self.pixarea_sqas
            if '/arcsec2' in self.fluxunit_out:
                self.fluxunit_out = self.fluxunit_out.replace('/arcsec2', '')
        
        self.dat = self.dat * convert_flux
        if self.var is not None:
            self.var = self.var * convert_flux**2
        if self.err is not None:
            self.err = self.err * convert_flux

        self.fluxnorm = fluxnorm
        self.dat = self.dat / fluxnorm
        if self.var is not None:
            self.var = self.var / fluxnorm**2
        if self.err is not None:
            self.err = self.err / fluxnorm
        
        '''
        This feature is not currently implemented.
        # linearize in the wavelength direction
        if linearize:
            waveold = copy.copy(self.wave)
            datold = copy.copy(self.dat)
            varold = copy.copy(self.var)
            dqold = copy.copy(self.dq)
            self.crpix = 1
            self.cdelt = (waveold[-1]-waveold[0]) / (self.nwave-1)
            wave = np.linspace(waveold[0], waveold[-1], num=self.nwave)
            self.wave = wave
            if self.cubedim == 1:
                spldat = interpolate.splrep(waveold, datold, s=0)
                self.dat = interpolate.splev(wave_in, spldat, der=0)
                splvar = interpolate.splrep(waveold, varold, s=0)
                self.var = interpolate.splev(wave, splvar, der=0)
                spldq = interpolate.splrep(waveold, dqold, s=0)
                self.dq = interpolate.splev(wave, spldq, der=0)
            elif self.cubedim == 2:
                for i in np.arange(self.ncols):
                    spldat = interpolate.splrep(waveold, datold[i, :], s=0)
                    self.dat[i, :] = interpolate.splev(wave, spldat, der=0)
                    if self.var is not None:
                        splvar = interpolate.splrep(waveold, varold[i, :], s=0)
                        self.var[i, :] = interpolate.splev(wave, splvar, der=0)
                    if self.dq is not None:
                        spldq = interpolate.splrep(waveold, dqold[i, :], s=0)
                        self.dq[i, :] = interpolate.splev(wave, spldq, der=0)
            elif self.cubedim == 3:
                for i in np.arange(self.ncols):
                    for j in np.arange(self.nrows):
                        spldat = interpolate.splrep(waveold, datold[i, j, :],
                            s=0)
                        self.dat[i, j, :] = interpolate.splev(wave, spldat,
                            der=0)
                        if self.var is not None:
                            splvar = interpolate.splrep(waveold, varold[i, j, :],
                                s=0)
                            self.var[i, j, :] = interpolate.splev(wave, splvar,
                                der=0)
                        if self.dq is not None:
                            spldq = interpolate.splrep(waveold, dqold[i, j, :],
                                s=0)
                            self.dq[i, j, :] = interpolate.splev(wave, spldq,
                                der=0)
            if not quiet:
                print('Cube: Interpolating DQ; values > 0.01 set to 1.',
                      file=logfile)
            if self.dq is not None:
                ibd = np.where(self.dq > 0.01)
                if np.size(ibd) > 0:
                    self.dq[ibd] = 1
            if self.var is not None:
                self.err = copy.copy(self.var)**0.5
        '''

        # close the fits file
        hdu.close()


    def about(self):
        '''
        Print information about the cube to stdout.
        '''
        print(f"Size of data cube: [{self.ncols}, {self.nrows}, {self.nwave}]")
        print(f"Wavelength range: [{self.wave[0]:0.5f},",
              f"{self.wave[self.nwave-1]:0.5f}]",
              f"{self.waveunit_out}")
        print(f"Dispersion: {self.cdelt:0.5f} {self.waveunit_out}")
        print(f"Input flux units: {self.fluxunit_in}")
        print(f"Input wave units: {self.waveunit_in}")
        print(f"Output flux units: {self.fluxunit_out}")
        print(f"Output wave units: {self.waveunit_out}")
        print(f"NB: q3dfit uses output units for internal calculations.")


    def convolve(self,
                 refsize: float, 
                 wavescale: bool=False,
                 method: Literal['circle','Gaussian','boxcar']='circle'):
        '''
        Spatially smooth a cube. Updates the data, variance, and error attributes.

        Parameters
        ----------
        refsize
            Pixel size for smoothing algorithm.
        wavescale
            Optional. Scale smoothing size with wavelength. Default is False.
        method
            Optional. Smoothing method. Options are 'circle', 'Gaussian', and 
            'boxcar'. Default is 'circle'.
        '''

        from astropy.convolution import convolve, Box2DKernel, Gaussian2DKernel, \
            Tophat2DKernel

        # check for flux/err = inf or bad dq
        # set to nan, as convolve will deal with nans
        # set dq to 1 to flag as bad
        # [astropy.convolve can interpolate over nans;
        # does this mean we don't need to set dq to 1?]
        indx_bd = (np.isinf(self.dat) | np.isinf(self.var) |
                   (self.dq != 0)).nonzero()
        indx_bd = (np.isinf(self.dat) | np.isinf(self.var) |
                   (self.dq != 0)).nonzero()
        self.dat[indx_bd] = np.nan
        self.err[indx_bd] = np.nan
        self.var[indx_bd] = np.nan
        self.dq[indx_bd] = 1

        # decreasing convolution size with increasing wavelength
        # reference size is at the longest wavelength
        if wavescale:
            cscale = refsize/max(self.wave)
            sizes = cscale/self.wave
        else:
            sizes = np.ndarray(self.nwave)
            sizes[:] = np.float64(refsize)

        for i in np.arange(0, self.nwave):
            if method == 'circle':
                mask = Tophat2DKernel(sizes[i])
            if method == 'Gaussian':
                mask = Gaussian2DKernel(sizes[i])
            if method == 'boxcar':
                mask = Box2DKernel(sizes[i])
            self.dat[:, :, i] = convolve(self.dat[:, :, i], mask)
            self.var[:, :, i] = convolve(self.var[:, :, i], mask)

        # re-compute error
        self.err = np.sqrt(self.var)


    def makeqsotemplate(self,
                        outpy: str,
                        col: Optional[int]=None,
                        row: Optional[int]=None,
                        norm: float=1.,
                        plot: bool=True,
                        radius: float=1.):
        '''
        Extract the quasar spectrum and save it to a numpy file.

        Parameters
        ----------
        outpy
            Name of the numpy save file for the resulting qso spectrum.
        col
            Optional. x value of aperture center, in unity-offset
            coordinates. Default is None, in which case the cube is median-
            collapsed in wavelength and the peak spaxel is used.
        row
            Optional.yx value of aperture center, in unity-offset
            coordinates. Default is None, in which case the cube is median-
            collapsed in wavelength and the peak spaxel is used.
        norm
            Optional. Factor by which to divide output spectrum. Default is 1.
        plot
            Optional. Plot extracted spectrum. Default is True.
        radius
            Optional. Radius for circular extraction. Must be > 0. Default is 1.

        '''

        if col is None and row is None:
            print('makeqsotempate: using peak spaxel in white light as center')
            white_light_image = np.median(self.dat, axis=2)
            white_light_image[np.where(np.isnan(white_light_image))] = 0

            loc_max = np.where(white_light_image == white_light_image.max())
            col = loc_max[0][0]+1
            row = loc_max[1][0]+1

        qsotemplate = {'wave': self.wave}
        spec = self.specextract(col, row, norm=norm, plot=plot, radius=radius)
        if self.dat is not None:
            qsotemplate['flux'] = spec[:, 0]
        if self.var is not None:
            qsotemplate['var'] = spec[:, 1]
        if self.dq is not None:
            qsotemplate['dq'] = spec[:, 2]

        np.save(outpy, qsotemplate)


    def specextract(self,
                    col: float,
                    row: float,
                    radius: float=1.,
                    norm: float=1.,
                    plot: bool=True,
                    ylim: Optional[tuple]=None):
        '''
        Extract a spectrum in a single spaxel or a circular aperture.

        Parameters
        ----------
        col
            x value of aperture center, in unity-offset coordinates.
        row
            y value of aperture center, in unity-offset coordinates.
        radius
            Optional. Radius for circular extraction. If 0, extract single 
            spaxel. Default is 1.
        norm
            Optional. Factor by which to divide output spectrum. Default is 1.
        plot
            Optional. Plot extracted spectrum.
        ylim
            Optional. Y-axis limits for plot. Default is None, in which case
            the plot will autoscale.

        Returns
        -------
        numpy.ndarray
            Array of size nwave x (1-3), including data and var and/or dq if
            present.
        '''

        import matplotlib.pyplot as plt
        import photutils

        # second dimension of output array
        next = 1
        if self.var is not None:
            next += 1
        if self.dq is not None:
            next += 1
        # create output array
        # Note that all planes have same dtype (float)
        spec = np.ndarray((self.nwave, next))

        # Set extraction to single pixel if zero radius is specified
        # Round col, row to nearest integer first
        if self.nrows == 1:
            if self.ncols == 1:
                spec[:, 0] = self.dat / norm
                if self.var is not None:
                    spec[:, 1] = self.var / norm / norm
                if self.dq is not None:
                    spec[:, 2] = self.dq
            else:
                intcol = round(col)
                spec[:, 0] = self.dat[intcol-1, :] / norm
                if self.var is not None:
                    spec[:, 1] = self.var[intcol-1, :] / norm / norm
                if self.dq is not None:
                    spec[:, 2] = self.dq[intcol-1, :]
        elif radius == 0.:
            intcol = round(col)
            introw = round(row)
            spec[:, 0] = self.dat[intcol-1, introw-1, :] / norm
            if self.var is not None:
                spec[:, 1] = self.var[intcol-1, introw-1, :] / norm / norm
            if self.dq is not None:
                spec[:, 2] = self.dq[intcol-1, introw-1, :]
        else:
            # create circular mask
            cent = np.array([row-1., col-1.])
            aper = photutils.aperture.CircularAperture(cent, radius)

            # loop through wavelengths
            for i in np.arange(0, self.nwave):
                specdat = \
                    photutils.aperture.aperture_photometry(
                        self.dat[:, :, i], aper)
                spec[i, 0] = specdat['aperture_sum'].data[0] / norm
                # for now simply sum var as well
                if self.var is not None:
                    specvar = \
                        photutils.aperture.aperture_photometry(
                            self.var[:, :, i], aper)
                    spec[i, 1] = specvar['aperture_sum'].data[0] / norm / norm
                # for now simply sum dq
                if self.dq is not None:
                    specdq = \
                        photutils.aperture.aperture_photometry(
                            self.dq[:, :, i], aper)
                    spec[i, 2] = specdq['aperture_sum'].data[0]

        if plot:
            plt.plot(self.wave, spec[:, 0])
            plt.plot(self.wave, np.sqrt(spec[:, 1]))
            if ylim is not None:
                plt.ylim(ylim)
            plt.show()

        return spec


    def reproject(self,
                  newheader: object,
                  new2dshape: ArrayLike,
                  newpixarea_sqas: float,
                  parallel: bool=True):
        '''
        Reproject cube onto a new WCS. Overwrites the data, variance, dq, and
        error attributes. Also updates the pixel area, nrows, ncols, and
        PIXEL_A2 keyword in the header.

        Parameters
        ----------
        newheader
            Header object containing new WCS information.
        new2dshape
            2-element array containing (nrows, ncols).
        newpixarea_sqas
            Pixel area in square arcseconds.
        parallel
            Optional. Use parallel processing when computing
            reprojection. Default is True.
        '''
        from reproject import reproject_exact

        head2d = copy.copy(self.header_dat)
        if 'NAXIS3' in head2d:
            for key in list(head2d.keys()):
                if "3" in key:
                    head2d.remove(key)
            head2d['NAXIS'] = 2
            head2d['WCSAXES'] = 2

        nstack = 3 # for dat, var, dq
        dqind = 2
        # temporary array has size (nstack, ncols, nrows, nwave)
        # reproject needs (nrows, ncols), but cube stores data
        # as (ncols, nrows)
        stack_in = np.ndarray((nstack,) + (self.ncols, self.nrows) +
                              (self.nwave,))
        stack_rp = np.ndarray((nstack,) + new2dshape[::-1] +
                              (self.nwave,))
        stack_in[0, :, :, :] = self.dat
        stack_in[1, :, :, :] = self.var
        stack_in[dqind, :, :, :] = self.dq
        for i in np.arange(self.nwave):
            # transpose data to (nrows, ncols)
            tmp_rp, _ = \
                reproject_exact((
                    np.transpose(stack_in[:, :, :, i], axes=(0, 2, 1)),
                    head2d), newheader, shape_out=new2dshape,
                    parallel=parallel)
            # transpose back
            stack_rp[:, :, :, i] = np.transpose(tmp_rp, axes=(0, 2, 1))
        self.dat = stack_rp[0, :, :, :]
        self.var = stack_rp[1, :, :, :]
        self.dq = stack_rp[dqind, :, :, :]
        # do surface brightness conversion
        # reproject expects SB units
        if self.pixarea_sqas is not None:
            self.dat /= self.pixarea_sqas / newpixarea_sqas
            self.var /= (self.pixarea_sqas / newpixarea_sqas)**2
            self.pixarea_sqas = newpixarea_sqas
        else:
            print('WARNING: reproject needs original pixel area in square ' +
                  'arcseconds to properly calculate flux units. Input as ' +
                  'pixarea_sqas=VALUE in call to cube if not in header.')

        # store new shape
        self.nrows = new2dshape[0]
        self.ncols = new2dshape[1]

        # re-compute error
        self.err = np.sqrt(self.var)

        # in case we need to write this to disk for later access
        if 'PIXAR_A2' in self.header_dat:
            self.header_dat['PIXAR_A2'] = self.pixarea_sqas


    def writefits(self,
                  outfile: str):
        '''
        Write cube to file outfile. Assumes extension order empty phu (if
        present in original fits file), datext, varext, dqext, and wmapext.

        Parameters
        ----------
        outfile
            Name of output file.

        '''

        print("Output flux units: ", self.fluxunit_out)
        print("Output wave units: ", self.waveunit_out)

        if 'BUNIT' in self.header_dat:
            self.header_dat['BUNIT'] = self.fluxunit_out

        if self.datext != 0:
            # create empty PHU
            hdu1 = fits.PrimaryHDU(header=self.header_phu)
            # create HDUList
            hdul = fits.HDUList([hdu1])
            # add data
            hdul.append(fits.ImageHDU(self.dat.T, header=self.header_dat))
        else:
            # create PHU with data
            hdu1 = fits.PrimaryHDU(self.dat.T, header=self.header_dat)
            # create HDUList
            hdul = fits.HDUList([hdu1])

        # add variance, DQ, and wmap if present
        hdul.append(fits.ImageHDU(self.var.T, header=self.header_var))
        hdul.append(fits.ImageHDU(self.dq.T, header=self.header_dq))
        if self.wmapext is not None:
            hdul.append(fits.ImageHDU(self.wmap.T, header=self.header_wmap))

        hdul.writeto(outfile, overwrite=True)


    def writespec(self,
                  spec: np.ndarray[float],
                  outfile: str):
        '''
        Write extracted spectrum to disk. Assumes extension order empty phu (if
        present in original fits file), datext, varext, dqext, and wmapext.

        Parameters
        ----------
        spec
            Array of size nwave x (1-3), including data and var and/or dq if
            present.
        outfile
            Name of output file.
        '''

        print("Output flux units: ", self.fluxunit_out)
        print("Output wave units: ", self.waveunit_out)

        # deal with simplest case for now
        if self.wavext is None:
            hdr = fits.Header({'CUNIT1': self.waveunit_out,
                               'CRPIX1': self.crpix,
                               'CDELT1': self.cdelt,
                               'CRVAL1': self.crval,
                               'CTYPE1': 'WAVE',
                               'BUNIT': self.fluxunit_out})

        if self.datext != 0:
            # create empty PHU
            hdu1 = fits.PrimaryHDU(header=self.header_phu)
            # create HDUList
            hdul = fits.HDUList([hdu1])
            # add data
            hdul.append(fits.ImageHDU(spec[:, 0], header=hdr))
        else:
            # create PHU with data
            hdu1 = fits.PrimaryHDU(spec[:, 0], header=hdr)
            # create HDUList
            hdul = fits.HDUList([hdu1])

        # add variance, DQ, and wmap if present
        hdul.append(fits.ImageHDU(spec[:, 1], header=hdr))
        hdul.append(fits.ImageHDU(spec[:, 2], header=hdr))
        # if self.wmapext is not None:
        #     hdul.append(fits.ImageHDU(self.wmap.T, header=self.header_wmap))

        hdul.writeto(outfile, overwrite=True)
