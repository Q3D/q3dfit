'''
This module contains the Cube class for reading in and containing a data cube
'''

import copy
import numpy as np
import re
import warnings

from astropy import units as u
from astropy.constants import c
from astropy.io import fits
from scipy import interpolate
from sys import stdout

from q3dfit.exceptions import CubeError

__all__ = [
    'Cube'
    ]


class Cube:
    '''
    Read in and contain a data cube and associated information.

    Parameters
    -----------
    infile : str
         Name of input FITS file.
    datext : int, optional
    varext : int, optional
    dqext : int, optional
    wmapext : int, optional
        Extension numbers of data, variance, data quality, and weight. Set to
        None if not present. Defaults are 1, 2, 3, and 4.
    wavext : int, optional
        Extension number of wavelength data.
    error : bool, optional
    fluxnorm : float, optional
        Factor by which to divide the data.
    fluxunit_in : str, optional
    fluxunit_out : str, optional
        Flux unit of input data cube and unit carried by object.
        Defaults are MJy/sr (JWST default) and erg/s/cm2/micron/sr. Must be
        parseable by astropy.units.
    invvar : bool, optional
        If the variance extension contains errors or inverse variance rather
        than variance, the routine will convert to variance.
    linearize : bool, optional
        Resample the input wavelength scale so it is linearized.
    logfile : str, optional
        File for progress messages.
    pixarea_sqas : float, optional
        Pixel (spaxel) area in square arcseconds. If present and flux units are
        per sr or per square arcseconds, will convert surface brightness
        units to flux per voxel.
    quiet : bool, optional
        Suppress progress messages.
    vormap : array_like, optional
        2D array specifying a Voronoi bin to which each spaxel belongs.
    waveunit_in : str, optional
    waveunit_out : str, optional
        Wavelength unit of input data cube and unit carried by object.
        Default is micron; other option is Angstrom.
    zerodq : bool, optional
        Zero out the DQ array.

    Attributes
    ----------
    dat : ndarray
    var : ndarray
    err : ndarray
    dq : ndarray
    wmap : ndarray
        Data, variance, error, data quality, and weight (1/variance) arrays.
        Errors are recorded as nan when the variances are negative.
    wave : ndarray
        Wavelength array.
    cdelt : float
    crpix : int
    crval : float
        WCS wavelength variables.
    header_phu :
    header_dat :
    header_var :
    header_dq :
    header_wmap :
        Headers of the various extensions (phu = first, or primary, header
        data unit)
    cubedim : int
    ncols : int
    nrows : int
    nwave : int
        Dimensions of the cube.

    Examples
    --------
    >>> from q3dfit import readcube
    >>> cube = readcube.Cube('infile.fits')
    >>> cube.convolve(2.5)
    >>> cube.writefits('outfile.fits')

    Notes
    -----
    '''

    def __init__(self, infile, datext=1, varext=2, dqext=3, wmapext=4,
                 error=False, fluxunit_in='MJy/sr',
                 fluxnorm=None, fluxunit_out='erg/s/cm2/micron/sr',
                 invvar=False, linearize=False, logfile=stdout, quiet=True,
                 pixarea_sqas=None,
                 vormap=None, waveunit_in='micron', waveunit_out='micron',
                 wavext=None, zerodq=False):

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
        try:
            self.dat = np.array((hdu[datext].data).T, dtype='float32')
        except (IndexError or KeyError):
            raise CubeError('Data extension not properly specified or absent')

        # Variance/Error
        if varext is not None:
            try:
                self.var = np.array((hdu[varext].data).T, dtype='float32')
            except (IndexError or KeyError):
                raise CubeError('Variance extension not properly specified ' +
                                'or absent')
            if invvar:
                self.var = 1./self.var
                self.err = np.sqrt(self.var)
            if error:
                self.err = self.var
                self.var = self.var**2.
            else:
                badvar = np.where(self.var < 0)
                if np.size(badvar) > 0:
                    print('Cube: Negative values encountered in variance ' +
                          'array. Taking absolute value.', file=logfile)
                self.var = np.abs(self.var)
                self.err = np.array(copy.copy(self.var) ** 0.5)
        else:
            self.var = None
            self.err = None

        # Data quality
        if dqext is not None:
            try:
                self.dq = np.array((hdu[dqext].data).T)
            except (IndexError or KeyError):
                raise CubeError('DQ extension not properly specified ' +
                                'or absent')
        else:
            self.dq = None
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
        if self.var is not None:
            self.header_var = hdu[varext].header
        if self.dq is not None:
            self.header_dq = hdu[dqext].header
        if self.wmap is not None:
            self.header_wmap = hdu[wmapext].header
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
                                '[nwave, nrows, ncols] format')
            CDELT = 'CDELT3'
            CRVAL = 'CRVAL3'
            CRPIX = 'CRPIX3'
            CDELT = 'CDELT3'
            CD = 'CD3_3'
            CUNIT = 'CUNIT3'
            BUNIT = 'BUNIT'
        # 1d array of spectra
        elif np.size(datashape) == 2:
            if not quiet:
                print('Cube: Reading 2D image. Assuming dispersion ' +
                      'is along rows.', file=logfile)
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
        waveunit_tmp = waveunit_in
        fluxunit_tmp = fluxunit_in
        self.waveunit_out = waveunit_out
        self.fluxunit_out = fluxunit_out
        # set with header if available
        try:
            self.waveunit_in = header[CUNIT]
        except KeyError:
            self.waveunit_in = waveunit_tmp
            if not quiet:
                print('Cube: No wavelength units in header; assuming micron.',
                      file=logfile)
        try:
            self.fluxunit_in = header[BUNIT]
        except KeyError:
            self.fluxunit_in = fluxunit_tmp
            if not quiet:
                print('Cube: No flux units in header; assuming MJy/sr',
                      file=logfile)
        # Remove whitespace from units
        self.waveunit_in.strip()
        self.fluxunit_in.strip()

        # Look for pixel area in square arcseconds
        try:
            self.pixarea_sqas = header['PIXAR_A2']
        except:
            self.pixarea_sqas = pixarea_sqas
            if not quiet and pixarea_sqas is None:
                print('Cube: No pixel area in header or specified in call;' +
                      'no correction for surface brightness flux units.',
                      file=logfile)

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
        match_number = re.compile('-?\ *[0-9]+\.?[0-9]*(?:[Ee]' +
                                  '\ *-?\ *[0-9]+)\ *')
        self.fluxunit_in = re.sub(match_number, '', self.fluxunit_in)

        # reading wavelength from wavelength extention
        self.wavext = wavext
        if wavext is not None:
            try:
                self.wave = hdu[wavext].data
            except (IndexError or KeyError):
                raise CubeError('Wave extension not properly specified ' +
                                'or absent')
            self.crval = 0
            self.crpix = 1
            self.cdelt = 1
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
        # explicitly cast as float32
        self.wave = self.wave.astype('float32')

        # convert wavelengths if requested
        if self.waveunit_in != self.waveunit_out:
            wave_in = self.wave * u.Unit(self.waveunit_in)
            self.wave = wave_in.to(u.Unit(self.waveunit_out)).value

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
                vorvar[i, 0, :] = self.var[xyvor[0], xyvor[1], :]
                vordq[i, 0, :] = self.dq[xyvor[0], xyvor[1], :]
                vorcoords[i, :] = xyvor
                nvor[i] = (np.shape(ivor))[1]
            self.dat = vordat
            self.var = vorvar
            self.dq = vordq

        # Flux unit conversions
        # default working flux unit is erg/s/cm^2/um/sr or erg/s/cm^2/um
        convert_flux = np.float32(1.)
        if 'MJy' in self.fluxunit_in and 'MJy' not in self.fluxunit_out:
            # IR units: https://coolwiki.ipac.caltech.edu/index.php/Units
            # default input flux unit is MJy/sr
            # 1 Jy = 10^-26 W/m^2/Hz
            # first fac: MJy to Jy
            # second fac: 10^-26 W/m^2/Hz / Jy * 10^-4 m^2/cm^-2 * 10^7 erg/s/W
            # third fac: c/lambda**2 Hz/micron, with lambda^2 in m*micron
            wave_out = self.wave * u.Unit(self.waveunit_out)
            convert_flux = 1e6 * 1e-23 * c.value / wave_out.to('m').value / \
                wave_out.to('micron').value

        # case of input flux_lambda cgs units per Angstrom:
        elif '/Angstrom' in self.fluxunit_in:
            convert_flux = np.float32(1e4)

        # case of output flux_lambda cgs units per Angstrom:
        if '/Angstrom' in self.fluxunit_out:
            convert_flux /= np.float32(1e4)

        # remove /sr or /arcsec2
        if '/sr' in self.fluxunit_in and \
            self.pixarea_sqas is not None:
            convert_flux *= 1./(206265.*206265.)*self.pixarea_sqas
            if '/sr' in self.fluxunit_out:
                self.fluxunit_out = self.fluxunit_out.replace('/sr', '')
        if '/arcsec2' in self.fluxunit_in and \
            self.pixarea_sqas is not None:
            convert_flux *= self.pixarea_sqas
            if '/arcsec2' in self.fluxunit_out:
                self.fluxunit_out = self.fluxunit_out.replace('/arcsec2', '')

        self.dat = self.dat * convert_flux
        self.var = self.var * convert_flux**2
        self.err = self.err * convert_flux

        if fluxnorm is not None:
            self.dat = self.dat / fluxnorm
            self.var = self.var / fluxnorm**2
            self.err = self.err / fluxnorm

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
            spldat = interpolate.splrep(waveold, datold, s=0)
            self.dat = interpolate.splev(waveold, spldat, der=0)
            splvar = interpolate.splrep(waveold, varold, s=0)
            self.var = interpolate.splev(waveold, splvar, der=0)
            spldq = interpolate.splrep(waveold, dqold, s=0)
            self.dq = interpolate.splev(waveold, spldq, der=0)
            if not quiet:
                print('Cube: Interpolating DQ; values > 0.01 set to 1.',
                      file=logfile)
            ibd = np.where(self.dq > 0.01)
            if np.size(ibd) > 0:
                self.dq[ibd] = 1

        # close the fits file
        hdu.close()

    def about(self):
        print("Size of data cube: [", self.ncols, ",", self.nrows, ",",
              self.nwave, "]")
        print("Wavelength range: [", self.wave[0], ",",
              self.wave[self.nwave-1], "] ", self.waveunit_out)

    def convolve(self, refsize, wavescale='none', method='circle'):

        import photutils
        from scipy.ndimage import convolve  # gaussian_filter, median_filter

        '''
        Spatially smooth a cube.

        Parameters
        ----------
        refsize : float
            Pixel size for smoothing algorithm.
        wavescale : str, optional
            Option to scale smoothing size with wavelength
        method : str, optional
            Smoothing method.

        Returns
        -------
        obj
            A modified cube object.
        '''

        # check for flux/err = inf/nan or bad dq
        indx_bd = (np.isnan(self.dat) | np.isinf(self.dat) |
                   np.isnan(self.var) | np.isinf(self.var) |
                   (self.dq != 0)).nonzero()
        self.dat[indx_bd] = 0.
        self.err[indx_bd] = 0.
        self.var[indx_bd] = 0.
        self.dq[indx_bd] = 1

        # decreasing convolution size with increasing wavelength
        if wavescale == 'diff':
            cscale = refsize/max(self.wave)
            sizes = cscale/self.wave
        elif wavescale == 'none':
            sizes = np.ndarray(self.nwave)
            sizes[:] = np.float(refsize)

        for i in np.arange(0, self.nwave):
            # 2D Gaussian with sigma = 1.5 at long wavelength, 1 at short
            #if method == 'Gaussian':
            #   cube_convolved[:,:,i] = gaussian_filter(cube_convolved[:,:,i],
            #                                           1+sizes[i])
            #if method == 'median-circular':
            #    cube_convolved[:,:,i] = median_filter(cube_convolved[:,:,i],
            #                                          footprint=circular_mask(10))
            if method == 'circle':
                # mask = circular_mask(np.int(sizes[i]))
                mask = \
                    ((photutils.aperture.CircularAperture(
                        (0., 0.), sizes[i])).to_mask()).data
                self.dat[:, :, i] = convolve(self.dat[:, :, i], mask)
                self.var[:, :, i] = convolve(self.var[:, :, i], mask)
        #if method == 'boxcar':
        #    cube_convolved[:,:,i] = convolve(cube_convolved[:,:,i],box_mask(2+np.int(sizes[i])))

        # re-compute error
        self.err = np.sqrt(self.var)

    def makeqsotemplate(self, outpy, norm=1., plot=True, radius=1.):

        import matplotlib.pyplot as plt

        '''Extract the quasar spectrum

        Parameters
        ----------
        outpy : string
                Name of the numpy save file for the resulting qso spectrum
        norm : float, optional
            Factor by which to divide output spectrum.
        plot : bool, optional
            Plot extracted spectrum.
        radius : float, optional
            Radius for circular extraction. Must be > 0.

        Returns
        -------
        dictionary
            {wave,flux,dq}
        '''

        white_light_image = np.median(self.dat, axis=2)
        white_light_image[np.where(np.isnan(white_light_image))] = 0

        loc_max = np.where(white_light_image == white_light_image.max())
        map_x = np.tile(np.indices((self.ncols, 1))[0], (1, self.nrows))
        map_y = np.tile(np.indices((self.nrows, 1))[0].T[0], (self.ncols, 1))
        map_r = np.sqrt((map_x - loc_max[0][0])**2 +
                        (map_y - loc_max[1][0])**2)
        iap = np.where(map_r <= radius)

        qsotemplate = {'wave': self.wave}
        if self.dat is not None:
            # norm = np.median(self.dat[loc_max[0][0], loc_max[1][0]])
            qsotemplate['flux'] = self.dat[iap[0][:], iap[1][:], :].sum(0) \
                / norm
        if plot:
            fig, ax = plt.subplots()
            ax.plot(self.wave, qsotemplate['flux'])
            #plt.plot(self.wave, qsotemplate['flux'])
            #plt.show()
        if self.var is not None:
            qsotemplate['var'] = self.var[iap[0][:], iap[1][:], :].sum(0) \
                / norm / norm
        if self.dq is not None:
            qsotemplate['dq'] = self.dq[iap[0][:], iap[1][:], :].sum(0)

        np.save(outpy, qsotemplate)

    def specextract(self, col, row, method='circle', norm=1., plot=True,
                    radius=1.):

        import matplotlib.pyplot as plt
        import photutils

        '''
        Extract a spectrum

        Parameters
        ----------
        cent : ndarray
            x, y pixel coordinates of aperture center
        method : string, optional
            Method of extraction. Default is circular aperture of radius r.
        radius : float, optional
            Radius for circular extraction. Must be > 0.
        norm : float, optional
            Factor by which to divide output spectrum.
        plot : bool, optional
            Plot extracted spectrum.

        Returns
        -------
        spec : ndarray
            Array of size nwave x (1-3), including data and var and/or dq if
            present.

        '''

        cent = np.array([row-1., col-1.])

        # Set radius to some value v. near 0 if zero is specified, otherwise
        # something chokes
        if radius == 0.:
            radius = 0.01

        # create circular mask
        if method == 'circle':
            aper = photutils.aperture.CircularAperture(cent, radius)

        # second dimension of output array
        next = 1
        if self.var is not None:
            next += 1
        if self.dq is not None:
            next += 1

        # create output array
        # Note that all planes have same dtype (float)
        spec = np.ndarray((self.nwave, next))
        # loop through wavelengths
        for i in np.arange(0, self.nwave):
            specdat = \
                photutils.aperture.aperture_photometry(self.dat[:, :, i], aper)
            spec[i, 0] = specdat['aperture_sum'].data[0] / norm
            # for now simply sum var as well
            if self.var is not None:
                specvar = \
                    photutils.aperture.aperture_photometry(self.var[:, :, i],
                                                           aper)
                spec[i, 1] = specvar['aperture_sum'].data[0] / norm / norm
            # for now simply sum dq
            if self.dq is not None:
                specdq = \
                    photutils.aperture.aperture_photometry(self.dq[:, :, i],
                                                           aper)
                spec[i, 2] = specdq['aperture_sum'].data[0]

        if plot:
            plt.plot(self.wave, spec[:, 0])
            plt.show()

        return spec

    def writefits(self, outfile):
        '''
        Write cube to file outfile. Assumes extension order empty phu (if
        present in original fits file), datext, varext, dqext, and wmapext.
        '''

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
        if self.varext is not None:
            hdul.append(fits.ImageHDU(self.var.T, header=self.header_var))
        if self.dqext is not None:
            hdul.append(fits.ImageHDU(self.dq.T, header=self.header_dq))
        if self.wmapext is not None:
            hdul.append(fits.ImageHDU(self.wmap.T, header=self.header_wmap))

        hdul.writeto(outfile, overwrite=True)

    def writespec(self, spec, outfile):
        '''
        Write extracted spectrum to disk. Assumes extension order empty phu (if
        present in original fits file), datext, varext, dqext, and wmapext.
        '''

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
        if self.varext is not None:
            hdul.append(fits.ImageHDU(spec[:, 1], header=hdr))
        if self.dqext is not None:
            hdul.append(fits.ImageHDU(spec[:, 2], header=hdr))
        # if self.wmapext is not None:
        #     hdul.append(fits.ImageHDU(self.wmap.T, header=self.header_wmap))

        hdul.writeto(outfile, overwrite=True)


if __name__ == "__main__":
    cube = Cube('data.fits')
