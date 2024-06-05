#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: Yuzo Ishikawa

Convolves an input spectrum by the JWST NIRSPEC + MIRI grating resolution. By
default, this will convolve the input spectrum with a wavelength dependent
dispersion relation. The user also can choose alternative methods.

METHOD 0 = flat convolution by wavelength bins (takes median resolving power
    at middle of wavelength bin)
METHOD 1 = NOT IMPLEMENTED. Convolution by dispersion curve by looping through
    each pixel element
METHOD 2 = DEFAULT. Convolution by dispersion curve using Cappellari method

How to run:
> import spectConvol
Create a spectConvol object for fitting:
> spConv = spectConvol.spectConvol()

Within the code itself, to convolve:
> spectOUT = spConv.gauss_convolve(waveIN, fluxIN, INST='nirspec',
                                   GRATING='G140M/F070LP', METHOD=2)
This requires a wrapper that calls gauss_convolve() and organizes the
convolved spectra.

JWST provides NIRSpec dispersion files. However, MIRI is not provided. Instead,
we take a Cubic Spline interpolation based on the curves in the Jdocs website.

"""

import numpy as np
import os
import re
from astropy.constants import c
from astropy.io import fits
from astropy.stats import gaussian_sigma_to_fwhm
import glob
import copy
from scipy.ndimage import gaussian_filter1d
from scipy.interpolate import CubicSpline
import q3dfit.data.dispersion_files


class spectConvol:

    def __init__(self, q3di, overwrite=False):
        '''
        Instantiate the spectConvol class.

        Parameters
        ----------
        q3di : object
            q3dinit class
        overwrite : bool, optional
            Overwrite custom dispersion files.

        Attributes
        ----------
        datDIR : str
            Full directory of dispersion files in the repository.
        init_inst : dict
            Dictionary of dictionaries to hold the dispersion data.
        init_meth : int
            Method of convolution.
        R : array
            Resolution of the instrument; added / updated by spect_convolver().
        
        Raises
        ------
        SystemExit

        Returns
        -------
        None

        '''

        # full directory of dispersion files in repo
        self.datDIR = os.path.join(os.path.abspath(q3dfit.data.__file__)[:-11],
                                   'dispersion_files')
        # Default to method 2
        if 'ws_method' not in q3di.spect_convol:
            q3di.spect_convol['ws_method'] = 2
        if q3di.spect_convol['ws_method'] != 0 and \
            q3di.spect_convol['ws_method'] != 2:
            raise SystemExit('spect_convol: Please specify Method 0 or 2. ' +
                             'Method 1 is not yet implemented.')
        self.init_meth = q3di.spect_convol['ws_method']

        # get list of acutal files in repo
        dispfiles = [dfile.split('/')[-1] for dfile in
                          glob.glob(os.path.join(self.datDIR, '*.fits'))]

        # create the empty dictionary-of-dictionaries to hold the
        # dispersion data
        self.init_inst = {}
        # inst is a telescope+instrument string
        # gratlist is dictionary of grating strings
        for inst, gratdict in q3di.spect_convol['ws_instrum'].items():
            self.init_inst[inst.upper()] = {}
            for grat in gratdict:
                self.init_inst[inst.upper()][grat.upper()] = None

        # load dispersion files and get dispersion data
        for inst, gratlist in self.init_inst.items():
            # For MIRI, use resolution vs. wavelength from Jones et al. 2023
            # https://ui.adsabs.harvard.edu/abs/2023MNRAS.523.2519J/abstract
            if inst != 'JWST_MIRI':
                # cycle through gratings and look for corresponding file
                for grat in gratlist:
                    dfile = '_'.join([inst, grat]).lower() + '_disp.fits'
                    if dfile in dispfiles:
                        self.init_inst[inst][grat] = \
                            self.get_dispersion_data(dfile)
                    # if no file is found, try to make one
                    elif inst == 'flat':
                        # look for convolution method and dispersion
                        # in grating name
                        convmethod = ''.join(re.findall("[a-zA-Z]", grat))
                        dispvalue = '.'.join(re.findall("\d+", grat))
                        # create new dispersion file
                        dobj = dispFile()
                        newfile = \
                            dobj.make_dispersion(dispvalue, TYPE=convmethod)
                        # get data from file
                        self.init_inst[inst][grat] = \
                            self.get_dispersion_data(newfile)
                    else:
                        print('WARNING: no dispersion file found or created ' +
                              'for ' + inst + grat + ' combination. No ' +
                              'convolution performed.')
        return

    def get_dispersion_data(self, dispfile):
        '''
        Get dispersion data from a dispersion file(s).

        Parameters
        ----------
        dispfile : str
            Filename of the dispersion file to load.
        
        Returns
        -------
        outdict : dict
            Dictionary of dispersion data.

        '''

        fullfilepath = os.path.join(self.datDIR, dispfile)
        idispDat = fits.open(fullfilepath)[1].data
        icols = idispDat.columns.names
        # lists of wavelength, resolution R, and delta_lambda
        iwvln = idispDat['WAVELENGTH']  # wavelength [μm]
        irsln, idelw = [], []
        if 'VELOCITY' in icols:
            irsln = c.to('km/s').value/idispDat['VELOCITY']
            idelw = iwvln/irsln
        elif 'R' in icols:
            irsln = idispDat['R']
            if 'DLAM' in icols:
                idelw = idispDat['DLAM']
            else:
                idelw = iwvln/irsln
        elif 'DLAM' in icols:
            idelw = idispDat['DLAM']
            if 'R' in icols:
                irsln = idispDat['R']
            else:
                irsln = iwvln/idelw
        # median wavelength and resolution at median wavelength
        gwvln_med = np.median(iwvln)
        # technically argmin returns first wavelength if
        # the median is between two wavelength points
        grsln_med = irsln[np.argmin(np.abs(iwvln - gwvln_med))]
        outdict = {'gwave': iwvln,
                   'gdwvn': idelw,
                   'gres': irsln,
                   'glamC': gwvln_med,
                   'gResC': grsln_med,
                   'gwvRng': [min(iwvln), max(iwvln)]}
        return outdict

    def spect_convolver(self, wvlIN, fluxIN, wvlcen=None, upsample=False):
        '''
        Convolves an input spectrum by the spectral resolution.

        Parameters
        ----------
        wvlIN : array
            Wavelength array of input spectrum
        fluxIN : array
            Flux array of input spectrum
        wvlcen : float, optional
            Central wavelength of a spectral feature, to check if it falls in the range
            of a grating. Default is None.
        upsample : bool, optional
            Upsample the spectrum by a factor of 10 before convolution. Default is False.

        Adds/updates the following attributes:
        R : array
            Resolution of the instrument

        Returns
        -------
        datOUT : array
            Convolved spectrum

        Raises
        ------
        SystemExit

        '''

        # This assumes that the spectrum has a discrete feature with
        # central wavelength wvlcen, and checks to see if it's in the
        # wavelength range of one (or two) of the gratings specified
        if wvlcen is not None:
            instList = []
            for inst, gratlist in self.init_inst.items():
                if inst != 'JWST_MIRI':
                    for grat in gratlist:
                        wvrng = self.init_inst[inst][grat]['gwvRng']
                        if (wvlcen > wvrng[0]) and (wvlcen < wvrng[1]):
                            instList.append([inst, grat, wvrng])
                        else:
                            raise SystemExit('spectConvol.spect_convolver: ' +
                                             'no specified tel/inst/grating ' +
                                             'combination covers the line ' +
                                             'at ' + wvlcen)
                else:
                    instList.append([inst, 'n/a',
                                     [wvlIN[0], wvlIN[len(wvlIN)-1]]])

        # case MIRI
        if instList[0][0] == 'JWST_MIRI':

            # https://jwst-docs.stsci.edu/jwst-mid-infrared-instrument/miri-observing-modes/miri-medium-resolution-spectroscopy#MIRIMediumResolutionSpectroscopy-wavelengthMRSwavelengthcoverageandspectralresolution
            R = 4603. - 128. * wvlIN + 10.**(-7.4 * wvlIN)
            igdisp2 = wvlIN / R
            igResM = np.median(R)
            iwvIN = wvlIN
            idatIN = fluxIN

        else:

            # grating wavelength overlap check --> take the middle point and
            # stitch together
            if len(instList) == 2:
                wvmin, wvmax = 0, 0
                jwvrng, jgwave, jgdwvn, jgResM = [], [], [], []
                for instoverlap in instList:
                    jinst = instoverlap[0]
                    jigrat = instoverlap[1]
                    jwvrng.append(instoverlap[2])
                    jgwave.append(self.init_inst[jinst][jigrat]['gwave'])
                    jgdwvn.append(self.init_inst[jinst][jigrat]['gdwvn'])
                    jgResM.append(self.init_inst[jinst][jigrat]['gResC'])
                jmin, jmax = [jwvrng[0][0], jwvrng[1][0]], \
                    [jwvrng[0][1], jwvrng[1][1]]
                wvmin = np.max(jmin)
                wvmax = np.min(jmax)
                wvmid = np.median([wvmin, wvmax])
                wlo = np.where(jmin == wvmin)[0][0]
                wup = np.where(jmax == wvmax)[0][0]
                wj1 = np.where(jgwave[wlo] < wvmid)[0]
                wj2 = np.where(jgwave[wup] > wvmid)[0]
                igwave = np.concatenate((jgwave[wlo][wj1], jgwave[wup][wj2]))
                igdwvn = np.concatenate((jgdwvn[wlo][wj1],  jgdwvn[wup][wj2]))
                igResM = np.median(jgResM)

            else:

                jinst = instList[0][0]
                jigrat = instList[0][1]
                igwave = self.init_inst[jinst][jigrat]['gwave']
                igdwvn = self.init_inst[jinst][jigrat]['gdwvn']
                igResM = self.init_inst[jinst][jigrat]['gResC']

            w1 = np.where(wvlIN >= min(igwave))[0]
            w1x = np.where(wvlIN < min(igwave))[0]
            w2 = np.where(wvlIN <= max(igwave))[0]
            w2x = np.where(wvlIN > max(igwave))[0]
            ww = np.intersect1d(w1, w2)
            iwvIN = wvlIN[ww]
            idatIN = fluxIN[ww]

            func1 = CubicSpline(igwave, igdwvn)
            igdisp2 = func1(iwvIN)

            #iR_datconv = np.zeros(len(ww))

        if self.init_meth == 0:
            iR_datconv = \
                self.flat_convolve(iwvIN, idatIN, igResM, WCEN=None)
            self.R = igResM
        # elif self.init_meth == 1:  # needs debugging
        #     iR_datconv = \
        #         self.gaussian_filter1d_looper(iwvIN, idatIN, igdisp)
        elif self.init_meth == 2:
            iR_datconv = \
                self.gaussian_filter1d_ppxf(iwvIN, idatIN, igdisp2,
                                            upsample=upsample)
            self.R = iwvIN / igdisp2

        if instList[0][0] == 'JWST_MIRI':
            datOUT = iR_datconv
        else:
            datOUT = np.concatenate((fluxIN[w1x], iR_datconv, fluxIN[w2x]))

        return datOUT

    # METHOD 0
    def flat_convolve(self, wvlIN, fluxIN, Rspec, WCEN=None):
        '''
        Convolves an input spectrum by a flat resolution R.

        Parameters
        ----------
        wvlIN : array
            Wavelength array of input spectrum 
        fluxIN : array
            Flux array of input spectrum
        Rspec : float
            Resolution of the instrument
        WCEN : float, optional
            Central wavelength of a spectral feature. Default is None.
        
        Returns
        -------
        datconvol : array
            Convolved spectrum

        '''

        wdiff = wvlIN[1]-wvlIN[0]
        mw = WCEN
        if WCEN is None:
            mw = np.median(wvlIN)
        fwhm = (mw/Rspec)  # km/s
        sigma = fwhm/gaussian_sigma_to_fwhm/wdiff
        datconvol = gaussian_filter1d(fluxIN, sigma)
        return datconvol

    '''
    # METHOD 1
    def gaussian_filter1d_looper(self, wvlIN, flxIN, DISP_INFO):
        
        Convolves an input spectrum by ...

        Parameters
        ----------
        wvlIN : array
            Wavelength array of input spectrum
        flxIN : array
            Flux array of input spectrum
        DISP_INFO : array

        Returns
        -------
        dcOUT : array
            Convolved spectrum

        wdiff = wvlIN[1]-wvlIN[0]
        fwhm = DISP_INFO[1]
        sigma = fwhm/gaussian_sigma_to_fwhm/wdiff
        pwvn = []
        pdat = []
        psig = []
        ww = np.where(wvlIN >= min(DISP_INFO[0]))[0]
        cwvn = wvlIN[ww]
        cflx = flxIN[ww]

        wpix = 5
        wi = 0
        while cwvn[wi+wpix] <= max(DISP_INFO[0]):
            iwvn = cwvn[wi:wi+wpix]
            idat = cflx[wi:wi+wpix]
            isig = sigma[wi:wi+wpix]
            pwvn.append(iwvn)
            pdat.append(idat)
            psig.append(np.median(isig))
            wi += wpix

        datconvol = []
        for ip in range(len(pwvn)):
            iflx = pdat[ip]
            gg = gaussian_filter1d(iflx, psig[ip], mode='constant',
                                   cval=pdat[ip][0])
            datconvol.append(gg)
        dcOUT = np.array(datconvol).flatten()
        wvOUT = np.array(pwvn).flatten()
        return wvOUT, dcOUT
    '''
   
    # METHOD 2
    def gaussian_filter1d_ppxf(self, wvlIN, flxIN, DISP_INFO, upsample=False):
        '''
        Convolves an input spectrum by a wavelength dependent dispersion curve.
        
        Parameters
        ----------
        wvlIN : array
            Wavelength array of input spectrum
        flxIN : array
            Flux array of input spectrum
        DISP_INFO : array
            Dispersion information
        upsample : bool, optional
            Upsample the spectrum by a factor of 10 before convolution. Default is False.
        
        Returns
        -------
        conv_spectrum : array
            Convolved spectrum
        
        '''

        wdiff = wvlIN[1]-wvlIN[0]

        # option to upsample by factor of 10 before convolution
        if upsample:
            specint = CubicSpline(wvlIN, flxIN)
            fwhmint = CubicSpline(wvlIN, DISP_INFO)
            ss = 10
            nwvl = wvlIN.shape[0]
            nwvlss = (nwvl-1)*ss+1
            wvlss = np.arange(nwvlss)*(wdiff/float(ss)) + wvlIN[0]
            spec = specint(wvlss)
            fwhm = fwhmint(wvlss)
        else:
            spec = copy.deepcopy(flxIN)
            fwhm = DISP_INFO

        sigma = np.divide(fwhm, (2*np.sqrt(2*np.log(2))))/wdiff

        p = int(np.ceil(np.max(3*sigma)))
        m = 2*p + 1
        x2 = np.linspace(-p, p, m)**2
        n = spec.size
        a = np.zeros((m, n))

        for j in range(m):
            a[j, p:-p] = spec[j:n-m+j+1]

        gau = np.exp(-x2[:, None]/(2*sigma**2))
        gau = np.divide(gau, np.sum(gau, 0)[None, :])

        if upsample:
            conv_spectrumss = np.sum(np.multiply(a, gau), 0)
            conv_spectrumint = CubicSpline(wvlss, conv_spectrumss)
            conv_spectrum = conv_spectrumint(wvlIN)
        else:
            conv_spectrum = np.sum(np.multiply(a, gau), 0)

        return conv_spectrum


class dispFile:
    '''
    Class with methods to write dispersion files. Methods output files.

    Attributes
    ----------
    saveDIR : str
        Directory to save dispersion files.
    
    '''

    def __init__(self):
        '''
        Instantiate the dispFile class.
        '''
        self.saveDIR = \
            os.path.join(os.path.abspath(q3dfit.data.__file__)[:-11],
                         'dispersion_files')
        return

    def make_custom_dispersion(self, wavelen, R=None, KMS=None, DLAMBDA=None,
                               FILENAME=None, OVERWRITE=False):
        '''
        Create dispersion file from array of R, KMS, or DLAMBDA.

        Parameters
        ----------
        wavelen : array
            Wavelength array
        R : array, optional
            Resolving power array. Default is None.
        KMS : array, optional
            Velocity array. Default is None.
        DLAMBDA : array, optional
            Delta lambda array. Default is None.
        FILENAME : str, optional
            Filename of the output dispersion file. Default is timestamped 
            custom_disp.fits.
        OVERWRITE : bool, optional
            Overwrite existing dispersion file. Default is False.
        
        Returns
        -------
        filepath : str
            Full path of the output dispersion file.
        
        Raises
        ------
        SystemExit

        '''
        c1 = fits.Column(name='WAVELENGTH', format='E', array=wavelen)
        clist = [c1]
        if R is not None:
            clist.append(fits.Column(name='R', format='E', array=R))
        elif DLAMBDA is not None:
            clist.append(fits.Column(name='DLAM', format='E', array=DLAMBDA))
        elif KMS is not None:
            clist.append(fits.Column(name='VELOCITY', format='E', array=KMS))

        cols = fits.ColDefs(clist)
        tbhdu = fits.BinTableHDU.from_columns(cols)

        if FILENAME is None:
            import time
            filename = 'custom_'+str(int(time.time()))+'_disp.fits'
        else:
            filename = FILENAME+'_disp.fits'
        filepath = os.path.join(self.saveDIR, filename)
        print('writing dispersion to: ', filepath)
        tbhdu.writeto(filepath, overwrite=OVERWRITE)

        return filepath

    def make_dispersion(self, dispValue, WAVELEN=[0.05, 100.], TYPE='R',
                        OVERWRITE=True):
        '''
        Create dispersion file with constant R, KMS, or DLAMBDA.

        Parameters
        ----------
        dispValue : float
            Dispersion value
        WAVELEN : array, optional
            Wavelength range. Default is [0.05, 100.], in micron.
        TYPE : str, optional
            Type of dispersion. Default is 'R'.
        OVERWRITE : bool, optional
            Overwrite existing dispersion file. Default is True.
        
        Returns
        -------
        filepath : str
            Full path of the output dispersion file.

        Raises
        ------
        SystemExit

        '''
        if not isinstance(dispValue, (int, float)):
            raise SystemExit("dispFile.make_dispersion: incorrect syntax. " +
                             "dispValue must be a real number.")

        yrange = dispValue
        if type(yrange) is not tuple or type(yrange) is not list:
            yrange = [yrange, yrange]

        gwvln = np.linspace(WAVELEN[0], WAVELEN[1], 10000)
        c1 = fits.Column(name='WAVELENGTH', format='E', array=gwvln)

        yy = CubicSpline(WAVELEN, yrange)
        yintp = yy(gwvln)
        clist = [c1]
        ig = TYPE.lower()
        if TYPE.upper() == 'R':
            print("R = ", dispValue)
            grsln = yintp
            clist.append(fits.Column(name='R', format='E', array=grsln))
        elif TYPE.upper() == 'DLAMBDA':
            print("dlambda [Å] = ", dispValue)
            dlambda = yintp * 1e-4   # convert from Angstrom to micron
            gdisp = dlambda
            clist.append(fits.Column(name='DLAM', format='E', array=gdisp))
        elif TYPE.upper() == 'KMS':
            print("velocity = ", dispValue, "km/s")
            vel = yintp
            clist.append(fits.Column(name='VELOCITY', format='E', array=vel))
        else:
            raise SystemExit("dispFile.make_dispersion: incorrect syntax. " +
                             "TYPE must be 'R', 'kms' , or 'dlambda'")

        cols = fits.ColDefs(clist)
        tbhdu = fits.BinTableHDU.from_columns(cols)

        filename = 'flat_'+ig+str(dispValue)+'_disp.fits'
        filepath = os.path.join(self.saveDIR, filename)
        print('Writing dispersion to: ', filename)
        tbhdu.writeto(filepath, overwrite=OVERWRITE)

        return filepath
