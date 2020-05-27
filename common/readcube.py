import numpy as np
import scipy.integrate as integrate
from matplotlib import pyplot as plt
from astropy.io import fits
import pdb
#from specutils.wcs import specwcs
from astropy import units as u
from astropy.io.fits import update
from astropy.cosmology import WMAP9 as cosmo
from astropy.convolution import convolve
from astropy.modeling import models


#from plotcos_general import movingaverage

from spectres import spectres

from scipy.io import readsav
from scipy import constants as ct
import copy
from pathlib import Path
import fnmatch
import os



'''
CUBE: the main class of ifs data cube

Extract data and information from an IFS data cube FITS file.

:Categories:
   IFSFIT

:Returns:
    python class CUBE
    attributes:
        phu: primary fits extension
        dat: data array
        var: variance array
        err: error array
        dq: dq array
        wave: wavelength array
        header_phu: header for the primary fits extension
        header_dat: eader for the data extension
        header_var: header for the variance extension
        header_dq: header for the dq extension
        ncols: number of columns in the cube
        nrows: number of rows in the cube
        nz: number of elements in wavelength array
:Params:
    infile: in, required, type=string
     Name of input FITS file.
:Keywords:
    fp: in, required, type=string
      Path of input FITS file.
    datext: in, optional, type=integer, default=1
      Extension # of data plane. Set to a negative number if the correct
      extension is 0, since an extension of 0 ignores the keyword.
    dqext: in, optional, type=integer, default=3
      Extension # of DQ plane. Set to a negative number if there is no DQ; DQ
      plane is then set to 0.
    error: in, optional, type=byte, default=False
      If the data cube contains errors rather than variance, the routine will
      convert to variance.
    invvar: in, optional, type=byte
      Set if the data cube holds inverse variance instead of variance. The
      output structure will still contain the variance.
    linearize: in, optional, type=byte, default=False
      If set, resample the input wavelength scale so it is linearized.
    quiet: in, optional, type=byte
      Suppress progress messages.
    varext: in, optional, type=integer, default=2
      Extension # of variance plane.
    waveext: in, optional, type=integer
      The extention number of a wavelength array.
    zerodq: in, optional, type=byte
      Zero out the DQ array.
'''

class CUBE:
    def __init__(self,**kwargs):
        fp = kwargs.get('fp','')
        self.resultfp = resultfp
        self.fp = fp
        self.cspeed = 299792.458
        infile=kwargs.get('infile','')
        self.infile = infile
        try os.path.isfile(infile)
            hdu = fits.open(fp+infile,ignore_missing_end=True)
            hdu.info()
        except:
            print(infile+' does not exist!')
        # fits extensions to be read 
        datext = kwargs.get('datext',1)
        varext = kwargs.get('varext',2)
        dqext =  kwargs.get('dqext',3)
        error = kwargs.get('error',False)
        invvar = kwargs.get('invvar',False)
        linearize = kwargs.get('linearize',False)
        quiet = kwargs.get('quiet',True)
        waveext = kwargs.get('waveext',None)
        zerodq = kwargs.get('zerodq',False)
        self.datext = datext
        self.varext = varext
        self.dqext = dqext
        self.hdu = hdu
        self.phu = hdu[0]
        self.dat = hdu[datext].data
        self.var = hdu[varext].data
        self.err = (hdu[varext].data) ** 0.5
        self.dq = hdu[dqext].data
        self.header_phu = hdu[0].header
        self.header_dat = hdu[datext].header
        self.header_var = hdu[varext].header
        self.header_dq = hdu[dqext].header 


        if np.size(np.shape(self.dat)) == 3:
            nrows = (np.shape(self.dat))[2]
            ncols = (np.shape(self.dat))[1]
            nw = (np.shape(self.dat))[0]
            self.nx = int(nx)
            self.ny = int(ny)
            self.nw = int(nw)
            # obtain the wavelenghth array
            if 'CDELT1' in header:
                self.wav0 = header['CRVAL1'] - (header['CRPIX1'] - 1) * header['CDELT1']
                self.wav = self.wav0 + np.arange(nw)*header['CDELT1']
            if 'CD1_1' in header:
                self.wav0 = header['CRVAL1'] - (header['CRPIX1'] - 1) * header['CD1_1']
                self.wav = self.wav0 + np.arange(nw)*header['CD1_1']

        

if __name__ == "__main__":
    c = constants.c/1000.
    #main(J0906=True)
    main()
