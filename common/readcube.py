import numpy as np
from matplotlib import pyplot as plt
from astropy.io import fits
import pdb
from astropy import units as u
from astropy.io.fits import update

from scipy import constants as ct
import copy
from scipy import interpolate
import os

import warnings


'''
CUBE: the main class of ifs data cube

Extract data and information from an IFS data cube FITS file.

:Categories:
   IFSFIT

:Usage:
from readcube import *
cube = CUBE(fp='/jwst1/lwz/KCWI_dwarf/pg1411/PG1411/',infile='pg1411rb3.fits')

:Returns:
    python class CUBE
    attributes:
        phu: primary fits extension
        dat: data array
        var: variance array
        err: error array, the values are recorded as nan when the variances are negative.
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
    vormap: in, optional, 2D array for the voronoi binning map
; :Author:
;    David S. N. Rupke::
;      Rhodes College
;      Department of Physics
;      2000 N. Parkway
;      Memphis, TN 38104
;      drupke@gmail.com
;
; :History:
;    ChangeHistory::
;      2010jun08, DSNR, created as GMOS_READCUBE
;      2013dec17, DSNR, ported to IFSF_READCUBE
;      2014jan29, DSNR, added ability to change default extensions
;      2014aug05, DSNR, small tweak to allow single spectra and no DQ
;      2015may20, DSNR, updated logic in reading # of rows, cols, and wavelength
;                       points to be more flexible; added wavedim to output
;                       structure
;      2016sep12, DSNR, fixed DATEXT logic so can specify an extension of 0;
;                       added warning when correct wavelength keywords not found;
;                       added second dispersion keyword option.
;      2018feb08, DSNR, added WAVEEXT, INVVAR, and ZERODQ keywords
;      2018feb23, DSNR, added LINEARIZE keyword
;      2018aug12, DSNR, ensure values of DATEXT, VAREXT, DQEXT don't get changed
;      2020may05, DSNR, new treatment of default axes in 2D images; added CUNIT
;                       and BUNIT to output
;      2020may31, Weizhe, translated into python 3
;
; :Copyright:
;    Copyright (C) 2013--2020 David S. N. Rupke
;
;    This program is free software: you can redistribute it and/or
;    modify it under the terms of the GNU General Public License as
;    published by the Free Software Foundation, either version 3 of
;    the License or any later version.
;
;    This program is distributed in the hope that it will be useful,
;    but WITHOUT ANY WARRANTY; without even the implied warranty of
;    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU
;    General Public License for more details.
;
;    You should have received a copy of the GNU General Public License
;    along with this program.  If not, see
;    http://www.gnu.org/licenses/.
;

'''

class CUBE:
    def __init__(self,**kwargs):
        warnings.filterwarnings("ignore")
        fp = kwargs.get('fp','')
        self.fp = fp
        self.cspeed = 299792.458
        infile=kwargs.get('infile','')
        self.infile = infile
        try:
            os.path.isfile(fp+infile)
            #hdu = fits.open(fp+infile,ignore_missing_end=True)
            #hdu.info()
        except:
            print(infile+' does not exist!')
        hdu = fits.open(fp+infile,ignore_missing_end=True)
        #hdu.info()
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
        vormap = kwargs.get('vormap',None)
        self.datext = datext
        self.varext = varext
        self.dqext = dqext
        self.hdu = hdu
        self.phu = hdu[0]
        try:
            self.dat = (hdu[datext].data).T
        except:
            print('data extension does not exist')
        try:
            self.var = (hdu[varext].data).T
            self.err = copy.copy(self.var) ** 0.5
            badvar = np.where(self.var < 0)
            if np.size(badvar) > 0:
                print('CUBE: Negative values encountered in variance array.')
        except:
            print('variance extension does not exist')
        try:
            self.dq = (hdu[dqext].data).T
        except:
            print('quality flag extension does not exist')
        if zerodq == True:
            self.dq = np.zeros(np.shape(self.data))
        if waveext:
            self.wave = hdu[waveext].data
            self.crval = 0
            self.crpix = 1
            self.cdelt = 1
        self.header_phu = hdu[0].header
        self.header_dat = hdu[datext].header
        self.header_var = hdu[varext].header
        self.header_dq = hdu[dqext].header

        datashape = np.shape(self.dat)
        if np.size(datashape) == 3:
            ncols = (datashape)[0]
            nrows = (datashape)[1]
            nw = (datashape)[2]
            try:
                np.max([nrows,ncols]) < nw
            except:
                print('data cube dimensions not in [nw,nrows,ncols] format')
            CDELT = 'CDELT3'
            CRVAL ='CRVAL3'
            CRPIX = 'CRPIX3'
            CDELT = 'CDELT3'
            CD = 'CD3_3'
            CUNIT = 'CUNIT3'
            BUNIT = 'BUNIT'
        elif np.size(datashape) == 2:
            print('READCUBE: Reading 2D image. Assuming dispersion direction is along rows')
            nrows = 1
            ncols = (datashape)[1]
            nw = (datashape)[-1]
            CDELT = 'CDELT1'
            CRVAL ='CRVAL1'
            CRPIX = 'CRPIX1'
            CDELT = 'CDELT1'
            CD = 'CD1_1'
        elif np.size(datashape) == 1:
            nrows = 1
            ncols = 1
            nw = (datashape)[0]
            CDELT = 'CDELT1'
            CRVAL ='CRVAL1'
            CRPIX = 'CRPIX1'
            CDELT = 'CDELT1'
            CD = 'CD1_1'
        self.ncols = int(ncols)
        self.nrows = int(nrows)
        self.nw = int(nw)
        # obtain the wavelenghth array using header
        if not waveext:
            header = copy.copy(self.header_dat)
            if CDELT in header:
                self.wav0 = header[CRVAL] - (header[CRPIX] - 1) * header[CDELT]
                self.wave = self.wav0 + np.arange(nw)*header[CDELT]
                self.cdelt = header[CDELT]
            if CD in header:
                self.wav0 = header[CRVAL] - (header[CRPIX] - 1) * header[CD]
                self.wave = self.wav0 + np.arange(nw)*header[CD]
                self.cdelt = header[CD]
        self.crval = header[CRVAL]
        self.crpix = header[CRPIX]
        if CUNIT in header:
            self.cunit = header[CUNIT]
        BUNIT = 'BUNIT'
        if BUNIT in header:
            self.bunit = header[BUNIT]
        if vormap:
            ncols = np.max(vormap)
            nrows = 1
            vordat = np.zeros((ncols,nrows,nz))
            vorvar = np.zeros((ncols,nrows,nz))
            vordq = np.zeros((ncols,nrows,nz))
            vorcoords = np.zeros((ncols,2),dtype=int)
            nvor = np.zeros((ncols))
            for i in np.arange(ncols):
                ivor = np.where(vormap == i+1)
                xyvor = [ivor[0][0],ivor[0][1]]
                vordat[i,0,:] = dat[xyvor[0],xyvor[1],:]
                vorvar[i,0,:] = var[xyvor[0],xyvor[1],:]
                vordq[i,0,:] = dq[xyvor[0],xyvor[1],:]
                vorcoords[i,:] = xyvor
                nvor[i] = (np.shape(ivor))[1]
            dat = vordat
            var = vorvar
            dq = vordq

        if linearize:
            waveold = copy.copy(self.wave)
            datold = copy.copy(self.dat)
            varold = copy.copy(self.var)
            dqold = copy.copy(self.dq)
            self.crpix = 1
            self.cdelt = (waveold[-1]-waveold[0]) / (self.nz-1)
            wave = np.linspace(aveold[0],waveold[-1],num=self.nz)
            self.wave = wave
            spldat = interpolate.splrep(waveold,datold,s=0)
            self.dat = interpolate.splev(waveold,spldat,der=0)
            splvar = interpolate.splrep(waveold,varold,s=0)
            self.var = interpolate.splev(waveold,splvar,der=0)
            spldq = interpolate.splrep(waveold,dqold,s=0)
            self.dq = interpolate.splev(waveold,spldq,der=0)
            print('READCUBE: Interpolating DQ; values > 0.01 set to 1.')
            ibd = np.where(self.dq > 0.01)
            if np.size(ibd) > 0:
                dq[ibd] = 1
        hdu.close()
if __name__ == "__main__":
    #c = constants.c/1000.
    #main(J0906=True)
    cube=CUBE(fp='/jwst1/lwz/KCWI_dwarf/pg1411/PG1411/',infile='pg1411rb3.fits')
