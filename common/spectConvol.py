#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""

@author: Yuzo Ishikawa

Convolves an input spectrum by the JWST NIRSPEC + MIRI grating resolution. 
By default, this will convolve the input spectrum with a wavelength dependent dispersion relation. 
The user also can choose alternative methods. 
METHOD 0 = flat convolution by wavelength bins (takes median resolving power at middle of wavelength bin)
METHOD 1 = convolution by dispersion curves: loop through each pixel element)
METHOD 2 = PPXF dispersion curve convolution (convolution by dispersion curves) - DEFAULT  

How to run this :
import spectConvol
spConv = spectConvol.spectConvol() # --> must create an spectConvol object that will get passed through the q3dfit pipeline
spectOUT = spConv.gauss_convolve(waveIN,fluxIN,INST='nirspec',GRATING='G140M/F070LP',METHOD=2)
==== 
                       
- Will require a wrapper that calls gauss_convolve() and organizes the convolved spectra
- Will need to set the paths to the dispersion files correctly

JWST provides NIRSpec dispersion files. By default, the dispersion data files are used to set the resolution
MIRI is not provided. Instead, I take a linear interpolation based on the curves in the Jdocs website. 

EDIT: 
- fixed typos
- some changes to the initialization

"""

import numpy as np
import os
from astropy.io import fits
import glob
import copy
from scipy.ndimage import gaussian_filter1d
from scipy.interpolate import interp1d

class spectConvol:
    
    def __init__(self,initdat):
        self.datDIR = '../data/dispersion_files'
        self.printSILENCE = False
        self.init_inst = [wsi.upper() for wsi in initdat['spect_convol']['ws_instrum']]
        self.init_grat = [wsg.upper() for wsg in initdat['spect_convol']['ws_grating']]
        nirspec_grating = {'PRISM/CLEAR':None,
                           'G140M/F070LP':None,'G140M/F100LP':None,'G235M/F170LP':None,'G395M/F290LP':None,
                           'G140H/F070LP':None,'G140H/F100LP':None,'G235H/F170LP':None,'G395H/F290LP':None}
        miri_grating = {'Ch1A':[(4.88,5.75),(3320,3710)],'Ch1B':[(5.63,6.63),(3190,3750)],'Ch1C':[(6.41,7.52),(3100,3610)],
                        'Ch2A':[(7.48,8.76),(2990,3110)],'Ch2B':[(8.71,10.23),(2750,3170)],'Ch2C':[(10.02,11.75),(2860,3300)],
                        'Ch3A':[(11.52,13.49),(2530,2880)],'Ch3B':[(13.36,15.65),(1790,2640)],'Ch3C':[(15.43,18.08),(1980,2790)],
                        'Ch4A':[(17.65,20.94),(1460,1930)],'Ch4B':[(20.41,24.22),(1680,1770)],'Ch4C':[(23.88,28.34),(1630,1330)]}
        self.grating_info = {'NIRSPEC':nirspec_grating,'MIRI':miri_grating}

        if 'NIRSPEC' in self.init_inst:
            self.nirspec_dispersion()
        elif 'MIRI' in self.init_inst:
            self.miri_dispersion()
        return
    
    # now cycle through grating selections and extract the dispersion relations
    def nirspec_dispersion(self):
        print(':: NIRSpec - extracting dispersion relations')
        displist = self.get_dispersion(NIRSPEC=True)
        for ig,igrat in self.grating_info['NIRSPEC'].items():
            if ig in self.init_grat:
                gwvln,gdisp,grsln = self.dispersion_data(displist,ig)
                gwvln_med = np.median(gwvln)
                wdiff = np.abs(gwvln - gwvln_med)
                grsln_med = np.median(grsln[np.where(wdiff == min(wdiff))[0]])
                self.grating_info['NIRSPEC'][ig] = {'gwave':gwvln,'gdisp':gdisp,'gres':grsln,'glamC':gwvln_med,'gResC':grsln_med}
        return
    
    def miri_dispersion(self):#,DISP_INFO,WSTEP=0.01,NWVLN=100):
        print(':: MIRI - extracting dispersion relations')
        for ig,igrat in self.grating_info['MIRI'].items():
            if ig in self.init_grat:
                gwvln = np.linspace(igrat[0][0], igrat[0][1],500)
                yy = interp1d(igrat[0], igrat[1])
                grsln = yy(gwvln)
                gdisp = gwvln/grsln
                gwvln_med = np.median(gwvln)
                wdiff = np.abs(gwvln - gwvln_med)
                grsln_med = np.median(grsln[np.where(wdiff == min(wdiff))[0]])
                add_igrat = {'gwave':gwvln,'gdisp':gdisp,'gres':grsln,'glamC':gwvln_med,'gResC':grsln_med}
                self.grating_info['MIRI'][ig].append(add_igrat)
        return 
        
    def get_dispersion(self,NIRSPEC=False,MIRI=False):
        dispfiles = [dfile.split('/')[-1] for dfile in glob.glob(os.path.join(self.datDIR,'*.fits'))]
        dispOUT = []
        for dfile in dispfiles:
            if (NIRSPEC == True and 'nirspec' in dfile) :
                dispOUT.append(os.path.join(self.datDIR,dfile))
            elif (MIRI == True and 'miri' in dfile):
                dispOUT.append(os.path.join(self.datDIR,dfile))
        return dispOUT
    
    def dispersion_data(self,filepath,gratName):
        gname = gratName.split('/')[0].lower()
        gfilepath = ''
        for fp in filepath :
            if gname in fp :
                gfilepath = fp
                break
        if gfilepath == '' :
            print('ERROR: cannot find dispersion file')
            return None
        else:
            dispDat = fits.open(gfilepath)[1].data
            wvln = dispDat['WAVELENGTH'] # wavelength [μm]
            disp = dispDat['DLDS']       # dispersion [μm/pixel]
            rsln = dispDat['R']          # resolution [λ/Δλ] unitless 
            return wvln,disp,rsln
        
    # now do the convolutions -- CALL THIS
    def gauss_convolve(self,wvlIN,fluxIN,INST=None,GRATING='PRISM/CLEAR',METHOD=2,SILENCE=False):
        ''' 
        METHOD 0 = flat convolution by wavelength bins
        METHOD 1 = convolution by dispersion curves: loop through each pixel element)
        METHOD 2 = PPXF method (convolution by dispersion curves) - DEFAULT
        '''
        self.printSILENCE = SILENCE
        # print('- Convolving by JWST grating resolution')
        # print(INST,GRATING,METHOD)
        if INST == None or METHOD > 2:
            print('ERROR: select the instrument NIRSpec or MIRI')
            return None
        elif INST.upper() == 'NIRSPEC' :
            wvnOUT,datOUT = self.do_NIRSPEC(wvlIN,fluxIN,GRATING=GRATING,METHOD=METHOD)
            return wvnOUT,datOUT
        elif INST.upper() == 'MIRI' :
            wvnOUT,datOUT = self.do_MIRI(wvlIN,fluxIN,GRATING=GRATING,METHOD=METHOD)
            return wvnOUT,datOUT
    
    def gaussian_filter1d_looper(self,wvlIN,flxIN,DISP_INFO):
        wdiff = wvlIN[1]-wvlIN[0]
        fwhm = DISP_INFO[1]
        sigma =  fwhm/(2*np.sqrt(2*np.log(2)))/wdiff
        pwvn = []
        pdat = []
        psig = []
        ww = np.where(wvlIN >= min(DISP_INFO[0]))[0]
        cwvn = wvlIN[ww]
        cflx = flxIN[ww]
        
        wpix = 5
        wi=0
        while cwvn[wi+wpix] <= max(DISP_INFO[0]):
            iwvn = cwvn[wi:wi+wpix]
            idat = cflx[wi:wi+wpix]
            isig = sigma[wi:wi+wpix]
            pwvn.append(iwvn)
            pdat.append(idat)
            psig.append(np.median(isig))
            wi+=wpix
        
        datconvol = []
        for ip in range(len(pwvn)):
            iflx = pdat[ip]
            # gg = Gaussian1DKernel(stddev=psig[ip])
            gg = gaussian_filter1d(iflx,psig[ip],mode='constant',cval=pdat[ip][0])
            # datconvol.append(convolve(iflx, gg,boundary='fill',fill_value=pdat[ip][0]))
            datconvol.append(gg)
        dcOUT = np.array(datconvol).flatten()
        # dcOUT = boxcar_5px(np.array(datconvol).flatten())
        wvOUT = np.array(pwvn).flatten()
        # print(len(dcOUT),len(wvOUT))
        # return datconvol
        return wvOUT,dcOUT
        
        
    def flat_convolve(self,wvlIN,fluxIN,Rspec,WCEN=None):
        wdiff = wvlIN[1]-wvlIN[0]
        mw = WCEN
        if WCEN == None:
            mw = np.median(wvlIN)
        fwhm = (mw/Rspec) # km/s
        sigma =  (fwhm/2.355)/wdiff
        datconvol = gaussian_filter1d(fluxIN, sigma)
        return datconvol
    
    def gaussian_filter1d_ppxf(self,wvlIN,flxIN,DISP_INFO):
        spec = copy.deepcopy(flxIN)
        fwhm = np.array(DISP_INFO[0])
        if len(DISP_INFO) != 1:
            fwhm = DISP_INFO[1]
        wdiff = wvlIN[1]-wvlIN[0]
        
        sigma =  np.divide(fwhm,2.355)/wdiff
        
        p = int(np.ceil(np.max(3*sigma)))
        m = 2*p + 1
        x2 = np.linspace(-p,p,m)**2
        n = spec.size
        a = np.zeros((m,n))
        
        for j in range(m):
            a[j,p:-p] = spec[j:n-m+j+1]
            
        gau = np.exp(-x2[:,None]/(2*sigma**2)) 
        gau = np.divide(gau,np.sum(gau,0)[None,:])
        conv_spectrum = np.sum(np.multiply(a,gau),0)
        
        return conv_spectrum
    
    
    def do_NIRSPEC(self,wvIN,datIN,GRATING='',METHOD=2):
        # print(self.printSILENCE )
        if self.printSILENCE != True:
            print(':: NIRSpec - convolution',GRATING,METHOD)
        # now do the convolution 
        igrat = self.grating_info['NIRSPEC'][GRATING]
        
        igwave = igrat['gwave']
        igdisp = igrat['gdisp']
        # igresp = igrat['gres']
        # igwvnM = igrat['glamC']
        igResM = igrat['gResC']
        
        w1 = np.where(wvIN >= min(igwave))[0]
        w2 = np.where(wvIN <= max(igwave))[0]
        ww = np.intersect1d(w1,w2)
        iwvIN  = wvIN[ww]
        idatIN = datIN[ww]
        
        func1 = interp1d(igwave,igdisp)#,fill_value=0)#,fill_value="extrapolate")
        igdisp = func1(iwvIN)
        
        if METHOD == 0:
            iR_datconv = self.flat_convolve(iwvIN,idatIN,igResM,WCEN=None)
            return iwvIN,iR_datconv
        elif METHOD == 1:
            self.gaussian_filter1d_looper(iwvIN,idatIN,igdisp)
            return
        elif METHOD == 2:
            iR_datconv = self.gaussian_filter1d_ppxf(iwvIN,idatIN,[igdisp])
            return iwvIN,iR_datconv
    
    def do_MIRI(self,wvIN,datIN,GRATING='',METHOD=2):
        if self.printSILENCE != True:
            print(':: MIRI - convolution',GRATING,METHOD)
        igrat = self.grating_info['MIRI'][GRATING]
        
        igwave = igrat[2]['gwave']
        igdisp = igrat[2]['gdisp']
        # igresp = igrat['gres']
        # igwvnM = igrat['glamC']
        igResM = igrat[2]['gResC']
        
        w1 = np.where(wvIN >= min(igwave))[0]
        w2 = np.where(wvIN <= max(igwave))[0]
        ww = np.intersect1d(w1,w2)
        iwvIN  = wvIN[ww]
        idatIN = datIN[ww]
        
        func1 = interp1d(igwave,igdisp)#,fill_value=0)#,fill_value="extrapolate")
        igdisp = func1(iwvIN)
        
        if METHOD == 0:
            iR_datconv = self.flat_convolve(iwvIN,idatIN,igResM,WCEN=None)
            return iwvIN,iR_datconv
        elif METHOD == 1:
            self.gaussian_filter1d_looper(iwvIN,idatIN,igdisp)
            return
        elif METHOD == 2:
            iR_datconv = self.gaussian_filter1d_ppxf(iwvIN,idatIN,[igdisp])
            return iwvIN,iR_datconv
    
