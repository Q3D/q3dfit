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
- Fixed typos
- Some changes to the initialization
- Simplified the MIRI and NIRSpec reading
- Fixed the wavelength matching. Now the convolution function can read in any spectrum and convolve lines according to desired JWST instrument/channel
- cleaned/slimmed the __init__() call and generalized get_dispersion_data() call. The convolution method solely depends on the dispersion file selection
- adding km/s reading implementation

"""

import numpy as np
import os
import re
from astropy.io import fits
import glob
import copy
from scipy.ndimage import gaussian_filter1d
from scipy.interpolate import interp1d
import q3dfit.data.dispersion_files

class spectConvol:
    def __init__(self,initdat,SILENCE=True):
        self.printSILENCE = SILENCE
        #self.datDIR = '../data/dispersion_files'
        self.datDIR = os.path.join(os.path.abspath(q3dfit.data.__file__)[:-11],'dispersion_files')
        if 'ws_method' not in initdat['spect_convol']:
            initdat['spect_convol']['ws_method'] = 2
        self.init_meth = initdat['spect_convol']['ws_method']
        
        # reorganize the input list of instruments
        self.init_inst = {}
        for wsi,inst in initdat['spect_convol']['ws_instrum'].items():
            self.init_inst[wsi.upper()] = {}
            for grat in inst:
                self.init_inst[wsi.upper()][grat.upper()]=None
        self.wavelength = initdat['argsreadcube']['waveunit_in']  
        
        dispfiles = [dfile.split('/')[-1] for dfile in glob.glob(os.path.join(self.datDIR,'*.fits'))]
        self.get_dispersion_data(dispfiles)
        
        return
    
    # generalized dispersion file organizer and reader
    # cycle through all specified grating selections, extract the dispersion relations, and save the relevant ones to memory
    def get_dispersion_data(self,dispfiles):
        # from q3dfit.common.make_dispersion import make_dispersion
        displist = copy.deepcopy(dispfiles)
        for inst,gratlist in self.init_inst.items():
            for igrat in gratlist:
                
                name_match = '_'.join([inst,igrat]).lower()+'_'
                matched = False
                for dfile in displist:
                    if name_match in dfile:
                        matched = True
                        # print(inst,igrat,dfile)
                        gfilepath = os.path.join(self.datDIR,dfile)
                        idispDat  = fits.open(gfilepath)[1].data
                        icols = idispDat.columns.names
                        
                        iwvln = idispDat['WAVELENGTH'] # wavelength [μm]
                        irsln,idelw = [],[]
                        if 'VELOCITY' in icols:
                            irsln = 299792/idispDat['VELOCITY']
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
                        #idisp = idispDat['DLDS']       # dispersion [μm/pixel]
                        #irsln = idispDat['R']          # resolution [λ/Δλ] unitless 
                        if len(idelw) == 0 :
                            idelw = iwvln/irsln
                        
                        displist.remove(dfile)
                        gwvln_med = np.median(iwvln)
                        wdiff = np.abs(iwvln - gwvln_med)
                        grsln_med = np.median(irsln[np.where(wdiff == min(wdiff))[0]])
                        self.init_inst[inst][igrat] = {'gwave':iwvln,#'gdisp':idisp,
                                                       'gdwvn':idelw,'gres':irsln,
                                                       'glamC':gwvln_med,'gResC':grsln_med,'gwvRng':[min(iwvln),max(iwvln)]}
                if matched == False:
                    dispvalue  = int(re.findall(r'\d+', igrat)[0])
                    convmethod = re.findall(r'[a-zA-Z]', igrat)[0] 
                    self.make_dispersion(dispvalue,TYPE=convmethod)#,OVERWRITE=True)
                    dispfiles = [dfile.split('/')[-1] for dfile in glob.glob(os.path.join(self.datDIR,'*.fits'))]
                    self.get_dispersion_data(dispfiles)
        return
    
    # now do the convolutions -- CALL THIS
    def spect_convolver(self,wvlIN,fluxIN,wvlcen):#,SILENCE=True):#,#INST=None,GRATING='PRISM/CLEAR',SILENCE=False):
        ''' 
        METHOD 0 = flat convolution by wavelength bins
        METHOD 1 = convolution by dispersion curves: loop through each pixel element)
        METHOD 2 = PPXF method (convolution by dispersion curves) - DEFAULT
        '''
        # self.printSILENCE = SILENCE
        if self.init_inst == {} or self.init_meth > 2:
            print('ERROR: select the instrument or correct method')
            return None
        
        wvnOUT,datOUT = [],[]
        found = False
        for inst,gratlist in self.init_inst.items():
            for igrat in gratlist:
                wvrng = self.init_inst[inst][igrat]['gwvRng']
                if ((wvlcen > wvrng[0])  & (wvlcen < wvrng[1])) == True:
                    found = True
                    #print('conv w/',inst,igrat,self.init_meth,wvrng[0],wvrng[1],wvlcen)
                    break
                #else:
                #    wvnOUT = []
                #    datOUT = []
        if found == False:
            return fluxIN   
        if self.printSILENCE != True:
            print(':: '+inst.upper()+' - convolution',igrat,self.init_meth)
        
        # now do the convolution 
        igwave = self.init_inst[inst][igrat]['gwave']
        #igdisp = self.init_inst[inst][igrat]['gdisp']
        igdwvn = self.init_inst[inst][igrat]['gdwvn']
        igResM = self.init_inst[inst][igrat]['gResC']
        
        w1 = np.where(wvlIN >= min(igwave))[0]
        w2 = np.where(wvlIN <= max(igwave))[0]
        w1x = np.where(wvlIN < min(igwave))[0]
        w2x = np.where(wvlIN > max(igwave))[0]
        ww = np.intersect1d(w1,w2)
        iwvIN  = wvlIN[ww]
        idatIN = fluxIN[ww]
        
        func1 = interp1d(igwave,igdwvn)#,fill_value=0)#,fill_value="extrapolate")
        igdisp2 = func1(iwvIN)
        
        iR_datconv = np.zeros(len(ww))
        if self.init_meth == 0:
            #print('method 0')
            iR_datconv = self.flat_convolve(iwvIN,idatIN,igResM,WCEN=None)
        elif self.init_meth == 1: # needs debugging
            # self.gaussian_filter1d_looper(iwvIN,idatIN,igdisp)
            #print('method 1')
            iR_datconv = idatIN
        elif self.init_meth == 2:
            #print('method 2')
            iR_datconv = self.gaussian_filter1d_ppxf(iwvIN,idatIN,igdisp2)
        #wvnOUT = np.concatenate((wvlIN[w1x],iwvIN,wvlIN[w2x]))
        datOUT = np.concatenate((fluxIN[w1x],iR_datconv,fluxIN[w2x]))
        return datOUT
    
    # METHOD 0
    def flat_convolve(self,wvlIN,fluxIN,Rspec,WCEN=None):
        wdiff = wvlIN[1]-wvlIN[0]
        mw = WCEN
        if WCEN == None:
            mw = np.median(wvlIN)
        fwhm = (mw/Rspec) # km/s
        sigma =  (fwhm/2.355)/wdiff
        datconvol = gaussian_filter1d(fluxIN, sigma)
        return datconvol
    
    # METHOD 1
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
        
    # METHOD 2
    def gaussian_filter1d_ppxf(self,wvlIN,flxIN,DISP_INFO):
        spec = copy.deepcopy(flxIN)
        fwhm = DISP_INFO
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
    
    def make_dispersion(self,dispValue,WAVELEN=None,TYPE=None,OVERWRITE=True):
        saveDIR = os.path.join(os.path.abspath(q3dfit.data.__file__)[:-11],'dispersion_files')
        if self.printSILENCE != True:
            print(':: make flat dispersion file')
        xrange = WAVELEN
        if WAVELEN is None:
            xrange = [0.05,100] # wavelength in micron

        yrange = dispValue
        if type(yrange) is not tuple or type(yrange) is not list:
            yrange = [yrange,yrange]

        gwvln = np.linspace(xrange[0], xrange[1],10000)
        c1 = fits.Column(name='WAVELENGTH', format='E', array=gwvln)

        yy = interp1d(xrange, yrange)
        yintp = yy(gwvln)
        clist = [c1]
        ig = TYPE.lower()
        if TYPE == 'R' or TYPE is None :
            # default is Resolving power
            print("R = ",dispValue)
            grsln = yintp
            clist.append(fits.Column(name='R', format='E', array=grsln))
        elif TYPE == 'dlambda' :
            print("dlambda [Å] = ",dispValue)
            dlambda = yintp * 1e-4 #convert from Angstrom to micron
            gdisp = dlambda
            clist.append(fits.Column(name='DLAM', format='E', array=gdisp))
        elif TYPE == 'kms':
            print("velocity = ",dispValue,"km/s")
            vel = yintp
            clist.append(fits.Column(name='VELOCITY', format='E', array=vel))
        
        cols = fits.ColDefs(clist)
        tbhdu = fits.BinTableHDU.from_columns(cols)
        
        filename = 'flat_'+ig+str(dispValue)+'_disp.fits'
        filepath = os.path.join(saveDIR,filename)
        if  (os.path.exists(filepath) == True and OVERWRITE == False):
            pass
        if saveDIR is not None and (os.path.exists(filepath) != True or OVERWRITE != False):
            if self.printSILENCE != True:
                print('create dispersion to:',filename)
            tbhdu.writeto(filepath,overwrite=OVERWRITE)
        return 
