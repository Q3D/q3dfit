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

"""

class spectConvol:
    def __init__(self,initdat):
        self.datDIR = '../data/dispersion_files'
        self.printSILENCE = False
        if initdat['spect_convol']['ws_instrum'] is not list:
            initdat['spect_convol']['ws_instrum'] = list(initdat['spect_convol']['ws_instrum'])
        if initdat['spect_convol']['ws_grating'] is not list:
            initdat['spect_convol']['ws_grating'] = list(initdat['spect_convol']['ws_grating'])
        if 'ws_method' not in initdat['spect_convol']:
            initdat['spect_convol']['ws_method'] = 2
        self.init_meth = initdat['spect_convol']['ws_method']
        self.init_inst = [wsi.upper() for wsi in initdat['spect_convol']['ws_instrum']]
        self.init_grat = [wsg.upper() for wsg in initdat['spect_convol']['ws_grating']]
        nirspec_grating = {'PRISM/CLEAR':None,
                           'G140M/F070LP':None,'G140M/F100LP':None,'G235M/F170LP':None,'G395M/F290LP':None,
                           'G140H/F070LP':None,'G140H/F100LP':None,'G235H/F170LP':None,'G395H/F290LP':None}
        miri_grating = {'CH1A':None,'CH1B':None,'CH1C':None,
                        'CH2A':None,'CH2B':None,'CH2C':None,
                        'CH3A':None,'CH3B':None,'CH3C':None,
                        'CH4A':None,'CH4B':None,'CH4C':None}
        self.grating_info = {'NIRSPEC':nirspec_grating,'MIRI':miri_grating}
        self.jwst_dispersions()
        print('convolving!')
        return
    
    # now cycle through grating selections and extract the dispersion relations
    def jwst_dispersions(self,INST = None):
        dispfiles = [dfile.split('/')[-1] for dfile in glob.glob(os.path.join(self.datDIR,'*.fits'))]
        displist = []
        for dfile in dispfiles:
            if ('jwst' in dfile ) and ('nirspec' in dfile or 'miri' in dfile) :
                displist.append(os.path.join(self.datDIR,dfile))
        for inst in self.init_inst:
            for ig,igrat in self.grating_info[inst.upper()].items():
                if ig in self.init_grat:
                    gwvln,gdisp,grsln = self.dispersion_data(displist,ig)
                    gwvln_med = np.median(gwvln)
                    wdiff = np.abs(gwvln - gwvln_med)
                    grsln_med = np.median(grsln[np.where(wdiff == min(wdiff))[0]])
                    self.grating_info[inst.upper()][ig] = {'gwave':gwvln,'gdisp':gdisp,'gres':grsln,'glamC':gwvln_med,'gResC':grsln_med,'gwvRng':[min(gwvln),max(gwvln)]}
        return
    
    # read the FITS files
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
    def spect_convolver(self,wvlIN,fluxIN,wvlcen,SILENCE=True):#,#INST=None,GRATING='PRISM/CLEAR',SILENCE=False):
        ''' 
        METHOD 0 = flat convolution by wavelength bins
        METHOD 1 = convolution by dispersion curves: loop through each pixel element)
        METHOD 2 = PPXF method (convolution by dispersion curves) - DEFAULT
        '''
        INST   = self.init_inst
        GRATING = self.init_grat
        METHOD = self.init_meth
        
        self.printSILENCE = SILENCE
        if INST == None or METHOD > 2:
            print('ERROR: select the instrument NIRSpec or MIRI')
            return None
        
        wvnOUT,datOUT = [],[]
        for inst in INST:
            for grat in GRATING:
                if grat.upper() in self.grating_info[inst.upper()]:
                    wvrng = self.grating_info[inst.upper()][grat.upper()]['gwvRng']
                    if ((wvlcen > wvrng[0])  & (wvlcen < wvrng[1])) == True:
                        break
                    else:
                        wvnOUT = wvlIN
                        datOUT = fluxIN
            
        if self.printSILENCE != True:
            print(':: '+inst.upper()+' - convolution',grat,METHOD)
            
        # now do the convolution 
        igrat = self.grating_info[inst.upper()][grat]
        igwave = igrat['gwave']
        igdisp = igrat['gdisp']
        # igresp = igrat['gres']
        # igwvnM = igrat['glamC']
        igResM = igrat['gResC']
        
        w1 = np.where(wvlIN >= min(igwave))[0]
        w2 = np.where(wvlIN <= max(igwave))[0]
        w1x = np.where(wvlIN < min(igwave))[0]
        w2x = np.where(wvlIN > max(igwave))[0]
        ww = np.intersect1d(w1,w2)
        iwvIN  = wvlIN[ww]
        idatIN = fluxIN[ww]
        
        func1 = interp1d(igwave,igdisp)#,fill_value=0)#,fill_value="extrapolate")
        igdisp = func1(iwvIN)
        
        if METHOD == 0:
            iR_datconv = self.flat_convolve(iwvIN,idatIN,igResM,WCEN=None)
        elif METHOD == 1: # needs debugging
            # self.gaussian_filter1d_looper(iwvIN,idatIN,igdisp)
            iR_datconv = idatIN
        elif METHOD == 2:
            iR_datconv = self.gaussian_filter1d_ppxf(iwvIN,idatIN,[igdisp])
        wvnOUT = np.concatenate((wvlIN[w1x],iwvIN,wvlIN[w2x]))
        datOUT = np.concatenate((wvlIN[w1x],iR_datconv,wvlIN[w2x]))
        return wvnOUT,datOUT
    
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
