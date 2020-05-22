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

import matplotlib.gridspec as gridspec
from scipy.io import readsav
import copy
from pathlib import Path
import fnmatch
import os


def movingaverage_wz(values, weights, window):
    indices = np.where((weights < 0) | (np.isfinite(weights)==0))
    for ij in indices[0]:
        i = 0
        while(np.isfinite(weights[ij-i]) == False):
            i+=1
        id0 = ij-i
        i=0
        while(np.isfinite(weights[ij+i]) == False):
            i+=1
        id1 = ij+i
        values[ij] = (values[id0]+values[id1]) /2.
        print(ij,id0,id1)
    mvavg = np.convolve(values, np.ones(window)/window,'same')
    return mvavg




'''
CUBE: the main class of ifs data cube
def maskbad: mask the given wavelength range of the ifs datacube 
    maskfile: txt with the wavelength range to be masked, now only work for all spaxels, needs to be updated

def plotspec: plot spectrum of specicfic spaxel

def linecompare: Compare velocities of Ha, Hb, and OIII emission

def Keck: compare the J0842 gmos spectra with the Keck LRIS spectra

def raw_sn: calculate the snr in the raw spectra or continuum subtracted spectrum (emlonly) in the given wavelength window.

def sum: stack the spectra according to the binning maps

def binwave: spectrally bin the spectral in the wavelength space. 

def fito3: fit the o3 emission line using linefit




'''

def psf(x=np.arange(0.,3,0.01),sigma=0.3):
    y = models.Gaussian1D(mean=0.,stddev=sigma,amplitude=1)
    return x,y(x)



def wavtovel(wave0=5006.83):
    return ((wave-wave0)/wave0-z) * c

'''
Extract data and information from an IFS data cube FITS file.

:Categories:
   IFSFIT

:Returns:
   python class CUBE

:Params:
   infile: in, required, type=string
     Name and path of input FITS file.

:Keywords:
   datext: in, optional, type=integer, default=1
     Extension # of data plane. Set to a negative number if the correct
     extension is 0, since an extension of 0 ignores the keyword.
   dqext: in, optional, type=integer, default=3
     Extension # of DQ plane. Set to a negative number if there is no DQ; DQ
     plane is then set to 0.
   error: in, optional, type=byte, default=0
     If the data cube contains errors rather than variance, the routine will
     convert to variance.
   header: out, optional, type=structure
     Headers for each extension.
   invvar: in, optional, type=byte
     Set if the data cube holds inverse variance instead of variance. The
     output structure will still contain the variance.
   linearize: in, optional, type=byte
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
    def __init__(self,coadd=None,z=0.0466,loopfit=True,emlonly=False,contcube=True,emlonlyout=False,writeqsotemp=False,kcwicube=False,**kwargs):
        fp = kwargs.get('fp','/jwst/lwz/geminiFT/J0906/cube/')
        resultfp = kwargs.get('resultfp','/jwst/lwz/geminiFT/J0906/fitsd02/outputs/')
        self.resultfp = resultfp
        contxdr = kwargs.get('contxdr','3comp_allfreeJ0906.cont.xdr')
        xc = kwargs.get('xc',12)
        yc = kwargs.get('yc',17)
        self.fp = fp
        self.cspeed = 299792.458
        self.plotname = kwargs.get('plotname','J0906')
        infile=kwargs.get('infile','cstxeqxbrgN20190404S0175_3D.fits')
        self.absorb = kwargs.get('absorb',False)        
        self.infile = infile
        self.z=z
        try os.path.isfile(infile)
            hdu = fits.open(fp+infile,ignore_missing_end=True)
            hdu.info()
        except:
            print(infile+' does not exist!')
        # fits extensions to be read 
        datext = kwargs.get('datext',1)
        varext = kwargs.get('varext',2)
        dqext =  kwargs.get('dqext',3)
        if kcwicube:
            datext = 0
            varext = 1
            dqext = 2
        self.datext = datext
        self.varext = varext
        self.dqext = dqext
        self.hdu = hdu
        self.kcwicube = kcwicube
        if not kcwicube:
            self.pmu = hdu[0]
        self.dat = hdu[datext].data
        self.var = hdu[varext].data
        self.err = (hdu[varext].data) ** 0.5
        self.dq = hdu[dqext].data
        self.header = hdu[datext].header
        header =  hdu[datext].header
        self.dq_hdr = hdu[dqext].header 


        if np.shape
        nrows = (np.shape(self.dat))[2]
        ncols = (np.shape(self.dat))[1]
        nw = (np.shape(self.dat))[0]
        self.nx = int(nx)
        self.ny = int(ny)
        self.nw = int(nw)



        # obtain the wavelenghth array
        if 'CDELT3' in header:
            self.wav0 = header['CRVAL3'] - (header['CRPIX3'] - 1) * header['CDELT3']
            self.wav = self.wav0 + np.arange(nw)*header['CDELT3']
        if 'CD3_3' in header:
            self.wav0 = header['CRVAL3'] - (header['CRPIX3'] - 1) * header['CD3_3']
            self.wav = self.wav0 + np.arange(nw)*header['CD3_3']

        #write spectrum of the central spaxel as the AGN template to be used
        if writeqsotemp:
            if 'J0906' in self.plotname: 
                ind = np.where((self.wav > 5050))
            else:
                ind = np.where((self.wav > 0))
            fl = (self.dat[ind,yc,xc])[0]
            flerr = (self.err[ind,yc,xc])[0]
            

        #write emlonly fits cube
        emlonlyout=False
        if emlonlyout:
            hdu2 = copy.copy(hdu)
            outfile = fp+'rupketest_noscat_emlonly.fits'
            hdu2.writeto(outfile)
            update(outfile, self.emlonly, header, 'sci')  
        #pdb.set_trace()

    def image2d(self,w0,w1,usewav=False):
        if usewav:
            ind = np.where((self.wav > w0) & (self.wav < w1))
            #pdb.set_trace()
            w0 = ind[0][0]
            w1 = ind[0][-1]
        img = np.sum(self.dat[w0-1:w1,:,:],axis=0)
        hdu = fits.ImageHDU(img,header=self.hdu[self.datext].header)
        hdu.writeto(self.resultfp+self.plotname+'2D.fits',overwrite=True)
        #pdb.set_trace()

    def marklines(self,xc=30,yc=49):
        lines = ['[NeV]3345','[NeV]3426','[OII]3726','[OII]3729','H9','[NeIII]3869','[SII]4068','HeII4686','H8','Hepsilon','Hdelta','Hgamma','Hbeta','[NI]5198','[NI]5200','Halpha','[OIII]4363','[OI]6364','[FeX]6375','NaD2','[ArIV]4740','[CaV]5309','[FeVII]5159','[FeVII]5276','[FeVII]5721','[FeVII]6087','[FeX]6375']
        lc = [3345.83,3425.87,3726.032,3728.815,3835.397,3868.76,4068.60,4686.7,3889.064,3970.075,4101.73,4340.47,4861.35,5197.9,5200.26,6562.80,4363.209,6363.78,6374.51,5889.95,4740.12,5309.11,5158.89,5276.38,5720.7,6087.0,6374.51]
        wave0 = self.wav[:]
        flux0 = self.dat[:,yc,xc]
        id1 = int(self.nw/4)-1
        id2 = int(self.nw/4*2)-1
        id3 = int(self.nw/4*3)-1
        plt.subplot(411)
        plt.plot(wave0[0:id1],flux0[0:id1])
        for li in range(np.size(lc)):
            plt.vlines(lc[li]*(1+self.z),0,0.7,linestyles='dotted')
            plt.text(lc[li]*(1+self.z)+1,0.3,lines[li],fontsize=7)
        plt.xlim(wave0[0]-1,wave0[id1]+1)
        plt.subplot(412)
        plt.plot(wave0[id1:id2],flux0[id1:id2])
        for li in range(np.size(lc)):
            plt.vlines(lc[li]*(1+self.z),0,0.7,linestyles='dotted')
            plt.text(lc[li]*(1+self.z)+1,0.3,lines[li],fontsize=7)
        plt.xlim(wave0[id1]-1,wave0[id2]+1)
        plt.subplot(413)
        plt.plot(wave0[id2:id3],flux0[id2:id3])
        for li in range(np.size(lc)):
            plt.vlines(lc[li]*(1+self.z),0,0.7,linestyles='dotted')
            plt.text(lc[li]*(1+self.z)+1,0.3,lines[li],fontsize=7)
        plt.xlim(wave0[id2]-1,wave0[id3]+1)
        plt.subplot(414)
        plt.plot(wave0[id3::],flux0[id3::])
        for li in range(np.size(lc)):
            plt.vlines(lc[li]*(1+self.z),0,0.7,linestyles='dotted')
            plt.text(lc[li]*(1+self.z)+1,0.3,lines[li],fontsize=7)
        plt.xlim(wave0[id3]-1,wave0[-1]+1)
        plt.savefig(self.plotname+'linelist.pdf')
        plt.legend()
        plt.show()

    def maskbad(self,fitsfile,maskfile):
        wave = self.wav
        w1,w2 = np.loadtxt(maskfile,unpack=True)
        if np.size(w1) == 1:
            w1 = np.array([w1])
            w2 = np.array([w2])
        print(np.size(w1))
        
        bid = np.array([])
        for i in np.arange(np.size(w1)):
            print(bid)
            bid = np.append(np.where((wave > w1[i]) & (wave < w2[i])),bid)
        bid=np.array(bid,dtype=int)
        print(bid)
        #pdb.set_trace()
        if np.size(bid) > 0:
            (self.dq)[bid,:,:]=1
        hdu2 = copy.copy(self.hdu)
        hdu2.writeto('m'+fitsfile,overwrite=True)
        print('haggsdgs',fitsfile)
        update('m'+fitsfile, self.dq, self.dq_hdr, 'dq')

    def plotradialspec(self,scalewindow=False,xc=12,yc=17,nx=4,ny=8,xran=[4970,5020]):
        ax1 = plt.subplot(111)
        colors=['red','yellow','green','blue','magenta']
        ind = np.where((self.wav > 4980*(1+self.z)) & (self.wav < 5030*(1+self.z)))
        p0 = np.max(self.dat[ind,yc,xc])
        ax1.plot(self.wav,self.dat[:,yc,xc],'k--',label='center',linewidth=3)
        for i in range(nx):
            x = int(xc - nx +i)
            if x != xc:
                label=str(x)
                #pdb.set_trace()
                
                if scalewindow:
                    sid = np.where((self.wav < scalewindow[1]) & (self.wav > scalewindow[0]))
                    scale = np.median(self.dat[sid,yc,xc])/np.median(self.dat[sid,yc,x])
                    print(scale)
                else:
                    scale = p0/np.max(self.dat[ind,yc,x])
                ax1.plot(self.wav,scale*self.dat[:,yc,x],label=label,c=colors[int(abs(x-xc))-1])
        plt.legend()
        plt.xlim(xran[0]*(1+self.z),xran[1]*(1+self.z))
        plt.show()
        plt.savefig(self.plotname+'radialemlonlyspec'+'.pdf')

    def plotsimple(self,x=12,y=17,scale=1.,label=''):
        x = x-1
        y = y-1
        plt.plot(self.wav,scale*self.dat[:,y,x],label=label)

    def plotspec(self,x=12,y=17,ax=None,label=None,zoom=(5100,5300),zoom2=(6400,6800),zoom3=(6000,6400),scale=1.):
        ax1 = plt.subplot(311)
        ax2 = plt.subplot(312)
        ax3 = plt.subplot(313)
        ax1.plot(self.wav,scale*self.dat[:,y,x],label=label)
        ax1.set_xlim(zoom)
        ax2.plot(self.wav,scale*self.dat[:,y,x],label=label)
        ax2.set_xlim(zoom2)
        ax3.plot(self.wav,scale*self.dat[:,y,x],label=label)
        ax3.set_xlim(zoom3)
        plt.savefig('rawspec_'+self.plotname+'_'+str(x)+'_'+str(y)+'.png')
        ax1 = plt.subplot(311)
        ax2 = plt.subplot(312)
        ax3 = plt.subplot(313)
        yran = (0-np.median(scale*self.dat[:,y,x]),np.median(scale*self.dat[:,y,x])*3)
        ax1.plot(self.wav,scale*self.dat[:,y,x],label=label)
        ax1.set_xlim(zoom)
        ax1.set_ylim(yran)
        ax2.plot(self.wav,scale*self.dat[:,y,x],label=label)
        ax2.set_xlim(zoom2)
        ax1.set_ylim(yran)
        ax3.plot(self.wav,scale*self.dat[:,y,x],label=label)
        ax3.set_xlim(zoom3)
        ax1.set_ylim(yran)
        plt.savefig('rawspec2_'+self.plotname+'_'+str(x)+'_'+str(y)+'.png')

    def line_compare(self,z0=0.0466,x0=23,y0=31,name='J0906'):
        c = 299792.458
        o3 = 5006.84
        o3b = 4958.91
        n2a = 6548.05
        n2b = 6583.45
        ha = 6562.80
        hb = 4861.35
        s2a = 6716.44
        s2b = 6730.82
        o1 = 6300.30
        o3b = 4958.91        

        wav = self.wav
        flux = (self.emlonly)[:,y0,x0]
        #flux = (self.dat)[:,y0,x0]
        err = (self.var)[:,y0,x0]**0.5
        vo3 = (wav/o3-1-z0) * c
        vha = (wav/ha-1-z0) * c
        vhb = (wav/hb-1-z0) * c
        vs2b = (wav/s2b-1-z0) * c
        vo3b = (wav/o3b-1-z0) * c
        vo1 = (wav/o1-1-z0) * c
        vn2b = (wav/n2b-1-z0) * c

        ido3 = np.where((wav > o3*(1+z0)-10.) & (wav < o3*(1+z0)+10.))
        idhb = np.where((wav > hb*(1+z0)-10.) & (wav < hb*(1+z0)+10.))



        '''
        compare the [OIII] and [OI] emission
        '''

        plt.figure()
        plt.plot(vo3,flux,label='[OIII]5007')
        plt.plot(-1*vo3,flux,label='inversed [OIII]5007')
        plt.legend()
        plt.figure()
        plt.plot(vo3,flux,label='[OIII]5007')
        plt.plot(vo3,(self.dat)[:,y0,x0],label='[OIII]5007')
        plt.figure()
        plt.plot(vo3,flux,label='[OIII]5007')



        #plt.plot(vo3,(self.dat)[:,y0,x0],label='[OIII]5007')
        #plt.show()
        indo3 = np.where((vo3<200) & (vo3>-200))
        indo3b = np.where((vo3b<200) & (vo3b>-200))
        indo1 = np.where((vo1<200) & (vo1>-200))
        indha = np.where((vha<200) & (vha>-200))
        inds2b = np.where((vs2b<200) & (vs2b>-200))
        indn2b = np.where((vn2b<200) & (vn2b>-200))
        indhb = np.where((vhb<200) & (vhb>-200))


        sc1 = np.max(flux[indo3]) /  np.max(flux[indo3b])
        sc2 = np.max(flux[indo3]) /  np.max(flux[indo1])
        sc3 = np.max(flux[indo3]) /  np.max(flux[indha])
        sc4 = np.max(flux[indo3]) /  np.max(flux[inds2b])
        sc5 = np.max(flux[indo3]) /  np.max(flux[indn2b])
        sc6 = np.max(flux[indo3]) /  np.max(flux[indhb])


        plt.plot(vo3b,flux*sc1,label='[OIII]4959')
        plt.plot(vo1,flux*sc2,label='[OI]6300')
        plt.xlim(-2000,2000)
        plt.legend()
        plt.savefig(name+'compareo1o3.png')
        
        plt.figure()
        plt.plot(vo3,flux,label='[OIII]5007')
        plt.plot(vha,flux*sc3,label='Halpha')
        plt.xlim(-2000,2000)
        plt.legend()
        plt.savefig(name+'comparehao3.png')
        #plt.show()
        #pdb.set_trace()

        plt.figure()
        plt.plot(vo3,flux,label='[OIII]5007')
        plt.plot(vo1,flux*sc2,label='[OI]6300')
        plt.plot(vs2b,flux*sc4,label='[SII]6730')
        plt.plot(vn2b,flux*sc5,label='[NII]6584')
        plt.xlim(-2000,2000)
        plt.ylim(-0.02,0.1)

        plt.legend()
        plt.savefig(name+'compareo1s2.png')

        plt.figure()
        plt.plot(vo1,flux*sc2,label='[OI]6300')
        plt.plot(vo3,flux,label='[OIII]5007')
        plt.plot(vha,flux*sc3,label='Halpha')
        plt.plot(vhb,flux*sc6,label='Hbeta')
        plt.xlim(-1550,1550)
        plt.ylim(-0.02,0.1)
        plt.legend()
        plt.savefig(name+'compareo1s2b.png')
        plt.show()



        plt.figure()
        plt.subplot(411)
        plt.plot(vo3,flux/np.max(flux[ido3]),label='[OIII]')
        #pdb.set_trace()
        plt.plot(vhb,flux/np.max(flux[idhb]),label='Hbeta')
        plt.ylim(0.,1.02)
        plt.legend()
        plt.xlim(-1000,1000)
        plt.subplot(412)
        plt.plot(vha,flux,label='Halpha')
        plt.xlim(-1000,1000)
        plt.legend()
        plt.subplot(413)
        plt.plot(vhb,flux,label='Hbeta')
        plt.xlim(-1000,1000)
        plt.ylim(-0.01,0.03)
        plt.legend()
        plt.subplot(414)
        plt.plot(vs2b,flux,label='[SII]')
        plt.xlim(-1000,1000)
        plt.ylim(-0.01,0.02)
        
        plt.axvline(x=0,linestyle='--',color='black')
        plt.legend()
        plt.xlabel('km s$^{-1}$')
        plt.ylabel('10$^{-15}$ F$_{\lambda}$')
        plt.savefig('compareo3ha'+name+'.png')
        plt.show()



    def keck(self,x0=0,y0=0,fluxscale=25,pngtag='',swin=[5030,5040],**kwargs):
        kspec1 = kwargs.get('kspec1','/jwst/lwz/geminiFT/keck/J0842+03_lris_spec_blue.dat')
        kspec2 = kwargs.get('kspec2','/jwst/lwz/geminiFT/keck/J0842+03_lris_spec_red.dat')
        kspec = kwargs.get('kspec','/jwst/lwz/geminiFT/keck/J0842+03_spectrum.txt')
        #kspec = kwargs.get('kspec','/jwst/lwz/geminiFT/keck/J0906+56_spectrum.txt')
        w_keck1, f_keck1 = np.loadtxt(kspec1, unpack=True)
        w_keck2, f_keck2 = np.loadtxt(kspec2, unpack=True)
        w_keck, f_keck, e_keck = np.loadtxt(kspec, unpack=True)
        w0, f0, e0 = np.loadtxt('/jwst/lwz/geminiFT/J0906/fitso3d02_forcebroadwing_zoom_5007/outputs/box_2line_flex_raw2sn3bJ0906_0012_0017_lin1.prt', unpack=True)
        f0 = f0 *1e-15 * 25
        #y0=17
        #x0=12
        flux = (self.dat)[:,y0,x0]*1e-15 * fluxscale  #1''*1'' = 5*5 =25 spaxel


        '''
        Compare sensitivity (noise) in two data sets
        '''
        gind = np.where((self.wav>swin[0]) & (self.wav<swin[1]))
        gflux = np.sum(self.dat[gind,y0,x0])*1e-15*fluxscale
        gerr = (np.sum(self.var[gind,y0,x0])*1e-30*fluxscale)**0.5
        kind = np.where((w_keck>swin[0]) & (w_keck<swin[1]))
        kflux = np.sum(f_keck[kind])
        kgscale = kflux/gflux
        kerr = (np.sum(e_keck[kind]**2))**0.5
        print('window',swin)
        print('gmos: ',gerr,'scale: ',kgscale,gerr*kgscale)
        print('keck: ',kflux,kerr)
        print('S/N',kflux/kerr,gflux/gerr)
        print('std GMOS',np.std(self.dat[gind,y0,x0])*25*1e-15)
        print('std Keck',np.std(f_keck[kind]))
        pdb.set_trace()
        plt.figure()
        plt.subplot(212)
        #plt.plot(w_keck1,f_keck1,'b',label='LRIS_blue')
        #plt.plot(w_keck2,f_keck2,'r',label='LRIS_red')
        plt.plot(w_keck,f_keck,'r',label='LRIS')
        plt.plot(self.wav,flux,'g',label='GMOS')
        #plt.xlim(5000,6930)
        plt.ylim(-0.1e-15,1.5e-15)
        #plt.xlim(4800,6930)
        plt.legend()
        plt.show()
        #pdb.set_trace()
        plt.subplot(221)
        #flux = (self.dat)[:,y0,x0]*1e-15
        plt.plot(w_keck,f_keck,'r')
        #plt.plot(w_keck1,f_keck1,'b')#,label='LRIS_blue')
        plt.plot(self.wav,flux,'g')#,label='GMOS')
        plt.axvline(x=5016.4787*(1+self.z))
        plt.axvline(x=4985.5469*(1+self.z))
        plt.xlim(5220,5270)
        plt.subplot(222)
        plt.plot(w_keck,f_keck,'r')#,label='LRIS_red')
        plt.plot(self.wav,flux,'g')#,label='GMOS')
        #plt.xlim(6700,6800)
        plt.xlim(6830,6900)
        #plt.ylim(-0.1,0.8)
        plt.legend()
        plt.savefig(pngtag+'comparekeck1.png') #b means the GMOS spectra are continuum subtracted
        #plt.savefig(pngtag+'comparekeck1.png')
        #pdb.set_trace()
        plt.show()
        plt.figure()

        '''
        #for J0906 only
        plt.subplot(212)
        #plt.plot(w_keck1,f_keck1,'b',label='LRIS_blue')
        #plt.plot(w_keck2,f_keck2,'r',label='LRIS_red')
        plt.plot(w_keck,f_keck,'r',label='LRIS')
        plt.plot(w0,f0,'g',label='GMOS')
        #plt.plot(self.wav,flux,'g',label='GMOS')
        plt.xlim(5000,6930)
        plt.ylim(-0.1e-15,1.5e-15)
        #plt.xlim(4800,6930)
        plt.legend()
        plt.subplot(221)
        #flux = (self.dat)[:,y0,x0]*1e-15
        plt.plot(w_keck,f_keck,'r')
        #plt.plot(w_keck1,f_keck1,'b')#,label='LRIS_blue')
        #plt.plot(self.wav,flux,'g')#,label='GMOS')
        plt.plot(w0,f0,'g')
        plt.axvline(x=5016.4787*(1+self.z))
        plt.axvline(x=4985.5469*(1+self.z))

        plt.xlim(5220,5270)
        plt.subplot(222)
        plt.plot(w_keck,f_keck,'r')#,label='LRIS_red')
        #plt.plot(self.wav,flux,'g')#,label='GMOS')
        plt.plot(w0,f0,'g')
        #plt.xlim(6700,6800)
        plt.xlim(6830,6900)
        #plt.ylim(-0.1,0.8)
        plt.legend()
        plt.savefig(pngtag+'comparekeck1b.png') #b means the GMOS spectra are continuum subtracted
        #plt.savefig(pngtag+'comparekeck1.png')
        plt.figure()
        '''

        plt.figure()
        plt.axvline
        plt.axvline
        ind = np.where((self.wav > 5270) & (self.wav < 5300))
        plt.plot(self.wav[ind],flux[ind],'g',label='GMOS')
        ind2 = np.where((w_keck > 5270) & (w_keck < 5300))
        plt.plot(w_keck[ind2],f_keck[ind2],'r',label='LRIS')
        plt.legend()
        plt.savefig(pngtag+'comparekeck11.png')

        plt.figure()
        plt.axvline
        plt.axvline
        ind = np.where((self.wav > 5235) & (self.wav < 5265))
        plt.plot(self.wav[ind],flux[ind],'g',label='GMOS')
        ind2 = np.where((w_keck > 5235) & (w_keck < 5265))
        plt.plot(w_keck[ind2],f_keck[ind2],'r',label='LRIS')
        plt.legend()
        plt.savefig(pngtag+'comparekeck111.png')
        
        plt.figure()
        plt.plot(w_keck1,f_keck1,'b',label='LRIS_blue')
        plt.plot(w_keck2,f_keck2,'r',label='LRIS_red')
        plt.savefig(pngtag+'comparekeck2.png')
        plt.figure()
        plt.plot(self.wav,flux,'g',label='GMOS')
        plt.savefig(pngtag+'comparekeck3.png')
        plt.show()
        #plt.plot(self.dat
#c1=header['CDELT1'],unit=u.Unit(header['CUNIT1']))

        #,window2=[4997,5004],window3=[5110,5310]
    
    #def raw_flux(self,window=[4997,5004.5],output=False,plot=True,plotname='wing',emlonly=False\
    #,xr=[10,14],yr=[15,19],contwav=np.array([[4970,4982],[5025,5050]])):
    #    self.rawsn_window = np.array(window) * (1+self.z)
    #    z1 = copy.copy(1+self.z)
    #    wid = np.where((self.wav > window[0]*(1+self.z)) & (self.wav < window[1]*(1+self.z)))
    #    #sig = np.sum(self.dat[wid,:,:],axis=1)
    #    #err = (np.sum(self.var[wid,:,:],axis=1))**0.5
    #    if not emlonly:
    #        from linefit import fitcont
    #        for i in range(int(np.size(contwav)/2)):
    #            temp = np.where((wav > contwav[i,0]*z1) & (wav < contwav[i,1]*z1))
    #            if i == 0:
    #                cont_ind = temp
    #            else:
    #                cont_ind = np.append(cont_ind,temp)
    #        cont_p = fitcont(wav[cont_ind], flux[cont_ind], err[cont_ind])
    #        emlonly = flux - np.polyval(cont_p,wav)
    #    else:
    #        emlonly = copy.copy(self.emlonly)
    #    sig = np.median(emlonly[wid,:,:],axis=1)

    def raw_sn(self,window=[4997,5004.5],output=False,plot=True,plotname='wing',emlonly=False\
,xr=[10,14],yr=[15,19],contwav=np.array([[4970,4982],[5025,5050]]),raw_flux=None,useraw=False):
        if useraw:
            emlonly= True
        self.rawsn_window = np.array(window) * (1+self.z)
        z1 = copy.copy(1+self.z)
        wid = np.array([])
        for i in range(int(np.size(window)/2)):
            wid = np.append(wid,np.where((self.wav > window[2*i]*(1+self.z)) & (self.wav < window[2*i+1]*(1+self.z))))
        wid=np.array(wid,dtype=int)
        #pdb.set_trace()
        #sig = np.sum(self.dat[wid,:,:],axis=1)
        #err = (np.sum(self.var[wid,:,:],axis=1))**0.5
        if not emlonly:
            emlonly = np.zeros((self.nw,self.ny,self.nx))
            wav = copy.copy(self.wav)
            flux = copy.copy(self.dat)
            err = copy.copy(self.err)
            from linefit import fitcont
            for i in range(int(np.size(contwav)/2)):
                temp = np.where((wav > contwav[i,0]*z1) & (wav < contwav[i,1]*z1))
                if i == 0:
                    cont_ind = temp
                else:
                    cont_ind = np.append(cont_ind,temp)
            #pdb.set_trace()
            for ix in range(self.nx):
                for iy in range(self.ny):
                    cont_p = fitcont(wav[cont_ind], flux[cont_ind,iy,ix], err[cont_ind,iy,ix])
                    emlonly[:,iy,ix] = flux[:,iy,ix] - np.polyval(cont_p,wav)
        else:
            if useraw:
                emlonly = copy.copy(self.dat)
            else:
                emlonly = copy.copy(self.emlonly)
        #pdb.set_trace()
        #used to have axis=1?? why it was correct?
        #rawflux = np.sum(emlonly[wid,:,:],axis=0)
        #pdb.set_trace()
        #rawflux = rawflux[0,:,:]
        sig = np.median(emlonly[wid,:,:],axis=0)
        err = (np.median(self.var[wid,:,:],axis=0))**0.5
        sig = np.reshape(sig,(self.ny,self.nx))
        err = np.reshape(err,(self.ny,self.nx))
        sn = sig/err
        #why I used wid2 here before, change it now anyway
        #wid2 = np.where((self.wav > window[0]) & (self.wav < window[1]))
        sig2 = np.median(emlonly[wid,:,:],axis=0)
        std = np.std(emlonly[wid,:,:],axis=0)
        sig2 = np.reshape(sig,(self.ny,self.nx))
        std = np.reshape(std,(self.ny,self.nx))
        sn2 = sig2/std
        sig3 = np.max(emlonly[wid,:,:],axis=0)
        sig3 = np.reshape(sig3,(self.ny,self.nx))
        sn3 = sig3/err
        rawflux = np.sum(emlonly[wid,:,:],axis=0)
        err2 = (np.sum(self.var[wid,:,:],axis=0))**0.5
        sn4 = rawflux/err2

        #pdb.set_trace()
        #err_cont = np.reshape(np.median(self.err[wid,:,:],axis=1),(self.ny,self.nx))
        #print((window3[0]),window3[1],err_cont[yr[0]:yr[1],xr[0]:xr[1]]*1e-15,5*err_cont[yr[0]:yr[1],xr[0]:xr[1]]*1e-15)       
    
        if output:
            f=open(output+'.txt','w')
            for i in range(self.nx):
                for j in range(self.ny):
                    if np.isfinite(sn[j,i]) and sn[j,i]>0.3:
                        print(sn[j,i])
                        f=open(output+'.txt','a')
                        f.write('{} {} {} {}\n'.format(i, j, sig[j,i], err[j,i]))
                        #f.write("%d %d %f" % i,j,sn[i,j])
            f=open(output+'2.txt','w')
            for i in range(self.nx):
                for j in range(self.ny):
                    if np.isfinite(sn2[j,i]) and sn2[j,i]>2:
                        print(sn[j,i])
                        f=open(output+'2.txt','a')
                        f.write('{} {} {} {}\n'.format(i, j, sig2[j,i], std[j,i]))
                        #f.write("%d %d %f" % i,j,sn[i,j])
            f=open(output+'3.txt','w')
            for i in range(self.nx):
                for j in range(self.ny):
                    if np.isfinite(sn3[j,i]) and sn3[j,i]>0.5:
                        print(sn[j,i])
                        f=open(output+'3.txt','a')
                        f.write('{} {} {} {}\n'.format(i, j, sig3[j,i], err[j,i]))
                        #f.write("%d %d %f" % i,j,sn[i,j])
            f=open(output+'4.txt','w')
            for i in range(self.nx):
                for j in range(self.ny):
                    if np.isfinite(sn4[j,i]) and sn4[j,i]>0.5:
                        print(sn4[j,i])
                        f=open(output+'4.txt','a')
                        f.write('{} {} {} {}\n'.format(i, j,rawflux[j,i], err2[j,i]))
                        #f.write("%d %d %f" % i,j,sn[i,j])



        
        if plot:
            #raw_flux = True
            if raw_flux:
                plt.figure()
                plt.imshow(np.arcsinh(rawflux), cmap='inferno', origin='lower', interpolation='none')
                plt.colorbar()
                plt.xlabel('arcsec')
                plt.ylabel('arcsec')
                plt.title(raw_flux)
                plt.savefig(self.plotname+'_'+plotname+'raw_flux.png')
                plt.show()
            else:
                plt.figure()
                plt.imshow(np.arcsinh(sn), cmap='inferno', vmin=0.1, origin='lower', interpolation='none')
                plt.colorbar()
                plt.xlabel('arcsec')
                plt.ylabel('arcsec')
                plt.title('arcsinh(S/N)')
                plt.savefig(self.plotname+'_'+plotname+'_rawsn.png')
                plt.figure()
                plt.imshow(sn2, cmap='inferno', vmin=0.1, origin='lower', interpolation='none')
                plt.colorbar()
                plt.xlabel('arcsec')
                plt.ylabel('arcsec')
                plt.title('arcsinh(S/N)')
                plt.savefig(self.plotname+'_'+plotname+'_rawsn2.png')
                plt.figure()
                plt.imshow(np.arcsinh(sn3), cmap='inferno', vmin=0.1, origin='lower', interpolation='none')
                plt.colorbar()
                plt.xlabel('arcsec')
                plt.ylabel('arcsec')
                plt.title('arcsinh(S/N)')
                plt.savefig(self.plotname+'_'+plotname+'_rawsn3.png')
                plt.figure()
                plt.imshow(np.arcsinh(sn4), cmap='inferno', vmin=0.1, origin='lower', interpolation='none')
                plt.colorbar()
                plt.xlabel('arcsec')
                plt.ylabel('arcsec')
                plt.title('arcsinh(S/N)')
                plt.savefig(self.plotname+'_'+plotname+'_rawsn4.png')

        #pdb.set_trace()
        return sn

    def radial(self,window=[6000,7000],xc=12,yc=17,sscale=0.2,ppsf=False,label=False):
        xc0 = xc-1
        yc0 = yc-1
        
        ind = np.where((self.wav < window[1]) & (self.wav > window[0]))
        flux = (np.sum(self.dat[ind,:,:],axis=1))[0].T
        ferr = ((np.sum(self.var[ind,:,:],axis=1))**0.5)[0].T
        yy,xx = np.meshgrid(np.arange(self.ny),np.arange(self.nx))
        r = ((yy-yc0)**2+(xx-xc0)**2)**0.5 * sscale
        arckpc = cosmo.kpc_proper_per_arcmin(self.z).value/60.
        r_kpc = r * arckpc 
        if ppsf:
            xpsf,ypsf = psf(sigma=ppsf/2.35)
            plt.plot(xpsf*arckpc,ypsf,'y',label=r"Gaussian PSF, FWHM="+str(ppsf)+"''")
        ind = np.where(r_kpc < 1.5)
        if label:
            labelname = label
        else:
            labelname=str(window[0])+'-'+str(window[1])
        plt.errorbar((r_kpc[ind]).flatten(),((flux/flux[xc0,yc0])[ind]).flatten(),yerr=(ferr[ind]/flux[xc0,yc0]).flatten(),fmt='o',label=labelname)
        #plt.legend()
        #plt.show()
        #pdb.set_trace()
    
    #cos aperture size is 2.5'')
    def sumaper(self,xc=18,yc=27, bsize=13, write=False,fpcube='J0906',outname=''):
        #xc, yc are the physical, starting from 1
        fluxb = np.sum(self.dat[:,(yc-bsize-1):(yc+bsize),(xc-bsize-1):(xc+bsize)],axis=(1,2))
        varb = np.sum(self.var[:,(yc-bsize-1):(yc+bsize),(xc-bsize-1):(xc+bsize)],axis=(1,2))
        errb = varb**0.5
        dqb = self.dq[:,16,11]
        #pdb.set_trace()
        if write:
            fluxb0 = np.zeros((self.nw,2,2))
            varb0 = np.zeros((self.nw,2,2))
            dqb0 =  np.zeros((self.nw,2,2))
            fpcube = self.fp
            #if fpcube == 'J0906' or fpcube == 'J0842':
            #    fpcube = '/jwst/lwz/geminiFT/'+fpcube+'/cube/'
            #fpcube = '/jwst/lwz/geminiFT/'+fpcube+'/cube/'
            out = fpcube+'sumaper'+outname+str(bsize)+'_'+self.infile
            (self.hdu).writeto(out,overwrite=True,output_verify='ignore')
            fluxb = fluxb.reshape((self.nw,1,1))
            varb = varb.reshape((self.nw,1,1))
            dqb = dqb.reshape((self.nw,1,1))
            hdus = [fits.PrimaryHDU(fluxb,header=self.hdu[self.datext].header),fits.ImageHDU(varb,header=self.hdu[self.varext].header,),fits.ImageHDU(dqb,header=self.hdu[self.dqext].header)]
            hdul = fits.HDUList(hdus)
            hdul.writeto(out,overwrite=True,output_verify='ignore')
            #fits.update cause weird error now, don't now why
            #fits.update(out, fluxb, self.hdu[self.datext].header, self.datext) #headers are different!!
            #fits.update(out, varb, self.hdu[self.varext].header, self.varext)
            #fits.update(out, dqb, self.hdu[self.dqext].header, self.dqext)
            hduout = fits.open(out,ignore_missing_end=True)
            hduout.info()
            #pdb.set_trace()
        return fluxb,errb


    def sum_old(self,plotsnwindow=False,fpcube='J0906',**kwargs):
        '''
        The output fits has a shape of (nbins,nwavelength), and the spectra in each bin is just summed but not divided by the size of the bin.
        '''
        z1 = copy.copy(self.z+1)
        fpcube = '/jwst/lwz/geminiFT/'+fpcube+'/cube/'
        binname= kwargs.get('binname','radialbin')
        binmap = kwargs.get("binmap", '/jwst/lwz/geminiFT/J0906/pyscript/radialbin')
        sumsize = kwargs.get('sumsize',-1)
        out = fpcube+binname+'_'+self.infile
        (self.hdu).writeto(out,overwrite=True)
        if sumsize < 0:
            px,py,bins = np.loadtxt(binmap,unpack=True)
            px = px.astype(int)
            py = py.astype(int)
            bins = bins.astype(int)
            #pdb.set_tracen()
            bmax = int(np.max(bins))+1
            flux_b = np.zeros((bmax,self.nw))
            var_b = np.zeros((bmax,self.nw))
            dq_b = np.zeros((bmax,self.nw))
            #for b,ax,ax2 in zip(np.arange(bmax),axes.flat,axes2.flat):
            ifig=0
            ipanel=0
            for b in range(bmax):
                print(b)
                newid = b
                idx = np.where(bins == newid)[0]
                print(newid,px[idx])

                #default dqmask does not mask bad pixels with large negative value automatically.
                for wk in range(self.nw):
                    ctgood=0
                    for xk in range(np.size(idx)):
                        #pdb.set_trace()
                        if self.dq[wk,py[idx[xk]],px[idx[xk]]] == 0:
                            flux_b[newid,wk] = flux_b[newid,wk] + self.dat[wk,py[idx[xk]],px[idx[xk]]]
                            var_b[newid,wk] = var_b[newid,wk] + self.var[wk,py[idx[xk]],px[idx[xk]]]
                            ctgood = ctgood+1  # nspaxels at wavelength wk are added together
                    flux_b[newid,wk] = flux_b[newid,wk] / (ctgood*1.) # divided by the number of spaxels added
                #flux_b[newid,:] = np.sum(self.dat[:,py[idx],px[idx]],axis=(1))
                #var_b[newid,:] = np.sum(self.var[:,py[idx],px[idx]],axis=(1))
                #dq_b[newid,:] = np.sum(self.dq[:,py[idx],px[idx]],axis=(1))

                #pdb.set_trace()
                #plottings

                ipanel = b % 6
                if b % 6 == 0:
                    fig, axes = plt.subplots(nrows=3, ncols=2)
                    fig2, axes2 = plt.subplots(nrows=3, ncols=2)
                    ifig+=1
                axes.flat[ipanel].plot(self.wav,flux_b[newid,:])
                wid = np.where((self.wav > (5007.-80)*z1) & (self.wav < (5007.+40)*z1))
                axes2.flat[ipanel].plot(self.wav[wid],np.reshape(flux_b[newid,wid],-1),label=str(b+1))
                axes2.flat[ipanel].plot(self.wav[wid],np.reshape(var_b[newid,wid],-1)**0.5,label=str(b+1))
                #ax2.set_xlim(5100,5300)
                #ax2.set_ylim(-0.1,0.4)
                if plotsnwindow:
                    axes2.flat[ipanel].axvline(self.rawsn_window[0],ls='--',color='r')
                    axes2.flat[ipanel].axvline(self.rawsn_window[1],ls='--',color='r')
                plt.legend()
                fig.savefig(self.plotname+binname+'_'+str(ifig)+'_showspec.png')
                fig2.savefig(self.plotname+binname+'_'+str(ifig)+'_showspec_zoom.png')
            flux_b=np.reshape(flux_b.T,(self.nw,1,bmax))
            var_b=np.reshape(var_b.T,(self.nw,1,bmax))
            dq_b=np.reshape(dq_b.T,(self.nw,1,bmax))
            fits.update(out, flux_b, self.hdu[self.datext].header, self.datext) #headers are different!!
            fits.update(out, var_b, self.hdu[self.varext].header, self.varext)
            fits.update(out, dq_b, self.hdu[self.dqext].header, self.dqext)
            hduout = fits.open(out,ignore_missing_end=True)
            hduout.info()
            plt.ylim(-0.1,0.4)
            #plt.show()
            return flux_b,var_b,var_b**0.5
        else:
            flux_b = np.sum(self.dat[:,sumsize:self.ny-sumsize,sumsize:self.nx-sumsize],axis=(1,2))
            var_b = np.sum(self.var[:,sumsize:self.ny-sumsize,sumsize:self.nx-sumsize],axis=(1,2))
            err_b = (var_b**0.5)
            return flux_b,var_b,err_b
    

        # to make a binned cube the same shape as the original one
        #dimensions are totally inversed in IDL mrdfits and astropy fits io
        '''
        The output fits has a shape of (nx,ny,nwavelength), and the spectra in each bin is divided by the size of the bin (number of spaxels).
        '''
    def sum(self,x0=4,y0=4,oned=False,plotsnwindow=False,fpcube='J0906',doperpixel=True,outname='',plotspec=False,**kwargs):
        sumsize = kwargs.get('sumsize',False)
        if sumsize == False:
            z1 = copy.copy(self.z+1)
            if fpcube == 'J0906' or fpcube == 'J0842':
                fpcube = '/jwst/lwz/geminiFT/'+fpcube+'/cube/'
            binname= kwargs.get('binname','radialbin')
            binmap = kwargs.get("binmap", '/jwst/lwz/geminiFT/J0906/pyscript/radialbin')
            out = fpcube+binname+'_'+self.infile
            #pdb.set_trace()
            #(self.hdu).writeto(out,overwrite=True,output_verify='ignore')
            #pdb.set_trace()
            #px, py are the index, so they are col-1, row-1
            px,py,bins = np.loadtxt(binmap,unpack=True)
            px = px.astype(int)
            py = py.astype(int)
            bins = bins.astype(int)
            bmax = int(np.max(bins))+1
            flux_b = np.zeros((bmax,self.nw))
            var_b = np.zeros((bmax,self.nw))
            dq_b = np.zeros((bmax,self.nw))

            flux_new = np.zeros((self.nw,self.ny,self.nx))
            var_new = np.zeros((self.nw,self.ny,self.nx))
            dq_new = np.zeros((self.nw,self.ny,self.nx))
            #for b,ax,ax2 in zip(np.arange(bmax),axes.flat,axes2.flat):
            ifig=0
            ipanel=0
            for b in range(bmax):
                print(b) # b=0 is ignored and has no values since bin number starts from 1
                newid = b  
                idx = np.where(bins == newid)[0]
                print(newid,px[idx])

                #default dqmask does not mask bad pixels with large negative value automatically.
                for wk in range(self.nw):
                    ctgood=0
                    for xk in range(np.size(idx)):
                        if self.dq[wk,py[idx[xk]],px[idx[xk]]] == 0:
                            flux_b[newid,wk] = flux_b[newid,wk] + self.dat[wk,py[idx[xk]],px[idx[xk]]]
                            var_b[newid,wk] = var_b[newid,wk] + self.var[wk,py[idx[xk]],px[idx[xk]]]
                            ctgood = ctgood+1  # nspaxels at wavelength wk are added together
                    if doperpixel:
                        flux_b[newid,wk] = flux_b[newid,wk] / (ctgood*1.) # divided by the number of spaxels added
                        var_b[newid,wk] = var_b[newid,wk] / (ctgood*1.)
                if plotspec:
                    ipanel = b % 6
                    if b % 6 == 0:
                        fig, axes = plt.subplots(nrows=3, ncols=2)
                        fig2, axes2 = plt.subplots(nrows=3, ncols=2)
                        ifig+=1
                    axes.flat[ipanel].plot(self.wav,flux_b[newid,:])
                    wid = np.where((self.wav > (5007.-80)*z1) & (self.wav < (5007.+40)*z1))
                    axes2.flat[ipanel].plot(self.wav[wid],np.reshape(flux_b[newid,wid],-1),label=str(b+1))
                    axes2.flat[ipanel].plot(self.wav[wid],np.reshape(var_b[newid,wid],-1)**0.5,label=str(b+1))
                    #ax2.set_xlim(5100,5300)
                    #ax2.set_ylim(-0.1,0.4)
                    if plotsnwindow:
                        axes2.flat[ipanel].axvline(self.rawsn_window[0],ls='--',color='r')
                        axes2.flat[ipanel].axvline(self.rawsn_window[1],ls='--',color='r')
                    plt.legend()
                    fig.savefig(self.plotname+binname+'_'+str(ifig)+'_showspec.png')
                    fig2.savefig(self.plotname+binname+'_'+str(ifig)+'_showspec_zoom.png')
            for bid in range(np.size(bins)):
                flux_new[:,py[bid],px[bid]] = flux_b[bins[bid],:]
                var_new[:,py[bid],px[bid]] = var_b[bins[bid],:]
                dq_new[:,py[bid],px[bid]] = dq_b[bins[bid],:]
            if self.kcwicube == False:
                hdus = [self.pmu,fits.ImageHDU(flux_new,header=self.hdu[self.datext].header),fits.ImageHDU(var_new,header=self.hdu[self.varext].header,),fits.ImageHDU(dq_new,header=self.hdu[self.dqext].header)]
            else:           
                hdus = [fits.PrimaryHDU(flux_new,header=self.hdu[self.datext].header),fits.ImageHDU(var_new,header=self.hdu[self.varext].header,),fits.ImageHDU(dq_new,header=self.hdu[self.dqext].header)]
            hdul = fits.HDUList(hdus)
            hdul.writeto(out,overwrite=True,output_verify='ignore')
            #pdb.set_trace()
            #these update do not work on the KCWI datacubes, do not know why.....
            #fits.update(out, flux_new, self.hdu[self.datext].header, self.datext,ignore_missing_end=True,output_verify='ignore') #headers are different!!
            #fits.update(out, var_new, self.hdu[self.varext].header, self.varext,ignore_missing_end=True,output_verify='ignore')
            #fits.update(out, dq_new, self.hdu[self.dqext].header, self.dqext,ignore_missing_end=True,output_verify='ignore')
            hduout = fits.open(out,ignore_missing_end=True)
            hduout.info()
            plt.ylim(-0.1,0.4)
            #pdb.set_trace()
            return flux_b,var_b,var_b**0.5
        else:
            #pdb.set_trace()
            (self.hdu).writeto(outname,overwrite=True)
            x1 = sumsize[0]-1
            x2 = sumsize[1]-1
            y1 = sumsize[2]-1
            y2 = sumsize[3]-1
            flux_b = np.sum(self.dat[:,x1:x2,y1:y2],axis=(1,2))
            var_b = np.sum(self.var[:,x1:x2,y1:y2],axis=(1,2))
            err_b = (var_b**0.5)
            dq_b = np.sum(self.dq[:,x1:x2,y1:y2],axis=(1,2))
            fits.update(outname, np.reshape(flux_b, (self.nw,1,1)), self.hdu[self.datext].header, self.datext)
            fits.update(outname, np.reshape(var_b, (self.nw,1,1)), self.hdu[self.varext].header, self.varext)
            fits.update(outname, np.reshape(dq_b, (self.nw,1,1)), self.hdu[self.dqext].header, self.dqext)
            #flux_b = np.sum(self.dat[:,sumsize:self.ny-sumsize,sumsize:self.nx-sumsize],axis=(1,2))
            #var_b = np.sum(self.var[:,sumsize:self.ny-sumsize,sumsize:self.nx-sumsize],axis=(1,2))
            #err_b = (var_b**0.5)
            #return flux_b,var_b,err_b

    def binwave(self, binfactor=int(4)):
        nw = copy.copy(self.nw)
        nw_2 = int(nw) // binfactor
        flux = self.dat[0:nw_2*binfactor]
        var = self.var[0:nw_2*binfactor]
        dq = self.dq[0:nw_2*binfactor]
        flux_new = flux[0::binfactor,:,:]
        var_new = var[0::binfactor,:,:]
        dq_new = dq[0::binfactor,:,:]
        for i in range(binfactor-1):
            flux_new = flux_new + flux[i+1::binfactor,:,:]
            var_new = var_new + var[i+1::binfactor,:,:]
            dq_new = dq_new + dq[i+1::binfactor,:,:]
        flux_new = flux_new / binfactor
        var_new = var_new / binfactor
        err_new = var_new ** 0.5
        out = 'wavebin'+str(binfactor)+'_'+self.infile       
        (self.hdu).writeto(out,overwrite=True)
        header1 = self.hdu[self.datext].header
        header2 = self.hdu[self.varext].header
        header3 = self.hdu[self.dqext].header
        header1['CDELT3'] = header1['CDELT3'] *binfactor
        header2['CDELT3'] = header2['CDELT3'] *binfactor
        header3['CDELT3'] = header3['CDELT3'] *binfactor
        header1['CD3_3'] = header1['CD3_3'] *binfactor  # this keyword is used in ifsf_readcube, not CDELT3
        header2['CD3_3'] = header2['CD3_3'] *binfactor
        header3['CD3_3'] = header3['CD3_3'] *binfactor
        

        fits.update(out, flux_new, header1, 1)
        fits.update(out, var_new, header2, 2)
        fits.update(out, dq_new, header3, 3)
        hduout = fits.open(out,ignore_missing_end=True)
        hduout.info()
        pdb.set_trace()
        

    def fito3(self,contwav=np.array([[5128,5136],[5350,5420]]),emlwav=np.array([[5163,5278]]),x=12,y=17,fitline=True):
        from linefit import fito3
        from linefit import fitcont
        wav = (copy.copy(self.wav))
        flux = (copy.copy(self.dat))[:,y,x]
        err = (copy.copy(self.err))[:,y,x]
        for i in range(int(np.size(contwav)/2)):
            #pdb.set_trace()
            temp = np.where((wav > contwav[i,0]) & (wav < contwav[i,1]))
            if i == 0:
                cont_ind = temp
            else:
                cont_ind = np.append(cont_ind,temp)
        for i in range(int(np.size(emlwav)/2)):
            temp2 = np.where((wav > emlwav[i,0]) & (wav < emlwav[i,1]))
            if i == 0:
                eml_ind = temp2
            else:
                eml_ind = np.append(eml_ind,temp2)
        #pdb.set_trace()
        scale = 100.
        cont_p = fitcont(wav[cont_ind], flux[cont_ind], err[cont_ind])
        emission_only = flux - np.polyval(cont_p,wav)
        #plt.plot(wav[cont_ind],cont_p[0]*wav[cont_ind]+cont_p[1])
        #plt.plot(wav,flux)
        #plt.show()
        if fitline:
            flagout, singleo3fit, o3fit, o32fit, non_param, o3flux = fito3(wav[eml_ind], emission_only[eml_ind]*scale, err[eml_ind]*scale, self.z)
            #if plot:
            #    plt.plot(wav[eml_ind], emission_only[eml_ind]*scale)
            return flagout, singleo3fit, o3fit, o32fit, non_param, o3flux
        else:
            return cont_p
        #plt.show()
        #pdb.set_trace()
    #def plot(self):
    #    plt.plot(wav,flux_s)
    #    plt.plot(wav,err_s)
    #    
    #    plt.xlim(5570,5584)
    #    plt.ylim(0,16)
    #    plt.show()


    def flux_v(self, wave0=5006.83, vmin=-1500, vmax = -500, useraw=True,eps=0.1, plot=True, x=11, y=16):
        if useraw:
            wave = copy.copy(self.wav)
            #vel = ((wave-wave0)/wave0-z) * c
            c = self.cspeed
            tmp = wave/wave0 - self.z
            vel = (tmp**2-1)/(tmp**2+1) *c
            self.vel = copy.copy(vel)
            flux0 = (self.dat)[:,y,x]
            rescale = np.median(flux0)
            flux = flux0 / rescale
            var0 = (self.var)[:,y,x]
            var = var0 / rescale**2
            contipar = self.fito3(fitline=False,x=x,y=y)
            continuum0 = np.polyval(contipar,wave)
            #contipar = copy.copy(self.contipar)
            #contipar_err = copy.copy(self.contipar_err)
            #renorm = copy.copy(self.renorm)
            #contipar[1] = contipar[1]/rescale
            #contipar_err[1] = contipar_err[1]/rescale #normalized
            #continuum = contipar[1]*pow(wave,contipar[0])
            if self.absorb:
                flux_nocon0 = continuum0 - flux0
            else:
                flux_nocon0 = flux0 - continuum0

            
            vind = np.where((vel <= vmax) & (vel >= vmin))
            flx_v0 = integrate.simps(flux_nocon0[vind],wave[vind])
            flx_v_err0 = (integrate.simps(var0[vind],wave[vind]))**0.5
            #flx_v_err02 = (np.median(var0[vind]) * (wave[vind[0][-1]]-wave[vind[0][0]])) **0.5
            #pdb.set_trace()
            return flx_v0,flx_v_err0

    def nonparam(self, wave0=5006.83, vthres = 0.2, useraw=True,eps=0.1, plot=True, x=11, y=16):
        print('wolegequ  ',self.contipar)
        if useraw:
            wave = copy.copy(self.wave)
            vel = ((wave-wave0)/wave0-z) * c
            self.vel = vel
            flux0 = (self.dat)[:,y,x]
            rescale = np.median(flux0)
            flux = flux0 / rescale
            err0 = (self.err)[:,y,x]
            err = err0 / rescale
            contipar = copy.copy(self.contipar)
            contipar_err = copy.copy(self.contipar_err)
            renorm = copy.copy(self.renorm)
            contipar[1] = contipar[1]/rescale
            contipar_err[1] = contipar_err[1]/rescale #normalized
            continuum = contipar[1]*pow(wave,contipar[0])
            self.continuum = continuum*rescale
            flux_nocon = flux - continuum
            if self.absorb:
                flux_nocon = continuum - flux
            flux_t = integrate.simps(flux_nocon,wave)
            n_wave = np.size(wave)
            dv1 = vel[1] - vel[0]
            flux_tmp_err2 = 0. #flux error squared
            flux_tmp_con_err2 = 0.
            for i in np.arange(n_wave-2)+1:
                flux_tmp = integrate.simps(flux_nocon[:i],wave[:i])
                #add up the errors for each flux bin (d_wave*err)
                #err2 means it is error squared
                flux_tmp_err2 = flux_tmp_err2 + (err[i]*(wave[i+1] - wave[i]))**2
                #errors consider the uncertainties in continuum
                con_err2 = (pow(wave[i],contipar[0])*contipar_err[1])**2 + (contipar[0]*contipar[1]*pow(wave[i],contipar[0]-1.)*contipar_err[0])**2
                flux_tmp_con_err2 = flux_tmp_con_err2 + con_err2*(wave[i+1] - wave[i])**2
                test = flux_tmp/flux_t - vthres
                print(flux_tmp, test,' !!!!?????')
                if abs(test) < eps:
                    flux_tmp_2 = integrate.simps(flux_nocon[:i+1],wave[:i+1])
                    test2 = flux_tmp_2/flux_t - vthres
                    print(test2,test,'dasdadasd')
                    if test2*test < 0:

                        if abs(test2) < abs(test):
                            v_out = vel[i+1]
                            flux_out = flux_tmp_2
                            flux_tmp_err2 = flux_tmp_err2+(err[i+1]*(wave[i+2] - wave[i+1]))**2
                            con_err2 = (pow(wave[i+1],contipar[0])*contipar_err[1])**2 + (contipar[0]*contipar[1]*pow(wave[i+1],contipar[0]-1.)*contipar_err[0])**2
                            flux_tmp_con_err2 = flux_tmp_con_err2 + con_err2*(wave[i+2] - wave[i+1])**2
                            print(flux_tmp_err2,' ?????')
                            j = i+2
                            j_con = i+2
                        else:
                            v_out = vel[i]
                            flux_out = flux_tmp
                            flux_tmp_err2 = flux_tmp_err2+(err[i]*(wave[i+1] - wave[i]))**2
                            con_err2 = (pow(wave[i],contipar[0])*contipar_err[1])**2 + (contipar[0]*contipar[1]*pow(wave[i],contipar[0]-1.)*contipar_err[0])**2
                            flux_tmp_con_err2 = flux_tmp_con_err2 + con_err2*(wave[i+1] - wave[i])**2
                            print(flux_tmp_err2,' !!!!')
                            j = i+1
                            j_con = i+1
                        flux_tmp_err = flux_tmp_err2**0.5
                        flux_tmp_con_err = flux_tmp_con_err2**0.5
                        #The error in v_out is the velocity difference between flux_tmp & flux_tmp+flux_tmp_err
                        while(j < n_wave-1):
                            flux_tmp_uperror = integrate.simps(flux_nocon[:j],wave[:j])
                            print(n_wave,j,flux_tmp_uperror,flux_tmp,flux_tmp_err)
                            if abs(flux_tmp_uperror-flux_tmp) > flux_tmp_err:
                                vout_err = abs(vel[j] - v_out)
                                break
                            j = j+1
                            print(j)
                        # or considering the error in the fitted continuum: vout_con_err
                        if j == n_wave-1:
                            vout_err = abs(vel[-1] - v_out)
                        while(j_con < n_wave-1):
                            flux_tmp_con_uperror = integrate.simps(flux_nocon[:j_con],wave[:j_con])
                            if abs(flux_tmp_con_uperror-flux_tmp) > flux_tmp_con_err:
                                vout_con_err = abs(vel[j_con] - v_out)
                                #if vthres == 0.5:
                                #if vthres == 0.5:
                                #    pdb.set_trace()
                                break
                            j_con = j_con+1
                        #the maximum error is the end velocity - vout, when error is too large
                        if j_con == n_wave-1:
                            vout_con_err = abs(vel[-1] - v_out)
                        #vout_err = abs(vel[i+1]-vel[i])
                        #why flux_err and vout_err so small sometimes
                        if vout_err <= 0:
                            vout_err = vel[j]-vel[j-1]
                        if vout_con_err <= 0:
                            vout_con_err = vel[j]-vel[j-1]
                        print('!!!!!!')
                        #return v_out, vout_err, vout_con_err, flux_out, flux_tmp_err, flux_tmp_con_err
                        #plot the regions for the 
                        return v_out, vout_err, vout_con_err, flux_out*renorm, flux_tmp_err*renorm, flux_tmp_con_err*renorm



def mask(fname = 'J0842d02_noscat.fits',maskfile = 'J0842cube_badmask'):
    fp = '/jwst/lwz/geminiFT/J0842/cube/'
    c = CUBE(fp=fp,infile=fname)
    c.maskbad(fname,maskfile)




def main(J0906=False):
    #from radialbin import radialbin
    J0906=True
    if J0906:
    #    #J0906
    #    #radialbin()
        cube = CUBE()
    #    cube = CUBE(emlonly=True)
    #    cube.line_compare()
    #    #cube.line_compare(emlonly=True)
    #    #sn = cube.raw_sn()
    #    #sn3 = cube3.raw_sn()    
    #    


    #fit a single spectrum by python routine
        #cube = CUBE()
        #cube.fito3()
        '''
        generate raw_sn maps
        '''
        #cube.raw_sn(output="J0906_sn",plot=True,xr=[10,14],yr=[15,19],emlonly=True,window=[4997,5003.])
        #
        ##voronoi bin maps
        #from voronoi_2d_binning_J0906 import voronoi_binning_example
        #rpix='5'
        #voronoi_binning_example(rpix=rpix)
        ##spatial binning
        ##cube.sum(binmap='J0906_voronoi_2d_binning_output400kms.txt',binname='vornoi_sn2')
        #cube.sum(binmap='J0906_voronoi_2d_binning_outputrawsn_gt'+rpix+'pix.txt',binname='vornoi_sn_gt'+rpix+'pix')
        #pdb.set_trace()


    #    #sn=cube.raw_sn(output="J0906_sn")
    #    cube.sum(binmap='J0906_voronoi_2d_binning_output.txt',binname='vornoi_sn05')
    #    #cube.sum(binmap='manbin',binname='manbin')
        #cube.sum_old(binmap='boxbin',binname='boxbin',doperpixel=False)
        
        #compare with Keck spectra
        cube = CUBE('boxbin_rupketest_d02noscat.fits')
        cube.keck(pngtag='d02',x0=1,y0=0,kspec='/jwst/lwz/geminiFT/keck/J0906+56_spectrum.txt')

        pdb.set_trace()
    #    #cube.sum(binmap='pandabin',binname='pandabin')
    #    #cube.sum(binmap='twoconebin',binname='twoconebin')
    #    
    #    pdb.set_trace()
    #    #compare the same spaxel from two cubes
    #    #cube1 = CUBE(fp='/jwst/lwz/geminiFT/J0906/redux1/',infile='cstxeqxbrgN20190404S0175_d023D.fits')
    #    cube2 = CUBE(fp='/jwst/lwz/geminiFT/J0906/redux1_testwavelength/',infile='00_cstxeqxrgN20190404S0175_d023D.fits')
    #    plt.figure()
    #    cube1.plotspec(label='redux1')
    #    cube2.plotspec(label='testwave')
    #    plt.xlim(5000,7000)

    #    plt.figure()
    #    cube1.plotspec(label='redux1')
    #    cube2.plotspec(label='testwave')
    #    plt.xlim(5150,5300)
    #    
    #    plt.legend()
    #    plt.show()

    ##J0842
    else:
        #m with O2 absorption masked
        #cube3 = CUBE(fp='/jwst/lwz/geminiFT/J0842/cube/',infile='mJ0842d02_noscat.fits',emlonly=True\
        #,resultfp='/jwst/lwz/geminiFT/J0842/fitsd02_alllines/outputs/',contxdr='allfreeoffJ0842.cont.xdr',plotname='0842',z=0.02875)
        #pdb.set_trace()
        #
        #'''
        #generate raw_sn maps
        #'''
        #cube3.raw_sn(output="J0842_sn",window=[5003,5005],plot=True,xr=[10,14],yr=[15,19],emlonly=True)
        #pdb.set_trace()
        ##voronoi binning
        #rpix = '4'
        #from voronoi_2d_binning_J0842 import voronoi_binning_example
        #voronoi_binning_example(rpix=rpix)
        #'''
        #stack spectra
        #'''
        ##cube3.sum(binmap='J0842_voronoi_2d_binning_outputrawsn.txt',binname='vornoi_sn')
        #
        ##cube3.sum(binmap='J0842_voronoi_2d_binning_outputrawsn_gt3pix.txt',binname='vornoi_sn_gt3pix')
        #cube3.sum(binmap='J0842_voronoi_2d_binning_outputrawsn_gt'+rpix+'pix.txt',binname='vornoi_sn_gt'+rpix+'pix')
        #pdb.set_trace()


        #cube.sum(binmap='J0842_manbin',binname='manbin')
        #cube.sum(binmap='J0842_boxbin',binname='boxbin')
        #cube.sum(binmap='J0842_pandabin',binname='pandabin')
        #cube.sum(binmap='J0842_twoconebin',binname='twoconebin')

        
        '''
        compare with keck spectra
        '''


        #cube33 = CUBE(fp='/jwst/lwz/geminiFT/J0842/cube/',infile='boxbin_J0842d02_noscat.fits',emlonly=False,contcube=False)
        #cube33.keck(pngtag='d02_noscat')
        
        #cube333 = CUBE(fp='/jwst/lwz/geminiFT/J0842/cube/',infile='J0842offd02_noscat.fits',emlonly=True\
        #,resultfp='/jwst/lwz/geminiFT/J0842/fitsd02_alllines/outputs/',contxdr='allfreeoffJ0842.cont.xdr',plotname='0842',z=0.02875)
        #cube333.sum_old(binmap='J0842_boxbin',binname='boxbin')
        cube333 = CUBE(fp='/jwst/lwz/geminiFT/J0842/cube/',infile='boxbin_J0842offd02_noscat.fits',emlonly=False, contcube=False,\
        resultfp='/jwst/lwz/geminiFT/J0842/fitsd02_alllines/outputs/',contxdr='allfreeoffJ0842.cont.xdr',plotname='0842',z=0.02875)
        cube333.keck(pngtag='offd02',x0=1,y0=0)
        pdb.set_trace()
        
        '''
        compare spectra
        '''
        #plt.figure()
        cube4 = CUBE(fp='/jwst/lwz/geminiFT/J0842/redux1/',infile='cstxeqxbrgS20190429S0082_d023D.fits')
        cube4.plotspec(x=7,y=14,label='old')
        cube5 = CUBE(fp='/jwst/lwz/geminiFT/J0842/redux1_testwave2/',infile='cstxeqxbrgS20190429S0082_d023D.fits')
        cube5.plotspec(x=7,y=14)
        plt.legend()
        plt.show()

        #cube3 = CUBE(fp='/jwst/lwz/geminiFT/J0842/cube/',infile='J0842d02_noscat.fits')
        #cube3.sum(binmap='boxbin',binname='boxbin')
        #cube3.line_compare(z0=0.0288,x0=12,y0=17,name='J0842')
        #pdb.set_trace()
        
        cube33 = CUBE(fp='/jwst/lwz/geminiFT/J0842/cube/',infile='boxbin_J0842d02_noscat.fits')
        cube33.keck()

    #mask in each frame
    #basepath = Path('/jwst/lwz/geminiFT/J0906/cube/')
    #files_in_basepath = basepath.iterdir()
    #fp = '/jwst/lwz/geminiFT/J0906/cube/'
    #fp2 = '/jwst/lwz/geminiFT/J0906/pyscript/'
    #for item in files_in_basepath:
    #    fname = item.name
    #    if fnmatch.fnmatch(fname, 'd00*0429*d023D.fits'):
    #        print(fname[-13:-11])
    #        fid = int(fname[-13:-11])
    #        if fid < 86:
    #            sid = str(fid - 81)
    #        else:
    #            sid = str(fid - 83)
    #        maskfile = 'J0842_badmask'+sid
    #        print(maskfile,fname)
    #        c = CUBE(fp=fp,infile=fname)
    #        print('asdasdqwd',os.path.isfile(fp2+maskfile),fp2+maskfile)
    #        if os.path.isfile(fp2+maskfile):
    #            c.maskbad(fname,fp2+maskfile)
    # mask in the whole cube


    #cube3 = CUBE(fp='/jwst/lwz/geminiFT/J0906/redux1/',infile='cstxeqxbrgN20190404S0175_d023D.fits',z=0.0288)
if __name__ == "__main__":
    c = 299792.458 
    #mask(fname='J0842offd02_noscat.fits',maskfile='J0842cubeoff_badmask')
    #main(J0906=True)
    main()
    #cubewave = CUBE()
    #cubewave.binwave(binfactor=int(6))
    
    '''
    ######################################################################
    #fit lorentz to J0842
    #####################################################################
    #cube = CUBE(fp='/jwst/lwz/geminiFT/J0842/cube/',infile='mJ0842d02_noscat.fits',z=0.02953)
    #pdb.set_trace()
    #cube.fito3(contwav=np.array([[5030,5050],[5200,5250]]),emlwav=np.array([[5050,5170]]),x=12,y=17,fitline=True)
    '''
