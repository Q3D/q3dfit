#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue May 31 15:31:13 2022

@author: yuyuzo12
"""

import copy as copy
import importlib
# import matplotlib as mpl
import numpy as np
import os,sys
import pdb

# import q3dfit 

from astropy.table import Table
from ppxf.ppxf_util import log_rebin
from q3dfit.linelist import linelist
from q3dfit.readcube import Cube
from q3dfit.sepfitpars import sepfitpars
from q3dfit.qsohostfcn import qsohostfcn
from q3dfit.exceptions import InitializationError
from numpy.polynomial import legendre
from scipy import interpolate
from astropy.io import fits


import q3dfit.data

from matplotlib import pyplot as plt
from  matplotlib import cm
from matplotlib import colors as co
import matplotlib.gridspec as gridspec
from astropy.cosmology import WMAP9 as cosmo
from astropy import units as u
import copy

from mpl_toolkits.axes_grid1 import make_axes_locatable
from matplotlib.ticker import MaxNLocator,LinearLocator,FixedLocator
from scipy.spatial import distance
from plotbin.sauron_colormap import register_sauron_colormap
import matplotlib.gridspec as gridspec
from matplotlib.colors import LogNorm

plt.rcParams["font.family"] = "serif"
plt.rcParams["mathtext.fontset"] = "dejavuserif"
plt.rcParams['figure.constrained_layout.use'] = False

class Q3Dpro:
    # =============================================================================================
    # instantiate the q3dpro object
    # =============================================================================================
    def __init__(self,initproc,SILENT=True,NOCONT=False,NOLINE=False,PLATESCALE=0.15):
        # read in the q3d initproc file and unpack 
        self.q3dinit = initproc
        # unpack initproc
        self.target_name = self.q3dinit['name']
        self.silent = SILENT
        if self.silent != True:
            print('processing outputs...')
            print('Target name:',self.target_name)
            
        self.pix = PLATESCALE # pixel size
        self.bad = 1e99
        self.dataDIR = initproc['outdir']
        # instantiate the Continuum (npy) and Emission Line (npz) objects
        if NOCONT == False:
            self.contdat = ContData(initproc,self.dataDIR)
        if NOLINE == False:
            self.linedat = LineData(initproc,self.dataDIR)
        
        return
    
    # =============================================================================================
    # processing the q3dfit cube data
    # =============================================================================================
    def get_psfsubcube(self):
        qso_subtract = self.contdat.host_mod - self.contdat.qso_mod
        return qso_subtract
    
    def get_lineprop(self,LINESELECT,LINEVAC=True):
        listlines = linelist(self.linedat.lines,vacuum=LINEVAC)
        ww = np.where(listlines['name'] == LINESELECT)[0]
        linewave = listlines['lines'][ww].value[0]
        linename = listlines['linelab'][ww].value[0]
        # output in MICRON
        return linewave,linename
    
    def make_linemap(self,LINESELECT,SNRCUT=None,LINEVAC=True,
                     xyCenter=None,VMINMAX=None,
                     XYSTYLE=None,PLTNUM=1,CMAP=None,
                     SAVEDATA=False,SILENT=None):
        print('Plotting emission line maps')
        c_kms = 2.99792458e5
        # breakpoint()
        if SILENT == None :
            SILENT = self.silent
        
        if SILENT != True: 
            print('create linemap:',LINESELECT)
        
        
        redshift = self.q3dinit['zinit_gas'][LINESELECT]
        ncomp = self.q3dinit['ncomp'][LINESELECT][0,0]
        
        matrix_size = redshift.shape
        # if len(matrix_size) > 2:
        #     if matrix_size[2] > 1:
        #         redshift = redshift[:,:,0]
        #     else:
        #         redshift = redshift.reshape(matrix_size[0],matrix_size[1])
        
        kpc_arcsec = cosmo.kpc_proper_per_arcmin(redshift).value/60.
        # arckpc = cosmo.kpc_proper_per_arcmin(redshift).value/60.
        # argscheckcomp = self.q3dinit['argscheckcomp']
        # xycen = xyCenter # (nx,ny) of the center
        

        if SNRCUT == None :
            SNRCUT = 5 # S/N threshold for the map
            
        # cid = -1 # index of the component to plot, -1 is the broadest component
        # central wavelength --> need to get from linereader 
        wave0,linetext = self.get_lineprop(LINESELECT,LINEVAC=LINEVAC)

        # --------------------------
        # EXTRACT THE DATA HERE
        # --------------------------
        # EXTRACT TOTAL FLUX
        fluxsum     = self.linedat.get_flux(LINESELECT,FLUXSEL='ftot')['flux']
        fluxsum_err = self.linedat.get_flux(LINESELECT,FLUXSEL='ftot')['fluxerr']
        fsmsk = clean_mask(fluxsum,BAD=self.bad)
        
        dataOUT = {'Ftot':{'data':fluxsum*fsmsk,'err':fluxsum_err*fsmsk,'name':['F$_{tot}$']},
                   'Fci' :{'data':np.zeros(matrix_size),'err':np.zeros(matrix_size),'name':[]},
                   'Sig' :{'data':np.zeros(matrix_size),'err':np.zeros(matrix_size),'name':[]},
                   'v50' :{'data':np.zeros(matrix_size),'err':None,'name':[]}} 
        # EXTRACT COMPONENTS
        for ci in range(0,ncomp) :
            ici = ci+1
            fcl = 'fc'+str(ici)
            iflux = self.linedat.get_flux(LINESELECT,FLUXSEL=fcl)['flux']
            ifler = self.linedat.get_flux(LINESELECT,FLUXSEL=fcl)['fluxerr']
            isigm = self.linedat.get_sigma(LINESELECT,COMPSEL=ici)['sig']
            isger = self.linedat.get_sigma(LINESELECT,COMPSEL=ici)['sigerr']
            iwvcn = self.linedat.get_wave(LINESELECT,COMPSEL=ici)['wav']
            ireds = redshift[:,:,ci]

            # now process them
            iv50 = ((iwvcn-wave0)/wave0 - ireds) * c_kms
            # mask out the bad values
            ifmask = clean_mask(iflux,BAD=self.bad)
            isgmsk = clean_mask(isigm,BAD=self.bad)
            iwvmsk = clean_mask(iwvcn,BAD=self.bad)
            
            # save to the processed matrices
            dataOUT['Fci']['data'][:,:,ci] = iflux*ifmask
            dataOUT['Sig']['data'][:,:,ci] = isigm*isgmsk
            dataOUT['v50']['data'][:,:,ci] = iv50*iwvmsk
            dataOUT['Fci']['err'][:,:,ci]  = ifler*ifmask
            dataOUT['Sig']['err'][:,:,ci]  = isger*isgmsk
            
            dataOUT['Fci']['name'].append('F$_{c'+str(ici)+'}$')
            dataOUT['Sig']['name'].append('$\sigma_{c'+str(ici)+'}$')
            dataOUT['v50']['name'].append('v$_{50,c'+str(ici)+'}$')

        #read in the information
        # for i in range(nspec):
        #     wav[i,:] = w[i*ncomp:(i+1)*ncomp]
        #     v50[i,:] = ((wav[i,:]-wav0)/wav0 - redshift) * c
        #     s[i,:] = sigma[i*ncomp:(i+1)*ncomp]
        #     colnew[i] = Col[i*ncomp]
        #     rownew[i] = Row[i*ncomp]
        #     flux[i,:] = flux0[i*ncomp:(i+1)*ncomp]
        #     fluxerr[i,:] = flux0err[i*ncomp:(i+1)*ncomp]
        #     tgid = np.where(flux[i,:] > 0)
        #     fluxsum[i] = np.sum(flux[i,tgid])
        #     fluxsum_err[i] = (np.sum(fluxerr[i,tgid]**2))**0.5
        #     if ncomp > 1:
        #         goodcomp = np.where((s[i,:]>0) & (s[i,:]<bad) & (flux[i,:]>0) & (flux[i,:]<bad))
        #         if np.size(goodcomp) > 1:
        #             sindex = np.argsort(s[i,goodcomp])
        #             wav[i,goodcomp] = wav[i,goodcomp][0][sindex]
        #             v50[i,goodcomp] = v50[i,goodcomp][0][sindex]
        #             s[i,goodcomp] = s[i,goodcomp][0][sindex]
        #             flux[i,goodcomp] = flux[i,goodcomp][0][sindex]
        #             fluxerr[i,goodcomp] = fluxerr[i,goodcomp][0][sindex]
        #             fluxsum[i] = np.sum(flux[i,:])
        #             fluxsum_err[i] = (np.sum(fluxerr[i,:]**2))**0.5

        # sn_tot = fluxsum/fluxsum_err
        # w80 = sigma*2.563   # w80 linewidth from the velocity dispersion
        # sn = flux/flux_err
        fluxsum_snc,gdindx,bdindx = snr_cut(fluxsum,fluxsum_err,SNRCUT=SNRCUT)
        
        # nnpix = 30

        # #pad the outer region of the cubes with nan, for those smaller cubes 
        # xarr0 = np.arange(np.min(xcol)-nnpix*pix,np.max(xcol)+nnpix*pix,pix)
        # yarr0 = np.arange(np.min(xcol)-nnpix*pix,np.max(xcol)+nnpix*pix,pix)
        # xarr,yarr = np.meshgrid(xarr0,yarr0)
        # xarr00 =np.reshape(xarr,-1)
        # yarr00 =np.reshape(yarr,-1)
        # inid = np.where((xarr00 >= np.min(xcol)) & (xarr00 <= np.max(xcol)+1) & (yarr00 >= np.min(ycol)) & (yarr00 <= np.max(ycol)))
        # ouid = np.array([])
        # for  i in np.arange(np.size(xarr)):
        #     if np.size(np.where(i == inid)) == 0:
        #         ouid = np.append(ouid,i)
        # ouid = np.array(ouid,dtype=int)
        
        
        
        # nadd = np.size(ouid)
        # zarr = np.zeros(nadd)+np.nan

        # w80c3 = w80[:,cid]
        # v50c3 = v50[:,cid]
        # sn_c3 = sn[:,cid]
        # fluxc3 = flux[:,cid]
        # wavc3 = wav[:,cid]

        # if SNRCUT != None1
        # # trim the X Y range (default is 3 arcsec) of the map and set an S/N threshold
        # try:
        #     xb = argscheckcomp['sigcut']*arckpc
        #     yb = argscheckcomp['sigcut']*arckpc
        #     gid = np.where(((xcolkpc < 1.15*xb) & (xcolkpc > -1.1*xb) & (ycolkpc < 1.2*yb) & (ycolkpc > -1.2*yb)))
            
        #     bid = np.where((w80c3 < 1e-5) | (w80c3 > 5e98) | (np.isfinite(w80c3) == False) | (sn_tot < sncut) | (np.isfinite(sn_tot) == False) | \
        #         (v50c3 > 5e98) | (wavc3 > 5e98) | (np.isfinite(wavc3) == False) | (fluxc3 <= 1e-10) | (fluxc3 > 5e98) | (np.isfinite(fluxc3) == False)) # |
            
            
        #     w80c3[bid] = np.nan
        #     v50c3[bid] = np.nan

        # except (IndexError or KeyError):
        #     print('x, y ranges or S/N threshold not properly set')
        
        FLUXLOG = False
        
        # --------------------------
        # Do the plotting here
        # --------------------------
        # pixkpc = self.pix*arckpc
        # Here, we determine the plot axis 
        xgrid = np.arange(0, matrix_size[1])
        ygrid = np.arange(0, matrix_size[0])
        xcol = xgrid
        ycol = ygrid
        
        qsoCenter = xyCenter
        if xyCenter == None :
            qsoCenter = [int(np.ceil(matrix_size[0]/2)),int(np.ceil(matrix_size[1]/2))]
    
        XYtitle = 'Spaxel'
        if XYSTYLE != None and XYSTYLE != False:
            #if XYSTYLE.lower() == 'center':
            xcol = (xgrid-xyCenter[1]) 
            ycol = (ygrid-xyCenter[0])
            if xyCenter != None :
                qsoCenter = [0,0]
            if XYSTYLE.lower() == 'kpc':
                kpc_pix = np.median(kpc_arcsec)* self.pix
                xcolkpc = xcol*kpc_pix
                ycolkpc = ycol*kpc_pix
                xcol,ycol = xcolkpc,ycolkpc
                XYtitle = 'Relative distance [kpc]'
        plt.close(PLTNUM)
        figDIM = [ncomp,4]
        figOUT = set_figSize(figDIM,matrix_size)
        fig,ax = plt.subplots(figDIM[0],figDIM[1])
        fig.set_figheight(figOUT[1])
        fig.set_figwidth(figOUT[0])
        if CMAP == None :
            CMAP = 'YlOrBr_r'
        ici = ''
        i,j=0,0
        for icomp,ipdat in dataOUT.items():
            pixVals = ipdat['data']
            ipshape = pixVals.shape
            if VMINMAX != None :
                vminmax = VMINMAX[icomp]
            if icomp == 'Ftot':
                doPLT = True
                NTICKS = 4
                ici=''
                vticks = None
                if 'fluxlog' in VMINMAX :
                    FLUXLOG = VMINMAX['fluxlog']
            else:
                i=0
                if icomp.lower() == 'fci' :
                    NTICKS = 4
                    vticks = None
                    if 'fluxlog' in VMINMAX :
                        FLUXLOG = VMINMAX['fluxlog']
                elif icomp.lower() == 'sig':
                    CMAP = 'YlOrBr_r'
                    NTICKS = 5
                    vticks = None
                    FLUXLOG = False
                elif icomp.lower() == 'v50' :
                    vticks = [vminmax[0],vminmax[0]/2,0,vminmax[1]/2,vminmax[1]]
                    CMAP = 'RdYlBu'
                    CMAP += '_r'
                    FLUXLOG = False
            for ci in range(0,ncomp) :
                i = ci
                if len(ipshape) > 2:
                    pixVals = ipdat['data'][:,:,ci]
                    doPLT = True
                    ici = '_c'+str(ci+1)
                if j == 0 and ci > 0:
                    fig.delaxes(ax[i,j])
                    doPLT = False
                if doPLT == True:
                    cmap_r = cm.get_cmap(CMAP)
                    cmap_r.set_bad(color='black')
                    xx, yy = xcol,ycol
                    axi = []
                    if ncomp < 2:
                        axi = ax[j]
                    else:
                        axi = ax[i,j]
                    display_pixels_wz(yy, xx,pixVals,CMAP=CMAP,AX=axi,COLORBAR=True,PLOTLOG=FLUXLOG,
                                      VMIN=vminmax[0],VMAX=vminmax[1],TICKS=vticks,NTICKS=NTICKS)
                    if xyCenter != None :
                        axi.errorbar(qsoCenter[0],qsoCenter[1],color='black',mew=1,mfc='red',fmt='*',markersize=15,zorder=2)
                    axi.set_xlabel(XYtitle,fontsize=16)
                    axi.set_ylabel(XYtitle,fontsize=16)
                    axi.set_title(ipdat['name'][ci],fontsize=20,pad=45)
                    # axi.set_ylim([max(xx),np.ceil(min(xx))])
                    # axi.set_xlim([min(yy),np.ceil(max(yy))])
                    if SAVEDATA == True:
                        linesave_name = self.target_name+'_'+LINESELECT+'_'+icomp+ici+'_map.fits'
                        print('Saving line map:',linesave_name)
                        savepath = os.path.join(self.dataDIR,linesave_name)
                        save_to_fits(pixVals,[],savepath)
            j+=1
        fig.suptitle(self.target_name+' : '+linetext+' maps',fontsize=20,snap=True,
                     horizontalalignment='right')
                     # verticalalignment='center',
                     # fontweight='semibold')
        fig.tight_layout(pad=0.15,h_pad=0.1)
        if SAVEDATA == True:
            pltsave_name = LINESELECT+'_emlin_map'
            print('Saving figure:',pltsave_name)
            plt.savefig(os.path.join(self.dataDIR,pltsave_name+'.png'),format='png')
            plt.savefig(os.path.join(self.dataDIR,pltsave_name+'.pdf'),format='pdf')
        # fig.subplots_adjust(top=0.88)
        plt.show()
        return
    
    def make_contmap(self):
        
        return
    
    def make_BPT(self,SNRCUT=3,
                 SAVEDATA=False,PLTNUM=5,KPC=False):
        
        
        # ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
        # first identify the lines, extract fluxes, and apply the SNR cuts
        # ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
        BPTlines = {'Hbeta':None,
                    '[OIII]5007':None,
                    '[OI]6300':None,
                    'Halpha' :None,
                    '[NII]6548':None,
                    '[NII]6583':None,
                    '[SII]6716':None,
                    '[SII]6731':None}
        
        # redshift = self.q3dinit['zinit_gas'][LINESELECT]
        # matrix_size = redshift.shape
        # redshift = redshift.reshape(matrix_size[0],matrix_size[1])
        # arckpc = cosmo.kpc_proper_per_arcmin(redshift).value/60.
        for lin in self.linedat.lines:
            if lin in BPTlines:
               fluxsum     = self.linedat.get_flux(lin,FLUXSEL='ftot')['flux']
               fluxsum_err = self.linedat.get_flux(lin,FLUXSEL='ftot')['fluxerr']
               fmask       = clean_mask(fluxsum,BAD=self.bad)
               flux_snc,gud_indx,bad_indx = snr_cut(fluxsum,fluxsum_err,SNRCUT=SNRCUT)
               redshift = self.q3dinit['zinit_gas'][lin]
               matrix_size = redshift.shape
               redshift = redshift.reshape(matrix_size[0],matrix_size[1])
               arckpc = cosmo.kpc_proper_per_arcmin(redshift).value/60.
               BPTlines[lin] = [fluxsum,fluxsum_err,flux_snc,fmask,gud_indx]

        lineratios = {'OiiiHb':None,
                      'SiiHa':None,
                      'OiHa':None,
                      'NiiHa':None,}
        BPTmod = {'OiiiHb/NiiHa':None,
                    'OiiiHb/SiiHa':None,
                    'OiiiHb/OiHa':None}
        # ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
        # make the theoretical BPT models
        # ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
        for bpt in BPTmod:
            if bpt == 'OiiiHb/NiiHa' :
                xkew1 = 0.05*np.arange(110)-5
                ykew1 = 0.61 / (xkew1-0.47)+1.19
                xkew2 = 0.05*np.arange(41)-2
                ykew2 = 0.61 / (xkew2-0.05)+1.3
                BPTmod[bpt] = [[xkew1,ykew1],[xkew2,ykew2]]
            elif bpt == 'OiiiHb/SiiHa' :
                xkew1 = 0.05*np.arange(105)-5
                ykew1 = 0.72 / (xkew1-0.32)+1.30
                xkew2 = 0.5*np.arange(2)-0.4
                ykew2 = 1.89*xkew2+0.76
                BPTmod[bpt] = [[xkew1,ykew1],[xkew2,ykew2]]
            elif bpt == 'OiiiHb/OiHa' :
                xkew1 = 0.05*np.arange(85)-5
                ykew1 = 0.73 / (xkew1+0.59)+1.33
                xkew2 = 0.5*np.arange(2)-1.1
                ykew2 = 1.18*xkew2 + 1.30
                BPTmod[bpt] = [[xkew1,ykew1],[xkew2,ykew2]]
            
        # ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
        # Calculate the line ratios
        # ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
        cntr = 0
        bptc = 0
        # ++++++++++++++++++
        # [OIII]/Hbeta
        # ++++++++++++++++++
        if BPTlines['[OIII]5007'] != None and BPTlines['Hbeta'] != None :
            f1,f1_err,f1_snc,m1,g1_gindx = BPTlines['[OIII]5007']
            f2,f2_err,f2_snc,m2,g2_gindx = BPTlines['Hbeta']
            mm = m1*m2
            frat = f1/f2
            # gdindx = np.intersect1d(g1_gindx, g2_gindx)
            # cc = list(set.intersection(*map(set,[g1_gindx, g2_gindx])))
            frat10 = np.log10(frat)
            frat10err = lgerr(f1_snc,f2_snc,f1_err,f2_err)
            pltrange = [-1,1.5]
            lineratios['OiiiHb'] = [frat10*mm,frat10err,'[OIII]/H$\\beta$',pltrange]
            cntr += 1
        # ++++++++++++++++++
        # [SII]/Halpha 
        # ++++++++++++++++++
        if (BPTlines['[SII]6716'] != None or BPTlines['[SII]6731'] != None) and BPTlines['Halpha'] != None :
            f2,f2_err,f2_snc,m2,g2_gindx = BPTlines['Halpha']
            f1,f1a,f1b = [],[],[]
            f1_err,f1a_err,f1b_err = [],[],[]
            if BPTlines['[SII]6716'] != None :
                f1a,f1a_err,f1a_snc,m1a,g1a_gindx = BPTlines['[SII]6716']
            else:
                f1a,f1a_err,f1a_snc = np.zeros(f1.shape),np.zeros(f2.shape),np.zeros(f2.shape)
                m1a,g1a_gindx = np.zeros(f2.shape)+1,[]
            if BPTlines['[SII]6731'] != None :
                f1b,f1b_err,f1b_snc,m1b,g1b_gindx = BPTlines['[SII]6731']
            else:
                f1b,f1b_err,f1b_snc = np.zeros(f1.shape),np.zeros(f1.shape),np.zeros(f1.shape)
                m1b,g1b_gindx = np.zeros(f1.shape)+1,[]
            f1     = f1a+f1b
            f1_snc = f1a_snc+f1b_snc
            f1_err = np.sqrt(np.array(f1a_err)**2+np.array(f1b_err)**2)
            mm = m2*m1a*m1b
            frat = f1/f2
            frat10 = np.log10(frat)
            frat10err = lgerr(f1_snc,f2_snc,f1_err,f2_err)
            pltrange = [-1.8,0.1]
            pltrange = [-1.8,0.9]
            lineratios['SiiHa'] = [frat10,frat10err,'[SII]/H$\\alpha$',pltrange]
            cntr += 1
            bptc += 1
        # ++++++++++++++++++
        # [OI]/Halpha
        # ++++++++++++++++++
        if BPTlines['[OI]6300'] != None and BPTlines['Halpha'] != None :
            f1,f1_err,f1_snc,m1,g1_gindx = BPTlines['[OI]6300']
            f2,f2_err,f2_snc,m2,g2_gindx = BPTlines['Halpha']
            mm = m1*m2
            frat = f1/f2
            frat10 = np.log10(frat)
            frat10err = lgerr(f1_snc,f2_snc,f1_err,f2_err)
            pltrange = [-2.1,0]
            lineratios['OiHa'] = [frat10*mm,frat10err,'[OI]/H$\\alpha$',[-1.8,0.1]]
            cntr += 1
            bptc += 1
        # ++++++++++++++++++
        # [NII]/Halpha
        # ++++++++++++++++++
        if BPTlines['[NII]6583'] != None and BPTlines['Halpha'] != None :
            f1,f1_err,f1_snc,m1,g1_gindx = BPTlines['[OIII]5007']
            f2,f2_err,f2_snc,m2,g2_gindx = BPTlines['Halpha']
            mm = m1*m2
            frat = f1/f2
            frat10 = np.log10(frat)
            frat10err = lgerr(f1_snc,f2_snc,f1_err,f2_err)
            pltrange = [-1.8,0.3]
            lineratios['NiiHa'] = [frat10*mm,frat10err,'[NII]/H$\\alpha$',[-1.8,0.1]]
            cntr += 1
            bptc += 1
            
        # breakpoint()
        # ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
        # Do the plotting here
        # ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
        if cntr != 0 and bptc != 0:
            matrix_size = BPTlines['[OIII]5007'][0].shape
            xyCenter = [int(np.ceil(matrix_size[0]/2)),int(np.ceil(matrix_size[1]/2))]
            xgrid = np.arange(0, matrix_size[1])
            ygrid = np.arange(0, matrix_size[0])
            xcol = xgrid
            ycol = ygrid
            if KPC == True :
                xcol = (xgrid-xyCenter[1]) 
                ycol = (ygrid-xyCenter[0]) 
            # --------------------------
            # Plot ine ratio map
            # --------------------------
            plt.close(PLTNUM)
            figDIM = [1,cntr]
            figOUT = set_figSize(figDIM,matrix_size)
            fig,ax = plt.subplots(1,cntr,num=PLTNUM,constrained_layout=True)#, gridspec_kw={'height_ratios': [1, 2]})
            fig.set_figheight(figOUT[1])
            fig.set_figwidth(figOUT[0])
            
            CMAP='inferno'
            cmap_r = cm.get_cmap(CMAP)
            cmap_r.set_bad(color='black')
            cf = 0
            for linrat in lineratios:
                if linrat != None :
                    xx,yy = xcol,ycol
                    iax = ax[cf]
                    frat10,frat10err,pltname,pltrange = lineratios[linrat]
                    display_pixels_wz(yy,xx,frat10,CMAP=CMAP,AX=iax,
                                      VMIN=-1,VMAX=1,
                                      NTICKS=5,COLORBAR=True)
                    # iax.errorbar(0,0,color='black',fmt='*',markersize=10,zorder=2)
                    iax.set_xlabel('spaxel',fontsize=13)
                    iax.set_ylabel('spaxel',fontsize=13)
                    if KPC == True:
                        iax.set_xlabel('Relative distance [kpc]',fontsize=13)
                        iax.set_ylabel('Relative distance [kpc]',fontsize=13)
                    iax.set_title('log$_{10}$ '+pltname,fontsize=15,pad=45)
                    # iax.set_ylim([max(xx),np.ceil(min(xx))])
                    # iax.set_xlim([min(yy),np.ceil(max(yy))])
                    cf += 1
            plt.tight_layout(pad=1.5,h_pad=0.1)
            if SAVEDATA == True:
                pltsave_name = 'emlin_ratio_map.png'
                print('Saving figure:',pltsave_name)
                plt.savefig(os.path.join(self.dataDIR,pltsave_name))
            plt.show()
            # --------------------------
            # Plot BPT here
            # --------------------------
            PLTNUM += 1
            plt.close(PLTNUM)
            fig,ax = plt.subplots(1,bptc,figsize=(bptc*5,5),num=PLTNUM,constrained_layout=True)
            cf=0
            # breakpoint()
            pixkpc = self.pix*arckpc
            xgrid = np.arange(0, matrix_size[1])
            ygrid = np.arange(0, matrix_size[0])
            xcol = (xgrid-xyCenter[1]) 
            ycol = (ygrid-xyCenter[0]) 
            xcolkpc = xcol*np.median(arckpc)* self.pix
            ycolkpc = ycol*np.median(arckpc)* self.pix
            for bpt in BPTmod:
                if bpt != None :
                    iax = ax[cf]
                    # first plot the theoretical curves
                    iBPTmod = BPTmod[bpt]
                    iax.plot(iBPTmod[0][0],iBPTmod[0][1],'k-',zorder=1,linewidth=1.5)
                    iax.plot(iBPTmod[1][0],iBPTmod[1][1],'k--',zorder=1,linewidth=1.5)
                    
                    iax.minorticks_on()
                    fnames = bpt.split('/')
                    frat1,frat1err,fratnam1,pltrng1 = lineratios[fnames[0]]
                    frat2,frat2err,fratnam2,pltrng2 = lineratios[fnames[1]]
                    
                    # print(fratnam1,fratnam2)
                    
                    gg = np.where(~np.isnan(frat1) & ~np.isnan(frat2))
                    xfrat,xfraterr = [],[[],[]]
                    yfrat,yfraterr = [],[[],[]]
                    for pi in range(len(gg[0])):
                        yfrat.append(frat1[gg[0][pi],gg[1][pi]])
                        yfraterr[0].append(frat1err[0][gg[0][pi],gg[1][pi]])
                        yfraterr[1].append(frat1err[1][gg[0][pi],gg[1][pi]])
                        xfrat.append(frat2[gg[0][pi],gg[1][pi]])
                        xfraterr[0].append(frat2err[0][gg[0][pi],gg[1][pi]])
                        xfraterr[1].append(frat2err[1][gg[0][pi],gg[1][pi]])
                    iax.errorbar(xfrat,yfrat,fmt='.',
                                 # xerr=xfraterr,yerr=yfraterr,
                                 color='black',markersize=5,zorder=2)
                    iax.errorbar(xfrat,yfrat,fmt='.',
                                 xerr=xfraterr,yerr=yfraterr,
                                 color='black',markersize=0,
                                 elinewidth=1.5,ecolor='blue',alpha=0.5,zorder=3)
                                  # color=cmap(norm(r_kpc[kk])),ms=1)
                    iax.errorbar(np.median(xfrat),np.median(yfrat),
                                 fillstyle='none',color='red',fmt='*',markersize=17,mew=2.5,zorder=3)
                    iax.set_ylim(pltrng1)
                    # iax.set_xlim(pltrng2)
                    iax.set_xlim([-2.1,0.7])
                    if cf == 0:
                        iax.set_ylabel(fratnam1,fontsize=16)
                        iax.tick_params(axis='y',which='major', length=10, width=1, direction='in',labelsize=13,
                                      bottom=True, top=True, left=True, right=True,color='black')
                    else:
                        iax.tick_params(axis='y',which='major', length=10, width=1, direction='in',labelsize=0,
                                      bottom=True, top=True, left=True, right=True,color='black')
                    iax.set_xlabel(fratnam2,fontsize=16)
                    
                    iax.tick_params(axis='x',which='major', length=10, width=1, direction='in',labelsize=13,
                                  bottom=True, top=True, left=True, right=True,color='black')
                    iax.tick_params(which='minor', length=5, width=1, direction='in',
                                  bottom=True, top=True, left=True, right=True,color='black')
                    cf+=1
                    # breakpoint()
    
            plt.tight_layout(pad=1.5,h_pad=0.1)
            if SAVEDATA == True:
                pltsave_name = 'BPT_map.png'
                print('Saving figure:',pltsave_name)
                plt.savefig(os.path.join(self.dataDIR,pltsave_name))
            # plt.savefig(tname+'_c'+str(int(ic+1))+'_bpt.pdf')
            plt.show()
            # breakpoint()
        return
    
    # =============================================================================================
    # reading in the .npy and .npz files
    # =============================================================================================
    # Emission line data
    # ---------------------------------------------------------------------------------------------

class LineData:
    def __init__(self,initproc,dataDIR):
        filename = initproc['label']+'.lin.npz'
        datafile = os.path.join(dataDIR,filename)
        # print(datafile)
        if os.path.exists(datafile) != True:
            print('ERROR: emission line ('+filename+') file does not exist')
            return
        self.data = self.read_npz(datafile)
        self.lines = initproc['lines']
        self.zgas  = initproc['zinit_gas']
        # self.flux    = self.get_flux()
        # self.siga    = self.get_sigma()
        # self.wavelen = self.get_wave()
        # self.eq      = self.get_weq()
        return
    
    def read_npz(self,datafile):
        dataread = np.load(datafile,allow_pickle=True)
        self.colname = sorted(dataread)
        return dataread
    
    def get_flux(self,lineselect,FLUXSEL='ftot'):
        # FLUXSEL = 'ftot' by default --> select from ('ftot', 'fc1', 'fc1pk')
        if lineselect not in self.lines:
            print('ERROR: line does not exist')
            return None
        emlflx     = self.data['emlflx'].item()
        emlflxerr  = self.data['emlflxerr'].item()
        dataout = {'flux':emlflx[FLUXSEL][lineselect],'fluxerr':emlflxerr[FLUXSEL][lineselect]}
        return dataout
    
    def get_sigma(self,lineselect,COMPSEL=1):
        if lineselect not in self.lines:
            print('ERROR: line does not exist')
            return None
        # 'c1'
        emlsig     = self.data['emlsig'].item()
        emlsigerr  = self.data['emlsigerr'].item()
        csel = 'c'+str(COMPSEL)
        dataout = {'sig':emlsig[csel][lineselect],'sigerr':emlsigerr[csel][lineselect]} 
        return dataout
    
    def get_wave(self,lineselect,COMPSEL=1):
        if lineselect not in self.lines:
            print('ERROR: line does not exist')
            return None
        # 'c1'
        emlwav     = self.data['emlwav'].item()
        emlwaverr  = self.data['emlwaverr'].item()
        csel = 'c'+str(COMPSEL)
        dataout = {'wav':emlwav[csel][lineselect],'waverr':emlwaverr[csel][lineselect]}
        return dataout
    
    def get_weq(self,lineselect,FLUXSEL='ftot'):
        # FLUXSEL = 'ftot' by default --> select from ('ftot', 'fc1')
        if lineselect not in self.lines:
            print('ERROR: line does not exist')
            return None
        # 'ftot', 'fc1'
        emlweq = self.data['emlweq'].item()
        dataout = emlweq[FLUXSEL][lineselect]
        return dataout
    
# ---------------------------------------------------------------------------------------------
# Continuum data
# ---------------------------------------------------------------------------------------------
class ContData:
    def __init__(self,initproc,dataDIR):
        filename = initproc['label']+'.cont.npy'
        datafile = os.path.join(dataDIR,filename)
        if os.path.exists(datafile) != True:
            print('ERROR: continuum ('+filename+') file does not exist')
            return None
        self.data = self.read_npy(datafile)
        self.wave           = self.data['wave']
        self.qso_mod        = self.data['qso_mod']
        self.qso_poly_mod   = self.data['qso_poly_mod']
        self.host_mod       = self.data['host_mod']
        self.poly_mod       = self.data['poly_mod']
        self.npts           = self.data['npts']
        self.stel_sixgma     = self.data['stel_sigma']
        self.stel_sigma_err = self.data['stel_sigma_err']
        self.stel_z         = self.data['stel_z']
        self.stel_z_err     = self.data['stel_z_err']
        self.stel_rchisq    = self.data['stel_rchisq']
        self.stel_ebv       = self.data['stel_ebv']
        self.stel_ebv_err   = self.data['stel_ebv_err']
        return
    
    def read_npy(self,datafile):
        dataout = np.load(datafile,allow_pickle=True).item()
        self.colname = dataout.keys()
        return dataout

##############################################################################
# code from W.Liu  - adapted from PPXF
# edited by Y.Ishikawa
##############################################################################
"""
Copyright (C) 2014-2017, Michele Cappellari
E-mail: michele.cappellari_at_physics.ox.ac.uk

Updated versions of the software are available from my web page
http://purl.org/cappellari/software

See example at the bottom for usage instructions.

MODIFICATION HISTORY:
    V1.0.0: Created to emulate my IDL procedure with the same name.
        Michele Cappellari, Oxford, 28 March 2014
    V1.0.1: Fixed treatment of optional parameters. MC, Oxford, 6 June 2014
    V1.0.2: Avoid potential runtime warning. MC, Oxford, 2 October 2014
    V1.0.3: Return axis. MC, Oxford, 26 March 2015
    V1.0.4: Return image instead of axis. MC, Oxford, 15 July 2015
    V1.0.5: Removes white gaps from rotated images using edgecolors.
        MC, Oxford, 5 October 2015
    V1.0.6: Pass kwargs to graphics functions.
        MC, Campos do Jordao, Brazil, 23 November 2015
    V1.0.7: Check that input (x,y) come from an axis-aligned image.
        MC, Oxford, 28 January 2016
    V1.0.8: Fixed deprecation warning in Numpy 1.11. MC, Oxford, 22 April 2016
    V1.1.0: Fixed program stop with kwargs. Included `colorbar` keyword.
        MC, Oxford, 18 May 2016
    V1.1.1: Use interpolation='nearest' to avoid crash on MacOS.
        MC, Oxford, 14 June 2016
    V1.1.2: Specify origin=`upper` in imshow() for consistent results with older
        Matplotlib version. Thanks to Guillermo Bosch for reporting the issue.
        MC, Oxford, 6 January 2017
    V1.1.3: Simplified passing of default keywords. MC, Oxford, 20 February 2017
    V1.1.4: Use register_sauron_colormap(). MC, Oxford, 29 March 2017
    V1.1.5: Request `pixelsize` when dataset is large. Thanks to Davor
        Krajnovic (Potsdam) for the feedback. MC, Oxford, 10 July 2017
    V1.1.6: Fixed new incompatibility with Matplotlib 2.1.
        MC, Oxford, 9 November 2017
    V1.1.7: Changed imports for plotbin as a package. MC, Oxford, 17 April 2018
    
"""

def display_pixels_wz(y, x, datIN, PIXELSIZE=None, VMIN=None, VMAX=None,TICKS=None,PLOTLOG=False,
                   ANGLE=None, COLORBAR=False, AUTOCBAR=False,LABEL=None, NTICKS=3,
                   CMAP='RdYlBu',SKIPTICK=False, **kwargs):
    """
    Display vectors of square pixels at coordinates (x,y) coloured with "val".
    An optional rotation around the origin can be applied to the whole image.
    
    The pixels are assumed to be taken from a regular cartesian grid with 
    constant spacing (like an axis-aligned image), but not all elements of
    the grid are required (missing data are OK).

    This routine is designed to be fast even with large images and to produce
    minimal file sizes when the output is saved in a vector format like PDF.

    """
    if VMIN is None:
        VMIN = np.min(datIN[datIN != np.nan])

    if VMAX is None:
        VMAX = np.max(datIN[datIN != np.nan])

    # print(VMIN,VMAX)
    xmin, xmax = np.ceil(np.min(x)), np.ceil(np.max(x))
    ymin, ymax = np.ceil(np.min(y)), np.ceil(np.max(y))
    xmax = 5*np.round(np.array(xmax)/5)
    ymax = 5*np.round(np.array(ymax)/5)

    ax = kwargs.get('AX',None)
    imgPLT = None
    if PLOTLOG == False :
        imgPLT = ax.imshow(np.rot90(datIN,1) ,
                           # origin='lower', 
                            cmap=CMAP,
                            extent=[ymin, ymax,xmin, xmax],
                           vmin=VMIN, vmax=VMAX,
                           interpolation='none')
    else :
        imgPLT = ax.imshow(np.rot90(datIN,1) ,
                           # origin='lower', 
                           cmap=CMAP,
                            extent=[ymin, ymax,xmin, xmax],
                           norm=LogNorm(vmin=VMIN, vmax=VMAX),
                           interpolation='none')
    
    # current_cmap = cm.get_cmap()
    # current_cmap.set_bad(color='white')

    if COLORBAR != False:
        divider = make_axes_locatable(ax)
        cax = divider.append_axes("top", size="5%", pad=0.1)
        cax.xaxis.set_label_position('top')
        if TICKS == None :
            if AUTOCBAR:
                TICKS = MaxNLocator(NTICKS).tick_values(VMIN, VMAX)
            else:    
                TICKS = LinearLocator(NTICKS).tick_values(VMIN, VMAX)
        cax.tick_params(labelsize=10)
        plt.colorbar(imgPLT, cax=cax,ticks=TICKS, orientation='horizontal',ticklocation='top')
        if np.abs(VMIN) >= 1:
            plt.colorbar(imgPLT, cax=cax,ticks=TICKS, orientation='horizontal',ticklocation='top')
        if np.abs(VMIN) <= 0.1:
            plt.colorbar(imgPLT, cax=cax,ticks=TICKS, orientation='horizontal',ticklocation='top', format='%.0e')
        plt.sca(ax)  # Activate main plot before returning
    ax.set_facecolor('black')
    
    if SKIPTICK != True :
        ax.minorticks_on()
        ax.tick_params(axis='x',which='major', length=10, width=1, direction='inout',labelsize=11,
                      bottom=True, top=False, left=True, right=True,color='black')
        ax.tick_params(axis='y',which='major', length=10, width=1, direction='inout',labelsize=11,
                      bottom=True, top=False, left=True, right=True,color='black')
        ax.tick_params(which='minor', length=5, width=1, direction='inout',
                      bottom=True, top=False, left=True, right=True,color='black')
    return

##############################################################################
# other functions
##############################################################################

# estimate the errors in logarithm from linear errors
def lgerr(x1,x2,x1err,x2err,):
    yd0 = x1/x2
    yd = np.log10(yd0)
    yderr0 = ((x1err/x2)**2+(x2err*x1/x2**2)**2)**0.5
    lgyerrup = np.log10(yd0+yderr0) - yd
    lgyerrlow = yd - np.log10(yd0-yderr0)
    return lgyerrlow,lgyerrup


def clean_mask(dataIN,BAD=1e99):
    dataOUT = copy.deepcopy(dataIN)
    dataOUT[dataIN != BAD] = 1
    dataOUT[dataIN == BAD] = np.nan
    # breakpoint()
    return dataOUT 

def snr_cut(dataIN,errIN,SNRCUT=2):
    snr = dataIN/errIN
    gud_indx = np.where(snr >= SNRCUT)
    bad_indx = np.where(snr < SNRCUT)
    dataOUT = copy.deepcopy(dataIN)
    dataOUT[snr < SNRCUT] = np.nan 
    return dataOUT,gud_indx,bad_indx

def save_to_fits(dataIN,hdrIN,savepath):
    # if not os.path.isfile(savepath):
        # shutil.copy(datpath1,savepath)
    if hdrIN == None or hdrIN == []:
        hdu_0 = fits.PrimaryHDU(dataIN)
    else:
        hdu_0 = fits.PrimaryHDU(dataIN,header=hdrIN)
    hdul = fits.HDUList([hdu_0])
    hdul.writeto(savepath,overwrite=True)
    return

def set_figSize(dim,plotsize):
    # breakpoint()
    dim = np.array(dim)
    xy = [12,14]
    # print(dim,plotsize)
    figSIZE = 5*np.round(np.array(plotsize)/5)[0:2]#*dim
    # figSIZE = [figSIZE[0],figSIZE[0]]
    # print(figSIZE)
    figSIZE = figSIZE/np.min(figSIZE)#*dim
    # print(figSIZE)  
    if dim[0] == dim[1]:
        if figSIZE[0] == figSIZE[1]:
            figSIZE = figSIZE*max(xy)
        if figSIZE[0] < figSIZE[1]:
            figSIZE = figSIZE*min(xy)
        # print(figSIZE)
    elif dim[0] < dim[1]:
        figSIZE = np.ceil(figSIZE*min(xy))
        # print(figSIZE)
        # figSIZE = [figSIZE[1],figSIZE[0]]
        figSIZE = [np.max(figSIZE),np.min(figSIZE)]
        if dim[0] == 1:
            figSIZE[1] = np.ceil(figSIZE[1]/2)
    elif dim[0] > dim[1]:
        figSIZE = np.ceil(figSIZE*max(xy))
        # print(figSIZE)
        # figSIZE = [figSIZE[0],figSIZE[1]]
        figSIZE = [np.min(figSIZE),np.max(figSIZE)]
    figSIZE = np.array(figSIZE).astype(int)
    # print(figSIZE)
    return figSIZE


