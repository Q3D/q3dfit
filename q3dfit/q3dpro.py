#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue May 31 15:31:13 2022

@author: yuyuzo12
"""

import copy as copy
import numpy as np
import os
import q3dfit.q3dutil as q3dutil

from astropy.constants import c
from astropy.cosmology import WMAP9 as cosmo
from astropy.io import fits
from matplotlib import pyplot as plt
from matplotlib import cm
from matplotlib.colors import LogNorm
from matplotlib.ticker import MaxNLocator, LinearLocator
from mpl_toolkits.axes_grid1 import make_axes_locatable
from q3dfit.q3dmath import cmpcvdf
from q3dfit.linelist import linelist

plt.rcParams["font.family"] = "serif"
plt.rcParams["mathtext.fontset"] = "dejavuserif"
plt.rcParams['figure.constrained_layout.use'] = False


class Q3Dpro:
    '''
    Class to process the q3dfit output data

    Attributes
    ----------
    q3dinit : object
        q3dinit object. Added/updated by :method:init.
    target_name : str
        Label from q3dinit. Added/updated by :method:init.
    silent : bool
        Flag to suppress print statements. Default is True. 
        Added/updated by :method:init.
    pix : float
        Plate scale of the data. Added/updated by :method:init.
    bad : float
        Value for bad data. Default is np.nan. Added/updated by :method:init.
    dataDIR : str
        Directory of the data, from q3dinit. Added/updated by :method:init.
    contdat : object
        Continuum data object. Added/updated by :method:init, using 
        :class:ContData.
    linedat : object
        Emission line data object. Added/updated by :method:init, using 
        :class:LineData.
    map_bkg : str
        Background color for the maps. Default is 'white'. Added/updated by 
        :method:init.
    zsys_gas : float
        Systemic redshift of the galaxy. Added/updated by :method:init.
    
    Methods
    -------
    get_lineprop(LINESELECT)
        Get the wavelength and line label of an emission line.
    get_linemap(LINESELECT, APPLYMASK=True, SAVEDATA=False)
    make_linemap(LINESELECT, SNRCUT=5., xyCenter=None, 
        VMINMAX=None, XYSTYLE=None, PLTNUM=1, CMAP=None, VCOMP=None, 
        SAVEDATA=False)
    make_lineratio_map(lineA, lineB, SNRCUT=3, SAVEDATA=False, 
        PLTNUM=5, KPC=False, VMINMAX=[-1,1])
    make_BPT(SNRCUT=3, SAVEDATA=False, PLTNUM=5, KPC=False)
    resort_line_components(dataOUT, NCOMP=None, COMP_SORT=None)

    '''

    def __init__(self, q3di, SILENT=True, NOCONT=False, NOLINE=False,
                 PLATESCALE=0.15, BACKGROUND='white', BAD=np.nan, 
                 zsys_gas=None):
        '''
        Initialize the Q3Dpro object.

        Parameters
        ----------
        See class attributes.

        Returns
        -------
        None.

        '''
        # read in the q3di file and unpack
        self.q3dinit = q3dutil.get_q3dio(q3di)
        # unpack initproc
        self.target_name = self.q3dinit.name
        self.silent = SILENT
        if self.silent is False:
            print('processing outputs...')
            print('Target name:', self.target_name)

        self.pix = PLATESCALE  # pixel size
        self.bad = BAD
        self.dataDIR = self.q3dinit.outdir
        # instantiate the Continuum (npy) and Emission Line (npz) objects
        if not NOCONT:
            self.contdat = ContData(self.q3dinit)
        if not NOLINE:
            self.linedat = LineData(self.q3dinit)
        self.map_bkg = BACKGROUND

        self.zsys_gas = zsys_gas

        return


    def get_zsys_gas(self):
        '''
        Get the systemic redshift of the galaxy from the zsys_gas attribute
        or q3dinit object, in that order.

        Returns
        -------
        redshift : float
            Systemic redshift of the galaxy.

        '''
        if self.zsys_gas is None:
            if self.q3dinit.zsys_gas is None:
                raise ValueError('Redshift not set in q3dinit or q3dpro ' +
                                 'objects. Please use zsys_gas parameter ' +
                                 'in q3dpro to set the systemic redshift ' +
                                 'of the galaxy for computing line properties.')
            else:
                redshift = self.q3dinit.zsys_gas
        else:
            redshift = self.zsys_gas
        return redshift
    

    def get_lineprop(self, LINESELECT):
        '''
        Get the wavelength and line label of an emission line.

        Parameters
        ----------
        LINESELECT : str
            Name of the line to select.
        
        Returns
        -------
        linewave : float
            Wavelength of the line in microns.
        linename : str
            Label of the line.

        '''
        listlines = linelist(self.linedat.lines, vacuum=self.q3dinit.vacuum)
        ww = np.where(listlines['name'] == LINESELECT)[0]
        linewave = listlines['lines'][ww].value[0]
        linename = listlines['linelab'][ww].value[0]
        # output in MICRON
        return linewave, linename

    def get_linemap(self, LINESELECT, APPLYMASK=True, SAVEDATA=False):
        '''
        Create maps of properties of an emission line.

        Parameters
        ----------
        LINESELECT : str
            Name of the line for which to create maps.
        
        '''
        print('getting line data...',LINESELECT)

        redshift = self.get_zsys_gas()
        ncomp = np.max(self.q3dinit.ncomp[LINESELECT])

        # kpc_arcsec = cosmo.kpc_proper_per_arcmin(redshift).value/60.
        # arckpc = cosmo.kpc_proper_per_arcmin(redshift).value/60.
        # argscheckcomp = self.q3dinit.argscheckcomp
        # xycen = xyCenter # (nx,ny) of the center

        # cid = -1 # index of the component to plot, -1 is the broadest component
        # central wavelength --> need to get from linereader
        wave0,linetext = self.get_lineprop(LINESELECT)

        fluxsum     = self.linedat.get_flux(LINESELECT,FLUXSEL='ftot')['flux']
        fluxsum_err = self.linedat.get_flux(LINESELECT,FLUXSEL='ftot')['fluxerr']
        fsmsk = clean_mask(fluxsum, BAD=self.bad)

        matrix_size = (fluxsum.shape[0],fluxsum.shape[1],ncomp)

        dataOUT = {'Ftot':{'data':fluxsum,
                           'err':fluxsum_err,
                           'name':['F$_{tot}$'],'mask':fsmsk},
                   'Fci' :{'data':np.zeros(matrix_size),'err':np.zeros(matrix_size),'name':[],'mask':np.zeros(matrix_size)},
                   'Sig' :{'data':np.zeros(matrix_size),'err':np.zeros(matrix_size),'name':[],'mask':np.zeros(matrix_size)},
                   'v50' :{'data':np.zeros(matrix_size),'err':np.zeros(matrix_size),'name':[],'mask':np.zeros(matrix_size)},
                   'w80' :{'data':np.zeros(matrix_size),'err':np.zeros(matrix_size),'name':[],'mask':np.zeros(matrix_size)}}

        # EXTRACT COMPONENTS
        for ci in range(0,ncomp) :
            ici = ci+1
            fcl = 'fc'+str(ici)
            iflux = self.linedat.get_flux(LINESELECT,FLUXSEL=fcl)['flux']
            ifler = self.linedat.get_flux(LINESELECT,FLUXSEL=fcl)['fluxerr']
            isigm = self.linedat.get_sigma(LINESELECT,COMPSEL=ici)['sig']
            isger = self.linedat.get_sigma(LINESELECT,COMPSEL=ici)['sigerr']
            iwvcn = self.linedat.get_wave(LINESELECT,COMPSEL=ici)['wav']
            iwver = self.linedat.get_wave(LINESELECT,COMPSEL=ici)['waverr']
            #ireds = redshift[:,:,ci]

            # now process them
            iv50 = ((iwvcn - wave0)/wave0 - redshift)/(1.+redshift)*c.to('km/s').value
            iw80  = isigm*2.563 #w80 linewidth from the velocity dispersion
            # mask out the bad values
            ifmask = np.array(clean_mask(iflux,BAD=self.bad))
            isgmsk = np.array(clean_mask(isigm,BAD=self.bad))
            iwvmsk = np.array(clean_mask(iwvcn,BAD=self.bad))

            # save to the processed matrices
            dataOUT['Fci']['data'][:,:,ci] = iflux#*ifmask
            dataOUT['Sig']['data'][:,:,ci] = isigm#*isgmsk
            dataOUT['v50']['data'][:,:,ci] = iv50#*iwvmsk
            dataOUT['w80']['data'][:,:,ci] = iw80#*isgmsk

            dataOUT['Fci']['err'][:,:,ci]  = ifler#*ifmask
            dataOUT['Sig']['err'][:,:,ci]  = isger#*isgmsk
            dataOUT['v50']['err'][:,:,ci]  = iwver#*ifmask
            dataOUT['w80']['err'][:,:,ci]  = isger#*isgmsk

            dataOUT['Fci']['name'].append('F$_{c'+str(ici)+'}$')
            dataOUT['Sig']['name'].append('$\sigma_{c'+str(ici)+'}$')
            dataOUT['v50']['name'].append('v$_{50,c'+str(ici)+'}$')
            dataOUT['w80']['name'].append('w$_{80,c'+str(ici)+'}$')

            dataOUT['Fci']['mask'][:,:,ci] = ifmask
            dataOUT['Sig']['mask'][:,:,ci] = isgmsk
            dataOUT['v50']['mask'][:,:,ci] = iwvmsk
            dataOUT['w80']['mask'][:,:,ci] = isgmsk

        if APPLYMASK:
            for ditem in dataOUT:
                if len(dataOUT[ditem]['data'].shape) > 2:
                    for ci in range(0,ncomp):
                        dataOUT[ditem]['data'][:,:,ci] = dataOUT[ditem]['data'][:,:,ci]*dataOUT[ditem]['mask'][:,:,ci]
                        dataOUT[ditem]['err'][:,:,ci]  = dataOUT[ditem]['err'][:,:,ci]*dataOUT[ditem]['mask'][:,:,ci]
                else:
                    dataOUT[ditem]['data'] = dataOUT[ditem]['data']*dataOUT[ditem]['mask']
                    dataOUT[ditem]['err']  = dataOUT[ditem]['err']*dataOUT[ditem]['mask']

        return wave0,linetext,dataOUT

    def make_linemap(self, LINESELECT, SNRCUT=5.,
                     xyCenter=None, VMINMAX=None,
                     XYSTYLE=None, PLTNUM=1, CMAP=None,
                     VCOMP=None,
                     SAVEDATA=False):
        '''
        Create maps of properties of an emission line.

        Parameters
        ----------
        LINESELECT : str
            Name of the line for which to create maps.
        SNRCUT : float
            SNR cut for the data. Default is 5.
        xyCenter : list
            Center of the map. Default is None.
        '''

        print('Plotting emission line maps')
        if self.silent is False:
            print('create linemap:', LINESELECT)

        redshift = self.get_zsys_gas()
        ncomp = np.max(self.q3dinit.ncomp[LINESELECT])

        kpc_arcsec = cosmo.kpc_proper_per_arcmin(redshift).value/60.
        # arckpc = cosmo.kpc_proper_per_arcmin(redshift).value/60.
        # argscheckcomp = self.q3dinit.argscheckcomp
        # xycen = xyCenter # (nx,ny) of the center

        # cid = -1 # index of the component to plot, -1 is the broadest component
        # central wavelength --> need to get from linereader
        # wave0,linetext = self.get_lineprop(LINESELECT)

        # --------------------------
        # EXTRACT THE DATA HERE
        # --------------------------
        wave0, linetext, dataOUT = self.get_linemap(LINESELECT, APPLYMASK=True)
        dataOUT, ncomp = self.resort_line_components(dataOUT, NCOMP=ncomp, 
                                                     COMP_SORT=VCOMP)
        # EXTRACT TOTAL FLUX
        # sn_tot = fluxsum/fluxsum_err
        # w80 = sigma*2.563   # w80 linewidth from the velocity dispersion
        # sn = flux/flux_err
        fluxsum     = dataOUT['Ftot']['data']
        fluxsum_err = dataOUT['Ftot']['err']

        fluxsum_snc, gdindx, bdindx = snr_cut(fluxsum, fluxsum_err, 
                                              SNRCUT=SNRCUT)
        matrix_size = (fluxsum.shape[0],fluxsum.shape[1],ncomp)
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
            qsoCenter = [int(np.ceil(matrix_size[0]/2)),
                         int(np.ceil(matrix_size[1]/2))]

        XYtitle = 'Spaxel'
        if XYSTYLE != None and XYSTYLE != False:
            #if XYSTYLE.lower() == 'center':
            xcol = (xgrid-xyCenter[1])
            ycol = (ygrid-xyCenter[0])
            if xyCenter != None :
                qsoCenter = [0, 0]
            if XYSTYLE.lower() == 'kpc':
                kpc_pix = np.median(kpc_arcsec)* self.pix
                xcolkpc = xcol*kpc_pix
                ycolkpc = ycol*kpc_pix
                xcol,ycol = xcolkpc, ycolkpc
                XYtitle = 'Relative distance [kpc]'
        plt.close(PLTNUM)
        figDIM = [ncomp+1, 4]
        figOUT = set_figSize(figDIM, matrix_size)
        fig, ax = plt.subplots(figDIM[0], figDIM[1], dpi=100)
        fig.set_figheight(figOUT[1]+2)  # (12)
        fig.set_figwidth(figOUT[0]-1)  # (14)
        if CMAP == None :
            CMAP = 'YlOrBr_r'
        ici = ''
        i,j=0,0
        for icomp, ipdat in dataOUT.items():
            doPLT = False
            pixVals = ipdat['data']
            ipshape = pixVals.shape
            if VMINMAX != None :
                vminmax = VMINMAX[icomp]
            if icomp == 'Ftot':
                NTICKS = 4
                ici=''
                vticks = \
                    [vminmax[0], np.power(10, np.median([np.log10(vminmax[0]),
                    np.log10(vminmax[1])])), vminmax[1]]
                if 'fluxlog' in VMINMAX :
                    FLUXLOG = VMINMAX['fluxlog']
            else:
                i=1
                if icomp.lower() == 'fci' :
                    j = 0
                    NTICKS = 3
                    vticks = \
                        [vminmax[0], np.power(10, np.median([np.log10(vminmax[0]),
                        np.log10(vminmax[1])])),vminmax[1]]
                    if 'fluxlog' in VMINMAX :
                        FLUXLOG = VMINMAX['fluxlog']
                if icomp.lower() == 'sig':
                    j+=1
                    CMAP = 'YlOrBr_r'
                    NTICKS  = 3
                    vticks = [vminmax[0],(vminmax[0]+vminmax[1])/2.,vminmax[1]]
                    FLUXLOG = False
                elif icomp.lower() == 'v50' :
                    j+=1
                    NTICKS = 5
                    vticks = [vminmax[0],vminmax[0]/2,0,vminmax[1]/2,vminmax[1]]
                    CMAP = 'RdYlBu'
                    CMAP += '_r'
                    FLUXLOG = False
                elif icomp.lower() == 'w80' :
                    j+=1
                    NTICKS = 3
                    vticks = [vminmax[0],(vminmax[0]+vminmax[1])/2.,vminmax[1]]
                    CMAP = 'RdYlBu'
                    CMAP += '_r'
                    FLUXLOG = False
            if j != 0:
                doPLT = False
                fig.delaxes(ax[0,j])
            for ci in range(0,ncomp) :
                ipixVals = []
                if icomp != 'Ftot' and len(ipshape) > 2:
                    doPLT = True
                    i = ci+1
                    ici = '_c'+str(ci+1)
                    ipixVals = pixVals[:,:,ci]
                elif icomp == 'Ftot':
                    doPLT = True
                    if ci > 0 :
                        doPLT = False
                        break
                    else:
                        ipixVals = pixVals
                if doPLT == True:
                    cmap_r = cm.get_cmap(CMAP)
                    # cmap_r.set_bad(color='black')
                    cmap_r.set_bad(color=self.map_bkg)
                    xx, yy = xcol,ycol
                    axi = ax[i,j]
                    if ncomp < 2:
                        axi = ax[i,j]
                    else:
                        axi = ax[i,j]
                    display_pixels_wz(yy, xx, ipixVals, CMAP=CMAP, AX=axi,
                                      COLORBAR=True, PLOTLOG=FLUXLOG,
                                      VMIN=vminmax[0], VMAX=vminmax[1],
                                      TICKS=vticks, NTICKS=NTICKS)
                    if xyCenter != None :
                        axi.errorbar(qsoCenter[0],qsoCenter[1],color='black',mew=1,mfc='red',fmt='*',markersize=15,zorder=2)
                    axi.set_xlabel(XYtitle,fontsize=16)
                    axi.set_ylabel(XYtitle,fontsize=16)
                    axi.set_title(ipdat['name'][ci],fontsize=20,pad=45)
                    # axi.set_ylim([min(xx),np.ceil(max(xx))])
                    # axi.set_xlim([min(yy),np.ceil(max(yy))])
                    if SAVEDATA == True:
                        linesave_name = self.target_name+'_'+LINESELECT+'_'+icomp+ici+'_map.fits'
                        print('Saving line map:',linesave_name)
                        savepath = os.path.join(self.dataDIR,linesave_name)
                        save_to_fits(pixVals,[],savepath)


            # j+=1
        fig.suptitle(self.target_name+' : '+linetext+' maps',fontsize=20,snap=True,
                     horizontalalignment='right')
                     # verticalalignment='center',
                     # fontweight='semibold')
        fig.tight_layout()#pad=0.15,h_pad=0.1)
        if SAVEDATA == True:
            pltsave_name = LINESELECT+'_emlin_map'
            print('Saving figure:',pltsave_name)
            plt.savefig(os.path.join(self.dataDIR,pltsave_name+'.png'),format='png')
            plt.savefig(os.path.join(self.dataDIR,pltsave_name+'.pdf'),format='pdf')
        # fig.subplots_adjust(top=0.88)
        plt.show()
        return

    # def make_contmap(self):
    #     return

    def make_lineratio_map(self, lineA, lineB, SNRCUT=3, SAVEDATA=False, 
                           PLTNUM=5, KPC=False, VMINMAX=[-1,1]):
        # ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
        # first identify the lines, extract fluxes, and apply the SNR cuts
        # ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
        linelist = {lineA:None,lineB:None}

        redshift = self.get_zsys_gas()

        arckpc = cosmo.kpc_proper_per_arcmin(redshift).value/60.
        mshaps = [None,None]
        li = 0
        for lin in self.linedat.lines:
            if lin in linelist:
                ncomp = np.max(self.q3dinit.ncomp[lin])
                # fluxsum     = self.linedat.get_flux(lin,FLUXSEL='ftot')['flux']
                # fluxsum_err = self.linedat.get_flux(lin,FLUXSEL='ftot')['fluxerr']
                wave0, linetext, dataOUT = self.get_linemap(lin, APPLYMASK={})
                istruct = {'wavcen':wave0,'wname':linetext,'data':dataOUT,'snr':{}}
                for ditem in dataOUT:
                    if len(dataOUT[ditem]['data'].shape) > 2:
                        istruct['snr'][ditem] = [[],[],[]]
                        if li == 0:
                            mshaps[1] = dataOUT[ditem]['data'].shape
                            istruct['snr'][ditem][0] = np.zeros(mshaps[1])
                        for ci in range(0,ncomp):
                            i_snc,i_gindx,i_bindx = snr_cut(dataOUT[ditem]['data'][:,:,ci],dataOUT[ditem]['err'][:,:,ci],SNRCUT=SNRCUT)
                            istruct['snr'][ditem][0][:,:,ci] = i_snc
                            istruct['snr'][ditem][1].append(i_gindx)
                            istruct['snr'][ditem][2].append(i_bindx)
                        li+=0
                    else:
                        if li == 0:
                            mshaps[0] = dataOUT[ditem]['data'].shape
                        i_snc,i_gindx,i_bindx = snr_cut(dataOUT[ditem]['data'],dataOUT[ditem]['err'],SNRCUT=SNRCUT)
                        istruct['snr'][ditem] = [i_snc,i_gindx,i_bindx]
                linelist[lin] = istruct

        lineratios = {'dothis':{'lines':[lineA,lineB],'pltname':linelist[lineA]['wname']+'/'+linelist[lineB]['wname'],'lrat':{'Ftot':None,'Fci':None}}}

        # ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
        # Calculate the line ratios
        # ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
        print('calculating line ratios...')
        cntr = 0
        for lrat,lratdat in lineratios.items():
            f1tot,f1tot_err,m1tot,f1scntot,f1ci,f1ci_err,m1ci,f1scnci = [],[],[],[],[],[],[],[]#np.zeros(mshaps[1]),np.zeros(mshaps[1]),np.zeros(mshaps[1])
            f2tot,f2tot_err,m2tot,f2scntot,f2ci,f2ci_err,m2ci,f2scnci = [],[],[],[],[],[],[],[]#np.zeros(mshaps[1]),np.zeros(mshaps[1]),np.zeros(mshaps[1])
            fratdat = [{'Ftot':[f1tot,f1tot_err,m1tot,f1scntot],'Fci':[f1ci,f1ci_err,m1ci,f1scnci]},
                       {'Ftot':[f2tot,f2tot_err,m2tot,f2scntot],'Fci':[f2ci,f2ci_err,m2ci,f2scnci]}]
            for li,lin in enumerate(lratdat['lines']):
                if isinstance(lin,list):
                    nogood = 0
                    for pp in range(0,4):
                        fratdat[li]['Ftot'][pp] = np.zeros(mshaps[0])
                        fratdat[li]['Fci'][pp]  = np.zeros(mshaps[1])
                    fratdat[li]['Ftot'][2] +=1
                    fratdat[li]['Fci'][2] +=1
                    for jlin in lin :
                        if linelist[jlin] == None:
                            nogood+=1
                        else:
                            lij_datOUT = linelist[jlin]['data']
                            lij_snrOUT = linelist[jlin]['snr']
                            lij_ftot,lij_ftotER,lij_ftotMASK = lij_datOUT['Ftot']['data'],lij_datOUT['Ftot']['err'],lij_datOUT['Ftot']['mask']
                            lij_fci,lij_fciER,lij_fciMASK    = lij_datOUT['Fci']['data'],lij_datOUT['Fci']['err'],lij_datOUT['Fci']['mask']
                            fratdat[li]['Ftot'][0] += lij_ftot
                            fratdat[li]['Ftot'][1] += lij_ftotER
                            fratdat[li]['Ftot'][2] *= lij_ftotMASK
                            fratdat[li]['Ftot'][3] += lij_snrOUT['Ftot'][0]
                            fratdat[li]['Fci'][0] += lij_fci
                            fratdat[li]['Fci'][1] += lij_fciER
                            fratdat[li]['Fci'][2] *= lij_fciMASK
                            fratdat[li]['Fci'][3] += lij_snrOUT['Fci'][0]
                    if nogood == len(lin):
                        lineratios[lrat] = None
                        break
                elif linelist[lin] == None:
                    lineratios[lrat] = None
                    break
                else:
                    li_datOUT = linelist[lin]['data']
                    li_snrOUT = linelist[lin]['snr']
                    li_ftot,li_ftotER,li_ftotMASK = li_datOUT['Ftot']['data'],li_datOUT['Ftot']['err'],li_datOUT['Ftot']['mask']
                    li_fci,li_fciER,li_fciMASK    = li_datOUT['Fci']['data'],li_datOUT['Fci']['err'],li_datOUT['Fci']['mask']
                    fratdat[li]['Ftot'][0] = li_ftot
                    fratdat[li]['Ftot'][1] = li_ftotER
                    fratdat[li]['Ftot'][2] = li_ftotMASK
                    fratdat[li]['Ftot'][3] = li_snrOUT['Ftot'][0]
                    fratdat[li]['Fci'][0] = li_fci
                    fratdat[li]['Fci'][1] = li_fciER
                    fratdat[li]['Fci'][2] = li_fciMASK
                    fratdat[li]['Fci'][3] = li_snrOUT['Fci'][0]
            if lineratios[lrat] != None:
                cntr +=1
                for fi,lratF in enumerate(lratdat['lrat'] ):
                    # print(fratdat[0])
                    fi_mask = fratdat[0][lratF][2]*fratdat[1][lratF][2]
                    fi_frat = fratdat[0][lratF][3]/fratdat[1][lratF][3]
                    frat10 = np.log10(fi_frat)*fi_mask
                    frat10err = lgerr(fratdat[0][lratF][3],fratdat[1][lratF][3],
                                      fratdat[0][lratF][1],fratdat[1][lratF][1])
                    lineratios[lrat]['lrat'][lratF]=[frat10,frat10err]

        # ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
        # Do the plotting here
        # ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
        # if cntr != 0 and bptc != 0:
        nps = mshaps[1][2]+1
        xyCenter = [int(np.ceil(mshaps[1][0]/2)),int(np.ceil(mshaps[1][1]/2))]
        xgrid = np.arange(0, mshaps[1][1])
        ygrid = np.arange(0, mshaps[1][0])
        xcol = xgrid
        ycol = ygrid
        if KPC == True :
            xcol = (xgrid-xyCenter[1])
            ycol = (ygrid-xyCenter[0])
        # --------------------------
        # Plot ine ratio map
        # --------------------------
        plt.close(PLTNUM)
        figDIM = [1,nps]
        figOUT = set_figSize(figDIM,mshaps[1])
        fig,ax = plt.subplots(1,nps,num=PLTNUM,dpi=100)#, gridspec_kw={'height_ratios': [1, 2]})
        fig.set_figheight(figOUT[1])
        fig.set_figwidth(figOUT[0])

        CMAP='inferno'
        cmap_r = cm.get_cmap(CMAP)
        cmap_r.set_bad(color=self.map_bkg)
        cf = 0

        for linrat in lineratios:
            if lineratios[linrat] != None :
                xx,yy = xcol,ycol
                for ni in range(0,nps):
                    ax[ni].set_xlabel('spaxel',fontsize=13)
                    ax[ni].set_ylabel('spaxel',fontsize=13)
                    if KPC == True:
                        ax[ni].set_xlabel('Relative distance [kpc]',fontsize=13)
                        ax[ni].set_ylabel('Relative distance [kpc]',fontsize=13)
                    prelud = ''
                    if ni == 0:
                        prelud = 'Ftot: '
                    else:
                        prelud = 'Fc'+str(ni)+': '
                    ax[ni].set_title(prelud+'log$_{10}$ '+lineratios[linrat]['pltname'],fontsize=15,pad=45)
                    # ax[ni].set_ylim([min(xx),np.ceil(max(xx))])
                    # ax[ni].set_xlim([min(yy),np.ceil(max(yy))])
                for li,lratF in enumerate(['Ftot','Fci']):
                    if lratF == 'Fci':
                        frat10,frat10err = lineratios[linrat]['lrat'][lratF][0],lineratios[linrat]['lrat'][lratF][1]
                        for ci in range(0,ncomp):
                            display_pixels_wz(yy, xx, frat10[:,:,ci], CMAP=CMAP, AX=ax[li+ci],
                                              VMIN=VMINMAX[0], VMAX=VMINMAX[1], NTICKS=5, COLORBAR=True)
                    else:
                        frat10,frat10err = lineratios[linrat]['lrat'][lratF][0],lineratios[linrat]['lrat'][lratF][1]
                        display_pixels_wz(yy, xx, frat10, CMAP=CMAP, AX=ax[li],
                                          VMIN=VMINMAX[0], VMAX=VMINMAX[1], NTICKS=5, COLORBAR=True)
                # pltname,pltrange = lineratios[linrat]['pltname'],lineratios[linrat]['pltrange']
                cf += 1
        plt.tight_layout(pad=1.5,h_pad=0.1)
        if SAVEDATA == True:
            pltsave_name = 'emlin_ratio_map.png'
            print('Saving figure:',pltsave_name)
            plt.savefig(os.path.join(self.dataDIR,pltsave_name))
        plt.show()
        return


    def make_BPT(self, SNRCUT=3, SAVEDATA=False, PLTNUM=5, KPC=False):
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

        redshift = self.get_zsys_gas()

        arckpc = cosmo.kpc_proper_per_arcmin(redshift).value/60.
        mshaps = [None,None]
        li = 0
        for lin in self.linedat.lines:
            if lin in BPTlines:
                ncomp = np.max(self.q3dinit.ncomp[lin])
                # fluxsum     = self.linedat.get_flux(lin,FLUXSEL='ftot')['flux']
                # fluxsum_err = self.linedat.get_flux(lin,FLUXSEL='ftot')['fluxerr']
                wave0, linetext, dataOUT = self.get_linemap(lin, APPLYMASK={})
                istruct = {'wavcen':wave0,'wname':linetext,'data':dataOUT,'snr':{}}
                for ditem in dataOUT:
                    if len(dataOUT[ditem]['data'].shape) > 2:
                        istruct['snr'][ditem] = [[],[],[]]
                        if li == 0:
                            mshaps[1] = dataOUT[ditem]['data'].shape
                            istruct['snr'][ditem][0] = np.zeros(mshaps[1])
                        for ci in range(0,ncomp):
                            i_snc,i_gindx,i_bindx = snr_cut(dataOUT[ditem]['data'][:,:,ci],dataOUT[ditem]['err'][:,:,ci],SNRCUT=SNRCUT)
                            istruct['snr'][ditem][0][:,:,ci] = i_snc
                            istruct['snr'][ditem][1].append(i_gindx)
                            istruct['snr'][ditem][2].append(i_bindx)
                        li+=0
                    else:
                        if li == 0:
                            mshaps[0] = dataOUT[ditem]['data'].shape
                        i_snc,i_gindx,i_bindx = snr_cut(dataOUT[ditem]['data'],dataOUT[ditem]['err'],SNRCUT=SNRCUT)
                        istruct['snr'][ditem] = [i_snc,i_gindx,i_bindx]
                BPTlines[lin] = istruct

               #redshift = self.q3dinit.zinit_gas[lin]
               #matrix_size = redshift.shape
               #redshift = redshift.reshape(matrix_size[0],matrix_size[1])
               #arckpc = cosmo.kpc_proper_per_arcmin(redshift).value/60.
               # BPTlines[lin] = [fluxsum,fluxsum_err,flux_snc,fmask,gud_indx]

        lineratios = {'OiiiHb':{'lines':['[OIII]5007','Hbeta'],'pltname':'[OIII]/H$\\beta$','pltrange':[-1,1.5],
                                'lrat':{'Ftot':None,'Fci':None}},
                        'SiiHa' :{'lines':[['[SII]6716','[SII]6731'],'Halpha'],'pltname':'[SII]/H$\\alpha$','pltrange':[-1.8,0.9],
                                  'lrat':{'Ftot':None,'Fci':None}},
                        'OiHa'  :{'lines':['[OI]6300','Halpha'],'pltname':'[OI]/H$\\alpha$','pltrange':[-1.8,0.1],
                                  'lrat':{'Ftot':None,'Fci':None}},
                       'NiiHa' :{'lines':['[NII]6583','Halpha'],'pltname':'[NII]/H$\\alpha$','pltrange':[-1.8,0.1],
                                 'lrat':{'Ftot':None,'Fci':None}},
                      }

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
        print('calculating line ratios...')
        cntr = 0
        # bptc = 0
        for lrat,lratdat in lineratios.items():
            f1tot,f1tot_err,m1tot,f1scntot,f1ci,f1ci_err,m1ci,f1scnci = [],[],[],[],[],[],[],[]#np.zeros(mshaps[1]),np.zeros(mshaps[1]),np.zeros(mshaps[1])
            f2tot,f2tot_err,m2tot,f2scntot,f2ci,f2ci_err,m2ci,f2scnci = [],[],[],[],[],[],[],[]#np.zeros(mshaps[1]),np.zeros(mshaps[1]),np.zeros(mshaps[1])
            fratdat = [{'Ftot':[f1tot,f1tot_err,m1tot,f1scntot],'Fci':[f1ci,f1ci_err,m1ci,f1scnci]},
                       {'Ftot':[f2tot,f2tot_err,m2tot,f2scntot],'Fci':[f2ci,f2ci_err,m2ci,f2scnci]}]
            for li,lin in enumerate(lratdat['lines']):
                if isinstance(lin,list):
                    nogood = 0
                    for pp in range(0,4):
                        fratdat[li]['Ftot'][pp] = np.zeros(mshaps[0])
                        fratdat[li]['Fci'][pp]  = np.zeros(mshaps[1])
                    fratdat[li]['Ftot'][2] +=1
                    fratdat[li]['Fci'][2] +=1
                    for jlin in lin :
                        if BPTlines[jlin] == None:
                            nogood+=1
                        else:
                            lij_datOUT = BPTlines[jlin]['data']
                            lij_snrOUT = BPTlines[jlin]['snr']
                            lij_ftot,lij_ftotER,lij_ftotMASK = lij_datOUT['Ftot']['data'],lij_datOUT['Ftot']['err'],lij_datOUT['Ftot']['mask']
                            lij_fci,lij_fciER,lij_fciMASK    = lij_datOUT['Fci']['data'],lij_datOUT['Fci']['err'],lij_datOUT['Fci']['mask']
                            fratdat[li]['Ftot'][0] += lij_ftot
                            fratdat[li]['Ftot'][1] += lij_ftotER
                            fratdat[li]['Ftot'][2] *= lij_ftotMASK
                            fratdat[li]['Ftot'][3] += lij_snrOUT['Ftot'][0]
                            fratdat[li]['Fci'][0] += lij_fci
                            fratdat[li]['Fci'][1] += lij_fciER
                            fratdat[li]['Fci'][2] *= lij_fciMASK
                            fratdat[li]['Fci'][3] += lij_snrOUT['Fci'][0]
                    if nogood == len(lin):
                        lineratios[lrat] = None
                        break
                elif BPTlines[lin] == None:
                    lineratios[lrat] = None
                    break
                else:
                    li_datOUT = BPTlines[lin]['data']
                    li_snrOUT = BPTlines[lin]['snr']
                    li_ftot,li_ftotER,li_ftotMASK = li_datOUT['Ftot']['data'],li_datOUT['Ftot']['err'],li_datOUT['Ftot']['mask']
                    li_fci,li_fciER,li_fciMASK    = li_datOUT['Fci']['data'],li_datOUT['Fci']['err'],li_datOUT['Fci']['mask']
                    fratdat[li]['Ftot'][0] = li_ftot
                    fratdat[li]['Ftot'][1] = li_ftotER
                    fratdat[li]['Ftot'][2] = li_ftotMASK
                    fratdat[li]['Ftot'][3] = li_snrOUT['Ftot'][0]
                    fratdat[li]['Fci'][0] = li_fci
                    fratdat[li]['Fci'][1] = li_fciER
                    fratdat[li]['Fci'][2] = li_fciMASK
                    fratdat[li]['Fci'][3] = li_snrOUT['Fci'][0]
            if lineratios[lrat] != None:
                cntr +=1
                for fi,lratF in enumerate(lratdat['lrat'] ):
                    # print(fratdat[0])
                    fi_mask = fratdat[0][lratF][2]*fratdat[1][lratF][2]
                    fi_frat = fratdat[0][lratF][3]/fratdat[1][lratF][3]
                    frat10 = np.log10(fi_frat)*fi_mask
                    frat10err = lgerr(fratdat[0][lratF][3],fratdat[1][lratF][3],
                                      fratdat[0][lratF][1],fratdat[1][lratF][1])
                    lineratios[lrat]['lrat'][lratF]=[frat10,frat10err]

        # ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
        # Do the plotting here
        # ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
        # if cntr != 0 and bptc != 0:
        nps = mshaps[1][2]+1
        xyCenter = [int(np.ceil(mshaps[1][0]/2)),int(np.ceil(mshaps[1][1]/2))]
        xgrid = np.arange(0, mshaps[1][1])
        ygrid = np.arange(0, mshaps[1][0])
        xcol = xgrid
        ycol = ygrid
        if KPC == True :
            xcol = (xgrid-xyCenter[1])
            ycol = (ygrid-xyCenter[0])
        # --------------------------
        # Plot line ratio map
        # --------------------------
        plt.close(PLTNUM)
        figDIM = [nps,cntr]
        figOUT = set_figSize(figDIM,mshaps[1])
        fig,ax = plt.subplots(nps,cntr,num=PLTNUM,dpi=100)#, gridspec_kw={'height_ratios': [1, 2]})
        fig.set_figheight(figOUT[1]+1)
        fig.set_figwidth(figOUT[0])

        CMAP='inferno'
        cmap_r = cm.get_cmap(CMAP)
        cmap_r.set_bad(color=self.map_bkg)
        cf = 0

        for linrat in lineratios:
            if lineratios[linrat] != None :
                xx,yy = xcol,ycol
                # iax = ax[cf]
                pltname,pltrange = lineratios[linrat]['pltname'],lineratios[linrat]['pltrange']
                for ni in range(0,nps):
                    ax[ni,cf].set_xlabel('spaxel',fontsize=13)
                    ax[ni,cf].set_ylabel('spaxel',fontsize=13)
                    if KPC == True:
                        ax[ni,cf].set_xlabel('Relative distance [kpc]',fontsize=13)
                        ax[ni,cf].set_ylabel('Relative distance [kpc]',fontsize=13)
                    prelud = ''
                    if ni == 0:
                        prelud = 'Ftot: '
                    else:
                        prelud = 'Fc'+str(ni)+': '
                    ax[ni,cf].set_title(prelud+'log$_{10}$ '+pltname,fontsize=15,pad=45)
                    # ax[ni,cf].set_ylim([max(xx),np.ceil(min(xx))])
                    # ax[ni,cf].set_xlim([min(yy),np.ceil(max(yy))])

                for li,lratF in enumerate(['Ftot','Fci']):
                    if lratF == 'Fci':
                        frat10,frat10err = lineratios[linrat]['lrat'][lratF][0],lineratios[linrat]['lrat'][lratF][1]
                        for ci in range(0,ncomp):
                            display_pixels_wz(yy, xx,
                                              frat10[:,:,ci], CMAP=CMAP, AX=ax[li+ci,cf],
                                              VMIN=-1, VMAX=1,NTICKS=5, COLORBAR=True)
                    else:
                        frat10,frat10err = lineratios[linrat]['lrat'][lratF][0],lineratios[linrat]['lrat'][lratF][1]
                        display_pixels_wz(yy, xx,
                                          frat10, CMAP=CMAP, AX=ax[li,cf],
                                          VMIN=-1, VMAX=1, NTICKS=5, COLORBAR=True)
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
        figDIM = [nps,cntr-1]
        figOUT = set_figSize(figDIM,mshaps[1],SQUARE=True)
        fig,ax = plt.subplots(nps,cntr-1,figsize=((cntr-1)*5,5),num=PLTNUM,dpi=100)
        fig.set_figheight(int(figOUT[1]))
        fig.set_figwidth(figOUT[0])
        cf=0
        pixkpc = self.pix*arckpc
        xgrid = np.arange(0, mshaps[1][1])
        ygrid = np.arange(0, mshaps[1][0])
        xcol = (xgrid-xyCenter[1])
        ycol = (ygrid-xyCenter[0])
        xcolkpc = xcol*np.median(arckpc)* self.pix
        ycolkpc = ycol*np.median(arckpc)* self.pix
        for bpt in BPTmod:
            if bpt != None :
                for li,lratF in enumerate(['Ftot','Fci']):
                    fnames = bpt.split('/')
                    # pltname,pltrange = lineratios[linrat]['pltname'],lineratios[fnames[0]]['pltrange']
                    pltname1,pltrange1 = lineratios[fnames[0]]['pltname'],lineratios[fnames[0]]['pltrange']
                    pltname2,pltrange2 = lineratios[fnames[1]]['pltname'],lineratios[fnames[1]]['pltrange']
                    if lratF == 'Fci':
                        frat10_A,frat10errA = lineratios[fnames[0]]['lrat'][lratF][0],lineratios[fnames[0]]['lrat'][lratF][1]
                        frat10_B,frat10errB = lineratios[fnames[1]]['lrat'][lratF][0],lineratios[fnames[1]]['lrat'][lratF][1]
                        for ci in range(0,ncomp):
                            gg = np.where(~np.isnan(frat10_A[:,:,ci]) & ~np.isnan(frat10_B[:,:,ci]))
                            xfract,yfract = frat10_B[:,:,ci][gg],frat10_A[:,:,ci][gg]
                            xfracterr,yfracterr = [frat10errB[0][:,:,ci][gg].flatten(), frat10errB[1][:,:,ci][gg].flatten()],[frat10errA[0][:,:,ci][gg].flatten(),frat10errA[1][:,:,ci][gg].flatten()]
                            ee = np.where(~np.isnan(xfracterr[0]) & ~np.isnan(xfracterr[1]) &  ~np.isnan(yfracterr[0]) & ~np.isnan(yfracterr[1]))
                            ax[li+ci,cf].errorbar(xfract.flatten()[ee],yfract.flatten()[ee],fmt='.',alpha=0.7,
                                                  color='black',markersize=5,zorder=2)
                            ax[li+ci,cf].errorbar(xfract.flatten()[ee],yfract.flatten()[ee],fmt='.',alpha=0.7,
                                                xerr=[xfracterr[0][ee],xfracterr[1][ee]],yerr=[yfracterr[0][ee],yfracterr[1][ee]],
                                            elinewidth=0.8,ecolor='dodgerblue',
                                            color='black',markersize=0,zorder=2)
                            ax[li+ci,cf].errorbar(np.median(frat10_B[gg[0],gg[1],ci].flatten()),np.median(frat10_A[gg[0],gg[1],ci].flatten()),
                                                  fillstyle='none',color='red',fmt='*',markersize=15,mew=2,zorder=3)
                    else:
                        frat10_A,frat10errA = lineratios[fnames[0]]['lrat'][lratF][0],lineratios[fnames[0]]['lrat'][lratF][1]
                        frat10_B,frat10errB = lineratios[fnames[1]]['lrat'][lratF][0],lineratios[fnames[1]]['lrat'][lratF][1]
                        gg = np.where(~np.isnan(frat10_B) & ~np.isnan(frat10_A))
                        xfract,yfract       = frat10_B[gg],frat10_A[gg]
                        xfracterr,yfracterr = [frat10errB[0][gg].flatten(),frat10errB[1][gg].flatten()],[frat10errA[0][gg].flatten(),frat10errA[1][gg].flatten()]
                        ee = np.where(~np.isnan(xfracterr[0]) & ~np.isnan(xfracterr[1])&  ~np.isnan(yfracterr[0]) & ~np.isnan(yfracterr[1]))
                        ax[li,cf].errorbar(xfract.flatten()[ee],yfract.flatten()[ee],fmt='.',alpha=0.7,
                                            color='black',markersize=5,zorder=2)
                        ax[li,cf].errorbar(xfract.flatten()[ee],yfract.flatten()[ee],fmt='.',alpha=0.7,
                                            xerr=[xfracterr[0][ee],xfracterr[1][ee]],yerr=[yfracterr[0][ee],yfracterr[1][ee]],
                                            elinewidth=0.8,ecolor='dodgerblue',
                                            color='black',markersize=0,zorder=2)
                        ax[li,cf].errorbar(np.median(xfract.flatten()),np.median(yfract.flatten()),
                                            fillstyle='none',color='red',fmt='*',markersize=15,mew=2,zorder=3)
                for ni in range(0,nps):
                    ax[ni,cf].set_xlim([-2.1,0.7])
                    ax[ni,cf].set_ylim(pltrange1)
                    # first plot the theoretical curves
                    iBPTmod = BPTmod[bpt]
                    ax[ni,cf].plot(iBPTmod[0][0],iBPTmod[0][1],'k-',zorder=1,linewidth=1.5)
                    ax[ni,cf].plot(iBPTmod[1][0],iBPTmod[1][1],'k--',zorder=1,linewidth=1.5)
                    compName = ''
                    if ni == 0:
                        compName = 'Ftot'
                    else:
                        compName = 'Fc'+str(ni)

                    ax[ni,cf].minorticks_on()
                    if cf == 0:
                        ax[ni,cf].set_ylabel(pltname1+', '+compName,fontsize=16)
                        ax[ni,cf].tick_params(axis='y',which='major', length=10, width=1, direction='in',labelsize=13,
                                      bottom=True, top=True, left=True, right=True,color='black')
                    else:
                        ax[ni,cf].tick_params(axis='y',which='major', length=10, width=1, direction='in',labelsize=0,
                                      bottom=True, top=True, left=True, right=True,color='black')
                    ax[ni,cf].set_xlabel(pltname2,fontsize=16)
                    ax[ni,cf].tick_params(axis='x',which='major', length=10, width=1, direction='in',labelsize=13,
                                  bottom=True, top=True, left=True, right=True,color='black')
                    ax[ni,cf].tick_params(which='minor', length=5, width=1, direction='in',
                                  bottom=True, top=True, left=True, right=True,color='black')
                cf+=1
        plt.tight_layout(pad=1.5,h_pad=0.1)
        if SAVEDATA == True:
            pltsave_name = 'BPT_map.png'
            print('Saving figure:',pltsave_name)
            plt.savefig(os.path.join(self.dataDIR,pltsave_name))
        # plt.savefig(tname+'_c'+str(int(ic+1))+'_bpt.pdf')
        plt.show()
            # breakpoint()
        return

    def resort_line_components(self,dataIN,NCOMP=1,COMP_SORT=None):

        if COMP_SORT == None:
            return dataIN,NCOMP
        else:
            if COMP_SORT['sort_by'] in dataIN.keys() :
                if 'sort_range' not in COMP_SORT:
                    print('sort comp by:',COMP_SORT['sort_by'])
                    dataOUT = copy.deepcopy(dataIN)
                    sortDat = dataIN[COMP_SORT['sort_by']]
                    mshap = sortDat['data'].shape
                    for ii in range (0,mshap[0]):
                        for jj in range (0,mshap[1]):
                            sij = np.argsort(np.abs(sortDat['data'][ii,jj,:]))
                            dataOUT['Fci']['data'][ii,jj,:] = dataIN['Fci']['data'][ii,jj,sij]
                            dataOUT['Sig']['data'][ii,jj,:] = dataIN['Sig']['data'][ii,jj,sij]
                            dataOUT['v50']['data'][ii,jj,:] = dataIN['v50']['data'][ii,jj,sij]
                            dataOUT['w80']['data'][ii,jj,:] = dataIN['w80']['data'][ii,jj,sij]
                            dataOUT['Fci']['err'][ii,jj,:]  = dataIN['Fci']['err'][ii,jj,sij]
                            dataOUT['Sig']['err'][ii,jj,:]  = dataIN['Sig']['err'][ii,jj,sij]
                            dataOUT['v50']['err'][ii,jj,:]  = dataIN['v50']['err'][ii,jj,sij]
                            dataOUT['w80']['err'][ii,jj,:]  = dataIN['w80']['err'][ii,jj,sij]
                            dataOUT['Fci']['mask'][ii,jj,:] = dataIN['Fci']['mask'][ii,jj,sij]
                            dataOUT['Sig']['mask'][ii,jj,:] = dataIN['Sig']['mask'][ii,jj,sij]
                            dataOUT['v50']['mask'][ii,jj,:] = dataIN['v50']['mask'][ii,jj,sij]
                            dataOUT['w80']['mask'][ii,jj,:] = dataIN['w80']['mask'][ii,jj,sij]
                    return dataOUT,mshap[2]
                else:
                    sort_rang = COMP_SORT['sort_range']
                    print('========================')
                    print('sort comp by:',COMP_SORT['sort_by'])
                    for si,srang in enumerate(sort_rang):
                        print('c'+str(si+1),srang)
                    sortDat = dataIN[COMP_SORT['sort_by']]
                    mshap = sortDat['data'].shape
                    dataOUT = {'Ftot':dataIN['Ftot'],
                               'Fci':{'data':None,'err':None,'name':None,'mask':None},
                               'Sig':{'data':None,'err':None,'name':None,'mask':None},
                               'v50':{'data':None,'err':None,'name':None,'mask':None},
                               'w80':{'data':None,'err':None,'name':None,'mask':None}}
                    for ditem in dataOUT:
                        if ditem != 'Ftot':
                            dataOUT[ditem]['data'] = np.zeros((mshap[0],mshap[1],len(sort_rang)))+np.nan
                            dataOUT[ditem]['err']  = np.zeros((mshap[0],mshap[1],len(sort_rang)))+np.nan
                            dataOUT[ditem]['name'] = []
                            dataOUT[ditem]['mask'] = np.zeros((mshap[0],mshap[1],len(sort_rang)))+np.nan

                    for ii in range (0,mshap[0]):
                        for jj in range (0,mshap[1]):
                            for cc in range(0,mshap[2]):
                                for sri,sr in enumerate(sort_rang):
                                    # dataOUT['Fci']['name'].append()
                                    ici = sri
                                    dataOUT['Fci']['name'].append('F$_{c'+str(ici)+'}$')
                                    dataOUT['Sig']['name'].append('$\sigma_{c'+str(ici)+'}$')
                                    dataOUT['v50']['name'].append('v$_{50,c'+str(ici)+'}$')
                                    dataOUT['w80']['name'].append('w$_{80,c'+str(ici)+'}$')
                                    if sr[0] <= sortDat['data'][ii,jj,cc] <= sr[1]:
                                        dataOUT['Fci']['data'][ii,jj,sri] = dataIN['Fci']['data'][ii,jj,cc]
                                        dataOUT['Sig']['data'][ii,jj,sri] = dataIN['Sig']['data'][ii,jj,cc]
                                        dataOUT['v50']['data'][ii,jj,sri] = dataIN['v50']['data'][ii,jj,cc]
                                        dataOUT['w80']['data'][ii,jj,sri] = dataIN['w80']['data'][ii,jj,cc]
                                        dataOUT['Fci']['err'][ii,jj,sri]  = dataIN['Fci']['err'][ii,jj,cc]
                                        dataOUT['Sig']['err'][ii,jj,sri]  = dataIN['Sig']['err'][ii,jj,cc]
                                        dataOUT['v50']['err'][ii,jj,sri]  = dataIN['v50']['err'][ii,jj,cc]
                                        dataOUT['w80']['err'][ii,jj,sri]  = dataIN['w80']['err'][ii,jj,cc]
                                        dataOUT['Fci']['mask'][ii,jj,sri] = dataIN['Fci']['mask'][ii,jj,cc]
                                        dataOUT['Sig']['mask'][ii,jj,sri] = dataIN['Sig']['mask'][ii,jj,cc]
                                        dataOUT['v50']['mask'][ii,jj,sri] = dataIN['v50']['mask'][ii,jj,cc]
                                        dataOUT['w80']['mask'][ii,jj,sri] = dataIN['w80']['mask'][ii,jj,cc]
                                # print('---')
                    return dataOUT,len(sort_rang)
            else:
                print('SORT ERROR...')
                pass



# =============================================================================================
# reading in the .npy and .npz files
# =============================================================================================
# Emission line data
# ---------------------------------------------------------------------------------------------


class LineData:
    '''
    Read in and store line data for all lines found in *.lin.npz file
    (which is output by q3da).

    Individual line measurements can be obtained with corresponding methods.

    Parameters
    -----------
    q3di : object

    Attributes
    ----------
    lines : dict
        Line names.
    maxncomp : int
        Maximum number of components fit to a line.
    data : dict
        Contents of the line data (.npz) file.
    Examples
    --------
    >>>

    Notes
    -----
    '''

    def __init__(self, q3di):

        filename = q3di.label+'.line.npz'
        datafile = os.path.join(q3di.outdir, filename)
        # print(datafile)
        if not os.path.exists(datafile):
            print('ERROR: emission line ('+filename+') file does not exist')
            return
        self.lines = q3di.lines
        self.maxncomp = q3di.maxncomp
        self.data = self.read_npz(datafile)
        # book-keeping inheritance from initproc
        self.ncols = self.data['ncols'].item()
        self.nrows = self.data['nrows'].item()
        self.bad = np.nan
        self.dataDIR = q3di.outdir
        self.target_name = q3di.name
        # self.flux    = self.get_flux()
        # self.siga    = self.get_sigma()
        # self.wavelen = self.get_wave()
        # self.eq      = self.get_weq()
        return

    def read_npz(self, datafile):
        ''' Load binary line data file.

        Parameters
        ----------
        datafile : str

        Returns
        -------
        Contents of datafile.

        '''
        dataread = np.load(datafile, allow_pickle=True)
        self.colname = sorted(dataread)
        return dataread

    def get_flux(self, lineselect, FLUXSEL='ftot'):
        ''' Get flux and error of a given line.

        Parameters
        ----------
        lineselect : str
            Which line to grab.
        fluxsel : str, default 'ftot' (total flux)
            Which flux to grab. String names defined in q3da.

        Returns
        -------
        dict
            keys flux, fluxerr contain ndarray(ncols, nrows, ncomp)

        '''

        # FLUXSEL = 'ftot' by default --> select from ('ftot', 'fc1', 'fc1pk')
        if lineselect not in self.lines:
            print('ERROR: line does not exist')
            return None
        emlflx = self.data['emlflx'].item()
        emlflxerr = self.data['emlflxerr'].item()
        dataout = {'flux': emlflx[FLUXSEL][lineselect],
                   'fluxerr': emlflxerr[FLUXSEL][lineselect]}
        return dataout

    def get_ncomp(self, lineselect):
        ''' Get # components fit to a given line.

        Parameters
        ----------
        lineselect : str

        Returns
        -------
        ndarray(ncols, nrows)

        '''
        if lineselect not in self.lines:
            print('ERROR: line does not exist')
            return None
        return (self.data['emlncomp'].item())[lineselect]

    def get_sigma(self, lineselect, COMPSEL=1):
        ''' Get sigma and error of a given line and component.

        Parameters
        ----------
        lineselect : str
            Which line to grab.
        compsel : int, default 1

        Returns
        -------
        dict
            keys sig, sigerr contain ndarray(ncols, nrows)

        '''
        if lineselect not in self.lines:
            print('ERROR: line does not exist')
            return None
        # 'c1'
        emlsig = self.data['emlsig'].item()
        emlsigerr = self.data['emlsigerr'].item()
        csel = 'c'+str(COMPSEL)
        dataout = {'sig': emlsig[csel][lineselect],
                   'sigerr': emlsigerr[csel][lineselect]}
        return dataout

    def get_wave(self, lineselect, COMPSEL=1):
        ''' Get central wavelength and error of a given line and component.

        Parameters
        ----------
        lineselect : str
            Which line to grab.
        compsel : int, default 1

        Returns
        -------
        dict
            keys wav, waverr contain ndarray(ncols, nrows)

        '''
        if lineselect not in self.lines:
            print('ERROR: line does not exist')
            return None
        # 'c1'
        emlwav = self.data['emlwav'].item()
        emlwaverr = self.data['emlwaverr'].item()
        csel = 'c'+str(COMPSEL)
        dataout = {'wav': emlwav[csel][lineselect],
                   'waverr': emlwaverr[csel][lineselect]}
        return dataout

    # def get_weq(self, lineselect, FLUXSEL='ftot'):
    #     # FLUXSEL = 'ftot' by default --> select from ('ftot', 'fc1')
    #     if lineselect not in self.lines:
    #         print('ERROR: line does not exist')
    #         return None
    #     # 'ftot', 'fc1'
    #     emlweq = self.data['emlweq'].item()
    #     dataout = emlweq[FLUXSEL][lineselect]
    #     return dataout


class OneLineData:
    '''
    Parse all line data for a given emission line from a LineData object.

    Parameters
    -----------
    linedata :
        Data from *.lin.npz file, stored in LineData.data.

    Attributes
    ----------
    flux : ndarray(ncols, nrows, maxncomp)
    fpklux : ndarray(ncols, nrows, maxncomp)
    line : str
    ncomp : ndarray(ncols, nrows)
    sig : ndarray(ncols, nrows, maxncomp)
    wave : ndarray(ncols, nrows, maxncomp)

    Examples
    --------
    >>>

    Notes
    -----
    '''

    def __init__(self, linedata, lineselect):
        # inherit some stuff from linedata
        self.ncols = linedata.ncols
        self.nrows = linedata.nrows
        self.bad = linedata.bad
        self.dataDIR = linedata.dataDIR
        self.target_name = linedata.target_name

        self.line = lineselect
        # initialize arrays
        self.flux = \
            np.zeros((linedata.ncols, linedata.nrows, linedata.maxncomp),
                     dtype=float) + linedata.bad
        self.pkflux = \
            np.zeros((linedata.ncols, linedata.nrows, linedata.maxncomp),
                     dtype=float) + linedata.bad
        self.sig = \
            np.zeros((linedata.ncols, linedata.nrows, linedata.maxncomp),
                     dtype=float) + linedata.bad
        self.wave = \
            np.zeros((linedata.ncols, linedata.nrows, linedata.maxncomp),
                     dtype=float) + linedata.bad
        # cycle through components to get maps
        for i in range(0, linedata.maxncomp):
            self.flux[:, :, i] = \
                (linedata.get_flux(lineselect, FLUXSEL='fc'+str(i+1)))['flux']
            self.pkflux[:, :, i] = \
                (linedata.get_flux(lineselect,
                                   FLUXSEL='fc'+str(i+1)+'pk'))['flux']
            self.sig[:, :, i] = \
                (linedata.get_sigma(lineselect, COMPSEL=i+1))['sig']
            self.wave[:, :, i] = \
                (linedata.get_wave(lineselect, COMPSEL=i+1))['wav']
        # No. of components on a spaxel-by-spaxel basis
        self.ncomp = linedata.get_ncomp(lineselect)

    def calc_cvdf(self, zref, vlimits=[-1e4, 1e4], vstep=1.):
        '''
        Compute CVDF for this line, for each spaxel.

        Parameters
        -----------
        zref : float
            Reference redshift for computing velocities
        vlimits : ndarray(2)
            limits for model velocities, in km/s
        vstep : float
            step for model velocities, in km/s

        Attributes
        ----------
        cvdf_vel : ndarray(nmod)
            Model velocities.
        vdf : ndarray(ncols, nrows, nmod)
            Velocity distribution in flux space.
        cvdf : ndarray(ncols, nrows, nmod)
            Cumulative velocity distribution function.

        '''
        self.cvdf_zref = zref
        self.cvdf_vel, self.vdf, self.cvdf = \
            cmpcvdf(self.wave, self.sig, self.pkflux, self.ncomp,
                    self.line, zref, vlimits=vlimits, vstep=vstep)
        self.cvdf_nmod = len(self.cvdf_vel)

    def calc_cvdf_vel(self, pct, calc_from_posvel=True):
        '''
        Compute a velocity at % pct from one side of the CVDF.

        Parameters
        -----------
        pct : float
            Percentage at which to calculate velocity.
        calc_from_posvel : bool
            If True, v0% is at positive velocities.

        Returns
        -------
        ndarray(ncols, nrows)

        '''

        if calc_from_posvel:
            pct_use = 100. - pct
        else:
            pct_use = pct

        varr = np.zeros((self.ncols, self.nrows), dtype=float) + self.bad
        for i in range(self.ncols):
            for j in range(self.nrows):
                ivel = np.searchsorted(self.cvdf[i, j, :], pct_use/100.)
                if ivel != self.cvdf_nmod:
                    # for now, just interpolate between two points around value
                    varr[i, j] = (self.cvdf_vel[ivel] +
                                  self.cvdf_vel[ivel-1]) / 2.
        return varr

    def make_cvdf_map(self, pct, velran=None, cmap=None,
                      center=None, markcenter=None, axisunit='spaxel',
                      platescale=None, outfile=False, outformat='png'):

        pixVals = self.calc_cvdf_vel(pct)

        kpc_arcsec = cosmo.kpc_proper_per_arcmin(self.cvdf_zref).value/60.

        # single-offset column and row values
        cols = np.arange(1, self.ncols+1, dtype=float)
        rows = np.arange(1, self.nrows+1, dtype=float)
        if center is None:
            center = [0., 0.]
        cols_cent = (cols - center[0])
        rows_cent = (rows - center[1])
        # This makes the axis span the spaxel values, with integer coordinate
        # being a pixel center. So a range of [1,5] spaxels will have an axis
        # range of [0.5,5.5]. This is what the extent keyword to imshow expects.
        # https://matplotlib.org/stable/tutorials/intermediate/imshow_extent.html
        xran = np.array([cols_cent[0], cols_cent[self.ncols-1]+1.]) - 0.5
        yran = np.array([rows_cent[0], rows_cent[self.nrows-1]+1.]) - 0.5

        XYtitle = 'Spaxel'
        #if axisunit == 'kpc' and platescale is not None:
        #    kpc_pix = np.median(kpc_arcsec)*platescale
        #    xcolkpc = xcol*kpc_pix
        #    ycolkpc = ycol*kpc_pix
        #    xcol, ycol = xcolkpc, ycolkpc
        #    XYtitle = 'Relative distance [kpc]'

        fig, ax = plt.subplots()
        if cmap is None:
            cmap = 'RdYlBu_r'

        vticks = [velran[0], velran[0]/2., 0., velran[1]/2., velran[1]]
        nticks = 4
        cmap_r = cm.get_cmap(cmap)
        cmap_r.set_bad(color='black')
        # cmap_r.set_bad(color=self.map_bkg)
        display_pixels_wz(cols_cent, rows_cent, pixVals, CMAP=cmap, AX=ax,
                          COLORBAR=True, VMIN=velran[0], VMAX=velran[1],
                          TICKS=vticks, NTICKS=nticks, XRAN=xran, YRAN=yran)
        if markcenter is not None:
            ax.errorbar(markcenter[0], markcenter[1], color='black', mew=1,
                        mfc='red', fmt='*', markersize=10, zorder=2)
        ax.set_xlabel(XYtitle, fontsize=12)
        ax.set_ylabel(XYtitle, fontsize=12)
        ax.set_title(self.target_name + ' ' + self.line + ' ' + 'v' +
                     str(int(pct)) + ' (km/s)', fontsize=16, pad=45)
        #axi.set_title(ipdat['name'][ci],fontsize=20,pad=45)

            #if savedata:
            #    linesave_name = self.target_name+'_'+LINESELECT+'_'+icomp+ici+'_map.fits'
            #    print('Saving line map:',linesave_name)
            #    savepath = os.path.join(self.dataDIR,linesave_name)
            #            save_to_fits(pixVals,[],savepath)
        #fig.suptitle(self.target_name+' : '+linetext+' maps',fontsize=20,snap=True,
        #             horizontalalignment='right')
        #             # verticalalignment='center',
        #             # fontweight='semibold')
        fig.tight_layout(pad=0.15, h_pad=0.1)
        fig.set_dpi(300.)

        if outfile:
            pltsave_name = self.target_name + '-' + self.line + '-v' + \
                str(int(pct)) + '-map'
            print('Saving ', pltsave_name, ' to ', self.dataDIR)
            plt.savefig(os.path.join(self.dataDIR, pltsave_name + '.' +
                                     outformat), format=outformat, dpi=100.)
        plt.show()

        return


# ---------------------------------------------------------------------------------------------
# Continuum data
# ---------------------------------------------------------------------------------------------
class ContData:
    def __init__(self, q3di):
        filename = q3di.label+'.cont.npy'
        datafile = os.path.join(q3di.outdir,filename)
        if os.path.exists(datafile) != True:
            print('ERROR: continuum ('+filename+') file does not exist')
            return None
        self.data = self.read_npy(datafile)
        self.wave           = self.data['wave']
        self.qso_mod        = self.data['qso_mod']
        self.host_mod       = self.data['host_mod']
        self.poly_mod       = self.data['poly_mod']
        self.npts           = self.data['npts']
        self.stel_sixgma    = self.data['stel_sigma']
        self.stel_sigma_err = self.data['stel_sigma_err']
        self.stel_z         = self.data['stel_z']
        self.stel_z_err     = self.data['stel_z_err']
        self.stel_rchisq    = self.data['stel_rchisq']
        self.stel_ebv       = self.data['stel_ebv']
        self.stel_ebv_err   = self.data['stel_ebv_err']
        return

    def read_npy(self, datafile):
        dataout = np.load(datafile, allow_pickle=True).item()
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


def display_pixels_wz(x, y, datIN, PIXELSIZE=None, VMIN=None, VMAX=None,
                      TICKS=None, PLOTLOG=False, ANGLE=None, COLORBAR=False,
                      AUTOCBAR=False, LABEL=None, NTICKS=3, CMAP='RdYlBu',
                      SKIPTICK=False, AX=None, XRAN=None, YRAN=None):
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
    if XRAN is not None:
        xmin, xmax = XRAN[0], XRAN[1]
    else:
        xmin, xmax = np.ceil(np.min(x)), np.ceil(np.max(x))
        xmax = 5*np.round(np.array(xmax)/5)
    if YRAN is not None:
        ymin, ymax = YRAN[0], YRAN[1]
    else:
        ymin, ymax = np.ceil(np.min(y)), np.ceil(np.max(y))
        ymax = 5*np.round(np.array(ymax)/5)

    imgPLT = None
    if not PLOTLOG:
        imgPLT = AX.imshow(np.rot90(datIN, 1),
                           # origin='lower',
                           cmap=CMAP,
                           extent=[xmin, xmax, ymin, ymax],
                           vmin=VMIN, vmax=VMAX,
                           interpolation='none')
    else:
        imgPLT = AX.imshow(np.rot90(datIN, 1),
                           # origin='lower',
                           cmap=CMAP,
                           extent=[xmin, xmax, ymin, ymax],
                           norm=LogNorm(vmin=VMIN, vmax=VMAX),
                           interpolation='none')

    current_cmap = cm.get_cmap()
    current_cmap.set_bad(color='white')

    if COLORBAR:
        divider = make_axes_locatable(AX)
        cax = divider.append_axes("top", size="5%", pad=0.1)
        cax.xaxis.set_label_position('top')
        if TICKS is None:
            if AUTOCBAR:
                TICKS = MaxNLocator(NTICKS).tick_values(VMIN, VMAX)
            else:
                TICKS = LinearLocator(NTICKS).tick_values(VMIN, VMAX)
        cax.tick_params(labelsize=10)
        # plt.colorbar(imgPLT, cax=cax, ticks=TICKS, orientation='horizontal',
        #              ticklocation='top')
        if np.abs(VMIN) >= 1:
            plt.colorbar(imgPLT, cax=cax, ticks=TICKS,
                         orientation='horizontal', ticklocation='top')
        if np.abs(VMIN) <= 0.1:
            plt.colorbar(imgPLT, cax=cax, ticks=TICKS,
                         orientation='horizontal', ticklocation='top',
                         format='%.0e')
        # cax.formatter.set_powerlimits((0, 0))
        plt.sca(AX)  # Activate main plot before returning
    AX.set_facecolor('black')

    if not SKIPTICK:
        AX.minorticks_on()
        AX.tick_params(axis='x', which='major', length=10, width=1,
                       direction='inout', labelsize=11,
                       bottom=True, top=False, left=True, right=True,
                       color='black')
        AX.tick_params(axis='y', which='major', length=10, width=1,
                       direction='inout', labelsize=11, bottom=True, top=False,
                       left=True, right=True, color='black')
        AX.tick_params(which='minor', length=5, width=1, direction='inout',
                       bottom=True, top=False, left=True, right=True,
                       color='black')
    return

##############################################################################
# other functions
##############################################################################

# estimate the errors in logarithm from linear errors
def lgerr(x1,x2,x1err,x2err,):
    x1,x2,x1err,x2err = np.array(x1),np.array(x2),np.array(x1err),np.array(x2err)
    yd0 = x1/x2
    yd = np.log10(yd0)
    yderr0 = ((x1err/x2)**2+(x2err*x1/x2**2)**2)**0.5
    lgyerrup = np.log10(yd0+yderr0) - yd
    lgyerrlow = yd - np.log10(yd0-yderr0)
    return [lgyerrlow,lgyerrup]

def clean_mask(dataIN, BAD=np.nan):
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
    dataOUT[np.where(np.isnan(errIN))] = np.nan
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

def set_figSize(dim,plotsize,SQUARE=False):
    dim = np.array(dim, dtype='float')
    xy = [12.,14.]
    figSIZE = 5.*np.round(np.array(plotsize, dtype='float')/5.)[0:2]
    if figSIZE[0]==0 and figSIZE[1]==0:
        figSIZE = plotsize
    figSIZE = figSIZE/np.min(figSIZE)
    if dim[0] == dim[1] :
        if figSIZE[0] >= figSIZE[1]:
            figSIZE = figSIZE*max(xy)
        if figSIZE[0] < figSIZE[1]:
            figSIZE = figSIZE*min(xy)
    elif dim[0] < dim[1]:
        figSIZE = np.ceil(figSIZE*min(xy))
        figSIZE = [np.max(figSIZE),np.min(figSIZE)]
        if dim[0] == 1:
            figSIZE[1] = np.ceil(figSIZE[1]/2.)
    elif dim[0] > dim[1]:
        figSIZE = np.ceil(figSIZE*max(xy))
        figSIZE = [np.min(figSIZE),np.max(figSIZE)]
    figSIZE = np.array(figSIZE).astype(int)
    return figSIZE


