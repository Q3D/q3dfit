#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue May 31 15:31:13 2022

@author: yuyuzo12
"""
from __future__ import annotations

from typing import Any, Literal, Optional

import copy as copy
import numpy as np
import os

from numpy.typing import ArrayLike

from astropy.constants import c
from astropy.cosmology import WMAP9 as cosmo
from astropy.io import fits
from matplotlib import pyplot as plt
from matplotlib import cm
from matplotlib.colors import LogNorm
from matplotlib.ticker import MaxNLocator, LinearLocator
from mpl_toolkits.axes_grid1 import make_axes_locatable

from q3dfit.linelist import linelist
from . import q3din, q3dutil

plt.rcParams["font.family"] = "serif"
plt.rcParams["mathtext.fontset"] = "dejavuserif"
plt.rcParams['figure.constrained_layout.use'] = False


class Q3Dpro:
    '''
    Class to process the q3dfit output data.

    Parameters
    ----------
    q3di
        Path to the q3dfit output file or q3din object. Also updates
        :py:attr:`~q3dfit.q3dpro.Q3Dpro.q3dinit`.
    quiet
        Optional. Flag to suppress print statements. Default is True. 
        Also updates :py:attr:`~q3dfit.q3dpro.Q3Dpro.quiet`.
    nocont
        Optional. If False, create :py:class:`~q3dfit.q3dpro.ContData` object
        and assign to :py:attr:`~q3dfit.q3dpro.Q3Dpro.contdat`. Default is False.
    noline
        Optional. If False, create :py:class:`~q3dfit.q3pro.LineData` object
        and assign to :py:attr:`~q3dfit.q3dpro.Q3Dpro.linedat`. Default is False.
    zsys
        Optional. Systemic redshift of the galaxy. Default is None, which
        will use the value from the q3din object. Also updates
        :py:attr:`~q3dfit.q3dpro.Q3Dpro.zsys` and :py:attr:`~q3dfit.q3dpro.Q3Dpro.zsys_gas`.
    zsys_gas
        Optional. Kept for backwards compatibility; zsys is preferred. 
        Default is None. Setting this will update 
        :py:attr:`~q3dfit.q3dpro.Q3Dpro.zsys` and
        :py:attr:`~q3dfit.q3dpro.Q3Dpro.zsys_gas`.

    Attributes
    ----------
    q3dinit : q3din.q3din
        Copy of :py:class:`~q3dfit.q3din.q3din` object. Added/updated by
        :py:meth:`__init__`.
    target_name : str
        Full name of source for plot labels, etc. . Added/updated by :method:init.
    pix : float
        Plate scale of the data. Added/updated by :method:init.
    bad : float
        Value for bad data. Default is np.nan. Added/updated by :method:init.
    dataDIR : str
        Directory of the data, from q3dinit. Added/updated by :method:init.
    contdat : Optional[ContData]
        Continuum data object. Added/updated by :method:init, using 
        :class:ContData. If nocont is True, this will be None.
    linedat : Optional[LineData]
        Emission line data object. Added/updated by :method:init, using 
        :class:LineData. If noline is True, this will be None.
    map_bkg : str
        Background color for the maps. Default is 'white'. Added/updated by 
        :method:init.
    zsys : float
        Systemic redshift of the galaxy. Added/updated by :py:meth:`__init__`.
    zsys_gas : float
        Kept for backwards compatibility; zsys is preferred. Equal to zsys.
        Added/updated by :py:meth:`__init__`.
    '''
    def __init__(self,
                 q3di: str | q3din.q3din,
                 quiet: bool=True,
                 nocont: bool=False,
                 noline: bool=False,
                 platescale: float=0.15,
                 background: str='white',
                 bad: float=np.nan, 
                 zsys: Optional[float]=None,                 
                 zsys_gas: Optional[float]=None,
                 **kwargs):
        
        # take care of old upper case parameters
        if 'NOCONT' in kwargs:
            nocont = kwargs['NOCONT']
        if 'NOLINE' in kwargs:
            noline = kwargs['NOLINE']
        if 'PLATESCALE' in kwargs:
            platescale = kwargs['PLATESCALE']
        if 'BACKGROUND' in kwargs:
            background = kwargs['BACKGROUND']
        if 'BAD' in kwargs:
            bad = kwargs['BAD']

        # read in the q3di file and unpack
        self.q3dinit: q3din.q3din = q3dutil.get_q3dio(q3di)
        # unpack initproc
        self.target_name = self.q3dinit.name
        if self.target_name is None:
            self.target_name = 'NoName'
            q3dutil.write_msg(f'No target name set in q3din. Please set Q3Dpro.target_name.', quiet=quiet)
        self.quiet = quiet
        q3dutil.write_msg(f'Processing outputs.', quiet=quiet)
        q3dutil.write_msg(f'Target name: {self.target_name}', quiet=quiet)

        self.pix = platescale  # pixel size
        self.bad = bad
        self.dataDIR = self.q3dinit.outdir
        # instantiate the Continuum (npy) and Emission Line (npz) objects
        if not nocont:
            self.contdat = ContData(self.q3dinit)
        else :
            self.contdat = None
        if not noline:
            self.linedat = LineData(self.q3dinit)
        else :
            self.linedat = None
        self.map_bkg = background

        self.zsys = self.get_zsys(zsys, zsys_gas)
        self.zsys_gas = self.zsys


    def get_zsys(self,
                 zsys: Optional[float],
                 zsys_gas: Optional[float]) -> float:
        '''
        Get the systemic redshift of the galaxy from the zsys or zsys_gas attribute
        or q3din object, in that order.

        Returns
        -------
        float
            Systemic redshift of the galaxy.

        '''
        if zsys is not None:
            redshift = zsys
        elif zsys_gas is not None:
            redshift = zsys_gas
        elif self.q3dinit.zsys_gas is not None:
            redshift = self.q3dinit.zsys_gas
            q3dutil.write_msg('Using redshift from q3dinit object.', quiet=self.quiet)
        else:
            raise ValueError('Redshift not set in q3dinit or q3dpro ' +
                             'objects. Please use zsys parameter ' +
                             'in q3dpro to set the systemic redshift ' +
                             'of the galaxy for computing line properties.')
        return redshift
    

    def get_lineprop(self,
                     line: str,
                     **kwargs) -> tuple[float, str]:
        '''
        Get the rest wavelength and line label of an emission line.

        Parameters
        ----------
        line
            Name of the line to select.
        
        Returns
        -------
        float
            Wavelength of the line in microns.
        str
            Full label of the line for plotting.
        '''
        if self.linedat is None:
            raise ValueError('No line data available.')

        # backwards compatibility
        if 'LINESELECT' in kwargs:
            line = kwargs['LINESELECT']
        
        listlines = linelist(self.linedat.lines, vacuum=self.q3dinit.vacuum)
        ww = np.where(listlines['name'] == line)[0]
        linewave = listlines['lines'][ww].value[0]
        linename = listlines['linelab'][ww].value[0]
        return linewave, linename


    def get_linemap(self,
                    line: str,
                    applymask: bool=True,
                    **kwargs) -> \
                        tuple[float, str, dict[Literal['Ftot', 'Fci', 'Sig', 'v50', 'w80'], 
                                               dict[Literal['data', 'err', 'line', 'mask'], 
                                                    np.ndarray | list]]]:
        '''
        Create arrays holding maps of properties of an emission line.

        Parameters
        ----------
        line
            Name of the line to select.
        applymask
            Optional. Flag to apply the mask to the data. Default is True.
        
        Returns
        -------
        float
            Rest wavelength of the line in microns.
        str
            Full label of the line for plotting.
        dict[Literal['Ftot','Fci','Sig','v50','w80'], dict[Literal['data','err','line','mask'], np.ndarray | list]]
            Dictionary of line properties. Keys are 'Ftot', 'Fci', 'Sig', 'v50', 'w80'.
            Each key contains a dictionary with keys 'data', 'err', 'name', 'mask'.
            'data' and 'err' are the data and error arrays, respectively, with shape (ncols, nrows, ncomp). 
            'name' is the properly formatted name of the property, for plotting. 'mask' is the mask array, 
            with shape (ncols, nrows, ncomp). In the mask, 1 is good data and :py:attr:`~q3dfit.q3dpro.q3dpro.bad`
            is bad data.
        '''
        if self.linedat is None:
            raise ValueError('No line data available.')
        
        # backwards compatibility
        if 'LINESELECT' in kwargs:
            line = kwargs['LINESELECT']
        if 'APPLYMASK' in kwargs:
            applymask = kwargs['APPLYMASK']

        q3dutil.write_msg(f'Getting line data for {line}.', quiet=self.quiet)
        ncomp = np.max(self.q3dinit.ncomp[line])
        wave0, linetext = self.get_lineprop(line)
        # total fluxes and errors
        fluxsum = self.linedat.get_flux(line, fluxsel='ftot')['flux']
        fluxsum_err = self.linedat.get_flux(line, fluxsel='ftot')['fluxerr']
        fsmsk = _clean_mask(fluxsum, BAD=self.bad)

        # tuple giving ncols, nrows, ncomp
        matrix_size = (fluxsum.shape[0], fluxsum.shape[1], ncomp)

        # Create the output dictionary. Presently, we're only saving the properties of the last component
        # and the total flux.
        dataOUT = {'Ftot':
                   {'data': fluxsum,
                    'err': fluxsum_err,
                    'name': ['F$_{tot}$'],
                    'mask': fsmsk},
                   'Fci':
                   {'data': np.zeros(matrix_size),
                    'err': np.zeros(matrix_size),
                    'name': [],
                    'mask': np.zeros(matrix_size)},
                   'Sig':
                   {'data': np.zeros(matrix_size),
                    'err': np.zeros(matrix_size),
                    'name': [],
                    'mask': np.zeros(matrix_size)},
                   'v50':
                   {'data': np.zeros(matrix_size),
                    'err': np.zeros(matrix_size),
                    'name': [],
                    'mask': np.zeros(matrix_size)},
                   'w80':
                    {'data': np.zeros(matrix_size),
                     'err': np.zeros(matrix_size),
                     'name': [],
                     'mask': np.zeros(matrix_size)}
                   }

        # EXTRACT COMPONENTS
        for ci in range(0,ncomp) :
            ici = ci+1
            fcl = 'fc'+str(ici)
            iflux = self.linedat.get_flux(line, FLUXSEL=fcl)['flux']
            ifler = self.linedat.get_flux(line, FLUXSEL=fcl)['fluxerr']
            isigm = self.linedat.get_sigma(line, COMPSEL=ici)['sig']
            isger = self.linedat.get_sigma(line, COMPSEL=ici)['sigerr']
            iwvcn = self.linedat.get_wave(line, COMPSEL=ici)['wav']
            iwver = self.linedat.get_wave(line, COMPSEL=ici)['waverr']

            # now process them
            # this is the velocity of the line, though note that
            # it is not the special relativistic velocity
            iv50 = ((iwvcn - wave0)/wave0 - self.zsys)/(1.+self.zsys)*c.to('km/s').value
            iw80  = isigm*2.563 #w80 linewidth from the velocity dispersion
            # mask out the bad values
            ifmask = np.array(_clean_mask(iflux, BAD=self.bad))
            isgmsk = np.array(_clean_mask(isigm, BAD=self.bad))
            iwvmsk = np.array(_clean_mask(iwvcn, BAD=self.bad))

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
            dataOUT['Sig']['name'].append('$\\sigma_{c'+str(ici)+'}$')
            dataOUT['v50']['name'].append('v$_{50,c'+str(ici)+'}$')
            dataOUT['w80']['name'].append('w$_{80,c'+str(ici)+'}$')

            dataOUT['Fci']['mask'][:,:,ci] = ifmask
            dataOUT['Sig']['mask'][:,:,ci] = isgmsk
            dataOUT['v50']['mask'][:,:,ci] = iwvmsk
            dataOUT['w80']['mask'][:,:,ci] = isgmsk

        if applymask:
            for ditem in dataOUT:
                if len(dataOUT[ditem]['data'].shape) > 2:
                    for ci in range(0,ncomp):
                        dataOUT[ditem]['data'][:, :, ci] = \
                            dataOUT[ditem]['data'][:, :, ci]*dataOUT[ditem]['mask'][:, :, ci]
                        dataOUT[ditem]['err'][:,:,ci]  = \
                            dataOUT[ditem]['err'][:, :, ci]*dataOUT[ditem]['mask'][:, :, ci]
                else:
                    dataOUT[ditem]['data'] = dataOUT[ditem]['data']*dataOUT[ditem]['mask']
                    dataOUT[ditem]['err']  = dataOUT[ditem]['err']*dataOUT[ditem]['mask']

        return wave0, linetext, dataOUT


    def make_linemap(self,
                     line: str,
                     #snrcut: float=5.,
                     xyCenter: Optional[list]=None,
                     xyStyle: Optional[str]=None,
                     fluxcmap: str='YlOrBr_r',
                     fluxlog: bool=False,
                     ranges: Optional[dict[Literal['Ftot', 'Fci', 'Sig', 'v50', 'w80'], list]]=None,
                     #pltnum: int=1,
                     compSort: Optional[dict[Literal['sort_by','sort_range'], Any]]=None,
                     saveData: bool=False,
                     saveFormat: Literal['png', 'ps', 'pdf', 'svg']='png',
                     dpi: int=100,
                     **kwargs):
        '''
        Plot maps of properties of an emission line.

        Parameters
        ----------
        line
            Name of the line for which to create plots.
        compSort
            Optional. The 'sort_by' key
            must take one of the following values, which are keys of the dictionary output by
            :py:meth:`~q3dfit.q3dpro.Q3Dpro.get_linemap`: 'Ftot', 'Fci', 'Sig', 'v50', or 'w80'. 
            The 'sort_range' key is optional and is a tuple of lists. Each list contains two
            values that define the range of the component values to be used in the sorting.
            For the nth element of the tuple, the last existing component that lies in the range 
            will be the nth component in the re-sorted line maps. If 'sort_range' is not provided, 
            the components will be sorted based on the absolute value of the 'sort_by' parameter.
            Default is None, which means no sorting.
        xyCenter
            Optional. Center of the plot in 0-offset spaxel coordinates. Default is None, which means
            the center is the spatial middle of the FOV.
        xyStyle
            Optional. Set this to 'kpc' to label the axes in kpc from xyCenter. Set to any other
            string to label the axes in pixels from xyCenter. Default is None, which means the axes are 
            labeled in pixels from the lower left corner. 
        fluxcmap
            Optional. Color palette for the flux maps. Default is 'YlOrBr_r'.
        fluxlog
            Optional. If True, plot the flux maps with a log scale. Default is False.
        ranges
            Optional. Dictionary with one or more of the following keys: 'Ftot', 'Fci', 'Sig', 'v50', 'w80'.
            The values of each key are lists of two values that define the range of the parameter to be plotted.
            Default is None, which means the full range of the parameter is plotted.
        saveData
            Optional. If True, save the plots to a file. Default is False.
        saveFormat
            Optional. Format for saving the plots. Default is 'png'.
        dpi
            Optional. Dots per inch for the saved plots. Default is 100.
        '''
        # backwards compatibility
        if 'LINESELECT' in kwargs:
            line = kwargs['LINESELECT']
        #if 'SNRCUT' in kwargs:
        #    snrcut = kwargs['SNRCUT']
        if 'VCOMP' in kwargs:
            compSort = kwargs['VCOMP']
        if 'XYSTYLE' in kwargs:
            xyStyle = kwargs['XYSTYLE']
            if xyStyle is False:
                xyStyle = None
        if 'VMINMAX' in kwargs:
            ranges = kwargs['VMINMAX']
        #if 'PLTNUM' in kwargs:
        #    pltnum = kwargs['PLTNUM']
        if 'CMAP' in kwargs:
            fluxcmap = kwargs['CMAP']
        if 'SAVEDATA' in kwargs:
            saveData = kwargs['SAVEDATA']

        q3dutil.write_msg('Plotting emission line maps.', quiet=self.quiet)
        q3dutil.write_msg(f'Creating linemap of {line}.', quiet=self.quiet)
        # max number of components fitted
        ncomp = np.max(self.q3dinit.ncomp[line])
        # kpc per arcsec
        kpc_arcsec = cosmo.kpc_proper_per_arcmin(self.zsys).value/60.

        # linemaps
        wave0, linetext, linemaps = self.get_linemap(line, applymask=True)
        # sort the components if requested
        if compSort is not None:
            dataOUT = self._resort_line_components(linemaps, sort_pars=compSort)
        else:
            dataOUT = copy.deepcopy(linemaps)
        # Total line fluxes and errores
        fluxsum = dataOUT['Ftot']['data']
        #fluxsum_err = dataOUT['Ftot']['err']

        # Apply the SNR cut
        #fluxsum_snc, gdindx, bdindx = _snr_cut(fluxsum, fluxsum_err, SNRCUT=snrcut)
        matrix_size = fluxsum.shape #(fluxsum.shape[0], fluxsum.shape[1], fluxsum.shape[2])

        # --------------------------
        # Do the plotting here
        # --------------------------

        # range of the spaxels, in 0-offset coordinates
        xgrid = np.arange(0, matrix_size[1])
        ygrid = np.arange(0, matrix_size[0])
        xcol = xgrid
        ycol = ygrid

        # Set xyCenter. Assume the center of the FOV if not provided
        if xyCenter is None:
            xyCenter = [int(np.ceil(matrix_size[0]/2)),
                        int(np.ceil(matrix_size[1]/2))]

        xTitle = 'Column [pix]'
        yTitle = 'Row [pix]'
        if xyStyle is not None:
            # recenter the spaxel coordinates on the map center
            xcol = (xgrid - xyCenter[1])
            ycol = (ygrid - xyCenter[0])
            xTitle = 'Distance [pix]'
            yTitle = 'Distance [pix]'
            #qsoCenter = [0, 0]
            if xyStyle.lower() == 'kpc':
                kpc_pix = np.median(kpc_arcsec)* self.pix
                xcolkpc = xcol*kpc_pix
                ycolkpc = ycol*kpc_pix
                xcol, ycol = xcolkpc, ycolkpc
                xTitle = 'Distance [kpc]'
                yTitle = 'Distance [kpc]'
        #plt.close(PLTNUM)
        figDIM = [ncomp+1, 4]
        figOUT = _set_figsize(figDIM, matrix_size)
        # create the figure
        fig, ax = plt.subplots(figDIM[0], figDIM[1], dpi=dpi) #, facecolor=self.map_bkg)
        fig.set_figheight(figOUT[1]+2)  # (12)
        fig.set_figwidth(figOUT[0]-1)  # (14)

        # string to append to the line parameter name with component information
        ici = '' 
        # i and j are the row and column indices of the subplot
        # i is the component number, j is the line parameter number
        i, j = 0, 0
        # cycle through line parameters
        # icomp is the line parameter string
        # ipdat is the dictionary containing the line parameter data
        for icomp, ipdat in dataOUT.items():
            doPLT = False
            pixVals = ipdat['data'] # select just the parameter values
            ipshape = pixVals.shape
            # Set range to plot to the full range of the parameter
            # unless a range is provided
            iranges = [np.nanmin(pixVals), np.nanmax(pixVals)]
            if ranges is not None:
                if icomp in ranges:
                    iranges = ranges[icomp]
            if icomp == 'Ftot':
                ici='' # no component information for the total flux
                cmap = fluxcmap
                #nticks = 4
                vticks = [iranges[0], 
                          np.power(10, np.median([np.log10(iranges[0]), 
                                                  np.log10(iranges[1])])),
                          iranges[1]]
                if fluxlog:
                    logplot = True
                else:
                    logplot = False
            else:
                i=1 # start with the first component
                if icomp.lower() == 'fci':
                    j = 0 # first column
                    #nticks = 3
                    vticks = [iranges[0], 
                              np.power(10, np.median([np.log10(iranges[0]), 
                                                      np.log10(iranges[1])])),
                              iranges[1]]
                    if fluxlog:
                        logplot = True
                    else:
                        logplot = False
                if icomp.lower() == 'sig':
                    j+=1
                    cmap = 'YlOrBr_r'
                    #nticks  = 3
                    vticks = [iranges[0], (iranges[0]+iranges[1])/2., iranges[1]]
                    logplot = False
                elif icomp.lower() == 'v50' :
                    j+=1
                    cmap = 'RdYlBu_r'
                    #nticks = 3
                    vticks = [iranges[0], (iranges[0]+iranges[1])/2., iranges[1]]
                    logplot = False
                elif icomp.lower() == 'w80' :
                    j+=1
                    cmap = 'RdYlBu_r'
                    #nticks = 3
                    vticks = [iranges[0], (iranges[0]+iranges[1])/2., iranges[1]]
                    logplot = False
            # If this is the first row (i=0) but not the first col (j!=0), plot the total flux only 
            # and skip the rest of the line parameters
            if j != 0:
                doPLT = False
                fig.delaxes(ax[0, j])
            # If this is not the first row (i>0), plot the component maps
            for ci in range(0, ncomp) :
                ipixVals = []
                if icomp != 'Ftot' and len(ipshape) > 2:
                    doPLT = True
                    i = ci+1 # component no.
                    ici = '_c'+str(ci+1) # component string
                    ipixVals = pixVals[:, :, ci]
                # if this is a component row, don't plot the total flux
                elif icomp == 'Ftot':
                    doPLT = True
                    if ci > 0 :
                        doPLT = False
                        break
                    else:
                        ipixVals = pixVals
                if doPLT is True:
                    cmap_r = cm.get_cmap(cmap)
                    # cmap_r.set_bad(color='black')
                    cmap_r.set_bad(color=self.map_bkg)
                    axi = ax[i, j]
                    _display_pixels_wz(ycol, xcol, ipixVals, axi, CMAP=cmap,
                                       COLORBAR=True, PLOTLOG=logplot,
                                       VMIN=iranges[0], VMAX=iranges[1],
                                       TICKS=vticks) #, NTICKS=nticks)
                    # Plot the center
                    axi.errorbar(xyCenter[0]-1, xyCenter[1]-1, color='black', mew=1, mfc='red', fmt='*', 
                                 markersize=15, zorder=2)
                    axi.set_xlabel(xTitle, fontsize=16)
                    axi.set_ylabel(yTitle, fontsize=16)
                    axi.set_title(ipdat['name'][ci], fontsize=20, pad=45)
                    # axi.set_ylim([min(xx),np.ceil(max(xx))])
                    # axi.set_xlim([min(yy),np.ceil(max(yy))])
                    if saveData:
                        linesave_name = self.target_name+'_'+line+'_'+icomp+ici+'_map.fits'
                        q3dutil.write_msg(f'Saving line map {linesave_name}', quiet=self.quiet)
                        savepath = os.path.join(self.dataDIR,linesave_name)
                        _save_to_fits(pixVals, None, savepath)


            # j+=1
        fig.suptitle(self.target_name+' : '+linetext+' maps',fontsize=20,snap=True,
                     horizontalalignment='right')
                     # verticalalignment='center',
                     # fontweight='semibold')
        fig.tight_layout()#pad=0.15,h_pad=0.1)
        if saveData:
            pltsave_name = f'{self.target_name}-{line}-map.{saveFormat}'
            q3dutil.write_msg(f'Saving {pltsave_name} to {self.dataDIR}.', 
                quiet=self.quiet)
            plt.savefig(os.path.join(self.dataDIR, f'{pltsave_name}'))
        # fig.subplots_adjust(top=0.88)
        plt.show()


    def make_lineratio_map(self,
                           lineA: str,
                           lineB: str,
                           snrcut: float=3.,
                           vminmax: list=[None, None],
                           kpc: bool=False,
                           cmap: str='inferno',
                           saveData: bool=False,
                           saveFormat: str='png',
                           dpi: int=100,
                           **kwargs):
        '''
        Plot map of the line ratio of two emission lines.

        Parameters
        ----------
        lineA
            Name of the first line for the line ratio.
        lineB
            Name of the second line for the line ratio.
        snrcut
            Optional. Signal-to-noise ratio cut for the line maps. Default is 3.
        vminmax
            Optional. List of two values that define the range of the line ratio to be plotted.
            Default is [None, None], which means the full range of the line ratio is plotted.
        kpc
            Optional. If True, label the axes in kpc from the galaxy center. Default is False.
        cmap
            Optional. Color palette for the line ratio map. Default is 'inferno'.
        saveData
            Optional. If True, save the plots to a file. Default is False.
        saveFormat
            Optional. Format for saving the plots. Default is 'png'.
        dpi
            Optional. Dots per inch for the saved plots. Default is 100.
        '''

        if self.linedat is None:
            raise ValueError('No line data available.')
        if lineA not in self.linedat.lines or lineB not in self.linedat.lines:
            raise ValueError('One or both of the lines are not in the line data.')

        # backwards compatibility
        if 'SNRCUT' in kwargs:
            snrcut = kwargs['SNRCUT']
        if 'SAVEDATA' in kwargs:
            saveData = kwargs['SAVEDATA']
        if 'VMINMAX' in kwargs:
            vminmax = kwargs['VMINMAX']
        if 'KPC' in kwargs:
            kpc = kwargs['KPC']


        # ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
        # first identify the lines, extract fluxes, and apply the SNR cuts
        # ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

        #arckpc = cosmo.kpc_proper_per_arcmin(self.zsys).value/60.
        lines = [lineA, lineB]
        linelist = dict()
        # first element holds the shape of the total flux data (ncols, nrows)
        # second element holds the shape of the component flux data (ncols, nrows, ncomp)
        mshaps = list()
        for lin in lines:
            ncomp = np.max(self.q3dinit.ncomp[lin])
            wave0, linetext, dataOUT = self.get_linemap(lin, applymask=False)
            # dictionary to hold the line map information
            istruct = {'wavcen': wave0,
                       'wname': linetext,
                       'data': dataOUT,
                       'snr':{}}
            # Total flux data
            # cols, rows
            mshaps.append(dataOUT['Ftot']['data'].shape)
            istruct['snr']['Ftot'] = \
                list(_snr_cut(dataOUT['Ftot']['data'],
                              dataOUT['Ftot']['err'],
                              snrcut))

            # Component fluxdata
            istruct['snr']['Fci'] = [[],[],[]]
            # cols, rows, components
            mshaps.append(dataOUT['Fci']['data'].shape)
            istruct['snr']['Fci'][0] = np.zeros(mshaps[1])
            for ci in range(0, ncomp):
                i_snc, i_gindx, i_bindx = \
                    _snr_cut(dataOUT['Fci']['data'][:, :, ci],
                             dataOUT['Fci']['err'][:, :, ci],
                             snrcut)
                istruct['snr']['Fci'][0][:, :, ci] = i_snc
                istruct['snr']['Fci'][1].append(i_gindx)
                istruct['snr']['Fci'][2].append(i_bindx)
            linelist[lin] = istruct

        # ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
        # Calculate the line ratios
        # ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
        q3dutil.write_msg('Calculating line ratio.', quiet=self.quiet)

        # dictionary to hold the line ratios
        lineratios = {'lines': [lineA, lineB],
                       'pltname': linelist[lineA]['wname']+'/'+linelist[lineB]['wname'],
                       'lrat': {'Ftot': None, 
                                'Fci': None}
                      }

        # dictionary to hold the line flux data
        fratdat = [{'Ftot': list(),
                    'Fci': list()},
                   {'Ftot': list(),
                    'Fci': list()}]
        # loop over the lines and collect the flux data
        for li, lin in enumerate(lineratios['lines']):
            # case of multiple lines to be combined for either side of the ratio
            if isinstance(lin, list):
                for i in range(0, 4):
                    fratdat[li]['Ftot'].append(np.zeros(mshaps[0]))
                    fratdat[li]['Fci'].append(np.zeros(mshaps[1]))
                fratdat[li]['Ftot'][2] +=1
                fratdat[li]['Fci'][2] +=1
                for jlin in lin :
                    lij_datOUT = linelist[jlin]['data']
                    lij_snrOUT = linelist[jlin]['snr']
                    lij_ftot, lij_ftotER, lij_ftotMASK = \
                        lij_datOUT['Ftot']['data'], lij_datOUT['Ftot']['err'], lij_datOUT['Ftot']['mask']
                    lij_fci, lij_fciER, lij_fciMASK = \
                        lij_datOUT['Fci']['data'], lij_datOUT['Fci']['err'], lij_datOUT['Fci']['mask']
                    fratdat[li]['Ftot'][0] += lij_ftot
                    fratdat[li]['Ftot'][1] += lij_ftotER
                    fratdat[li]['Ftot'][2] *= lij_ftotMASK
                    fratdat[li]['Ftot'][3] += lij_snrOUT['Ftot'][0]
                    fratdat[li]['Fci'][0] += lij_fci
                    fratdat[li]['Fci'][1] += lij_fciER
                    fratdat[li]['Fci'][2] *= lij_fciMASK
                    fratdat[li]['Fci'][3] += lij_snrOUT['Fci'][0]
            else:
                li_datOUT = linelist[lin]['data']
                li_snrOUT = linelist[lin]['snr']
                li_ftot, li_ftotER, li_ftotMASK = \
                    li_datOUT['Ftot']['data'], li_datOUT['Ftot']['err'], li_datOUT['Ftot']['mask']
                li_fci, li_fciER, li_fciMASK = \
                    li_datOUT['Fci']['data'], li_datOUT['Fci']['err'], li_datOUT['Fci']['mask']
                fratdat[li]['Ftot'].append(li_ftot)
                fratdat[li]['Ftot'].append(li_ftotER)
                fratdat[li]['Ftot'].append(li_ftotMASK)
                fratdat[li]['Ftot'].append(li_snrOUT['Ftot'][0])
                fratdat[li]['Fci'].append(li_fci)
                fratdat[li]['Fci'].append(li_fciER)
                fratdat[li]['Fci'].append(li_fciMASK)
                fratdat[li]['Fci'].append(li_snrOUT['Fci'][0])
        # calculate the line ratio
        for lratF in lineratios['lrat']:
            fi_mask = fratdat[0][lratF][2] * fratdat[1][lratF][2]
            fi_frat = fratdat[0][lratF][3] / fratdat[1][lratF][3]
            frat10 = np.log10(fi_frat)*fi_mask
            frat10err = _lgerr(fratdat[0][lratF][3], fratdat[1][lratF][3],
                               fratdat[0][lratF][1], fratdat[1][lratF][1])
            lineratios['lrat'][lratF]=[frat10, frat10err]

        # ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
        # Do the plotting here
        # ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
        # if cntr != 0 and bptc != 0:
        # No. of panels. 1 for 'Ftot', ncomp for 'Fci'
        nps = mshaps[1][2] + 1
        xyCenter = [int(np.ceil(mshaps[1][0]/2)),
                    int(np.ceil(mshaps[1][1]/2))]
        xgrid = np.arange(0, mshaps[1][1])
        ygrid = np.arange(0, mshaps[1][0])
        xcol = xgrid
        ycol = ygrid
        if kpc:
            xcol = (xgrid - xyCenter[1])
            ycol = (ygrid - xyCenter[0])
        # --------------------------
        # Plot ine ratio map
        # --------------------------
        figDIM = [1, nps]
        figOUT = _set_figsize(figDIM, mshaps[1])
        fig, ax = plt.subplots(1, nps, dpi=dpi)#, gridspec_kw={'height_ratios': [1, 2]})
        fig.set_figheight(figOUT[1])
        fig.set_figwidth(figOUT[0])

        cmap_r = cm.get_cmap(cmap)
        cmap_r.set_bad(color=self.map_bkg)
        cf = 0

        xx,yy = xcol,ycol
        for ni in range(0, nps):
            ax[ni].set_xlabel('spaxel', fontsize=13)
            ax[ni].set_ylabel('spaxel', fontsize=13)
            if kpc:
                ax[ni].set_xlabel('Relative distance [kpc]', fontsize=13)
                ax[ni].set_ylabel('Relative distance [kpc]', fontsize=13)
            prelud = ''
            if ni == 0:
                prelud = 'Ftot: '
            else:
                prelud = 'Fc'+str(ni)+': '
            ax[ni].set_title(f'{prelud}log$_{{10}}$ {lineratios['pltname']}', fontsize=15, pad=45)
        frat10, frat10err = lineratios['lrat']['Ftot'][0], lineratios['lrat']['Ftot'][1]
        _display_pixels_wz(yy, xx, frat10, ax[0], CMAP=cmap,
                           VMIN=vminmax[0], VMAX=vminmax[1], NTICKS=5, COLORBAR=True)
        frat10, frat10err = lineratios['lrat']['Fci'][0], lineratios['lrat']['Fci'][1]
        for ci in range(0,ncomp):
            _display_pixels_wz(yy, xx, frat10[:,:,ci], ax[1+ci], CMAP=cmap,
                               VMIN=vminmax[0], VMAX=vminmax[1], NTICKS=5, COLORBAR=True)
        plt.tight_layout(pad=1.5, h_pad=0.1)
        if saveData:
            pltsave_name = f'{self.target_name}-{lineA}-{lineB}-ratio-map.{saveFormat}'
            q3dutil.write_msg(f'Saving {pltsave_name} to {self.dataDIR}.', quiet=self.quiet)
            plt.savefig(os.path.join(self.dataDIR, f'{pltsave_name}'))
        plt.show()


    def make_BPT(self,
                 snrcut: float=3.,
                 #vminmax: list=[None, None],
                 kpc: bool=False,
                 cmap: str='inferno',
                 saveData: bool=False,
                 saveFormat: str='png',
                 dpi: int=100,
                 **kwargs):
        '''
        Plot usual BPT diagrams.

        Parameters
        ----------
        snrcut
            Optional. Signal-to-noise ratio cut for the line maps. Default is 3.
        kpc
            Optional. If True, label the axes in kpc from the galaxy center. Default is False.
        cmap
            Optional. Color palette for the line ratio map. Default is 'inferno'.
        saveData
            Optional. If True, save the plots to a file. Default is False.
        saveFormat
            Optional. Format for saving the plots. Default is 'png'.
        dpi
            Optional. Dots per inch for the saved plots. Default is 100.
        '''

        if self.linedat is None:
            raise ValueError('No line data available.')

        # backwards compatibility
        if 'SNRCUT' in kwargs:
            snrcut = kwargs['SNRCUT']
        if 'SAVEDATA' in kwargs:
            saveData = kwargs['SAVEDATA']
        #if 'VMINMAX' in kwargs:
        #    vminmax = kwargs['VMINMAX']
        if 'KPC' in kwargs:
            kpc = kwargs['KPC']

        # ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
        # first identify the lines, extract fluxes, and apply the SNR cuts
        # ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
        BPTlist = {'Hbeta', '[OIII]5007', '[OI]6300', 'Halpha', '[NII]6548', '[NII]6583', 
                   '[SII]6716', '[SII]6731'}
        for line in BPTlist:
            if line not in self.linedat.lines:
                raise ValueError('One of the BPT lines is not in the line data.')

        BPTlines = dict()
        mshaps = list()
        li = 0
        for lin in BPTlist:
            ncomp = np.max(self.q3dinit.ncomp[lin])
            wave0, linetext, dataOUT = self.get_linemap(lin, APPLYMASK={})
            istruct = {'wavcen': wave0,
                        'wname': linetext,
                        'data': dataOUT,
                        'snr': dict()}

            # Total flux data
            mshaps.append(dataOUT['Ftot']['data'].shape)
            istruct['snr']['Ftot'] = \
                list(_snr_cut(dataOUT['Ftot']['data'], 
                              dataOUT['Ftot']['err'], 
                              snrcut))

            # Component fluxdata
            istruct['snr']['Fci'] = [[],[],[]]
            mshaps.append(dataOUT['Fci']['data'].shape)
            istruct['snr']['Fci'][0] = np.zeros(mshaps[1])
            for ci in range(0, ncomp):
                i_snc, i_gindx, i_bindx = \
                    _snr_cut(dataOUT['Fci']['data'][:, :, ci], 
                             dataOUT['Fci']['err'][:, :, ci],
                             snrcut)
                istruct['snr']['Fci'][0][:, :, ci] = i_snc
                istruct['snr']['Fci'][1].append(i_gindx)
                istruct['snr']['Fci'][2].append(i_bindx)
            BPTlines[lin] = istruct

        lineratios = {'OiiiHb': {'lines': ['[OIII]5007', 'Hbeta'],
                                 'pltname': '[OIII]/H$\\beta$', 
                                 'pltrange': [-1,1.5],
                                'lrat': {'Ftot':None,
                                         'Fci': None}},
                        'SiiHa': {'lines': [['[SII]6716', '[SII]6731'], 'Halpha'],
                                  'pltname': '[SII]/H$\\alpha$',
                                  'pltrange': [-1.8,0.9],
                                  'lrat': {'Ftot': None,
                                           'Fci': None}},
                        'OiHa': {'lines': ['[OI]6300', 'Halpha'], 
                                 'pltname': '[OI]/H$\\alpha$', 
                                 'pltrange': [-1.8, 0.1],
                                 'lrat': {'Ftot':None,
                                          'Fci':None}},
                       'NiiHa': {'lines': ['[NII]6583', 'Halpha'], 
                                 'pltname': '[NII]/H$\\alpha$',
                                 'pltrange': [-1.8, 0.1],
                                 'lrat': {'Ftot': None,
                                          'Fci': None}},
                      }

        
        # ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
        # make the theoretical BPT models
        # ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
        BPTmod = {'OiiiHb/NiiHa': list(),
                  'OiiiHb/SiiHa': list(),
                  'OiiiHb/OiHa': list()}
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
        q3dutil.write_msg('Calculating line ratios.', quiet=self.quiet)
        cntr = 0
        # loop over the line ratios
        for lrat,lratdat in lineratios.items():
            # dictionary to hold the line flux data
            fratdat = [{'Ftot': list(),
                        'Fci': list()},
                       {'Ftot': list(),
                        'Fci': list()}]
            # loop over the lines and collect the flux data
            for li, lin in enumerate(lratdat['lines']):
                if isinstance(lin, list):
                    for i in range(0, 4):
                        fratdat[li]['Ftot'].append(np.zeros(mshaps[0]))
                        fratdat[li]['Fci'].append(np.zeros(mshaps[1]))
                    fratdat[li]['Ftot'][2] +=1
                    fratdat[li]['Fci'][2] +=1
                    for jlin in lin :
                        lij_datOUT = BPTlines[jlin]['data']
                        lij_snrOUT = BPTlines[jlin]['snr']
                        lij_ftot, lij_ftotER, lij_ftotMASK = \
                            lij_datOUT['Ftot']['data'], lij_datOUT['Ftot']['err'], lij_datOUT['Ftot']['mask']
                        lij_fci, lij_fciER, lij_fciMASK = \
                            lij_datOUT['Fci']['data'], lij_datOUT['Fci']['err'], lij_datOUT['Fci']['mask']
                        fratdat[li]['Ftot'][0] += lij_ftot
                        fratdat[li]['Ftot'][1] += lij_ftotER
                        fratdat[li]['Ftot'][2] *= lij_ftotMASK
                        fratdat[li]['Ftot'][3] += lij_snrOUT['Ftot'][0]
                        fratdat[li]['Fci'][0] += lij_fci
                        fratdat[li]['Fci'][1] += lij_fciER
                        fratdat[li]['Fci'][2] *= lij_fciMASK
                        fratdat[li]['Fci'][3] += lij_snrOUT['Fci'][0]
                else:
                    li_datOUT = BPTlines[lin]['data']
                    li_snrOUT = BPTlines[lin]['snr']
                    li_ftot, li_ftotER, li_ftotMASK = \
                        li_datOUT['Ftot']['data'], li_datOUT['Ftot']['err'], li_datOUT['Ftot']['mask']
                    li_fci, li_fciER, li_fciMASK = \
                        li_datOUT['Fci']['data'], li_datOUT['Fci']['err'], li_datOUT['Fci']['mask']
                    fratdat[li]['Ftot'].append(li_ftot)
                    fratdat[li]['Ftot'].append(li_ftotER)
                    fratdat[li]['Ftot'].append(li_ftotMASK)
                    fratdat[li]['Ftot'].append(li_snrOUT['Ftot'][0])
                    fratdat[li]['Fci'].append(li_fci)
                    fratdat[li]['Fci'].append(li_fciER)
                    fratdat[li]['Fci'].append(li_fciMASK)
                    fratdat[li]['Fci'].append(li_snrOUT['Fci'][0])
            cntr +=1
            # calculate the line ratio
            for lratF in lratdat['lrat']:
                fi_mask = fratdat[0][lratF][2] * fratdat[1][lratF][2]
                fi_frat = fratdat[0][lratF][3] / fratdat[1][lratF][3]
                frat10 = np.log10(fi_frat)*fi_mask
                frat10err = _lgerr(fratdat[0][lratF][3],fratdat[1][lratF][3],
                                   fratdat[0][lratF][1],fratdat[1][lratF][1])
                lineratios[lrat]['lrat'][lratF]=[frat10, frat10err]

        # ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
        # Do the plotting here
        # ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
        # No. of panels. 1 for 'Ftot', ncomp for 'Fci'
        nps = mshaps[1][2]+1
        xyCenter = [int(np.ceil(mshaps[1][0]/2)),
                    int(np.ceil(mshaps[1][1]/2))]
        xgrid = np.arange(0, mshaps[1][1])
        ygrid = np.arange(0, mshaps[1][0])
        xcol = xgrid
        ycol = ygrid
        if kpc:
            xcol = (xgrid-xyCenter[1])
            ycol = (ygrid-xyCenter[0])
        # --------------------------
        # Plot line ratio maps
        # --------------------------
        figDIM = [nps, cntr]
        figOUT = _set_figsize(figDIM, mshaps[1])
        fig,ax = plt.subplots(nps, cntr, dpi=dpi)#, gridspec_kw={'height_ratios': [1, 2]})
        fig.set_figheight(figOUT[1]+1)
        fig.set_figwidth(figOUT[0])

        cmap_r = cm.get_cmap(cmap)
        cmap_r.set_bad(color=self.map_bkg)
        cf = 0
        for linrat in lineratios:
            xx,yy = xcol,ycol
                # iax = ax[cf]
            pltname, pltrange = lineratios[linrat]['pltname'], lineratios[linrat]['pltrange']
            for ni in range(0, nps):
                ax[ni,cf].set_xlabel('spaxel', fontsize=13)
                ax[ni,cf].set_ylabel('spaxel', fontsize=13)
                if kpc:
                    ax[ni,cf].set_xlabel('Relative distance [kpc]',fontsize=13)
                    ax[ni,cf].set_ylabel('Relative distance [kpc]',fontsize=13)
                prelud = ''
                if ni == 0:
                    prelud = 'Ftot: '
                else:
                    prelud = 'Fc'+str(ni)+': '
                ax[ni,cf].set_title(prelud+'log$_{{10}}$ '+pltname, fontsize=15,pad=45)

            frat10,frat10err = lineratios[linrat]['lrat']['Ftot'][0],lineratios[linrat]['lrat']['Ftot'][1]
            _display_pixels_wz(yy, xx, frat10, ax[0, cf], CMAP=cmap, 
                                VMIN=-1, VMAX=1, NTICKS=5, COLORBAR=True)
            frat10,frat10err = lineratios[linrat]['lrat'][lratF][0],lineratios[linrat]['lrat'][lratF][1]
            for ci in range(0,ncomp):
                _display_pixels_wz(yy, xx, frat10[:, :, ci], ax[1+ci, cf], CMAP=cmap, 
                                   VMIN=-1, VMAX=1, NTICKS=5, COLORBAR=True)
            cf += 1
        plt.tight_layout(pad=1.5, h_pad=0.1)
        if saveData:
            pltsave_name = f'{self.target_name}-BPT-linemaps.{saveFormat}'
            q3dutil.write_msg(f'Saving {pltsave_name} to {self.dataDIR}.', quiet=self.quiet)
            plt.savefig(os.path.join(self.dataDIR, pltsave_name))
        plt.show()

        # --------------------------
        # Plot BPT here
        # --------------------------

        figDIM = [nps, cntr-1]
        figOUT = _set_figsize(figDIM, mshaps[1])
        fig,ax = plt.subplots(nps,cntr-1, figsize=((cntr-1)*5,5), dpi=dpi)
        fig.set_figheight(int(figOUT[1]))
        fig.set_figwidth(figOUT[0])
        cf=0
        xgrid = np.arange(0, mshaps[1][1])
        ygrid = np.arange(0, mshaps[1][0])
        xcol = (xgrid-xyCenter[1])
        ycol = (ygrid-xyCenter[0])
        for bpt in BPTmod:
            for li, lratF in enumerate(['Ftot', 'Fci']):
                fnames = bpt.split('/')
                pltname1, pltrange1 = lineratios[fnames[0]]['pltname'], lineratios[fnames[0]]['pltrange']
                pltname2, pltrange2 = lineratios[fnames[1]]['pltname'], lineratios[fnames[1]]['pltrange']
                if lratF == 'Fci':
                    frat10_A, frat10errA = lineratios[fnames[0]]['lrat'][lratF][0], \
                        lineratios[fnames[0]]['lrat'][lratF][1]
                    frat10_B, frat10errB = lineratios[fnames[1]]['lrat'][lratF][0], \
                        lineratios[fnames[1]]['lrat'][lratF][1]
                    for ci in range(0, ncomp):
                        gg = np.where(~np.isnan(frat10_A[:,:,ci]) & ~np.isnan(frat10_B[:,:,ci]))
                        xfract, yfract = frat10_B[:,:,ci][gg], frat10_A[:,:,ci][gg]
                        xfracterr,yfracterr = \
                            [frat10errB[0][:, :, ci][gg].flatten(),
                             frat10errB[1][:, :, ci][gg].flatten()], \
                            [frat10errA[0][:, :, ci][gg].flatten(), 
                             frat10errA[1][:, :, ci][gg].flatten()]
                        ee = np.where(~np.isnan(xfracterr[0]) & \
                                      ~np.isnan(xfracterr[1]) & \
                                      ~np.isnan(yfracterr[0]) & \
                                      ~np.isnan(yfracterr[1]))
                        ax[li+ci, cf].errorbar(xfract.flatten()[ee], 
                                               yfract.flatten()[ee], fmt='.', alpha=0.7,
                                               color='black', markersize=5, zorder=2)
                        ax[li+ci, cf].errorbar(xfract.flatten()[ee],
                                               yfract.flatten()[ee], fmt='.', alpha=0.7,
                                               xerr=[xfracterr[0][ee],
                                                     xfracterr[1][ee]],
                                               yerr=[yfracterr[0][ee],
                                                     yfracterr[1][ee]],
                                               elinewidth=0.8, ecolor='dodgerblue',
                                               color='black', markersize=0, zorder=2)
                        ax[li+ci,cf].errorbar(np.median(frat10_B[gg[0], gg[1],ci].flatten()),
                                              np.median(frat10_A[gg[0], gg[1],ci].flatten()),
                                              fillstyle='none', color='red', fmt='*', markersize=15,
                                              mew=2, zorder=3)
                else:
                    frat10_A, frat10errA = \
                        lineratios[fnames[0]]['lrat'][lratF][0], \
                        lineratios[fnames[0]]['lrat'][lratF][1]
                    frat10_B, frat10errB = \
                        lineratios[fnames[1]]['lrat'][lratF][0], \
                        lineratios[fnames[1]]['lrat'][lratF][1]
                    gg = np.where(~np.isnan(frat10_B) & ~np.isnan(frat10_A))
                    xfract, yfract = frat10_B[gg], frat10_A[gg]
                    xfracterr, yfracterr = [frat10errB[0][gg].flatten(),
                                            frat10errB[1][gg].flatten()], \
                                           [frat10errA[0][gg].flatten(), 
                                            frat10errA[1][gg].flatten()]
                    ee = np.where(~np.isnan(xfracterr[0]) &
                                  ~np.isnan(xfracterr[1]) &  
                                  ~np.isnan(yfracterr[0]) & 
                                  ~np.isnan(yfracterr[1]))
                    ax[li,cf].errorbar(xfract.flatten()[ee], yfract.flatten()[ee], fmt='.', alpha=0.7,
                                        color='black', markersize=5, zorder=2)
                    ax[li,cf].errorbar(xfract.flatten()[ee], yfract.flatten()[ee], fmt='.', alpha=0.7,
                                        xerr=[xfracterr[0][ee], xfracterr[1][ee]], 
                                        yerr=[yfracterr[0][ee], yfracterr[1][ee]],
                                        elinewidth=0.8, ecolor='dodgerblue',
                                        color='black', markersize=0, zorder=2)
                    ax[li,cf].errorbar(np.median(xfract.flatten()), np.median(yfract.flatten()),
                                        fillstyle='none', color='red', fmt='*', markersize=15, mew=2, zorder=3)
            for ni in range(0,nps):
                ax[ni,cf].set_xlim([-2.1, 0.7])
                ax[ni,cf].set_ylim(pltrange1)
                # first plot the theoretical curves
                iBPTmod = BPTmod[bpt]
                ax[ni,cf].plot(iBPTmod[0][0], iBPTmod[0][1], 'k-', zorder=1, linewidth=1.5)
                ax[ni,cf].plot(iBPTmod[1][0], iBPTmod[1][1], 'k--', zorder=1, linewidth=1.5)
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
        if saveData:
            pltsave_name = f'{self.target_name}-BPT.{saveFormat}'
            q3dutil.write_msg(f'Saving {pltsave_name} to {self.dataDIR}.', quiet=self.quiet)
            plt.savefig(os.path.join(self.dataDIR, pltsave_name))
        plt.show()


    def _resort_line_components(self,
                                linemaps: dict[Literal['Ftot', 'Fci', 'Sig', 'v50', 'w80'], 
                                               dict[Literal['data', 'err', 'name', 'mask'], Any]],
                                sort_pars: dict[Literal['sort_by', 'sort_range'], Any]) \
                                      -> dict[Literal['Ftot', 'Fci', 'Sig', 'v50', 'w80'], 
                                              dict[Literal['data', 'err', 'name', 'mask'], Any]]:
        '''
        Re-sort the line components based on the values of a given line parameter.

        Parameters
        ----------
        linemaps
            Line maps. This equals the third output of the :py:meth:`~q3dfit.q3dpro.get_linemap` 
            method.
        sort_pars
            Parameters for sorting components. See `compSort` parameter of
            :py:meth:`~q3dfit.q3dpro.make_linemap` for more information.
        
        Returns
        -------
        dict
            The re-sorted line maps. This is the same as the input linemaps, but with the components
            re-ordered based on the sort_by parameter.
        '''
        if sort_pars['sort_by'] not in linemaps.keys():
            raise ValueError('The sort_by parameter in sort_pars is not valid.')
        else:
            sortDat = linemaps[sort_pars['sort_by']]
            mshap = sortDat['data'].shape

        q3dutil.write_msg('Sorting components by ', sort_pars['sort_by'])

        if 'sort_range' not in sort_pars:
            dataOUT = copy.deepcopy(linemaps)
            for ii in range (0, mshap[0]):
                for jj in range (0, mshap[1]):
                    sij = np.argsort(np.abs(sortDat['data'][ii,jj,:]))
                    dataOUT['Fci']['data'][ii,jj,:] = linemaps['Fci']['data'][ii,jj,sij]
                    dataOUT['Sig']['data'][ii,jj,:] = linemaps['Sig']['data'][ii,jj,sij]
                    dataOUT['v50']['data'][ii,jj,:] = linemaps['v50']['data'][ii,jj,sij]
                    dataOUT['w80']['data'][ii,jj,:] = linemaps['w80']['data'][ii,jj,sij]
                    dataOUT['Fci']['err'][ii,jj,:]  = linemaps['Fci']['err'][ii,jj,sij]
                    dataOUT['Sig']['err'][ii,jj,:]  = linemaps['Sig']['err'][ii,jj,sij]
                    dataOUT['v50']['err'][ii,jj,:]  = linemaps['v50']['err'][ii,jj,sij]
                    dataOUT['w80']['err'][ii,jj,:]  = linemaps['w80']['err'][ii,jj,sij]
                    dataOUT['Fci']['mask'][ii,jj,:] = linemaps['Fci']['mask'][ii,jj,sij]
                    dataOUT['Sig']['mask'][ii,jj,:] = linemaps['Sig']['mask'][ii,jj,sij]
                    dataOUT['v50']['mask'][ii,jj,:] = linemaps['v50']['mask'][ii,jj,sij]
                    dataOUT['w80']['mask'][ii,jj,:] = linemaps['w80']['mask'][ii,jj,sij]

        else:
            sort_rang = sort_pars['sort_range']
            for si, srang in enumerate(sort_rang):
                q3dutil.write_msg(f'The component c{si+1} will equal the last component that '+
                                  f'lies in the range {srang}.', quiet=self.quiet)
                q3dutil.write_msg(f'Setting values to np.nan otherwise.', quiet=self.quiet)
            # Set up output dictionary
            dataOUT = {'Ftot':linemaps['Ftot'],
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
            # loop through spaxels
            for ii in range (0, mshap[0]):
                for jj in range (0, mshap[1]):
                    # this tracks the input component index
                    for cc in range(0, mshap[2]):
                        # this tracks the output component index and the sort range
                        for sri, sr in enumerate(sort_rang):
                            dataOUT['Fci']['name'].append('F$_{c'+str(sri)+'}$')
                            dataOUT['Sig']['name'].append('$\\sigma_{c'+str(sri)+'}$')
                            dataOUT['v50']['name'].append('v$_{50,c'+str(sri)+'}$')
                            dataOUT['w80']['name'].append('w$_{80,c'+str(sri)+'}$')
                            if sr[0] <= sortDat['data'][ii,jj,cc] <= sr[1]:
                                dataOUT['Fci']['data'][ii,jj,sri] = linemaps['Fci']['data'][ii,jj,cc]
                                dataOUT['Sig']['data'][ii,jj,sri] = linemaps['Sig']['data'][ii,jj,cc]
                                dataOUT['v50']['data'][ii,jj,sri] = linemaps['v50']['data'][ii,jj,cc]
                                dataOUT['w80']['data'][ii,jj,sri] = linemaps['w80']['data'][ii,jj,cc]
                                dataOUT['Fci']['err'][ii,jj,sri]  = linemaps['Fci']['err'][ii,jj,cc]
                                dataOUT['Sig']['err'][ii,jj,sri]  = linemaps['Sig']['err'][ii,jj,cc]
                                dataOUT['v50']['err'][ii,jj,sri]  = linemaps['v50']['err'][ii,jj,cc]
                                dataOUT['w80']['err'][ii,jj,sri]  = linemaps['w80']['err'][ii,jj,cc]
                                dataOUT['Fci']['mask'][ii,jj,sri] = linemaps['Fci']['mask'][ii,jj,cc]
                                dataOUT['Sig']['mask'][ii,jj,sri] = linemaps['Sig']['mask'][ii,jj,cc]
                                dataOUT['v50']['mask'][ii,jj,sri] = linemaps['v50']['mask'][ii,jj,cc]
                                dataOUT['w80']['mask'][ii,jj,sri] = linemaps['w80']['mask'][ii,jj,cc]
            
        # return the re-sorted line maps
        return dataOUT
 

class LineData:
    '''
    Read in and store line data for all fitted lines in the numpy save file created by 
    :py:meth:`~q3dfit.q3dcollect.q3dcollect`.

    Individual line measurements can be obtained with corresponding methods.

    Parameters
    -----------
    q3di
        q3d initialization object containing line names and other parameters.

    Attributes
    ----------
    lines : list
        Copy of :py:attr:`~q3dfit.q3din.q3din.lines`.
    maxncomp : int
        Copy of :py:attr:`~q3dfit.q3din.q3din.maxncomp`.
    data : numpy.lib.npyio.NpzFile
        Contents of the line data (.npz) file.
    ncols : int
        Copy of :py:attr:`~q3dfit.q3din.q3din.ncols`.
    nrows : int
        Copy of :py:attr:`~q3dfit.q3din.q3din.nrows`.
    bad : float
        Value of bad pixels. Default is np.nan.
    dataDIR : str
        Copy of :py:attr:`~q3dfit.q3din.q3din.outdir`.
    target_name : str
        Copy of :py:attr:`~q3dfit.q3din.q3din.name`.
    colname : list
        Sorted list of dictionary names in the :py:attr:`~q3dfit.q3din.q3din.data` attribute.
        Set in :py:meth:`~q3dfit.q3dpro.LineData._read_npz`.
    '''

    def __init__(self,
                 q3di: q3din.q3din):

        filename = q3di.label+'.line.npz'
        datafile = os.path.join(q3di.outdir, filename)
        if not os.path.exists(datafile):
            raise FileNotFoundError(f'ERROR: emission line file {filename} does not exist.')
        self.lines = q3di.lines
        self.maxncomp = q3di.maxncomp
        self.data = self._read_npz(datafile)
        # book-keeping inheritance from initproc
        self.ncols = q3di.ncols # self.data['ncols'].item()
        self.nrows = q3di.nrows # self.data['nrows'].item()
        self.bad = np.nan
        self.dataDIR = q3di.outdir
        self.target_name = q3di.name


    def _read_npz(self,
                  file: str):
        '''
        Load compressed file archive.

        Parameters
        ----------
        file
            Name of the .npz file to read.

        Returns
        -------
        numpy.lib.npyio.NpzFile
        '''
        dataread = np.load(file, allow_pickle=True)
        self.colname = sorted(dataread)
        return dataread


    def get_flux(self,
                 lineselect: str,
                 fluxsel: str='ftot',
                 **kwargs) -> dict[Literal['flx', 'flxerr'], np.ndarray]:
        ''' 
        Get flux and error of a given line.

        Parameters
        ----------
        lineselect
            Which line to grab.
        fluxsel
            Optional. Which type of flux to grab, as defined in :py:func:`~q3dfit.q3dcollect.q3dcollect`.
            Options are 'ftot', 'fc1', 'fc1pk', 'fc2', 'fc2pk', ..., 'fcN', 'fcNpk' for N total components.
            Default is 'ftot'.

        Returns
        -------
        dict[Literal['flx', 'flxerr'], numpy.ndarray]
            Each key contains an array of size (ncols, nrows, ncomp).

        '''
        if 'FLUXSEL' in kwargs:
            fluxsel = kwargs['FLUXSEL']
        if lineselect not in self.lines:
            raise ValueError(f'Line {lineselect} is not present in the data.')
        emlflx = self.data['emlflx'].item()
        emlflxerr = self.data['emlflxerr'].item()
        dataout = {'flux': emlflx[fluxsel][lineselect],
                   'fluxerr': emlflxerr[fluxsel][lineselect]}
        return dataout


    def get_ncomp(self,
                  lineselect: str) -> np.ndarray:
        '''
        Get # components fit to a given line.

        Parameters
        ----------
        lineselect
            Which line to grab.

        Returns
        -------
        numpy.ndarray
            Array of size (ncols, nrows) containing the number of components fit to each spaxel.
        '''
        if lineselect not in self.lines:
            raise ValueError(f'Line {lineselect} is not present in the data.')
        return (self.data['emlncomp'].item())[lineselect]


    def get_sigma(self,
                  lineselect: str,
                  compsel: int=1,
                  **kwargs) -> dict[Literal['sig', 'sigerr'], np.ndarray]:
        '''
        Get sigma and error of a given line and component.

        Parameters
        ----------
        lineselect
            Which line to grab.
        compsel
            Optional. Which component to grab. Default is 1.

        Returns
        -------
        dict[Literal['sig', 'sigerr'], numpy.ndarray]
            Each key contains an array of size (ncols, nrows).

        '''
        if 'COMPSEL' in kwargs:
            compsel = kwargs['COMPSEL']
        if lineselect not in self.lines:
            raise ValueError(f'Line {lineselect} is not present in the data.')
        emlsig = self.data['emlsig'].item()
        emlsigerr = self.data['emlsigerr'].item()
        csel = 'c'+str(compsel)
        dataout = {'sig': emlsig[csel][lineselect],
                   'sigerr': emlsigerr[csel][lineselect]}
        return dataout


    def get_wave(self,
                 lineselect: str,
                 compsel: int=1,
                 **kwargs) -> dict[Literal['wav', 'waverr'], np.ndarray]:
        '''
        Get central wavelength and error of a given line and component.

        Parameters
        ----------
        lineselect
            Which line to grab.
        compsel
            Optional. Which component to grab. Default is 1.

        Returns
        -------
        dict[Literal['wav', 'waverr'], numpy.ndarray]
            Each key contains an array of size (ncols, nrows).
        '''
        if 'COMPSEL' in kwargs:
            compsel = kwargs['COMPSEL']
        if lineselect not in self.lines:
            raise ValueError(f'Line {lineselect} is not present in the data.')
        emlwav = self.data['emlwav'].item()
        emlwaverr = self.data['emlwaverr'].item()
        csel = 'c'+str(compsel)
        dataout = {'wav': emlwav[csel][lineselect],
                   'waverr': emlwaverr[csel][lineselect]}
        return dataout


class OneLineData:
    '''
    Parse all line data for a given emission line from a LineData object.

    Parameters
    -----------
    linedata
        All raw line data from the fit.
    line
        Which line to grab.
    quiet
        Optional. Suppress messages. Default is True.

    Attributes
    ----------
    ncols : int
        Copy of :py:attr:`~q3dfit.q3dpro.LineData.ncols`.
    nrows : int
        Copy of :py:attr:`~q3dfit.q3dpro.LineData.nrows`.
    bad : float
        Copy of :py:attr:`~q3dfit.q3dpro.LineData.bad`.
    dataDIR : str
        Copy of :py:attr:`~q3dfit.q3dpro.LineData.dataDIR`.
    target_name : str
        Copy of :py:attr:`~q3dfit.q3dpro.LineData.target_name`.
    flux : numpy.ndarray
        Total flux of each spaxel and component for this line.
    fpklux : numpy.ndarray
        Peak flux of each spaxel and component for this line.
    sig : numpy.ndarray
        Sigma of each spaxel and component for this line.
    wave : numpy.ndarray
        Central wavelength of each spaxel and component for this line.
    ncomp : numpy.ndarray
        Number of components fit to each spaxel for this line.
    cvdf_zref : float
        Reference redshift for computing velocities. Defined in :py:meth:`~q3dfit.q3dpro.OneLineData.calc_cvdf`.
    cvdf_vel : numpy.ndarray
        Model velocities. 1D array. Defined in :py:meth:`~q3dfit.q3dpro.OneLineData.calc_cvdf`.
    vdf : numpy.ndarray
        Velocity distribution in flux space. 3D data with two dimensions of
        imaging plane and third of model points. Defined in :py:meth:`~q3dfit.q3dpro.OneLineData.calc_cvdf`.
    cvdf : numpy.ndarray
        Cumulative velocity distribution function. 3D data with two dimensions of
        imaging plane and third of model points. Defined in :py:meth:`~q3dfit.q3dpro.OneLineData.calc_cvdf`.
    cvdf_nmod : int
        Number of model points. Defined in :py:meth:`~q3dfit.q3dpro.OneLineData.calc_cvdf`.
    '''
    def __init__(self,
                 linedata: LineData,
                 line: str,
                 quiet: bool=True,
                 **kwargs):

        # back-compatible with old code
        if 'lineselect' in kwargs:
            line = kwargs['lineselect']

        self.quiet = quiet

        # inherit some stuff from linedata
        self.ncols = linedata.ncols
        self.nrows = linedata.nrows
        self.bad = linedata.bad
        self.dataDIR = linedata.dataDIR
        self.target_name = linedata.target_name
        
        if line not in linedata.lines:
            raise ValueError(f'Line {line} is not present in the data.')
        self.line = line

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
                (linedata.get_flux(line, FLUXSEL='fc'+str(i+1)))['flux']
            self.pkflux[:, :, i] = \
                (linedata.get_flux(line, FLUXSEL='fc'+str(i+1)+'pk'))['flux']
            self.sig[:, :, i] = \
                (linedata.get_sigma(line, COMPSEL=i+1))['sig']
            self.wave[:, :, i] = \
                (linedata.get_wave(line, COMPSEL=i+1))['wav']
        # No. of components on a spaxel-by-spaxel basis
        self.ncomp = linedata.get_ncomp(line)


    def calc_cvdf(self,
                  zref: float,
                  vlimits: ArrayLike=[-1e4, 1e4],
                  vstep: float=1.):
        '''
        Compute cumulative velocity distribution function for this line, for each spaxel.

        Parameters
        -----------
        zref
            Reference redshift for computing velocities.
        vlimits
            Limits for model velocities, in km/s.
        vstep
            Step size for model velocities, in km/s.
        ''' 
        self.cvdf_zref = zref
        
        # these are allegedly the smallest numbers recognized
        minexp = -310
        # this is the experimentally determined limit for when
        # I can take a log of a 1e-minexp
        # mymin = np.exp(minexp)

        # establish the velocity array from the inputs or from the defaults
        modvel = np.arange(vlimits[0], vlimits[1] + vstep, vstep)
        beta = modvel/c.to('km/s').value
        dz = np.sqrt((1. + beta)/(1. - beta)) - 1.

        # central (rest) wavelength of the line in question
        listlines = linelist([self.line])
        cwv = listlines['lines'].value[0]
        modwaves = cwv*(1. + dz)*(1. + zref)

        # output arrays
        size_cube = np.shape(self.pkflux)
        nmod = np.size(modvel)

        vdf = np.zeros((size_cube[0], size_cube[1], nmod))
        cvdf = np.zeros((size_cube[0], size_cube[1], nmod))
        for i in range(np.max(self.ncomp)):
            rbpkflux = np.repeat((self.pkflux[:, :, i])[:, :, np.newaxis], nmod, axis=2)
            rbsigma = np.repeat((self.sig[:, :, i])[:, :, np.newaxis], nmod, axis=2)
            rbpkwave = np.repeat((self.wave[:, :, i])[:, :, np.newaxis], nmod, axis=2)
            rbncomp = np.repeat(self.ncomp[:, :, np.newaxis], nmod, axis=2)
            rbmodwave = \
                np.broadcast_to(modwaves, (size_cube[0], size_cube[1], nmod))

            inz = ((rbsigma > 0) & (rbsigma != np.nan) &
                (rbpkwave > 0) & (rbpkwave != np.nan) &
                (rbpkflux > 0) & (rbpkflux != np.nan) &
                (rbncomp > i))
            if np.sum(inz) > 0:
                exparg = np.zeros((size_cube[0], size_cube[1], nmod)) - minexp
                exparg[inz] = ((rbmodwave[inz]/rbpkwave[inz] - 1.) /
                            (rbsigma[inz]/c.to('km/s').value))**2. / 2.
                i_no_under = (exparg < -minexp)
                if np.sum(i_no_under) > 0:
                    vdf[i_no_under] += rbpkflux[i_no_under] * \
                        np.exp(-exparg[i_no_under])

        # size of each model bin
        dmodwaves = modwaves[1:nmod] - modwaves[0:nmod-1]
        # supplement with the zeroth element to make the right length
        dmodwaves = np.append(dmodwaves[0], dmodwaves)
        # rebin to full cube
        rbdmodwaves = \
            np.broadcast_to(dmodwaves, (size_cube[0], size_cube[1], nmod))
        fluxnorm = vdf * rbdmodwaves
        #fluxnormerr = emlcvdf['fluxerr'][line]*dmodwaves
        fluxint = np.repeat((np.sum(fluxnorm, 2))[:, :, np.newaxis], nmod, axis=2)
        inz = fluxint != 0
        if np.sum(inz) > 0:
            fluxnorm[inz] /= fluxint[inz]
            #fluxnormerr[inz] /= fluxint[inz]

        cvdf[:, :, 0] = fluxnorm[:, :, 0]
        for i in range(1, nmod):
            cvdf[:, :, i] = cvdf[:, :, i-1] + fluxnorm[:, :, i]
            #emlcvdf['cvdferr'][line] = fluxnormerr

        self.cvdf_vel = modvel
        self.vdf = vdf
        self.cvdf = cvdf
        self.cvdf_nmod = len(modvel)


    def calc_cvdf_vel(self,
                      pct: float,
                      calc_from_posvel: bool=True) -> np.ndarray:
        '''
        Compute a velocity at % pct from one side of the CVDF.

        Parameters
        -----------
        pct
            Percentage at which to calculate velocity. Must be between 0 and 100.
        calc_from_posvel
            Optional. If True, the zero-point is at positive velocities. Default is True.

        Returns
        -------
        numpy.ndarray
            Array of size (ncols, nrows) containing the velocity at the given percentile.
        '''
        if pct < 0 or pct > 100:
            raise ValueError('Percentile must be between 0 and 100.')
 
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


    def make_cvdf_map(self,
                      pct: float,
                      velran: ArrayLike=[-3e2, 3e2],
                      calc_from_posvel: bool=True,
                      cmap: str='RdYlBu_r',
                      center: ArrayLike=[0.,0.],
                      markcenter: Optional[ArrayLike]=None,
                      saveData: bool=False,
                      saveFormat: str='png',
                      dpi: int=100,
                      **kwargs):
        '''
        Make a map of the cumulative velocity distribution function at a given percentile.

        Parameters
        ----------
        pct
            Percentile to plot. Must be between 0 and 100.
        velran
            Optional. Range of velocities to plot, in km/s. Default is [-3e2, 3e2].
        calc_from_posvel
            Optional. If True, the zero-point is at positive velocities. Default is True.
        cmap
            Optional. Colormap to use. Default is 'RdYlBu_r'.
        center
            Optional. Location, in zero-offset spaxel coordinates, from which to compute
            axis offsets. Default is [0., 0.].
        markcenter
            Optional. Location, in axes coordinates, to mark with a star.
            Default is None.
        outfile
            Optional. Name of file to save the plot. Default is None.
        outformat
            Optional. Format of the output file. Default is 'png'.
        outdpi
            Optional. DPI of the output file. Default is 100.
        '''
        # back-compatibility
        if 'outfile' in kwargs:
            saveData = kwargs['outfile']

        if pct < 0 or pct > 100:
            raise ValueError('Percentile must be between 0 and 100.')

        pixVals = self.calc_cvdf_vel(pct, calc_from_posvel)

        #kpc_arcsec = cosmo.kpc_proper_per_arcmin(self.cvdf_zref).value/60.

        # single-offset column and row values
        cols = np.arange(1, self.ncols+1, dtype=float)
        rows = np.arange(1, self.nrows+1, dtype=float)
        cols_cent = (cols - center[0])
        rows_cent = (rows - center[1])
        # This makes the axis span the spaxel values, with integer coordinate
        # being a pixel center. So a range of [1,5] spaxels will have an axis
        # range of [0.5,5.5]. This is what the extent keyword to imshow expects.
        # https://matplotlib.org/stable/tutorials/intermediate/imshow_extent.html
        xran = np.array([cols_cent[0], cols_cent[self.ncols-1]+1.]) - 0.5
        yran = np.array([rows_cent[0], rows_cent[self.nrows-1]+1.]) - 0.5

        #if axisunit == 'kpc' and platescale is not None:
        #    kpc_pix = np.median(kpc_arcsec)*platescale
        #    xcolkpc = xcol*kpc_pix
        #    ycolkpc = ycol*kpc_pix
        #    xcol, ycol = xcolkpc, ycolkpc
        #    XYtitle = 'Relative distance [kpc]'

        XYtitle = 'Spaxel'
        fig, ax = plt.subplots()

        vticks = [velran[0], velran[0]/2., 0., velran[1]/2., velran[1]]
        nticks = 4
        cmap_r = cm.get_cmap(cmap)
        cmap_r.set_bad(color='black')
        # cmap_r.set_bad(color=self.map_bkg)
        _display_pixels_wz(cols_cent, rows_cent, pixVals, ax, CMAP=cmap, 
                          COLORBAR=True, VMIN=velran[0], VMAX=velran[1],
                          TICKS=vticks, NTICKS=nticks, XRAN=xran, YRAN=yran)
        if markcenter is not None:
            ax.errorbar(markcenter[0], markcenter[1], color='black', mew=1,
                        mfc='red', fmt='*', markersize=10, zorder=2)
        ax.set_xlabel(XYtitle, fontsize=12)
        ax.set_ylabel(XYtitle, fontsize=12)
        ax.set_title(f'{self.target_name} {self.line} v{int(pct)} (km/s)',
            fontsize=16, pad=45)
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
        fig.set_dpi(dpi)

        if saveData:
            pltsave_name = f'{self.target_name}-{self.line}-v{int(pct)}-map.{saveFormat}'
            q3dutil.write_msg(f'Saving {pltsave_name} to {self.dataDIR}.', quiet=self.quiet)
            plt.savefig(os.path.join(self.dataDIR, pltsave_name))
        plt.show()

        return


class ContData:
    '''
    Read in and store continuum data for all spaxels in the numpy save file created by
    :py:meth:`~q3dfit.q3dcollect.q3dcollect`.

    Parameters
    ----------
    q3di

    Attributes
    ----------

    '''

    def __init__(self,
                 q3di: q3din.q3din):

        filename = q3di.label+'.cont.npy'
        datafile = os.path.join(q3di.outdir, filename)
        if not os.path.exists(datafile):
            raise FileNotFoundError(f'Continuum file {filename} does not exist.')

        self.data = self._read_npy(datafile)
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


    def _read_npy(self,
                  datafile: str) -> dict:
        '''
        Load numpy save file.
        '''
        dataout = np.load(datafile, allow_pickle=True).item()
        self.colname = dataout.keys()
        return dataout


def _display_pixels_wz(x: np.ndarray,
                       y: np.ndarray,
                       datIN: np.ndarray,
                       AX: plt.Axes,
                       VMIN: Optional[float]=None,
                       VMAX: Optional[float]=None,
                       XRAN: Optional[ArrayLike]=None,
                       YRAN: Optional[ArrayLike]=None,
                       PLOTLOG: bool=False,
                       CMAP: str='RdYlBu',
                       COLORBAR: bool=False,
                       TICKS: Optional[list]=None,
                       NTICKS: int=3,
                       AUTOCBAR: bool=False,
                       SKIPTICK: bool=False):
    """
    Adapted from v1.1.7 version (circa 2017) of a routine by Michele Cappellari,
    by Weizhe Liu and Yuzo Ishikawa.
    
    Display vectors of square pixels at coordinates (x,y) coloured with "val".
    An optional rotation around the origin can be applied to the whole image.

    The pixels are assumed to be taken from a regular cartesian grid with
    constant spacing (like an axis-aligned image), but not all elements of
    the grid are required (missing data are OK).

    This routine is designed to be fast even with large images and to produce
    minimal file sizes when the output is saved in a vector format like PDF.

    Parameters
    ----------
    x
        2D array with the x-coordinates of the pixels.
    y
        2D array with the y-coordinates of the pixels.
    datIN
        2D array with the values of the pixels.
    AX
        The matplotlib axes to plot the image.
    VMIN
        Optional. Minimum value of the color scale. Default is np.nanmin(datIN).
    VMAX
        Optional. Maximum value of the color scale. Default is np.nanmax(datIN).
    XRAN
        Optional. Range of the x-axis: [xmin, xmax]. Default is [np.min(x), np.max(x)].
    YRAN
        Optional. Range of the y-axis: [ymin, ymax]. Default is [np.min(y), np.max(y)].
    PLOTLOG
        Optional. If True, plot the logarithm of the data. Default is False.
    CMAP
        Optional. Colormap. Default is 'RdYlBu'.
    COLORBAR
        Optional. If True, plot a colorbar. Default is False.
    TICKS
        Optional. List of ticks for the colorbar. Default is None. If None, the
        ticks are automatically determined using NTICKS.
    NTICKS
        Optional. Number of ticks for the colorbar. Default is 3. Ignored if TICKS is not None.
    AUTOCBAR
        Optional. If True, and TICKS is None, the colorbar ticks are determined using 
        :py:class:`matplotlib.ticker.MaxNLocator'. If False, :py:class:`matplotlib.ticker.LinearLocator`
        is used. Default is False.
    SKIPTICK
        Optional. If True, do not plot ticks. Default is False.
    """
    if VMIN is None:
        VMIN = np.nanmin(datIN)

    if VMAX is None:
        VMAX = np.nanmax(datIN)

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
    #AX.set_facecolor('black')

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


def _lgerr(x1: ArrayLike | float,
           x2: ArrayLike | float,
           x1err: ArrayLike | float,
           x2err: ArrayLike | float) -> list[np.ndarray]:
    '''
    Estimate the lower- and upper-errors in the base-10 log of the ratio of two numbers x1 and x2 from 
    linear errors. 
    '''
    x1,x2,x1err,x2err = np.array(x1),np.array(x2),np.array(x1err),np.array(x2err)
    yd0 = x1/x2
    yd = np.log10(yd0)
    yderr0 = ((x1err/x2)**2+(x2err*x1/x2**2)**2)**0.5
    lgyerrup = np.log10(yd0+yderr0) - yd
    lgyerrlow = yd - np.log10(yd0-yderr0)
    return [lgyerrlow,lgyerrup]


def _clean_mask(dataIN: np.ndarray,
                BAD: float=np.nan) -> np.ndarray:
    '''
    Create a mask for bad pixels in an array. Set good pixels to 1 and bad pixels to BAD.
    '''
    dataOUT = copy.copy(dataIN)
    dataOUT[dataIN != BAD] = 1
    dataOUT[dataIN == BAD] = np.nan
    return dataOUT


def _snr_cut(dataIN: np.ndarray,
             errIN: np.ndarray,
             snrcut: float=2,
             **kwargs) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    '''
    Apply a signal-to-noise ratio cut to data. Pixels with SNR < SNRCUT are set to np.nan.
    
    Returns
    -------
    numpy.ndarray
        Data with bad pixels set to np.nan.
    numpy.ndarray
        Indices of good pixels.
    numpy.ndarray
        Indices of bad pixels.
    '''
    # back-compatibility
    if 'SNRCUT' in kwargs:
        snrcut = kwargs['SNRCUT']
    snr = dataIN/errIN
    gud_indx = np.where(snr >= snrcut)
    bad_indx = np.where(snr < snrcut)
    dataOUT = copy.copy(dataIN)
    dataOUT[snr < snrcut] = np.nan
    dataOUT[np.where(np.isnan(errIN))] = np.nan
    return dataOUT, gud_indx, bad_indx


def _save_to_fits(dataIN: np.ndarray,
                  hdrIN: Optional[fits.Header],
                  savepath: str):
    '''
    Save data to a FITS file.
    '''
    if hdrIN is None:
        hdu_0 = fits.PrimaryHDU(dataIN)
    else:
        hdu_0 = fits.PrimaryHDU(dataIN, header=hdrIN)
    hdul = fits.HDUList([hdu_0])
    hdul.writeto(savepath,overwrite=True)


def _set_figsize(dim: ArrayLike,
                 plotsize: ArrayLike) -> np.ndarray:
    '''
    Set the size of a figure based on the dimensions of the data and the desired plot size.

    Parameters
    ----------
    dim
        A 2-element ArrayLike giving the dimensions of the data.
    plotsize
        A 2-element ArrayLike giving the desired size of the plot in units of ...?

    Returns
    -------
    numpy.ndarray
        A 2-element array giving the size of the figure in units of ...?
    '''
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
