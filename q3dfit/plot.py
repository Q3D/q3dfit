#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import annotations

from typing import Literal, Optional
from numpy.typing import ArrayLike
from copy import copy

from astropy.constants import c
from astropy import units as u
import math
import matplotlib.pyplot as plt
import numpy as np
from matplotlib import gridspec, rcParams
from matplotlib.ticker import StrMethodFormatter

from q3dfit.contfit import readcf
from q3dfit.exceptions import InitializationError
from q3dfit.q3din import q3din
from q3dfit.spectConvol import spectConvol
from q3dfit.q3dout import q3dout
from q3dfit.q3dutil import lmlabel

def plotcont(q3do: q3dout,
             savefig: bool=False,
             outfile: Optional[str]=None,
             argssavefig: dict={'bbox_inches': 'tight', 'dpi': 300},
             questfit: bool=False,
             q3di: Optional[q3din]=None,
             compspec: Optional[np.ndarray]=None,
             complabs: Optional[list]=None,
             compcols: Optional[list]=None,
             title: Optional[str]=None,
             xran: Optional[list]=None,
             zeroymin: bool=False,
             xstyle: Literal['log', 'lin']='lin',
             ystyle: Literal['log', 'lin']='lin',
             figsize: tuple=(10, 5),
             waveunit_in: Literal['micron','Angstrom']='micron',
             waveunit_out: Optional[str]=None,
             fluxunit_out: Literal['flambda', 'lambdaflambda', 'fnu'] = 'flambda',
             mode: Literal['dark', 'light'] = 'dark'
             ):
    '''
    Created on Tue Jun  1 13:32:37 2021
    @author: annamurphree

    Plots continuum fit of optical data (fit by fitqsohost or ppxf)
    or IR data (fit by questfit).

    Parameters
    ----------
    q3do
        :py:class:`~q3dfit.q3dout.q3dout` object containing results of fit.
    savefig
        Optional. If True, saves the plot to a file. Defaults to False.
    outfile
        Optional. Full path and name of output plot. Defaults to None, which
        means no output file is created.
    argssavefig
        Optional. Dictionary of arguments to pass to 
        :py:meth:`~matplotlib.pyplt.savefig()`. Defaults to
        {'bbox_inches': 'tight', 'dpi': 300}.
    questfit
        Optional. If True, indicates that the fit is an IR spectrum fit by
        questfit. Defaults to False.
    q3di
        Optional. :py:class:`~q3dfit.q3din.q3din` object containing fit initialization.
        Only needed for IR spectra fit by questfit. Defaults to None.
    compspec
        Optional. Array of component spectra to overplot on continuum data. 
        If None, no components are overplotted.
    complabs
        Optional. List of labels for component spectra. Defaults to None, which
        means labels are automatically generated as 'Component 1', 'Component 2', etc.
    compcols
        Optional. List of colors for component spectra. Defaults to None, which
        means colors are automatically generated as 'C0', 'C1', etc.
    title
        Optional. Title for the plot. Defaults to None.
    xran
        Optional. Range of x-axis (wavelength) to plot. If None, applies the
       `fitrange` attribute of :py:class:`~q3dfit.q3dout.q3dout`.
    zeroymin
        Optional. If True, sets the minimum y-axis value to 0. Defaults to False.
    xstyle
        Optional. Style of x-axis scale, either 'log' or 'lin'. Defaults to 'lin'.
    ystyle
        Optional. Style of y-axis scale, either 'log' or 'lin'. Defaults to 'lin'.
    figsize
        Optional. Size of the figure in inches, specified as a tuple (width, height).
        Defaults to (10, 5).
    mode
        Optional. Plot style, either 'dark' or 'light'. Defaults to 'dark'.
        This affects the background color and text color of the plot.
    waveunit_in
        Optional. Input wavelength unit, either 'micron' or 'Angstrom'.
        Defaults to 'micron'.
    waveunit_out
        Optional. Output wavelength unit, either 'micron' or 'Angstrom'.
        If None, defaults to waveunit_in.
    fluxunit_out
        Optional. Output flux unit, either 'flambda', 'lambdaflambda', or 'fnu'.
        Defaults to 'flambda'. The input flux unit is always 'flambda'. Choosing
        'lambdaflambda' multiplies the flux by wavelength (in the units of 
        waveunit_out), and 'fnu' multiplies the flux by wavelength^2/c, 
        where c is the speed of light in the units of waveunit_out.
    '''

    rcParamsOrig = rcParams.copy()

    # dark mode just for fun:
    if mode == 'dark':
        pltstyle = 'dark_background'
        dcolor = 'w'
    else:
        pltstyle = 'seaborn-v0_8-ticks'
        dcolor = 'k'

    wave = q3do.wave.copy()
    specstars = copy(q3do.cont_dat)
    modstars = copy(q3do.cont_fit)

    if waveunit_out is None:
        # if no output wavelength unit is specified, use the input wavelength unit
        waveunit_out = waveunit_in
    wavein = wave.copy()*getattr(u,waveunit_in)
    waveout = wavein.copy().to(waveunit_out)

    # for optical spectra fit by fitqsohost or ppxf:
    if not questfit:

        if compspec is not None:
            ccompspec = copy(compspec)
            if len(ccompspec) > 1:
                ncomp = len(ccompspec)
            else:
                ncomp = 1
            if compcols is None:
                compcols = ['C' + str(i) for i in range(ncomp)]
            if complabs is None:
                complabs = ['Component ' + str(i + 1) for i in range(ncomp)]
        else:
            ncomp = 0

        if xran is None:
            xran = copy(q3do.fitrange)

        if waveunit_in == 'Angstrom' and waveunit_out == 'micron':
            # convert angstrom to microns
            xran = list(np.divide(xran, 10**4))
        elif waveunit_in == 'micron' and waveunit_out == 'Angstrom':
            # convert microns to angstroms
            xran = list(np.multiply(xran, 10**4))

        if fluxunit_out == 'lambdaflambda':
            # multiply the flux by wavelength
            specstars = list(np.multiply(specstars, wave))
            modstars = list(np.multiply(modstars, wave))
            if ncomp > 0:
                for i in range(0, ncomp):
                    ccompspec[i] = list(np.multiply(ccompspec[i], wave))
            ytit = '$\lambda$F$_\lambda$'
        elif fluxunit_out == 'fnu':
            # multiply the flux by wavelength^2/c
            specstars = \
                list(np.multiply(specstars,
                                 np.divide(np.multiply(wavein.value, waveout.value), 
                                           c.to(waveunit_out+'/s').value)))
            modstars = \
                list(np.multiply(modstars,
                                 np.divide(np.multiply(wavein.value, waveout.value), 
                                           c.to(waveunit_out+'/s').value)))
            if ncomp > 0:
                for i in range(0, ncomp):
                    ccompspec[i] = \
                        list(np.multiply(ccompspec[i],
                                         np.divide(np.multiply(wavein.value, waveout.value), 
                                           c.to(waveunit_out+'/s').value)))
            ytit = 'F$_\u03BD$'
        else:
            ytit = 'F$_\lambda$'

        # plot on a log scale:
        if xstyle == 'log' or ystyle == 'log':
            plt.style.use(pltstyle)
            # CB: Otherwise the background becomes black and the axes ticks
            # unreadable when saving the figure
            if mode == 'light':
                rcParams['savefig.facecolor'] = 'white'
            fig = plt.figure(figsize=figsize)
            # fig = plt.figure()
            plt.axis('off')  # so the subplots don't share a y-axis

            fig.add_subplot(1, 1, 1)
            ydat = specstars
            ymod = modstars
            # plotting
            plt.xlim(xran[0], xran[1])

            fig.axes[0].axis('off')  # so the subplots don't share a y-axis
            fig.axes[1].axis('off')  # so the subplots don't share a y-axis

            gs = fig.add_gridspec(4, 1)
            ax1 = fig.add_subplot(gs[:3, :])
            # ax1.legend(ncol=2)
            if xstyle == 'log':
                ax1.set_xscale('log')
            # ax1.set_xticklabels([])
            if ystyle == 'log':
                ax1.set_yscale('log')

            ax1.set_ylabel(ytit, fontsize=20)
            #if title == 'QSO':
            #    ax1.set_ylim(10e-7)

            # actually plotting
            plt.plot(waveout.value, ydat, dcolor, linewidth=1)
            plt.plot(waveout.value, ymod, 'r', linewidth=2, label='Total')
            if ncomp > 0:
                for i in range(0, ncomp):
                    plt.plot(waveout.value, ccompspec[i], compcols[i], linewidth=2,
                             label=complabs[i])

            # tick formatting
            yticks_used = ax1.get_yticks()
            ylim_used = ax1.get_ylim()
            yticks_used = np.append(np.append(ylim_used[0], yticks_used),
                                    ylim_used[1])
            ax1.set_yticks(yticks_used)
            ax1.set_ylim(ylim_used)

            ax1.minorticks_on()
            #ax1.tick_params(which='major', length=20, pad=10, labelsize=10)
            #ax1.tick_params(which='minor', length=7, labelsize=8)


            l = ax1.legend(loc='upper right', fontsize=16)
            for text in l.get_texts():
                text.set_color(dcolor)
            ax2 = fig.add_subplot(gs[-1, :], sharex=ax1)
            ax2.plot(waveout.value, np.divide(specstars, modstars), color=dcolor)
            ax2.axhline(1, color='grey', linestyle='--', alpha=0.7, zorder=0)
            ax2.set_ylabel('Data/Model', fontsize=19)

#            ax2.tick_params(which='major', length=20, pad=20, labelsize=9)
#            ax2.tick_params(which='minor', length=7, labelsize=8)

            if waveunit_out == 'micron':
                ax2.set_xlabel('Wavelength ($\mu$m)', fontsize=20)
            elif waveunit_out == 'Angstrom':
                ax2.set_xlabel('Wavelength ($\AA$)', fontsize=20)
            gs.update(wspace=0.0, hspace=0.05)
            plt.gcf().subplots_adjust(bottom=0.1)

            if title is not None:
                plt.suptitle(title, fontsize=20)

            if savefig and outfile is not None:
                plt.savefig(outfile[0], **argssavefig)

        elif xstyle == 'lin' or ystyle == 'lin':
            dxran = xran[1] - xran[0]
            xran1 = [xran[0], xran[0] + np.around(dxran/3.0, 3)]
            xran2 = [xran[0] + np.around(dxran/3.0, 3),
                     xran[0] + 2.0 * np.around(dxran/3.0, 3)]
            xran3 = [xran[0] + 2.0 * np.around(dxran/3.0, 3),
                     xran[1]]
            i1 = [None]
            i2 = [None]
            i3 = [None]

            i1.pop(0)
            i2.pop(0)
            i3.pop(0)

            ydat = specstars
            ymod = modstars

            for i in range(0, len(waveout.value)):
                if waveout.value[i] > xran1[0] and waveout.value[i] < xran1[1]:
                    i1.append(i)
                if waveout.value[i] > xran2[0] and waveout.value[i] < xran2[1]:
                    i2.append(i)
                if waveout.value[i] > xran3[0] and waveout.value[i] < xran3[1]:
                    i3.append(i)

            maxthresh = 0.2
            ntop = 20
            nbottom = 20
            if len(waveout.value) < 100:
                ntop = 10
                nbottom = 10
            ++ntop
            --nbottom

            if waveunit_out == 'micron':
                xtit = 'Observed Wavelength ($\mu$m)'
            elif waveunit_out == 'Angstrom':
                xtit = 'Observed Wavelength ($\AA$)'

            plt.style.use(pltstyle)
            fig = plt.figure(figsize=figsize)
            plt.axis('off')  # so the subplots don't share a y-axis

            maximum = 0
            minimum = 0
            ''
            idict = {1: i1, 2: i2, 3: i3}
            xrans = {1: xran1, 2: xran2, 3: xran3}
            for group in range(1, 4):
                if len(idict[group]) > 0:
                    fig.add_subplot(3, 1, group)

                    # finding min/max values at indices from idict
                    dat_et_mod = np.concatenate((ydat[idict[group]],
                                                 ymod[idict[group]]))
                    maximum = np.nanmax(dat_et_mod)
                    minimum = np.nanmin(dat_et_mod)
                    # set min and max in yran
                    yran = [minimum, maximum]
                    if zeroymin:
                        yran[0] = 0.

                    # finding yran[1] aka max
                    ydi = np.zeros(len(idict[group]))
                    ydi = np.array(ydat)[idict[group]]

                    ymodi = np.zeros(len(idict[group]))
                    ymodi = np.array(ymod)[idict[group]]
                    y = np.array(ydi - ymodi)
                    ny = len(y)

                    iysort = np.argsort(y)
                    ysort = np.array(y)[iysort]

                    ymodisort = ymodi[iysort]
                    if ysort[ny - ntop] < ysort[ny - 1] * maxthresh:
                        yran[1] = np.nanmax(ysort[0:ny - ntop] +
                                            ymodisort[0:ny - ntop])

                    # plotting
                    plt.xlim(xrans[group][0], xrans[group][1])
                    plt.ylim(yran[0], yran[1])
                    plt.ylabel(ytit, fontsize=15)
                    if group == 3:
                        plt.xlabel(xtit, fontsize=15, labelpad=10)
                    if ystyle == 'log':
                        plt.yscale('log')

                    # tick formatting
                    plt.minorticks_on()
                    plt.tick_params(which='major', length=10, pad=5)
                    plt.tick_params(which='minor', length=5)
                    #if waveunit_out == 'micron':
                    #    xticks = np.arange(np.around(xrans[group][0],1)-0.025,
                    #                       np.around(xrans[group][1],1), 0.025)[:-1]
                    #    plt.xticks(xticks, fontsize=10)
                    #elif waveunit_out == 'Angstrom':
                    #    xticks = np.arange(math.floor(xrans[group][0]/100.0)*100,
                    #                       (math.floor(xrans[group][1]/100)*100)+100, 100)
                    #    plt.xticks(xticks, fontsize=10)
                    if np.nanmin(ydat) > 1e-10:
                        # this will fail if fluxes are very low (<~1e-10)
                        plt.yticks(np.arange(yran[0], yran[1],
                                             np.around((yran[1] - yran[0])/5.,
                                                   decimals=2)), fontsize=10)
                    else:
                        plt.yticks()

                    # actually plotting
                    plt.plot(waveout.value, ydat, dcolor, linewidth=1)

                    if ncomp > 0:
                        for i in range(0, ncomp):
                            plt.plot(waveout.value, ccompspec[i], compcols[i], 
                                     linewidth=2, label=complabs[i])

                    plt.plot(waveout.value, ymod, 'r', linewidth=2, label=title)
                    if group == 1:
                        plt.legend(loc='upper right')

            # more formatting
            plt.subplots_adjust(hspace=0.25)
            #plt.tight_layout(pad=5)
            #plt.gcf().subplots_adjust(bottom=0.1)

            if title is not None:
                plt.suptitle(title, fontsize=20)

            if savefig and outfile is not None:
                if len(outfile[0])>1:
                    plt.savefig(outfile[0], **argssavefig)
                else:
                    plt.savefig(outfile, **argssavefig)

            plt.show()

    # for IR spectra fit with questfit:
    else:

        comp_best_fit = q3do.ct_coeff['comp_best_fit']
        if xstyle == 'log' or ystyle == 'log':
            fig = plt.figure(figsize=figsize)
            gs = fig.add_gridspec(4,1)
            ax1 = fig.add_subplot(gs[:3, :])

            MIRgdlambda = wave #[q3do.ct_indx]
            MIRgdflux = q3do.spec #[q3do.ct_indx]
            MIRcontinuum = modstars #[q3do.ct_indx]

            if waveunit_in =='micron' and waveunit_out == 'Angstrom':
                # convert microns to angstroms
                MIRgdlambda = list(np.multiply(MIRgdlambda, 10**4))
            elif waveunit_in =='Angstrom' and waveunit_out == 'micron':
                # convert angstroms to microns
                MIRgdlambda = list(np.divide(MIRgdlambda, 10**4))

            if fluxunit_out == 'lambdaflambda':
                # multiply the flux by wavelength
                MIRgdflux = list(np.multiply(MIRgdflux, MIRgdlambda))
                MIRcontinuum = list(np.multiply(MIRcontinuum, MIRgdlambda))
                if len(comp_best_fit.keys()) > 0:
                    for i in range(0, len(comp_best_fit.keys())):
                        comp_best_fit[list(comp_best_fit.keys())[i]] = \
                            np.multiply(comp_best_fit[list(comp_best_fit.keys())[i]],
                                        MIRgdlambda)
                ytit = '$\lambda$F$_\lambda$'
            elif fluxunit_out == 'fnu':
                # multiply the flux by wavelength^2/c
                MIRgdflux = list(np.multiply(MIRgdflux, MIRgdlambda))
                MIRcontinuum = list(np.multiply(MIRgdflux, MIRgdlambda))
                ytit = 'F$_\u03BD$'
            else:
                ytit = 'F$_\lambda$'

            plt.style.use(pltstyle)

            ax1.plot(MIRgdlambda, MIRgdflux, label='Data',color=dcolor)
            ax1.plot(MIRgdlambda, MIRcontinuum, label='Model', color='r')

            if 'global_ext_model' in q3di.argscontfit:
                for i in np.arange(0,len(comp_best_fit.keys())-2,1):
                    ax1.plot(MIRgdlambda,
                                np.multiply(comp_best_fit[list(comp_best_fit.keys())[i]],
                                            np.multiply(comp_best_fit[list(comp_best_fit.keys())[-2]],
                                                        comp_best_fit[list(comp_best_fit.keys())[-1]])),
                                label=list(comp_best_fit.keys())[i],
                                linestyle='--',alpha=0.5)
            else:
                for i in np.arange(0,len(comp_best_fit.keys()),3):
                    ax1.plot(MIRgdlambda,
                                np.multiply(comp_best_fit[list(comp_best_fit.keys())[i]],
                                            np.multiply(comp_best_fit[list(comp_best_fit.keys())[i+1]],
                                                        comp_best_fit[list(comp_best_fit.keys())[i+2]])),
                                label=list(comp_best_fit.keys())[i],
                                linestyle='--',alpha=0.5)

                for comp_i in comp_best_fit.keys():
                        if 'ext' not in comp_i and 'abs' not in comp_i:
                            spec_out = comp_best_fit[comp_i]
                            if comp_i+'_ext' in comp_best_fit.keys():
                                spec_out *= comp_best_fit[comp_i+'_ext']
                            if comp_i+'_abs' in comp_best_fit.keys():
                                spec_out *= comp_best_fit[comp_i+'_abs']
                            plt.plot(MIRgdlambda, spec_out, label=comp_i,linestyle='--',alpha=0.5)


            #ax1.legend(ncol=2)
            ax1.legend(loc='upper right',bbox_to_anchor=(1.15, 1),prop={'size': 10})
            if xstyle == 'log':
                ax1.set_xscale('log')
            if ystyle == 'log':
                ax1.set_yscale('log')
            ax1.set_ylim(1e-4)
            ax1.set_ylabel(ytit, fontsize=12)

            ax2 = fig.add_subplot(gs[-1, :], sharex=ax1)
            ax2.plot(MIRgdlambda,np.divide(MIRgdflux,MIRcontinuum),color=dcolor)
            ax2.axhline(1, color='grey', linestyle='--', alpha=0.7, zorder=0)
            ax2.set_ylabel('Data/Model', fontsize=12)
            if waveunit_out == 'Angstrom':
                ax2.set_xlabel('Wavelength ($\AA$)', fontsize=12)
            elif waveunit_out == 'micron':
                ax2.set_xlabel('Wavelength ($\mu$m)', fontsize=12)
            gs.update(wspace=0.0, hspace=0.05)
            plt.suptitle('Total', fontsize=30)

        elif xstyle == 'lin' or ystyle == 'lin':

            if xran is None:
                xran = q3do.fitrange

            MIRgdlambda = wave #[q3do.ct_indx]
            MIRgdflux = q3do.spec #[q3do.ct_indx]
            MIRcontinuum = modstars #[q3do.ct_indx]

            xtit = ''
            if waveunit_in == 'microns' and waveunit_out == 'Angstrom':
                # convert wave list from microns to angstroms
                MIRgdlambda = list(np.multiply(MIRgdlambda, 10**4))
                xtit = 'Observed Wavelength ($\AA$)'
            elif waveunit_in == 'Angstrom' and waveunit_out == 'micron':
                # convert wave list from angstroms to microns
                MIRgdlambda = list(np.divide(MIRgdlambda, 10**4))
                xtit = 'Observed Wavelength ($\mu$m)'

            if fluxunit_out == 'lambdaflambda':
                # multiply the flux by wavelength
                MIRgdflux = list(np.multiply(MIRgdflux, MIRgdlambda))
                MIRcontinuum = list(np.multiply(MIRcontinuum, MIRgdlambda))
                if len(comp_best_fit.keys()) > 0:
                    for i in range(0, len(comp_best_fit.keys())):
                        comp_best_fit[list(comp_best_fit.keys())[i]] = \
                            list(np.multiply(comp_best_fit[list(comp_best_fit.keys())[i]],
                                             MIRgdlambda))
                ytit = '$\lambda$F$_\lambda$'
            elif fluxunit_out == 'fnu':
                # multiply the flux by wavelength
                MIRgdflux = list(np.multiply(MIRgdflux, MIRgdlambda))
                MIRcontinuum = list(np.multiply(MIRgdflux, MIRgdlambda))
                ytit = 'F$_\u03BD$'
            else:
                ytit = 'F$_\lambda$'

            wave = MIRgdlambda
            ydat = MIRgdflux
            ymod = MIRcontinuum

            dxran = xran[1] - xran[0]
            xran1 = [xran[0], xran[0] + np.around(dxran/3.0,3)]
            xran2 = [xran[0] + np.around(dxran/3.0,3), xran[0] + 2.0 * np.around(dxran/3.0,3)]
            xran3 = [xran[0] + 2.0 * np.around(dxran/3.0,3), xran[1]]
            i1 = [None]
            i2 = [None]
            i3 = [None]

            i1.pop(0)
            i2.pop(0)
            i3.pop(0)

            for i in range(0, len(wave)):
                if wave[i] > xran1[0] and wave[i] < xran1[1]:
                    i1.append(i)
                if wave[i] > xran2[0] and wave[i] < xran2[1]:
                    i2.append(i)
                if wave[i] > xran3[0] and wave[i] < xran3[1]:
                    i3.append(i)

            maxthresh = 0.2
            ntop = 20
            nbottom = 20
            if len(wave) < 100:
                ntop = 10
                nbottom = 10
            ++ntop
            --nbottom

            plt.style.use(pltstyle)
            fig = plt.figure(figsize=figsize)
            #fig = plt.figure()
            plt.axis('off')  # so the subplots don't share a y-axis

            maximum = 0
            minimum = 0

            idict = {1:i1, 2:i2, 3:i3}
            xrans = {1:xran1, 2:xran2, 3:xran3}
            for group in range(1,4):
                if len(idict[group]) > 0:
                    fig.add_subplot(3, 1, group)
                    ax = plt.subplot(3, 1, group)
                    # shrink current axis by 10% to fit legend on side
                    box = ax.get_position()
                    ax.set_position([box.x0, box.y0, box.width * 0.8, box.height])
                    # finding max value between ydat and ymod at indices from i1
                    for i in idict[group]:
                        bigboy = np.nanmax([ydat[i], ymod[i]])
                        if bigboy > maximum:
                            maximum = bigboy
                    # finding min
                    for i in idict[group]:
                        smallboy = np.nanmin([ydat[i], ymod[i]])
                        if smallboy < minimum:
                            minimum = smallboy
                    # set min and max in yran
                    yran = [minimum, maximum]
                    if zeroymin:
                        yran[0] = 0.

                    # finding yran[1] aka max
                    ydi = np.zeros(len(idict[group]))
                    ydi = np.array(ydat)[idict[group]]

                    ymodi = np.zeros(len(idict[group]))
                    ymodi = np.array(ymod)[idict[group]]
                    y = np.array(ydi - ymodi)
                    ny = len(y)

                    iysort = np.argsort(y)
                    ysort = np.array(y)[iysort]

                    ymodisort = ymodi[iysort]
                    if ysort[ny - ntop] < ysort[ny - 1] * maxthresh:
                        yran[1] = np.nanmax(ysort[0:ny - ntop] +
                                            ymodisort[0:ny - ntop])

                    # plotting
                    plt.xlim(xrans[group][0], xrans[group][1])
                    plt.ylim(yran[0], yran[1])
                    plt.ylabel(ytit, fontsize=15)
                    if group == 3:
                        plt.xlabel(xtit, fontsize=15, labelpad=10)
                    if ystyle == 'log':
                        plt.yscale('log')

                    # tick formatting
                    plt.minorticks_on()
                    plt.tick_params(which='major', length=10, pad=5)
                    plt.tick_params(which='minor', length=5)
                    if waveunit_out == 'micron':
                        xticks = np.arange(np.around(xrans[group][0]), np.around(xrans[group][1]), 1)
                        plt.xticks(xticks, fontsize=10)
                    elif waveunit_out == 'Angstrom':
                        xticks = np.arange(math.floor(xrans[group][0]/1000.0)*1000,
                                           (math.floor(xrans[group][1]/1000.0)*1000)+1000, 10000)
                        plt.xticks(xticks, fontsize=10)
                    if fluxunit_out != 'fnu':
                        # this will fail if fluxes are very low (<~1e-10)
                        plt.yticks(np.arange(yran[0], yran[1],
                                             np.around((yran[1] - yran[0])/5.,
                                                       decimals=2)),
                                   fontsize=10)
                    else:
                        plt.yticks()

                    # actually plotting
                    plt.plot(MIRgdlambda, MIRgdflux, label='Data',
                             color=dcolor)
                    plt.plot(MIRgdlambda, MIRcontinuum, label='Model',
                             color='red')

                    if 'global_ext_model' in q3di.argscontfit:
                           for i in np.arange(0,len(comp_best_fit.keys())-2,1):
                              plt.plot(MIRgdlambda,
                                       np.multiply(comp_best_fit[list(comp_best_fit.keys())[i]],
                                                   np.multiply(comp_best_fit[list(comp_best_fit.keys())[-2]],
                                                               comp_best_fit[list(comp_best_fit.keys())[-1]])),
                                       label=list(comp_best_fit.keys())[i],linestyle='--',alpha=0.5)

                    else:
                        for comp_i in comp_best_fit.keys():
                            if 'ext' not in comp_i and 'abs' not in comp_i:
                                spec_out = comp_best_fit[comp_i]
                                if comp_i+'_ext' in comp_best_fit.keys():
                                    spec_out *= comp_best_fit[comp_i+'_ext']
                                if comp_i+'_abs' in comp_best_fit.keys():
                                    spec_out *= comp_best_fit[comp_i+'_abs']
                                plt.plot(MIRgdlambda, spec_out, label=comp_i,linestyle='--',alpha=0.5)

                    if group == 1:
                        ax.legend(loc='upper right',bbox_to_anchor=(1.22, 1),prop={'size': 10})

        # more formatting
        plt.subplots_adjust(hspace=0.25)
        plt.tight_layout(pad=15)
        plt.gcf().subplots_adjust(bottom=0.1)
        plt.gcf().subplots_adjust(right=0.85)

        if title is not None:
            plt.suptitle(title, fontsize=20)

        if savefig and outfile is not None:
            if len(outfile[0])>1:
                plt.savefig(outfile[0], **argssavefig)
            else:
                plt.savefig(outfile, **argssavefig)

        plt.show()

    rcParams.update(rcParamsOrig)


def plotline(q3do: q3dout,
             nx: int=1,
             ny: int=1,
             figsize: tuple=(16,13),
             line: Optional[ArrayLike]=None,
             center_obs: Optional[ArrayLike]=None,
             center_rest: Optional[ArrayLike]=None,
             waveunit_in: Literal['micron','Angstrom']='micron',
             waveunit_out: Optional[str]=None,
             specConv: Optional[spectConvol]=None,
             size: float|ArrayLike=300.,
             savefig: bool=False,
             outfile: Optional[str]=None,
             argssavefig: dict={'bbox_inches': 'tight', 'dpi': 300}):
    """
    Plot emission line fit and output to JPG

    Parameters
    ----------
    q3do
        :py:class:`~q3dfit.q3dout.q3dout` object containing the output of the
        fit.
    nx
        Number of columns in the plot grid. Defaults to 1.
    ny
        Number of rows in the plot grid. Defaults to 1.
    figsize
        Size of the figure in inches, specified as a tuple (width, height).
    line
        Optional. List of lines in which to center the subplots. If None,
        the center of the plot window is determined from center_obs or
        center_rest.
    center_obs
        Optional. List of wavelengths in the observed frame to center the
        subplots. If None, the center of the plot window is determined from
        line or center_rest.
    center_rest
        Optional. List of wavelengths in the rest frame to center the
        subplots. If None, the center of the plot window is determined from
        line or center_obs.
    waveunit_in
        Optional. Input wavelength unit, either 'micron' or 'Angstrom'.
        Defaults to 'micron'.
    waveunit_out
        Optional. Output wavelength unit, either 'micron' or 'Angstrom'.
        If None, defaults to waveunit_in.
    specConv
        Optional. :py:class:`~q3dfit.spectConvol.spectConvol` object for
        spectral convolution. If None, no convolution is applied.
    size
        Optional. Size of the plot window in Angstroms. If a single float is
    savefig
        Optional. If True, saves the plot to a file. Defaults to False.
    outfile
        Optional. Full path and name of output plot. Defaults to None, which
        means no output file is created.
    argssavefig
        Optional. Dictionary of arguments to pass to 
        :py:meth:`~matplotlib.pyplt.savefig()`. Defaults to
        {'bbox_inches': 'tight', 'dpi': 300}.
    """

    rcParamsOrig = rcParams.copy()

    ncomp = q3do.maxncomp
    colors = ['Magenta', 'Green', 'Orange', 'Teal']

    if waveunit_out is None:
        # if no output wavelength unit is specified, use the input wavelength unit
        waveunit_out = waveunit_in

    wave = q3do.wave.copy()
    spectot = q3do.spec
    specstars = q3do.cont_dat
    modstars = q3do.cont_fit
    modlines = q3do.line_fit
    modtot = modstars + modlines

    if waveunit_in == 'Angstrom' and waveunit_out == 'micron':
        # convert angstrom to microns
        wave = list(np.divide(wave, 10**4))
    elif waveunit_in == 'micron' and waveunit_out == 'Angstrom':
        # convert microns to angstroms
        wave = list(np.multiply(wave, 10**4))


    # To-do: Allow output wavelengths in Angstrom
    #'waveunit_out' = 'micron'
    # if 'waveunit_out' in pltpar:
    #     if pltpar['waveunit_out = 'Angstrom':
    #         waveunit_out = 'Angstrom'

    # To-do: Get masking code from pltcont

    # lines
    linelist = q3do.linelist['lines']
    linelabel = q3do.linelist['name']
    linetext = q3do.linelist['linelab']
    # Sort in wavelength order
    isortlam = np.argsort(linelist)
    linelist = linelist[isortlam]
    linelabel = linelabel[isortlam]
    linetext = linetext[isortlam]

    #
    # Plotting parameters
    #
    # Look for line list, then determine center of plot window from fitted
    # wavelength
    if line is not None:
        sub_linlab = line
        linwav = np.empty(len(sub_linlab), dtype='float32')
        for i in range(0, len(sub_linlab)):
            # Get wavelength from zeroth component
            if sub_linlab[i] != '':
                lmline = lmlabel(sub_linlab[i])
                # if ncomp > 0
                if f'{lmline.lmlabel}_0_cwv' in q3do.param.keys():
                    linwav[i] = q3do.param[f'{lmline.lmlabel}_0_cwv']
                # otherwise
                else:
                    idx = np.where(q3do.linelist['name'] == sub_linlab[i])
                    if len(idx) > 0:
                        linwav[i] = q3do.linelist['lines'][idx] * \
                            (1. + q3do.zstar)
                    else:
                        raise InitializationError(f'Line {sub_linlab[i]} not fit.')
            else:
                linwav[i] = 0.
    # If linelist not present, get cwavelength enter of plot window from list
    # first option: wavelength center specified in observed (plotted) frame
    elif center_obs is not None:
        linwav = np.array(center_obs)
    # second option: wavelength center specified in rest frame, then converted
    # to observed (plotted) frame
    elif center_rest is not None:
        linwav = np.array(center_rest) * q3do.zstar
    else:
        raise InitializationError('LINE, CENTER_OBS, or CENTER_REST ' +
                                  'list not given in ARGSPLTLIN dictionary')
    nlin = len(linwav)
    # Size of plot in wavelength, in observed frame
    # case of single size for all panels
    if isinstance(size, float):
        size = np.full(nlin, size)  # default size currently 300 A ... fix for
    # case of array of sizes
    else:
        size = np.array(size)
        # other units!
    off = np.array([-1.*size/2., size/2.])
    off = off.transpose()

    plt.style.use('dark_background')
    fig = plt.figure(figsize=figsize)
    for i in range(0, nlin):

        outer = gridspec.GridSpec(ny, nx, wspace=0.2, hspace=0.2)
        inner = \
            gridspec.GridSpecFromSubplotSpec(2, 1,
                                             subplot_spec=outer[i],
                                             wspace=0.1, hspace=0,
                                             height_ratios=[4, 2],
                                             width_ratios=None)

        # create xran and ind
        linwavtmp = linwav[i]
        offtmp = off[i, :]
        xran = linwavtmp + offtmp
        ind = np.array([0])
        for h in range(0, len(wave)):
            if wave[h] > xran[0] and wave[h] < xran[1]:
                ind = np.append(ind, h)
        ind = np.delete(ind, [0])
        ct = len(ind)
        if ct > 0:
            # create subplots
            ax0 = plt.Subplot(fig, inner[0])
            ax1 = plt.Subplot(fig, inner[1])
            fig.add_subplot(ax0)
            fig.add_subplot(ax1)
            # create x-ticks
            xticks = np.linspace(xran[0],xran[1],num=5,endpoint=False)
            xticks = np.delete(xticks, [0])
            if waveunit_out == 'Angstrom':
                plt.gca().xaxis.set_major_formatter(StrMethodFormatter('{x:.0f}'))
            elif waveunit_out == 'micron':
                plt.gca().xaxis.set_major_formatter(StrMethodFormatter('{x:.2f}'))
            #     xticks = xticks * 1e4
            # create minor x-ticks
            xmticks = np.linspace(xran[0],xran[1],num=25,endpoint=False)
            xmticks = np.delete(xmticks, [0])
            #if waveunit_out == 'Angstrom':
            #     xmticks = xticks * 1e4
            # set ticks
            ax0.set_xticks(xticks)
            ax1.set_xticks(xticks)
            ax0.set_xticks(xmticks, minor=True)
            ax1.set_xticks(xmticks, minor=True)
            ax0.tick_params('x', which='major', direction='in', length=7,
                            width=2, color='white')
            ax0.tick_params('x', which='minor', direction='in', length=5,
                            width=1, color='white')
            ax1.tick_params('x', which='major', direction='in', length=7,
                            width=2, color='white')
            ax1.tick_params('x', which='minor', direction='in', length=5,
                            width=1,  color='white')

            # create yran
            ydat = spectot
            ymod = modtot
            ydattmp = np.zeros((ct), dtype=float)
            ymodtmp = np.zeros((ct), dtype=float)
            for j in range(0, len(ind)):
                ydattmp[j] = ydat[(ind[j])]
                ymodtmp[j] = ymod[(ind[j])]
            ydatmin = min(ydattmp)
            ymodmin = min(ymodtmp)
            if ydatmin <= ymodmin:
                yranmin = ydatmin
            else:
                yranmin = ymodmin
            ydatmax = max(ydattmp)
            ymodmax = max(ymodtmp)
            if ydatmax >= ymodmax:
                yranmax = ydatmax
            else:
                yranmax = ymodmax
            yran = [yranmin, yranmax]
            icol = (float(i))/(float(nx))
            if icol % 1 == 0:
                ytit = 'Fit'
            else:
                ytit = ''
            ax0.set(ylabel=ytit)
            ax0.set_xlim([xran[0], xran[1]])
            ax0.set_ylim([yran[0], yran[1]])
            # plots on ax0
            ax0.plot(wave, ydat, color='White', linewidth=1)
            if waveunit_out == 'micron':
                xtit = 'Wavelength ($\mu$m)'
            elif waveunit_out == 'Angstrom':
                xtit = 'Wavelength ($\AA$)'
            ytit = ''
            ax0.plot(wave, ymod, color='Red', linewidth=2)
            # Plot all lines visible in plot range
            for j in range(0, ncomp):
                ylaboff = 0.07
                for k, line in enumerate(linelabel):
                    lmline = lmlabel(line)
                    if f'{lmline.lmlabel}_{j}_cwv' in q3do.param.keys():
                        refwav = q3do.param[f'{lmline.lmlabel}_{j}_cwv']
                    else:
                        irefwav = np.where(q3do.linelist['name'] == line)
                        refwav = q3do.linelist['lines'][irefwav] * \
                            (1. + q3do.zstar)
                    if refwav >= xran[0] and refwav <= xran[1]:
                        if f'{lmline.lmlabel}_{j}_cwv' in \
                            q3do.param.keys():
                            flux = q3do.cmplin(line, j)
                            #import pdb; pdb.set_trace()
                            if specConv is not None:
                                conv = specConv.spect_convolver(wave, flux, refwav)
                            else:
                                conv = flux
                            ax0.plot(wave, yran[0] + conv, color=colors[j],
                                     linewidth=2, linestyle='dashed')
                        ax0.annotate(linetext[k], (0.05, 1. - ylaboff),
                                     xycoords='axes fraction',
                                     va='center', fontsize=10)
                        ylaboff += 0.07

        # if nmasked > 0:
        #   for r in range(0,nmasked):
        #        ax0.plot([masklam[r,0], masklam[r,1]], [yran[0], yran[0]],linewidth=8, color='Cyan')
            # set new value for yran
            ydat = specstars
            ymod = modstars
            ydattmp = np.zeros((len(ind)), dtype=float)
            ymodtmp = np.zeros((len(ind)), dtype=float)
            for j in range(0, len(ind)):
                ydattmp[j] = ydat[(ind[j])]
                ymodtmp[j] = ymod[(ind[j])]
            ydatmin = min(ydattmp)
            ymodmin = min(ymodtmp)
            if ydatmin <= ymodmin:
                yranmin = ydatmin
            else:
                yranmin = ymodmin
            ydatmax = max(ydattmp)
            ymodmax = max(ymodtmp)
            if ydatmax >= ymodmax:
                yranmax = ydatmax
            else:
                yranmax = ymodmax
            yran = [yranmin, yranmax]
            if icol % 1 == 0:
                ytit = 'Residual'
            else:
                ytit = ''
            ax1.set(ylabel=ytit)
            # plots on ax1
            ax1.set_xlim([xran[0], xran[1]])
            ax1.set_ylim([yran[0], yran[1]])
            ax1.plot(wave, ydat, linewidth=1)
            ax1.plot(wave, ymod, color='Red')

    # title
    if waveunit_out == 'micron':
        xtit = 'Wavelength ($\mu$m)'
    elif waveunit_out == 'Angstrom':
        xtit = 'Wavelength ($\AA$)'
    fig.suptitle(xtit, fontsize=20)

    if savefig and outfile is not None:
        if len(outfile[0])>1:
            fig.savefig(outfile[0], **argssavefig)
        else:
            fig.savefig(outfile, **argssavefig)

    plt.show()

    rcParams.update(rcParamsOrig)


def adjust_ax(ax,
              fig,
              fs=20,
              minor=False):
        '''
        CB: Function defined to adjust the sizes of xlabel, ylabel, and the 
        ticklabels (in an inelegant way for the latter).
        
        Presently just a utility function to be used in plotquest.

        Further documentation pending more testing and development.

        Parameters
        -----
        ax: matplotlib axis object
        ax object of the plot you want to adjust

        fig: matplotlib fig object
        fig object that contains the ax object

        returns
        -------
        Nothing
        '''

        fig.canvas.draw()
        xlabel = ax.get_xlabel()
        ylabel = ax.get_ylabel()
        ax.set_xlabel(xlabel, fontsize=fs)
        ax.set_ylabel(ylabel, fontsize=fs)
        ax.tick_params(labelsize=fs-3)

        # -- Trying to prune xtickslabels if increasing the fontsize made them overlap
        xticks_old = ax.get_xticks()
        if minor:
            xticks_old = ax.get_xticks(minor=True)

        xfigsize = fig.get_size_inches()[0]                # in inches
        textstrlen = len(ax.get_xticklabels()[0]._text.replace('\\mathdefault', ''))    # length of tick labels depends on nr of decimals specified
        textwidth_inch = textstrlen * (fs-3)*0.7 / 72.    # Assume width of number in text = 0.7* height. Matplotlib uses 72 Points per inch (ppi): https://stackoverflow.com/questions/47633546/relationship-between-dpi-and-figure-size

        if (len(xticks_old)+1)*textwidth_inch > 0.9* xfigsize * ax.get_position().width:
            xticks_new = np.array([])
            for i in range(len(xticks_old)):
                if i%2==1:
                    xticks_new = np.append(xticks_new, xticks_old[i])
            if not minor:
                ax.set_xticks(xticks_new, fontsize=fs-3)
            else:
                ax.set_xticks(xticks_new, fontsize=fs-3, minor=True)
        ax.set_xticklabels(ax.get_xticks(), fontsize=fs-3)
        ax.tick_params(axis='x', which='both', labelsize=fs-3)

        fig.tight_layout()


def plotdecomp(q3do,
               q3di,
               savefig=True,
               outfile=None,
               templ_mask=[],
               do_lines=False,
               show=False,
               mode='light',
               ymin=-1,
               ymax=-1,
               try_adjust_ax=True):
    '''
    Calls plotquest to plot the quasar-host galaxy decomposition. Not sure what
    the difference is between this and plotquest.

    Further documentation pending more testing and development.
    '''

    wave = q3do.wave
    specstars = q3do.cont_dat
    modstars = q3do.cont_fit
    MIRgdlambda = wave
    MIRgdflux = q3do.spec
    MIRcontinuum = modstars

    if outfile is None:
        outfile=q3do.filelab + '_decomp'

    if do_lines:
        plotquest(q3do.wave, q3do.spec, q3do.cont_fit, q3do.ct_coeff, q3di, zstar=q3do.zstar, savefig=savefig, outfile=outfile, 
            templ_mask=templ_mask, lines=q3do.linelist['lines'], linespec=q3do.line_fit, show=show, mode=mode, ymin=ymin, ymax=ymax, 
            try_adjust_ax=try_adjust_ax, row=q3do.row, col=q3do.col)
    else:
        plotquest(q3do.wave, q3do.spec, q3do.cont_fit, q3do.ct_coeff, q3di, zstar=q3do.zstar, savefig=savefig, outfile=outfile, 
            templ_mask=templ_mask, show=show, mode=mode, ymin=ymin, ymax=ymax, try_adjust_ax=try_adjust_ax, row=q3do.row, col=q3do.col)



def plotquest(MIRgdlambda,
              MIRgdflux,
              MIRcontinuum,
              ct_coeff,
              q3di,
              zstar=0.,
              savefig=True,
              outfile=None,
              templ_mask=[],
              lines=[],
              linespec=[],
              show=False,
              mode='light',
              ymin=-1,
              ymax=-1,
              try_adjust_ax=True,
              row=-1,
              col=-1):
    '''
    Plot the fit to the residual of the quasar-host galaxy decomposition, if
    refit is done with questfit. This function is presently only called
    internally by :py:func:`~q3dfit.contfit.fitqsohost`.

    Further documentation pending more testing and development.
    '''

    rcParamsOrig = rcParams.copy()

    # dark mode just for fun:
    if mode == 'dark':
        pltstyle = 'dark_background'
        dcolor = 'w'
    else:
        pltstyle = 'seaborn-v0_8-ticks'
        dcolor = 'k'

    plt.style.use(pltstyle)
    # CB: Otherwise the background becomes black and the axes ticks
    # unreadable when saving the figure
    if mode == 'light':
        rcParams['savefig.facecolor'] = 'white'

    comp_best_fit = ct_coeff['comp_best_fit']


    plot_noext = False  # Remove dust contribution and plot intrinstic components

    if 'plot_decomp' in q3di.argscontfit:
        config_file = readcf(q3di.argscontfit['config_file'])
        global_extinction = False
        for key in config_file:
            try:
                if 'global' in config_file[key][3]:
                        global_extinction = True
            except:
                continue

        fig = plt.figure(figsize=(6, 9))
        gs = fig.add_gridspec(6,1, top=0.95, bottom=0.08, left=0.2)
        ax1 = fig.add_subplot(gs[:5, :])

        ax1.plot(MIRgdlambda, MIRgdflux,color='black')
        if len(lines)==0:
            ax1.plot(MIRgdlambda, MIRcontinuum, color='r')
        else:
            ax1.plot(MIRgdlambda, MIRcontinuum + linespec, color='darkorange')

        if len(templ_mask)>0:
          MIRgdlambda_temp = MIRgdlambda[templ_mask]
        else:
          MIRgdlambda_temp = MIRgdlambda

        if len(lines)>0:
            for line_i in lines:
              ax1.axvline(line_i * (1. + zstar), color='grey', linestyle='--', alpha=0.7, zorder=0)
            #ax1.axvspan(line_i-max(q3di.siglim_gas), line_i+max(q3di.siglim_gas))
            ax1.plot(MIRgdlambda, linespec, color='r', linestyle='-', alpha=0.7, linewidth=1.5)


        colour_list = ['dodgerblue', 'mediumblue', 'salmon', 'palegreen', 'orange', 'purple', 'forestgreen', 'darkgoldenrod', 'mediumblue', 'magenta', 'plum', 'yellowgreen']

        if global_extinction:
            str_global_ext = list(comp_best_fit.keys())[-2]
            str_global_ice = list(comp_best_fit.keys())[-1]
            # global_ext is a multi-dimensional array
            if len(comp_best_fit[str_global_ext].shape) > 1:
                comp_best_fit[str_global_ext] = comp_best_fit[str_global_ext] [:,0,0]
            # global_ice is a multi-dimensional array
            if len(comp_best_fit[str_global_ice].shape) > 1:
                comp_best_fit[str_global_ice] = comp_best_fit[str_global_ice] [:,0,0]
            count = 0
            for i, el in enumerate(comp_best_fit):
                if (el != str_global_ext) and (el != str_global_ice):
                    if len(comp_best_fit[el].shape) > 1:              # component is a multi-dimensional array
                        comp_best_fit[el] = comp_best_fit[el] [:,0,0]
                    if plot_noext:
                        if count>len(colour_list)-1:
                            ax1.plot(MIRgdlambda_temp, comp_best_fit[el]/comp_best_fit[str_global_ext]/comp_best_fit[str_global_ice], label=el,linestyle='--',alpha=0.5)
                        else:
                            ax1.plot(MIRgdlambda_temp, comp_best_fit[el]/comp_best_fit[str_global_ext]/comp_best_fit[str_global_ice], color=colour_list[count], label=el,linestyle='--',alpha=0.5)
                    else:
                        if count>len(colour_list)-1:
                            ax1.plot(MIRgdlambda_temp, comp_best_fit[el], label=el,linestyle='--',alpha=0.5)
                        else:
                            ax1.plot(MIRgdlambda_temp, comp_best_fit[el], color=colour_list[count], label=el,linestyle='--',alpha=0.5)
                    count += 1

        else:
            count = 0

            for i, el in enumerate(comp_best_fit):
                if len(comp_best_fit[el].shape) > 1:
                  comp_best_fit[el] = comp_best_fit[el] [:,0,0]

                if not ('_ext' in el or '_abs' in el):
                    spec_i = comp_best_fit[el]
                    label_i = el

                    if not plot_noext:
                      if el+'_ext' in comp_best_fit.keys():
                          spec_i = spec_i*comp_best_fit[el+'_ext']
                      if el+'_abs' in comp_best_fit.keys():
                          spec_i = spec_i*comp_best_fit[el+'_abs']


                    if count>len(colour_list)-1:
                      ax1.plot(MIRgdlambda_temp, spec_i, label=label_i,linestyle='--',alpha=0.5)
                    else:
                      ax1.plot(MIRgdlambda_temp, spec_i, label=label_i, color=colour_list[i], linestyle='--',alpha=0.5)
                    count += 1

        ax1.legend(ncol=2)
        ax1.set_xscale('log')
        ax1.set_yscale('log')


        #ax1.set_ylim(1e-5,1e2)
        ax1.set_ylabel('Flux')
        if try_adjust_ax:
            adjust_ax(ax1, fig, minor=True)
        ax1.tick_params(axis='x', which='both', bottom=False, top=False, labelbottom=False) # turn off major & minor ticks on the x-axis

        ax2 = fig.add_subplot(gs[5:6, :], sharex=ax1)
        if len(lines)>=1:
            ax1.set_ylim(min(MIRcontinuum)/1e3, 3*max(MIRcontinuum + linespec))
            ax2.plot(MIRgdlambda,MIRgdflux/(MIRcontinuum + linespec),color='black')
        else:
            ax1.set_ylim(min(MIRcontinuum)/1e3, 3*max(max(MIRgdflux), max(MIRcontinuum)))
            ax2.plot(MIRgdlambda,MIRgdflux/MIRcontinuum,color='black')
        if ymin>0.:
            ax1.set_ylim(bottom=ymin)
        if ymax>0.:
            ax1.set_ylim(top=ymax)
        ax2.axhline(1, color='grey', linestyle='--', alpha=0.7, zorder=0)
        ax2.set_ylabel('Data/Model')
        ax2.set_xlabel('Wavelength [micron]')

        from matplotlib.ticker import ScalarFormatter
        ax2.xaxis.set_major_formatter(ScalarFormatter())
        ax2.xaxis.set_minor_formatter(ScalarFormatter())
        ax2.ticklabel_format(style='plain')

        if row>-1 and col>-1:
            ax1.set_title('Spaxel [{}, {}]'.format(col, row), fontsize=20)

        gs.update(wspace=0.0, hspace=0.05)
        adjust_ax(ax2, fig)

        if savefig and outfile is not None:
            if len(outfile[0])>1:
                plt.savefig(outfile[0]+'.jpg')
            else:
                plt.savefig(outfile+'.jpg')
        else:
            fig.savefig(outfile + '.jpg')

        if show:
            plt.show()

    rcParams.update(rcParamsOrig)
