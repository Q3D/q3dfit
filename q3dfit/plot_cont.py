#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jun  1 13:32:37 2021

@author: annamurphree

Plots continuum fit of optical data (fit by fitqsohost or ppxf)
or IR data (fit by questfit).
Called by Q3DA.

Init file optional parameters ('argscontplot'):
    xstyle = log or lin (linear),
    ystyle = log or lin (linear),
    waveunit_in = micron or Angstrom,
    waveunit_out = micron or Angstrom,
    fluxunit_in = flambda, lambdaflambda (= nufnu), or fnu,
    fluxunit_out = flambda, lambdaflambda (= nufnu), or fnu,
    mode = light or dark
The first options are the defaults.

"""
import math
import matplotlib.pyplot as plt
import numpy as np
import pdb
from matplotlib import rcParams


def plot_cont(instr, outfile, ct_coeff=None, initdat=None,
              compspec=None, comptitles=None, ps=None,
              title=None, fitran=None, yranminmax=None, IR=False, compcols=None):

    xstyle = 'log'
    ystyle = 'log'
    # JWST defaults:
    waveunit_in = 'micron'
    waveunit_out = 'micron'
    fluxunit_in = 'flambda'
    fluxunit_out = 'flambda'
    mode = 'light'

    if 'argscontplot' in initdat:
        if 'xstyle' in initdat['argscontplot']:
            xstyle = initdat['argscontplot']['xstyle']
        if 'ystyle' in initdat['argscontplot']:
            ystyle = initdat['argscontplot']['ystyle']
        if 'waveunit_in' in initdat['argscontplot']:
            waveunit_in = initdat['argscontplot']['waveunit_in']
        if 'waveunit_out' in initdat['argscontplot']:
            waveunit_out = initdat['argscontplot']['waveunit_out']
        if 'fluxunit_in' in initdat['argscontplot']:
            fluxunit_in = initdat['argscontplot']['fluxunit_in']
        if 'fluxunit_out' in initdat['argscontplot']:
            fluxunit_out = initdat['argscontplot']['fluxunit_out']
        if 'mode' in initdat['argscontplot']:
            mode = initdat['argscontplot']['mode']

    # dark mode just for fun:
    if mode == 'dark':
        pltstyle = 'dark_background'
        dcolor = 'w'
    else:
        pltstyle = 'seaborn-ticks'
        dcolor = 'k'

    wave = instr['wave']
    specstars = instr['cont_dat']
    modstars = instr['cont_fit']

    # for optical spectra fit by fitqsohost or ppxf:
    if not IR:

        if compspec is not None:
            if len(compspec) > 1:
                ncomp = len(compspec)
            else:
                ncomp = 1
            compcolors = ['c', 'plum', 'm']
            complabels = ['QSO', 'Host', 'Wind']
            if comptitles is not None:
                complabels = comptitles
            if compcols is not None:
                compcolors = compcols
        else:
            ncomp = 0

        if fitran is not None:
            xran = fitran
        else:
            xran = instr['fitran']

        if waveunit_in == 'Angstrom' and waveunit_out == 'micron':
            # convert angstrom to microns
            xran = list(np.divide(xran, 10**4))
            wave = list(np.divide(wave, 10**4))
            # speed of light in microns/s
            c = 2.998e+14
        elif waveunit_in == 'micron' and waveunit_out == 'Angstrom':
            # convert microns to angstroms
            xran = list(np.multiply(xran, 10**4))
            wave = list(np.multiply(wave, 10**4))
            # speed of light in angstroms/s
            c = 2.998e+18

        if fluxunit_in == 'flambda' and fluxunit_out == 'lambdaflambda':
            # multiply the flux by wavelength
            specstars = list(np.multiply(specstars, wave))
            modstars = list(np.multiply(modstars, wave))
            if ncomp > 0:
                for i in range(0, ncomp):
                    compspec[i] = list(np.multiply(compspec[i], wave))
            ytit = '$\lambda$F$_\lambda$'
        elif fluxunit_in == 'flambda' and fluxunit_out == 'fnu':
            # multiply the flux by wavelength^2/c
            specstars = \
                list(np.multiply(specstars,
                                 np.divide(np.multiply(wave, wave), c)))
            modstars = \
                list(np.multiply(modstars,
                                 np.divide(np.multiply(wave, wave), c)))
            if ncomp > 0:
                for i in range(0, ncomp):
                    compspec[i] = \
                        list(np.multiply(compspec[i],
                                         np.divide(np.multiply(wave, wave), c)))
            ytit = 'F$_\u03BD$'
        else:
            ytit = 'F$_\lambda$'

        # plot on a log scale:
        if xstyle == 'log' or ystyle == 'log':
            plt.style.use(pltstyle)
            if mode=='light':
                rcParams['savefig.facecolor'] = 'white'    # CB: Otherwise the background becomes black and the axes ticks unreadable when saving the figure
            fig = plt.figure(figsize=(20, 10))
            #fig = plt.figure()
            plt.axis('off')  # so the subplots don't share a y-axis

            fig.add_subplot(1, 1, 1)
            ydat = specstars
            ymod = modstars
            # plotting
            plt.xlim(xran[0], xran[1])

            fig.axes[0].axis('off') # so the subplots don't share a y-axis
            fig.axes[1].axis('off') # so the subplots don't share a y-axis

            gs = fig.add_gridspec(4,1)
            ax1 = fig.add_subplot(gs[:3, :])
            #ax1.legend(ncol=2)
            if xstyle == 'log':
                ax1.set_xscale('log')
            #ax1.set_xticklabels([])
            if ystyle == 'log':
                ax1.set_yscale('log')

            ax1.set_ylabel(ytit, fontsize=20)
            if title == 'QSO':
                ax1.set_ylim(10e-7)

            # actually plotting
            plt.plot(wave, ydat, dcolor, linewidth=1)
            plt.plot(wave, ymod, 'r', linewidth=3, label='Total')
            if ncomp > 0:
                for i in range(0, ncomp):
                    plt.plot(wave, compspec[i], compcolors[i], linewidth=3,
                             label=complabels[i])

            # tick formatting
            yticks_used = ax1.get_yticks()
            ylim_used = ax1.get_ylim()
            yticks_used = np.append(np.append(ylim_used[0], yticks_used), ylim_used[1])
            ax1.set_yticks(yticks_used)
            ax1.set_ylim(ylim_used)

            ax1.minorticks_on()
            ax1.tick_params(which='major', length=20, pad=10, labelsize = 20)
            ax1.tick_params(which='minor', length=7, labelsize = 17)


            l = ax1.legend(loc='upper right', fontsize=16)
            for text in l.get_texts():
                text.set_color(dcolor)
            ax2 = fig.add_subplot(gs[-1, :], sharex=ax1)
            ax2.plot(wave, np.divide(specstars, modstars), color=dcolor)
            ax2.axhline(1, color='grey', linestyle='--', alpha=0.7, zorder=0)
            ax2.set_ylabel('Data/Model', fontsize=19)

            ax2.tick_params(which='major', length=20, pad=20, labelsize = 18)
            ax2.tick_params(which='minor', length=7, labelsize = 17)

            if waveunit_out == 'micron':
                ax2.set_xlabel('Wavelength ($\mu$m)', fontsize=20)
            elif waveunit_out == 'Angstrom':
                ax2.set_xlabel('Wavelength ($\AA$)', fontsize=20)
            gs.update(wspace=0.0, hspace=0.05)
            plt.gcf().subplots_adjust(bottom=0.1)

            if title is not None:
                plt.suptitle(title, fontsize=30)

            plt.savefig(outfile + '.jpg')

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

            if waveunit_out == 'micron':
                xtit = 'Observed Wavelength ($\mu$m)'
            elif waveunit_out == 'Angstrom':
                xtit = 'Observed Wavelength ($\AA$)'

            plt.style.use(pltstyle)
            fig = plt.figure(figsize=(20, 20))
            plt.axis('off')  # so the subplots don't share a y-axis

            maximum = 0
            minimum = 0
            ''
            idict = {1: i1, 2: i2, 3: i3}
            xrans = {1: xran1, 2: xran2, 3: xran3}
            for group in range(1,4):
                if len(idict[group]) > 0:
                    fig.add_subplot(3, 1, group)

                    # finding min/max values at indices from idict
                    dat_et_mod = np.concatenate((ydat[idict[group]],
                                                 ymod[idict[group]]))
                    maximum = np.nanmax(dat_et_mod)
                    minimum = np.nanmin(dat_et_mod)
                    # set min and max in yran
                    if yranminmax is not None:
                        yran = [minimum, maximum]
                    else:
                        yran = [0, maximum]

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
                        xticks = np.arange(np.around(xrans[group][0],1)-0.025,
                                           np.around(xrans[group][1],1), 0.025)[:-1]
                        plt.xticks(xticks, fontsize=10)
                    elif waveunit_out == 'Angstrom':
                        xticks = np.arange(math.floor(xrans[group][0]/100.0)*100,
                                           (math.floor(xrans[group][1]/100)*100)+100, 100)
                        plt.xticks(xticks, fontsize=10)
                    if np.nanmin(ydat) > 1e-10:
                        # this will fail if fluxes are very low (<~1e-10)
                        plt.yticks(np.arange(yran[0], yran[1],
                                             np.around((yran[1] - yran[0])/5.,
                                                   decimals=2)), fontsize=10)
                    else:
                        plt.yticks()

                    # actually plotting
                    plt.plot(wave, ydat, dcolor, linewidth=1)

                    if ncomp > 0:
                        for i in range(0, ncomp):
                            plt.plot(wave, compspec[i], compcolors[i], linewidth=3, label=complabels[i])

                    plt.plot(wave, ymod, 'r', linewidth=4, label=title)
                    if group == 1:
                        plt.legend(loc='upper right')

            # more formatting
            plt.subplots_adjust(hspace=0.25)
            #plt.tight_layout(pad=5)
            #plt.gcf().subplots_adjust(bottom=0.1)

            if title is not None:
                plt.suptitle(title, fontsize=40)

            plt.savefig(outfile + '.jpg')

    # for IR spectra fit with questfit:
    else:

        comp_best_fit = ct_coeff['comp_best_fit']

        if xstyle == 'log' or ystyle == 'log':
            if 'plotMIR' in initdat.keys():
              if initdat['plotMIR']:
                fig = plt.figure(figsize=(50, 30))
                gs = fig.add_gridspec(4,1)
                ax1 = fig.add_subplot(gs[:3, :])

                MIRgdlambda = wave #[instr['ct_indx']]
                MIRgdflux = instr['spec'] #[instr['ct_indx']]
                MIRcontinuum = modstars #[instr['ct_indx']]

                if waveunit_in =='micron' and waveunit_out == 'Angstrom':
                    # convert microns to angstroms
                    MIRgdlambda = list(np.multiply(MIRgdlambda, 10**4))
                elif waveunit_in =='Angstrom' and waveunit_out == 'micron':
                    # convert angstroms to microns
                    MIRgdlambda = list(np.divide(MIRgdlambda, 10**4))

                if fluxunit_in == 'flambda' and fluxunit_out == 'lambdaflambda':
                    # multiply the flux by wavelength
                    MIRgdflux = list(np.multiply(MIRgdflux, MIRgdlambda))
                    MIRcontinuum = list(np.multiply(MIRcontinuum, MIRgdlambda))
                    if len(comp_best_fit.keys()) > 0:
                        for i in range(0, len(comp_best_fit.keys())):
                            comp_best_fit[list(comp_best_fit.keys())[i]] = \
                                np.multiply(comp_best_fit[list(comp_best_fit.keys())[i]],
                                            MIRgdlambda)
                    ytit = '$\lambda$F$_\lambda$'
                elif fluxunit_in == 'flambda' and fluxunit_out == 'fnu':
                    # multiply the flux by wavelength^2/c
                    MIRgdflux = list(np.multiply(MIRgdflux, MIRgdlambda))
                    MIRcontinuum = list(np.multiply(MIRgdflux, MIRgdlambda))
                    ytit = 'F$_\u03BD$'
                else:
                    ytit = 'F$_\lambda$'

                plt.style.use(pltstyle)

                ax1.plot(MIRgdlambda, MIRgdflux, label='Data',color=dcolor)
                ax1.plot(MIRgdlambda, MIRcontinuum, label='Model', color='r')

                if 'argscontfit' in initdat:
                    if 'global_ext_model' in initdat['argscontfit']:
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
                else:
                    print('argscontfit  missing in the init dict \n --> Not plotting MIR fitting results correctly.')

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

            if fitran is not None:
                xran = fitran
            else:
                xran = instr['fitran']

            if waveunit_in == 'microns' and waveunit_out == 'Angstrom':
                # convert wave list from microns to angstroms
                MIRgdlambda = list(np.multiply(MIRgdlambda, 10**4))
                xtit = 'Observed Wavelength ($\AA$)'
            elif waveunit_in == 'Angstrom' and waveunit_out == 'micron':
                # convert wave list from angstroms to microns
                MIRgdlambda = list(np.divide(MIRgdlambda, 10**4))
                xtit = 'Observed Wavelength ($\mu$m)'

            if fluxunit_in == 'flambda' and fluxunit_out == 'lambdaflambda':
                # multiply the flux by wavelength
                MIRgdflux = list(np.multiply(MIRgdflux, MIRgdlambda))
                MIRcontinuum = list(np.multiply(MIRcontinuum, MIRgdlambda))
                if len(comp_best_fit.keys()) > 0:
                    for i in range(0, len(comp_best_fit.keys())):
                        comp_best_fit[list(comp_best_fit.keys())[i]] = \
                            list(np.multiply(comp_best_fit[list(comp_best_fit.keys())[i]],
                                             MIRgdlambda))
                ytit = '$\lambda$F$_\lambda$'
            elif fluxunit_in == 'flambda' and fluxunit_out == 'fnu':
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
            fig = plt.figure(figsize=(10, 10))
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
                        bigboy = np.nanmax(ydat[i], ymod[i])
                        if bigboy > maximum:
                            maximum = bigboy
                    # finding min
                    for i in idict[group]:
                        smallboy = np.nanmin(ydat[i], ymod[i])
                        if smallboy < minimum:
                            minimum = smallboy
                    # set min and max in yran
                    if yranminmax is not None:
                        yran = [minimum, maximum]
                    else:
                        yran = [0, maximum]

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

                    if 'argscontfit' in initdat:
                        if 'global_ext_model' in initdat['argscontfit']:
                           for i in np.arange(0,len(comp_best_fit.keys())-2,1):
                              plt.plot(MIRgdlambda,
                                       np.multiply(comp_best_fit[list(comp_best_fit.keys())[i]],
                                                   np.multiply(comp_best_fit[list(comp_best_fit.keys())[-2]],
                                                               comp_best_fit[list(comp_best_fit.keys())[-1]])),
                                       label=list(comp_best_fit.keys())[i],linestyle='--',alpha=0.5)

                        else:
                           for i in np.arange(0,len(comp_best_fit.keys()),3):
                              plt.plot(MIRgdlambda,
                                       np.multiply(comp_best_fit[list(comp_best_fit.keys())[i]],
                                                   np.multiply(comp_best_fit[list(comp_best_fit.keys())[i+1]],
                                                               comp_best_fit[list(comp_best_fit.keys())[i+2]])),
                                       label=list(comp_best_fit.keys())[i],linestyle='--',alpha=0.5)

                    if group == 1:
                        ax.legend(loc='upper right',bbox_to_anchor=(1.22, 1),prop={'size': 10})

        # more formatting
        plt.subplots_adjust(hspace=0.25)
        plt.tight_layout(pad=15)
        plt.gcf().subplots_adjust(bottom=0.1)
        plt.gcf().subplots_adjust(right=0.85)

        if title is not None:
            plt.suptitle(title, fontsize=30)

        plt.savefig(outfile + '.jpg')
