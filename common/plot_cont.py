#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jun  1 13:32:37 2021

@author: annamurphree

Plots continuum fit of optical data (fit by fitqsohost or ppxf) or IR data (fit by questfit).
Called by Q3DA.

Init file optional parameters ('argscontplot'):
    xstyle = log or lin (linear),
    ystyle = log or lin (linear), 
    xunit = micron or ang (angstrom),
    yunit = flambda, lambdaflambda (= nufnu), or fnu
    mode = light or dark
The first options are the defaults.

"""
import matplotlib.pyplot as plt
#from matplotlib.lines import Line2D
import numpy as np
import math

def plot_cont(instr, outfile, MIRgdlambda=None, MIRgdflux=None, 
              MIRcontinuum=None, ct_coeff=None, initdat=None,
              compspec=None, comptitles=None, ps=None,
              title=None, fitran=None, yranminmax=None, IR=None):

    if 'argscontplot' in initdat:
        xstyle = initdat['argscontplot']['xstyle']
        ystyle = initdat['argscontplot']['ystyle']
        xunit = initdat['argscontplot']['xunit']
        yunit = initdat['argscontplot']['yunit']
        mode = initdat['argscontplot']['mode']
    else:
        xstyle = 'log'
        ystyle = 'log'
        xunit = 'micron'
        yunit = 'flambda'
        mode = 'light'
    
    if mode == 'dark':
        pltstyle = 'dark_background'
        dcolor = 'w'
    else:
        pltstyle = 'seaborn-ticks'
        dcolor = 'k'
    
    # for optical spectra fit by fitqsohost or ppxf:
    if IR == None:
        
        if compspec is not None:
            if len(compspec) > 1:
                ncomp = len(compspec)
            else:
                ncomp = 1
            compcolors = ['c', 'plum', 'm']
            complabels = ['QSO', 'Host', 'Wind']
        else:
            ncomp = 0
    
        wave = instr['wave']
        specstars = instr['cont_dat']
        modstars = instr['cont_fit']
        
        if fitran is not None:
            xran = fitran
        else:
            xran = instr['fitran']
        
        # speed of light in angstroms/s
        c = 2.998e+18
        if xunit == 'micron':
            # convert angstrom to microns
            xran = list(np.divide(xran, 10**4))
            wave = list(np.divide(wave, 10**4))
            # speed of light in microns/s
            c = 2.998e+14
        
        if yunit == 'lambdaflambda':
            # multiply the flux by wavelength
            specstars = list(np.multiply(specstars, wave))
            modstars = list(np.multiply(modstars, wave))
            if ncomp > 0:
                for i in range(0, ncomp):
                    compspec[i] = list(np.multiply(compspec[i], wave))
            ytit = '$\lambda$F$_\lambda$'
        elif yunit == 'fnu':
            # multiply the flux by wavelength
            specstars = list(np.multiply(specstars, np.divide(np.multiply(wave,wave),c)))
            modstars = list(np.multiply(modstars, np.divide(np.multiply(wave,wave),c)))
            if ncomp > 0:
                for i in range(0, ncomp):
                    compspec[i] = list(np.multiply(compspec[i], np.divide(np.multiply(wave,wave),c)))
            ytit = 'F$_\u03BD$'
        else: 
            ytit = 'F$_\lambda$'
            
        # plot on a log scale:
        if xstyle == 'log' or ystyle == 'log':
            plt.style.use(pltstyle)
            fig = plt.figure(figsize=(30, 10))
            plt.axis('off')  # so the subplots don't share a y-axis
            
            fig.add_subplot(1, 1, 1)
            ydat = specstars
            ymod = modstars
            # plotting
            plt.xlim(xran[0], xran[1])
            # tick formatting
            plt.minorticks_on()
            plt.tick_params(which='major', length=20, pad=30)
            plt.tick_params(which='minor', length=10)
    
            gs = fig.add_gridspec(4,1)
            ax1 = fig.add_subplot(gs[:3, :])
            ax1.legend(ncol=2)
            if xstyle == 'log':
                ax1.set_xscale('log')
            #ax1.set_xticklabels([])
            if ystyle == 'log':
                ax1.set_yscale('log')
            ax1.set_ylabel(ytit, fontsize=12)
            if title == 'QSO':
                ax1.set_ylim(10e-7)
            
            # actually plotting
            plt.plot(wave, ydat, dcolor, linewidth=1)
            plt.plot(wave, ymod, 'r', linewidth=3, label='Total')
            if ncomp > 0:
                for i in range(0, ncomp):
                    plt.plot(wave, compspec[i], compcolors[i], linewidth=3, label=complabels[i])
            
            l = plt.legend(loc='upper right')
            for text in l.get_texts():
                text.set_color(dcolor)
            ax2 = fig.add_subplot(gs[-1, :], sharex=ax1)
            ax2.plot(wave, np.divide(specstars,modstars), color=dcolor)
            ax2.axhline(1, color='grey', linestyle='--', alpha=0.7, zorder=0)
            ax2.set_ylabel('Data/Model', fontsize=12)
            if xunit == 'micron':
                ax2.set_xlabel('Wavelength ($\mu$m)', fontsize=12)
            else:
                ax2.set_xlabel('Wavelength ($\AA$)', fontsize=12)
            gs.update(wspace=0.0, hspace=0.05)
            plt.gcf().subplots_adjust(bottom=0.1)
        
            if title is not None:
                plt.suptitle(title, fontsize=30)
            
            plt.savefig(outfile + '.jpg')
            
        #else:
        elif xstyle == 'lin' or ystyle == 'lin':
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
            
            ydat = specstars
            ymod = modstars
            
            for i in range(0, len(wave)):
                if wave[i] > xran1[0] and wave[i] < xran1[1]:
                    i1.append(i)
                if wave[i] > xran2[0] and wave[i] < xran2[1]:
                    i2.append(i)
                if wave[i] > xran3[0] and wave[i] < xran3[1]:
                    i3.append(i)
            ct1 = len(i1)
            ct2 = len(i2)
            ct3 = len(i3)
            
            maxthresh = 0.2
            ntop = 20
            nbottom = 20
            if len(wave) < 100:
                ntop = 10
                nbottom = 10
            ++ntop
            --nbottom
            
            if xunit == 'micron':
                xtit = 'Observed Wavelength ($\mu$m)'
            else:
                xtit = 'Observed Wavelength ($\AA$)'
            
            plt.style.use(pltstyle)
            fig = plt.figure(figsize=(20, 20))
            plt.axis('off')  # so the subplots don't share a y-axis
            
            maximum = 0
            minimum = 0
            ''
            cts = {1:ct1, 2:ct2, 3:ct3}
            idict = {1:i1, 2:i2, 3:i3}
            xrans = {1:xran1, 2:xran2, 3:xran3}
            count = 1
            for ct in cts.values():
                if ct > 0:
                    fig.add_subplot(3, 1, count)
        
                    # finding max value between ydat and ymod at indices from i1
                    #for i in idict[cts[ct]]:
                    for i in idict[count]:
                        bigboy = max(ydat[i], ymod[i])
                        if bigboy > maximum:
                            maximum = bigboy
                    # finding min
                    for i in idict[count]:
                        smallboy = min(ydat[i], ymod[i])
                        if smallboy < minimum:
                            minimum = smallboy
                    # set min and max in yran
                    if yranminmax is not None:
                        yran = [minimum, maximum]
                    else:
                        yran = [0, maximum]
            
                    # finding yran[1] aka max
                    ydi = np.zeros(len(idict[count]))
                    ydi = np.array(ydat)[idict[count]]
            
                    ymodi = np.zeros(len(idict[count]))
                    ymodi = np.array(ymod)[idict[count]]
                    y = np.array(ydi - ymodi)
                    ny = len(y)
            
                    iysort = np.argsort(y)
                    ysort = np.array(y)[iysort]
            
                    ymodisort = ymodi[iysort]
                    if ysort[ny - ntop] < ysort[ny - 1] * maxthresh:
                        yran[1] = max(ysort[0:ny - ntop] + ymodisort[0:ny - ntop])
            
                    # plotting
                    plt.xlim(xrans[count][0], xrans[count][1])
                    plt.ylim(yran[0], yran[1])
                    plt.ylabel(ytit, fontsize=15)
                    if count == 3:
                        plt.xlabel(xtit, fontsize=15, labelpad=10)
                    if ystyle == 'log':
                        plt.yscale('log')
            
                    # tick formatting
                    plt.minorticks_on()
                    plt.tick_params(which='major', length=10, pad=5)
                    plt.tick_params(which='minor', length=5)
                    if xunit == 'micron':
                        xticks = np.arange(np.around(xrans[count][0],1)-0.025, np.around(xrans[count][1],1), 0.025)[:-1]
                        #print(xran1[0], xran1[1], xticks)
                        plt.xticks(xticks, fontsize=10)
                        #plt.xticks(np.arange(xran1[0], xran1[1], .02), fontsize=10)
                    else:
                        xticks = np.arange(math.floor(xrans[count][0]/100.0)*100, (math.floor(xrans[count][1]/100)*100)+100, 100)
                        #print(xran1[0], xran1[1], xticks)
                        plt.xticks(xticks, fontsize=10)
                        #plt.xticks(np.arange(xran1[0], xran1[1], 200), fontsize=10)
                    if yunit != 'fnu':
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
                    if count == 1:
                        plt.legend(loc='upper right')
                    #print('plot', count)
                    count+=1
            '''
            if ct1 > 0:
                fig.add_subplot(3, 1, 1)
        
                # finding max value between ydat and ymod at indices from i1
                for i in i1:
                    bigboy = max(ydat[i], ymod[i])
                    if bigboy > maximum:
                        maximum = bigboy
                # finding min
                for i in i1:
                    smallboy = min(ydat[i], ymod[i])
                    if smallboy < minimum:
                        minimum = smallboy
                # set min and max in yran
                if yranminmax is not None:
                    yran = [minimum, maximum]
                else:
                    yran = [0, maximum]
        
                # finding yran[1] aka max
                ydi = np.zeros(len(i1))
                ydi = np.array(ydat)[i1]
        
                ymodi = np.zeros(len(i1))
                ymodi = np.array(ymod)[i1]
                y = np.array(ydi - ymodi)
                ny = len(y)
        
                iysort = np.argsort(y)
                ysort = np.array(y)[iysort]
        
                ymodisort = ymodi[iysort]
                if ysort[ny - ntop] < ysort[ny - 1] * maxthresh:
                    yran[1] = max(ysort[0:ny - ntop] + ymodisort[0:ny - ntop])
        
                # plotting
                plt.xlim(xran1[0], xran1[1])
                plt.ylim(yran[0], yran[1])
                plt.ylabel(ytit, fontsize=15)
                #if ystyle == 'log':
                #    plt.yscale('log')
        
                # tick formatting
                plt.minorticks_on()
                plt.tick_params(which='major', length=10, pad=5)
                plt.tick_params(which='minor', length=5)
                if xunit == 'micron':
                    xticks = np.arange(np.around(xran1[0],1)-0.025, np.around(xran1[1],1), 0.025)[:-1]
                    #print(xran1[0], xran1[1], xticks)
                    plt.xticks(xticks, fontsize=10)
                    #plt.xticks(np.arange(xran1[0], xran1[1], .02), fontsize=10)
                else:
                    xticks = np.arange(math.floor(xran1[0]/100.0)*100, (math.floor(xran1[1]/100)*100)+100, 100)
                    #print(xran1[0], xran1[1], xticks)
                    plt.xticks(xticks, fontsize=10)
                    #plt.xticks(np.arange(xran1[0], xran1[1], 200), fontsize=10)
                if yunit != 'fnu':
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
        
                plt.plot(wave, ymod, 'r', linewidth=4, label='Total')
                plt.legend(loc='upper right')
        
            # like previous section
            if ct2 > 0:
                fig.add_subplot(3, 1, 2)
                plt.minorticks_on()
                ymod = modstars
        
                # finding max
                for i in i2:
                    bigboy = max(ydat[i], ymod[i])
                    if bigboy > maximum:
                        maximum = bigboy
                # finding min
                for i in i2:
                    smallboy = min(ydat[i], ymod[i])
                    if smallboy < minimum:
                        minimum = smallboy
        
                if yranminmax is not None:
                    yran = [minimum, maximum]
                else:
                    yran = [0, maximum]
        
                # finding yran[1] aka max
                ydi = np.zeros(len(i2))
                ydi = np.array(ydat)[i2]
        
                ymodi = np.zeros(len(i2))
                ymodi = np.array(ymod)[i2]
                y = np.array(ydi - ymodi)
                ny = len(y)
        
                iysort = np.argsort(y)
                ysort = np.array(y)[iysort]
        
                ymodisort = ymodi[iysort]
                if ysort[ny - ntop] < ysort[ny - 1] * maxthresh:
                    yran[1] = max(ysort[0:ny - ntop] + ymodisort[0:ny - ntop])
        
                # plotting
                plt.xlim(xran2[0], xran2[1])
                plt.ylim(yran[0], yran[1])
                plt.ylabel(ytit, fontsize=15)
                if xunit == 'micron':
                    xticks = np.arange(np.around(xran2[0],1)-0.05, np.around(xran2[1],1)+0.025, 0.025)
                    #print(xran2[0], xran2[1], xticks)
                    plt.xticks(xticks, fontsize=10)
                    #plt.xticks(np.arange(xran2[0], xran2[1], .02), fontsize=10)
                else:
                    xticks = np.arange(math.floor(xran2[0]/100.0)*100, (math.floor(xran2[1]/100.0)*100)+100, 100)
                    #print(xran2[0], xran2[1], xticks)
                    plt.xticks(xticks, fontsize=10)
                    #plt.xticks(np.arange(xran2[0], xran2[1], 200), fontsize=10)
                plt.yticks(fontsize=10)
                plt.tick_params(which='major', length=10, pad=5)
                plt.tick_params(which='minor', length=5)
                if ystyle == 'log':
                    plt.yscale('log')
        
                # yay more plotting
                plt.plot(wave, ydat, dcolor, linewidth=1)
        
                if ncomp > 0:
                    for i in range(0, ncomp):
                        plt.plot(wave, compspec[i], compcolors[i], linewidth=3, label=complabels[i])
        
                plt.plot(wave, ymod, 'r', linewidth=4)
        
            # and again
            if ct3 > 0:
                fig.add_subplot(3, 1, 3)
                plt.subplots_adjust(hspace=2)
                plt.minorticks_on()
        
                # finding max
                for i in i3:
                    bigboy = max(ydat[i], ymod[i])
                    if bigboy > maximum:
                        maximum = bigboy
                # finding min
                for i in i3:
                    smallboy = min(ydat[i], ymod[i])
                    if smallboy < minimum:
                        minimum = smallboy
        
                # finding yran[1] aka max
                if yranminmax is not None:
                    yran = [minimum, maximum]
                else:
                    yran = [0, maximum]
        
                ydi = np.zeros(len(i3))
                ydi = np.array(ydat)[i3]
        
                ymodi = np.zeros(len(i3))
                ymodi = np.array(ymod)[i3]
                y = np.array(ydi - ymodi)
                ny = len(y)
        
                iysort = np.argsort(y)
                ysort = np.array(y)[iysort]
        
                ymodisort = ymodi[iysort]
                if ysort[ny - ntop] < ysort[ny - 1] * maxthresh:
                    yran[1] = max(ysort[0:ny - ntop] + ymodisort[0:ny - ntop])
        
                # plotting
                plt.xlim(xran3[0], xran3[1])
                plt.ylim(yran[0], yran[1])
                plt.xlabel(xtit, fontsize=15, labelpad=10)
                plt.ylabel(ytit, fontsize=15)
                if xunit == 'micron':
                    xticks = np.arange(np.around(xran3[0],1), np.around(xran3[1],1)+0.05, 0.025)[:-1]
                    #print(xran3[0], xran3[1], xticks)
                    plt.xticks(xticks, fontsize=10)
                    #plt.xticks(np.arange(xran3[0], xran3[1], .02), fontsize=10)
                else:
                    xticks = np.arange(math.floor(xran3[0]/100.0)*100, (math.floor(xran3[1]/100.0)*100)+100, 100)
                    #print(xran3[0], xran3[1], xticks)
                    plt.xticks(xticks, fontsize=10)
                    #plt.xticks(np.arange(xran3[0], xran3[1], 200), fontsize=10)
                plt.yticks(fontsize=10)
                plt.tick_params(which='major', length=10, pad=5)
                plt.tick_params(which='minor', length=5)
                if ystyle == 'log':
                    plt.yscale('log')
        
                # plotting more
                plt.plot(wave, ydat, dcolor, linewidth=1)
                if ncomp > 0:
                    for i in range(0, ncomp):
                        plt.plot(wave, compspec[i], compcolors[i], linewidth=3, label=complabels[i])
        
                plt.plot(wave, ymod, 'r', linewidth=4)
                '''
            # more formatting
            plt.subplots_adjust(hspace=0.25)
            #plt.tight_layout(pad=5)
            #plt.gcf().subplots_adjust(bottom=0.1)
        
            if title is not None:
                plt.suptitle(title, fontsize=40)
        
            plt.savefig(outfile + '.jpg')
    
    # for IR spectra fit with questfit:
    elif IR == 1:
        comp_best_fit = ct_coeff['comp_best_fit']
        
        if xstyle == 'log' or ystyle == 'log':
            if 'plotMIR' in initdat.keys(): 
              if initdat['plotMIR']:
                #fig.add_subplot(3, 1, 1)
                fig = plt.figure(figsize=(50, 30))
                gs = fig.add_gridspec(4,1)
                ax1 = fig.add_subplot(gs[:3, :])
                
                if xunit == 'ang':
                    # convert angstrom to microns
                    MIRgdlambda = list(np.multiply(MIRgdlambda, 10**4))
                
                if yunit == 'lambdaflambda':
                    # multiply the flux by wavelength
                    MIRgdflux = list(np.multiply(MIRgdflux, MIRgdlambda))
                    MIRcontinuum = list(np.multiply(MIRcontinuum, MIRgdlambda))
                    if len(comp_best_fit.keys()) > 0:
                        for i in range(0, len(comp_best_fit.keys())):
                            comp_best_fit[list(comp_best_fit.keys())[i]] = np.multiply(comp_best_fit[list(comp_best_fit.keys())[i]], MIRgdlambda)
                    ytit = '$\lambda$F$_\lambda$'
                elif yunit == 'fnu':
                    # multiply the flux by wavelength
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
                #ax1.set_xticklabels([])
                if ystyle == 'log':
                    ax1.set_yscale('log')
                ax1.set_ylim(1e-4)
                ax1.set_ylabel(ytit, fontsize=12)
        
                ax2 = fig.add_subplot(gs[-1, :], sharex=ax1)
                ax2.plot(MIRgdlambda,np.divide(MIRgdflux,MIRcontinuum),color=dcolor)
                ax2.axhline(1, color='grey', linestyle='--', alpha=0.7, zorder=0)
                ax2.set_ylabel('Data/Model', fontsize=12)
                if xunit == 'Angstrom':
                    ax2.set_xlabel('Wavelength ($\AA$)', fontsize=12)
                else:
                    ax2.set_xlabel('Wavelength ($\mu$m)', fontsize=12)
                gs.update(wspace=0.0, hspace=0.05)
                plt.suptitle('Total', fontsize=30)
        
        elif xstyle == 'lin' or ystyle == 'lin':
            
            if fitran is not None:
                xran = fitran
            else:
                xran = instr['fitran']
            
            if xunit == 'Angstrom':
                # convert wave list from microns to angstroms
                MIRgdlambda = list(np.multiply(MIRgdlambda, 10**4))
                xtit = 'Observed Wavelength ($\AA$)'
            elif xunit == 'micron':
                # convert xrange from angstroms to microns
                xran = list(np.divide(xran, 10**4))
                xtit = 'Observed Wavelength ($\mu$m)'
            
            if yunit == 'lambdaflambda':
                # multiply the flux by wavelength
                MIRgdflux = list(np.multiply(MIRgdflux, MIRgdlambda))
                MIRcontinuum = list(np.multiply(MIRcontinuum, MIRgdlambda))
                if len(comp_best_fit.keys()) > 0:
                    for i in range(0, len(comp_best_fit.keys())):
                        comp_best_fit[list(comp_best_fit.keys())[i]] = list(np.multiply(comp_best_fit[list(comp_best_fit.keys())[i]], MIRgdlambda))
                ytit = '$\lambda$F$_\lambda$'
            elif yunit == 'fnu':
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
            ct1 = len(i1)
            ct2 = len(i2)
            ct3 = len(i3)
            
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
            
            cts = {1:ct1, 2:ct2, 3:ct3}
            idict = {1:i1, 2:i2, 3:i3}
            xrans = {1:xran1, 2:xran2, 3:xran3}
            count = 1
            for ct in cts.values():
                if ct > 0:
                    fig.add_subplot(3, 1, count)
                    ax = plt.subplot(3, 1, count)
                    # shrink current axis by 10% to fit legend on side
                    box = ax.get_position()
                    ax.set_position([box.x0, box.y0, box.width * 0.8, box.height])
                    # finding max value between ydat and ymod at indices from i1
                    #for i in idict[cts[ct]]:
                    for i in idict[count]:
                        bigboy = max(ydat[i], ymod[i])
                        if bigboy > maximum:
                            maximum = bigboy
                    # finding min
                    for i in idict[count]:
                        smallboy = min(ydat[i], ymod[i])
                        if smallboy < minimum:
                            minimum = smallboy
                    # set min and max in yran
                    if yranminmax is not None:
                        yran = [minimum, maximum]
                    else:
                        yran = [0, maximum]
            
                    # finding yran[1] aka max
                    ydi = np.zeros(len(idict[count]))
                    ydi = np.array(ydat)[idict[count]]
            
                    ymodi = np.zeros(len(idict[count]))
                    ymodi = np.array(ymod)[idict[count]]
                    y = np.array(ydi - ymodi)
                    ny = len(y)
            
                    iysort = np.argsort(y)
                    ysort = np.array(y)[iysort]
            
                    ymodisort = ymodi[iysort]
                    if ysort[ny - ntop] < ysort[ny - 1] * maxthresh:
                        yran[1] = max(ysort[0:ny - ntop] + ymodisort[0:ny - ntop])
            
                    # plotting
                    plt.xlim(xrans[count][0], xrans[count][1])
                    plt.ylim(yran[0], yran[1])
                    plt.ylabel(ytit, fontsize=15)
                    if count == 3:
                        plt.xlabel(xtit, fontsize=15, labelpad=10)
                    if ystyle == 'log':
                        plt.yscale('log')
            
                    # tick formatting
                    plt.minorticks_on()
                    plt.tick_params(which='major', length=10, pad=5)
                    plt.tick_params(which='minor', length=5)
                    if xunit == 'micron':
                        xticks = np.arange(np.around(xrans[count][0]), np.around(xrans[count][1]), 1)
                        #print(xran1[0], xran1[1], xticks)
                        plt.xticks(xticks, fontsize=10)
                        #plt.xticks(np.arange(xran3[0], xran3[1], .02), fontsize=10)
                    else:
                        xticks = np.arange(math.floor(xrans[count][0]/1000.0)*1000, (math.floor(xrans[count][1]/1000.0)*1000)+1000, 10000)
                        #print(xran1[0], xran1[1], xticks)
                        plt.xticks(xticks, fontsize=10)
                        #plt.xticks(np.arange(xran3[0], xran3[1], 200), fontsize=10)\
                    if yunit != 'fnu':
                        # this will fail if fluxes are very low (<~1e-10)
                        plt.yticks(np.arange(yran[0], yran[1],
                                             np.around((yran[1] - yran[0])/5.,
                                                       decimals=2)), fontsize=10)
                    else: 
                        plt.yticks()
            
                    # actually plotting
                    plt.plot(MIRgdlambda, MIRgdflux, label='Data', color=dcolor)
                    plt.plot(MIRgdlambda, MIRcontinuum, label='Model', color='red')
                    
                    if 'argscontfit' in initdat:
                        if 'global_ext_model' in initdat['argscontfit']:
                           for i in np.arange(0,len(comp_best_fit.keys())-2,1):
                              plt.plot(MIRgdlambda,np.multiply(comp_best_fit[list(comp_best_fit.keys())[i]],np.multiply(comp_best_fit[list(comp_best_fit.keys())[-2]],comp_best_fit[list(comp_best_fit.keys())[-1]])),label=list(comp_best_fit.keys())[i],linestyle='--',alpha=0.5)
            
                        else:
                           for i in np.arange(0,len(comp_best_fit.keys()),3):
                              plt.plot(MIRgdlambda,np.multiply(comp_best_fit[list(comp_best_fit.keys())[i]],np.multiply(comp_best_fit[list(comp_best_fit.keys())[i+1]],comp_best_fit[list(comp_best_fit.keys())[i+2]])),label=list(comp_best_fit.keys())[i],linestyle='--',alpha=0.5)
                    
                    if count == 1:
                        ax.legend(loc='upper right',bbox_to_anchor=(1.22, 1),prop={'size': 10})
                    
                    #print('plot', count)
                    count+=1
            '''
            if ct1 > 0:
                fig.add_subplot(3, 1, 1)
                ''
                ax = plt.subplot(311)
                # shrink current axis by 10% to fit legend on side
                box = ax.get_position()
                ax.set_position([box.x0, box.y0, box.width * 0.9, box.height])
                ''
                # finding max value between ydat and ymod at indices from i1
                for i in i1:
                    bigboy = max(ydat[i], ymod[i])
                    if bigboy > maximum:
                        maximum = bigboy
                # finding min
                for i in i1:
                    smallboy = min(ydat[i], ymod[i])
                    if smallboy < minimum:
                        minimum = smallboy
                    
                # set min and max in yran
                if yranminmax is not None:
                    yran = [minimum, maximum]
                else:
                    yran = [0, maximum]
                
                # finding yran[1] aka max
                ydi = np.zeros(len(i1))
                ydi = np.array(ydat)[i1]
        
                ymodi = np.zeros(len(i1))
                ymodi = np.array(ymod)[i1]
                y = np.array(ydi - ymodi)
                ny = len(y)
        
                iysort = np.argsort(y)
                ysort = np.array(y)[iysort]
        
                ymodisort = ymodi[iysort]
                if ysort[ny - ntop] < ysort[ny - 1] * maxthresh:
                    yran[1] = max(ysort[0:ny - ntop] + ymodisort[0:ny - ntop])
                    
                # plotting
                plt.xlim(xran1[0], xran1[1])
                plt.ylim(yran[0], yran[1])
                plt.ylabel(ytit, fontsize=15)
                if ystyle == 'log':
                    plt.yscale('log')
        
                # tick formatting
                plt.minorticks_on()
                plt.tick_params(which='major', length=10, pad=5)
                plt.tick_params(which='minor', length=5)
                if xunit == 'micron':
                    xticks = np.arange(np.around(xran1[0]), np.around(xran1[1]), 1)
                    #print(xran1[0], xran1[1], xticks)
                    plt.xticks(xticks, fontsize=10)
                    #plt.xticks(np.arange(xran3[0], xran3[1], .02), fontsize=10)
                else:
                    xticks = np.arange(math.floor(xran1[0]/1000.0)*1000, (math.floor(xran1[1]/1000.0)*1000)+1000, 10000)
                    #print(xran1[0], xran1[1], xticks)
                    plt.xticks(xticks, fontsize=10)
                    #plt.xticks(np.arange(xran3[0], xran3[1], 200), fontsize=10)
                plt.yticks(fontsize=10)
                
                # actually plotting
                plt.plot(MIRgdlambda, MIRgdflux, label='Data', color=dcolor)
                plt.plot(MIRgdlambda, MIRcontinuum, label='Model', color='red')
                
                if 'argscontfit' in initdat:
                    if 'global_ext_model' in initdat['argscontfit']:
                       for i in np.arange(0,len(comp_best_fit.keys())-2,1):
                          plt.plot(MIRgdlambda,np.multiply(comp_best_fit[list(comp_best_fit.keys())[i]],np.multiply(comp_best_fit[list(comp_best_fit.keys())[-2]],comp_best_fit[list(comp_best_fit.keys())[-1]])),label=list(comp_best_fit.keys())[i],linestyle='--',alpha=0.5)
        
                    else:
                       for i in np.arange(0,len(comp_best_fit.keys()),3):
                          plt.plot(MIRgdlambda,np.multiply(comp_best_fit[list(comp_best_fit.keys())[i]],np.multiply(comp_best_fit[list(comp_best_fit.keys())[i+1]],comp_best_fit[list(comp_best_fit.keys())[i+2]])),label=list(comp_best_fit.keys())[i],linestyle='--',alpha=0.5)
                
                ax.legend(loc='upper right',bbox_to_anchor=(1.19, 1),prop={'size': 10})
            
            if ct2 > 0:
                fig.add_subplot(3, 1, 2)
                ax = plt.subplot(312)
                # shrink current axis by 10% to fit legend on side
                box = ax.get_position()
                ax.set_position([box.x0, box.y0, box.width * 0.9, box.height])
                
                # finding max value between ydat and ymod at indices from i1
                for i in i2:
                    bigboy = max(ydat[i], ymod[i])
                    if bigboy > maximum:
                        maximum = bigboy
                # finding min
                for i in i2:
                    smallboy = min(ydat[i], ymod[i])
                    if smallboy < minimum:
                        minimum = smallboy
                # set min and max in yran
                if yranminmax is not None:
                    yran = [minimum, maximum]
                else:
                    yran = [0, maximum]
                
                # finding yran[1] aka max
                ydi = np.zeros(len(i2))
                ydi = np.array(ydat)[i2]
        
                ymodi = np.zeros(len(i2))
                ymodi = np.array(ymod)[i2]
                y = np.array(ydi - ymodi)
                ny = len(y)
        
                iysort = np.argsort(y)
                ysort = np.array(y)[iysort]
        
                ymodisort = ymodi[iysort]
                if ysort[ny - ntop] < ysort[ny - 1] * maxthresh:
                    yran[1] = max(ysort[0:ny - ntop] + ymodisort[0:ny - ntop])
                
                # plotting
                plt.xlim(xran2[0], xran2[1])
                plt.ylim(yran[0], yran[1])
                plt.ylabel(ytit, fontsize=15)
                if ystyle == 'log':
                    plt.yscale('log')
        
                # tick formatting
                plt.minorticks_on()
                plt.tick_params(which='major', length=10, pad=5)
                plt.tick_params(which='minor', length=5)
                if xunit == 'micron':
                    xticks = np.arange(np.around(xran2[0]), np.around(xran2[1]), 1)
                    #print(xran2[0], xran2[1], xticks)
                    plt.xticks(xticks, fontsize=10)
                    #plt.xticks(np.arange(xran3[0], xran3[1], .02), fontsize=10)
                else:
                    xticks = np.arange(math.floor(xran2[0]/1000.0)*1000, (math.floor(xran2[1]/1000.0)*1000)+1000, 10000)
                    #print(xran2[0], xran2[1], xticks)
                    plt.xticks(xticks, fontsize=10)
                    #plt.xticks(np.arange(xran3[0], xran3[1], 200), fontsize=10)
                plt.yticks(fontsize=10)
                
                # actually plotting
                plt.plot(MIRgdlambda, MIRgdflux, label='Data', color=dcolor)
                plt.plot(MIRgdlambda, MIRcontinuum, label='Model', color='r')
                
                if 'argscontfit' in initdat:
                    if 'global_ext_model' in initdat['argscontfit']:
                       for i in np.arange(0,len(comp_best_fit.keys())-2,1):
                          plt.plot(MIRgdlambda,np.multiply(comp_best_fit[list(comp_best_fit.keys())[i]],np.multiply(comp_best_fit[list(comp_best_fit.keys())[-2]],comp_best_fit[list(comp_best_fit.keys())[-1]])),label=list(comp_best_fit.keys())[i],linestyle='--',alpha=0.5)
        
                    else:
                       for i in np.arange(0,len(comp_best_fit.keys()),3):
                          plt.plot(MIRgdlambda,np.multiply(comp_best_fit[list(comp_best_fit.keys())[i]],np.multiply(comp_best_fit[list(comp_best_fit.keys())[i+1]],comp_best_fit[list(comp_best_fit.keys())[i+2]])),label=list(comp_best_fit.keys())[i],linestyle='--',alpha=0.5)
            
            if ct3 > 0:
                fig.add_subplot(3, 1, 3)
                ax = plt.subplot(313)
                # shrink current axis by 10% to fit legend on side
                box = ax.get_position()
                ax.set_position([box.x0, box.y0, box.width * 0.9, box.height])
                
                # finding max value 
                for i in i3:
                    bigboy = max(ydat[i], ymod[i])
                    if bigboy > maximum:
                        maximum = bigboy
                # finding min
                for i in i3:
                    smallboy = min(ydat[i], ymod[i])
                    if smallboy < minimum:
                        minimum = smallboy
                # set min and max in yran
                if yranminmax is not None:
                    yran = [minimum, maximum]
                else:
                    yran = [0, maximum]
                
                # finding yran[1] aka max
                ydi = np.zeros(len(i3))
                ydi = np.array(ydat)[i3]
        
                ymodi = np.zeros(len(i3))
                ymodi = np.array(ymod)[i3]
                y = np.array(ydi - ymodi)
                ny = len(y)
        
                iysort = np.argsort(y)
                ysort = np.array(y)[iysort]
        
                ymodisort = ymodi[iysort]
                if ysort[ny - ntop] < ysort[ny - 1] * maxthresh:
                    yran[1] = max(ysort[0:ny - ntop] + ymodisort[0:ny - ntop])
                
                # plotting
                plt.xlim(xran3[0], xran3[1])
                plt.ylim(yran[0], yran[1])
                plt.xlabel(xtit, fontsize=15)
                plt.ylabel(ytit, fontsize=15)
                if ystyle == 'log':
                    plt.yscale('log')
        
                # tick formatting
                plt.minorticks_on()
                plt.tick_params(which='major', length=10, pad=5)
                plt.tick_params(which='minor', length=5)
                if xunit == 'micron':
                    xticks = np.arange(np.around(xran3[0]), np.around(xran3[1]), 1)
                    #print(xran3[0], xran3[1], xticks)
                    plt.xticks(xticks, fontsize=10)
                    #plt.xticks(np.arange(xran3[0], xran3[1], .02), fontsize=10)
                else:
                    xticks = np.arange(math.floor(xran3[0]/1000.0)*1000, (math.floor(xran3[1]/1000.0)*1000)+1000, 10000)
                    #print(xran3[0], xran3[1], xticks)
                    plt.xticks(xticks, fontsize=10)
                    #plt.xticks(np.arange(xran3[0], xran3[1], 200), fontsize=10)
                plt.yticks(fontsize=10)
                
                # actually plotting
                plt.plot(MIRgdlambda, MIRgdflux, label='Data', color=dcolor)
                plt.plot(MIRgdlambda, MIRcontinuum, label='Model', color='r')
                
                if 'argscontfit' in initdat:
                    if 'global_ext_model' in initdat['argscontfit']:
                       for i in np.arange(0,len(comp_best_fit.keys())-2,1):
                          plt.plot(MIRgdlambda,np.multiply(comp_best_fit[list(comp_best_fit.keys())[i]],np.multiply(comp_best_fit[list(comp_best_fit.keys())[-2]],comp_best_fit[list(comp_best_fit.keys())[-1]])),label=list(comp_best_fit.keys())[i],linestyle='--',alpha=0.5)
        
                    else:
                       for i in np.arange(0,len(comp_best_fit.keys()),3):
                          plt.plot(MIRgdlambda,np.multiply(comp_best_fit[list(comp_best_fit.keys())[i]],np.multiply(comp_best_fit[list(comp_best_fit.keys())[i+1]],comp_best_fit[list(comp_best_fit.keys())[i+2]])),label=list(comp_best_fit.keys())[i],linestyle='--',alpha=0.5)
            '''
        # more formatting
        plt.subplots_adjust(hspace=0.25)
        plt.tight_layout(pad=15)
        plt.gcf().subplots_adjust(bottom=0.1)
        plt.gcf().subplots_adjust(right=0.85)
    
        if title is not None:
            plt.suptitle(title, fontsize=30)
            
        plt.savefig(outfile + '.jpg')