# -*- coding: utf-8 -*-
"""
Plots continuum fit and outputs to JPG.
@author: hadley
"""
import pdb
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np
#from more_itertools import consecutive_groups


def pltcont(instr, outfile, compspec=None, comptitles=None, ps=None,
            title=None, fitran=None, yranminmax=None):

    if compspec is not None:
        if len(compspec) > 1:
            ncomp = len(compspec)
        else:
            ncomp = 1
        compcolors = ['c', 'y', 'm']
    else:
        ncomp = 0

    wave = instr['wave']
    specstars = instr['cont_dat']
    # speclines = instr['emlin_dat']
    modstars = instr['cont_fit']

    if fitran is not None:
        xran = fitran
    else:
        xran = instr['fitran']
    dxran = xran[1] - xran[0]
    xran1 = [xran[0], xran[0] + dxran//3.0]
    xran2 = [xran[0] + dxran//3.0, xran[0] + 2.0 * dxran//3.0]
    xran3 = [xran[0] + 2.0 * dxran//3.0, xran[1]]
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

    # Find masked regions
    # nmasked = 0
    # imasked = []
    # Groups of consecutive integers
    # for group in consecutive_groups(instr['ct_indx']):
    #     imasked = [imasked] + list(group)

    # if nct > 1:
    #     nmasked = nct - 1
    #     masklam=[[0] * 2] * nmasked
    #     masklam = np.array(masklam, dtype = float)
    #     for i in range(0, nmasked):
    #         masklam[i][0] = wave[instr['ct_indx'][hct[i]]]
    #         masklam[i][1] = wave[instr['ct_indx'][lct[i + 1]]]

    # if instr['ct_indx'][0] != 0:
    #     ++nmasked
    #     masklam = [[wave[0], wave[instr['ct_indx'][lct[0] - 1]]]] + masklam
    # if instr['ct_indx'][len(instr['ct_indx']) - 1] != len(wave)-1:
    #     ++nmasked
    #     masklam = masklam + [[wave[instr['ct_indx'][hct[nct - 1]]], \
    #                           wave[len(wave)-1]]]

    maxthresh = 0.2
    ntop = 20
    nbottom = 20
    if len(wave) < 100:
        ntop = 10
        nbottom = 10
    ++ntop
    --nbottom

    xtit = 'Observed Wavelength ($\AA$)'
    ytit = ''
    plt.style.use('dark_background')
    fig = plt.figure(figsize=(50, 30))
    plt.axis('off')  # so the subplots don't share a y-axis

#    masklam = np.array(masklam)

    if ct1 > 0:
        fig.add_subplot(3, 1, 1)
        # custom legend
        if ncomp > 0 and comptitles is not None:
            custom_lines = []
            for i in range(0, ncomp):
                custom_lines.append(Line2D([0], [0],
                                           color=compcolors[i], lw=4))
            custom_lines.append(Line2D([0], [0], color='r', lw=4))
            plt.legend(custom_lines, comptitles + ['total'],
                       prop={"size": 30})

        ydat = specstars
        ymod = modstars

        maximum = 0
        # finding max value between ydat and ymod at indices from i1
        for i in i1:
            bigboy = max(ydat[i], ymod[i])
            if bigboy > maximum:
                maximum = bigboy
        # finding min
        minimum = 0
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
        plt.ylabel(ytit)

        # Tick formatting
        plt.minorticks_on()
        plt.tick_params(which='major', length=20, pad=30)
        plt.tick_params(which='minor', length=10)
        plt.xticks(np.arange(xran1[0], xran1[1], 200), fontsize=30)
        # This will fail if fluxes are very low (<~1e-10)
        plt.yticks(np.arange(yran[0], yran[1],
                             np.around((yran[1] - yran[0])/5.,
                                       decimals=10)), fontsize=25)

        # actually plotting
        plt.plot(wave, ydat, 'w', linewidth=1)

        if ncomp > 0:
            for i in range(0, ncomp):
                plt.plot(wave, compspec[i], compcolors[i], linewidth=3)

        plt.plot(wave, ymod, 'r', linewidth=4, label='Total')

        # if nmasked > 0:
        #     for i in range(0, nmasked):
        #         plt.plot([masklam[i][0], masklam[i][1]], [yran[0], yran[0]],
        #                  'c', linewidth=20,  solid_capstyle="butt")

    # like previous section
    if ct2 > 0:
        fig.add_subplot(3, 1, 2)
        plt.minorticks_on()

        ydat = specstars
        ymod = modstars

        # finding max
        maximum = 0
        for i in i2:
            bigboy = max(ydat[i], ymod[i])
            if bigboy > maximum:
                maximum = bigboy
        # finding min
        minimum = 0
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
        plt.ylabel(ytit)
        plt.xticks(np.arange(xran2[0], xran2[1], 200), fontsize=30)
        plt.yticks(fontsize=25)
        plt.tick_params(which='major', length=20, pad=30)
        plt.tick_params(which='minor', length=10)

        # yay more plotting
        plt.plot(wave, ydat, 'w', linewidth=1)

        if ncomp > 0:
            for i in range(0, ncomp):
                plt.plot(wave, compspec[i], compcolors[i], linewidth=3)

        plt.plot(wave, ymod, 'r', linewidth=4)

        # if nmasked > 0:
        #     for i in range(0, nmasked):
        #         plt.plot([masklam[i][0],masklam[i][1]], [yran[0], yran[0]], \
        #                  'c', linewidth=20,  solid_capstyle="butt")

    # and again
    if ct3 > 0:
        fig.add_subplot(3, 1, 3)
        plt.subplots_adjust(hspace=2)
        plt.minorticks_on()

        ydat = specstars
        ymod = modstars

        # finding max
        maximum = 0
        for i in i3:
            bigboy = max(ydat[i], ymod[i])
            if bigboy > maximum:
                maximum = bigboy
        # finding min
        minimum = 0
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
        plt.xlabel(xtit, fontsize=45, labelpad=35)
        plt.ylabel(ytit)
        plt.xticks(np.arange(xran3[0], xran3[1], 200), fontsize=30)
        plt.yticks(fontsize=25)
        plt.tick_params(which='major', length=20, pad=30)
        plt.tick_params(which='minor', length=10)

        # plotting more
        plt.plot(wave, ydat, 'w', linewidth=1)
        if ncomp > 0:
            for i in range(0, ncomp):
                plt.plot(wave, compspec[i], compcolors[i], linewidth=3)

        plt.plot(wave, ymod, 'r', linewidth=4)

        # if nmasked > 0:
        #     for i in range(0, nmasked):
        #         plt.plot([masklam[i][0],masklam[i][1]], [yran[0], yran[0]], \
        #                  'c', linewidth=20, solid_capstyle="butt")

    # more formatting
    plt.subplots_adjust(hspace=0.25)
    plt.tight_layout(pad=10)

    if title is not None:
        plt.suptitle(title, fontsize=50)

    plt.savefig(outfile + '.jpg')
