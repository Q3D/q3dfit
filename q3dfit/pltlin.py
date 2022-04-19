# -*- coding: utf-8 -*-

from q3dfit.cmplin import cmplin
from q3dfit.lmlabel import lmlabel
from q3dfit.exceptions import InitializationError

import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import numpy as np
import pdb


def pltlin(instr, pltpar, outfile):
    """

    Plot emission line fit and output to JPG

    Parameters
    ----------
    instr : dict
       contains results of fit
    pltpar : dict
       contains parameters to control plot
        label=np.arrange(Nlines)
        label= str(label)
        line labels for plot
        wave= np.arrange((Nlines), float)
        rest wavelengths of lines
        lineoth= np.arrange((Notherlines, Ncomp), float)
        wavelengths of other lines to plot
        nx # of plot columns
        ny # of plot rows
    outfile : str
        Full path and name of output plot.

    """
    ncomp = instr['maxncomp']
    colors = ['Magenta', 'Green', 'Orange', 'Teal']

    wave = instr['wave']
    spectot = instr['spec']
    specstars = instr['cont_dat']
    modstars = instr['cont_fit']
    modlines = instr['emlin_fit']
    modtot = modstars + modlines

    # To-do: Allow output wavelengths in Angstrom
    #'waveunit_out' = 'micron'
    # if 'waveunit_out' in pltpar:
    #     if pltpar['waveunit_out'] = 'Angstrom':
    #         waveunit_out = 'Angstrom'

    # To-do: Get masking code from pltcont

    # lines
    linelist = instr['linelist']['lines']
    linelabel = instr['linelist']['name']
    linetext = instr['linelist']['linelab']
    # Sort in wavelength order
    isortlam = np.argsort(linelist)
    linelist = linelist[isortlam]
    linelabel = linelabel[isortlam]
    linetext = linetext[isortlam]

    #
    # Plotting parameters
    #
    nx = pltpar['nx']
    ny = pltpar['ny']
    # Look for line list, then determine center of plot window from fitted
    # wavelength
    if 'line' in pltpar:
        sub_linlab = pltpar['line']
        linwav = np.empty(len(sub_linlab), dtype='float32')
        for i in range(0, len(sub_linlab)):
            # Get wavelength from zeroth component
            if sub_linlab[i] != '':
                lmline = lmlabel(sub_linlab[i])
                # if ncomp > 0
                if f'{lmline.lmlabel}_0_cwv' in instr['param'].keys():
                    linwav[i] = instr['param'][f'{lmline.lmlabel}_0_cwv']
                # otherwise
                else:
                    idx = np.where(instr['linelist']['name'] == sub_linlab[i])
                    linwav[i] = instr['linelist']['lines'][idx] * \
                        (1. + instr['zstar'])
            else:
                linwav[i] = 0.
    # If linelist not present, get cwavelength enter of plot window from list
    # first option: wavelength center specified in observed (plotted) frame
    elif 'center_obs' in pltpar:
        linwav = np.array(pltpar['center_obs'])
    # second option: wavelength center specified in rest frame, then converted
    # to observed (plotted) frame
    elif 'center_rest' in pltpar:
        linwav = np.array(pltpar['center_rest']) * instr['zstar']
    else:
        raise InitializationError('LINE, CENTER_OBS, or CENTER_REST ' +
                                  'list not given in ARGSPLTLIN dictionary')
    nlin = len(linwav)
    # Size of plot in wavelength, in observed frame
    if 'size' in pltpar:
        size = np.array(pltpar['size'])
    else:
        size = np.full(nlin, 300.)  # default size currently 300 A ... fix for
        # other units!
    off = np.array([-1.*size/2., size/2.])
    off = off.transpose()

    plt.style.use('dark_background')
    fig = plt.figure(figsize=(16, 13))
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
            # if waveunit_out == 'Angstrom':
            #     xticks = xticks * 1.E4
            # create minor x-ticks
            xmticks = np.linspace(xran[0],xran[1],num=25,endpoint=False)
            xmticks = np.delete(xmticks, [0])
            # if waveunit_out == 'Angstrom':
            #     xmticks = xticks * 1.E4
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
            xtit = 'Observed Wavelength ($\mu$m)'
            # if waveunit_out == 'Angstrom':
            #     xtit = 'Observed Wavelength ($\AA$)'
            ytit = ''
            ax0.plot(wave, ymod, color='Red', linewidth=2)
            # Plot all lines visible in plot range
            for j in range(0, ncomp):
                ylaboff = 0.07
                for k, line in enumerate(linelabel):
                    lmline = lmlabel(line)
                    if f'{lmline.lmlabel}_{j}_cwv' in instr['param'].keys():
                        refwav = instr['param'][f'{lmline.lmlabel}_{j}_cwv']
                    else:
                        irefwav = np.where(instr['linelist']['name'] == line)
                        refwav = instr['linelist']['lines'][irefwav] * \
                            (1. + instr['zstar'])
                    if refwav >= xran[0] and refwav <= xran[1]:
                        if f'{lmline.lmlabel}_{j}_cwv' in \
                            instr['param'].keys():
                            flux = cmplin(instr, line, j, velsig=True)
                            ax0.plot(wave, yran[0] + flux, color=colors[j],
                                     linewidth=2, linestyle='dashed')
                        ax0.annotate(linetext[k], (0.05, 1. - ylaboff),
                                     xycoords='axes fraction',
                                     va='center', fontsize=15)
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
    xtit = 'Observed Wavelength ($\mu$m)'
    # if waveunit_out == 'Angstrom':
    #     xtit = 'Observed Wavelength ($\AA$)'
    fig.suptitle(xtit, fontsize=25)

    fig.savefig(outfile + '.jpg')
