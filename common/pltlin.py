# -*- coding: utf-8 -*-
"""
Created on Mon Jun 29 08:38:08 2020

@author: lily
"""
# docformat= 'rst'

#+

#Plot emission line fit and output to JPG

#Categories: IFSFIT

#Returns: None.

#Params:
#instr:in, required, type=structure
   #contains results of fit
#pltpar:in, required, type=structure
   #contains parameters to control plot

#label=np.arrange(Nlines)
#label= str(label)
#line labels for plot
#wave= np.arrange((Nlines), float)
#rest wavelengths of lines
#lineoth= np.arrange((Notherlines, Ncomp), float)
#wavelengths of other lines to plot
#nx # of plot columns
#ny # of plot rows
#

#outfile: in, required, type=string
#Full path and name of output plot.

#Keywords:
#micron: in, optional, type=byte
#Label output plots in um rather than A. 
   #Input wavelengths still assumed to be in A.

#Author:
#  David S. N. Rupke:
#      Department of Physics
#      2000 N. Parkway
#      Memphis, TN 38104
#      drupke@gmail.com

#History:
#   ChangeHistory:
#      2009, DSNR, created
#      13sep12, DSNR, re-written
#      2013oct, DSNR, documented
#      2013nov21, DSNR, renamed, added license and copyright 
#      2015may13, DSNR, switched from using LAYOUT keyword to using CGLAYOUT
#                       procedure to fix layout issues
#      2016aug31, DSNR, added overplotting of continuum ranges masked during
#                       continuum fit with thick cyan line
#      2016sep13, DSNR, added MICRON keyword
#Copyright:
#    Copyright (C) 2013--2016 David S. N. Rupke
#
#    This program is free software: you can redistribute it and/or
#    modify it under the terms of the GNU General Public License as
#    published by the Free Software Foundation, either version 3 of
#    the License or any later version.

#   This program is distributed in the hope that it will be useful,
#    but WITHOUT ANY WARRANTY; without even the implied warranty of
#    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU
#    General Public License for more details.
#
#    You should have received a copy of the GNU General Public License
#    along with this program.  If not, see
#    http://www.gnu.org/licenses/.
#
#-
#

import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.lines import Line2D
import numpy as np
from astropy import modeling
import math
import pdb
from decimal import *

def pltlin(instr, pltpar, outfile):

 param=instr['param']
 ncomp= param[1]
 ncomp=int(ncomp)
 colors= ['Magenta', 'Green', 'Orange', 'Teal']

 wave = instr['wave']
 spectot = instr['spec']
 specstars = instr['cont_dat']
 speclines = instr['emlin_dat']
 modstars = instr['cont_fit']
 modlines = instr['emlin_fit']
 modtot = modstars + modlines
 ct_indx= instr['ct_indx']

#norm = max(modstars)
#spectot /= norm
#specstars /= norm
#speclines /= norm
#modtot /= norm
#modstars /= norm
#modlines /= norm

 zbase= instr['zstar']
   

#Find masked regions during continuum fit
 nmasked=0 # number of masked regions
#Find consecutive unmasked 
 lct, hct, nct = consec(ct_indx)
#Set interior masked regions
 if nct> 1:
    nmasked= nct-1
    masklam= np.zeros((nmasked, 2), dtype=float)
    for i in range (0, nmasked):
        masklam[i,0] = wave[ct_indx[hct[i]]]
        masklam[i,1]=wave[ct_indx[lct[i+1]]]       
        
#Set masked region if it occurs at beginning of lambda array
 if ct_indx[0] != 0:
      ++nmasked
      a=[wave[0], wave[ct_indx[lct[0]-1]]]
      masklam= np.concatenate(a, masklam)

#Set masked region if it occurs at end of lambda array
 if ct_indx[len(ct_indx)-1] != len(wave)-1: 
      ++nmasked
      b=[wave[ct_indx[hct[nct-1]]], wave[len(wave)]]
      masklam= np.concatenate(masklam, b)
      
 nlin= len(pltpar['label'])
 linlab= pltpar['label']
 linwav= pltpar['wave']
 off= pltpar['off']
 nx=pltpar['nx']
 ny=pltpar['ny']
 if 'linoth' in pltpar:
    linoth=pltpar['linoth']
 else: 
    linoth= np.arange(1, nlin)
    str(linoth) 
 

 plt.style.use('dark_background') 
 fig = plt.figure(figsize=(16,13))
 for i in range (0,nlin):
    outer = gridspec.GridSpec(ny, nx, wspace=0.2, hspace=0.2)
    inner = gridspec.GridSpecFromSubplotSpec (2, 1, \
            subplot_spec=outer[i], wspace=0.1, hspace=0, height_ratios=[4,2], width_ratios=None)
 #create xran and ind
    linwavtmp= linwav[i]
    offtmp=np.array(off)[i,:]
    xran = (linwavtmp + offtmp)
    xran= xran*(1 + zbase)
    ind=np.array([0])
    for h in range(0, len(wave)):
        if wave[h]>xran[0] and wave[h]<xran[1]:
            ind=np.append(ind, h)
    ind=np.delete(ind,[0])
    ct=len(ind)
    if ct > 0:
 #create subplots
        ax0 = plt.Subplot(fig, inner[0])
        ax1 = plt.Subplot(fig, inner[1])
        ax0.annotate(linlab[i], (0.05, 0.9), xycoords='axes fraction', va='center', fontsize=15)
        fig.add_subplot(ax0)
        fig.add_subplot(ax1)
 #create x-ticks
        xticks=np.array([0])
        if 'micron' in pltpar:
            for t in range (int(xran[0]), int(xran[1])):
                if t%(0.5E4)==0:
                    xticks=np.append(xticks, t)
        elif 'meter' in pltpar:
            for t in range (int(xran[0]), int(xran[1])):
                if t%(0.5E10)==0:
                    xticks=np.append(xticks, t)
        else:
            for t in range (int(xran[0]), int(xran[1])):
                if t%50==0:
                    xticks=np.append(xticks, t)
        xticks=np.delete(xticks,[0])
 #create minor x-ticks
        xmticks=np.array([0])
        if 'micron' in pltpar:
            for t in range (int(xran[0]), int(xran[1])):
                if t%(0.5E4)==0:
                    xmticks=np.append(xmticks, t)
        elif 'meter' in pltpar:
            for t in range (int(xran[0]), int(xran[1])):
                if t%(0.5E10)==0:
                    xmticks=np.append(xmticks, t)
        else:
            for t in range (int(xran[0]), int(xran[1])):
                if t%10==0:
                    xmticks=np.append(xmticks, t)
        xmticks=np.delete(xmticks,[0])
 #set ticks
        ax0.set_xticks(xticks)
        ax1.set_xticks(xticks)
        ax0.set_xticks(xmticks, minor=True)
        ax1.set_xticks(xmticks, minor=True)
        ax0.tick_params('x', which='major', direction='in', length=7, width=2, color='white')
        ax0.tick_params('x', which='minor', direction='in', length=5, width=1, color='white')
        ax1.tick_params('x', which='major', direction='in', length=7, width=2, color='white')
        ax1.tick_params('x', which='minor', direction='in', length=5, width=1,  color='white')
 #create yran
        ydat = spectot
        ymod = modtot
        ydattmp=np.zeros((ct), dtype=float)
        ymodtmp=np.zeros((ct), dtype=float)
        for l in range (0, len(ind)):
             ydattmp[l]= ydat[(ind[l])]
             ymodtmp[l]= ymod[(ind[l])]
        ydatmin=min(ydattmp)
        ymodmin=min(ymodtmp)
        if ydatmin <= ymodmin:
            yranmin=ydatmin
        else:
            yranmin=ymodmin
        ydatmax=max(ydattmp)
        ymodmax=max(ymodtmp)
        if ydatmax >= ymodmax:
            yranmax=ydatmax
        else:
            yranmax=ymodmax
        yran=[yranmin, yranmax]
        icol = (float(i))/(float(nx))
        if icol%1==0:
            ytit='Fit'
        else:
            ytit= ''
        ax0.set(ylabel=ytit)
        ax0.set_xlim([xran[0], xran[1]])
        ax0.set_ylim([yran[0], yran[1]])
 #plots on ax0
        ax0.plot (wave,ydat, color='White', linewidth=1)
        xtit = 'Observed Wavelength ($\AA$)'
        ytit=''
        ax0.plot (wave, ymod,color='Red', linewidth=2)   
        for j in range(1, ncomp+1):
          flux= cmplin(instr, linlab[i], j, velsig=1)
          for p in range (0, len(flux)):
              flux[p]=float(flux[p])
          ax0.plot(wave, (yran[0]+flux), color=colors[j-1], linewidth=1, linestyle='dashed')
          if linoth[0, i] != '':
             for k in range (0, (len(linoth[:,i]))):
                  if linoth[k,i] != '':
                       flux=cmplin(instr, linoth[k,i], j, velsig=1)
                       for p in range(0, len(flux)):
                           flux[p]=float(flux[p])
                       ax0.plot(wave,(yran[0]+flux), color=colors[j-1], linewidth=1, linestyle='dashed')
        xloc=xran[0]+(xran[1]-xran[0])*(float(0.05))
        yloc=yran[0]+(yran[1]-yran[0])*(float(0.85))
        plt.text(xloc, yloc, linlab[i], fontsize=2)        
        if nmasked > 0:
          for r in range (0,nmasked):
               ax0.plot([masklam[r,0], masklam[r,1]], [yran[0], yran[0]],linewidth=8, color='Cyan')
 #set new value for yran
        ydat = specstars
        ymod = modstars
        ydattmp=np.zeros((len(ind)), dtype=float)
        ymodtmp=np.zeros((len(ind)), dtype=float)
        for l in range (0, len(ind)):
             ydattmp[l]= ydat[(ind[l])]
             ymodtmp[l]= ymod[(ind[l])]
        ydatmin=min(ydattmp)
        ymodmin=min(ymodtmp)
        if ydatmin <= ymodmin:
            yranmin=ydatmin
        else:
            yranmin=ymodmin
        ydatmax=max(ydattmp)
        ymodmax=max(ymodtmp)
        if ydatmax >= ymodmax:
            yranmax=ydatmax
        else:
            yranmax=ymodmax
        yran=[yranmin, yranmax]
        if icol%1==0:
            ytit = 'Residual' 
        else:
            ytit = ''
        ax1.set(ylabel=ytit)
 #plots on ax1
        ax1.set_xlim([xran[0], xran[1]])
        ax1.set_ylim([yran[0], yran[1]])
        ax1.plot(wave, ydat, linewidth=1)
        ax1.plot(wave,ymod,color='Red')
   
#titles
 if 'micron' in pltpar:
    xtit= 'Observed Wavelength (\u03BC)'
 elif 'meter' in pltpar:
    xtit = 'Observed Wavelength (m)'
 else: 
    xtit = 'Observed Wavelength ($\AA$)'
 fig.suptitle(xtit, fontsize=25)

 tmpfile = outfile
 plt.show()
 fig.savefig(tmpfile + '.jpg')
