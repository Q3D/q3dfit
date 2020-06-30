# -*- coding: utf-8 -*-
"""
Created on Mon Jun 29 08:38:08 2020

@author: lily
"""

import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np
from astropy import modeling
import math
import pdb
from decimal import *
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
#int(nx) # of plot columns
#int(ny) # of plot rows
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

def pltlin(instr, pltpar, outfile):
#in IDL this next section of code directs the output to z-buffer pseudo device
#and sets the resolution, depth, character size/thickness, line thickness
#and erases it
 # plt.minorticks_on()
 # plt.xticks(step=50)
 # plt.xminorticks(step=10)

 # if hasattr(pltpar,'micron'):
 #     plt.xticks(step=0.5E4)
 #     float(plt.xticks)
 #     plt.xminorticks(step=0.5E4)
 #     float(plt.xminorticks)
 # if hasattr(pltpar,'meter'):
 #    plt.xticks(step=0.5E10)
 #    float(plt.xticks)
 #    plt.xminorticks(step=0.5E10)
 #    float(plt.xminorticks)
 
 pos=np.zeros((pltpar['nx'], pltpar['ny']), dtype=int)
 #stuff about margins
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

 # if hasattr (pltpar, 'micron'):
 #    wave=(1E4)
 #    float(wave)
 # if hasattr (pltpar, 'meter'):
 #    wave=(1E10)
 #    float(wave)
   

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
 if 'linoth' in pltpar:
    linoth=pltpar['linoth']
 else: 
    linoth= np.arange(1, nlin)
    str(linoth) 
 
 #axes[0,0].plot([0])
 plt.style.use('dark_background') 
 plt.axis('off') #so the subplots don't share a y-axis   
 
 for i in range (0,nlin):
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
    print(ct)
    if ct > 0:
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
        icol = (float(i))/(float(pltpar['nx']))
        if icol == int(icol):
            ytit='Fit'
        else:
            ytit= ''
        
        plt.plot (wave,ydat, color='White', linewidth=1)
        xtit = 'Observed Wavelength ($\AA$)'
        ytit=''
        plt.xlim([xran[0], xran[1]])
        plt.ylim([yran[0], yran[1]])
        plt.plot (wave, ymod,color='Red', linewidth=2)   
        for j in range(1, ncomp+1):
          flux= cmplin(instr, linlab[i], j, velsig=1)
          for p in range (0, len(flux)):
              flux[p]=float(flux[p])
          plt.plot(wave, (yran[0]+flux), color=colors[j-1], linewidth=2, linestyle='dashed')
          if linoth[0, i] != '':
             for k in range (0, (len(linoth[:,i]))):
                  if linoth[k,i] != '':
                       flux=cmplin(instr, linoth[k,i], j, velsig=1)
                       for p in range(0, len(flux)):
                         flux[p]=float(flux[p])
                       plt.plot(wave,(yran[0]+flux), color=colors[j-1], linestyle='dashed')
        xloc=xran[0]+(xran[1]-xran[0])*(float(0.05))
        yloc=yran[0]+(yran[1]-yran[0])*(float(0.85))
        plt.text(xloc, yloc, linlab[i], fontsize=2)        
        if nmasked > 0:
          for r in range (0,nmasked):
               plt.plot([masklam[r,0], masklam[r,1]], [yran[0], yran[0]],linewidth=8, color='Cyan')
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
        if icol == int(icol):
            ytit = 'Residual' 
        else:
            ytit = ''
        #plt.plot(wave, ydat, linewidth=1)
        #plt.plot(wave,ymod,color='Red',thick=4)
        plt.show()
        
 if 'micron' in pltpar:
    xtit= 'Observed Wavelength (\u03BC)'
 elif 'meter' in pltpar:
    xtit = 'Observed Wavelength (m)'
 else: 
    xtit = 'Observed Wavelength ($\AA$)'
    plt.text(0.5, 0.02, xtit, fontsize=2)
    tmpfile = outfile
 plt.savefig(tmpfile + '.jpg')
 
def consec(a, l = None, h = None, n = None, same = None, distribution = None):
   #a=array
    nel = len(a)
    if nel == 1: 
        l = 0
        h = 0
        n = 1        
    elif nel == 2:
        if same != None:
            if a[1] - a[0] == 0:
                l = 0
                h = 1
                n = 1
            else: 
                l = -1
                h = -1
                n = 0                
        else:
            if abs(a[1] - a[0]) == 1:
                l = 0
                h = 1
                n = 1
            else:
                l = -1
                h = -1
                n = 0      
    else:
        if same == None:
            #adding padding
            temp = np.concatenate(([a[0]], a))
            arr = np.concatenate((temp, [a[-1]]))
            
            shiftedright = np.roll(arr, 1)
            shiftedleft = np.roll(arr, -1)

           
            cond1 = np.absolute(np.subtract(arr, shiftedright)) == 1
            cond2 = np.absolute(np.subtract(arr, shiftedleft)) == 1

        else:
            #adding padding
            temp = np.concatenate(([a[0] + 1], a))
            arr = np.concatenate((temp, [a[-1] - 1]))
            
            shiftedright = np.roll(arr, 1)
            shiftedleft = np.roll(arr, -1)
            
            cond1 = np.absolute(np.subtract(arr, shiftedright)) == 0
            cond2 = np.absolute(np.subtract(arr, shiftedleft)) == 0        
        
        #getting rid of padding
        cond1 = cond1[1: -1]
        cond2 = cond2[1: -1]
        
        #making l
        l = [0]
        l.pop(0)
        for i in range (0, nel):
            if cond2[i] and cond1[i] == False:
                l.append(i)
        nl = len(l)        

        #making h        
        h = [0]
        h.pop(0)
        for i in range (0, nel):
            if cond1[i] and cond2[i] == False:
                h.append(i)        
        nh = len(h)
        
        if nh * nl == 0: 
            l = -1
            h = -1
            n = 0 
        
        else: n = min(nh, nl)
    
    if l[0] != h[0]: dist = np.subtract(h, l) + 1 
    else: dist = 0

    return l, h, n

#placeholder for flux
fluxfile= open("fluxph.txt", 'r')
fluxph = np.zeros((6105), dtype=float)
count = 0
for i in fluxfile.readlines():
  for j in i.split():
      if count < 6105:
        row = count % 6105
        fluxph[row] = j
        count+=1
fluxfile.close()

outfile= 'pltlingraphs'
linoth = np.full((2,6),'', dtype=object)
linoth[0,2] = '[OIII]4959'
linoth[0,3] = '[OI]6364'
linoth[:,4] = ['[NII]6548','[NII]6583']
linoth[0,5] = '[SII]6716'
pltpar = {'nx': 3,'ny': 2,
'label': ['','Hbeta','[OIII]5007','[OI]6300','Halpha','[SII]6731'],
'wave': [0,4861,5007,6300,6563,6731],
'off': [[-120,90],[-80,50],[-130,50],
       [-80,120],[-95,70],[-95,50]],
'linoth': linoth}

#instr.param
paramfile= open("param.txt", 'r')
paramarr = np.zeros((37), dtype=float)
count = 0
for i in paramfile.readlines():
  for j in i.split():
      if count < 37:
        row = count % 37
        paramarr[row] = j
        count+=1
paramfile.close()

#instr.wave
wavefile = open("wave.txt", 'r')
wavearr= np.zeros((6105), dtype=float)
count = 0
for i in wavefile.readlines():
  for j in i.split():
      if count < 6105:
        row = count % 6105
        wavearr[row]= j
        count+=1
wavefile.close()

#instr.spec
specfile = open("spec.txt", 'r')
specarr= np.zeros((6105), dtype=float)
count = 0
for i in specfile.readlines():
  for j in i.split():
      if count < 6105:
        row = count % 6105
        specarr[row]= j
        count+=1
specfile.close()
        
#instr.cont_dat
cont_datfile = open("cont_dat.txt", 'r')
cont_datarr= np.zeros((6105), dtype=float)
count = 0
for i in cont_datfile.readlines():
  for j in i.split():
      if count < 6105:
        row = count % 6105
        cont_datarr[row]= j
        count+=1
cont_datfile.close()

#instr.emlin_dat
emlin_datfile = open("emlin_dat.txt", 'r')
emlin_datarr= np.zeros((6105), dtype=float)
count = 0
for i in emlin_datfile.readlines():
  for j in i.split():
      if count < 6105:
        row = count % 6105
        emlin_datarr[row] = j
        count+=1
emlin_datfile.close()

#instr.cont_fit
cont_fitfile = open("cont_fit.txt", 'r')
cont_fitarr= np.zeros((6105), dtype=float)
count = 0
for i in cont_fitfile.readlines():
  for j in i.split():
      if count < 6105:
        row = count % 6105
        cont_fitarr[row] = j
        count+=1
cont_fitfile.close()

#instr.emlin_fit
emlin_fitfile = open("emlin_fit.txt", 'r')
emlin_fitarr= np.zeros((6105), dtype=float)
count = 0
for i in emlin_fitfile.readlines():
  for j in i.split():
      if count < 6105:
        row = count % 6105
        emlin_fitarr[row] = j
        count+=1
emlin_fitfile.close()

#instr.zstar
zstar=0.089302800717023054

#instr.ct_indx
ct_indxfile= open("ct_indx.txt", 'r')
ct_indxarr = np.zeros((5800), dtype=int)
count = 0
for i in ct_indxfile.readlines():
  for j in i.split():
      if count < 5800:
        row = count % 5800
        ct_indxarr[row] = j
        count+=1
ct_indxfile.close()
linelabelarr=['Halpha','Hbeta', '[OI]6300', '[OI]6364', \
              '[OIII]4959', '[OIII]5007', '[NII]6548', \
              '[NII]6583',  '[SII]6716',    '[SII]6731']
parinfo = {'line': ['', '', '', '', '', '', '', \
                'Hbeta', 'Hbeta', 'Hbeta', 'Halpha', 'Halpha', 'Halpha', \
                '[OIII]4959', '[OIII]4959', '[OIII]4959', '[NII]6583', \
                '[NII]6583',  '[NII]6583',  '[SII]6716',  '[SII]6716', \
                '[SII]6716',  '[OI]6364',   '[OI]6364',   '[OI]6364', \
                '[SII]6731',  '[SII]6731',  '[SII]6731',  '[OIII]5007', \
                '[OIII]5007', '[OIII]5007', '[NII]6548',  '[NII]6548', \
                '[NII]6548',  '[OI]6300',   '[OI]6300',   '[OI]6300'],
            'comp':[0.0000000,       0.0000000,       0.0000000, \
                    1.0000000,       0.0000000,       1.0000000, \
                    0.0000000,       1.0000000,       1.0000000, \
                    1.0000000,       1.0000000,       1.0000000, \
                    1.0000000,       1.0000000,       1.0000000, \
                    1.0000000,       1.0000000,       1.0000000, \
                    1.0000000,       1.0000000,       1.0000000, \
                    1.0000000,       1.0000000,       1.0000000, \
                    1.0000000,       1.0000000,       1.0000000, \
                    1.0000000,       1.0000000,       1.0000000, \
                    1.0000000,       1.0000000,       1.0000000, \
                    1.0000000,       1.0000000,       1.0000000, \
                    1.0000000]}
instr= {"param":paramarr, "wave":wavearr, "spec":specarr, "cont_dat":cont_datarr, \
        "emlin_dat":emlin_datarr, "emlin_fit":emlin_fitarr, "cont_fit":cont_fitarr, \
            "zstar":zstar, "ct_indx":ct_indxarr, "linelabel":linelabelarr, "parinfo":parinfo}

def cmplin(instr, line, comp, velsig = None):

    c = 299792.458
        
    iline = [None]
    iline.pop(0)

    for i in range (0, len(instr['linelabel'])):
        if instr['linelabel'][i] == line:
            iline.append(i)
    ct = len(iline)
    ppoff = instr['param'][0]
    ncomp = instr['param'][1]
    specres = instr['param'][2]

    if 'parinfo' in instr:                
        indices = [None]
        indices.pop(0)
        for i in range (0, len(instr['parinfo']['line'])):
            if instr['parinfo']['line'][i] == line and \
                instr['parinfo']['comp'][i] == comp:
                indices.append(i)
    else:
        nline = len(instr['linelabel'])
        #not sure why it did where inelabel == line a Second time here? took it out
        indices = instr['param'][0] + (comp - 1) * nline * 3 + iline * 3
        indices = indices = indices[0] + np.arange(0, 3, dtype = float)

    if indices[0] != -1:
        gausspar = [None]
        for i in indices:
            gausspar.append(instr['param'][i])
        gausspar = gausspar[1:] #haha
        if velsig != None:
            gausspar[2] = np.sqrt((gausspar[2] * gausspar[1]/c)**2.0 \
                    + specres ** 2.0)
        else: gausspar[2] = np.sqrt(gausspar[2]**2.0 + specres**2.0)
        
        if gausspar[2] == 0.0: flux = 0.0
        else: flux = gaussian(instr['wave'], gausspar)

    else: 
        flux = 0.0
    return flux

def gaussian(xi, parms):
    getcontext().prec = 40
    a = parms[0] #amp
    b = parms[1] #mean
    c = parms[2] #standard dev
    g = [0.0] #gaussian
        
    for x in xi:
        hl = Decimal(a) * Decimal(-0.5 * ((x - b) / c)**2).exp()
        g.append(hl)
    g = g[1:] #lol
    
    return g

pltlin(instr, pltpar, outfile)
