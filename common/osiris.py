# -*- coding: utf-8 -*-
"""
; docformat = 'rst'
;
;+
;
; Initialize parameters for fitting.
;
; :Categories:
;    IFSFIT/INIT
;
; :Returns:
;    PARINFO structure for input into MPFIT.
;
; :Params:
;    linelist: in, required, type=hash(lines)
;      Emission line rest frame wavelengths.
;    linelistz: in, required, type=hash(lines\,maxncomp)
;      Emission line observed frame wavelengths.
;    linetie: in, required, type=hash(lines)
;      Name of emission line to which each emission line is tied
;      (in redshift and linewidth).
;    initflux: in, required, type=hash(lines\,maxncomp)
;      Initial guess for peak flux in each component.
;    initsig: in, required, type=hash(lines\,maxncomp)
;      Initial guess for emission lines widths.
;    maxncomp: in, required, type=double
;      Maximum no. of emission line components.
;    ncomp: in, required, type=hash(lines)
;      Number of velocity components.
;      
; :Keywords:
;    lratfix: in, optional, type=hash(lineratios,ncomp)
;      For each line ratio that should be fixed, input an array with each 
;      element set to either the BAD value (do not fix that component) or the 
;      value to which the line ratio will be fixed for that component.
;    siglim: in, optional, type=dblarr(2)
;      Lower and upper sigma limits in km/s.
;    sigfix: in, optional, type=hash(lines\,maxncomp)
;      Fix sigma at this value, for particular lines/components.
; 
; :Author:
;    David S. N. Rupke::
;      Rhodes College
;      Department of Physics
;      2000 N. Parkway
;      Memphis, TN 38104
;      drupke@gmail.com
;
; :History:
;    ChangeHistory::
;      2016sep13, DSNR, copied from IFSF_GMOS
;      2020may26, YI, translated to Python 3
;    
; :Copyright:
;    Copyright (C) 2016 David S. N. Rupke
;
;    This program is free software: you can redistribute it and/or
;    modify it under the terms of the GNU General Public License as
;    published by the Free Software Foundation, either version 3 of
;    the License or any later version.
;
;    This program is distributed in the hope that it will be useful,
;    but WITHOUT ANY WARRANTY; without even the implied warranty of
;    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU
;    General Public License for more details.
;
;    You should have received a copy of the GNU General Public License
;    along with this program.  If not, see
;    http://www.gnu.org/licenses/.
;
;-
"""
import numpy as np

def osiris(linelist, linelistz,linetie,initflux,initsig,maxncomp,ncomp,
               lratfix=None,siglim=None,sigfix=None,specres=None) :
    bad = 1e99
    c=299792.458
    
    # Sigma limits
    res = 3000.
    if not siglim :
        siglim = [299792./res/2.35,2000.]
    
    # Number of emission lines to fit
    nline = len(linelist)
    lines_arr = linelist #these 2 steps highly depend on the structure of the input
    # Number of initial parameters before Gaussian parameters begin
    ppoff0 = 3
    ppoff = ppoff0 
    
    pardict = {'value':0.,'fixed':0.,'limited':[b'0',b'0'],'tied':'',
               'limits':[0.,0.],'step':0.,'mpprint':b'0','mpside':2,
               'parname':'','line':'','comp':0.,'sigmawave_tie':'','flux_tie':''}
    parinfo = list(np.repeat(pardict,ppoff+maxncomp*(nline*3)))

    # Number of initial parameters before Gaussian parameters begin
    parinfo[0]['value'] = ppoff
    parinfo[0]['fixed'] = b'1'
    parinfo[0]['parname'] = 'No. of non-Gaussian parameters'
    
    # Maximum number of velocity components
    parinfo[1]['value'] = maxncomp
    parinfo[1]['fixed'] = b'1'
    parinfo[1]['parname'] = 'Maximum no. of velocity components'
    
    # Spectral resolution
    parinfo[2]['value'] = specres
    parinfo[2]['fixed'] = b'1'
    parinfo[2]['parname'] = 'Spectral resolution in wavelength space [sigma]'
    
    
    # cycle through velocity components
    for i,in range(0,maxncomp):
        # index offsets for this component
        foff = ppoff+i*nline*3
        woff = foff+1
        soff = foff+2
        # cycle through lines
        iline=0
        for line in lines_arr:
        # indices
            ifoff = foff + iline*3
            iwoff = woff + iline*3
            isoff = soff + iline*3
            parinfo[ifoff]['parname'] = 'flux_peak'
            parinfo[iwoff]['parname'] = 'wavelength'
            parinfo[isoff]['parname'] = 'sigma'
            parinfo[ifoff]['line'] = line
            parinfo[iwoff]['line'] = line
            parinfo[isoff]['line'] = line
            parinfo[ifoff]['comp'] = i+1
            parinfo[iwoff]['comp'] = i+1
            parinfo[isoff]['comp'] = i+1
            
            # if the number of components to be fit is exceeded, fix line fluxes to 0
            if i+1 > ncomp[line]:
                parinfo[ifoff]['value'] = 0.
                parinfo[iwoff]['value'] = 0.
                parinfo[isoff]['value'] = 0.
                parinfo[ifoff]['fixed'] = b'1'
                parinfo[iwoff]['fixed'] = b'1'
                parinfo[isoff]['fixed'] = b'1'
            else:
                # initial values
                parinfo[ifoff]['value'] = initflux[line][i]
                parinfo[iwoff]['value'] = linelistz[line][i]
                parinfo[isoff]['value'] = initsig[line][i]
                # limits
                parinfo[ifoff]['limited'][0] = b'1'
                parinfo[ifoff]['limited'][0]  = 0.
                parinfo[iwoff]['limited'] = [b'1',b'1']
                parinfo[iwoff]['limited'][0] = linelistz[line][i]*0.997
                parinfo[iwoff]['limited'][1] = linelistz[line][i]*1.003
                parinfo[isoff]['limited'] = [b'1',b'1']
                parinfo[isoff]['limited'] = siglim
                # ties
                if line == linetie[line]:
                    parinfo[iwoff]['tied'] = ''
                    parinfo[isoff]['tied'] = ''
                else:
                    indtie = np.where(lines_arr == linetie[line])[0]
                    parinfo[iwoff]['tied'] = '{0:0.6e}{1:1}{2:0.6e}{3:1}{4:1}{5:1}'.format(linelist[line],'/',linelist[linetie[line]],'* P[',woff+indtie*3,']')
                    parinfo[isoff]['tied'] = '{0:1}{1:1}{2:1}'.format('P[',soff+indtie*3,']') 
                    parinfo[iwoff]['sigmawave_tie'] = linetie[line]
                    parinfo[isoff]['sigmawave_tie'] = linetie[line]
                # fixed/free
                if sigfix:
                    if 'line' in sigfix:
                        if sigfix['line'][i] != 0:
                            parinfo[isoff]['fixed']=b'1'
                            parinfo[isoff]['value']=sigfix['line'][i] 
            iline+=1

# Check parinit initial values vs. limits
    badpar = [i for i, x in enumerate(parinfo) 
               if (x['limited'][0] == b'1' and x['value'] < x['limits'][0]) 
               or (x['limited'][1] == b'1' and x['value'] > x['limits'][1])]
    ct = len(badpar)
    if ct > 0 :
        print('IFSF_OSIRIS: Initial values are outside limits.')
        print('Offending parameters:')
        print('{0:20}{1:20}{2:5}{3:15}{4:15}{5:15}'.format('Quantity','Line','Comp','Value','Lower limit','Upper limit'))
        for i in range(0,ct):
            j = badpar[i]
            print('{0:20}{1:20}{2:5}{3:15.6e}{4:15.6e}{5:15.6e}'.format(parinfo[j]['parname'],parinfo[j]['line'],parinfo[j]['comp'],
                                                                        parinfo[j]['value'],parinfo[j]['limits'][0],parinfo[j]['limits'][1]))
        return 0
    else:
        return parinfo
