# -*- coding: utf-8 -*-
"""
; docformat = 'rst'
;
;+
;
; Initialize parameters for fitting. Specific to GMOS instrument.
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
;    blrcomp: in, optional, type=dblarr(N_BLR)
;      For each velocity component to model as a broad line region
;      (BLR), put the index (unity-offset) of that component into this
;      scalar (or array if more than one component) and all fluxes but
;      Balmer line fluxes will be zeroed.
;    blrlines: in, optional, type=strarr(N_lines)
;      List of lines to fit with BLR component.
;    lratfix: in, optional, type=hash(lineratios,ncomp)
;      For each line ratio that should be fixed, input an array with each 
;      element set to either the BAD value (do not fix that component) or the 
;      value to which the line ratio will be fixed for that component.
;    siglim: in, optional, type=dblarr(2)
;      Lower and upper sigma limits in km/s.
;    sigfix: in, optional, type=hash(lines\,maxncomp)
;      Fix sigma at this value, for particular lines/components.
;    specres: in, optional, type=double, def=0.64d
;      Estimated spectral resolution in wavelength units (sigma).
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
;      2009jun01, DSNR, created
;      2009jun08, DSNR, added multiple components
;      2010may27, DSNR, re-written to fit in observed frame
;      2013sep13, DSNR, re-written to allow more than one common redshift
;      2013dec12, DSNR, documented, renamed, added license and copyright 
;      2014jan13, DSNR, updated to use hashes, and to add parname, line, and 
;                       comp tags into output parinfo structure
;      2014apr10, DSNR, added if statements to remove IEEE exceptions
;      2014apr17, DSNR, adjusted upper limits for Ha/Hb and [NII]/Ha
;      2014apr23, DSNR, added SIGFIX keyword
;      2014jun05, DSNR, added LRATFIX keyword
;      2015jan20, DSNR, added check for both lines in line ratio constraints
;      2015may12, DSNR, added option to model BLR for a certain velocity 
;                       component by setting all lines to 0 except those
;                       specified; see BLRCOMP and BLRLINES
;      2016feb04, DSNR, added [OII] line ratio
;      2016sep26, DSNR, added SPECRES as a keyword and a parameter
;      2016oct05, DSNR, added SIGMAWAVE_TIE and FLUX_TIE to PARINFO
;      2016oct08, DSNR, turned off Ha/Hb limits b/c of issues with estimating
;                       Hbeta error in a noisy spectrum when pegged at lower
;                       limit
;      2018feb22, DSNR, added NeIII line ratio
;      2020may22, YI, translated from IDL to Python 3
;    
; :Copyright:
;    Copyright (C) 2013--2018 David S. N. Rupke
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
        
def ifsf_gmos(linelist, linelistz,linetie,initflux,initsig,maxncomp,ncomp,
               lratfix=False,siglim=False,sigfix=[],blrcomp=False,blrlines=False,specres=False) :
    bad = 1e99
    c=299792.458
    
# Estimated spectral resolution for B600 grating based on measurements.
# Website says R = 1688 at 4610 A for 0.5" slit, with IFU 0.31" eff. slit.
# This gives 1.69 A FWHM. I measure sometimes closer to 1.5-1.6.
# Sigma is then in the range 0.64 -- 0.72. Use the latter for flexibility.

    if blrlines != True:
        blrlines = ['Halpha','Hbeta','Hgamma','Hdelta','Hepsilon','H8','H9','H10','H11']
    if specres != True:
        specres = 0.64
    # A reasonable lower limit of 5d for physicality ... Assume line is resolved.
    if siglim != True:
        siglim = [5.,2000.]
    if blrcomp != True:
        blrcomp = -1
        
    print(blrlines)
    # Number of emission lines to fit
    nline = len(linelist)
    lines_arr = linelist #these 2 steps highly depend on the structure of the input
    # Number of initial parameters before Gaussian parameters begin
    lratlim = 4 
    ppoff0 = 3
    ppoff = ppoff0 + maxncomp*lratlim
    
    pardict = {'value':0.,'fixed':0.,'limited':[r'0',r'0'],'tied':'',
               'limits':[0.,0.],'step':0.,'mpprint':0.,'mpside':2,
               'parname':'','line':'','comp':0.,'sigmawave_tie':'','flux_tie':''}
    parinfo = list(np.repeat(pardict,ppoff+maxncomp*(nline*3)))
    
    # Number of initial parameters before Gaussian parameters begin
    parinfo[0]['value'] = ppoff
    parinfo[0]['fixed'] = r'1'
    parinfo[0]['parname'] = 'No. of non-Gaussian parameters'
    
    # Maximum number of velocity components
    parinfo[1]['value'] = maxncomp
    parinfo[1]['fixed'] = r'1'
    parinfo[1]['parname'] = 'Maximum no. of velocity components'
    
    # Spectral resolution
    parinfo[2]['value'] = specres
    parinfo[2]['fixed'] = r'1'
    parinfo[2]['parname'] = 'Spectral resolution in wavelength space [sigma]'
    
    if lratfix != True:
        lratfix = {}
    
# [SII] ratio
    ilratlim = 0
    lratlab = '[SII]6716/6731'
    if '[SII]6716' in ncomp:
        tmp_ncomp = ncomp['[SII]6716']
    else:
        tmp_ncomp = 0
    if tmp_ncomp > 0 :
        ip1 = ppoff0 + ilratlim*maxncomp
        ip2 = ip1+tmp_ncomp-1
        fa = initflux['[SII]6716'][0:tmp_ncomp-1]
        fb = initflux['[SII]6731'][0:tmp_ncomp-1]
        frat = np.zeros(tmp_ncomp)+1. # default if initial s2b flux = 0
        inz  = np.where(fb > 0)[0]
        ctnz = len(inz)
        if ctnz > 0 :
            frat[inz] = fa[inz]/fb[inz]
        parinfo[ip1:ip2+1]['value'] = frat      
        parinfo[ip1:ip2+1]['limited'] = np.resize([r'1',r'1'],[tmp_ncomp,2])
        parinfo[ip1:ip2+1]['limits']  = np.resize([0.44,1.43],[tmp_ncomp,2])
        parinfo[ip1:ip2+1]['parname'] = '[SII]6716/6731 line ratio'
        parinfo[ip1:ip2+1]['comp'] = np.arange(0,tmp_ncomp)+1
        # Check to see if line ratio is fixed
        ilratfix = [i for i, x in enumerate(lratfix) if x == lratlab]
        ctlratfix = len(ilratfix)
        for i in range(0,tmp_ncomp) :
        # If line ratio is fixed, then fix it
            lratfixed = r'0'
            if ctlratfix > 0 :
                if lratfix[lratlab][i] != bad :
                    parinfo[ip1+i]['value'] = lratfix[lratlab][i]
                    parinfo[ip1+i]['fixed'] = r'1'
                    parinfo[ip1+i]['limited'] = [r'0',r'0']
                    lratfixed = r'1'
            if not lratfixed:
                # case of pegging at or exceeding upper limit
                if parinfo[ip1+i]['value'] >= parinfo[ip1+i]['limits'][1] :
                    parinfo[ip1+i]['value'] = parinfo[ip1+i]['limits'][1] - (parinfo[ip1+i]['limits'][1] - parinfo[ip1+i]['limits'][0])*0.1
                # case of pegging at or dipping below lower limit
                if parinfo[ip1+i]['value'] <= parinfo[ip1+i]['limits'][0] :
                    parinfo[ip1+i]['value'] = parinfo[ip1+i]['limits'][0] + (parinfo[ip1+i]['limits'][1] - parinfo[ip1+i]['limits'][0])*0.1                
        
# [NI] ratio
# See Ferland+12 for collisional case, Bautista99 for other cases. Upper limit 
# was originally 3, but found that it would peg at that and then the error for 
# [NI]5200 would be artificially large, and it would be removed from the fit. 
# Can fix to 1.5 (low density collisional limit, applicable to n <~ 10^3 
# cm^-3; Ferland+12 Appendix A.3) to solve artificially large errors in [NI]5200. 
    ilratlim = 1    
    lratlab = '[NI]5200/5198'
    if '[NI]5198'  in ncomp :
        tmp_ncomp = ncomp['[NI]5198']
    else:
        tmp_ncomp = 0
    if tmp_ncomp > 0:
        ip1 = ppoff0 + ilratlim*maxncomp
        ip2 = ip1+tmp_ncomp-1
        fa = initflux['[NI]5200'][0:tmp_ncomp-1]
        fb = initflux['[NI]5198'][0:tmp_ncomp-1]
        frat = np.zeros(tmp_ncomp)+2.
        inz  = np.where(fb > 0)[0]
        ctnz = len(inz)
        if ctnz > 0 :
            frat[inz] = fa[inz]/fb[inz]
        parinfo[ip1:ip2+1]['value'] = frat
        parinfo[ip1:ip2+1]['limited'] = np.resize([r'1',r'1'],[tmp_ncomp,2])
        parinfo[ip1:ip2+1]['limits']  = np.resize([0.6,4.],[tmp_ncomp,2])
        parinfo[ip1:ip2+1]['parname'] = '[NI]5200/5198 line ratio'
        parinfo[ip1:ip2+1]['comp'] = np.arange(0,tmp_ncomp)+1
        # Check to see if line ratio is fixed
        ilratfix = [i for i, x in enumerate(lratfix) if x == lratlab]
        ctlratfix = len(ilratfix)
        for i in range(0,tmp_ncomp) :
        # If line ratio is fixed, then fix it
            lratfixed = r'0'
            if ctlratfix > 0 :
                if lratfix[lratlab][i] != bad :
                    parinfo[ip1+i]['value'] = lratfix[lratlab][i]
                    parinfo[ip1+i]['fixed'] = r'1'
                    parinfo[ip1+i]['limited'] = [r'0',r'0']
                    lratfixed = r'1'
                if not lratfixed:
                    # case of pegging at or exceeding upper limit
                    if parinfo[ip1+i]['value'] >= parinfo[ip1+i]['limits'][1] :
                        parinfo[ip1+i]['value'] = parinfo[ip1+i]['limits'][1] - (parinfo[ip1+i]['limits'][1] - parinfo[ip1+i]['limits'][0])*0.1
                        # case of pegging at or dipping below lower limit
                    if parinfo[ip1+i]['value'] <= parinfo[ip1+i]['limits'][0] :
                        parinfo[ip1+i]['value'] = parinfo[ip1+i]['limits'][0] + (parinfo[ip1+i]['limits'][1] - parinfo[ip1+i]['limits'][0])*0.1                
                    
# [NII]/Ha ratio
    ilratlim = 2
    lratlab = '[NII]6583/Ha'
    if ('Halpha' in ncomp) and ('[NII]6583'  in ncomp) :
        tmp_ncomp = ncomp['Halpha']
    else:
        tmp_ncomp = 0
    if tmp_ncomp > 0:
        ip1 = ppoff0 + ilratlim*maxncomp
        ip2 = ip1+tmp_ncomp-1
        fa = initflux['[NII]6583'][0:tmp_ncomp-1]
        fb = initflux['Halpha'][0:tmp_ncomp-1]
        frat = np.zeros(tmp_ncomp)+1.
        inz  = np.where(fb > 0)[0]
        ctnz = len(inz)
        if ctnz > 0 :
            frat[inz] = fa[inz]/fb[inz]
        parinfo[ip1:ip2+1]['value'] = frat
        parinfo[ip1:ip2+1]['limited'] = np.resize([r'1',r'1'],[tmp_ncomp,2])
        # This upper limit appears to be the maximum seen in Kewley+06 or 
        # Rich+14 ("Composite Spectra in ..."). The lower limit is appropriate 
        # for ULIRGs.
        parinfo[ip1:ip2+1]['limits']  = np.resize([0.1,4.],[tmp_ncomp,2])
        parinfo[ip1:ip2+1]['parname'] = '[NII]6583/Halpha line ratio'
        parinfo[ip1:ip2+1]['comp'] = np.arange(0,tmp_ncomp)+1
        # Check to see if line ratio is fixed
        ilratfix = [i for i, x in enumerate(lratfix) if x == lratlab]
        ctlratfix = len(ilratfix)
        for i in range(0,tmp_ncomp) :
        # If line ratio is fixed, then fix it
            lratfixed = r'0'
            if ctlratfix > 0 :
                if lratfix[lratlab][i] != bad :
                    parinfo[ip1+i]['value'] = lratfix[lratlab][i]
                    parinfo[ip1+i]['fixed'] = r'1'
                    parinfo[ip1+i]['limited'] = [r'0',r'0']
                    lratfixed = r'1'
                if not lratfixed:
                    # case of pegging at or exceeding upper limit
                    if parinfo[ip1+i]['value'] >= parinfo[ip1+i]['limits'][1] :
                        parinfo[ip1+i]['value'] = parinfo[ip1+i]['limits'][1] - (parinfo[ip1+i]['limits'][1] - parinfo[ip1+i]['limits'][0])*0.1
                   # case of pegging at or dipping below lower limit
                    if parinfo[ip1+i]['value'] <= parinfo[ip1+i]['limits'][0] :
                        parinfo[ip1+i]['value'] = parinfo[ip1+i]['limits'][0] + (parinfo[ip1+i]['limits'][1] - parinfo[ip1+i]['limits'][0])*0.1                
                    
# Ha/Hb ratio --> has been commented out in the original IDL code
# ilratlim = 3
#  lratlab = 'Ha/Hb'
#   if ncomp->haskey('Halpha') AND ncomp->haskey('Hbeta') $
#     then tmp_ncomp = ncomp['Halpha'] $
#   else tmp_ncomp = 0
#   if tmp_ncomp gt 0 then begin
#      ip1 = ppoff0 + ilratlim*maxncomp
#      ip2 = ip1 + tmp_ncomp - 1
#      fa = initflux['Halpha',0:tmp_ncomp-1]
#      fb = initflux['Hbeta',0:tmp_ncomp-1]
#      frat = dblarr(tmp_ncomp)+3d ; default if initial hb flux = 0
#      inz = where(fb gt 0,ctnz)
#      if ctnz gt 0 then frat[inz] = fa[inz]/fb[inz]
#      parinfo[ip1:ip2+1].value = frat
#      parinfo[ip1:ip2+1].limited = rebin([1B,0B],2,tmp_ncomp)
# ;    Upper limit of 50 corresponds to E(B-V) = 2.89 using CCM
#      parinfo[ip1:ip2+1].limits  = rebin([2.86d,0d],2,tmp_ncomp)
#      parinfo[ip1:ip2+1].parname = 'Halpha/Hbeta line ratio'
#      parinfo[ip1:ip2+1].comp = indgen(tmp_ncomp)+1
# ;    Check to see if line ratio is fixed
#      ilratfix = where(lratfix.keys() eq lratlab,ctlratfix)
#      for i=0,tmp_ncomp-1 do begin
# ;       If line ratio is fixed, then fix it
#         lratfixed = 0b
#         if ctlratfix gt 0 then begin
#            if lratfix[lratlab,i] ne bad then begin
#               parinfo[ip1+i].value = lratfix[lratlab,i]
#               parinfo[ip1+i].fixed = 1b
#               parinfo[ip1+i].limited = [0b,0b]
#               lratfixed = 1b
#            endif
#         endif
#         if ~ lratfixed then begin
# ;;          case of pegging at or exceeding upper limit
# ;           if parinfo[ip1+i].value ge parinfo[ip1+i].limits[1] then $
# ;              parinfo[ip1+i].value = parinfo[ip1+i].limits[1] - $
# ;                                     (parinfo[ip1+i].limits[1] - $
# ;                                      parinfo[ip1+i].limits[0])*0.1
# ;          case of pegging at or dipping below lower limit
#            if parinfo[ip1+i].value le parinfo[ip1+i].limits[0] then $
#               parinfo[ip1+i].value = parinfo[ip1+i].limits[0] + $
#                                      parinfo[ip1+i].limits[0]*0.1d
# ;                                     (parinfo[ip1+i].limits[1] - $
# ;                                      parinfo[ip1+i].limits[0])*0.1
#         endif
#      endfor
#   endif

# [OII] ratio
# Limits from Pradhan et al. 2006, MNRAS, 366, L6
# 28aug2016, DSNR, changed limits to be more physically reasonable for AGN physics
# 28mar2019, DSNR, changed back to defaults
    ilratlim = 3
    lratlab = '[OII]3729/3726'
    if '[OII]3726' in ncomp :
        tmp_ncomp = ncomp['[OII]3726']
    else:
        tmp_ncomp = 0
    if tmp_ncomp > 0:
        ip1 = ppoff0 + ilratlim*maxncomp
        ip2 = ip1+tmp_ncomp-1
        fa = initflux['[OII]3726'][0:tmp_ncomp-1]
        fb = initflux['[OII]3729'][0:tmp_ncomp-1]
        frat = np.zeros(tmp_ncomp)+1.
        inz  = np.where(fb > 0)[0]
        ctnz = len(inz)
        if ctnz > 0 :
            frat[inz] = fa[inz]/fb[inz]
        parinfo[ip1:ip2+1]['value'] = frat
        parinfo[ip1:ip2+1]['limited'] = np.resize([r'1',r'1'],[tmp_ncomp,2])
        parinfo[ip1:ip2+1]['limits']  = np.resize([0.35,1.5],[tmp_ncomp,2])
        parinfo[ip1:ip2+1]['parname'] = '[OII]3729/3726 line ratio'
        parinfo[ip1:ip2+1]['comp'] = np.arange(0,tmp_ncomp)+1
    # Check to see if line ratio is fixed
        ilratfix = [i for i, x in enumerate(lratfix) if x == lratlab]
        ctlratfix = len(ilratfix)
        for i in range(0,tmp_ncomp) :
        # If line ratio is fixed, then fix it
            lratfixed = r'0'
            if ctlratfix > 0 :
                if lratfix[lratlab][i] != bad :
                    parinfo[ip1+i]['value'] = lratfix[lratlab][i]
                    parinfo[ip1+i]['fixed'] = r'1'
                    parinfo[ip1+i]['limited'] = [r'0',r'0']
                    lratfixed = r'1'
                if not lratfixed:
                    # case of pegging at or exceeding upper limit
                    if parinfo[ip1+i]['value'] >= parinfo[ip1+i]['limits'][1] :
                        parinfo[ip1+i]['value'] = parinfo[ip1+i]['limits'][1] - (parinfo[ip1+i]['limits'][1] - parinfo[ip1+i]['limits'][0])*0.1
                        # case of pegging at or dipping below lower limit
                    if parinfo[ip1+i]['value'] <= parinfo[ip1+i]['limits'][0] :
                        parinfo[ip1+i]['value'] = parinfo[ip1+i]['limits'][0] + (parinfo[ip1+i]['limits'][1] - parinfo[ip1+i]['limits'][0])*0.1                
                    

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
            if ((i+1 > ncomp[line]) or (np.where(i+1 == blrcomp)[0][0] >= 0 and np.where(line == blrlines)[0][0] == -1)):
                parinfo[ifoff]['value'] = 0.
                parinfo[iwoff]['value'] = 0.
                parinfo[isoff]['value'] = 0.
                parinfo[ifoff]['fixed'] = r'1'
                parinfo[iwoff]['fixed'] = r'1'
                parinfo[isoff]['fixed'] = r'1'
            else:
                # initial values
                parinfo[ifoff]['value'] = initflux[line,i]
                parinfo[iwoff]['value'] = linelistz[line,i]
                parinfo[isoff]['value'] = initsig[line,i]
                # limits
                parinfo[ifoff]['limited'][0] = r'1'
                parinfo[ifoff]['limited'][0]  = 0.
                parinfo[iwoff]['limited'] = [r'1',r'1']
                parinfo[iwoff]['limited'][0] = linelistz[line,i]*0.997
                parinfo[iwoff]['limited'][1] = linelistz[line,i]*1.003
                parinfo[isoff]['limited'] = [r'1',r'1']
                parinfo[isoff]['limited'] = siglim
                # ties
                if line == linetie[line]:
                    parinfo[iwoff]['tied'] = ''
                    parinfo[isoff]['tied'] = ''
                else:
                    indtie = np.where(lines_arr == linetie[line])[0]
                    parinfo[iwoff]['tied'] = ''
                # fixed/free
                if sigfix == True:
                    if 'line' in sigfix:
                        if sigfix['line'][1] != 0:
                            parinfo[isoff]['fixed']=r'1'
                            parinfo[isoff]['value']=sigfix['line'][i] 
            iline+=1

        # the if statement here prevents MPFIT_TIE from issuing an IEEE exception,
        # since if we're not fitting [SII] then the ratio is set to 0 at the 
        # beginning of this routine
        if '[SII]6716' in ncomp:
            if ncomp['[SII]6716'] > 0:
                ilratlim = 0
        ilratlim = 1
        linea = np.where(lines_arr == '[SII]6716')[0]
        cta = len(linea)
        lineb = np.where(lines_arr == '[SII]6731')[0]
        ctb = len(lineb)
        if cta > 0 and ctb > 0 :
            parinfo[foff+lineb*3]['tied'] = ''
            parinfo[foff+lineb*3]['flux_tie'] = '[SII]6716'
        
        ilratlim = 1
        linea = np.where(lines_arr == '[NI]5198')[0]
        cta = len(linea)
        lineb = np.where(lines_arr == '[NI]5200')[0]
        ctb = len(lineb)
        if cta > 0 and ctb > 0 :
            parinfo[foff+lineb*3]['tied'] = ''
            parinfo[foff+lineb*3]['flux_tie'] = '[NI]5200'
        
        ilratlim = 2
        linea = np.where(lines_arr == 'Halpha')[0]
        cta = len(linea)
        lineb = np.where(lines_arr == '[NII]6583')[0]
        ctb = len(lineb)
        if cta > 0 and ctb > 0 :
            parinfo[foff+lineb*3]['tied'] = ''
            parinfo[foff+lineb*3]['flux_tie'] = 'Halpha'
        
        ilratlim = 3
        linea = np.where(lines_arr == '[OII]3726')[0]
        cta = len(linea)
        lineb = np.where(lines_arr == '[OII]3729')[0]
        ctb = len(lineb)
        if cta > 0 and ctb > 0 :
            parinfo[foff+linea*3]['tied'] = ''
            parinfo[foff+linea*3]['flux_tie'] = '[OII]3729'
            
            
        linea = np.where(lines_arr == '[OIII]4959')[0]
        cta = len(linea)
        lineb = np.where(lines_arr == '[OIII]5007')[0]
        ctb = len(lineb)
        if cta > 0 and ctb > 0 :
            parinfo[foff+linea*3]['tied'] = ''
            parinfo[foff+linea*3]['flux_tie'] = '[OIII]5007'
            # Make sure initial value is correct
            parinfo[foff+linea*3]['value'] = parinfo[foff+lineb*3]['value']/3.0
        
        linea = np.where(lines_arr == '[OI]6300')[0]
        cta = len(linea)
        lineb = np.where(lines_arr == '[OI]6364')[0]
        ctb = len(lineb)
        if cta > 0 and ctb > 0 :
            parinfo[foff+lineb*3]['tied'] = ''
            parinfo[foff+lineb*3]['flux_tie'] = '[OI]6300'
            # Make sure initial value is correct
            parinfo[foff+lineb*3]['value'] = parinfo[foff+linea*3]['value']/3.0
        
        linea = np.where(lines_arr == '[NII]6548')[0]
        cta = len(linea)
        lineb = np.where(lines_arr == '[NII]6583')[0]
        ctb = len(lineb)
        if cta > 0 and ctb > 0 :
            parinfo[foff+linea*3]['tied'] = ''
            parinfo[foff+linea*3]['flux_tie'] = '[NII]6583'
            # Make sure initial value is correct
            parinfo[foff+linea*3]['value'] = parinfo[foff+lineb*3]['value']/3.0
            
        linea = np.where(lines_arr == '[NeIII]3967')[0]
        cta = len(linea)
        lineb = np.where(lines_arr == '[NeIII]3869')[0]
        ctb = len(lineb)
        if cta > 0 and ctb > 0 :
            parinfo[foff+linea*3]['tied'] = ''
            parinfo[foff+linea*3]['flux_tie'] = '[NeIII]3869'
            # Make sure initial value is correct
            parinfo[foff+linea*3]['value'] = parinfo[foff+lineb*3]['value']/3.0
        
        
# Check parinit initial values vs. limits
    # badpar = where((parinfo.limited[0] AND $
    #                   parinfo.value lt parinfo.limits[0]) OR $
    #                  (parinfo.limited[1] AND $
    #                   parinfo.value gt parinfo.limits[1]),ct)
    # badpar = np.where(())
    badpar = []
    ct = 0
    if ct > 0 :
        print('Quantity','Line','Comp','Value','Lower limit','Upper limit')#,format='(2A20,A5,3A15)')
        for i in range(0,ct):
            j = badpar[i]
            print(parinfo[j]['parname'],parinfo[j]['line'],parinfo[j]['comp'],
                  parinfo[j]['value'],parinfo[j][0],parinfo[j]['limits'][1])#,format='(2A20,I5,3E15.6)')
        print('Initial values are outside limits.')
    else:
        return parinfo

