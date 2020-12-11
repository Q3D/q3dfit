# docformat = 'rst'
#
#+
#
# Convert output of MPFIT, with best-fit line parameters in a single
# array, into a structure with separate arrays for different line
# parameters. Compute total line fluxes from the best-fit line
# parameters.
#
# :Categories:
#    IFSFIT
#
# :Returns:
#    A structure with separate hashes for different line parameters. The hashes 
#    are indexed by line, and each value is an array over components. 
#    Tags: flux, fluxerr, fluxpk, fluxpkerr, nolines, wave, and sigma.
#
# :Params:
#    linelist: in, required, type=hash(lines)
#      List of emission line rest-frame wavelengths.
#    param: in, required, type=dblarr(N)
#      Best-fit parameter array output by MPFIT.
#    perror: in, optional, type=dblarr(N)
#      Errors in best fit parameters, output by MPFIT.
#    parinfo: in, required, type=structure
#      Structure input into MPFIT. Each tag has N values, one per parameter. 
#      Used to sort param and perror arrays.
#
# :Keywords:
#    waveran: in, optional, type=dblarr(2)
#      Set to upper and lower limits to return line parameters only
#      for lines within the given wavelength range. Lines outside this
#      range have fluxes set to 0.
#    tflux: out, optional, type=structure
#      A structure with separate hashes for total flux and error. The hashes 
#      are indexed by line. Tags: tflux, tfluxerr
# 
# :Author:
#    David S. N. Rupke::
#      Rhodes College
#      Department of Physics
#      2000 N. Parkway
#      Memphis, TN 38104
#      drupke@gmail.com
#
# :History:
#    ChangeHistory::
#      2009may26, DSNR, created
#      2009jun07, DSNR, added error propagation and rewrote
#      2013nov01, DSNR, added documentation
#      2013nov25, DSNR, renamed, added copyright and license
#      2014jan13, DSNR, re-written to use hashes rather than arrays
#      2014feb26, DSNR, replaced ordered hashes with hashes
#      2015sep20, DSNR, compute total line flux and error
#      2016sep26, DSNR, account for new treatment of spectral resolution#
#                       fix flux errors for tied lines
#      2016oct05, DSNR, include wavelength and sigma errors
#      2016oct08, DSNR, turned off Ha/Hb limits b/c of issues with estimating
#                       Hbeta error in a noisy spectrum when pegged at lower
#                       limit
#      2016oct10, DSNR, added option to combine doublets# changed calculation
#                       of error when line ratio pegged
#      2020may27, DW,   translated to Python
#    
# :Copyright:
#    Copyright (C) 2013--2016 David S. N. Rupke
#
#    This program is free software: you can redistribute it and/or
#    modify it under the terms of the GNU General Public License as
#    published by the Free Software Foundation, either version 3 of
#    the License or any later version.
#
#    This program is distributed in the hope that it will be useful,
#    but WITHOUT ANY WARRANTY# without even the implied warranty of
#    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU
#    General Public License for more details.
#
#    You should have received a copy of the GNU General Public License
#    along with this program.  If not, see
#    http://www.gnu.org/licenses/.
#
#-
import numpy as np
import math
import pdb
from astropy.table import Table
from scipy import constants
from q3dfit.common.gaussflux import gaussflux


def sepfitpars(linelist, param, perror, parinfo, waveran = None, tflux = None, 
               doublets = None):

    # DW: param, perror and parinfo are lists of dictionaries which are not easy to handle
    # I therefore re-structure them into a single dictionary each
    
    parinfo_new={}
    for k,v in [(key,d[key]) for d in parinfo for key in d]:
        if k not in parinfo_new: parinfo_new[k]=[v]
        else: parinfo_new[k].append(v)
    
    parinfo_new = Table(parinfo_new)
    
    param = np.array(param)
    perror = np.array(perror)
	
	# Return 0 if no lines were fit
    if len(param) == 1: ### DW: this needs to be double checked!!!!###:
        outstr = {'nolines': np.array([0])}
        return outstr

    else: 
        
        maxncomp = param[1] ### DW: this needs to be double checked!!!!###
            
        flux          = Table(np.full(linelist['name'].T.data.shape, np.nan), names = linelist['name'])
        fluxerr       = Table(np.full(linelist['name'].T.data.shape, np.nan), names = linelist['name'])
        fluxpk        = Table(np.full(linelist['name'].T.data.shape, np.nan), names = linelist['name'])
        fluxpkerr     = Table(np.full(linelist['name'].T.data.shape, np.nan), names = linelist['name'])
        fluxpk_obs    = Table(np.full(linelist['name'].T.data.shape, np.nan), names = linelist['name'])
        fluxpkerr_obs = Table(np.full(linelist['name'].T.data.shape, np.nan), names = linelist['name'])
        sigma         = Table(np.full(linelist['name'].T.data.shape, np.nan), names = linelist['name'])
        sigmaerr      = Table(np.full(linelist['name'].T.data.shape, np.nan), names = linelist['name'])
        sigma_obs     = Table(np.full(linelist['name'].T.data.shape, np.nan), names = linelist['name'])
        sigmaerr_obs  = Table(np.full(linelist['name'].T.data.shape, np.nan), names = linelist['name'])
        wave          = Table(np.full(linelist['name'].T.data.shape, np.nan), names = linelist['name'])
        waveerr       = Table(np.full(linelist['name'].T.data.shape, np.nan), names = linelist['name'])
    
        if not (tflux is None):
            tf =  Table(np.full(linelist['name'].T.data.shape, np.nan), names = linelist['name'])
            tfe = Table(np.full(linelist['name'].T.data.shape, np.nan), names = linelist['name'])
      
    
        in2ha = np.where(parinfo_new['parname'] == '[NII]/Halpha line ratio')
        ctn2ha = np.count_nonzero(parinfo_new['parname'] == '[NII]/Halpha line ratio')
    
        in1rat = np.where(parinfo_new['parname'] == '[NI]5200/5198 line ratio')
        ctn1rat = np.count_nonzero(parinfo_new['parname'] == '[NI]5200/5198 line ratio')

        is2rat = np.where(parinfo_new['parname'] == '[SII]6716/6731 line ratio')
        cts2rat = np.count_nonzero(parinfo_new['parname'] == '[SII]6716/6731 line ratio')

        #   ihahb = np.where(parinfo_new['parname'] == 'Halpha/Hbeta line ratio')
        #   cthahb = np.count_nonzero(parinfo_new['parname'] == 'Halpha/Hbeta line ratio')

        io2rat = np.where(parinfo_new['parname'] == '[OII]3729/3726 line ratio')
        cto2rat = np.count_nonzero(parinfo_new['parname'] == '[OII]3729/3726 line ratio')
    
        #   Populate Tables

        for line in linelist['name']:
        
            #indices
            iline =   np.where(parinfo_new['line'] == line)
            ifluxpk = np.intersect1d(iline, np.where(parinfo_new['parname'] == 'flux_peak'))
            isigma =  np.intersect1d(iline, np.where(parinfo_new['parname'] == 'sigma'))
            iwave =   np.intersect1d(iline, np.where(parinfo_new['parname'] == 'wavelength'))
            
            # indices for errors, which is more complicated if error comes from line to which this one is tied
            sigmawave_tie = parinfo_new['sigmawave_tie'][isigma[0]] # line tied to
            if sigmawave_tie == '':
                isigmaerr = isigma
                iwaveerr = iwave
            else:
                ilinetie = np.where(parinfo_new['line'] == sigmawave_tie)
                isigmaerr = np.intersect1d(ilinetie, np.where(parinfo_new['parname'] == 'sigma'))
                iwaveerr = np.intersect1d(ilinetie, np.where(parinfo_new['parname'] == 'wavelength'))
              
            wave[line] = param[iwave]
            sigma[line] = param[isigma]
            fluxpk[line] = param[ifluxpk]
            sigma_obs[line] = param[isigma]
            fluxpk_obs[line] = param[ifluxpk]
            # This bit of jujitsu ensures waveerr doesn't go NaN ...
            p1tmp = perror[iwaveerr]
            p2tmp = param[iwaveerr]
            p3tmp = param[iwave]
            inz = np.where((p1tmp != 0.) and (p2tmp != 0.) and (p3tmp != 0.))
            ctnz = np.count_nonzero((p1tmp != 0.) and (p2tmp != 0.) and (p3tmp != 0.))
            waveerrtmp = np.zeros(p1tmp.shape)
        
            if ctnz > 0:
                waveerrtmp[inz] = p1tmp[inz]/p2tmp[inz]*p3tmp[inz]
        
            waveerr[line] = waveerrtmp
            sigmaerr[line] = perror[isigmaerr]
            sigmaerr_obs[line] = perror[isigmaerr]
            fluxpkerr[line] = perror[ifluxpk]
            fluxpkerr_obs[line] = perror[ifluxpk]

            # Because of the way these lines are tied to others (with a division!) they
            # can yield NaNs in components that aren't fit. Correct this.
            # if line eq '[SII]6731' OR line eq 'Hbeta' OR line eq '[NI]5189' then begin

            if (line == '[SII]6731') or (line == '[NI]5189'):
                inan = np.where(np.isfinite(fluxpk[line]) == False)
                ctnan = np.count_nonzero(np.isfinite(fluxpk[line]) == False)
                if ctnan > 0:
                    fluxpk[line][inan] = 0.
                    fluxpkerr[line,inan] = 0.
                    fluxpk_obs[line,inan] = 0.
                    fluxpkerr_obs[line,inan] = 0.
                                      
        for line in linelist['name']:
                                      
        # Fix flux errors associated with line ratios. E.g., [NII]/Halpha is a fitted
        # parameter and [NII]6583 is tied to it, so the formal error in [NII]6583
        # flux is 0. Add errors in Halpha and [NII]/Halpha in quadrature to get
        #  error in [NII]6583.

            if (line == "[NII]6583") and (ctn2ha > 0):
                fluxpkerr_obs[line][0:ctn2ha] = fluxpk_obs[line][0:ctn2ha] * \
                np.sqrt((perror[in2ha]/param[in2ha])**2. + \
                (fluxpkerr_obs['Halpha'][0:ctn2ha]/ fluxpk_obs['Halpha'][0:ctn2ha])**2.)  
                # In pegged case, set errors equal to each other
                ipegged = np.where((perror[in2ha] == 0.) and (param[in2ha] != 0.))
                ctpegged = np.count_nonzero((perror[in2ha] == 0.) and (param[in2ha] != 0.))
                if ctpegged > 0:
                    fluxpkerr_obs['[NII]6583'][ipegged] = \
                    fluxpkerr_obs['Halpha'][ipegged]
                fluxpkerr[line] = fluxpkerr_obs[line]
            
            if (line == '[SII]6731') and (cts2rat > 0):
                fluxpkerr_obs[line][0:cts2rat] = \
                fluxpk_obs[line][0:cts2rat] * \
                np.sqrt((perror[is2rat]/param[is2rat])**2. + \
                (fluxpkerr_obs['[SII]6716'][0:cts2rat] / \
                fluxpk_obs['[SII]6716'][0:cts2rat])**2.)
                # In pegged case, set errors equal to each other
                ipegged = np.where((perror[is2rat] == 0.) and (param[is2rat] != 0.))
                ctpegged = np.count_nonzero((perror[in2ha] == 0.) and (param[in2ha] != 0.))
                if ctpegged > 0:
                    fluxpkerr_obs['[SII]6731'][ipegged] = \
                    fluxpkerr_obs['[SII]6716'][ipegged]
                fluxpkerr[line] = fluxpkerr_obs[line]
            
            if (line == '[NI]5198') and (ctn1rat > 0):
                fluxpkerr_obs[line][0:ctn1rat] = \
                fluxpk_obs[line][0:ctn1rat] * \
                np.sqrt((perror[in1rat]/param[in1rat])**2. + \
                (fluxpkerr_obs['[NI]5200'][0:ctn1rat]/ \
                fluxpk_obs['[NI]5200'][0:ctn1rat])**2.)
                
                fluxpkerr[line] = fluxpkerr_obs[line]
                
                # In pegged case, set errors equal to each other
                ipegged = np.where((perror[in1rat] == 0.) and (param[in1rat] != 0.))
                ctpegged = np.count_nonzero((perror[in1rat] == 0.) and (param[in1rat] != 0.))
                if ctpegged > 0:
                    fluxpkerr_obs['[NI]5198'][ipegged] = fluxpkerr_obs['[NI]5200'][ipegged]
               
                fluxpkerr[line] = fluxpkerr_obs[line]                          
    
## DW: the following is a commented section in the IDL code:
#      if line eq 'Hbeta' AND cthahb gt 0 then begin
#         fluxpkerr_obs[line,0:cthahb-1] = $
#            fluxpk_obs[line,0:cthahb-1]*sqrt($
#            (perror[ihahb]/param[ihahb])^2d + $
#            (fluxpkerr_obs['Halpha',0:cthahb-1]/$
#            fluxpk_obs['Halpha',0:cthahb-1])^2d)
#         fluxpkerr[line] = fluxpkerr_obs[line]
##        If Halpha/Hbeta gets too high, MPFIT sees it as an "upper limit" and
##        sets perror = 0d. Then the errors seem too low, and it registers as a
##        detection. This bit of code corrects that.
#         ipeggedupper = $
#            where(param[ihahb] gt 1d1 AND perror[ihahb] eq 0d,ctpegged)
#         if ctpegged gt 0 then begin
#            fluxpk[line,ipeggedupper] = 0d
#            fluxpk_obs[line,ipeggedupper] = 0d
#         endif

            if (line == 'Hbeta') and (np.count_nonzero(linelist['name'] == 'Halpha') == 1):
                # If Halpha/Hbeta goes belowlower limit, then we re-calculate the errors
                # add discrepancy in quadrature to currently calculated error. Assume
                # error in fitting is in Hbeta and adjust accordingly.
                fha = fluxpk['Halpha']
                ihahb = np.where((fluxpk['Halpha'] > 0.) and (fluxpk['Hbeta'] > 0.))
                cthahb = np.count_nonzero((fluxpk['Halpha'] > 0.) and (fluxpk['Hbeta'] > 0.))
                if cthahb > 0.:
                    itoolow = np.where(fluxpk['Halpha'][ihahb]/fluxpk['Hbeta'] < 2.86)
                    cttoolow = np.count_nonzero(fluxpk['Halpha'][ihahb]/fluxpk['Hbeta'] < 2.86)
                    if cttoolow > 0:
                        fluxpkdiff = fluxpk[line][itoolow] - fluxpk['Halpha'][itoolow]/2.86
                        fluxpk[line][itoolow] -= fluxpkdiff                  
                        fluxpk_obs[line][itoolow] -= fluxpkdiff   
                        fluxpkerr[line][itoolow] = np.sqrt(fluxpkerr[line][itoolow]**2. + fluxpkdiff**2.)
                        fluxpkerr_obs[line][itoolow] = np.sqrt(fluxpkerr_obs[line][itoolow]**2. + fluxpkdiff**2.)                
                                          
            if (line == '[OII]3729') and (cto2rat > 0.):
                fluxpkerr_obs[line][0:cto2rat] = \
                fluxpk_obs[line][0:cto2rat]*np.sqrt( \
                (perror[io2rat]/param[io2rat])**2. + \
                (fluxpkerr_obs['[OII]3726'][0:cto2rat]/ \
                fluxpk_obs['[OII]3726'][0:cto2rat])**2.)                          
                # In pegged case, set errors equal to each other
                ipegged = np.where((perror[io2rat] == 0.) and (param[io2rat] != 0.))
                ctpegged = np.count_nonzero((perror[io2rat] == 0.) and (param[io2rat] != 0.))           
                if ctpegged > 0:
                    fluxpkerr_obs['[OII]3729'][ipegged] = fluxpkerr_obs['[OII]3726'][ipegged]
                fluxpkerr[line] = fluxpkerr_obs[line]

#     Add back in spectral resolution
#     Can't use sigma = 0 as criterion since the line could be fitted but unresolved.

                                          
            sigmatmp = sigma[line]/(constants.c/1.e3)*wave[line]                               
            inz = np.where(fluxpk[line] > 0)                               
            ctnz = np.count_nonzero(fluxpk[line] > 0)                                                         
                                          
            if ctnz > 0:
            # Make sure we're not adding something to 0 -- i.e. the component wasn't fit.
                sigmatmp[inz] = np.sqrt(sigmatmp[inz]**2. + param[2]**2.)
                sigma_obs[line][inz] = sigmatmp[inz]/wave[line][inz]*(constants.c/1.e3) # in km/s                         
            # error propagation for adding in quadrature
                sigmaerr_obs[line][inz] *= \
                sigma[line][inz]/(constants.c/1.e3)*wave[line][inz] / sigmatmp[inz]
            # Correct peak flux and error for deconvolution
                fluxpk[line][inz] *= sigma_obs[line][inz]/sigma[line][inz]
                fluxpkerr[line][inz] *= sigma_obs[line][inz]/sigma[line][inz]
                                          
# Compute total Gaussian flux
# sigma and error need to be in wavelength space
            gflux = gaussflux(fluxpk_obs[line],sigmatmp,normerr=fluxpkerr_obs[line],sigerr=sigmaerr_obs[line]/(constants.c/1.e3)*wave[line])                          
            flux[line] = gflux['flux']
            fluxerr[line] = gflux['flux_err']
                                       
# Set fluxes to 0 outside of wavelength range, or if NaNs or infinite errors
             
            if waveran:
                inoflux = np.where( (waveran[0] > wave[line]*(1 - 3.*sigma[line]/(constants.c/1.e3))) or \
                                    (waveran[1] < wave[line]*(1 + 3.*sigma[line]/(constants.c/1.e3))) or \
                                    (np.isfinite(fluxerr[line]) == False) or \
                                    (np.isfinite(fluxpkerr[line]) == False) )
                
                ct = np.count_nonzero( (waveran[0] > wave[line]*(1 - 3.*sigma[line]/(constants.c/1.e3))) or \
                                       (waveran[1] < wave[line]*(1 + 3.*sigma[line]/(constants.c/1.e3))) or \
                                       (np.isfinite(fluxerr[line]) == False) or \
                                       (np.isfinite(fluxpkerr[line]) == False) )
                                          
                                          
                if ct > 0:
                    flux[line][inoflux] = 0.
                    fluxerr[line][inoflux] = 0.
                    fluxpk[line][inoflux] = 0.
                    fluxpkerr[line][inoflux] = 0.
                    fluxpk_obs[line][inoflux] = 0.
                    fluxpkerr_obs[line][inoflux] = 0.
                                        
# Compute total fluxes summed over components
                                        
            igd = np.where(flux[line] > 0.)
            ctgd = np.count_nonzero(flux[line] > 0.)
            
            if not (tflux is None):     
                if ctgd > 0:
                    tf[line] = np.sum(flux[line][igd])
                    tfe[line] = np.sqrt(np.sum(fluxerr[line][igd]**2.))                   
                else:
                    tf[line] = 0.
                    tfe[line] = 0.
                                          
                                          
# Special doublet cases: combine fluxes from each line
        if not (doublets is None):
            ndoublets = doublets.shape[0]
            
            for i in np.arange(0,ndoublets):  
                if (np.count_nonzero(linelist['name'] == doublets[i,0]) == 1) \
                    and (np.count_nonzero(linelist['name'] == doublets[i,1]) == 1):
# new line label
                    dkey = doublets[i,0]+'+'+doublets[i,1]
# add fluxes
                    tf[dkey] = tf[doublets[i,0]]+tf[doublets[i,1]]
                    flux[dkey] = flux[doublets[i,0]]+flux[doublets[i,1]]
                    fluxpk[dkey] = fluxpk[doublets[i,0]]+fluxpk[doublets[i,1]]
                    fluxpk_obs[dkey] = fluxpk_obs[doublets[i,0]]+fluxpk_obs[doublets[i,1]]
# add flux errors in quadrature
                    tfe[dkey] = np.sqrt(tfe[doublets[i,0]]**2. + tfe[doublets[i,1]]**2.)
                    fluxerr[dkey] = np.sqrt(fluxerr[doublets[i,0]]**2. + fluxerr[doublets[i,1]]**2.)
                    fluxpkerr[dkey] = np.sqrt(fluxpkerr[doublets[i,0]]**2. + fluxpkerr[doublets[i,1]]**2.)
                    fluxpkerr_obs[dkey] = np.sqrt(fluxpkerr_obs[doublets[i,0]]**2. + fluxpkerr_obs[doublets[i,1]]**2.)
# average waves and sigmas and errors
                    wave[dkey] = (wave[doublets[i,0]]+wave[doublets[i,1]])/2.
                    waveerr[dkey] = (waveerr[doublets[i,0]]+waveerr[doublets[i,1]])/2.
                    sigma[dkey] = (sigma[doublets[i,0]]+sigma[doublets[i,1]])/2.
                    sigmaerr[dkey] = (sigmaerr[doublets[i,0]]+sigmaerr[doublets[i,1]])/2.
                    sigma_obs[dkey] = (sigma_obs[doublets[i,0]]+sigma_obs[doublets[i,1]])/2.
                    sigmaerr_obs[dkey] = (sigmaerr_obs[doublets[i,0]]+sigmaerr_obs[doublets[i,1]])/2.
        
        outstr = {'nolines':0,'flux':flux,'fluxerr':fluxerr,'fluxpk':fluxpk,'fluxpkerr':fluxpkerr,\
                  'wave':wave,'waveerr':waveerr,'sigma':sigma,'sigmaerr':sigmaerr,'sigma_obs':sigma_obs,\
                  'sigmaerr_obs':sigmaerr_obs, 'fluxpk_obs':fluxpk_obs,'fluxpkerr_obs':fluxpkerr_obs}
        if not (tflux is None):
            tflux = {'tflux':tf,'tfluxerr':tfe}
        
        return outstr
                                          
                                          
                                          