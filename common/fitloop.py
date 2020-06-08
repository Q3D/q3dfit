#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jun  8 11:14:14 2020

@author: drupke

Take outputs from IFSF and perform fitting loop. If loop is split among multiple
cores, then DRT_BRIDGELOOP parses this file to feed into a batch file.

:Returns:
   None.
   
:Params:
   ispax: in, required, type=int
     Value of index over which to loop
   colarr: in, required,type=intarr(2)
     Column # of spaxel (0-offset)
   rowarr: in, required, type=intarr(2)
     Row # of spaxel (0-offset)
   cube: in, required, type=structure
     Output from IFSF_READCUBE, containing data
   initdat: in, required, type=structure
     Output from initialization routine, containing fit parameters
   linelist: in, required, type=hash
     Output from IFSF_LINELIST.
   oned: in, required, type=byte
     Whether data is in a cube or in one dimension (longslit)
   onefit: in, required, type=byte
     If set, ignore second fit
   quiet: in, required, type=byte
     verbosity switch from IFSF
   
:Keywords:
   logfile: in, optional, type=strarr
     Names of log filesone per spaxel.


:History:
   ChangeHistory::
     2016sep18, DSNR, copied from IFSF into standalone procedure
     2016sep26, DSNR, small change in masking for new treatment of spec. res.
     2016oct20, DSNR, fixed treatment of SIGINIT_GAS
     2016nov17, DSNR, added flux calibration
     2018jun25, DSNR, added MC error calculation on stellar parameters

"""

def fitloop(ispax, colarr, rowarr, cube, initdat, linelist, oned, onefit, \
            quiet, logfile=None):
    
    import pdb
    import numpy as np

    if logfile:
        if isinstance(logfile,str): uselogfile = logfile
        else: uselogfile = logfile[ispax]
        loglun = open(uselogfile,'w')
    
    masksig_secondfit_def = 2.
    colind = ispax % cube.ncols
    rowind = int(ispax / cube.ncols)
    i = colarr[colind,rowind]
    j = rowarr[colind,rowind]
    print(f'[col,row]=[{i+1},{j+1}] out of [{cube.ncols},{cube.nrows}]',\
          file=loglun)
    
    if oned:
        flux = cube.dat[:,i]
        err = abs(cube.var[:,i])**0.5
        dq = cube.dq[:,i]
    else:
        flux = cube.dat[i,j,:]
        err = abs(cube.var[i,j,:])**0.5
        dq = cube.dq[i,j,:]
    errmax = max(err)
    
    if initdat.__contains__('vormap'):
        tmpi = cube.vorcoords[i,0]
        tmpj = cube.vorcoords[i,1]
        i = tmpi
        j = tmpj
        print(f'Reference coordinate: [col,row]=[{i+1},{j+1}]',file=loglun)
        
    if oned:
        outlab = '{[outdir]}{[label]}_{:04d}'.format(initdat,initdat,i+1)
    else:
        outlab = '{[outdir]}{[label]}_{:04d}_{:04d}'.format(initdat,initdat,i+1,j+1)

#   Apply DQ plane
    indx_bad = np.nonzero(dq > 0)
    if indx_bad[0].size > 0:
        flux[indx_bad] = 0.
        err[indx_bad] = errmax*100.

#   Check that the flux is not filled with 0s, infs, or nans
    somedata = ((flux != 0.).any() or \
                (flux != np.inf).any() or \
                (flux != np.nan).any())
    if somedata:
        
        if not initdat.__contains__('noemlinfit'):
            
#           Extract # of components specific to this spaxel and write as dict
#           Each dict key (line) will have one value (# comp)
            ncomp = dict()
            for line in initdat['lines']:
                if oned: ncomp[line] = initdat['ncomp'][line][i]
                else: ncomp[line] = initdat['ncomp'][line][i,j]
                
#       First fit

        dofit = True
        abortfit = False
        while(dofit):
            
#           Find lines where ncomp set to 0
            ct_comp_emlist = 0
            if not initdat.__contains__('noemlinfit'):
                for i in ncomp.values():
                    if i == 0: ct_comp_emlist+=1
            
            dofit = False