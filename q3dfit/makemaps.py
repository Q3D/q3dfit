#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
This procedure makes maps of various quantities. 

Retuns jpg plots.
Params:
    initproc: in, required, type=string
    Name of procedure to initialize the fit.

@author: hadley
"""
import numpy as np
import importlib
import os.path
import math
import sys
import pdb
import bisect
import matplotlib.pyplot as plt
sys.path.append("/Users/hadley/Desktop/research") 

from q3dfit.linelist import linelist
from q3dfit.readcube import Cube
from q3dfit.rebin import rebin
from q3dfit.cmpcvdf import cmpcvdf
from q3dfit.cmpcompvals import cmpcompvals
from q3dfit.lineratios import lineratios

from astropy.cosmology import FlatLambdaCDM
from astropy.modeling import models, fitting
from astropy.io import fits

import scipy.interpolate
import scipy.ndimage
import scipy.stats as stats

def hstsubim(image, subimsize, ifsdims, ifsps, ifspa, ifsrefcoords, \
                    hstrefcoords, scllim, hstps = False, ifsbounds = False, \
                    fov = False, sclargs = False, noscl = False, badmask = False, \
                    buffac = False): return None

def moffat (x, a):
    #from: https://github.com/drupke/drtools/blob/master/drt_moffat.pro
    '''
    Author: David S. N. Rupke
      Rhodes College
      Department of Physics
      2000 N. Parkway
      Memphis, TN 38104
      drupke@gmail.com
    '''
    u = (x.value - a[1])/a[2]
    mof = a[0]/(u**2.0 + 1.0)**(a[3])
    if len(a) > 5: mof = mof + a[4]
    if len(a) == 6: mof = mof + a[5]*x.value

    return mof

def congrid(a, newdims, method='linear', centre=False, minusone=False):
    #from https://scipy-cookbook.readthedocs.io/items/Rebinning.html
    '''Arbitrary resampling of source array to new dimension sizes.
    Currently only supports maintaining the same number of dimensions.
    To use 1-D arrays, first promote them to shape (x,1).
    
    Uses the same parameters and creates the same co-ordinate lookup points
    as IDL''s congrid routine, which apparently originally came from a VAX/VMS
    routine of the same name.

    method:
    neighbour - closest value from original data
    nearest and linear - uses n x 1-D interpolations using
                         scipy.interpolate.interp1d
    (see Numerical Recipes for validity of use of n 1-D interpolations)
    spline - uses ndimage.map_coordinates

    centre:
    True - interpolation points are at the centres of the bins
    False - points are at the front edge of the bin

    minusone:
    For example- inarray.shape = (i,j) & new dimensions = (x,y)
    False - inarray is resampled by factors of (i/x) * (j/y)
    True - inarray is resampled by(i-1)/(x-1) * (j-1)/(y-1)
    This prevents extrapolation one element beyond bounds of input array.
    '''
    if not a.dtype in [np.float64, np.float32]:
        a = np.cast[float](a)

    m1 = np.cast[int](minusone)
    ofs = np.cast[int](centre) * 0.5
    old = np.array( a.shape )
    ndims = len( a.shape )
    if len( newdims ) != ndims:
        print ("[congrid] dimensions error. " \
              "This routine currently only support " \
              "rebinning to the same number of dimensions.")
        return None
    newdims = np.asarray( newdims, dtype=float )
    dimlist = []

    if method == 'neighbour':
        for i in range( ndims ):
            base = np.indices(newdims)[i]
            dimlist.append( (old[i] - m1) / (newdims[i] - m1) \
                            * (base + ofs) - ofs )
        cd = np.array( dimlist ).round().astype(int)
        newa = a[list( cd )]
        return newa

    elif method in ['nearest','linear']:
        # calculate new dims
        for i in range( ndims ):
            base = np.arange( newdims[i] )
            dimlist.append( (old[i] - m1) / (newdims[i] - m1) \
                            * (base + ofs) - ofs )
        # specify old dims
        olddims = [np.arange(i, dtype = np.float) for i in list( a.shape )]

        # first interpolation - for ndims = any
        mint = scipy.interpolate.interp1d( olddims[-1], a, kind=method )
        newa = mint( dimlist[-1] )

        trorder = [ndims - 1] + range( ndims - 1 )
        for i in range( ndims - 2, -1, -1 ):
            newa = newa.transpose( trorder )

            mint = scipy.interpolate.interp1d( olddims[i], newa, kind=method )
            newa = mint( dimlist[i] )

        if ndims > 1:
            # need one more transpose to return to original dimensions
            newa = newa.transpose( trorder )

        return newa
    elif method in ['spline']:
        oslices = [ slice(0,j) for j in old ]
        oldcoords = np.ogrid[oslices]
        nslices = [ slice(0,j) for j in list(newdims) ]
        newcoords = np.mgrid[nslices]

        newcoords_dims = range(np.rank(newcoords))
        #make first index last
        newcoords_dims.append(newcoords_dims.pop(0))
        newcoords_tr = newcoords.transpose(newcoords_dims)
        # makes a view that affects newcoords

        newcoords_tr += ofs

        deltas = (np.asarray(old) - m1) / (newdims - m1)
        newcoords_tr *= deltas

        newcoords_tr -= ofs

        newa = scipy.ndimage.map_coordinates(a, newcoords)
        return newa
    else:
        print ("Congrid error: Unrecognized interpolation type.\n", \
              "Currently only \'neighbour\', \'nearest\',\'linear\',", \
              "and \'spline\' are supported.")
        return None
    
def makemaps (initproc):
    
    fwhm2sig = 2.0 * np.sqrt(2.0 * np.log(2.0))
    plotquantum = 2.5                       #in inches
    bad = 1e99
    c_kms = 299792.458
    ncbdivmax = 7
    maxnadabscomp = 3
    maxnademcomp = 3
    taumax = 5.0
    
    #physical constants
    m_particleg = 1.4 * 1.672649e24              #(log) mass per particle, in solar grams                     
    m_particlesm = 1.4 * 1.672649e-24 / 1.989e33 #(log) mass per particle, in solar masses
    secperyr = 24.0 * 3600.0 * 365.25            #seconds in a year
    mperpc = 3.0856e16                           #meters in a parsec
    lsun = 3.826e33                              #solar luminosities, erg/s
    msun = 1.989e33                              #solar mass, g
    c_cms = 2.99792e10
    ionfrac = 0.9                                #Na ionization fraction
    oneminusionfrac_relerr = [0,0]
    naabund = -5.69                              #Na abundance
    nadep = 0.95                                 #Na depletion
    
    volemis = 2.63e-25                           #volume emissivity of Ha 
                                                 # = product of recomb. coeff. 
                                                 #and photon energy [erg cm^3 s^-1]
    elecden_default = 100.0                      #electron density, cm^-3
    elecden_err_default = [50.0,100.0]

    lineofdashes = '-' * (62 * 62)               #string of 62*62 dashes
    
    #factor by which to resample images for PS-to-PDF conversion
    samplefac = 10
    resampthresh = 500
    
    #Values for computing electron density from [SII] ratio
    #from Sanders, Shapley, et al. 2015
    s2_minratio = 0.4375
    s2_maxratio = 1.4484
    s2_a = 0.4315
    s2_b = 2107
    s2_c = 627.1
#   s2_maxden = (s2_c * s2_minrat - s2_a*s2_b)/(s2_a - s2_minrat)
#   s2_minden = (s2_c * s2_maxrat - s2_a*s2_b)/(s2_a - s2_maxrat)
    s2_maxdensity = 1e4
    s2_mindensity = 1e1
    
    '''''''''''''''''''''''''''''''''''''''''''''''''''''''''''
    ' Load initialization parameters and line data            '
    '''''''''''''''''''''''''''''''''''''''''''''''''''''''''''

    #Get initialization structures
    initmaps = np.load('/Users/hadley/Desktop/research/initmaps.npy', allow_pickle = True)
    initmaps = initmaps.item(0)
    
    module = importlib.import_module('q3dfit.init.' + initproc)
    fcninitproc = getattr(module, initproc)
    initdat = fcninitproc()
    
    if 'donad' in initdat:
        initnad = {'dumy': 1}
        
        module = importlib.import_module('q3dfit.init.' + initproc)
        fcninitproc = getattr(module, initproc)
        initdat = fcninitproc(initproc, initnad = initnad)
    
    #Get galaxy-specific parameters from initialization file
    center_axes = -1
    center_nuclei = -1
    if 'center_axes' in initmaps:
        center_axes = initmaps['center_axes']
    if 'center_nuclei' in initmaps:
        center_nuclei = initmaps['center_nuclei']

    #Get linelist
    if 'noemlinfit' not in initdat:
        linelabels = 1.0
        if 'argslinelist' in initdat:
            listlines = linelist(inlines = initdat['lines'], linelab = True)
        else:
            listlines = linelist(inlines = initdat['lines'], linelab = True)

    #Linelist with doublets to combine
    emldoublets = np.array([['[SII]6716','[SII]6731'],
                            ['[OII]3726','[OII]3729'],
                            ['[NI]5198','[NI]5200'],
                            ['[NeIII]3869','[NeIII]3967'],
                            ['[NeV]3345','[NeV]3426'],
                            ['MgII2796','MgII2803']])
     
    if emldoublets.ndim == 1: numdoublets = 1 
    else: numdoublets = emldoublets[0].size
    lines_with_doublets = initdat['lines']
     
    
    for i in range(numdoublets - 1):
        
        if (emldoublets[0,i] in listlines['name']) and (emldoublets[1,i] in listlines['name']):
           dkey = emldoublets[0,i] + '+' + emldoublets[1,i]
           lines_with_doublets = np.array([lines_with_doublets, dkey])
        
    if 'argslinelist' in initdat:
        linelist_with_doublets = \
            linelist(inlines = lines_with_doublets, linelab = True)
    else:
        linelist_with_doublets = \
            linelist(inlines = lines_with_doublets, linelab = True)

    if 'donad' in initdat:
        if 'argslinelist' in initdat:
            nadlinelist = linelist(inlines = ['NaD1','NaD2','HeI5876'])
        else:
            nadlinelist = linelist(inlines = ['NaD1','NaD2','HeI5876'])

    '''
    Get range file

    plot types, in order; used for correlating with input ranges (array 
                                                                  rangequant)
    '''

    hasrangefile = 0
    if 'rangefile' in initmaps:
      if os.path.isfile(initmaps['rangefile']): 
          #readcol,initmaps.rangefile,rangeline,rangequant,rangelo,rangehi,$
                 #rangencbdiv,format='(A,A,D,D,I)',/silent
         hasrangefile = 1
      else: print('Range file listed in INITMAPS but not found.')

      
    #TODO: file and struct names below (4)

    #Restore line maps
    if 'noemlinfit' not in initdat:
        file = initdat['outdir'] + initdat['label'] + '.lin.npz'
        struct1 = (np.load(file, allow_pickle='TRUE')).files #this is that thing with the emission line stuff
   
    #Restore continuum parameters
    if 'decompose_ppxf_fit' in initdat or 'decompose_qso_fit' in initdat:
        file = initdat['outdir'] + initdat['label'] + '.cont.npy'
        contcube = (np.load(file, allow_pickle='TRUE')).item()
    
    #Get NaD parameters
    if 'donad' in initdat:
        file = initdat['outdir'] + initdat['label'] + '.nadspec.xdr'
        nadcube = (np.load(file, allow_pickle='TRUE')).item()
        file = initdat['outdir'] + initdat['label'] + '.nadfit.xdr'
        nadfit = (np.load(file, allow_pickle='TRUE')).item()        
        
        if 'badnademp' in initmaps:
            tagstobad = np.array(['WEQ','IWEQ','EMFLUX','EMUL','VEL'])
            tagnames = nadcube.keys()
            
            ibad = np.where(initmaps['badnademp'] == 0)
            ctbad = len(ibad)
            
            if ctbad > 0:
                for i in range (tagstobad.size - 1):
                    itag = np.where(tagnames == tagstobad[i])
                    sizetag = nadcube[itag].size
                    for j in range (sizetag[3] - 1):
                        tmp = nadcube[itag][:, :, j]
                        tmp[ibad] = bad
                        nadcube[itag][:, :, j] = tmp

    if ('donad' not in initdat) and ('noemlinfit' in initmaps):
        print('No emission line or absorption line data specified.')
                        
        
    '''''''''''''''''''''''''''''''''''''''''''''''''''''''''''      
    ' Compute some things                                     '
    '''''''''''''''''''''''''''''''''''''''''''''''''''''''''''
    
    #Luminosity and angular size distances
    if 'distance' in initdat:
        ldist = initdat['distance']
        asdist = ldist / (1.0 + initdat['zsys_gas']) ** 2.0
    else:
        #Planck 2018 parameters: https://ui.adsabs.harvard.edu/#abs/arXiv:1807.06209
        cosmo = FlatLambdaCDM(H0=67.4, Om0=0.315) #Lambda0=0.685
        ldist = cosmo.luminosity_distance(initdat['zsys_gas'])
        asdist = ldist / (1.0 + initdat['zsys_gas']) ** 2.0
        
    kpc_per_as = asdist * 1000.0 / 206265.0
    kpc_per_pix = initdat['platescale'] * kpc_per_as


    '''''''''''''''''''''''''''''''''''''''''''''''''''''''''''
    ' Load and process continuum data                         '
    '''''''''''''''''''''''''''''''''''''''''''''''''''''''''''
    
    #Data cube
    if 'datext' not in initdat: datext = 1 
    else: datext = initdat['datext']
   
    if 'varext' not in initdat: varext = 2 
    else: varext = initdat['varext']
   
    if 'dqext' not in initdat: dqext = 3 
    else: dqext = initdat['dqext']
   
    datacube = Cube(initdat['infile'], datext=datext, varext=varext,
                    dqext=dqext)
    if 'fluxfactor' in initmaps:
        datacube.dat *= initmaps['fluxfactor']
        datacube.var *= (initmaps['fluxfactor']) ^ 2.0
   
    

    dx = datacube.nrows
    dy = datacube.ncols
    dz = datacube.nw
   
    #defined so that center of spaxel at bottom left has coordinates [1,1] 
    if center_axes[0] == -1: center_axes = [float(dx) / 2.0, float(dy) / 2.0] + 0.5
    if center_nuclei[0] == -1: center_nuclei = center_axes
    if 'vornorm' in initmaps:
        datacube.dat /= rebin(initmaps['vornorm'], (dx, dy, dz))
   

    #Image window
    if 'plotwin' in initmaps:
        plotwin = initmaps['plotwin']
        dxwin = plotwin[2] - plotwin[0] + 1
        dywin = plotwin[3]- plotwin[1] + 1
    else:
        plotwin = [1, 1, dx, dy]
        dxwin = dx
        dywin = dy
   
    #Figure aspect ratio multiplier
    if 'aspectrat' in initmaps: aspectrat = initmaps['aspectrat']
    else: aspectrat = 1.0

    #HST data
    dohst = False
    dohstbl = False
    dohstrd = False
    dohstsm = False
    dohstcol = False
    dohstcolsm = False
    if 'hst' in initmaps and 'hstbl' in initmaps:
        dohstbl = True
        if 'ext' in initmaps['hstbl']: hstblext = initmaps['hstbl']['ext']
        else: hstblext = True
        hstbl = fits.open(initmaps['hstbl']['file'])
        hstblhead = hstbl.header
        hst_big_ifsfov = np.zeros((4,3), dtype = float)
        if 'platescale' in initmaps['hstbl']:
            hstpsbl = initmaps['hstbl']['platescale']
        else: hstpsbl = 0.05
        if 'refcoords' in initmaps['hstbl']:
            hstrefcoords = initmaps['hstbl']['refcoords']
        else: hstrefcoords = initmaps['hst']['refcoords']
        if initmaps['hstbl']['buffac'] in initmaps['hstbl']:
            hstbl_buffac = initmaps['hstbl']['buffac']
        else: hstbl_buffac = 2.0
        
        if ('subim_sm' in initmaps['hst']) and ('sclargs_sm' in initmaps['hstbl']):
            hst_sm_ifsfov = np.zeros((4,2), dtype = float)
            bhst_sm = hstsubim(hstbl, [initmaps['hst']['subim_sm'], \
                                 initmaps['hst']['subim_sm']], \
                                 [dx,dy], initdat['platescale'], \
                                 initdat['positionangle'], center_nuclei, \
                                 hstrefcoords, \
                                 initmaps['hstbl']['scllim'], \
                                 sclargs = initmaps['hstbl.sclargs_sm'], \
                                 ifsbounds = hst_sm_ifsfov, hstps = hstpsbl)
      
        bhst_big = hstsubim(hstbl, [initmaps['hst']['subim_big'],\
                               initmaps['hst']['subim_big']], \
                               [dx,dy],initdat['platescale'], \
                               initdat['positionangle'], center_nuclei, \
                               hstrefcoords, \
                               initmaps['hstbl']['scllim'], \
                               sclargs = initmaps['hstbl']['sclargs_big'], \
                               ifsbounds = hst_big_ifsfov, hstps = hstpsbl)
        bhst_fov = hstsubim(hstbl,[0,0],[dx,dy], initdat['platescale'], \
                               initdat['positionangle'], center_nuclei, \
                               hstrefcoords, \
                               initmaps['hstbl']['scllim'], \
                               sclargs = initmaps['hstbl']['sclargs_fov'], \
                               fov = True, hstps = hstpsbl, buffac = hstbl_buffac)
        bhst_fov_ns = hstsubim(hstbl, [0,0], [dx,dy], initdat['platescale'], \
                                  initdat['positionangle'], center_nuclei, \
                                  hstrefcoords, [0,0], noscl = True, fov = True, \
                                  hstps = hstpsbl, buffac = hstbl_buffac)
      
        if 'hstblsm' in initmaps: 
            dohstsm = 1 
            #For F05189, mask central pixels before smoothing
            if initdat['label'] == 'f05189':
                size_tmp = hstbl.shape
                
                map_x_tmp = rebin(np.arange(size_tmp[0]), (size_tmp[0], size_tmp[1]))
                map_y_tmp = rebin(np.transpose(np.arange(size_tmp[1])), \
                              (size_tmp[0], size_tmp[1]))
                map_rkpc_tmp = np.sqrt((map_x_tmp - (hstrefcoords[0] + \
                               initmaps['hstbl']['nucoffset'][0] - 1)) ^ 2.0 + \
                               (map_y_tmp - (hstrefcoords[1] + \
                                initmaps['hstbl']['nucoffset'][1] - 1)) ^ 2.0) * \
                                initmaps['hstbl']['platescale'] * kpc_per_as
                ipsf = np.where(map_rkpc_tmp < 0.15)
                ipsf_bkgd = np.where((map_rkpc_tmp > 0.15) and (map_rkpc_tmp < 0.25))
                hstbl_tmp = hstbl
                hstbl_tmp[ipsf] = np.median(hstbl[ipsf_bkgd])
                #hstblsm = filter_image(hstbl_tmp,fwhm=initmaps.hst.smoothfwhm,/all)
                hstblsm = hstbl_tmp
            else:
                hstbltmp = hstbl
                ibadhst = np.where(hstbl == 0.0)
                ctbadhst = len(ibadhst)
                if ctbadhst > 0: hstbltmp[ibadhst] = math.nan
                fwhm = initmaps['hst']['smoothfwhm']
                boxwidth = round((fwhm ^ 2.0) / 2.0 + 1.0)
                if not boxwidth: boxwidth + 1
                #hstblsm = filter_image(hstbl,fwhm=initmaps.hst.smoothfwhm,/all)
                hstblsm = hstbl
                #hstblsm = filter_image(hstbltmp,smooth=boxwidth,/iter,/all)
                hstblsm = hstbltmp
                ibadhst = np.where(np.isfinite(hstblsm))
                ctbadhst = len(ibadhst)
                if ctbadhst > 0: hstblsm[ibadhst] = bad
                hstbltmp = 0
                  
                '''
                bhst_fov_sm = hstsubim(hstblsm, [0,0], [dx,dy], \
                                      initdat['platescale'], \
                                      initdat['positionangle'], center_nuclei, \
                                      hstrefcoords, \
                                      initmaps['hstblsm']['scllim'], \
                                      sclargs = initmaps['hstblsm']['sclargs'], \
                                      fov = True, hstps = hstpsbl, buffac = hstbl_buffac)
                '''
                
                bhst_fov_sm_ns= hstsubim(hstblsm, [0,0], [dx,dy], \
                                       initdat['platescale'], \
                                       initdat['positionangle'], center_nuclei, \
                                       hstrefcoords,[0,0], noscl = True, \
                                       fov = True, hstps = hstpsbl, buffac = hstbl_buffac)
                
                bhst_fov_sm_ns_rb = congrid(bhst_fov_sm_ns, dx, dy, center = True)
                if 'vormap' in initdat:
                    tmp_rb = bhst_fov_sm_ns_rb
                    nvor = max(initdat.vormap)
                    badvor = np.where(~np.isfinite(initdat['vormap']))
                    ctbad = len(badvor)
                    if ctbad > 0: tmp_rb[badvor] = 0.0
                    for i in range (1, nvor):
                        ivor = np.where(initdat['vormap'] == i)
                        ctbins = len(ivor)
                        tmp_rb[ivor] = sum(bhst_fov_sm_ns_rb[ivor]) / ctbins
                    bhst_fov_sm_ns_rb = tmp_rb
                 
                if 'bgsub' in initmaps['hstblsm']:
                    bhst_fov_sm_ns_rb -= initmaps['hstblsm']['bgsub']
                    
    if ('hst' in initmaps) and ('hstrd' in initmaps):
        dohstrd = 1
        if 'ext' in initmaps['hstrd']: hstrdext = initmaps['hstrd']['ext']
        else: hstrdext = 1
        hstrd = fits.open(initmaps['hstrd']['file'])
        hstrdhead = hstrd.header
        hst_big_ifsfov = np.zeros((4,2), dtype = float)
        if 'refcoords' in initmaps['hstrd']:
            hstrefcoords = initmaps['hstrd']['refcoords']
        else: hstrefcoords = initmaps['hst']['refcoords']
        if 'platescale' in initmaps['hstrd']:
            hstpsrd = initmaps['hstrd']['platescale']
        else: hstpsrd = 0.05
        if 'buffac' in initmaps['hstrd']:
            hstrd_buffac = initmaps['hstrd']['buffac']
        else: hstrd_buffac = 2.0
        if 'subim_sm' in initmaps['hst'] and \
            'sclargs_sm' in initmaps['hstrd']:
            hst_sm_ifsfov = np.zeros((4,2), dtype = float)
            rhst_sm = hstsubim(hstrd, [initmaps['hst']['subim_sm'],
                               initmaps['hst']['subim_sm']], [dx,dy],
                               initdat['platescale'], initdat['positionangle'],
                               center_nuclei, hstrefcoords, 
                               initmaps['hstrd']['scllim'],
                               sclargs = initmaps['hstrd']['sclargs_sm'],
                               ifsbounds = hst_sm_ifsfov, hstps = hstpsrd)
        rhst_big = hstsubim(hstrd,[initmaps['hst']['subim_big'],
                               initmaps['hst']['subim_big']],
                               [dx,dy], initdat['platescale'],
                               initdat['positionangle'], center_nuclei,
                               hstrefcoords, initmaps['hstrd']['scllim'],
                               sclargs = initmaps['hstrd']['sclargs_big'],
                               ifsbounds = hst_big_ifsfov, hstps = hstpsrd)
        rhst_fov_sc = hstsubim(hstrd, [0,0], [dx,dy], initdat['platescale'],
                               initdat['positionangle'], center_nuclei,
                               #0, center_nuclei, hstrefcoords,
                               initmaps['hstrd']['scllim'],
                               sclargs = initmaps['hstrd']['sclargs_fov'],
                               fov = True, hstps = hstpsrd, buffac = hstrd_buffac)
        rhst_fov_ns = hstsubim(hstrd, [0,0], [dx,dy], initdat['platescale'],
                               initdat['positionangle'], center_nuclei,
                               #0,center_nuclei, hstrefcoords, [0,0],
                               noscl = True, fov = True,
                               hstps = hstpsrd, buffac = hstrd_buffac)
        
        if 'hstrdsm' in initmaps: 
            dohstsm = 1
            
            #For F05189, mask central pixels before smoothing
            if initdat['label'] == 'f05189':
                size_tmp = hstrd.shape
                map_x_tmp = rebin(np.arange(size_tmp[0]), (size_tmp[0], size_tmp[1]))
                map_y_tmp = rebin(np.transpose(np.arange(size_tmp[1])), \
                              (size_tmp[0],size_tmp[1]))
                map_rkpc_tmp = np.sqrt((map_x_tmp - (hstrefcoords[0] + \
                                 initmaps['hstrd']['nucoffset'][0] - 1)) ^ 2.0 + \
                                (map_y_tmp - (hstrefcoords[1] + \
                                 initmaps['hstrd']['nucoffset'][1] - 1)) ^ 2.0) * \
                                 initmaps['hstrd']['platescale'] * kpc_per_as
                ipsf = np.where(map_rkpc_tmp < 0.15)
                ipsf_bkgd = np.where(map_rkpc_tmp > 0.15 and map_rkpc_tmp < 0.25)
                hstrd_tmp = hstrd
                hstrd_tmp[ipsf] = np.median(hstrd[ipsf_bkgd])
                #hstrdsm = filter_image(hstrd_tmp,fwhm=initmaps.hst.smoothfwhm,/all)
                hstrdsm = hstrd_tmp
            else:
                hstrdtmp = hstrd
                ibadhst = np.where(hstrd == 0)
                ctbadhst = len(ibadhst)
                if ctbadhst > 0: hstrdtmp[ibadhst] = math.nan
                fwhm = initmaps['hst']['smoothfwhm']
                boxwidth = round((fwhm ^ 2.0) / 2.0 + 1.0)
                if not boxwidth: boxwidth + 1
                #hstrdsm = filter_image(hstrd,fwhm=initmaps.hst.smoothfwhm,/all)
                #hstrdsm = filter_image(hstrdtmp,smooth=boxwidth,/iter,/all)
                hstrdsm = hstrdtmp
                ibadhst = np.where(np.finite(hstrdsm))
                ctbadhst = len(ibadhst)
                if ctbadhst > 0: hstrdsm[ibadhst] = bad
                hstrdtmp = 0

            '''
            rhst_fov_sm = hstsubim(hstrdsm,[0,0],[dx,dy], initdat['platescale'], \
                                      initdat['positionangle'], center_nuclei, \
                                      hstrefcoords, \
                                      initmaps['hstrdsm']['scllim'], \
                                      sclargs = initmaps['hstrdsm']['sclargs'], \
                                      fov = True, hstps = hstpsrd, buffac = hstrd_buffac)
            '''
            rhst_fov_sm_ns= hstsubim(hstrdsm,[0,0],[dx,dy], initdat['platescale'],
                                       initdat['positionangle'], center_nuclei,
                                       hstrefcoords,[0,0], noscl = True,
                                       fov = True, hstps = hstpsrd, buffac = hstrd_buffac)

            rhst_fov_sm_ns_rb = congrid(rhst_fov_sm_ns, dx, dy, center = True)
            if 'vormap' in initdat:
                tmp_rb = rhst_fov_sm_ns_rb
                nvor = max(initdat['vormap'])
                badvor = np.where(~np.finite(initdat.vormap))
                ctbad = len(badvor)
                if ctbad > 0: tmp_rb[badvor] = 0.0
                for i in range (1, nvor):
                   ivor = np.where(initdat.vormap == i)
                   ctbins = len(ivor)
                   tmp_rb[ivor] = np.sum(rhst_fov_sm_ns_rb[ivor]) / ctbins
                rhst_fov_sm_ns_rb = tmp_rb
            if 'bgsub' in initmaps['hstrdsm']:
                rhst_fov_sm_ns_rb -= initmaps['hstrdsm']['bgsub']
      
    if dohstbl or dohstrd: dohst = True
    if dohst and 'hstcol' in initmaps:
        dohstcol = 1
        if initmaps['hstbl']['platescale'] != initmaps['hstrd']['platescale']:
             print('MAKEMAPS: EROR: HST blue and red plate scales differ.')
             quit()
      
        '''
      HST ABmag computations. See
      http://hla.stsci.edu/hla_faq.html#Source11
      http://www.stsci.edu/hst/acs/documents/handbooks/currentDHB/acs_Ch52.html#102632
      http://www.stsci.edu/hst/acs/analysis/zeropoints
      Convert from e-/s to ABmags in the filters using:
         ABmag = -2.5 log(e-/s) + ZP
         ZP(ABmag) = -2.5 log(PHOTFLAM) - 2.408 - 5 log(PHOTPLAM)
      The result will be mags per arcsec^2.   
        '''    
        abortrd = 'No PHOTFLAM/PLAM keyword in red continuum image.'
        abortbl = 'No PHOTFLAM/PLAM keyword in blue continuum image.'
        if 'photplam' in initmaps['hstbl']:
            pivotbl=initmaps['hstbl']['photplam']
        else:
            pivotbl = hstblhead['PHOTPLAM']
            
        if 'photplam' in initmaps['hstrd']: pivotrd=initmaps['hstrd']['photplam']
        else:
            pivotrd = hstrdhead['PHOTPLAM']
        if 'zp' in initmaps['hstbl']: zpbl = initmaps['hstbl']['zp']
        else:
            hdu = hstblhead[0]
            zpbl = -2.5 * math.log10(hdu['PHOTFLAM']) - 2.408 - \
                5.0 * math.log10(hdu['PHOTFLAM'])
        if 'zp' in initmaps['hstrd']: zprd = initmaps['hstrd']['zp']
        else:
            zprd = -2.5 * math.log10(hstrdhead['PHOTFLAM'] - 2.408 - \
                5.0 * math.log10(hstrdhead['PHOTPLAM']))
        #Shift one image w.r.t. the other if necessary
        #Use red image as reference by default
        if ('refcoords' in initmaps['hstbl']) and ('refcoords' in initmaps['hstrd']):
            if initmaps['hstbl']['refcoords'][0] != initmaps['hstrd']['refcoords'][0] or \
                initmaps['hstbl']['refcoords'][1] != initmaps['hstrd']['refcoords'][1]:
                    idiff = initmaps['hstrd']['refcoords'] - initmaps['hstbl']['refcoords'] 
                    hstbl = np.roll(hstbl, round(idiff[0]), axis = 1) #shift horiz.
                    hstbl = np.roll(hstbl, round(idiff[1]), axis = 0) #shift vert.
                    #hstbl = shift(hstbl,fix(idiff[0]),fix(idiff[1]))
                    
            else: hstrefcoords = initmaps['hstrd']['refcoords']
        else: hstrefcoords = initmaps['hst']['refcoords']
      
        #Resize images to same size if necessary
        sizebl = hstbl.shape
        sizerd = hstrd.shape
        if (sizebl[0] != sizerd[0]) or (sizebl[1] != sizerd[1]):
            #fixing size of x
            if sizebl[0] != sizerd[0]:
                if sizebl[0] > sizerd[0]: hstbl = hstbl[sizerd[0] - 1,:]
                else: hstrd = hstrd[sizebl[1] - 1, :]
            #fixing size of y
            if sizebl[1] != sizerd[1]:
                if sizebl[1] > sizerd[1]: hstbl = hstbl[:, sizerd[1] - 1]
                else: hstrd = hstrd[:, sizebl[1] - 1]
        
        #Take a bunch of random samples of HST image
        if 'sdevuplim' in initmaps['hst']: uplim = initmaps['hst']['sdevuplim']
        else: uplim = 0.1 #this gets rid of cosmic rays and stars ...
        if 'sdevreg' in initmaps['hst']: sdevreg = initmaps['hst']['sdevreg']
        else:
            size_hst = hstrd.shape
            pxhst = round(size_hst[0] / 10)
            pyhst = round(size_hst[1] / 10)
            sdevreg[:, 0] = [3 * pxhst, 4 * pxhst, 3 * pyhst, 4 * pyhst]
            sdevreg[:, 1] = [3 * pxhst, 4 * pxhst, 6 * pyhst, 7 * pyhst]
            sdevreg[:, 2] = [6 * pxhst, 7 * pxhst, 3 * pyhst, 4 * pyhst]
            sdevreg[:, 3] = [6 * pxhst, 7 * pxhst, 6 * pyhst, 7 * pyhst]
      
        nsdevreg = sdevreg.size
        if sdevreg.ndim == 1:
            sdevreg = sdevreg.reshape(4,1)
            #TODO
            #sdevreg = reform(sdevreg,4,1)
            nsdevreg = 1
      
        sdev = np.zeros((nsdevreg), dtype = float)
        for i in range (0, nsdevreg - 1):
            #hsttmp = hstblsm[sdevreg[0, i] : sdevreg[1, i], sdevreg[2, i] : sdevreg[3, i]]
            hsttmp = hstblsm[np.arange(sdevreg[0, i], sdevreg[1, i] + 1), np.arange(sdevreg[2, i], sdevreg[3, i] + 1)]
            sdev[i] = np.std(hsttmp[np.where(hsttmp != 0 and hsttmp < uplim)])
      
        sdevbl = np.median(sdev)
        for i in range (0, nsdevreg - 1):
            hsttmp = hstrdsm[np.arange(sdevreg[0,i], sdevreg[1,i]), np.arange(sdevreg[2,i], sdevreg[3,i])]
            sdev[i] = np.std(np.where(hsttmp != 0 and hsttmp < uplim))
        sdevrd = np.median(sdev)
        
        
        #Find bad pixels
        if 'sigthr' in initmaps['hstcol']: colsigthr = initmaps['hstcol']['sigthr']
        else: colsigthr = 3.0
        ibdcol = np.where(hstrd < colsigthr * sdevrd or hstbl < colsigthr * sdevbl)
        ctbdcol = len(ibdcol)
        hstcol = -2.5 * np.log10(hstbl / hstrd) + zpbl - zprd
        
        #Galactic extinction correction:
        #(x - y)_intrinsic = (x - y)_obs - (A_x - A_y)
        #galextcor = A_x - A_y
        if 'galextcor' in initmaps['hstcol']:
            hstcol -= initmaps['hstcol']['galextcor']
        hstcol[ibdcol] = bad
        
        if hstbl_buffac > hstrd_buffac: hstcol_buffac = hstrd_buffac
        else: hstcol_buffac = hstbl_buffac
        
        #Extract and scale
        if 'subim_sm' in initmaps['hst']:
            chst_sm = hstsubim(hstcol,[initmaps['hst']['subim_sm'],
                               initmaps['hst']['subim_sm']],
                               [dx,dy], initdat['platescale'],
                               initdat['positionangle'], center_nuclei,
                               hstrefcoords, initmaps['hstcol']['scllim'],
                               sclargs = initmaps['hstcol']['sclargs'], hstps = hstpsbl)
            
        chst_big = hstsubim(hstcol, [initmaps['hst']['subim_big'],
                               initmaps['hst']['subim_big']],
                               [dx,dy], initdat['platescale'],
                               initdat['positionangle'], center_nuclei,
                               hstrefcoords, initmaps['hstcol']['scllim'],
                               sclargs = initmaps['hstcol']['sclargs'], hstps = hstpsbl)
        chst_fov = hstsubim(hstcol, [0,0], [dx,dy], initdat['platescale'],
                               initdat['positionangle'], center_nuclei,
                               hstrefcoords, initmaps['hstcol']['scllim'],
                               sclargs = initmaps['hstcol']['sclargs'],
                               fov = True, hstps = hstpsbl, buffac = hstcol_buffac)
        if ctbdcol > 0:
            hstcol_bad = hstcol * 0.0
            hstcol_bad[ibdcol] = 1.0
            chst_fov_bad = hstsubim(hstcol_bad, [0,0], [dx,dy], initdat['platescale'],
                                      initdat['positionangle'], center_nuclei,
                                      hstrefcoords, initmaps['hstcol']['scllim'],
                                      noscl = True, fov = True, hstps = hstpsbl, 
                                      badmask = True, buffac = hstcol_buffac)
            ibdcol_chst_fov = np.where(chst_fov_bad == 1.0)
            ctbdcol_chst_fov = len(ibdcol_chst_fov)
            if ctbdcol_chst_fov > 0: chst_fov[ibdcol_chst_fov] = bytes(255)
            hstcol_bad = 0
      
        #Extract unscaled color image
        chst_fov_ns = hstsubim(hstcol, [0,0], [dx,dy], initdat['platescale'],
                                  initdat['positionangle'], center_nuclei,
                                  hstrefcoords, initmaps['hstcol']['scllim'],
                                  noscl = True, fov = True, hstps = hstpsbl,
                                  buffac = hstcol_buffac)
        
    if 'hst' in initmaps and 'hstcolsm' in initmaps:
        dohstcolsm = 1
        #Shift one image w.r.t. the other if necessary
        #Use red image as reference by default
        if ('refcoords' in initmaps['hstbl']) and ('refcoords' in initmaps['hstrd']):
            if initmaps['hstbl']['refcoords'][0] != initmaps['hstrd']['refcoords'][0] and \
            initmaps['hstbl']['refcoords'][1] != initmaps['hstrd']['refcoords'][1]:
                idiff = initmaps['hstrd']['refcoords'] - initmaps['hstbl']['refcoords']
                hstblsm = np.roll(hstblsm, np.fix(idiff[0]), axis = 1)
                hstblsm = np.roll(hstblsm, np.fix(idiff[1]), axis = 0)
                #hstblsm = shift(hstblsm,fix(idiff[0]),fix(idiff[1]))
            else: hstrefcoords = initmaps['hstrd']['refcoords']
        else: hstrefcoords = initmaps['hst']['refcoords']
      
        #Resize images to same size if necessary
        sizebl = hstblsm.shape
        sizerd = hstrdsm.shape
        if sizebl[0] != sizerd[0] or sizebl[1] != sizerd[1]:
            if sizebl[0] != sizerd[0]:
                if sizebl[0] > sizerd[0]: hstblsm = hstblsm[sizerd[0] - 1, :]
                else: hstrdsm = hstrdsm[sizebl[0] - 1, :]
            if sizebl[1] != sizerd[1]:
                if sizebl[1] > sizerd[1]: hstblsm = hstblsm[:, sizerd[1] - 1]
                else: hstrdsm = hstrdsm[:, sizebl[1]-1]
      
        #Take a bunch of random samples of HST image
        if 'sdevuplim' in initmaps['hst']: uplim = initmaps['hst']['sdevuplim']
        else: uplim = 0.1 #this gets rid of cosmic rays and stars ...
        if 'sdevreg' in initmaps['hst']: sdevreg = initmaps['hst']['sdevreg']
        else:
            size_hst = hstrd.shape
            pxhst = round(size_hst[0] / 10)
            pyhst = round(size_hst[1] / 10)
            sdevreg[:, 0] = [3 * pxhst, 4 * pxhst, 3 * pyhst, 4 * pyhst]
            sdevreg[:, 1] = [3 * pxhst, 4 * pxhst, 6 * pyhst, 7 * pyhst]
            sdevreg[:, 2] = [6 * pxhst, 7 * pxhst, 3 * pyhst, 4 * pyhst]
            sdevreg[:, 3] = [6 * pxhst, 7 * pxhst, 6 * pyhst, 7 * pyhst]
      
        nsdevreg = sdevreg.size
        if sdevreg.ndim == 1 : #maybe
            flux = np.array(sdevreg)[4, 1].flatten()
            nsdevreg = 1
      
        
        sdev = np.zeros((nsdevreg), dtype = float)
        for i in range (0, nsdevreg - 1):
            #hsttmp = hstblsm[sdevreg[0, i] : sdevreg[1, i], sdevreg[2, i] : sdevreg[3, i]]
            hsttmp = hstblsm[np.arange(sdevreg[0, i], sdevreg[1, i] + 1), np.arange(sdevreg[2, i], sdevreg[3, i] + 1)]
            sdev[i] = np.std(hsttmp[np.where(hsttmp != 0 and hsttmp < uplim)])
      
        sdevbl = np.median(sdev)
        for i in range (0, nsdevreg - 1):
            hsttmp = hstrdsm[np.arange(sdevreg[0,i], sdevreg[1,i]), np.arange(sdevreg[2,i], sdevreg[3,i])]
            sdev[i] = np.std(np.where(hsttmp != 0 and hsttmp < uplim))
        sdevrd = np.median(sdev)
        
        
        
        #Find bad pixels
        if 'colsigthr' in initmaps['hst']: colsigthr = initmaps['hst']['colsigthr']
        else: colsigthr = 3.0
      
        #Color maps
        hstcolsm = -2.5 * np.log10(hstblsm / hstrdsm) + zpbl - zprd
        if 'galextcor' in initmaps['hstcol']:
            hstcolsm -= initmaps['hstcol']['galextcor']
        ibdcol = np.where(hstrdsm < colsigthr * sdevrd or hstblsm < colsigthr * sdevbl)
        hstcolsm[ibdcol] = bad

        #Extract and scale
        if 'subim_sm' in initmaps['hst']:
            
            cshst_sm = hstsubim(hstcolsm, [initmaps['hst']['subim_sm'],
                                initmaps['hst']['subim_sm']], [dx,dy],
                                initdat['platescale'],
                                initdat['positionangle'], center_nuclei,
                                hstrefcoords, initmaps['hstcolsm']['scllim'],
                                sclargs = initmaps['hstcolsm']['sclargs'],
                                hstps = hstpsbl)
      
        cshst_big = hstsubim(hstcolsm, [initmaps['hst']['subim_big'],
                                initmaps['hst']['subim_big']], [dx,dy],
                                initdat['platescale'],
                                initdat['positionangle'], center_nuclei,
                                hstrefcoords, initmaps['hstcolsm']['scllim'],
                                sclargs = initmaps['hstcolsm']['sclargs'],
                                hstps = hstpsbl)
        cshst_fov_s = hstsubim(hstcolsm, [0,0], [dx,dy], initdat['platescale'],
                                initdat['positionangle'], center_nuclei,
                                hstrefcoords, initmaps['hstcolsm']['scllim'],
                                sclargs = initmaps['hstcolsm']['sclargs'],
                                fov = True, hstps = hstpsbl, buffac = hstcol_buffac)
        
        #Extract unscaled color image and convert to same pixel scale as IFS data
        cshst_fov_ns = hstsubim(hstcolsm, [0,0], [dx,dy], initdat['platescale'],
                                initdat['positionangle'], center_nuclei,
                                hstrefcoords, [0,0], noscl = True, fov = True,
                                hstps = hstpsbl, buffac = hstcol_buffac)
        
        cshst_fov_rb = congrid(cshst_fov_ns, dx, dy, center = True)
   
    if dohst and dohstcol and 'vormap' in initdat:
        dohstcolvor = 1
        cshst_fov_rb = -2.5 * np.log10(bhst_fov_sm_ns_rb / rhst_fov_sm_ns_rb) + zpbl - zprd
        inan = np.where(np.isfinite(cshst_fov_rb))
        ctnan = len(inan)
        if 'galextcor' in initmaps['hstcol']:
            cshst_fov_rb -= initmaps['hstcol']['galextcor']
        if ctnan > 0: cshst_fov_rb[inan] = bad
   
    hstrd = 0
    hstbl = 0
    hstcol = 0
    hstrdsm = 0
    hstblsm = 0  

    '''''''''''''''''''''''''''''''''''''''''''''''''''''''''''
    ' Fit QSO PSF                                             '  
    '''''''''''''''''''''''''''''''''''''''''''''''''''''''''''
    #Radii in kpc
    #GMOS FOV
    
    #map_x = rebin(np.arange(1, dx), (dx, dy))
    map_x = np.full((dy, dx), np.arange(1, dx + 1))
    #map_y = rebin(np.transpose(np.arange(1, dy)), (dx, dy))
    map_y = np.full((dy, dx), np.transpose([np.arange(1, dy + 1)]))
    map_r = np.sqrt((map_x - center_axes[0])**2.0 + (map_y - center_axes[1])**2.0)
    map_rkpc_ifs = (map_r * kpc_per_pix).value

    #PA E of N, in degrees; spaxel with [0,0] has PA = bad
    '''
    map_pa = np.zeros((dx,dy), dtype = float)
    map_xaxis = map_x - center_axes[0]
    map_yaxis = map_y - center_axes[1]
    map_pa = ifsf_pa(map_xaxis, map_yaxis)
    map_pa += initdat['positionangle']
    iphase = np.where(map_pa > 360.0)
    ctphase = len(iphase)
    if ctphase > 0: map_pa[iphase] -= 360.0
    inuc = np.where(map_r == 0.0)
    ctnuc = len(inuc)
    if ctnuc > 0: map_pa[inuc] = bad
    '''
    
    if 'decompose_qso_fit' in initdat:
        inan = np.where(np.isfinite(contcube['qso_mod']))
        ctnan = len(inan)
        if ctnan > 0: contcube['qso_mod'][inan] = 0.0
        

        qso_map = np.sum(contcube['qso_mod'], axis = 2) / contcube['npts']
        maxqso_map = np.max(qso_map)
        
        #qso_err = stddev(contcube.qso,dim=3,/double)
        #qso_err = sqrt(total(datacubeube.var,3))
        qso_err = np.sqrt(np.median(datacube.var, axis = 2))
        ibd = np.where(~np.isfinite(qso_err))
        ctbd = len(ibd)
        if ctbd > 0:
            qso_err[ibd] = bad
            qso_map[ibd] = 0.0
      
        qso_map /= maxqso_map
        qso_err /= maxqso_map
        
        '''
;      izero = where(qso_map eq 0d,ctzero)
;      maxerr = max(qso_err)
;      lowthresh=1d-4
;      if ctzero gt 0 then begin
;         qso_map[izero] = lowthresh
;         qso_err[izero] = maxerr*100d
;      endif
;      ilow = where(qso_map lt lowthresh,ctlow)
;      if ctlow gt 0 then begin
;         qso_map[ilow] = lowthresh
;         qso_err[ilow] = maxerr*100d
;      endif
      
;      qso_err /= max(median(datacube.dat,dim=3,/double))
        '''
      
        #2D Moffat fit to continuum flux vs. radius
        
        #model parameters:
        amp, x0, y0, gamma, alpha = 0.0, center_nuclei[0] - 1.0, center_nuclei[1], 1.0, 3.0
        yp, xp = qso_map.shape
        y, x, = np.mgrid[:yp, :xp]
        
        #model for data
        moffat_init = models.Moffat2D(amp, x0, y0, gamma, alpha)
        #initialize a fitter
        fit = fitting.LevMarLSQFitter()
        #fit data with fitter
        qso_fit = fit(moffat_init, x, y, qso_map)
        
        qso_fitpar = qso_fit.parameters
        
        map_rnuc = np.sqrt((map_x - qso_fitpar[1] + 1) ** 2.0 + \
                              (map_y - qso_fitpar[2] + 1) ** 2.0)
        map_rnuckpc_ifs = map_rnuc * kpc_per_pix
        psf1d_x = np.arange(101.) / 100.0 * np.max(map_rnuckpc_ifs)
        psf1d_y = np.log10(moffat(psf1d_x, [qso_fitpar[0], 0.0,
                              qso_fitpar[2] * kpc_per_pix.value, qso_fitpar[4]])) #TODO: 2

    '''''''''''''''''''''''''''''''''''''''''''''''''''''''''''
    ' Compute radii and centers                               '  
    '''''''''''''''''''''''''''''''''''''''''''''''''''''''''''

    #coordinates in kpc
    xran_kpc = np.asarray([0 - (center_axes[0] - 0.5), dx - (center_axes[0] - 0.5)], float) \
              * kpc_per_pix
    yran_kpc = np.asarray([0 - (center_axes[1] - 0.5), dy - (center_axes[1] - 0.5)], float) \
              * kpc_per_pix
    center_nuclei_kpc_x = ([c - center_axes[0] for c in center_nuclei[0:]] ) \
                         * kpc_per_pix
    center_nuclei_kpc_y = ([c - center_axes[1] for c in center_nuclei[1:]] ) \
                         * kpc_per_pix
    #in image window
    xwinran_kpc = np.asarray([0 - (center_axes[0] - 0.5 - (plotwin[0] - 1)), \
                         dxwin - (center_axes[0] - 0.5 - (plotwin[0]-1))], float) \
                        * kpc_per_pix
    ywinran_kpc = np.asarray([0 - (center_axes[1] - 0.5 - (plotwin[1] - 1)), \
                         dywin - (center_axes[1] - 0.5 - (plotwin[1] - 1))], float) \
                        * kpc_per_pix
#   center_nuclei_kpc_xwin = (center_nuclei[0,*]-center_axes[0]-(plotwin[0]-1)) $
#                            * kpc_per_pix
#   center_nuclei_kpc_ywin = (center_nuclei[1,*]-center_axes[1]-(plotwin[1]-1)) $
#                            * kpc_per_pix
    center_nuclei_kpc_xwin = center_nuclei_kpc_x
    center_nuclei_kpc_ywin = center_nuclei_kpc_y
   
#   HST FOV
    if (dohstrd or dohstbl):
        if dohstbl: size_subim = bhst_fov.shape
        else: size_subim = rhst_fov_sc.shape
        map_x_hst = rebin(np.arange(size_subim[0]) + 1, (size_subim[0], size_subim[0]))
        map_y_hst = rebin(np.transpose(np.arange(size_subim[0]) + 1),
                        (size_subim[0], size_subim[0]))
#       Locations of [0,0] point on axes and nuclei, in HST pixels (single-offset
#       indices).
        center_axes_hst = (center_axes - 0.5) * float(size_subim[0] / dx) + 0.5
        center_nuclei_hst = (center_nuclei - 0.5) * float(size_subim[0] / dx) + 0.5
#       Radius of each HST pixel from axis [0,0] point, in HST pixels
        map_r_hst = np.sqrt((map_x_hst - center_axes_hst[0]) ^ 2.0 + \
                       (map_y_hst - center_axes_hst[1]) ^ 2.0)
        if dohstbl:
#           Radius of each HST pixel from axis [0,0] point, in kpc
            map_rkpc_hst = map_r_hst * initmaps['hstbl']['platescale'] * kpc_per_as
            hstplatescale = initmaps['hstbl']['platescale']
            kpc_per_hstpix = hstplatescale * kpc_per_as
            if dohstrd:
                if initmaps['hstbl']['platescale'] != initmaps['hstrd']['platescale']:
                    print('WARNING: HST blue and red plate scales differ;')
                    print('         using blue platescale for radius calculations.')
        else:
            map_rkpc_hst = map_r_hst * initmaps['hstrd']['platescale'] * kpc_per_as      
            hstplatescale = initmaps['hstrd']['platescale']
            kpc_per_hstpix = hstplatescale * kpc_per_as
      
        if dohstbl:
            if 'nucoffset' in initmaps['hstbl']:
                #Radius of each blue HST pixel from axis [0,0] point, in HST pixels, 
                #with by-hand offset applied
                map_r_bhst = \
                    np.sqrt((map_x_hst - \
                    (center_axes_hst[0] + initmaps['hstbl']['nucoffset'][0])) ^ 2.0 + \
                    (map_y_hst - \
                    (center_axes_hst[1] + initmaps['hstbl']['nucoffset'][1])) ^ 2.0)
#           ... and now in kpc
                map_rkpc_bhst = map_r_bhst * initmaps['hstbl']['platescale'] * kpc_per_as
            else: map_rkpc_bhst = map_rkpc_hst
        if dohstrd:
            if 'nucoffset' in initmaps['hstbl']:
                map_r_rhst = \
                    np.sqrt((map_x_hst - \
                    (center_axes_hst[0] + initmaps['hstrd']['nucoffset'][0])) ^ 2.0 + \
                    (map_y_hst - \
                    (center_axes_hst[1] + initmaps['hstrd']['nucoffset'][1])) ^ 2.0)
                map_rkpc_rhst = map_r_rhst * initmaps.hstrd.platescale * kpc_per_as
            else: map_rkpc_rhst = map_rkpc_hst 
        else: map_rkpc_hst = 0.0

#TODO: emission lines
    '''
    '''''''''''''''''''''''''''''''''''''''''''''''''''''''''''
    ' Process emission lines                                  '
    '''''''''''''''''''''''''''''''''''''''''''''''''''''''''''

    emlvel = 0.0
    emlflxcor_pp = 0.0
    emlflxcor_med = 0.0
    ebv = 0.0
    ebvmed = 0.0
    errebv = 0.0
    lr = dict()
    lrerrlo = dict()
    lrerrhi = dict()
    elecdenmap = dict()
    elecdenmap_errlo = dict()
    elecdenmap_errhi = dict()
    if 'noemlinfit' not in initdat:
    '''
    '''
;;     Sort emission line components if requested
;      if tag_exist(initmaps,'fcnsortcomp') AND $
;         tag_exist(initmaps,'sortlines') AND $
;         tag_exist(initmaps,'sorttype') then begin
;         if tag_exist(initmaps,'argssortcomp') then $
;            linmaps = call_function(initmaps.fcnsortcomp,dx,dy,linmaps,$
;                                    initdat.linetie,initmaps.sortlines,$
;                                    initmaps.sorttype,$
;                                    _extra=initmaps.argssortcomp) $
;         else $
;            linmaps = call_function(initmaps.fcnsortcomp,dx,dy,linmaps,$
;                                    initdat.linetie,initmaps.sortlines,$
;                                    initmaps.sorttype)
;      endif
        '''

    '''
;;     Set reference line for rotation curve, for defining outflows, by resorting
;;     OUTLINES so that reference line comes first.
;      if ~ tag_exist(initmaps,'diskline') then diskline='Halpha' $
;      else diskline = initmaps.diskline
;      idiskline = outlines.where(diskline,count=ctdiskline)
;      if ctdiskline gt 0 then outlines.move,idiskline,0 $
;      else message,$
;         'Disk line not found; using first line in OUTLINES hash.',/cont
        '''
    '''                            
;      ofpars_line=0b
;      sigthresh=0b
;      ofthresh=0b ; threshold for defining outflow
;      ofignore=0b
;      diffthresh=0b
;      if tag_exist(initmaps,'compof') then begin
;         ofpars_line=1b
;         if tag_exist(initmaps,'compof_sigthresh') then $
;            sigthresh=initmaps.compof_sigthresh
;         if tag_exist(initmaps,'compof_ofthresh') then $
;            ofthresh=initmaps.compof_ofthresh
;         if tag_exist(initmaps,'compof_diffthresh') then $
;            diffthresh=initmaps.compof_diffthresh
;         if tag_exist(initmaps,'compof_ignore') then $
;            ofignore=initmaps.compof_ignore
;      endif
;      if line eq diskline then diskrot=0b $
;      else diskrot=linspecpars[diskline].vpk
;      if tag_exist(initmaps,'compof') then ofpars[line] = ofpars_line
;      endforeach
;      linspecpars_tags = tag_names(linspecpars[outlines[0]])
;      if tag_exist(initmaps,'compof') then $
;         ofpars_tags = tag_names(ofpars[outlines[0]])
        '''
    '''
#     Compute CVDF velocities and fluxes
        if 'cvdf' in initmaps:
            if 'flux_maps' in initmaps['cvdf']:
                fluxvels = initmaps['cvdf']['flux_maps']
            else: fluxvels = 0.0
            if 'sigcut' in initmaps['cvdf']:
                sigcut=initmaps['cvdf']['sigcut']
            else: sigcut = 0.0
            
            
        #Loading in emission line dictionaries from q3da (line 852)  
        emldicts = np.load('{[outdir]}{[label]}'.format(initdat, initdat)+'.lin.npz')  
        emlwav = emldicts['emlwav']
        emlwaverr = emldicts['emlwaverr']
        emlsig = emldicts['emlsig'] 
        emlsigerr = emldicts['emlsigerr']
        emlflx = emldicts['emlflx'] 
        emlflxerr = emldicts['emlflxerr']
        emlweq = emldicts['emlweq']
        
        
        emlcompvel = cmpcompvals(emlwav,emlsig,emlflx,initdat['zsys_gas'], \
                                    emlwaverr = True, emlsigerr = True)
        emlvel = emlcompvel
        if 'nocvdf' not in initdat:
            emlcvdfvel = cmpcvdf(emlcvdf,emlflx,emlflxerr, \
                                       fluxvels = True, sigcut = True)
            emlvel = emlcvdfvel + emlcompvel
        
        #Extinction maps
        #flux summed over components
        if 'ebv' in initmaps:
            #Calculate ...
            if 'calc' in initmaps['ebv']:
                ebv = dict()
                errebv = dict()
                ebvmed = dict()
                for key in initmaps['ebv']['calc']:
                    if 'argslineratios' in initmaps:
                        
                        ;ebvtmp = lineratios(emlflx[key], emlflxerr[key], listlines, \
                                            ebvonly = True, errlo=errtmp, \
                                            extra = initmaps['argslineratios'])
                        
                        ebvtmp = lineratios(emlflx[key], emlflxerr[key], listlines, \
                                            ebvonly = True)
                        errtmp = ebvtmp[1]
                        ebvtmp = ebvtmp[0]
                    else: 
                        ebvtmp =  lineratios(emlflx[key], emlflxerr[key],listlines, \
                                             ebvonly = True)
                        errtmp = ebvtmp[1]
                        ebvtmp = ebvtmp[0]
                    ebv[key] = ebvtmp['ebv']
                    errebv[key] = errtmp['ebv']
                    igdebv = np.where(ebv[key] != bad)
                    ctgdebv = len(igdebv)
                    if ctgdebv > 0: ebvmed[key] = np.median(ebv[key,igdebv])
            
                #... and apply.
                if 'apply' in initmaps['ebv']:
                    ebvkey = 'ftot'
                    emlflxcor_pp = dict()
                    emlflxcor_med = dict()
                    emlflxerrcor_pp = dict()
                    emlflxerrcor_med = dict()
                    emlflxcor_pp['ftot'] = dict()
                    emlflxcor_med['ftot'] = dict()
                    emlflxerrcor_pp['ftot'] = dict()
                    emlflxerrcor_med['ftot'] = dict()
                    for i in range (1, initdat['maxncomp']):
                        stric = 'icomp'
                        emlflxcor_pp['fc' + stric] = hash()
                        emlflxcor_med['fc' + stric] = hash()
                        emlflxerrcor_pp['fc' + stric] = hash()
                        emlflxerrcor_med['fc' + stric] = hash()
                        for line in listlines.keys:
                            emlflxcor_pp['fc'+stric,line] = emlflx['fc'+stric,line]
                            emlflxcor_med['fc'+stric,line] = emlflx['fc'+stric,line]
                            emlflxerrcor_pp['fc'+stric,line] = emlflx['fc'+stric,line]
                            emlflxerrcor_med['fc'+stric,line] = emlflx['fc'+stric,line]
                            flx = emlflx['fc'+stric,line]
                            flxerr = emlflxerr['fc'+stric,line]
                            igdflx = np.where(flx > 0 and flx != bad)
                            ctgdflx = len(igdflx)
                            igdebv = np.where(ebv[ebvkey] != bad)
                            ctgdebv = len(igdebv)
                            if (ctgdflx > 0) and (ctgdebv > 0):
                                ebvuse = ebv[ebvkey]
                                ibadebv = list(set(igdflx) - set(igdebv))
                                ctbadebv = len(ibadebv)
                                if ctbadebv > 0: ebvuse[ibadebv] = ebvmed[ebvkey]
                                emlflxcor_pp['fc'+stric,line,igdflx] = \
                                    dustcor_ccm(listlines[line],flx[igdflx], \
                                            ebvuse[igdflx])
                                emlflxcor_med['fc'+stric,line,igdflx] = \
                                    dustcor_ccm(listlines[line],flx[igdflx], \
                                            ebvmed[ebvkey])
                                #Should really propagate E(B-V) error as well ...
                                emlflxerrcor_pp['fc'+stric,line,igdflx] = \
                                    dustcor_ccm(listlines[line],flxerr[igdflx], \
                                            ebvuse[igdflx])
                                emlflxerrcor_med['fc'+stric,line,igdflx] = \
                                    dustcor_ccm(lislines[line],flxerr[igdflx], \
                                            ebvmed[ebvkey])


         
            else:
                ebv = 0.0
                errebv = 0.0

        #Line ratios
        if 'lr' in initmaps:
            if 'calc' in initmaps['lr']:
                for key in initmaps['lr']['calc']:
                    lrtmp =  lineratios(emlflx[key], emlflxerr[key], listlines, \
                                        lronly = True)
                    errlotmp = lrtmp[1]
                    errhitmp = lrtmp[2]
                    lrtmp = lrtmp[0]
                    lr[key] = dict()
                    lrerrlo[key] = dict()
                    lrerrhi[key] = dict()
                    for lrloop in lrtmp.keys:
                        lr[key, lrloop] = lrtmp[lrloop]
                        lrerrlo[key, lrloop] = errlotmp[lrloop]
                        lrerrhi[key, lrloop] = errhitmp[lrloop]
               


                    #Compute electron densities
                    #Use numbers from Sanders, Shapley, et al. 2015
                    if 's2' in lrtmp.keys:
                        #Densities
                        tmps2map = lrtmp['s2']
                        igds2 = np.where(tmps2map != bad and np.isfinite(tmps2map))
                        ctgd = len(igds2)
                        #ibds2 = where(tmps2map eq bad OR ~ finite(tmps2map),ctgd)
                        tmps2mapgd = 10.0 ^ tmps2map[igds2]
                        tmpdenmap = np.zeros((dx, dy), float) + bad
                        igdden = np.where((tmps2mapgd > s2_minratio) and \
                                 (tmps2mapgd < s2_maxratio))
                        tmpdenmapgd = tmpdenmap[igds2]
                        tmpdenmapgd[igdden] = math.log10((s2_c * tmps2mapgd[igdden] - s2_a * s2_b)/ \
                                               (s2_a - tmps2mapgd[igdden]))
                        ilo = np.where((tmps2mapgd < s2_maxratio) or \
                              (tmpdenmapgd < math.log10(s2_mindensity)) )
                        ctlo = len(ilo)
                        ihi = np.where(tmps2mapgd < s2_minratio or \
                              (tmpdenmapgd > math.log10(s2_maxdensity) and \
                               tmpdenmapgd != bad))
                        cthi = len(ihi)
                        if ctlo > 0: tmpdenmapgd[ilo] = math.log10(s2_mindensity)
                        if cthi > 0: tmpdenmapgd[ihi] = math.log10(s2_maxdensity)
                        tmpdenmap[igds2] = tmpdenmapgd

                        #Density upper limits
                        tmps2maperrlo = errlotmp['s2']
                        tmps2mapgderrlo = tmps2mapgd - \
                                    10.0 ^ (tmps2map[igds2] - tmps2maperrlo[igds2])
                        tmpdenmaperrhi = np.zeros((dx,dy), float) + bad
                        tmpdenmaperrhigd = tmpdenmaperrhi[igds2]
                        tmpdenmaphi = np.zeros((dx,dy), float) + bad
                        tmpdenmaphigd = tmpdenmaphi[igds2]
                        igdden = np.where(tmps2mapgd - tmps2mapgderrlo > s2_minratio and \
                                 tmps2mapgd - tmps2mapgderrlo < s2_maxratio)
                        tmpdenmaphigd[igdden] = \
                            math.log10((s2_c * (tmps2mapgd[igdden]-tmps2mapgderrlo[igdden]) - s2_a * s2_b)/ \
                                       (s2_a - (tmps2mapgd[igdden]-tmps2mapgderrlo[igdden])))
                        tmpdenmaperrhigd[igdden] = tmpdenmaphigd[igdden] - \
                            tmpdenmapgd[igdden]
                        ilo = np.where(tmps2mapgd - tmps2mapgderrlo > s2_maxratio or \
                              tmpdenmaphigd < math.log10(s2_mindensity))
                        ctlo = len(ilo)
                        ihi = np.where((tmps2mapgd - tmps2mapgderrlo < s2_minratio and \
                              tmps2mapgd - tmps2mapgderrlo > 0.0) or \
                              (tmpdenmaphigd > math.log10(s2_maxdensity) and \
                               tmpdenmaphigd != bad))
                        cthi = len(ihi)
                        if ctlo > 0: tmpdenmaperrhigd[ilo] = 0.0
                        if cthi > 0: tmpdenmaperrhigd[ihi] = \
                            math.log10(s2_maxdensity) - tmpdenmapgd[ihi]
                        tmpdenmaperrhi[igds2] = tmpdenmaperrhigd

                        #Density lower limits
                        tmps2maperrhi = errhitmp['s2']
                        tmps2mapgderrhi = 10.0 ^ (tmps2map[igds2] + tmps2maperrhi[igds2]) - \
                                         tmps2mapgd
                        tmpdenmaperrlo = np.zeros((dx,dy), float) + bad
                        tmpdenmaperrlogd = tmpdenmaperrlo[igds2]
                        tmpdenmaplo = np.zeros((dx,dy), float) + bad
                        tmpdenmaplogd = tmpdenmaplo[igds2]
                        igdden = np.where(tmps2mapgd + tmps2mapgderrhi > s2_minratio and \
                                 tmps2mapgd + tmps2mapgderrhi < s2_maxratio)
                        tmpdenmaplogd[igdden] = \
                            math.log10((s2_c * (tmps2mapgd[igdden]+tmps2mapgderrhi[igdden]) - s2_a * s2_b)/ \
                            (s2_a - (tmps2mapgd[igdden] + tmps2mapgderrhi[igdden])))
                        tmpdenmaperrlogd[igdden] = tmpdenmapgd[igdden] - \
                            tmpdenmaplogd[igdden]
                        ilo = np.where(tmps2mapgd + tmps2mapgderrhi > s2_maxratio or \
                            tmpdenmaplogd < math.log10(s2_mindensity))
                        ctlo = len(ilo)
                        ihi = np.where((tmps2mapgd + tmps2mapgderrhi < s2_minratio and \
                              tmps2mapgd + tmps2mapgderrhi > 0.0) or \
                              (tmpdenmaplogd > math.log10(s2_maxdensity) and \
                               tmpdenmaplogd != bad))
                        cthi = len(ihi)
                        if ctlo > 0: tmpdenmaperrlogd[ilo] = \
                            tmpdenmapgd[ilo] - math.log10(s2_mindensity)
                        if cthi > 0: tmpdenmaperrlogd[ihi] = 0.0
                        tmpdenmaperrlo[igds2] = tmpdenmaperrlogd

                        elecdenmap[key] = tmpdenmap
                        elecdenmap_errlo[key] = tmpdenmaperrlo
                        elecdenmap_errhi[key] = tmpdenmaperrhi

    else:
        emlflx = 0.0


    if 'donad' in initdat:
        #Apply S/N cut. Funny logic is to avoid loops.
        #TODO: working w an xdr file
        nadmap = nadfit.weqabs[:, :, 0]
        maperravg = (nadfit.weqabserr[:, :,0] + nadfit.weqabserr[:, :, 1]) / 2.0
        mask1 = np.zeros((dx,dy), float)
        mask2 = np.zeros((dx,dy), float)
        igd = np.where(nadmap > 0.0 and \
                    nadmap != bad and \
                    nadmap > initmaps['nadabsweq_snrthresh'] * maperravg)
        ibd = np.where(map == 0.0 or \
                    map == bad or \
                    map < initmaps['nadabsweq_snrthresh'] * maperravg)
        mask1[igd] = 1.0
        mask2[ibd] = bad
        rbmask1 = rebin(mask1, dx, dy, initnad['maxncomp'])
        rbmask2 = rebin(mask2, dx, dy, initnad['maxncomp'])
        nadfit.waveabs *= rbmask1
        nadfit.waveabserr *= rbmask1
        nadfit.sigmaabs *= rbmask1
        nadfit.sigmaabserr *= rbmask1
        nadfit.tau *= rbmask1
        nadfit.tauerr *= rbmask1
        nadfit.waveabs += rbmask2
        nadfit.waveabserr += rbmask2
        nadfit.sigmaabs += rbmask2
        nadfit.sigmaabserr += rbmask2
        nadfit.tau += rbmask2
        nadfit.tauerr += rbmask2


        #Compute velocities and column densities of NaD model fits
        nadabscftau = np.zeros((dx, dy, maxnadabscomp + 1), float) + bad
        errnadabscftau = np.zeros((dx, dy, maxnadabscomp + 1, 2), float) + bad
        nadabstau = np.zeros((dx, dy, maxnadabscomp + 1), float) + bad
        errnadabstau = np.zeros((dx, dy, maxnadabscomp + 1, 2), float) + bad
        nadabscf = np.zeros((dx, dy, maxnadabscomp + 1), float) + bad
        nadabsvel = np.zeros((dx, dy, maxnadabscomp + 1), float) + bad
        errnadabsvel = np.zeros((dx, dy, maxnadabscomp + 1, 2), float) + bad
        nademvel = np.zeros((dx, dy, maxnademcomp + 1), float) + bad
        errnademvel = np.zeros((dx, dy, maxnademcomp + 1), float) + bad
        nadabssig = np.zeros((dx, dy, maxnadabscomp + 1), float) + bad
        errnadabssig = np.zeros((dx, dy, maxnadabscomp + 1, 2), float) + bad
        nademsig = np.zeros((dx, dy, maxnademcomp + 1), float) + bad
        errnademsig = np.zeros((dx, dy, maxnademcomp + 1), float) + bad
        nadabsv98 = np.zeros((dx, dy, maxnadabscomp + 1), float) + bad
        #errnadabsv98 = dblarr(dx,dy,maxnadabscomp+1,2)+bad
        nademv98 = np.zeros((dx, dy, maxnademcomp + 1), float) + bad
        #errnademv98 = dblarr(dx,dy,maxnademcomp+1,2)+bad
        nadabsnh = np.zeros((dx, dy), float) + bad
        #llnadabsnh = bytarr(dx,dy)
        llnadabsnh = np.array(bytearray(dx * dy), dtype=np.byte)
        llnadabsnh.shape(dx, dy)
        
        errnadabsnh = np.zeros((dx, dy, 2), float) + bad
        nadabslnnai = np.zeros((dx,dy), float) + bad
        
        llnadabslnnai = np.array(bytearray(dx * dy), dtype=np.byte)
        llnadabslnnai.shape(dx, dy)
        
        errnadabslnnai = np.zeros((dx, dy, 2), float) + bad
        nadabsnhcf = np.zeros((dx, dy), float) + bad
        errnadabsnhcf = np.zeros((dx, dy, 2), float) + bad
        nadabscnhcf = np.zeros((dx, dy, maxnadabscomp + 1), float) + bad
        errnadabscnhcf = np.zeros((dx, dy, maxnadabscomp + 1, 2), float) + bad
        nadabsncomp = np.zeros((dx, dy), dtype = int) + bad
        nademncomp = np.zeros((dx,dy), dtype = int) + bad
        for i in range (0, dx - 1):
            for j in range (0, dy - 1):
                igd = np.where(nadfit.waveabs[i, j, :] != bad and \
                        nadfit.waveabs[i, j, :] != 0)
                ctgd = len(igd)
                if ctgd > 0:
                    tmpcftau=nadfit.cf[i,j,igd] * nadfit.tau[i,j,igd]
                    tmpcf=nadfit.cf[i,j,igd]
                    tmptau=nadfit.tau[i,j,igd]
                    tmpcftauerr = nadfit.cferr[i,j,igd,:]
                    tmpltauerrlo = nadfit.tauerr[i,j,igd,0]     #errors are in log(tau) space
                    tmpltauerrhi = nadfit.tauerr[i,j,igd,1]     #errors are in log(tau) space
                    tmpwaveabs = nadfit.waveabs[i,j,igd]
                    tmpwaveabserr = nadfit.waveabserr[i,j,igd,:]
                    tmpsigabs=nadfit.sigmaabs[i,j,igd]
                    tmpsigabserr = nadfit.sigmaabserr[i,j,igd,:]
                    ineg = np.where(tmpcftau[0,0,igd] > 0.0 and \
                            tmpcftau[0,0,igd] < tmpcftauerr[0,0,igd,0])
                    ctneg = len(ineg)
                    if ctneg > 0: tmpcftau[0,0,igd,0] = 0.0
                    nnai = sum(tmptau * nadfit.sigmaabs[i,j,igd]) / \
                            (1.497 - 15 / np.sqrt(2.0) * nadlinelist['NaD1'] * 0.3180)
                    nadabslnnai[i,j] = math.log10(nnai)
                    nnaicf = sum(tmpcftau*nadfit.sigmaabs[i,j,igd] * \
                        nadfit.cf[i,j,igd]) / \
                        (1.497 - 15 / np.sqrt(2.0) * nadlinelist['NaD1'] * 0.3180)
                    tmptauerrlo = tmptau - 10.0 ^ (math.log10(tmptau) - tmpltauerrlo)
                    tmptauerrhi = 10.0 ^ (math.log10(tmptau) + tmpltauerrhi) - tmptau
                    #get saturated points
                    isat = np.where(tmptau == taumax)
                    ctsat = len(isat)
                    if ctsat > 0: tmptauerrhi[isat] = 0.0
                    nnaierrlo = \
                        nnai * np.sqrt(sum((tmptauerrlo / tmptau) ^ 2.0 + \
                                  (nadfit.sigmaabserr[i,j,igd,0]/ \
                                   nadfit.sigmaabs[i,j,igd])^2.0))
                    nnaierrhi = \
                        nnai * np.sqrt(sum((tmptauerrhi / tmptau) ^ 2.0 + \
                                  (nadfit.sigmaabserr[i,j,igd,1]/ \
                                   nadfit.sigmaabs[i,j,igd])^2.0))
                    nnaicferr = \
                        [nnaicf, nnaicf] * np.sqrt(sum((nadfit.cferr[i,j,igd,:]/ \
                        rebin(tmpcftau,ctgd,2))^2.0 + \
                        (nadfit.sigmaabserr[i,j,igd,:]/ \
                        rebin(nadfit.sigmaabs[i,j,igd],ctgd,2))^2.0,3))
                    nadabsnh[i,j] = nnai/(1-ionfrac)/10^(naabund - nadep)
                    nadabsnhcf[i,j] = nnaicf/(1-ionfrac)/10^(naabund - nadep)
                    errnadabsnh[i,j,:] = [nnaierrlo,nnaierrhi]/ \
                                    (1-ionfrac)/10^(naabund - nadep)
                    errnadabsnhcf[i,j,:] = \
                        nadabsnhcf[i,j] * np.sqrt((nnaicferr/nnaicf)^2.0 + \
                                       (oneminusionfrac_relerr)^2.0)
                    nadabsncomp[i,j] = ctgd
                    if ctsat > 0:
                        llnadabsnh[i,j] = bytes(1)
                        errnadabsnh[i,j,1] = 0.0
                        llnadabslnnai[i,j] = bytes(1)
                        errnadabslnnai[i,j,1] = 0.0
               
                
;           Sort absorption line wavelengths. In output arrays,
;           first element of third dimension holds data for spaxels with only
;           1 component. Next elements hold velocities for spaxels with more than
;           1 comp, in order of increasing blueshift. Formula for computing error
;           in velocity results from computing derivative in Wolfram Alpha w.r.t.
;           lambda and rearranging on paper.
                
                if ctgd == 1:
                    
;              Set low sigma error equal to best-fit minus 5 km/s 
;              (unlikely that it's actually lower than this!) 
;              if low is greater than best-fit value
                    
                    
                    if tmpsigabs < tmpsigabserr[0]: tmpsigabserr[0] = tmpsigabs-5.0
                    errnadabslnnai[i,j,0] = \
                        np.sqrt(tmpltauerrlo^2.0 + \
                       (math.log10(tmpsigabs)-math.log10(tmpsigabs-tmpsigabserr[0]))^2.0)
                    errnadabslnnai[i,j,1] = \
                        np.sqrt(tmpltauerrhi^2.0 + \
                       (math.log10(tmpsigabs+tmpsigabserr[1])-math.log10(tmpsigabs))^2.0)
                    nadabscnhcf[i,j,0] = nadabsnhcf[i,j]
                    errnadabscnhcf[i,j,0,:]=errnadabsnhcf[i,j,:]
                    zdiff = \
                        tmpwaveabs/(nadlinelist['NaD1']*(1.0 + initdat['zsys_gas'])) - 1.0
                    nadabsvel[i,j,0] = c_kms * ((zdiff+1.0)^2.0 - 1.0) / \
                                          ((zdiff+1.0)^2.0 + 1.0)
                    errnadabsvel[i,j,0,:] = \
                        c_kms * (4.0/(nadlinelist['NaD1']*(1.0 + initdat['zsys_gas']))* \
                           ([zdiff,zdiff]+1.0)/(([zdiff,zdiff]+1.0)^2.0 + 1.0)^2.0) * \
                           tmpwaveabserr
                    nadabssig[i,j,0] = tmpsigabs
                    errnadabssig[i,j,0,:] = tmpsigabserr
                    nadabsv98[i,j,0] = nadabsvel[i,j,0]-2.0*nadabssig[i,j,0]
                    #errnadabsv98[i,j,0] = $
                    #sqrt(errnadabsvel[i,j,0]^2d + 4d*errnadabssig[i,j,0]^2d)
                    nadabscftau[i,j,0] = tmpcftau
                    errnadabscftau[i,j,0,""] = tmpcftauerr
                    nadabstau[i,j,0] = tmptau
                    nadabscf[i,j,0] = tmpcf
                    errnadabstau[i,j,0,:] = [tmptauerrlo,tmptauerrhi]
                elif ctgd > 1:
                    errnadabslnnai[i,j,0] = 0.0
                    errnadabslnnai[i,j,1] = 0.0
                    for k in range (0, ctgd-1):
                        if tmpsigabs[0,0,k] < tmpsigabserr[0,0,k,0]:
                            tmpsigabserr[0,0,k,0] = tmpsigabs[0,0,k] - 5.0
                        errnadabslnnai[i,j,0] += \
                            tmpltauerrlo[0,0,k]^2.0 + \
                            (math.log10(tmpsigabs[0,0,k])- \
                             math.log10(tmpsigabs[0,0,k]-tmpsigabserr[0,0,k,0]))^2.0
                        errnadabslnnai[i,j,1] += \
                            tmpltauerrhi[0,0,k]^2.0 + \
                            (math.log10(tmpsigabs[0,0,k]+tmpsigabserr[0,0,k,1])- \
                             math.log10(tmpsigabs[0,0,k]))^2.0
                    errnadabslnnai[i,j,:] = np.sqrt(errnadabslnnai[i,j,:])
                    sortgd = np.argsort(tmpwaveabs)
                    #nnai = reverse(tmptau[sortgd]*tmpsigabs[sortgd]) / $
                        #(1.497d-15/sqrt(2d)*nadlinelist['NaD1']*0.3180d)
                    
                    nnaicf = (tmpcftau[sortgd]*tmpsigabs[sortgd])[::-1] / \
                        (1.497-15/np.sqrt(2.0)*nadlinelist['NaD1']*0.3180)
                    #nadabscnh[i,j,1:ctgd] = nnai/(1-ionfrac)/10^(naabund - nadep)
                    nadabscnhcf[i,j,1:ctgd] = nnaicf/(1-ionfrac)/10^(naabund - nadep)
                    
;               nnaierrlo = $
;                  nnai*sqrt(reverse((tmptauerrlo[sortgd]/tmptau[sortgd])^2d + $
;                                    (tmpsigabserr[sortgd]/tmpsigabs[sortgd])^2d))
;               nnaierrhi = $
;                  nnai*sqrt(reverse((tmptauerrhi[sortgd]/tmptau[sortgd])^2d + $
;                                    (tmpsigabserr[sortgd]/tmpsigabs[sortgd])^2d))
                    
                    nnaicferr = \
                        rebin(nnaicf,1,1,ctgd,2) * \
                        np.sqrt(((tmpcftauerr[0,0,sortgd,:]/ \
                        rebin(tmpcftau[sortgd],1,1,ctgd,2))^2.0 + \
                        (tmpsigabserr[0,0,sortgd,:]/ \
                        rebin(tmpsigabs[sortgd],1,1,ctgd,2))^2.0,3)[::-1])
                    
;               nnaicferrhi = $
;                  nnaicf*sqrt(reverse((tmpcftauerrhi[sortgd]/tmpcftau[sortgd])^2d + $
;                                    (tmpsigabserr[sortgd]/tmpsigabs[sortgd])^2d))
;               errnadabscnh[i,j,1:ctgd,0] = nnaierrlo / $
;                                            (1-ionfrac)/10^(naabund - nadep)
;               errnadabscnh[i,j,1:ctgd,1] = nnaierrhi / $
;                                            (1-ionfrac)/10^(naabund - nadep)
;               errnadabscnhcf[i,j,1:ctgd,*] = nnaicferr / $
;                  (1-ionfrac)/10^(naabund - nadep)
                    
                    errnadabscnhcf[i,j,1:ctgd,:] = \
                        rebin(nadabscnhcf[i,j,1:ctgd],1,1,ctgd,2)* \
                        np.sqrt((nnaicferr/rebin(nnaicf,1,1,ctgd,2))^2.0 + \
                        (rebin(oneminusionfrac_relerr,1,1,ctgd,2))^2.0)
                    zdiff = (tmpwaveabs[sortgd])[::-1]/ \
                       (nadlinelist['NaD1']*(1.0 + initdat['zsys_gas'])) - 1.0
                    nadabsvel[i,j,1:ctgd] = c_kms * ((zdiff+1.0)^2.0 - 1.0) / \
                        ((zdiff+1.0)^2.0 + 1.0)
                    errnadabsvel[i,j,1:ctgd,:] = \
                        c_kms * (4.0/(nadlinelist['NaD1']*(1.0 + initdat['zsys_gas']))* \
                           (rebin(zdiff,1,1,ctgd,2) + 1.0)/ \
                           ((rebin(zdiff,1,1,ctgd,2)+1.0)^2.0 + 1.0)^2.0) * \
                           (tmpwaveabserr[0,0,sortgd,:],3)[::-1]
                    nadabssig[i,j,1:ctgd] = (tmpsigabs[sortgd])[::-1]
                    errnadabssig[i,j,1:ctgd,:] = (tmpsigabserr[0,0,sortgd,:],3)[::-1]
                    nadabsv98[i,j,1:ctgd] = nadabsvel[i,j,1:ctgd]- \
                                       2.0*nadabssig[i,j,1:ctgd]
                    
;               errnadabsv98[i,j,1:ctgd] = $
;                  sqrt(errnadabsvel[i,j,1:ctgd]^2d + $
;                  4d*errnadabssig[i,j,1:ctgd]^2d)
                    
                    nadabscftau[i,j,1:ctgd] = (tmpcftau[sortgd])[::-1]
                    errnadabscftau[i,j,1:ctgd,:] = (tmpcftauerr[0,0,sortgd,:],3)[::-1]
                    nadabstau[i,j,1:ctgd] = (tmptau[sortgd])[::-1]
                    errnadabstau[i,j,1:ctgd,0] = (tmptauerrlo[sortgd])[::-1]
                    errnadabstau[i,j,1:ctgd,1] = (tmptauerrhi[sortgd])[::-1]
                    nadabscf[i,j,1:ctgd] = (tmpcf[sortgd])[::-1]
            
                
;           Sort emission line wavelengths. In output velocity array,
;           first element of third dimension holds data for spaxels with only
;           1 component. Next elements hold velocities for spaxels with more than
;           1 comp, in order of increasing redshift.
                
                igd = np.where(nadfit.waveem[i,j,:] != bad and 
                        nadfit.waveem[i,j,:] != 0)
                ctgd = len(igd)
                if ctgd > 0:
                    nademncomp[i,j] = ctgd
                    tmpwaveem = nadfit.waveem[i,j,igd]
                    tmpwaveemerr = np.mean(nadfit.waveemerr[i,j,:,:], 4)
                    tmpwaveemerr = tmpwaveemerr[igd]
                    tmpsigem = nadfit.sigmaem[i,j,igd]
                    tmpsigemerr = np.mean(nadfit.sigmaemerr[i,j,:,:])#dim=4
                    tmpsigemerr = tmpsigemerr[igd]
            
                if ctgd == 1:
                    zdiff = tmpwaveem/ \
                        (nadlinelist['NaD1']*(1.0 + initdat['zsys_gas'])) - 1.0
                    nademvel[i,j,0] = c_kms * ((zdiff+1.0)^2.0 - 1.0) / \
                        ((zdiff+1.0)^2.0 + 1.0)
                    errnademvel[i,j,0] = \
                        c_kms * (4.0/(nadlinelist['NaD1']*(1.0 + initdat['zsys_gas']))* \
                        (zdiff+1.0)/((zdiff+1.0)^2.0 + 1.0)^2.0) * tmpwaveemerr
                    nademsig[i,j,0] = tmpsigem
                    errnadabssig[i,j,0] = tmpsigemerr
                    nademv98[i,j,0] = nademvel[i,j,0]+2.0*nademsig[i,j,0]
                    #errnademv98[i,j,0] = $
                        #sqrt(errnademvel[i,j,0]^2d + 4d*errnademsig[i,j,0]^2d)
                elif ctgd > 1:
                    sortgd = np.argsort(tmpwaveem)
                    zdiff = tmpwaveem[sortgd]/ \
                        (nadlinelist['NaD1']*(1.0 + initdat['zsys_gas'])) - 1.0
                    nademvel[i,j,1:ctgd] = c_kms * ((zdiff+1.0)^2.0 - 1.0) / \
                        ((zdiff+1.0)^2.0 + 1.0)               
                    errnademvel[i,j,1:ctgd] = \
                        c_kms * (4.0/(nadlinelist['NaD1']*(1.0 + initdat['zsys_gas']))* \
                        (zdiff+1.0)/((zdiff+1.0)^2.0 + 1.0)^2.0) * \
                        tmpwaveemerr[sortgd]
                    nademsig[i,j,1:ctgd] = tmpsigem[sortgd]
                    errnademsig[i,j,1:ctgd] = tmpsigemerr[sortgd]
                    nademv98[i,j,1:ctgd] = nademvel[i,j,1:ctgd]+2.0*nademsig[i,j,1:ctgd]
                    
;               errnademv98[i,j,1:ctgd] = $
;                  sqrt(errnademvel[i,j,1:ctgd]^2d + $
;                  4d*errnademsig[i,j,1:ctgd]^2d)                 
                        

        #Parse actual numbers of NaD components
        ionecomp = np.where(nadabsncomp == 1)
        ctonecomp = len(ionecomp)
        if ctonecomp > 0: donadabsonecomp = bytes(1) 
        else: donadabsonecomp = bytes(0)      
        imulticomp = np.where(nadabsncomp > 1 and nadabsncomp != bad)
        ctmulticomp = len(imulticomp)
        if ctmulticomp > 0: donadabsmulticomp = bytes(1) 
        else: donadabsmulticomp = bytes(0)
        igd_tmp = np.where(nadabsncomp != bad)
        ctgd_tmp = len(igd_tmp)
        if ctgd_tmp > 0: maxnadabsncomp_act = np.max(nadabsncomp[igd_tmp])
        else: maxnadabsncomp_act = 0

        ionecomp = np.where(nademncomp == 1)
        ctonecomp = len(ionecomp)
        if ctonecomp > 0: donademonecomp = bytes(1) 
        else: donademonecomp = bytes(0)   
        imulticomp = np.where(nademncomp > 1 and nademncomp != bad)
        ctmulticomp = len(imulticomp)
        if ctmulticomp > 0: donademmulticomp = bytes(1) 
        else: donademmulticomp = bytes(0)     
        igd_tmp = np.where(nademncomp != bad)
        ctgd_tmp = len(igd_tmp)
        if ctgd_tmp > 0: maxnademncomp_act = np.max(nademncomp[igd_tmp])
        else: maxnademncomp_act = 0

        #Absorption lines: Cumulative velocity distribution functions
        nadabscvdf = \
            cmpcvdf_abs(nadfit.waveabs,
                          np.mean(nadfit.waveabserr), #TODO: dim=4
                          nadfit.sigmaabs,
                          np.mean(nadfit.sigmaabserr), #dim=4
                          nadfit.tau,
                          np.mean(nadfit.tauerr), #dim=4
                          initnad['maxncomp'], nadlinelist['NaD1'],
                          initdat['zsys_gas'])
        nadabscvdfvals = cmpcvdfvals_abs(nadabscvdf)

        #Emission lines: Cumulative velocity distribution functions
        nademcvdf = \
            cmpcvdf_abs(nadfit.waveem,
                          np.mean(nadfit.waveemerr), #dim=4
                          nadfit.sigmaem,
                          np.mean(nadfit.sigmaemerr), #dim=4
                          nadfit.flux,
                          np.mean(nadfit.fluxerr), #dim=4
                          initnad['maxncomp'], nadlinelist['NaD1'],
                          initdat['zsys_gas'])
        nademcvdfvals = cmpcvdfvals_abs(nademcvdf)


    else:
        nadabsvel = bytes(0)
        nadabsv98 = bytes(0)
        ibd_nadabs_fitweq = bytes(0)
        igd_nadabs_fitweq = bytes(0)

    #Compass rose
    angarr_rad = initdat['positionangle']* math.pi/180.0
    sinangarr = np.sin(angarr_rad)
    cosangarr = np.cos(angarr_rad)
    #starting point and length of compass rose normalized to plot panel
    xarr0_norm = 0.95
    yarr0_norm = 0.05
    rarr_norm = 0.2
    rlaboff_norm = 0.05
    laboff_norm = 0.0
    carr = 'White'
    
    
;  Coordinates in kpc for arrow coordinates:
;  Element 1: starting point
;  2: end of N arrow
;  3: end of E arrow
;  4: N label
;  5: E label
    
    
    xarr_kpc = np.zeros(5, float)
    yarr_kpc = np.zeros(5, float)
    #average panel dimension
    pdim = ((xran_kpc[1]-xran_kpc[0]) + (yran_kpc[1]-yran_kpc[0]))/2.0
    xarr_kpc[0] = xarr0_norm * (xran_kpc[1]-xran_kpc[0]) + xran_kpc[0]
    xarr_kpc[1] = xarr_kpc[0] + rarr_norm*pdim*sinangarr
    xarr_kpc[2] = xarr_kpc[0] - rarr_norm*pdim*cosangarr
    xarr_kpc[3] = xarr_kpc[0] + (rarr_norm+rlaboff_norm)*pdim*sinangarr
    xarr_kpc[4] = xarr_kpc[0] - (rarr_norm+rlaboff_norm)*pdim*cosangarr
    yarr_kpc[0] = yarr0_norm * (yran_kpc[1]-yran_kpc[0]) + yran_kpc[0]
    yarr_kpc[1] = yarr_kpc[0] + rarr_norm*pdim*cosangarr
    yarr_kpc[2] = yarr_kpc[0] + rarr_norm*pdim*sinangarr
    yarr_kpc[3] = yarr_kpc[0] + (rarr_norm+rlaboff_norm)*pdim*cosangarr
    yarr_kpc[4] = yarr_kpc[0] + (rarr_norm+rlaboff_norm)*pdim*sinangarr

    minyarr_kpc = np.min(yarr_kpc)
    if minyarr_kpc < yran_kpc[0]: yarr_kpc -= minyarr_kpc - yran_kpc[0]
    maxxarr_kpc = np.max(xarr_kpc)
    if maxxarr_kpc > xran_kpc[1]: xarr_kpc -= maxxarr_kpc - xran_kpc[1]
    
    #Compute coordinates for disk axis lines
    diskaxes_endpoints = bytes(0)
    if 'plotdiskaxes' in initmaps:
        diskaxes_endpoints = np.zeros((2,2,2), float)
        for i in range (0, 1):
            halflength=initmaps['plotdiskaxes']['length'][i]/2.0
            sinangle_tmp = np.sin(initmaps['plotdiskaxes']['angle'][i]*math.pi/180.0)
            cosangle_tmp = np.cos(initmaps['plotdiskaxes']['angle'][i]*math.pi/180.0)
            xends = initmaps['plotdiskaxes']['xcenter'][i]+[halflength*sinangle_tmp, \
                -halflength*sinangle_tmp]
            yends = initmaps['plotdiskaxes']['ycenter'][i]+[-halflength*cosangle_tmp, \
                 halflength*cosangle_tmp]
            diskaxes_endpoints[:, :, i] = [[xends],[yends]]
 
    
;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;
; Fit PSF to Emission Line Map
;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;

;   if tag_exist(initmaps,'fit_empsf') then begin
;      linmap_tmp = $
;         emlflx[string(initmaps.fit_empsf.vel,format='(I0)'),$
;                initmaps.fit_empsf.line]
;      vel = initmaps.fit_empsf.vel
;      ivel = value_locate(linmap_tmp.vel,vel)
;      empsf_map = linmap_tmp.flux[*,*,ivel]
;      maxempsf_map = max(empsf_map,imax)
;      empsf_map /= maxempsf_map
;
;      ;     Use error in total flux for error in line
;      empsf_err = tlinmaps[initmaps.fit_empsf.line,*,*,0,1]
;      empsf_err /= empsf_err[imax]
;
;      ;     2D Moffat fit to continuum flux vs. radius
;      parinfo = REPLICATE({fixed:0b},8)
;      parinfo[0].fixed = 1b
;      est=[0d,1d,1d,1d,center_nuclei[0]-1d,center_nuclei[1]-1d,0d,2.5d]
;      empsf_fit = $
;         mpfit2dpeak(empsf_map,empsf_fitpar,/moffat,/circular,est=est,$
;         parinfo=parinfo,error=empsf_err)
;
;      map_rempsf = sqrt((map_x - empsf_fitpar[4]+1)^2d + $
;         (map_y - empsf_fitpar[5]+1)^2d)
;      map_rempsfkpc_ifs = map_rempsf * kpc_per_pix
;      empsf1d_x = dindgen(101)/100d*max(map_rempsfkpc_ifs)
;      empsf1d_y = alog10(moffat(empsf1d_x,[empsf_fitpar[1],0d,$
;         empsf_fitpar[2]*kpc_per_pix,$
;         empsf_fitpar[7]]))
;
;   endif
    '''
    
    '''''''''''''''''''''''''''''''''''''''''''''''''''''''''''
    ' Continuum plots                                         '
    '''''''''''''''''''''''''''''''''''''''''''''''''''''''''''

    if 'ct' in initmaps:
        if dohst: ctsumrange_tmp = initmaps['ct']['sumrange_hstcomp']
        else: ctsumrange_tmp = initmaps['ct']['sumrange']
        
        capifs = str(ctsumrange_tmp[0]) + '-' + str(ctsumrange_tmp[1])
        if 'sumrange_lab' in initmaps['ct']:
            if initmaps['ct']['sumrange_lab'] == 'microns':
                capifs = str(ctsumrange_tmp[0]/1e4 + '-' + ctsumrange_tmp[1]/1e4)
      
        if 'charscale' in initmaps['ct']: charscale = initmaps['ct']['charscale']
        else: charscale = 1.0
   
    if dohst:
        if 'source' in initmaps['hst']: capsource = initmaps['hst']['source']
        else: capsource = 'HST'
        if dohstrd and dohstbl:
            caphst = str(initmaps['hstbl']['label']+'+'+initmaps['hstrd']['label']) #not sure abt special characters
        elif dohstrd:
            caphst = str(initmaps['hstrd']['label'])
        else:
            caphst = str(initmaps['hstbl']['label'])
   
    #arrays for positions for zoom box
    #posbox1x = np.zeros(2, float)
    #posbox1y = np.zeros(2, float)
    #posbox2x = np.zeros(2, float)
    #posbox2y = np.zeros(2, float)
   
    #Figure out correct image size in inches
    ysize_in = 2.2
    aspectrat_fov = float(dx)/float(dy)
    npanels_ifsfov = 0
    if 'ct' in initmaps: npanels_ifsfov = 1.0
    if dohst: npanels_ifsfov += 1.0
    if dohstsm: npanels_ifsfov += 1.0
    if npanels_ifsfov == 0:
        print('MAKEMAPS: Error -- no continuum images to plot.')
        exit #maybe
   
    imgheight_in = 1.6
    xmargin_in = 0.4
    ymargin_in = (ysize_in - imgheight_in)/2.0
    ifsimg_width_in = imgheight_in*aspectrat_fov*npanels_ifsfov
    #Sizes and positions of image windows in real and normalized coordinates
    if dohst:
        xsize_in = imgheight_in + xmargin_in + ifsimg_width_in
        xfrac_margin = xmargin_in / xsize_in
        xfrac_hstbig = imgheight_in / xsize_in
        xfrac_ifsfov_width = imgheight_in*aspectrat_fov / xsize_in
        yfracb = ymargin_in/ysize_in
        yfract = 1.0 - ymargin_in/ysize_in
        pos_hstbig = [0.0,yfracb,xfrac_hstbig,yfract]
        pos_ifsfov = np.zeros((4, int(npanels_ifsfov)), float)
        pos_ifsfov[:,0] = [xfrac_hstbig+xfrac_margin,yfracb,
                         xfrac_hstbig+xfrac_margin+xfrac_ifsfov_width,yfract]
        for i in range (1, int(npanels_ifsfov) - 1):
            pos_ifsfov[:,i] = pos_ifsfov[:,i-1] + [xfrac_ifsfov_width,0.0,
                                                   xfrac_ifsfov_width,0.0]
        #Instrument labels
        lineoff = 0.1 * xfrac_hstbig
        xhstline = [pos_hstbig[0]+lineoff,
                    pos_ifsfov[2,npanels_ifsfov-2]-lineoff]
        yhstline = [yfracb*0.75,yfracb*0.75]
        xhstline_tpos = (xhstline[1]+xhstline[0])/2.0
        yhstline_tpos = yfracb*0.15
        xifsline = [pos_ifsfov[0,npanels_ifsfov-1]+lineoff,
                    pos_ifsfov[2,npanels_ifsfov-1]-lineoff]
        yifsline = [yfracb*0.75,yfracb*0.75]
        xifsline_tpos = (xifsline[1]+xifsline[0])/2.0
        yifsline_tpos = yfracb*0.15
    else:
        ysize_in = 2.2 + ymargin_in
        xsize_in = xmargin_in + ifsimg_width_in
        yfracb = ymargin_in/ysize_in
        yfract = 1.0 - ymargin_in*2.0/ysize_in
        xfrac_margin = xmargin_in/xsize_in
        yfrac_margin = ymargin_in/ysize_in
        pos_ifsfov = [xfrac_margin,yfracb,1.0,yfract]
        #Instrument labels
        lineoff = 0.1*ifsimg_width_in/xsize_in
        xifsline = [pos_ifsfov[0]+lineoff,
                    pos_ifsfov[2]-lineoff]
        yifsline = [yfracb*0.75,yfracb*0.75]
        xifsline_tpos = (xifsline[1]+xifsline[0])/2.0
        yifsline_tpos = yfracb*0.15

    #start of plotting
    #cgps_open(initdat['mapdir']+initdat['label']+'cont.eps')
    
    plt.style.use('seaborn-white')
    if npanels_ifsfov == 1: #if only one plot
        contfig = plt.figure(figsize=(10, 10))
    else:
        contfig = plt.figure(figsize=(npanels_ifsfov * 4, npanels_ifsfov)) 
    #plt.axis("off")
    if dohst:
        
        plt.text(xhstline_tpos, yhstline_tpos, str(capsource + ': ' + caphst), c = 'red') #ie HST:ACS/F625W
        plt.text(0, .6, initdat['name']) #ie PG1411 + 442   
        #HST continuum, large scale
        if dohstbl: size_subim = np.shape(bhst_big)
        else: size_subim = np.shape(rhst_big)
        if dohstrd and dohstbl:
            mapscl = np.zeros((3,size_subim[0],size_subim[1]), bytes)
            mapscl[0,:,:] = rhst_big
            mapscl[2,:,:] = bhst_big
            mapscl[1,:,:] = bytes((float(rhst_big)+float(bhst_big))/2.0)
            if size_subim[0] < resampthresh or size_subim[1] < resampthresh:
                mapscl = rebin(mapscl, (3,size_subim[0]*samplefac,size_subim[1]*samplefac))
        else:
            mapscl = np.zeros((size_subim[0],size_subim[1]), bytes)
            if dohstrd: mapscl = rhst_big
            if dohstbl: mapscl = bhst_big
            if size_subim[0] < resampthresh or size_subim[1] < resampthresh:
                mapscl = rebin(mapscl, (size_subim[0]*samplefac,
                               size_subim[1]*samplefac))
  
        
        #Assumes IFS FOV coordinates are 0-offset, with [0,0] at a pixel center
        
        #plots the whole thing
        ax1 = contfig.add_subplot(1, 4, 1)
        ax1.grid(False)
        ax1.set_title(str(capsource + ': ' + caphst))
        ax1.set_xlim(-0.5, size_subim[0]-0.5)
        ax1.set_ylim(-0.5, size_subim[1]-0.5)
        ax1.imshow(mapscl, cmap = 'hot')
        
        ax1.plot([hst_big_ifsfov[:,0],hst_big_ifsfov[0,0]], 
                 [hst_big_ifsfov[:,1],hst_big_ifsfov[0,1]], c = 'red') #possibly the box arms

        imsize = str(int(initmaps['hst']['subim_big'] * kpc_per_as))
        ax1.text(size_subim[0]*0.05, size_subim[1]*0.9, str(imsize+' X '+imsize+' kpc'), color = 'w')    #ie 37x37 kpc 
            
        #;cgtext,size_subim[0]*0.05,size_subim[1]*0.05,caphst,color='white'
        '''
        posbox1x[0] = truepos[0]+(truepos[2]-truepos[0])* \
                hst_big_ifsfov[3,0]/size_subim[0]
        posbox1y[0] = truepos[1]+(truepos[3]-truepos[1])* \
                hst_big_ifsfov[3,1]/size_subim[1]
        posbox2x[0] = truepos[0]+(truepos[2]-truepos[0])* \
                hst_big_ifsfov[0,0]/size_subim[0]
        posbox2y[0] = truepos[1]+(truepos[3]-truepos[1])* \
                hst_big_ifsfov[0,1]/size_subim[1]
        '''

        #HST continuum, IFS FOV (the zoom in)
        if dohstbl: size_subim = np.shape(bhst_fov)
        else: size_subim = np.shape(rhst_fov_sc)
        if dohstbl and dohstrd:
            mapscl = np.zeros((3,size_subim[0],size_subim[1]), bytes)
            mapscl[0,:,:] = rhst_fov_sc
            mapscl[2,:,:] = bhst_fov
            mapscl[1,:,:] = bytes((float(rhst_fov_sc)+float(bhst_fov))/2.0)
            ctmap = (rhst_fov_ns+bhst_fov_ns)/2.0
            if size_subim[0] < resampthresh or size_subim[1] < resampthresh:
                mapscl = rebin(mapscl, (3,size_subim[0]*samplefac,size_subim[1]*samplefac))
        else:
            mapscl = np.zeros((size_subim[0],size_subim[1]), bytes)
            if dohstrd:
                mapscl = rhst_fov_sc
                ctmap = rhst_fov_ns
            elif dohstbl:
                mapscl = bhst_fov
                ctmap = bhst_fov_ns
            if size_subim[0] < resampthresh or size_subim[1] < resampthresh:
                mapscl = rebin(mapscl, (size_subim[0]*samplefac,size_subim[1]*samplefac))
                
        ax2 = contfig.add_subplot(1, 4, 2)
        ax2.grid(False)
        ax2.imshow(mapscl, cmap = 'hot')
        
        if 'fithstpeak' in initmaps['hst'] and 'fithstpeakwin_kpc' in initmaps['hst']:
            nucfit_dwin_kpc = initmaps['hst']['fithstpeakwin_kpc']
            nucfit_halfdwin_hstpix = round(nucfit_dwin_kpc/kpc_per_hstpix/2.0)
            #subsets of images for peak fitting, centered around (first) nucleus
            xhst_sub = round(center_nuclei_hst[0,0]) + \
                [-nucfit_halfdwin_hstpix,nucfit_halfdwin_hstpix]
            yhst_sub = round(center_nuclei_hst[1,0]) + \
                [-nucfit_halfdwin_hstpix,nucfit_halfdwin_hstpix]
            ctmap_center = ctmap[xhst_sub[0]:xhst_sub[1], \
                          yhst_sub[0]:yhst_sub[1]]
            #Circular moffat fit
            
            #model for data
            moffat_init = models.Moffat2D()
            #fit data with fitter
            yfit = fit(moffat_init, ctmap_center)

            a = [yfit.amplitude, yfit.x_0, yfit.y_0, yfit.gamma, yfit.alpha]
            
            #Fitted peak coordinate in HST pixels; single-offset coordinates,
            #[1,1] at a pixel center
            peakfit_hstpix = [a[1]+xhst_sub[0]+1,a[2]+yhst_sub[0]+1]
            peakfit_hst_distance_from_nucleus_hstpix = peakfit_hstpix - \
                                                center_nuclei_hst[:,0]
            peakfit_hst_distance_from_nucleus_kpc = \
            peakfit_hst_distance_from_nucleus_hstpix * kpc_per_hstpix
            size_hstpix = np.shape(ctmap)
            
            ax2.set_xlim(0.5,size_hstpix[0]+0.5)
            ax2.set_ylim(0.5,size_hstpix[1]+0.5)
            ax2.plot(peakfit_hstpix[0],peakfit_hstpix[1], 'b+', mew = 2, ms = 20) # the + zoomed in image
            
        else: 
            ax2.plot([0])
  
        #plotaxesnuc(xran_kpc,yran_kpc,center_nuclei_kpc_x, \
                   #center_nuclei_kpc_y,toplab = True)
        ax2.text(xran_kpc[0]+(xran_kpc[1]-xran_kpc[0])*0.05,
                 yran_kpc[1]-(yran_kpc[1]-yran_kpc[0])*0.1,
                 'IFS FOV', color='white')
        #ifsf_plotcompass,xarr_kpc,yarr_kpc,carr=carr,/nolab,hsize=150d,hthick=2d
        '''
        posbox1x[1] = truepos[0]
        posbox1y[1] = truepos[3]
        posbox2x[1] = truepos[0]
        posbox2y[1] = truepos[1]
        '''

        #smoothed HST continuum, IFS FOV
        if dohstsm:
            '''
            #3-color image
;         mapscl = bytarr(3,size_subim[0],size_subim[1])
;         if dohstrd then mapscl[0,*,*] = rhst_fov_sm
;         if dohstbl then mapscl[2,*,*] = bhst_fov_sm
;         if dohstrd AND dohstbl then $
;            mapscl[1,*,*] = byte((double(rhst_fov_sm)+double(bhst_fov_sm))/2d)
            '''
            #Flux image
            if dohstbl and dohstrd:
                ctmap = (float(rhst_fov_sm_ns_rb)+float(bhst_fov_sm_ns_rb))/2.0
                if 'beta' in initmaps['hstblsm']:
                    beta = initmaps['hstblsm']['beta']
                elif 'beta' in initmaps['hstrdsm']:
                    beta = initmaps['hstrdsm']['beta']
                else: beta = 1.0
                if 'stretch' in initmaps['hstblsm']:
                    stretch = initmaps['hstblsm']
                elif 'stretch' in initmaps['hstrdsm']:
                    stretch = initmaps['hstrdsm']['stretch']
                else: stretch = 1
                if 'scllim' in initmaps['hstblsm']:
                    scllim = initmaps['hstblsm']['scllim']
                elif 'scllim' in initmaps['hstrdsm']:
                    scllim = initmaps['hstrdsm']['scllim']
                else: scllim = [np.min(ctmap), np.max(ctmap)]               
            elif dohstbl:
                ctmap = bhst_fov_sm_ns_rb
                if 'beta' in initmaps['hstblsm']:
                    beta = initmaps['hstblsm']['beta']
                else: beta = 1.0
                if 'stretch' in initmaps['hstblsm']:
                    stretch = initmaps['hstblsm']['stretch']
                else: stretch = 1
                if 'scllim' in initmaps['hstblsm']:
                    scllim = initmaps['hstblsm']['scllim']
                else: scllim = [np.min(ctmap), np.max(ctmap)]
            else:
                ctmap = rhst_fov_sm_ns_rb
                if 'beta' in initmaps['hstrdsm']:
                    beta = initmaps['hstrdsm']['beta']
                else: beta = 1.0
                if 'stretch' in initmaps['hstrdsm']:
                    stretch = initmaps['hstrdsm']['stretch']
                else: stretch = 1
                if 'scllim' in initmaps['hstrdsm']:
                    scllim = initmaps['hstrdsm']['scllim']
                else: scllim = [np.min(ctmap), np.max(ctmap)]
     
            ctmap /= max(ctmap)
            zran = scllim
            dzran = zran[1]-zran[0]
            #;if tag_exist(initmaps.ct,'beta') then beta=initmaps.ct.beta else beta=1d
            #mapscl = cgimgscl(ctmap,minval=zran[0],max=zran[1],$
                       #stretch=stretch,beta=beta)
            if size_subim[0] < resampthresh or size_subim[1] < resampthresh:
                mapscl = rebin(mapscl, (dx*samplefac,dy*samplefac))
            
            ax4 = contfig.add_subplot(1, 4, 4)
            ax4.grid(False)
            ax4.imshow(mapscl, cmap = 'hot')
            
            if 'fitifspeak' in initmaps['ct'] and 'fitifspeakwin_kpc' in initmaps['ct']:
                nucfit_dwin_kpc = initmaps['ct']['fitifspeakwin_kpc']
                nucfit_halfdwin_pix = round(nucfit_dwin_kpc/kpc_per_pix/2.0)
                #subsets of images for peak fitting, centered around (first) nucleus
                x_sub = round(center_nuclei[0,0]) + \
                    [-nucfit_halfdwin_pix,nucfit_halfdwin_pix]
                y_sub = round(center_nuclei[1,0]) + \
                    [-nucfit_halfdwin_pix,nucfit_halfdwin_pix]
                ctmap_center = ctmap[x_sub[0]:x_sub[1], y_sub[0]:y_sub[1]]
                #Circular moffat fit
                
                moffat_init = models.Moffat2D()
                yfit = fit(moffat_init, ctmap_center)

                a = [yfit.amplitude, yfit.x_0, yfit.y_0, yfit.gamma, yfit.alpha]
                
                #Fitted peak coordinate in IFS pixels; single-offset coordinates,
                #[1,1] at a pixel center
                peakfit_pix = [a[1]+x_sub[0]+1,a[2]+y_sub[0]+1]
                peakfit_hstconv_distance_from_nucleus_pix = peakfit_pix - \
                                                    center_nuclei[:,0]
                peakfit_hstconv_distance_from_nucleus_kpc = \
                peakfit_hstconv_distance_from_nucleus_pix * kpc_per_pix
                
                
                ax4.set_xlim(0.5, dx+0.5)
                ax4.set_ylim(0.5, dy+0.5)
                
                ax4.plot(peakfit_pix[0], peakfit_pix[1], 'ro')
            else:
                ax4.plot([0])
                
            #plotaxesnuc(xran_kpc,yran_kpc, center_nuclei_kpc_x,
                        #center_nuclei_kpc_y, nolab = True)
            ax4.text(xran_kpc[0]+(xran_kpc[1]-xran_kpc[0])*0.05,
            yran_kpc[1]-(yran_kpc[1]-yran_kpc[0])*0.1,
            'IFS FOV, conv.', color='white')
            
        #ax4.plot(posbox1x, posbox1y, color='Red')
        #ax4.plot(posbox2x, posbox2y, color='Red')

    #third subplot in pg1411cont.jpg
    if 'ct' in initmaps:
        
        ictlo = bisect.bisect(datacube.wave, ctsumrange_tmp[0])
        icthi = bisect.bisect(datacube.wave, ctsumrange_tmp[1])
        zran = initmaps['ct']['scllim']
        dzran = zran[1]-zran[0]
        
        if 'domedian' in initmaps['ct']:
            ctmap = np.median(datacube.dat[:,:,ictlo:icthi],axis=3) * \
                              float(icthi-ictlo+1)
        else: ctmap = np.sum(datacube.dat[:,:,ictlo:icthi], 2) #this has to be right
        ctmap /= np.max(ctmap)
        if 'beta' in initmaps['ct']: beta = initmaps['ct']['beta'] 
        else: beta = 1.0
        
        mapscl = rebin(ctmap, (dy*samplefac, dx*samplefac))
        #np.save('mapscl', mapscl) 
        
        if npanels_ifsfov == 1: ax3 = contfig.add_subplot(111)
        else: ax3 = contfig.add_subplot(1, 4, 3)
        
        #axes stuff
        ax3.axes.xaxis.set_ticks([])
        ax3.axes.yaxis.set_ticks([])
        ax3.set_xlabel('IFS', color='Blue', fontsize = 25)
        
        ax3.imshow(mapscl.T, cmap = 'hot', aspect = "equal", origin = "lower")
        
        
        if 'fitifspeak' in initmaps['ct'] and 'fitifspeakwin_kpc' in initmaps['ct']:
            nucfit_dwin_kpc = initmaps['ct']['fitifspeakwin_kpc']
            nucfit_halfdwin_pix = round(nucfit_dwin_kpc/kpc_per_pix.value/2.0)
            x_sub = round(center_nuclei[0]) +  \
                np.array([-nucfit_halfdwin_pix, nucfit_halfdwin_pix])
            y_sub = round(center_nuclei[1]) + \
                np.array([-nucfit_halfdwin_pix, nucfit_halfdwin_pix])
            ctmap_center = ctmap[x_sub[0]:x_sub[1], y_sub[0]:y_sub[1]]
  
            
  #Circular moffat fit
            yp, xp = mapscl.shape
            y, x, = np.mgrid[:yp, :xp]
            
            moffat_init = models.Moffat2D(x_0 = mapscl.argmax(axis = 1)[0], y_0 = mapscl.argmax(axis = 0)[0])
            yfit = fit(moffat_init, x, y, mapscl)
            a = yfit.parameters
            
            peakfit_pix = [x_sub[0]+1, y_sub[0]+1]
            peakfit_pix_ifs = peakfit_pix #save for radcont
            peakfit_ifs_distance_from_nucleus_pix = np.array(peakfit_pix) - \
                                             np.array(center_nuclei[:][0])
            peakfit_ifs_distance_from_nucleus_kpc = \
               peakfit_ifs_distance_from_nucleus_pix * kpc_per_pix
            
            ax3.plot(yfit.y_0, yfit.x_0, 'b+', mew = 3, ms = 20)
    
    if npanels_ifsfov == 1: #make more general?
        plt.title(initdat['name'], fontsize = 30)
    if npanels_ifsfov > 1:
        nolab_tmp = True
        toplab_tmp = False
    else:
        nolab_tmp = False
        toplab_tmp = True
  
    ax3.text(mapscl.shape[0]/2, mapscl.shape[1]/20, capifs, color='white', fontsize = 20, ha = 'center')

    #contfig.savefig(initdat['mapdir']+initdat['label']+ 'cont.jpg')
    contfig.savefig('/Users/hadley/Desktop/research/mapdir/pg1411cont.png', facecolor = 'white')
    
    
    '''''''''''''''''''''''''''''''''''''''''''''''''''''''''''
    ' Continuum color plots                                   '
    '''''''''''''''''''''''''''''''''''''''''''''''''''''''''''
    
    '''
    if 'hstcol' in initmaps or 'hstcolsm' in initmaps:
        if 'ct' in initmaps:
            if dohst: ctsumrange_tmp = initmaps['ct']['sumrange_hstcomp']
            else: ctsumrange_tmp = initmaps['ct']['sumrange']
            capifs = str(ctsumrange_tmp[0] + '-' + ctsumrange_tmp[1])
            if 'sumrange_lab' in initmaps['ct']:
                if initmaps['ct']['sumrange_lab'] == 'microns':
                    capifs = str(ctsumrange_tmp[0]/1e4 + '-' + ctsumrange_tmp[1]/1e4)
      
            if 'charscale' in initmaps['ct']: charscale = initmaps['ct']['charscale']
            else: charscale = 1.0
   
        caphst = str(initmaps['hstbl']['label'] + '-' + initmaps['hstrd']['label'])
        #arrays for positions for zoom box
        posbox1x = np.zeros(2, float)
        posbox1y = np.zeros(2, float)
        posbox2x = np.zeros(2, float)
        posbox2y = np.zeros(2, float)
   
        #Figure out correct image size in inches
        ysize_in = 2.2
        aspectrat_fov = float(dx)/float(dy)
        npanels_ifsfov = 0
        if 'ct' in initmaps: npanels_ifsfov = 1.0
        npanels_ifsfov += 1.0
        if dohstcolsm: npanels_ifsfov += 1.0
        if npanels_ifsfov == 0:
            print('MAKEMAPS: Error -- no continuum images to plot.')
            exit
        imgheight_in = 1.6
        xmargin_in = 0.4
        ymargin_in = (ysize_in - imgheight_in)/2.0
        ifsimg_width_in = imgheight_in*aspectrat_fov*npanels_ifsfov
        #Sizes and positions of image windows in real and normalized coordinates
        if dohst:
            xsize_in = imgheight_in + xmargin_in + ifsimg_width_in
            xfrac_margin = xmargin_in / xsize_in
            xfrac_hstbig = imgheight_in / xsize_in
            xfrac_ifsfov_width = imgheight_in*aspectrat_fov / xsize_in
            yfracb = ymargin_in/ysize_in
            yfract = 1.0 - ymargin_in/ysize_in
            pos_hstbig = [0.0,yfracb,xfrac_hstbig,yfract]
            pos_ifsfov = np.zeros((4, int(npanels_ifsfov)), float)
            pos_ifsfov[:,0] = [xfrac_hstbig+xfrac_margin,yfracb,
                         xfrac_hstbig+xfrac_margin+xfrac_ifsfov_width,yfract]
            for i in range (1, int(npanels_ifsfov) - 1):
                pos_ifsfov[:,i] = pos_ifsfov[:,i-1] + [xfrac_ifsfov_width,0.0, \
                                                xfrac_ifsfov_width,0.0]
            #Instrument labels
            lineoff = 0.1*xfrac_hstbig
            xhstline = [pos_hstbig[0]+lineoff,
                        pos_ifsfov[2,npanels_ifsfov-2]-lineoff]
            yhstline = [yfracb*0.75,yfracb*0.75]
            xhstline_tpos = (xhstline[1]+xhstline[0])/2.0
            yhstline_tpos = yfracb*0.15
            xifsline = [pos_ifsfov[0,npanels_ifsfov-1]+lineoff,
                        pos_ifsfov[2,npanels_ifsfov-1]-lineoff]
            yifsline = [yfracb*0.75,yfracb*0.75]
            xifsline_tpos = (xifsline[1]+xifsline[0])/2.0
            yifsline_tpos = yfracb*0.15
        else:
            ysize_in = 2.2 + ymargin_in
            xsize_in = xmargin_in + ifsimg_width_in
            yfracb = ymargin_in/ysize_in
            yfract = 1.0 - ymargin_in*2.0/ysize_in
            xfrac_margin = xmargin_in/xsize_in
            yfrac_margin = ymargin_in/ysize_in
            pos_ifsfov = [xfrac_margin,yfracb,1.0,yfract]
            #Instrument labels
            lineoff = 0.1*ifsimg_width_in/xsize_in
            xifsline = [pos_ifsfov[0]+lineoff,
                        pos_ifsfov[2]-lineoff]
            yifsline = [yfracb*0.75,yfracb*0.75]
            xifsline_tpos = (xifsline[1]+xifsline[0])/2.0
            yifsline_tpos = yfracb*0.15

        #cgps_open,initdat.mapdir+initdat.label+'color.eps',$
             #charsize=1d*charscale,$
             #/encap,/inches,xs=xsize_in,ys=ysize_in,/qui,/nomatch

        colorfig = plt.figure()

        if dohst:
            plt.text(xhstline_tpos,yhstline_tpos, str(capsource+': '+caphst),
                    color='Red')
      
            #HST continuum, large scale
            size_subim = np.shape(chst_big)
            mapscl = chst_big
            if size_subim[0] < resampthresh or size_subim[1] < resampthresh:
                mapscl = rebin(mapscl, (size_subim[0]*samplefac,size_subim[1]*samplefac))
            
            ax1 = colorfig.add_subplot(1, 4, 1)
            ax1.imshow(mapscl, cmap ='hot')
            #  Assumes IFS FOV coordinates are 0-offset, with [0,0] at a pixel center
            ax1.set_xlim(-0.5,size_subim[0]-0.5)
            ax1.set_ylim(-0.5,size_subim[1]-0.5)
            ax1.set_title(initdat['name'])
                
            ax1.plt([hst_big_ifsfov[:,0],hst_big_ifsfov[0,0]],
                    [hst_big_ifsfov[:,1],hst_big_ifsfov[0,1]], color='Red')

            imsize = str(int(initmaps['hst']['subim_big'] * kpc_per_as))
            plt.text(size_subim[0]*0.05,size_subim[1]*0.9, 
                     str(imsize+'\times'+imsize+' kpc'), color='white')
            ;
            posbox1x[0] = truepos[0]+(truepos[2]-truepos[0])* \
                        hst_big_ifsfov[3,0]/size_subim[0]
            posbox1y[0] = truepos[1]+(truepos[3]-truepos[1])* \
                        hst_big_ifsfov[3,1]/size_subim[1]
            posbox2x[0] = truepos[0]+(truepos[2]-truepos[0])* \
                        hst_big_ifsfov[0,0]/size_subim[0]
            posbox2y[0] = truepos[1]+(truepos[3]-truepos[1])* \
                        hst_big_ifsfov[0,1]/size_subim[1]
            ;

            #HST continuum, IFS FOV
            size_subim = np.shape(chst_fov)
            mapscl = chst_fov
            if size_subim[0] < resampthresh or size_subim[1] < resampthresh:
                mapscl = rebin(mapscl, (size_subim[0]*samplefac,size_subim[1]*samplefac))
            #cgloadct,65
            #cgimage,mapscl,/keep,pos=pos_ifsfov[*,0],opos=truepos,$
                 #/noerase,missing_value=bad,missing_index=255,$
                 #missing_color='white'
            ax2 = colorfig.add_subplot(1, 4, 2)
            ax2.imshow(mapscl)
            #plotaxesnuc(xran_kpc,yran_kpc,center_nuclei_kpc_x,
                       #center_nuclei_kpc_y,toplab = True)
            plt.text(xran_kpc[0]+(xran_kpc[1]-xran_kpc[0])*0.05,
                     yran_kpc[1]-(yran_kpc[1]-yran_kpc[0])*0.1,
                     'IFS FOV',color='white')
            #plotcompass,xarr_kpc,yarr_kpc,carr=carr,/nolab,hsize=150d,hthick=2d
            
            ;posbox1x[1] = truepos[0]
            ;posbox1y[1] = truepos[3]
            ;posbox2x[1] = truepos[0]
            ;posbox2y[1] = truepos[1]
            

            #smoothed HST continuum, IFS FOV
            if dohstcolsm:
                colmap = cshst_fov_rb
                size_subim = np.shape(colmap)
                zran = initmaps['hstcolsm']['scllim']
                dzran = zran[1]-zran[0]
                #mapscl = cgimgscl(colmap,minval=zran[0],max=zran[1],\
                                  #stretch=initmaps.ct.stretch)
                if size_subim[0] < resampthresh or size_subim[1] < resampthresh:
                    mapscl = rebin(mapscl, (size_subim[0]*samplefac,size_subim[1]*samplefac))
                
                ax3 = colorfig.add_suplot(1, 4, 3)
                ax3.imshow(mapscl, extent=(-150,200,-150,200))
                #plotaxesnuc(xran_kpc,yran_kpc,
                          #center_nuclei_kpc_x,center_nuclei_kpc_y,nolab = True)
                plt.text(xran_kpc[0]+(xran_kpc[1]-xran_kpc[0])*0.05,
                yran_kpc[1]-(yran_kpc[1]-yran_kpc[0])*0.1,
                'IFS FOV, conv.', color='white')
      

            #plt.plot(posbox1x,posbox1y,color='Red')
            #plt.plot(posbox2x,posbox2y,color='Red')

        if 'ct' in initmaps:
            plt.text(xifsline_tpos,yifsline_tpos, 'IFS', color='Blue')

            ictlo = list(datacube.wave).index(ctsumrange_tmp[0])
            icthi = list(datacube.wave).index(ctsumrange_tmp[1])
            zran = initmaps['ct']['scllim']
            dzran = zran[1]-zran[0]
            if 'domedian' in initmaps['ct']:
                ctmap = np.median(datacube.dat[:,:,ictlo:icthi]) * \
                                  float(icthi-ictlo+1, 2)
            else:
                ctmap = np.sum(datacube.dat[:,:,ictlo:icthi], 2)
            ctmap /= max(ctmap)
            if 'beta' in initmaps['ct']: beta = initmaps['ct']['beta']
            else: beta=1.0
            #mapscl = cgimgscl(rebin(ctmap,dx*samplefac,dy*samplefac,/sample),$
                        #minval=zran[0],max=zran[1],$
                        #stretch=initmaps.ct.stretch,beta=beta)                        
            #cgloadct,65
            #cgimage,mapscl,/keep,pos=pos_ifsfov[*,fix(npanels_ifsfov) - 1],$
                  #opos=truepos,/noerase,missing_value=bad,missing_index=255,$
                  #missing_color='white'
            ax4 = colorfig.add_subplot(1, 4, 4)
            ax4.imshow(mapscl, cmap = 'hot')
            if 'fitifspeak' in initmaps['ct'] and 'fitifspeakwin_kpc' in initmaps['ct']:
                nucfit_dwin_kpc = initmaps['ct']['fitifspeakwin_kpc']
                nucfit_halfdwin_pix = round(nucfit_dwin_kpc/kpc_per_pix/2.0)
                #subsets of images for peak fitting, centered around (first) nucleus
                x_sub = round(center_nuclei[0,0]) + \
                        [-nucfit_halfdwin_pix,nucfit_halfdwin_pix]
                y_sub = round(center_nuclei[1,0]) + \
                        [-nucfit_halfdwin_pix,nucfit_halfdwin_pix]
                ctmap_center = ctmap[x_sub[0]:x_sub[1], y_sub[0]:y_sub[1]]
                #Circular moffat fit
                moffat_init = models.Moffat2D()
                yfit = fit(moffat_init, ctmap_center)
                a = [yfit.amplitude, yfit.x_0, yfit.y_0, yfit.gamma, yfit.alpha]

                #Fitted peak coordinate in IFS pixels; single-offset coordinates,
                #[1,1] at a pixel center
                peakfit_pix = [a[1]+x_sub[0]+1,a[2]+y_sub[0]+1]
                peakfit_pix_ifs = peakfit_pix #save for later
                peakfit_ifs_distance_from_nucleus_pix = peakfit_pix - \
                                                 center_nuclei[:,0]
                peakfit_ifs_distance_from_nucleus_kpc = \
                    peakfit_ifs_distance_from_nucleus_pix * kpc_per_pix
                ax4.plot([0])
                #ax4.set_xlim(0.5,dx+0.5)
                #ax4.set_ylim(0.5,dy+0.5)
                ax4.plot(peakfit_pix[0],peakfit_pix[1], 'ro')
            else:
                ax4.plot([0])
                #title=title_tmp
      
            if npanels_ifsfov == 1: plt.text(0.5, 1.0 - yfrac_margin, initdat['name'])
            if npanels_ifsfov > 1:
                nolab_tmp = True
                toplab_tmp = False
            else:
                nolab_tmp=False
                toplab_tmp=True
      
            #plotaxesnuc(xran_kpc,yran_kpc,
                       #center_nuclei_kpc_x,center_nuclei_kpc_y,
                       #nolab=nolab_tmp,toplab=toplab_tmp)
            plt.text(xran_kpc[0]+(xran_kpc[1]-xran_kpc[0])*0.05,
                     yran_kpc[1]-(yran_kpc[1]-yran_kpc[0])*0.1, 
                     capifs,color='white')
            if npanels_ifsfov == 1: print("do later")
                #plotcompass,xarr_kpc,yarr_kpc,carr=carr,/nolab,$
                          #hsize=150d,hthick=2d
                          
            #cgps_close
            
    '''
    
    '''''''''''''''''''''''''''''''''''''''''''''''''''''''''''
    ' Continuum radial profiles                               '
    '''''''''''''''''''''''''''''''''''''''''''''''''''''''''''
    #very not done
    
    '''
    #TODO
    if 'ct' in initmaps:
        #npy = 2
        npx = 1
        if 'decompose_qso_fit' in initdat: npx = 3
        if 'remove_scattered' in initdat: npx = 4

        #Figure out correct image size in inches
        panel_in = 2.0
        margin_in = 0.5
        halfmargin_in = margin_in/2.0
        xsize_in = panel_in * npx + margin_in
        aspectrat_fov=float(dx)/float(dy)
        ysize_in = margin_in*2.0 + panel_in * (1.0 + 1.0/aspectrat_fov)
        #Sizes and positions of image windows in real and normalized coordinates
        pan_xfrac = panel_in/xsize_in
        pan_yfrac = panel_in/ysize_in
        ifs_yfrac = panel_in/aspectrat_fov/ysize_in
        mar_xfrac = margin_in/xsize_in
        #mar_yfrac = margin_in/ysize_in
        #hmar_xfrac = halfmargin_in/xsize_in
        hmar_yfrac = halfmargin_in/ysize_in
        pos_top = np.zeros((4,npx), float)
        pos_bot = np.zeros((4,npx), float)
        for i in range(0, npx-1):
            pos_top[:,i] = [mar_xfrac+float(i)*pan_xfrac,
                            1.0 - (hmar_yfrac+pan_yfrac),
                            mar_xfrac+float(i+1)*pan_xfrac,
                            1.0 - hmar_yfrac]
            pos_bot[:,i] = [mar_xfrac+float(i)*pan_xfrac,
                            hmar_yfrac,
                            mar_xfrac+float(i+1)*pan_xfrac,
                            hmar_yfrac+ifs_yfrac]
        
        zran = initmaps['ct']['scllim_rad']
        #dzran = zran[1]-zran[0]
        
        #see pg1411cont_rad.jpg
        radfig = plt.figure(figsize=(12, 5))
        radfig.subplots_adjust(wspace=.5, hspace=.25)
        axs = radfig.subplots(2, 3)
      
        #Total (continuum-only) model flux. Plot fits if decompose tags set, otherwise
        #plot total cube flux within specified range.
        if 'decompose_qso_fit' in initdat:
            #Divide model flux by total # pixels for cases where total number of 
            #pixels varies by spaxel (due to contracted spectra at edges, e.g.)
            ctmap = np.sum(contcube['qso_mod']+contcube['host_mod'], 2) / contcube['npts'] #TODO
            ctsumrange_tmp = initdat['fitran']
        elif 'decompose_ppxf_fit' in initdat:
            tmpstel = contcube['stel_mod_tot']
            ibdtmp = np.where(tmpstel == bad)
            ctbdtmp = len(ibdtmp)
            if ctbdtmp > 0: tmpstel[ibdtmp] = 0.0
            tmppoly = contcube['poly_mod_tot']
            ibdtmp = np.where(tmppoly == bad)
            ctbdtmp = len(ibdtmp)
            if ctbdtmp > 0: tmppoly[ibdtmp] = 0.0
            ctmap = tmpstel+tmppoly
            ctsumrange_tmp = initdat['fitran']
        else:
            ictlo = list(datacube.wave).index(initmaps['ct']['sumrange'][0])
            icthi = list(datacube.wave).index(initmaps['ct']['sumrange'][1])
            ctmap = (datacube.dat[:,:,ictlo:icthi]).sum(2)
            ctsumrange_tmp = initmaps['ct']['sumrange']
        capran = str(ctsumrange_tmp[0]) + '-' + str(ctsumrange_tmp[1])
        if 'sumrange_lab' in initmaps['ct']:
            if initmaps['ct']['sumrange_lab'] == 'microns':
                capran = str(ctsumrange_tmp[0]/1e4) + '-' + str(ctsumrange_tmp[1]/1e4)
      
        maxctmap = np.max(ctmap)
        
        #pdb.set_trace()
        ctmap /= maxctmap
        axs[0,0].plot(map_rkpc_ifs, np.log10(ctmap))
        axs[0,0].plot([1, 5, 10], [0, -1, -2], 'r-')
        axs[0,0].set_ylim(-4,0)
        axs[0,0].set_xlim(0, np.max(map_rkpc_ifs))
        axs[0,0].set_title(initdat['name'])
        axs[0,0].set_xlabel('Radius (kpc)')
        axs[0,0].set_ylabel('log I/I(max)')
        if 'decompose_qso_fit' in initdat:
            axs[0,0].plot(psf1d_x, psf1d_y, 'r-')
        #elif 'fit_empsf' in initmaps:
            #axs[0,0].plot(empsf1d_x, empsf1d_y, 'r-')
        elif 'ctradprof_psffwhm' in initmaps:
            x = np.arange(101.)/100.0 * np.max(map_rkpc_ifs)
            fwhm=initmaps['ctradprof_psffwhm'] * kpc_per_as
            #Gaussian
            y = stats.norm.pdf(x, 1.0, 0.0, fwhm/2.35) #i think this only takes 3 args
            y = [math.log(z,10) for z in y]
            axs[0,0].plot(x, y, 'bo')
            #Moffat, index = 1.5
            y = math.log10(moffat(x,[1.0,0.0,fwhm/2.0/np.sqrt(2^(1/1.5)-1),1.5]))
            axs[0, 0].plot(x,y,'r-')
            #Moffat, index = 2.5
            y = math.log10(moffat(x,[1.0,0.0,fwhm/2.0/np.sqrt(2^(1/2.5)-1),2.5]))
            axs[0, 0].plot(x,y, color='Red')
            #Moffat, index = 5
            y = math.log10(moffat(x,[1.0,0.0,fwhm/2.0/np.sqrt(2^(1/5)-1),5]))
            axs[0,0].plot(x,y,color='Blue')
           
        #mapscl = rebin(np.log10(ctmap),(dx*samplefac, dy*samplefac))
      
        axs[1, 0].imshow(mapscl, extent=(-150,200,-150,200), cmap = 'hot') #arbitrary numbers
        #axs[1, 0].set_xlim(0.5, dx + 0.5)
        #axs[1, 0].set_ylim(0.5, dy + 0.5)
        if 'fitifspeak' in initmaps['ct'] and 'fitifspeakwin_kpc' in initmaps['ct']:
            axs[1, 0].plot(peakfit_pix_ifs[0],peakfit_pix_ifs[1],color='Red')         
        axs[1, 0].text(-120, 170,'Host Cont.+Quasar', color='white')
        axs[1, 0].text(-120,-dy*0.95,capran,color='white')
        #plotaxesnuc,xran_kpc,yran_kpc,center_nuclei_kpc_x,center_nuclei_kpc_y

        if 'decompose_qso_fit' in initdat:
            qso_map = np.sum(contcube['qso_mod'], axis = 2) / contcube['npts']
            #qso_map /= max(qso_map)
            qso_map /= maxctmap
            axs[0, 1].plot(map_rkpc_ifs, np.log10(qso_map), 'bo')
            axs[0, 1].set_ylim(-4,0)
            axs[0, 1].set_xlim(0, np.max(map_rkpc_ifs))
            #if tag_exist(initdat,'decompose_qso_fit') then begin
            axs[0, 1].plot(psf1d_x,psf1d_y,color='Red')
            axs[0, 1].text(np.max(map_rkpc_ifs)*0.5, -4.0*0.1, 'FWHM=' +  
                str(round(qso_fitpar[2]*initdat['platescale'], 2)) + ' asec') #TODO: 2
            axs[0, 1].text(np.max(map_rkpc_ifs)*0.5,-4.0*0.2,'FWHM='+
                str(round(qso_fitpar[2]*initdat['platescale']*kpc_per_as.value, 2))+' kpc') #TODO: 2
            axs[0, 1].text(np.max(map_rkpc_ifs)*0.5, -4.0*0.3, '\u03B3='+ \
                str(qso_fitpar[4]))
            
;         endif else if tag_exist(initmaps,'fit_empsf') then begin
;            cgoplot,empsf1d_x,empsf1d_y,color='Red'
;         endif else if tag_exist(initmaps,'ctradprof_psffwhm') then begin
;            x = dindgen(101)/100d*max(map_rkpc_ifs)
;            fwhm=initmaps.ctradprof_psffwhm * kpc_per_as
;;           Gaussian
;            y = alog10(gaussian(x,[1d,0d,fwhm/2.35]))
;            cgoplot,x,y,color='Black'
;;           Moffat, index = 1.5
;            y = alog10(moffat(x,[1d,0d,fwhm/2d/sqrt(2^(1/1.5d)-1),1.5d]))
;            cgoplot,x,y,color='Red',/linesty
;;           Moffat, index = 2.5
;            y = alog10(moffat(x,[1d,0d,fwhm/2d/sqrt(2^(1/2.5d)-1),2.5d]))
;            cgoplot,x,y,color='Red'
;;           Moffat, index = 5
;            y = alog10(moffat(x,[1d,0d,fwhm/2d/sqrt(2^(1/5d)-1),5d]))
;            cgoplot,x,y,color='Blue'
;         endif

;         mapscl = cgimgscl(rebin(qso_map,dx*samplefac,dy*samplefac,/sample),$
;                           minval=zran[0],max=zran[1],stretch=initmaps.ct.stretch)
            
            #mapscl = rebin(math.log10(qso_map), (dx*samplefac, dy*samplefac))
            #this has to be a lot more zoomed in
            axs[1, 1].imshow(mapscl, extent=(-150,200,-150,200), cmap = 'hot')
            #axs[1, 1].set_xlim(0.5, dx+0.5)
            #axs[1, 1].set_xlim(0.5, 200)
            
            if 'fitifspeak' in initmaps['ct'] and 'fitifspeakwin_kpc' in initmaps['ct']:
                peakfit_pix = [qso_fitpar[1]+1,qso_fitpar[2]+1]
                
                peakfit_ifs_qso_distance_from_nucleus_pix = np.array(peakfit_pix) - \
                                                        np.array(center_nuclei[0:])
                peakfit_ifs_qso_distance_from_nucleus_kpc = \
                    peakfit_ifs_qso_distance_from_nucleus_pix * kpc_per_pix
                axs[1, 1].plot(peakfit_pix[0],peakfit_pix[1],color='Red')        
            axs[1, 1].text(dx*0.05, dy*0.95, 'Quasar PSF', color='white')
            #plotaxesnuc(xran_kpc,yran_kpc,
                          #center_nuclei_kpc_x,center_nuclei_kpc_y,nolab = True)
            
            host_map = np.sum(contcube['host_mod'], axis = 2) / contcube['npts']
            #host_map /= max(host_map)
            host_map /= maxctmap
            axs[0, 2].plot(map_rkpc_ifs, np.log10(host_map))
            axs[0, 2].set_ylim(-4, 0)
            axs[0, 2].set_xlim(0, np.max(map_rkpc_ifs))
            #;if tag_exist(initdat,'decompose_qso_fit') then begin
            axs[0, 2].plot(psf1d_x,psf1d_y,color='Red')
            
;         endif else if tag_exist(initmaps,'fit_empsf') then begin
;            cgoplot,empsf1d_x,empsf1d_y,color='Red'
;         endif else if tag_exist(initmaps,'ctradprof_psffwhm') then begin
;            x = dindgen(101)/100d*max(map_rkpc_ifs)
;            fwhm=initmaps.ctradprof_psffwhm * kpc_per_as
;;           Gaussian
;            y = alog10(gaussian(x,[1d,0d,fwhm/2.35]))
;            cgoplot,x,y,color='Black'
;;           Moffat, index = 1.5
;            y = alog10(moffat(x,[1d,0d,fwhm/2d/sqrt(2^(1/1.5d)-1),1.5d]))
;            cgoplot,x,y,color='Red',/linesty
;;           Moffat, index = 2.5
;            y = alog10(moffat(x,[1d,0d,fwhm/2d/sqrt(2^(1/2.5d)-1),2.5d]))
;            cgoplot,x,y,color='Red'
;;           Moffat, index = 5
;            y = alog10(moffat(x,[1d,0d,fwhm/2d/sqrt(2^(1/5d)-1),5d]))
;            cgoplot,x,y,color='Blue'
;         endif
         
;         mapscl = cgimgscl(rebin(host_map,dx*samplefac,dy*samplefac,/sample),$
;                           minval=zran[0]),max=zran[1],stretch=initmaps.ct.stretch)
            
            #mapscl = cgimgscl(rebin(math.log10(host_map),dx*samplefac,dy*samplefac,sample = True),
                        #minval=zran[0],max=zran[1],stretch=initmaps['ct']['stretch'])
            
            axs[1, 2].imshow(mapscl, extent=(-150,200,-150,200), cmap = 'hot')
            #axs[1, 2].set_xlim(0.5,dx+0.5)
            #axs[1, 2].set_xlim(0.5,dy+0.5)
            axs[1, 2].text(dx*0.05,dy*0.95,'Host Cont.',color='white')
            #plotaxesnuc(xran_kpc,yran_kpc,
                          #center_nuclei_kpc_x,center_nuclei_kpc_y,nolab = True)
            
            if 'remove_scattered' in initdat: #Where is this in the figure
                scatt_map = np.sum(contcube['poly_mod'], axis = 2) / contcube['npts']
                   
;;           Use maximum flux for normalization unless it's much higher than 
;;           second highest flux
;            ifsort = reverse(sort(scatt_map))
;            if scatt_map[ifsort[0]]/scatt_map[ifsort[1]] gt 2 then $
;               scatt_map /= scatt_map[ifsort[1]] $
;            else scatt_map /= scatt_map[ifsort[0]]
                   
                scatt_map /= maxctmap
                plt.plot(map_rkpc_ifs, math.log10(scatt_map))
                plt.ylim(-4,0)
                plt.xlim(0, np.max(map_rkpc_ifs))
    
                plt.plot(psf1d_x,psf1d_y,color='Red')
                
                   
;            mapscl = cgimgscl(rebin(scatt_map,dx*samplefac,dy*samplefac,/sample),$
;                              minval=zran[0],max=zran[1],$
;                              stretch=initmaps.ct.stretch)
                   
                #mapscl = cgimgscl(rebin(math.log10(scatt_map),dx*samplefac,dy*samplefac, sample = True),
                        #minval=zran[0],max=zran[1],stretch=initmaps['ct']['stretch'])
                
                plt.imshow(mapscl, extent=(-150,200,-150,200), cmap = 'hot')
                plt.xlim(0.5,dx+0.5)
                plt.ylim(0.5,dy+0.5)
                plt.text(dx*0.05,dy*0.95,'Scattered Light',color='white')
                #plotaxesnuc(xran_kpc,yran_kpc, \
                             #center_nuclei_kpc_x,center_nuclei_kpc_y,nolab = True)
        
        #cgps_close
        radfig.savefig('/Users/hadley/Desktop/research/mapdir/pg1411cont_rad.eps', facecolor = 'white')
'''
#hello = 2
#makemaps('pg1411')
