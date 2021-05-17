#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
This procedure makes maps of various quantities. 
Contains one helper routine: IFSF_PA

Retuns Postscript plots.
Params:
    initproc: in, required, type=string
    Name of procedure to initialize the fit.

@author: hadley
"""
import numpy as np
import importlib
import os.path
import math
import cv2

from q3dfit.common.linelist import linelist
from q3dfit.common.readcube import readcube
from q3dfit.common.rebin import rebin
from q3dfit.common.hstsubim import hstsubim



from astropy.cosmology import luminosity_distance
from astropy.io import fits

#for congrid
import scipy.interpolate
import scipy.ndimage


def posangle (xaxis, yaxis): 
    return None

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
 
def mpfit2dpeak(): #need to find a replacement. returns arrays for fit and params
    return None   
 
def ifsf_pa(): #need to find a replacement
    return None   
 
def makemaps (initproc):
    fwhm2sig = 2.0 * np.sqrt(2.0 * np.log(2.0))
    plotquantum = 2.5                       #in inches
    bad = 1e99
    c_kmPerS = 299792.458
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
    initmaps = None
    
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
            #TODO: check if params match up
            listlines = linelist(initdat['lines'], linelab = linelabels, \
                                  _extra = initdat['argslinelist'])
        else:
            listlines = linelist(initdat['lines'], linelab = linelabels)

    #Linelist with doublets to combine
    emldoublets = np.array([['[SII]6716','[SII]6731'],
                    ['[OII]3726','[OII]3729'],
                    ['[NI]5198','[NI]5200'],
                    ['[NeIII]3869','[NeIII]3967'],
                    ['[NeV]3345','[NeV]3426'],
                    ['MgII2796','MgII2803']])
     
    if emldoublets.ndim == 1: ndoublets = 1 
    else: numdoublets = emldoublets.size
    lines_with_doublets = initdat['lines']
     
    for i in range(numdoublets - 1):
        if (emldoublets[0,i] in listlines) and (emldoublets[1,i] in listlines):
           dkey = emldoublets[0,i] + '+' + emldoublets[1,i]
           lines_with_doublets = np.array([lines_with_doublets, dkey])
        
    if 'argslinelist' in initdat:
        linelist_with_doublets = \
            listlines(lines_with_doublets, linelab = linelabels, \
                         _extra = initdat['argslinelist'])
    else:
        linelist_with_doublets = \
            listlines(lines_with_doublets, linelab = linelabels)

    if 'donad' in initdat:
        if 'argslinelist' in initdat:
            nadlinelist = linelist(['NaD1','NaD2','HeI5876'], \
                                     _extra = initdat['argslinelist'])
        else:
            nadlinelist = linelist(['NaD1','NaD2','HeI5876'])

    '''
    Get range file

    plot types, in order; used for correlating with input ranges (array 
                                                                  rangequant)
    '''
  
    hasrangefile = 0
    if 'rangefile' in initmaps:
      #TODO
      if os.path.isfile(initmaps['rangefile']): 
          #readcol,initmaps.rangefile,rangeline,rangequant,rangelo,rangehi,$
                 #rangencbdiv,format='(A,A,D,D,I)',/silent
         hasrangefile = 1
      else: print('Range file listed in INITMAPS but not found.')

      
    #TODO: file and struct names below (4)

    #Restore line maps
    if 'noemlinfit' not in initdat:
        file = initdat['outdir'] + initdat['label'] + '.lin.xdr'
        struct1 = (np.load(file, allow_pickle='TRUE')).item()
   
    #Restore continuum parameters
    if 'decompose_ppxf_fit' in initdat or 'decompose_qso_fit' in initdat:
        file = initdat['outdir'] + initdat['label'] + '.cont.xdr'
        contcube = (np.load(file, allow_pickle='TRUE')).item()
    

    #Get NaD parameters
    if 'donad' in initdat:
        file = initdat['outdir'] + initdat['label'] + '.nadspec.xdr'
        nadcube = (np.load(file, allow_pickle='TRUE')).item()
        file = initdat['outdir'] + initdat['label'] + '.nadfit.xdr'
        struct4 = (np.load(file, allow_pickle='TRUE')).item()        
        
        if 'badnademp' in initmaps:
            tagstobad = np.array(['WEQ','IWEQ','EMFLUX','EMUL','VEL'])
            #assuming nadcube is a dictionary
            tagnames = nadcube.keys
            
            #TODO: check what badnademp would be (used 0 for now)
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
        asdist = ldist / (1.0 + initdat['zsys_gas']) ^ 2.0
    else:
        #Planck 2018 parameters: https://ui.adsabs.harvard.edu/#abs/arXiv:1807.06209
        #TODO: fcn to calculate luminosity distance
        ldist = luminosity_distance(initdat['zsys_gas'])
        #ldist = lumdist(initdat.zsys_gas,H0=67.4d,Omega_m=0.315d,Lambda0=0.685d,/silent)
        asdist = ldist / (1.0 + initdat['zsys_gas']) ^ 2.0
        
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
   
    header = 1
    datacube = readcube(initdat['infile'], quiet = True, oned = True, \
                        header = header, datext = datext, varext = varext, \
                        dqext = dqext)
    if 'fluxfactor' in initmaps:
        datacube['dat'] *= initmaps['fluxfactor']
        datacube['var'] *= (initmaps['fluxfactor']) ^ 2.0
   
    dimension_sizes = datacube['dat'].shape
    dx = dimension_sizes[0]
    dy = dimension_sizes[1]
    dz = dimension_sizes[2]
   
    #defined so that center of spaxel at bottom left has coordinates [1,1] 
    if center_axes[0] == -1: center_axes = [float(dx) / 2.0, float(dy) / 2.0] + 0.5
    if center_nuclei[0] == -1: center_nuclei = center_axes
    if 'vornorm' in initmaps:
        datacube['dat'] /= rebin(initmaps['vornorm'], (dx, dy, dz))
   

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
    dohst = 0
    dohstbl = 0
    dohstrd = 0
    dohstsm = 0
    dohstcol = 0
    dohstcolsm = 0
    if 'hst' in initmaps and 'hstbl' in initmaps:
        dohstbl = 1
        if 'ext' in initmaps['hstbl']: hstblext = initmaps['hstbl']['ext']
        else: hstblext = 1
        hstbl = fits.open(initmaps['hstbl']['file'])
        hstblhead = hstbl.header #TODO
        hst_big_ifsfov = np.zeros((4,3), dtype = float)
        if 'platescale' in initmaps['hstbl']:
            hstpsbl = initmaps['hstbl']['platescale']
        else: hstpsbl = 0.05
        if 'refcoords' in initmaps['hstbl']:
            hstrefcoords = initmaps['hstbl']['refcoords']
        else: hstrefcoords = initmaps['hst']['refcoords']
        if initmaps['hstbl']['buffac'] in initmaps['hstbl']: #TODO: had a comma?
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
                              size_tmp[0], size_tmp[1])
                map_rkpc_tmp = np.sqrt((map_x_tmp - (hstrefcoords[0] + \
                               initmaps['hstbl']['nucoffset'][0] - 1)) ^ 2.0 + \
                               (map_y_tmp - (hstrefcoords[1] + \
                                initmaps['hstbl']['nucoffset'][1] - 1)) ^ 2.0) * \
                                initmaps['hstbl']['platescale'] * kpc_per_as
                ipsf = np.where(map_rkpc_tmp < 0.15)
                ipsf_bkgd = np.where((map_rkpc_tmp > 0.15) and (map_rkpc_tmp < 0.25))
                hstbl_tmp = hstbl
                hstbl_tmp[ipsf] = np.median(hstbl[ipsf_bkgd])
                #TODO: replacement for filter_image
                #hstblsm = filter_image(hstbl_tmp,fwhm=initmaps.hst.smoothfwhm,/all)
                hstblsm = cv2.blur(hstbl_tmp)
            else:
                hstbltmp = hstbl
                ibadhst = np.where(hstbl == 0.0)
                ctbadhst = len(ibadhst)
                #TODO: what does this mean?
                #if ctbadhst > 0: hstbltmp[ibadhst] = !values.d_nan
                fwhm = initmaps['hst']['smoothfwhm']
                boxwidth = round((fwhm ^ 2.0) / 2.0 + 1.0)
                if not boxwidth: ++boxwidth
                #hstblsm = filter_image(hstbl,fwhm=initmaps.hst.smoothfwhm,/all)
                hstblsm = cv2.blur(hstbl)
                #hstblsm = filter_image(hstbltmp,smooth=boxwidth,/iter,/all)
                hstblsm = cv2.blur(hstbltmp)
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
                
                #TODO: check
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
                hstrdsm = cv2.blur(hstrd_tmp)
            else:
                hstrdtmp = hstrd
                ibadhst = np.where(hstrd == 0)
                ctbadhst = len(ibadhst)
                #TODO
                #if ctbadhst > 0: hstrdtmp[ibadhst] = !values.d_nan
                fwhm = initmaps['hst']['smoothfwhm']
                boxwidth = round((fwhm ^ 2.0) / 2.0 + 1.0)
                if not boxwidth: ++boxwidth
                #hstrdsm = filter_image(hstrd,fwhm=initmaps.hst.smoothfwhm,/all)
                #hstrdsm = filter_image(hstrdtmp,smooth=boxwidth,/iter,/all)
                hstrdsm = cv2.blur(hstrdtmp)
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
      
    if dohstbl or dohstrd: dohst = 0 #TODO
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
            pivotbl = hstblhead['PHOTPLAM'] #TODO: check
            
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
                    hstbl = np.roll(hstbl, round(idiff[0]), axis = 1) #TODO: shift horiz.
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
            #nsdevreg = 1
      
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
            #TODO
            
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
        if sdevreg.ndim == 1 : #TODO
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
    map_x = rebin(np.arange(1, dx), dx, dy)
    map_y = rebin(np.transpose(np.arange(1, dy)), dx, dy)
    map_r = np.sqrt((map_x - center_axes[0])^2.0 + (map_y - center_axes[1])^2.0)
    map_rkpc_ifs = map_r * kpc_per_pix

    #PA E of N, in degrees; spaxel with [0,0] has PA = bad
    map_pa = np.zeros((dx,dy), dtype = float)
    map_xaxis = map_x - center_axes[0]
    map_yaxis = map_y - center_axes[1]
    map_pa = ifsf_pa(map_xaxis, map_yaxis) #TODO
    map_pa += initdat['positionangle']
    iphase = np.where(map_pa > 360.0)
    ctphase = len(iphase)
    if ctphase > 0: map_pa[iphase] -= 360.0
    inuc = np.where(map_r == 0.0)
    ctnuc = len(inuc)
    if ctnuc > 0: map_pa[inuc] = bad

    if 'decompose_qso_fit' in initdat:
        inan = np.where(np.isfinite(contcube['qso_mod']))
        ctnan = len(inan)
        if ctnan > 0: contcube['qso_mod'][inan] = 0.0 #TODO: array = 0?
        qso_map = sum(contcube.qso_mod, 3) / contcube['npts']
        maxqso_map = max(qso_map)
        #qso_err = stddev(contcube.qso,dim=3,/double)
        #qso_err = sqrt(total(datacubeube.var,3))
        qso_err = np.sqrt(np.median(datacube['var'], dim=3)) #TODO
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
        #parinit = REPLICATE({value:0d, fixed:0b, limited:[0B,0B], tied:'', limits:[0d,0d]},8)
        parinit = np.full(8, {'value': 0.0, 'fixed': 0.0, 
                   'limited':[bytes(0),bytes(0)], 
                   'tied':'', 'limits':[0.0, 0.0]})
        est = np.array([0.0, 1.0, 1.0, 1.0, center_nuclei[0] - 1.0, center_nuclei[1] - 1.0, 0.0, 3.0])
        parinit['value'] = est
        parinit[7]['limited'] = [bytes(1),bytes(1)]
        parinit[7]['limits'] = [0.0, 5.0]
        qso_fit = mpfit2dpeak(qso_map, qso_fitpar, circular = True, 
                              moffat = True, est = est,
                              error = qso_err, parinfo = parinit)
        map_rnuc = np.sqrt((map_x - qso_fitpar[4] + 1) ^ 2.0 + \
                              (map_y - qso_fitpar[5] + 1) ^ 2.0)
        map_rnuckpc_ifs = map_rnuc * kpc_per_pix
        psf1d_x = np.arange(101.) / 100.0 * max(map_rnuckpc_ifs)
        psf1d_y = np.log10(moffat(psf1d_x, [qso_fitpar[1], 0.0,
                              qso_fitpar[2] * kpc_per_pix, qso_fitpar[7]]))

    '''''''''''''''''''''''''''''''''''''''''''''''''''''''''''
    ' Compute radii and centers                               '  
    '''''''''''''''''''''''''''''''''''''''''''''''''''''''''''

    #coordinates in kpc
    xran_kpc = float([0 - (center_axes[0] - 0.5), dx - (center_axes[0] - 0.5)]) \
              * kpc_per_pix
    yran_kpc = float([0 - (center_axes[1] - 0.5), dy - (center_axes[1] - 0.5)]) \
              * kpc_per_pix
    center_nuclei_kpc_x = (center_nuclei[0, :] - center_axes[0]) \
                         * kpc_per_pix
    center_nuclei_kpc_y = (center_nuclei[1, :] - center_axes[1]) \
                         * kpc_per_pix
    #in image window
    xwinran_kpc = float([0 - (center_axes[0] - 0.5 - (plotwin[0] - 1)), \
                         dxwin - (center_axes[0] - 0.5 - (plotwin[0]-1))]) \
                        * kpc_per_pix
    ywinran_kpc = float([0 - (center_axes[1] - 0.5 - (plotwin[1] - 1)), \
                         dywin - (center_axes[1] - 0.5 - (plotwin[1] - 1))]) \
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
        map_x_hst = rebin(np.arange(size_subim[0]) + 1, size_subim[0], size_subim[1])
        map_y_hst = rebin(np.transpose(np.arange(size_subim[1]) + 1),
                        size_subim[0], size_subim[1])
#       Locations of [0,0] point on axes and nuclei, in HST pixels (single-offset
#       indices).
        center_axes_hst = (center_axes - 0.5) * float(size_subim[1] / dx) + 0.5
        center_nuclei_hst = (center_nuclei - 0.5) * float(size_subim[1] / dx) + 0.5
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



      