# -*- coding: utf-8 -*-

import numpy as np

from q3dfit.linelist import linelist
from q3dfit.q3dutil import lmlabel
from astropy.constants import c
from math import pi


def airtovac(wv, waveunit='Angstrom'):
    """
    Takes an array of wavelengths in air and converts them to vacuum
        using eq. 3 from Morton et al. 1991 ApJSS 77 119

    Parameters
    ----------
    wv : ndarray, shape (n,)
       Input wavelengths. Array of real elements with shape (n,)
       where 'n' is the number of wavelengths
    waveunit : str, optional
       Wavelength unit, could be 'Angstrom' or 'micron',
       default is Angstrom

    Returns
    -------
    ndarray, shape (n,)
       An array of the same dimensions as the input and in the same units
       as the input

    References
    ----------
    .. Morton et al. 1991 ApJSS 77 119

    Examples
    --------
    >>>wv=np.arange(3000,7000,1)
    >>>vac_wv=airtovac(wv)
    array([3000.87467224, 3001.87492143, 3002.87517064, ..., 6998.92971915,
       6999.92998844, 7000.93025774])

    """
    x=wv
    # get x to be in Angstroms for calculations if it isn't already
    if ((waveunit!='Angstrom') & (waveunit!='micron')):
        print ('Wave unit ',waveunit,' not recognized, returning Angstroms')
    if (waveunit=='micron'):
        x=wv*1.e4

    tmp=1.e4/x
    y=x*(1.+6.4328e-5+2.94981e-2/(146.-tmp**2)+2.5540e-4/(41.-tmp**2))
    # vacuum wavelengths are indeed slightly longer

    # get y to be in the same units as the input:
    if (waveunit=='micron'):
        y=y*1.e-4

    return(y)


def cmpcvdf(wave, sigma, pkflux, ncomp, line, zref,
            vlimits=[-1e4, 1e4], vstep=1.):
    """
    Computes cumulative velocity distribution function for a given line, for
    each spaxel in a data cube.

    Parameters
    ----------
    wave : array(ncols, nrows, ncomp)
    sigma : array(ncols, nrows, ncomp)
    pkflux : array(ncols, nrows, ncomp)
    ncomp : array(ncols, nrows)
    line : str
    zref : float

    Returns
    -------
    modvel : array(nmod)
    vdf : array(ncols, nrows, nmod)
        Velocity distribution in flux space. 3D data with two dimensions of
        imaging plane and third of model points.
    cvdf : array(ncols, nrows, nmod)
        Cumulative velocity distribution function.

    Notes
    -----
    """

    # these are allegedly the smallest numbers recognized
    minexp = -310
    # this is the experimentally determined limit for when
    # I can take a log of a 1e-minexp
    # mymin = np.exp(minexp)

    # establish the velocity array from the inputs or from the defaults
    modvel = np.arange(vlimits[0], vlimits[1]+vstep, vstep)
    beta = modvel/c.to('km/s').value
    dz = np.sqrt((1. + beta)/(1. - beta)) - 1.

    # central (rest) wavelength of the line in question
    listlines = linelist([line])
    cwv = listlines['lines'].value[0]
    modwaves = cwv*(1. + dz)*(1. + zref)

    # output arrays
    size_cube = np.shape(pkflux)
    nmod = np.size(modvel)

    vdf = np.zeros((size_cube[0], size_cube[1], nmod))
    #vdferr = np.zeros((size_cube[0], size_cube[1], nmod))
    cvdf = np.zeros((size_cube[0], size_cube[1], nmod))
    #cvdferr = np.zeros((size_cube[0], size_cube[1], nmod))
    for i in range(np.max(ncomp)):
        rbpkflux = np.repeat((pkflux[:, :, i])[:, :, np.newaxis], nmod, axis=2)
        rbsigma = np.repeat((sigma[:, :, i])[:, :, np.newaxis], nmod, axis=2)
        rbpkwave = np.repeat((wave[:, :, i])[:, :, np.newaxis], nmod, axis=2)
        rbncomp = np.repeat(ncomp[:, :, np.newaxis], nmod, axis=2)
        rbmodwave = \
            np.broadcast_to(modwaves, (size_cube[0], size_cube[1], nmod))

        inz = ((rbsigma > 0) & (rbsigma != np.nan) &
               (rbpkwave > 0) & (rbpkwave != np.nan) &
               (rbpkflux > 0) & (rbpkflux != np.nan) &
               (rbncomp > i))
        if np.sum(inz) > 0:
            exparg = np.zeros((size_cube[0], size_cube[1], nmod)) - minexp
            exparg[inz] = ((rbmodwave[inz]/rbpkwave[inz] - 1.) /
                           (rbsigma[inz]/c.to('km/s').value))**2. / 2.
            i_no_under = (exparg < -minexp)
            if np.sum(i_no_under) > 0:
                vdf[i_no_under] += rbpkflux[i_no_under] * \
                    np.exp(-exparg[i_no_under])
                # df_norm = rbpkfluxerrs[i_no_under]*np.exp(-exparg[i_no_under])
                #term1 = rbpkflux[i_no_under] * \
                #    np.abs(rbmodwave[i_no_under] - rbpkwave[i_no_under])
                #term2 = rbsigma[i_no_under]/c.to('km/s').value * \
                #    rbpkwave[i_no_under]
                # df_wave = term1/(term2**2)*rbpkwaveerrs[i_no_under]*np.exp(-exparg[i_no_under])
                #term3 = rbpkflux[i_no_under] * \
                #    (rbmodwave[i_no_under] - rbpkwave[i_no_under])**2
                #term4 = rbsigma[i_no_under]/c.to('km/s').value * \
                #    rbpkwave[i_no_under]
                #df_sig = term3/term4**2*rbsigmaerrs[i_no_under]/rbsigmas[i_no_under]*np.exp(-exparg[i_no_under])
                #dfsq = np.zeros((size_cube[0],size_cube[1],nmod))
                #dfsq = dfsq[i_no_under]
                #i_no_under_2 = ((df_norm > mymin) & (df_wave > mymin) & (df_sig > mymin))
                #if (sum(i_no_under_2)>0):
                #    dfsq[i_no_under_2] = (df_norm[i_no_under_2])**2+(df_wave[i_no_under_2])**2+(df_sig[i_no_under_2])**2
                #emlcvdf['fluxerr'][line][i_no_under] += dfsq

        #inz = (emlcvdf['flux'][line] > 0)
        #if (sum(inz)>0):
        #    emlcvdf['fluxerr'][line][inz] = np.sqrt(emlcvdf['fluxerr'][line][inz])

    # size of each model bin
    dmodwaves = modwaves[1:nmod] - modwaves[0:nmod-1]
    # supplement with the zeroth element to make the right length
    dmodwaves = np.append(dmodwaves[0], dmodwaves)
    # rebin to full cube
    rbdmodwaves = \
        np.broadcast_to(dmodwaves, (size_cube[0], size_cube[1], nmod))
    fluxnorm = vdf * rbdmodwaves
    #fluxnormerr = emlcvdf['fluxerr'][line]*dmodwaves
    fluxint = np.repeat((np.sum(fluxnorm, 2))[:, :, np.newaxis], nmod, axis=2)
    inz = fluxint != 0
    if np.sum(inz) > 0:
        fluxnorm[inz] /= fluxint[inz]
        #fluxnormerr[inz] /= fluxint[inz]

    cvdf[:, :, 0] = fluxnorm[:, :, 0]
    for i in range(1, nmod):
        cvdf[:, :, i] = cvdf[:, :, i-1] + fluxnorm[:, :, i]
        #emlcvdf['cvdferr'][line] = fluxnormerr

    return modvel, vdf, cvdf


def cmplin(q3do, line, comp, velsig=False):
    '''
    Function takes four parameters and returns specified flux

    Parameters:
    q3do : obj
    line : str
    comp : int
    velsig : bool

    Returns
    -------
    flux : float
    '''

    lmline = lmlabel(line)
    mName = '{0}_{1}_'.format(lmline.lmlabel, comp)
    gausspar = np.zeros(3)
    gausspar[0] = q3do.param[mName+'flx']
    gausspar[1] = q3do.param[mName+'cwv']
    gausspar[2] = q3do.param[mName+'sig']
    if velsig:
        gausspar[2] = gausspar[2] * gausspar[1]/c.to('km/s').value

    flux = gaussian(q3do.wave, gausspar)

    return flux


def cmpweq(instr, linelist, doublets=None):
    """
    Compute equivalent widths for the specified emission lines.
    Uses models of emission lines and continuum, and integrates over both using
    the "rectangle rule."

    Parameters
    ----------
    instr : dict
        Contains output of IFSF_FITSPEC.
    linelist: astropy Table
        Contains the output from linelist.
    doublets : ndarray
        A 2D array of strings combining doublets in pairs if it's
        desirable to return the total eq. width,
        for example:
            doublets=[['[OIII]4959','[OIII]5007'],['[SII]6716','[SII]6731']]
            or
            doublets=['[OIII]4959','[OIII]5007']
        default: None

    Returns
    -------
    ndarray
        Array of equivalent widths.

    """

    ncomp=instr['param'][1]
    nlam=len(instr['wave'])
    lines=linelist['name']

    tot={}
    comp={}
    dwave=instr['wave'][1:nlam]-instr['wave'][0:nlam-1]
    for line in lines:
        tot[line]=0.
        comp[line]=np.zeros(ncomp)
        for j in range(1, ncomp+1):
            modlines=cmplin(instr,line,j,velsig=True)
            if (len(modlines)!=1):
                comp[line][j-1]=np.sum(-modlines[1:nlam]/instr['cont_fit'][1:nlam]*dwave)
            else: comp[line][j-1]=0.
            tot[line]+=comp[line][j-1]

    #Special doublet cases: combine fluxes from each line
    if (doublets!=None):
        # this shouldn't hurt and should make it easier
        doublets=np.array(doublets)
        sdoub=np.shape(doublets)
        # this should work regardless of whether a single doublet is surrounded by single or double square parentheses:
        if (len(sdoub)==1):
            ndoublets=1
            # and let's put this all into a 2D array shape for consistency so we are easily able to iterate
            doublets=[doublets]
        else:
            ndoublets=sdoub[0]
        for i in range(ndoublets):
            if ((doublets[i][0] in lines) and (doublets[i][1] in lines)):
                #new line label
                dkey = doublets[i][0]+'+'+doublets[i][1]
                #add fluxes
                tot[dkey] = tot[doublets[i][0]]+tot[doublets[i][1]]
                comp[dkey] = comp[doublets[i][0]]+comp[doublets[i][1]]

    return({'tot': tot,'comp': comp})


def gaussian(xi, parms):
    '''
    '''
    a = parms[0]  # amp
    b = parms[1]  # mean
    c = parms[2]  # standard dev

    # Anything higher-precision than this (e.g., float64) slows things down
    # a bunch. longdouble completely chokes on lack of memory.
    arg = np.array(-0.5 * ((xi - b)/c)**2, dtype=np.float32)
    g = a * np.exp(arg)

    return g


def gaussflux(norm, sigma, normerr=None, sigerr=None):

    flux = norm * sigma * np.sqrt(2. * pi)
    fluxerr = 0.

    if normerr is not None and sigerr is not None:
        fluxerr = flux*np.sqrt((normerr/norm)**2. + (sigerr/sigma)**2.)

    outstr = {'flux': flux, 'flux_err': fluxerr}

    return outstr
