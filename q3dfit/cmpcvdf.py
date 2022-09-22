# -*- coding: utf-8 -*-

import numpy as np
import pdb

from q3dfit.linelist import linelist
from astropy.constants import c


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

    # I am sure I need to do something else here for the 'bad' values...
    bad = 1e99
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

        inz = ((rbsigma > 0) & (rbsigma != bad) &
               (rbpkwave > 0) & (rbpkwave != bad) &
               (rbpkflux > 0) & (rbpkflux != bad) &
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
