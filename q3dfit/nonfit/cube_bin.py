# -*- coding: utf-8 -*-

import numpy as np

from astropy.io import fits
from astropy.nddata import block_reduce
from q3dfit.readcube import Cube


def circular_mask(r):
    '''circular mask, integer radii only, no fractional flux'''
    mask = np.fromfunction(lambda i, j: ((i-r)**2 + (j-r)**2) <= r**2,
                           (r*2+1, r*2+1), dtype=int)
    return mask


def box_mask(r):
    '''box mask, integer sides only'''
    return np.ones((r*2, r*2))


def bin_cube(infits,bin_value=[2,2,1], datext=0, varext=1, dqext=2, wavext=None,
             wmapext=None, plot=True, waveunit_in='micron',
             waveunit_out='micron',outfits=None):

    cube = Cube(infits, datext=datext, dqext=dqext,
                varext=varext, wavext=wavext, wmapext=wmapext,
                waveunit_in=waveunit_in, waveunit_out=waveunit_out)

    flux = cube.dat
    err = cube.var
    dq = cube.dq

    indx_bd = np.where((flux == np.inf) | (err == np.inf))
    flux[indx_bd] = 0.
    err[indx_bd] = 0.

    #check for nan values
    indx_bd = np.where((flux == np.nan) | (err == np.nan))
    flux[indx_bd] = 0.
    err[indx_bd] = 0.

    #checking bad data quality flag
    indx_bd = np.where(dq!=0)
    flux[indx_bd] = 0.
    err[indx_bd] = 0.

    flux_binned = block_reduce(flux,bin_value)
    var_binned = block_reduce(err,bin_value)
    dq_binned = np.ones((flux_binned.shape[0],flux_binned.shape[1],flux_binned.shape[2]))
    if outfits != None:
        hdu = fits.open(infits)
        if 'SCI' and 'ERR' and 'DQ' in hdu:
            hdu['SCI'].data = flux_binned.T
            hdu['ERR'].data = var_binned.T
            hdu['DQ'].data = dq_binned.T
        else:
            hdu[datext].data = flux_binned.T
            hdu[varext].data = var_binned.T
            hdu[dqext].data = dq_binned.T

        hdu.writeto(outfits,overwrite=True)

    return flux_binned
