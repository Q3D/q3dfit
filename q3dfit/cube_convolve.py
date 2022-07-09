import numpy as np
from q3dfit.readcube import Cube
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter,median_filter,convolve
from astropy.io import fits
from astropy.nddata import block_reduce

def circular_mask(r):
    '''circular mask'''
    mask = np.fromfunction(lambda i, j: ((i-r)**2 + (j-r)**2) < r**2, (r*2+1, r*2+1), dtype=int)

    return mask


def convolve_cube(infits, datext=0, varext=1, dqext=2, wavext=None,
                   wmapext=None, plot=True, waveunit_in='micron',
                   waveunit_out='micron',wavelength_segments=[],outfits=None):
    
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
    

    cube_convolved = flux.copy()
    sizes = np.linspace(0.5,0,flux.shape[2])
    method = 'Gaussian'
    for i in np.arange(0,flux.shape[2]):
        if method == 'Gaussian':
           cube_convolved[:,:,i] = gaussian_filter(cube_convolved[:,:,i],1+sizes[i])
        if method == 'median-circular':
            cube_convolved[:,:,i] = median_filter(cube_convolved[:,:,i],footprint=circular_mask(10))
        if method == 'circular':
            cube_convolved[:,:,i] = convolve(cube_convolved[:,:,i],circular_mask(2+np.int(sizes[i])))

#    plt.plot(np.log10(flux[15,15]),label='original')
#    plt.plot(np.log10(cube_convolved[15,15]),label='convolved')
#    plt.legend()

    if outfits != None:
        hdu = fits.open(infits)
        if 'SCI' in hdu:
            hdu['SCI'].data = cube_convolved
        else:
            hdu[datext].data = cube_convolved
        hdu.writeto(outfits,overwrite=True)
    return cube_convolved

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
            hdu['SCI'].data = flux_binned
            hdu['ERR'].data = var_binned
            hdu['DQ'].data = dq_binned
        else:
            hdu[datext].data = flux_binned
            hdu[varext].data = var_binned
            hdu[dqext].data = dq_binned
        
        hdu.writeto(outfits,overwrite=True)

    return flux_binned

if __name__ == "__main__":
    #    cube = Cube(infile='../../NIRSpec_ETC_sim/NIRSpec_etc_cube_both_2.fits',datext=0, varext=1, dqext=2, wavext=None, wmapext=None)
    
    wavelength_segments = [[10,100],[200,250]]
    infits = '../NIRSpec_ETC_sim/NIRSpec_etc_cube_both_2.fits'
    infits = '../NIRspec_sim/NRS00001-QG-F100LP-G140H_comb_1234_g140h-f100lp_s3d.fits'
    outfits = '../NIRspec_sim/NRS00001-QG-F100LP-G140H_comb_1234_g140h-f100lp_s3d_smoothed.fits'
    outfits_bined = '../NIRspec_sim/NRS00001-QG-F100LP-G140H_comb_1234_g140h-f100lp_s3d_binned.fits'
    smoothed = convolve_cube(infits, datext=1, varext=2, dqext=3, wavext=None,
                                    wmapext=None, plot=True,
                                    waveunit_in='micron',wavelength_segments=[],waveunit_out='micron',outfits=outfits)



    bined = bin_cube(infits,[2,2,1], datext=1, varext=2, dqext=3, wavext=None,
                                    wmapext=None, plot=True,
                                    waveunit_in='micron',waveunit_out='micron',outfits=outfits_bined)
    
