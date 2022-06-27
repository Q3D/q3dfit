import numpy as np
from q3dfit.readcube import Cube
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter,median_filter,convolve
from astropy.io import fits


def circular_mask(r):
    '''circular mask'''
    mask = np.fromfunction(lambda i, j: ((i-r)**2 + (j-r)**2) < r**2, (r*2+1, r*2+1), dtype=int)

    return mask


def convolve_cube(infits, datext=0, varext=1, dqext=2, wavext=None,
                   wmapext=None, plot=True, waveunit_in='micron',
                   waveunit_out='micron',wavelength_segments=[]):
    
    cube = Cube(infits, datext=datext, dqext=dqext,
                varext=varext, wavext=wavext, wmapext=wmapext,
                waveunit_in=waveunit_in, waveunit_out=waveunit_out)

    flux = cube.dat
    err = cube.var
    dq = cube.dq


    cube_convolved = flux.copy()
    sizes = np.linspace(2,0,flux.shape[2])
    method = 'circular'
    for i in np.arange(0,flux.shape[2]):
        if method == 'Gaussian':
           cube_convolved[:,:,i] = gaussian_filter(cube_convolved[:,:,i],2.0+sizes[i])
        if method == 'median-circular':
            cube_convolved[:,:,i] = median_filter(cube_convolved[:,:,i],footprint=circular_mask(10))
        if method == 'circular':
            cube_convolved[:,:,i] = convolve(cube_convolved[:,:,i],circular_mask(2+np.int(sizes[i])))

    plt.plot(flux[15,15],label='original')
    plt.plot(cube_convolved[15,15],label='convolved')
    plt.legend()

    return cube_convolved

if __name__ == "__main__":
    #    cube = Cube(infile='../../NIRSpec_ETC_sim/NIRSpec_etc_cube_both_2.fits',datext=0, varext=1, dqext=2, wavext=None, wmapext=None)
    
    wavelength_segments = [[10,100],[200,250]]
    infits = '../NIRSpec_ETC_sim/NIRSpec_etc_cube_both_2.fits'
    infits = '../NIRspec_sim/NRS00001-QG-F100LP-G140H_comb_1234_g140h-f100lp_s3d.fits'
    smoothed = convolve_cube(infits, datext=1, varext=2, dqext=3, wavext=None,
                                    wmapext=None, plot=True,
                                    waveunit_in='micron',wavelength_segments=[],waveunit_out='micron')



    HDU= fits.PrimaryHDU(smoothed)
    HDU.writeto('../NIRspec_sim/NRS00001-QG-F100LP-G140H_comb_1234_g140h-f100lp_s3d_smoothed.fits',overwrite=True)



