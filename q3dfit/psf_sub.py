import numpy as np
import matplotlib.pyplot as plt
from q3dfit.readcube import Cube
from astropy.io import fits
from q3dfit.cube_convolve import convolve_cube

def make_PSF(datacube,cube,wavelength):

    wave = cube.wave

    flux = datacube
    err = cube.var
    dq = cube.dq

    m = cube.cdelt
    b = cube.crval

    slices = []
    for i in wavelength:
        px_min = np.int((i[0]-b)/m)
        px_max = np.int((i[1]-b)/m)
        slices.append(np.sum(flux[:,:,px_min:px_max+1],axis=2))

    psf = np.mean(slices,axis=0)
    return psf
                      
def subtract_psf(datacube,psf):
    '''Subtracts the PSF from the image at each wavelength slice and returns a PSF subtracted cube'''
    temp_cube=datacube.copy()
    
    if len(psf.shape) == 2:#checking if PSF is a 2D or 3D array
        psf = psf/psf.max() #normalize PSF
        loc_psf_max = np.where(psf == psf.max())
        for num in np.arange(0,datacube.shape[2]):
            scale=datacube[loc_psf_max[0][0],loc_psf_max[1][0],num]
            temp_cube[:,:,num]+=-psf*scale

    if len(psf.shape) == 3:#checking if PSF is a 2D or 3D array
        for num in np.arange(0,datacube.shape[2]):
            psf_num = psf[:,:,num]/psf[:,:,num].max() #normalize PSF at a specific wavelength location
            loc_psf_max = np.where(psf_num == psf_num.max) #location of maximum PSF
            scale=datacube[loc_psf_max[0][0],loc_psf_max[1][0],num]
            temp_cube[:,:,num]+=-psf_num*scale
    
    return temp_cube


if __name__ == "__main__":
    #    cube = Cube(infile='../../NIRSpec_ETC_sim/NIRSpec_etc_cube_both_2.fits',datext=0, varext=1, dqext=2, wavext=None, wmapext=None)
    
    wavelength_segments = [[10,100],[200,250]]
    infits = '../NIRSpec_ETC_sim/NIRSpec_etc_cube_both_2.fits'
    #    infits = '../NIRspec_sim/NRS00001-QG-F100LP-G140H_comb_1234_g140h-f100lp_s3d.fits'
    smoothed = convolve_cube(infits, datext=0, varext=1, dqext=2, wavext=None,
                             wmapext=None, plot=True,
                             waveunit_in='micron',wavelength_segments=[],waveunit_out='micron')


    cube = Cube(infits,datext=0, varext=1, dqext=2,wmapext=None,)
    psf = make_PSF(smoothed,cube,[[1.637,1.639]])

    psf_sub_data = subtract_psf(smoothed,psf)
    HDU= fits.PrimaryHDU(psf_sub_data)
    HDU.writeto('../NIRspec_sim/NRS00001-QG-F100LP-G140H_comb_1234_g140h-f100lp_s3d_psfsub.fits',overwrite=True)


