import numpy as np
import matplotlib.pyplot as plt


def subtract_psf(datacube,psf):
    '''Subtracts the PSF from the image at each wavelength slice and returns a PSF subtracted cube'''
    temp_cube=datacube.copy()
    
    if len(psf.shape) == 2:#checking if PSF is a 2D or 3D array
        psf = psf/psf.max() #normalize PSF
        loc_psf_max = where(psf == psf.max)
        for num in np.arange(0,datacube.shape[2]):
            scale=datacube[loc_psf_max[0][0],loc_psf_max[0][1],num]
            temp_cube[:,:,num]+=-psf*scale

    if len(psf.shape) == 3:#checking if PSF is a 2D or 3D array
        for num in np.arange(0,datacube.shape[2]):
            psf_num = psf[:,:,num]/psf[:,:,num].max() #normalize PSF at a specific wavelength location
            loc_psf_max = where(psf_num == psf_num.max) #location of maximum PSF
            scale=datacube[loc_psf_max[0][0],loc_psf_max[0][1],num]
            temp_cube[:,:,num]+=-psf_num*scale
    
    return temp

