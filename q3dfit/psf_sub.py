import numpy as np
import matplotlib.pyplot as plt
from q3dfit.readcube import Cube
from astropy.io import fits
from q3dfit.cube_convolve import convolve_cube

def make_PSF(datacube,cube,wavelength,plot=False):

    wave = cube.wave

    flux = cube.dat#datacube
    err = cube.var
    dq = cube.dq
    
    m = cube.cdelt
    b = cube.crval
    indx_bd = np.where((flux == np.inf) | (err == np.inf))
    flux[indx_bd] = 0.
    err[indx_bd] = 0.
    
    #check for nan values
    indx_bd = np.where((np.isnan(flux) == True) | (np.isnan(err) == True))
    flux[indx_bd] = 0.
    err[indx_bd] = 0.
    
    #checking bad data quality flag
    indx_bd = np.where(dq!=0)
    flux[indx_bd] = 0.
    err[indx_bd] = 0.

    slices = []
    for i in wavelength:
        px_min = np.int((i[0]-b)/m)
        px_max = np.int((i[1]-b)/m)
        slices.append(np.sum(flux[:,:,px_min:px_max+1],axis=2))

    psf = np.mean(slices,axis=0)
    loc_psf_max = np.where(psf == psf.max())

    if plot == True:
        lambda_range=np.array([])
        for i in wavelength:
            lambda_range=np.concatenate([lambda_range,np.arange(i[0],i[1])])
            plt.fill_between(np.arange(i[0],i[1],m),datacube[loc_psf_max[0][0],loc_psf_max[1][0]][np.int((i[0]-b)/m):np.int((i[1]-b)/m)],color='green')#if we want wavelength
#            plt.fill_between(np.arange(i[0],i[1]),datacube[loc_psf_max[0][0],loc_psf_max[1][0]][np.int((i[0]-b)/m):np.int((i[1]-b)/m)],color='green')#if we want pixel number

        plt.step(np.arange(0,datacube.shape[2])*m+b,datacube[loc_psf_max[0][0],loc_psf_max[1][0]],color='blue')#wavelength
    
    return psf
                      
def subtract_psf(infits,psf,datext=0, varext=1, dqext=2, wavext=None,
                 wmapext=None, waveunit_in='micron',
                 waveunit_out='micron', outfits = None):
    '''Subtracts the PSF from the image at each wavelength slice and returns a PSF subtracted cube'''

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
    indx_bd = np.where((np.isnan(flux) == True) | (np.isnan(err) == True))
    flux[indx_bd] = 0.
    err[indx_bd] = 0.
    
    #checking bad data quality flag
    indx_bd = np.where(dq!=0)
    flux[indx_bd] = 0.
    err[indx_bd] = 0.
    
    temp_cube=flux.copy()
    
    if len(psf.shape) == 2:#checking if PSF is a 2D or 3D array
        psf = psf/psf.max() #normalize PSF
        loc_psf_max = np.where(psf == psf.max())
        for num in np.arange(0,flux.shape[2]):
            scale=flux[loc_psf_max[0][0],loc_psf_max[1][0],num]
            temp_cube[:,:,num]+=-psf*scale

    if len(psf.shape) == 3:#checking if PSF is a 2D or 3D array
        for num in np.arange(0,datacube.shape[2]):
            psf_num = psf[:,:,num]/psf[:,:,num].max() #normalize PSF at a specific wavelength location
            loc_psf_max = np.where(psf_num == psf_num.max) #location of maximum PSF
            scale=flux[loc_psf_max[0][0],loc_psf_max[1][0],num]
            temp_cube[:,:,num]+=-psf_num*scale
    
    if outfits != None:
        hdu = fits.open(infits)
        if 'SCI' in hdu:
            hdu['SCI'].data = temp_cube.T
            hdu.writeto(outfits,overwrite=True)
        else:
            hdu[datext].data = temp_cube.T
            hdu.writeto(outfits,overwrite=True)


    return temp_cube


if __name__ == "__main__":
    #    cube = Cube(infile='../../NIRSpec_ETC_sim/NIRSpec_etc_cube_both_2.fits',datext=0, varext=1, dqext=2, wavext=None, wmapext=None)
    
    wavelength_segments = [[10,100],[200,250]]
    infits = '../NIRSpec_ETC_sim/NIRSpec_etc_cube_both_2.fits'
    outfits = '../NIRSpec_ETC_sim/NIRSpec_etc_cube_both_2_psf_sub.fits'
    #    infits = '../NIRspec_sim/NRS00001-QG-F100LP-G140H_comb_1234_g140h-f100lp_s3d.fits'
    smoothed = convolve_cube(infits, datext=0, varext=1, dqext=2, wavext=None,
                             wmapext=None, plot=True,
                             waveunit_in='micron',wavelength_segments=[],waveunit_out='micron')


    cube = Cube(infits,datext=0, varext=1, dqext=2,wmapext=None,)
    psf = make_PSF(cube.dat,cube,[[1.637,1.639]],plot=True)

    psf_sub_data = subtract_psf(infits,psf,datext=0, varext=1, dqext=2, wavext=None,
                            wmapext=None, waveunit_in='micron',
                            waveunit_out='micron', outfits = outfits)

    NB_img = np.mean(psf_sub_data[:,:,2832:2840],axis=2)
    plt.figure(333)
    plt.imshow(np.log10(NB_img),origin='lower')

#    HDU= fits.PrimaryHDU(psf_sub_data)
#    HDU.writeto('../NIRspec_sim/NRS00001-QG-F100LP-G140H_comb_1234_g140h-f100lp_s3d_psfsub.fits',overwrite=True)


