import numpy as np
from q3dfit.readcube import Cube
import matplotlib.pyplot as plt

def select_spaxels(infits, datext=0, varext=1, dqext=2, wavext=None,
                   wmapext=None, plot=True, waveunit_in='micron',
                   waveunit_out='micron',wavelength_segments=[]):

    cube = Cube(infits, datext=datext, dqext=dqext,
            varext=varext, wavext=wavext, wmapext=wmapext,
            waveunit_in=waveunit_in, waveunit_out=waveunit_out)

    #check for inf values in cubes
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



    if len(wavelength_segments) == 0:
        white_light_img = np.sum(flux,axis=2)
        white_light_img_std = np.sum(err,axis=2)**0.5
        SNR_map = white_light_img/white_light_img_std
        spaxels_to_fit = np.where(SNR_map>10.)

    if len(wavelength_segments) > 0:
        SNR_map = np.zeros((flux.shape[0],flux.shape[1]))
        white_light_img = np.zeros((flux.shape[0],flux.shape[1]))
        white_light_img_var = np.zeros((flux.shape[0],flux.shape[1]))
        for i in wavelength_segments:
            white_light_img += np.sum(flux[:,:,i[0]:i[1]],axis=2)
            white_light_img_var += np.sum(err[:,:,i[0]:i[1]],axis=2)

        SNR_map = white_light_img/white_light_img_var**0.5
        spaxels_to_fit = np.where(SNR_map>5.)


    if plot == True:
        plt.imshow(SNR_map,vmax=5,origin='lower')
        plt.plot(spaxels_to_fit[1],spaxels_to_fit[0],'x')

    number_spaxels_to_fit = len(spaxels_to_fit[1])
    print('Number of spaxels to fit: ' + str(number_spaxels_to_fit))


    return list(spaxels_to_fit[0]),list(spaxels_to_fit[1])

if __name__ == "__main__":
#    cube = Cube(infile='../../NIRSpec_ETC_sim/NIRSpec_etc_cube_both_2.fits',datext=0, varext=1, dqext=2, wavext=None, wmapext=None)

    wavelength_segments = [[10,100],[200,250]]
    infits = '../NIRSpec_ETC_sim/NIRSpec_etc_cube_both_2.fits'
    spaxles_to_fit = select_spaxels(infits, datext=0, dqext=2, varext=1, wavext=None,
               wmapext=None, plot=True, waveunit_in='micron',
               waveunit_out='micron',wavelength_segments=[])


##gd_indx_1 = set(np.where(flux != 0.)[0])
##gd_indx_2 = set(np.where(err > 0.)[0])
#gd_indx_3 = set(np.where(np.isfinite(flux)))
#gd_indx_4 = set(np.where(np.isfinite(err)))
#gd_indx_5 = set(np.where(dq == 1))
#gd_indx_full = gd_indx_3.intersection(gd_indx_4,
#                                       gd_indx_5)
#
#gd_indx_full = list(gd_indx_full)

#gd_indx_8 = set(np.where(wlambda >= fitran_tmp[0])[0])
#gd_indx_9 = set(np.where(wlambda <= fitran_tmp[1])[0])
