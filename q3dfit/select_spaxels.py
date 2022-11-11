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
        spaxels_to_fit = np.where(SNR_map>5.)

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


    return spaxels_to_fit,SNR_map


def mom8_map(infits, datext=0, varext=1, dqext=2, wavext=None,
                   wmapext=None, plot=True, waveunit_in='micron',
                   waveunit_out='micron',wavelength_segments=[],SNR_cut = 3):

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
    indx_bd = np.where((np.isnan(flux) == True) | (np.isnan(err) == True))
    flux[indx_bd] = 0.
    err[indx_bd] = 0.

    #checking bad data quality flag
    indx_bd = np.where(dq!=0)
    flux[indx_bd] = 0.
    err[indx_bd] = 0.

    if len(wavelength_segments) > 0:
        SNR_map = np.zeros((flux.shape[0],flux.shape[1]))
        mom8 = np.zeros((flux.shape[0],flux.shape[1]))
        mom8_var = np.zeros((flux.shape[0],flux.shape[1]))
        for i in wavelength_segments:
            mom8 += np.max(flux[:,:,i[0]:i[1]],axis=2)
            print(mom8.max())
            mom8_var += np.mean(err[:,:,i[0]:i[1]],axis=2)
            print(mom8_var.max())

        SNR_map = mom8/mom8_var**0.5
        indx_bd_SNR = np.where((SNR_map == np.inf) | (SNR_map == -np.inf) )
        SNR_map[indx_bd_SNR] = 0.

        indx_bd_SNR = np.where((np.isnan(SNR_map) == True))
        SNR_map[indx_bd_SNR] = 0.

        spaxels_to_fit = np.where(SNR_map>SNR_cut)


    return SNR_map,spaxels_to_fit

def M_N(v,i_v,m_0,m_1,n):
    M_N  = sum(i_v*(v-m_1)**n)/m_0
    if np.isnan(M_N):
        M_N = 0
    return M_N

if __name__ == "__main__":
#    cube = Cube(infile='../../NIRSpec_ETC_sim/NIRSpec_etc_cube_both_2.fits',datext=0, varext=1, dqext=2, wavext=None, wmapext=None)

#    wavelength_segments = [[10,100],[200,250]]
#    infits = '../NIRSpec_ETC_sim/NIRSpec_etc_cube_both_2.fits'
#    spaxles_to_fit = select_spaxels(infits, datext=0, dqext=2, varext=1, wavext=None,
#               wmapext=None, plot=True, waveunit_in='micron',
#               waveunit_out='micron',wavelength_segments=wavelength_segments)

    infits = '/Volumes/My Passport for Mac/JWST/ERS/SDSS1652/jw01335/reduced_level_2/montage_new_CRDS_no_imprint/NRS1NRS2_s3d_psfsub.fits'
    wavelength_segments = [[670,730]]
    SNR_map,spaxels_to_fit = mom8_map(infits, datext=0, varext=1, dqext=None, wavext=None,
                   wmapext=None, plot=True, waveunit_in='micron',
                   waveunit_out='micron',wavelength_segments=wavelength_segments)
    SNR_map[0:6] = 0
    SNR_map[79:] = 0
    SNR_map[:,72:] = 0
    cube = Cube(infile=infits,datext=0, varext=1, dqext=2, wavext=None, wmapext=None)
    data = cube.dat
    I = np.zeros((data.shape[0],data.shape[1]))
    V = np.zeros((data.shape[0],data.shape[1]))
    D = np.zeros((data.shape[0],data.shape[1]))
    loc_good_I = np.where((SNR_map>3)&(SNR_map!=np.inf))
    max_loc_I=np.argmax(data[:,:,670:730],axis=2)
    loc_max_spec = max_loc_I[loc_good_I]
    spec_good = data[loc_good_I][:,670:730]

    M_0 = np.zeros(spec_good.shape[0])
    M_1 = np.zeros(spec_good.shape[0])
    M_2 = np.zeros(spec_good.shape[0])
    sigma_I = 5
    v = np.arange(670,730,1.)#-604
    for i in np.arange(0,len(M_0)):
        spec_good[i] = spec_good[i]
        M_0[i] = np.sum(spec_good[i][loc_max_spec[i]-sigma_I:loc_max_spec[i]+sigma_I])
        M_1[i] = np.sum(v[loc_max_spec[i]-sigma_I:loc_max_spec[i]+sigma_I]*spec_good[i][loc_max_spec[i]-sigma_I:loc_max_spec[i]+sigma_I])/M_0[i]
        M_2[i] = M_N(v[loc_max_spec[i]-sigma_I:loc_max_spec[i]+sigma_I],spec_good[i][loc_max_spec[i]-sigma_I:loc_max_spec[i]+sigma_I],M_0[i],M_1[i],2)**0.5


    z=2.9489
    #v = np.arange(0,data.shape[2])
    v = np.arange(0,data.shape[2])*0.000395999988541007 + 1.700016528205231
    v = (v-0.5007*(1+z))/(0.5007*(1+z))*3e5
    loc_good_II = (loc_good_I[0][False==np.isnan(M_2)],loc_good_I[1][False==np.isnan(M_2)])
    spec_good = data[loc_good_II]
    M_0 = M_0[False==np.isnan(M_2)]
    M_1 = M_1[False==np.isnan(M_2)]
    M_2 = M_2[False==np.isnan(M_2)]

    loc_good_II = (loc_good_II[0][np.where(M_2>1.)],loc_good_II[1][np.where(M_2>1.)])
    spec_good = data[loc_good_II]
    M_0 = M_0[np.where(M_2>1.)]
    M_1 = M_1[np.where(M_2>1.)]
    M_2 = M_2[np.where(M_2>1.)]


    N_sigma = 3
    for i in np.arange(0,spec_good.shape[0]):

        spec_good[i] = spec_good[i]
        M_0[i] = np.sum(spec_good[i][np.int(M_1[i]-M_2[i]*N_sigma):int(M_1[i]+M_2[i]*N_sigma)])
        M_1[i] = np.sum(v[np.int(M_1[i]-M_2[i]*N_sigma):np.int(M_1[i]+M_2[i]*N_sigma)]*spec_good[i][np.int(M_1[i]-M_2[i]*N_sigma):np.int(M_1[i]+M_2[i]*N_sigma)])/M_0[i]
        M_2[i] = M_N(v[np.int(M_1[i]-M_2[i]*N_sigma):np.int(M_1[i]+M_2[i]*N_sigma)],spec_good[i][np.int(M_1[i]-M_2[i]*N_sigma):np.int(M_1[i]+M_2[i]*N_sigma)],M_0[i],M_1[i],2)**0.5

    I[loc_good_II] = M_0
    V[loc_good_II] = M_1
    D[loc_good_II] = M_2

    loc_zero = np.where(V==0.)
    V[loc_zero] = np.nan
