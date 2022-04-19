from numpy import *
from q3dfit.makeqsotemplate import makeqsotemplate
import q3dfit.readcube as readcube
from matplotlib.pyplot import *
from scipy.ndimage import gaussian_filter
from skimage.transform import rescale
from scipy.ndimage import fourier_shift, shift, rotate
from astropy.io import fits
from matplotlib import pyplot as plt


volume = '../../../MIRISIM/MIRI-ETC-SIM/'

hdul = fits.open('../../../MIRISIM/MIRI-ETC-SIM/miri_etc_cube_both.fits')
#prihdu = fits.PrimaryHDU( header=hdul[0].header, data=hdul[0].data[:, 4:-5, :])
prihdu = fits.PrimaryHDU( header=hdul[0].header, data=hdul[0].data[:, 5:-4, :])
hdus_list = [prihdu]
for i in range(1,len(hdul)):
    # hdus_list.append(fits.ImageHDU(data=hdul[i].data[:, 4:-5, :], header=hdul[i].header))
    hdus_list.append(fits.ImageHDU(data=hdul[i].data[:, 5:-4, :], header=hdul[i].header))

thdulist = fits.HDUList(hdus_list)
thdulist.writeto('../../../MIRISIM/MIRI-ETC-SIM/miri_etc_cube_both_cut.fits', overwrite=True)


cube_both = readcube.CUBE(infile=volume+'miri_etc_cube_both_cut.fits',dataext=1, varext=2, dqext=3, waveext=None, wmapext=None)
#cube_both = readcube.CUBE(infile='../NIRSpec_sim/NRS00001-QG-F100LP-G140H_comb_1234_g140h-f100lp_s3d.fits',dataext=1, varext=2, dqext=3, waveext=None)
cube_psf = readcube.CUBE(infile=volume+'miri_etc_cube_quasar.fits',dataext=1, varext=2, dqext=3, waveext=None, wmapext=None)
cube_galaxy = readcube.CUBE(infile=volume+'miri_etc_cube_galaxy.fits',dataext=1, varext=2, dqext=3, waveext=None, wmapext=None)


def write_psfsubcube(file_in, file_out, datext, flux_psfsub):
    
    hdul_in = fits.open(file_in)


    prihdu = fits.PrimaryHDU( header=hdul_in[0].header, data=hdul_in[0].data)
    hdus_list = [prihdu]
    for i in range(1,len(hdul_in)):
        if i == datext:
            hdus_list.append(fits.ImageHDU(data=flux_psfsub, header=hdul_in[i].header))
        else:
            hdus_list.append(fits.ImageHDU(data=hdul_in[i].data, header=hdul_in[i].header))
    thdulist = fits.HDUList(hdus_list)
    thdulist.writeto(file_out, overwrite=True)



def scaling_factors(wavelengths):
    '''
        A function defined to calculate the scaling factors for SDI cube transformation.
        Parameters
        ----------
        In parameters:
        ----------
        wavelengths : numpy array
        Array with the wavelength of each frame.
        Returns
        -------
        Out
        -------
        numpy array
        Scaling factors.
        '''
    #0.05
    return (max(wavelengths)/ wavelengths)**1.00

def scale_cube(cube_in,shift_back = None,scaling = None):
    '''Function defined to scale the data cube according to
        diffraction limited wavelength scaling relation
        
        In:
        ----------
        cube_in: 3D numpy array or readcube class
        
        optional:
        ----------
        shift_back: True/False. If set to True does the inverse shift
        scaling: numpy array of scaling factors as a function of wavelength
        
        out:
        ------------
        3D numpy array of scaled data cube
        
        
        
        '''
    cube_out = zeros((29,29,3926)) #for ETC
    cube_out= zeros((16, 25, 907)) # for MIRI ETC cube
    cube_out= zeros((16, 16, 907)) # for MIRI ETC cube cut
    #    cube_out = zeros((37,37,3945)) #for NIRSpec sim

    #checking if the loaded cube is a numpy array or a class object.
    if str(type(cube_in))=="<class 'numpy.ndarray'>":
        Bl = scaling
    else:
        Bl = scaling_factors(cube_in.wave) #making the scaling array in case in the input is a class object
    if shift_back == True: #creating the inverse scaling for the data cube.
        Bl = 1/Bl
    
    F_spax = np.zeros(len(Bl))
    F_spaxB = np.zeros(len(Bl))

    for i in arange(0,len(Bl)): #looping through all wavelength
        
        scaling_x = Bl[i] #scaling factor, assuming square spaxels.
        scaling_y = Bl[i]
        if str(type(cube_in))=="<class 'numpy.ndarray'>":
            image_in = cube_in[:,:,i]
        else:
            image_in = cube_in.dat[0:29,:,i]
        #        image_in = cube_in.dat[0:,0:37,i]


        #scaling an individual wavelength slice
        im_scale = rescale(image_in,
                           (scaling_y, scaling_x),
                           order=3,
                           mode='reflect',
                           multichannel=False,
                           anti_aliasing=True)

        #anti_aliasing_sigma = abs(1-scaling_y)/2.

        sum_before = np.sum(image_in)
        sum_after = np.sum(im_scale)
        im_scale = im_scale * (sum_before / sum_after) #conserving flux of the scaled image.


        image_out = np.zeros(image_in.shape)
        if shift_back==True:
            npix_del = image_out.shape[-1] - im_scale.shape[-1]
            npix_delx = image_out.shape[-2] - im_scale.shape[-2]      # x direction
        if shift_back==None:
            npix_del = im_scale.shape[-1] - image_out.shape[-1]
            npix_delx = im_scale.shape[-2] - image_out.shape[-2]      # x direction

        if npix_del == 0:
            image_out = im_scale
        else:
            if npix_del % 2 == 0:
                npix_del_a = int(npix_del/2)
                npix_del_b = int(npix_del/2)
            else:
                npix_del_a = int((npix_del-1)/2)
                npix_del_b = int((npix_del+1)/2)

            ### x direction
            if npix_delx % 2 == 0:
                npix_delx_a = int(npix_delx/2)
                npix_delx_b = int(npix_delx/2)
            else:
                npix_delx_a = int((npix_delx-1)/2)
                npix_delx_b = int((npix_delx+1)/2)

            if npix_delx_b==0:      # preventing getting sth like image_out = im_scale[0:-0, 0:-0] below, i.e. an empty array
                npix_delx_b = -im_scale.shape[-2]-1
            if npix_del_b==0:
                npix_del_b = -im_scale.shape[-2]-1

            if shift_back==True:
                # cube_out[npix_del_a:-npix_del_b, npix_del_a:-npix_del_b,i] = im_scale
                cube_out[npix_delx_a:-npix_delx_b, npix_del_a:-npix_del_b,i] = im_scale
            if shift_back==None:
                #image_out = im_scale[npix_del_a:-npix_del_b, npix_del_a:-npix_del_b]
                image_out = im_scale[npix_delx_a:-npix_delx_b, npix_del_a:-npix_del_b]
                cube_out[:,:,i] = image_out

        F_spax[i] = image_out[6,9]
        F_spaxB[i] = image_out[7,8]

        if npix_del % 2 == 1:
            if shift_back == True:
                cube_out[:,:,i] = shift(cube_out[:,:,i], (0.5, 0.5), order=3, mode='constant')
            else:
                image_out = shift(image_out, (-0.5, -0.5), order=3, mode='constant')
                cube_out[:,:,i] = image_out


    plot_rescale_output = False
    if plot_rescale_output:
        plt.plot(cube_in.wave, F_spax, label='resized')
        plt.plot(cube_in.wave, cube_in.dat[6,9, :], label='input cube')
        plt.title('Spaxel [6,9] of 16x16 cube')
        plt.xlabel('Wavelength')
        plt.ylabel('Flux')
        plt.legend()
        plt.savefig('/Users/caroline/Documents/ARI-Heidelberg/Q3D/Plots_random/rescale_fct_output_6_9.png')
        plt.show()
        plt.close()

        plt.plot(cube_in.wave, F_spaxB, label='resized')
        plt.plot(cube_in.wave, cube_in.dat[7,8, :], label='input cube')
        plt.title('Spaxel [7,8] of 16x16 cube')
        plt.xlabel('Wavelength')
        plt.ylabel('Flux')
        plt.legend()
        plt.savefig('/Users/caroline/Documents/ARI-Heidelberg/Q3D/Plots_random/rescale_fct_output_7_8.png')
        plt.show()
        plt.close()

    return cube_out


#testing
if __name__ == "__main__":
    S = scaling_factors(cube_both.wave)
    cube_scaled = scale_cube(cube_both)

    psf = (median(cube_scaled[:,:,0:1954],axis=2))#+median(cube_scaled[:,:,2171:3912],axis=2))/2.
    psf = psf/psf.max()

    #psf_sub = zeros((29,29,cube_both.dat.shape[2]))
    psf_sub = zeros(cube_both.dat.shape)
    for i in arange(0,cube_both.dat.shape[2]):
        psf_sub[:,:,i] = cube_scaled[:,:,i]-psf*cube_scaled[:,:,i].max()

    psf_sub_i = scale_cube(psf_sub,shift_back = True, scaling=S)


    write_cube_cut = False
    if write_cube_cut:
        hdu=fits.PrimaryHDU(cube_scaled.T)
        hdu.writeto(volume+'cube_scaled_cut.fits',overwrite='True')

        hdu = fits.PrimaryHDU(psf_sub_i)
        hdu.writeto(volume+'cube_psf_sub_cut.fits',overwrite=True)

        hdu = fits.PrimaryHDU(psf_sub_i.T)
        hdu.writeto(volume+'cube_psf_sub_cut_T.fits',overwrite=True)

        hdu = fits.PrimaryHDU(psf.T)
        hdu.writeto(volume+'psf_cut_T.fits',overwrite=True)


    breakpoint()

    write_psfsubcube(file_in=volume+'miri_etc_cube_both.fits', file_out=volume+'miri_etc_psf_sub.fits', datext=1, flux_psfsub=psf_sub_i.T)



    WL_sub = (sum(psf_sub_i[:,:,0:1500],axis=2))#+sum(psf_sub[:,:,2171:3912],axis=2))/1.


