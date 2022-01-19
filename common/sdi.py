from numpy import *
from q3dfit.common.makeqsotemplate import makeqsotemplate
import q3dfit.common.readcube as readcube
from matplotlib.pyplot import *
from scipy.ndimage import gaussian_filter
from skimage.transform import rescale
from scipy.ndimage import fourier_shift, shift, rotate
from astropy.io import fits

cube_both = readcube.CUBE(infile='miri_etc_cube_both.fits',dataext=1, varext=2, dqext=3, waveext=None)
#cube_both = readcube.CUBE(infile='../NIRSpec_sim/NRS00001-QG-F100LP-G140H_comb_1234_g140h-f100lp_s3d.fits',dataext=1, varext=2, dqext=3, waveext=None)
cube_psf = readcube.CUBE(infile='miri_etc_cube_quasar.fits',dataext=1, varext=2, dqext=3, waveext=None)
cube_galaxy = readcube.CUBE(infile='miri_etc_cube_galaxy.fits',dataext=1, varext=2, dqext=3, waveext=None)



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
    #    cube_out = zeros((37,37,3945)) #for NIRSpec sim

    #checking if the loaded cube is a numpy array or a class object.
    if str(type(cube_in))=="<class 'numpy.ndarray'>":
        Bl = scaling
    else:
        Bl = scaling_factors(cube_in.wave) #making the scaling array in case in the input is a class object
    if shift_back == True: #creating the inverse scaling for the data cube.
        Bl = 1/Bl
    
    
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
        if shift_back==None:
            npix_del = im_scale.shape[-1] - image_out.shape[-1]

        if npix_del == 0:
            image_out = im_scale
        else:
            if npix_del % 2 == 0:
                npix_del_a = int(npix_del/2)
                npix_del_b = int(npix_del/2)
            
            else:
                npix_del_a = int((npix_del-1)/2)
                npix_del_b = int((npix_del+1)/2)

            if shift_back==True:
                cube_out[npix_del_a:-npix_del_b, npix_del_a:-npix_del_b,i] = im_scale
            if shift_back==None:
                image_out = im_scale[npix_del_a:-npix_del_b, npix_del_a:-npix_del_b]
                cube_out[:,:,i] = image_out
        
        if npix_del % 2 == 1:
            if shift_back == True:
                cube_out[:,:,i] = shift(cube_out[:,:,i], (0.5, 0.5), order=3, mode='constant')
            else:
                image_out = shift(image_out, (-0.5, -0.5), order=3, mode='constant')
                cube_out[:,:,i] = image_out




    return cube_out

#testing
if __name__ == "__main__":
    S = scaling_factors(cube_both.wave)
    cube_scaled = scale_cube(cube_both)
    hdu=fits.PrimaryHDU(cube_scaled)
    hdu.writeto('cube_scaled.fits',overwrite='True')


    psf = (median(cube_scaled[:,:,0:1954],axis=2))#+median(cube_scaled[:,:,2171:3912],axis=2))/2.
    psf = psf/psf.max()

    psf_sub = zeros((29,29,cube_both.dat.shape[2]))
    for i in arange(0,cube_both.dat.shape[2]):
        psf_sub[:,:,i] = cube_scaled[:,:,i]#-psf#*cube_scaled[:,:,i].max()

    psf_sub_i = scale_cube(psf_sub,shift_back = True, scaling=S)

    hdu = fits.PrimaryHDU(psf_sub_i)
    hdu.writeto('cube_psf_sub.fits',overwrite=True)

    WL_sub = (sum(psf_sub_i[:,:,0:1500],axis=2))#+sum(psf_sub[:,:,2171:3912],axis=2))/1.


