# -*- coding: utf-8 -*-
"""
Routine to read in the PHOENIX stellar models extending to 5 micron.

"""

import os
import numpy as np
import glob
from astropy.io import fits
import ppxf.ppxf_util as util

def setup_phoenix():
    '''
    Naming scheme for the templates: <grid>/<subgrid>/(n)lte<Teff><log(g)><subgrid>.<grid>-HiRes.fits
    (see Husser+2013)
    '''
    path = '/Users/weizheliu/q3dfit-dev/q3dfit/data/phoenix/R10000FITS/'  #  /phoenix.astro.physik.uni-goettingen.de/data/MedResFITS/R10000FITS_all/'
    files_all = glob.glob(path+'*')
    
    metal = [-4., -3., -2. -1.5, -1., -0.5, 0., 0.5, 1.]
    alpha = np.arange(-0.2, 1.4, step=0.2)
    for Z_i in metal:
        for a_i in alpha:
            files_i = path+'PHOENIX-ACES-AGSS-COND-2011_R10000FITS_Z%+.1f.Alpha=%+.2f.zip' % (Z_i, a_i)
            #files_i = glob.glob('lte03500-0.50{:.2f}.Alpha={:.3f}.PHOENIX-ACES-AGSS-COND-2011-HiRes.zip'.format(Z_i, a_i))
            #if Z_i==0:
            #    
            #    files_i = glob.glob('lte03500-0.50-0.0.Alpha={:.2f}.PHOENIX-ACES-AGSS-COND-2011-HiRes.zip'.format(Z_i, a_i))
    #breakpoint()
            if os.path.isfile(files_i):
                hdu = fits.open(files_i)
                flux = hdu[0].data
                hdr = hdu[0].header
                nflux = np.size(flux)
                
                wave = (np.arange(nflux)+1-hdr['CRPIX1']) * hdr['CDELT1'] + hdr['CRVAL1']
                if hdr['CTYPE1'] == 'WAVE-LOG':
                    wave = 10**wave
                #ages_str = data.meta['comments'][43].split()
                #ages_strnp = np.array(ages_str[2:])
                #ages = ages_strnp.astype(float)
                #flux = np.zeros((len(data), ages.size), dtype=np.float)
                #cols = data.colnames[1::3]
                #for i in range(len(ages)):
                #    flux[:, i] = np.array(data[cols[i]])
                #    flux[:, i] /= flux[:, i].max()
                
                template = {'lambda': wave, 'flux': flux}
                np.save(outfile, template)
    
def setup_XSL():
    path = '/Users/weizheliu/q3dfit-dev/q3dfit/data/XSL/Kroupa_SSP_PC/'  #  /phoenix.astro.physik.uni-goettingen.de/data/MedResFITS/R10000FITS_all/'
    files_all = glob.glob(path+'*')
    files_i = path+'XSL_SSP_logT10.2_MH-1.8_Kroupa_PC.fits'
    #PHOENIX-ACES-AGSS-COND-2011_R10000FITS_Z%+.1f.Alpha=%+.2f.zip' % (Z_i, a_i)

    logT = np.arange(7.7,10.2,step=0.1)
    MH = np.arange(-2.2, -0.2, step=0.2)
    MH = np.append(MH,np.array([-0.1,-0.0,0.1,0.2]))
    for T_i in logT:
        for M_i in MH:
            if M_i < 1e-4:
                files_i = path+'XSL_SSP_logT%3.1f_MH%+2.1f_Kroupa_PC.fits' % (T_i,M_i)
            else:
                files_i = path+'XSL_SSP_logT%3.1f_MH%2.1f_Kroupa_PC.fits' % (T_i,M_i)
            hdu = fits.open(files_i)
            flux = hdu[0].data
            hdr = hdu[0].header
            nflux = np.size(flux)
            wave = (np.arange(nflux)+1-hdr['CRPIX1'])* hdr['CDELT1'] + hdr['CRVAL1']
            outfile = path+'XSL_SSP_logT%3.1f_MH%+2.1f_Kroupa_PC.npy' % (T_i,M_i)
            #if hdr['CTYPE1'] == 'WAVE-LOG':
            wave = 10**wave/1e3 # in um, linear
            template = {'lambda': wave, 'flux': flux, 'ages': T_i}
            np.save(outfile, template)
            #print(wave)
            #breakpoint()
setup_XSL()

#setup_phoenix()

