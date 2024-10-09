import os
from astropy.io import fits
import numpy as np
import shutil
import copy


def datacube_convert(infile,CLOBBER=True):
    # read in the datacube and check the extensions
    fdIN = fits.open(infile)
    enames = [dd.name.upper() for dd in fdIN]
    extens = {ext:(ext in enames) for ext in {'VAR':None,'DQ':None}}
    sciEXT = enames.index("SCI")
    if all(flag == True for fi,flag in extens.items()) == True:
        return infile
    else:
        # create new name
        dfsplit = infile.split('/')
        newfile = '/'.join(dfsplit[:-1])+'/'+dfsplit[-1].split('.fits')[0]+'_vardq.fits'
        if not os.path.isfile(newfile):
            shutil.copy(infile,newfile)
            print('create',newfile)
        hdul = fits.open(newfile,mode='update')
        datIN = hdul[sciEXT].data
        for ext in extens:
            if ext == 'VAR' and extens[ext] == False:
                var = copy.deepcopy(datIN)
                hdu_a = fits.ImageHDU(var,name='var')
                hdul.append(hdu_a)
            elif ext == 'DQ' and extens[ext] == False:
                dq  = np.zeros(datIN.shape)
                hdu_b = fits.ImageHDU(dq,name='dq')
                hdul.append(hdu_b)
        hdul.writeto(newfile,overwrite=CLOBBER,output_verify='ignore')
        hdul.close()
        return newfile
