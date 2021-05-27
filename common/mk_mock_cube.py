# -*- coding: utf-8 -*-
"""
This script can be used to transform a 1D spectrum into a mock cube (with 2 "empty" dimensions).
The resulting mock cube can then be fed to readcube.py .
"""


import numpy as np
from astropy.io import fits
import pdb


path_in = '../test/test_questfit/'
file_in = path_in + '4978688_0.ideos.npy'
data = np.load(file_in, allow_pickle=True)

file_in = path_in + 'IRAS21219m1757_dlw_qst.npy'
data = np.load(file_in, allow_pickle=True)[0]


def From_1D_to_mock3D(data, savename=''):
	'''
	This function takes a 1D spectrum and stores it under the form of a mock cube (with 2 "empty" dimensions).
	:Params: 
		data: in, type = recarray with columns named 'WAVE', 'FLUX' and 'EFLUX'
		   Input 1D wavelengths, fluxes and errors
		savename: in, type=str
		   Name of the output file containing the mock cube
	'''
	# -- Create primary HDU with header
	prihdr = fits.Header()
	prihdu = fits.PrimaryHDU(header=prihdr)

	# -- Create extension 1 - flux
	val_1D = data['FLUX'].astype(float)
	len_1D = len(val_1D)
	data1 = np.array([np.array([ val_1D ])]).reshape( len_1D, 1, 1 )
	len_str = str(data1.shape[0])
	hdu1 = fits.ImageHDU(data1, name='sci')
	hdu1.header

	# -- Create extension 2 - error spectrum
	try:
		val_1D = data['EFLUX'].astype(float)
	except:
		val_1D = data['STDEV'].astype(float)
	len_1D = len(val_1D)
	data2 = np.array([np.array([ val_1D**2 ])]).reshape( len_1D, 1, 1 )   # Note: error is translated into variance
	len_str = str(data2.shape[0])
	hdu2 = fits.ImageHDU(data2, name='var')
	hdu2.header

	# -- Create extension 3 - data quality flags being 0 if all is fine, 1 if there is an issue
	try:
		val_1D = data['FLAG'].astype(float)
	except: 
		val_1D = np.zeros(data1.shape)
	len_1D = len(val_1D)
	data3 = np.array([np.array([ val_1D ])]).reshape( len_1D, 1, 1 )
	len_str = str(data3.shape[0])
	hdu3 = fits.ImageHDU(data3, name='dq')
	hdu3.header

	# -- Create extension 4 - wavelengths
	val_1D = data['WAVE'].astype(float) * 1e4
	len_1D = len(val_1D)
	data4 = val_1D # np.array([np.array([ val_1D ])]).reshape( len_1D, 1, 1 )
	len_str = str(data4.shape[0])
	hdu4 = fits.ImageHDU(data4, name='wave')
	hdu4.header

	# -- Combine 
	thdulist = fits.HDUList([prihdu, hdu1, hdu2, hdu3, hdu4])
	thdulist.writeto(savename, overwrite=True)


From_1D_to_mock3D(data, savename=file_in.replace('.ideos','').split('.npy')[0]+'_mock_cube.fits')

