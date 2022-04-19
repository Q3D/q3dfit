# -*- coding: utf-8 -*-
"""
Created on March 30th 2021 by Caroline Bertemes.

This script saves the output of questfit.py, namely the best-fit parameters and
spectra of each component.

"""

import numpy as np
from astropy.io import fits
import sys


def write_dict_fits(filename, x, xname, dict_y, xfmt='20A', comm = ''):
	'''
	Create a new table fits from a "reference array" (e.g. wavelengths) and a dictionary 
	containing arrays of length equal to the reference array (e.g. fluxes of different components). 
	
	:Params: 
	  filename: in, type=str
		Name of the output FITS file
	  x: in, type = array
	    reference array (e.g. wavelength array)
	  xname: in, type=1D array
	    name of the reference array x
	  dict_y: in, type=dict
	    dict containing arrays (e.g. flux arrays of different components) of the same length as x. 
	    By default, these arrays are assumed to contain floats. To change the format for a given array 
	    named "key" in the dict, include another entry named "key_FMT" which specifies its format in dict_y.
	  xfmt: in, optional, type=str
	    format of the reference array x. Default is string.
	  comm: in, optional, type=str
	    A string with any additional information/instructions to be stored in the header.

	'''
	# --- Create table
	col1 = fits.Column(name=xname, format=xfmt, array=x)
	cols_list = [ col1 ]
	for key_i in dict_y.keys():
		if 'FMT' not in key_i:
			fmt_i = 'F' 
			if key_i+'_FMT' in dict_y.keys():	fmt_i = dict_y[key_i+'_FMT']
			col_i = fits.Column(name=key_i, format=fmt_i, array=dict_y[key_i])
			cols_list.append(col_i)
	cols = fits.ColDefs(cols_list)
	tbhdu = fits.BinTableHDU.from_columns(cols)
	tbhdu.header
	# --- Create primary HDU with header
	prihdr = fits.Header()
	prihdr['COMMENT'] = comm
	prihdu = fits.PrimaryHDU(header=prihdr)
	# --- Combine both in new .fits file
	thdulist = fits.HDUList([prihdu, tbhdu])
	thdulist.writeto(filename, overwrite=True)


def save_spectral_comp(wave, flux, best_fit, comp_best_fit, config_filename):
	'''
	Write out a FITS file containing the observed wavelengths and fluxes, the best-fit model, and 
	the latter's different components. Names of the components are taken from the config file.
	
	:Params: 
	  wave: in, type=1D array
	    wavelength array
	  flux: in, type=1D array
	    observed flux array
	  best_fit: in, type=1D array
	    best-fit modelled flux array
	  comp_best_fit: in, type=ordered dict
	    ordered dict containing each best-fit component unattenuated, followed by its extinction and absorption
	  config_filename: in, type=str
	    Name of the input config file. This is used to name the output file.
	'''
	dict_comp_out = {'FLUX': flux, 'best_fit': best_fit}
	for i in np.arange(0,len(comp_best_fit.keys()),3):
		dict_comp_out[list(comp_best_fit.keys())[i]] = comp_best_fit[list(comp_best_fit.keys())[i]]*comp_best_fit[list(comp_best_fit.keys())[i+1]]*comp_best_fit[list(comp_best_fit.keys())[i+2]]
	filename_comp = config_filename.replace('.cf', '_comp.fits')
	write_dict_fits(filename_comp, wave, 'WAVE', dict_comp_out, xfmt='F')



def save_params(result, config_filename):
	"""
	Write out a txt file containing a table with the best-fit parameters retrieved by lmfit.

	:Params:
	  result: in, type = lmfit model.fit() object
	    model.fit() object obtained from running lmfit. Contains the best-fit parameters result.params
	  config_filename: in, type=str
	    Name of the input config file. This is used to name the output file.
	"""
	original_stdout = sys.stdout 
	filename_par = config_filename.replace('.cf', '_par.txt')
	with open(filename_par, 'w') as f:
		sys.stdout = f # -- Re-direct standard output to file
		result.params.pretty_print()
		sys.stdout = original_stdout # -- Reset standard output




