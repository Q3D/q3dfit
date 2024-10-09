# -*- coding: utf-8 -*-
"""
Routine to read in the PHOENIX stellar models extending to 5 micron.

"""

import numpy as np
import glob
from astropy.io import fits
import ppxf.ppxf_util as util

def setup_spectral_library():
	'''
	Naming scheme for the templates: <grid>/<subgrid>/(n)lte<Teff><log(g)><subgrid>.<grid>-HiRes.fits
	(see Husser+2013)
	'''
	path = '/Users/caroline/Documents/ARI-Heidelberg/Q3D/StellLib/PHOENIX/phoenix.astro.physik.uni-goettingen.de/data/MedResFITS/R10000FITS_all/'
	files_all = glob.glob(path+'*')

	metal = [-4., -3., -2. -1.5, -1., -0.5, 0., 0.5, 1.]
	alpha = np.arange(-0.2, 1.4, step=0.2)

	for Z_i in metal:
		for a_i in alpha:
			files_i = glob.glob('lte03500-0.50{:.2f}.Alpha={:.2f}.PHOENIX-ACES-AGSS-COND-2011-HiRes.fits'.format(Z_i, a_i))
			if Z_i==0.:
				files_i = glob.glob('lte03500-0.50-0.0.Alpha={:.2f}.PHOENIX-ACES-AGSS-COND-2011-HiRes.fits'.format(Z_i, a_i))
