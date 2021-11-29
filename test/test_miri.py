import sys
#sys.path.append('/jwst0nb/lwz/')
import numpy as np
from matplotlib import pyplot as plt
from q3dfit.common import readcube

# Make quasar template
# from q3dfit.common.makeqsotemplate import makeqsotemplate
# volume = '/Volumes/fingolfin/ifs/gmos/cubes/pg1411/'
# outpy = volume + 'pg1411qsotemplate.npy'
# infits = volume + 'pg1411rb1.fits'
# makeqsotemplate(infits, outpy, dataext=None, dqext=None, waveext=None)


#cube = readcube.CUBE(infile='/jwst0nb/lwz/jwst_q3d_data/NRS00001-QG-F100LP-G140H_comb_1234_g140h-f100lp_s3d.fits')
#cube = readcube.CUBE(infile='/jwst0nb/lwz/jwst_q3d_data/det_image_seq1_MIRIFUSHORT_12SHORTexp1_s3d.fits')
#plt.plot(cube.wave,cube.dat[17,21,:])
#plt.show()
#plt.imshow(np.log10(cube.dat[:,:,2871]))
#plt.imshow(np.log10(cube.wmap[:,:,2871]))

def Get_flex_template(nrow, ncol):
	volume = '../../../MIRISIM/MIRI-ETC-SIM/'
	infits = volume + 'miri_etc_cube_quasar.fits'
	argsreadcube = {'fluxunit_in': 'Jy', 'waveunit_in': 'angstrom', 'waveunit_out': 'micron'} 
	cube_templ = readcube.CUBE(infile=infits, dataext=1, varext=2, dqext=3, waveext=None, **argsreadcube)
	#breakpoint()
	outpy = '../data/questfit_templates/' + 'miri_qsotemplate_flex.npy'
	qsotemplate = {'wave':cube_templ.wave,'flux':cube_templ.dat[ncol-1, nrow-1, :],'dq':cube_templ.dq[ncol-1, nrow-1, :]}
	np.save(outpy,qsotemplate)


template_exists = True
if not template_exists:
	# Make quasar template
	from q3dfit.common.makeqsotemplate import makeqsotemplate
	volume = '../../../MIRISIM/MIRI-ETC-SIM/'
	#outpy = volume + 'miri_qsotemplate_B.npy'
	outpy = '../data/questfit_templates/' + 'miri_qsotemplate.npy'
	infits = volume + 'miri_etc_cube.fits'
	makeqsotemplate(infits, outpy, dataext=1, varext=2, dqext=3, waveext=None)


flextempl = True


run_all_spaxels = False
if run_all_spaxels:
	ncol_FitFailed = np.array([])
	nrow_FitFailed = np.array([])
	for ncol_i in range(16):		# There are 16 cols, 25 rows in total
		for nrow_j in range(25):
			try:
				if flextempl:
					Get_flex_template(nrow_j, ncol_i)
				from q3dfit.common.q3df import q3df
				q3df('miritest', cols=ncol_i, rows=nrow_j, quiet=False)
				from q3dfit.common.q3da import q3da
				q3da('miritest', cols=ncol_i, rows=nrow_j, quiet=False)
			except:
				ncol_FitFailed = np.append(ncol_FitFailed, ncol_i)
				nrow_FitFailed = np.append(nrow_FitFailed, nrow_j)
	print('\nFit failed for the following:')
	print('ncol_FitFailed: ', ncol_FitFailed)
	print('nrow_FitFailed: ', nrow_FitFailed)
else:
	ncol = 7
	nrow = 13
	#ncol = 6
	#nrow = 11
	if flextempl:
		Get_flex_template(nrow, ncol)
	from q3dfit.common.q3df import q3df
	q3df('miritest', cols=ncol, rows=nrow, quiet=False)		# There are 16 cols, 25 rows in total
	from q3dfit.common.q3da import q3da
	q3da('miritest', cols=ncol, rows=nrow, quiet=False)




# Test creation of Gonzalez-Delgado templates
# from q3dfit.common.gdtemp import gdtemp
# gdtemp('/Users/drupke/Documents/stellar_models/gonzalezdelgado/SSPGeneva.z020',
#        '/Users/drupke/Documents/stellar_models/gonzalezdelgado/SSPGeneva_z020.npy')
