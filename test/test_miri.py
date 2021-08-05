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


template_exists = True
if not template_exists:
	# Make quasar template
	from q3dfit.common.makeqsotemplate import makeqsotemplate
	volume = '../../../MIRISIM/MIRI-ETC-SIM/'
	outpy = volume + 'miri_qsotemplate_B.npy'
	infits = volume + 'miri_etc_cube.fits'
	makeqsotemplate(infits, outpy, dataext=1, varext=2, dqext=3, waveext=None)



from q3dfit.common.q3df import q3df
q3df('miritest', cols=7, rows=14, quiet=False)
from q3dfit.common.q3da import q3da
q3da('miritest', cols=7, rows=14, quiet=False)




# Test creation of Gonzalez-Delgado templates
# from q3dfit.common.gdtemp import gdtemp
# gdtemp('/Users/drupke/Documents/stellar_models/gonzalezdelgado/SSPGeneva.z020',
#        '/Users/drupke/Documents/stellar_models/gonzalezdelgado/SSPGeneva_z020.npy')
