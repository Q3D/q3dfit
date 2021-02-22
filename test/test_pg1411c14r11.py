from q3dfit.common.q3df import q3df
from q3dfit.common.q3da import q3da
# from q3dfit.common.makeqsotemplate import makeqsotemplate

volume = '/Volumes/fingolfin/ifs/gmos/cubes/pg1411/'
outpy = volume + 'pg1411qsotemplate.npy'
infits = volume + 'pg1411rb1.fits'

# makeqsotemplate(infits, outpy, dataext=None, dqext=None, waveext=None)
q3df('pg1411', ncores=10)
q3da('pg1411')

# Test creation of Gonzalez-Delgado templates
# from q3dfit.common.gdtemp import gdtemp
# gdtemp('/Users/drupke/Documents/stellar_models/gonzalezdelgado/SSPGeneva.z020',
#        '/Users/drupke/Documents/stellar_models/gonzalezdelgado/SSPGeneva_z020.npy')
