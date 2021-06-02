# Make quasar template
# from q3dfit.common.makeqsotemplate import makeqsotemplate
# volume = '/Users/drupke/Box Sync/q3d/pg1411/'
# outpy = volume + 'pg1411qsotemplate.npy'
# infits = volume + 'pg1411rb1.fits'
# makeqsotemplate(infits, outpy, dataext=None, dqext=None, waveext=None)

from q3dfit.common.q3df import q3df
q3df('pg1411', cols=14, rows=11, quiet=False)

from q3dfit.common.q3da import q3da
q3da('pg1411', cols=14, rows=11, quiet=False)

# Test creation of Gonzalez-Delgado templates
# from q3dfit.common.gdtemp import gdtemp
# gdtemp('/Users/drupke/Documents/stellar_models/gonzalezdelgado/SSPGeneva.z020',
#        '/Users/drupke/Documents/stellar_models/gonzalezdelgado/SSPGeneva_z020.npy')
