from q3dfit.common import q3df,q3da,makeqsotemplate

volume = '/Volumes/fingolfin/ifs/gmos/cubes/pg1411/'
outpy = volume + 'pg1411qsotemplate.npy'
infits = volume + 'pg1411rb1.fits'

makeqsotemplate.makeqsotemplate(infits,outpy,dataext=None,dqext=None,waveext=None)
q3df.q3df('pg1411',cols=[14],rows=[11])
q3da.q3da('pg1411',cols=[14],rows=[11])

# Test creation of Gonzalez-Delgado templates
#from q3dfit.common.gdtemp import gdtemp
#gdtemp('/Users/drupke/Documents/stellar_models/gonzalezdelgado/SSPGeneva.z020',\
#       '/Users/drupke/Documents/stellar_models/gonzalezdelgado/SSPGeneva_z020.npy')