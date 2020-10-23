from q3dfit.common import q3df,makeqsotemplate

volume = '/Volumes/fingolfin/ifs/gmos/cubes/pg1411/'
outpy = volume + 'pg1411qsotemplate.npy'
infits = volume + 'pg1411rb3.fits'

makeqsotemplate.makeqsotemplate(infits,outpy,dataext=None,dqext=None,waveext=None)
q3df.q3df('pg1411',cols=[14],rows=[11])
