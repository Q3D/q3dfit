import numpy
from q3dfit.common import readcube

def makeqsotemplate(infits,outxdr,dataext=None,dqext=None,waveext=None):
    '''Function defined to extract the quasar spectrum
        
        
        Parameters
        ----------
        infits: string
                Name of the fits file to load in.
        
        outxdr
              : string
                Name of the directory where to save the qso spectrum
                
                
        Returns
        -------
        dictionary
        {wave,flux,dq}
        
        
        
        
        
        
        '''
    

    if not dataext:
        dataext = 1.

    if dataext == -1:
        dataext = 0

    if not dqext:
        dqext = 3

    if not waveext:
        waveext = 0


    cube = readcube.CUBE(infile=infits,dataext=dataext,dqext=dqext,waveext=waveext)

    white_light_image = numpy.median(cube.dat,axis=2)

    loc_max = numpy.where(white_light_image == white_light_image.max())


    qsotemplate = {'wave':cube.wave,'flux':cube.dat[loc_max[0][0],loc_max[1][0]],'dq':cube.dq[loc_max[0][0],loc_max[1][0]]}



    numpy.save(outxdr+'nucleartemplate.npy',qsotemplate)

    return qsotemplate
