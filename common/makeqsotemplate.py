import numpy as np
# import pdb
from matplotlib import pyplot as plt
from q3dfit.common import readcube

'''Function defined to extract the quasar spectrum


        Parameters
        ----------
        infits: string
                Name of the fits file to load in.

        outpy
              : string
                Name of the numpy save file for the resulting qso spectrum


        Returns
        -------
        dictionary
        {wave,flux,dq}

'''


def makeqsotemplate(infits, outpy, datext=1, dqext=2, varext=3, wavext=None,
                    wmapext=4, plot=True, waveunit_in='micron'):

    cube = readcube.CUBE(infile=infits, datext=datext, dqext=dqext,
                         varext=varext, wavext=wavext, wmapext=wmapext,
                         waveunit_in=waveunit_in)

    white_light_image = np.median(cube.dat, axis=2)

    loc_max = np.where(white_light_image == white_light_image.max())

    qsotemplate = {'wave': cube.wave}
    if cube.dat is not None:
        norm = np.median(cube.dat[loc_max[0][0], loc_max[1][0]])
        qsotemplate['flux'] = cube.dat[loc_max[0][0], loc_max[1][0]] / norm
        if plot:
            plt.plot(cube.wave, qsotemplate['flux'])
            plt.show()
    if cube.var is not None:
        qsotemplate['var'] = cube.var[loc_max[0][0], loc_max[1][0]] / norm
    if cube.dq is not None:
        qsotemplate['dq'] = cube.dq[loc_max[0][0], loc_max[1][0]]

    np.save(outpy, qsotemplate)
