import numpy as np
import pdb
from matplotlib import pyplot as plt
from .readcube import Cube

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
                    wmapext=4, plot=True, waveunit_in='micron',
                    waveunit_out='micron', radius=0.):

    cube = Cube(infits, datext=datext, dqext=dqext,
                varext=varext, wavext=wavext, wmapext=wmapext,
                waveunit_in=waveunit_in, waveunit_out=waveunit_out)

    white_light_image = np.median(cube.dat, axis=2)
    white_light_image[np.where( np.isnan(white_light_image))] = 0

    loc_max = np.where(white_light_image == white_light_image.max())

    map_x = np.tile(np.indices((cube.ncols,1))[0],(1,cube.nrows))
    map_y = np.tile(np.indices((cube.nrows,1))[0].T[0],(cube.ncols,1))
    map_r = np.sqrt((map_x - loc_max[0][0])**2 + (map_y - loc_max[1][0])**2)
    iap = np.where(map_r <= radius)

    qsotemplate = {'wave': cube.wave}
    if cube.dat is not None:
        norm = 1#np.median(cube.dat[loc_max[0][0], loc_max[1][0]])
        qsotemplate['flux'] = cube.dat[iap[0][:], iap[1][:], :].sum(0) / norm
        if plot:
            plt.plot(cube.wave, qsotemplate['flux'])
            plt.show()
    if cube.var is not None:
        qsotemplate['var'] = cube.var[iap[0][:], iap[1][:], :].sum(0) / norm
    if cube.dq is not None:
        qsotemplate['dq'] = cube.dq[iap[0][:], iap[1][:], :].sum(0)

    np.save(outpy, qsotemplate)
