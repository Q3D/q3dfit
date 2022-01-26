import sys
#sys.path.append('/jwst0nb/lwz/')
import numpy as np
from matplotlib import pyplot as plt
from q3dfit.common.readcube import CUBE
from q3dfit.common import readcube
from q3dfit.common.q3df import q3df
from q3dfit.common.q3da import q3da
from scipy import constants
from astropy import units as u
import os


# Make quasar template
# from q3dfit.common.makeqsotemplate import makeqsotemplate
# volume = '/Volumes/fingolfin/ifs/gmos/cubes/pg1411/'
# outpy = volume + 'pg1411qsotemplate.npy'
# infits = volume + 'pg1411rb1.fits'
# makeqsotemplate(infits, outpy, datext=None, dqext=None, wavext=None)


#cube = readcube.CUBE(infile='/jwst0nb/lwz/jwst_q3d_data/NRS00001-QG-F100LP-G140H_comb_1234_g140h-f100lp_s3d.fits')
#cube = readcube.CUBE(infile='/jwst0nb/lwz/jwst_q3d_data/det_image_seq1_MIRIFUSHORT_12SHORTexp1_s3d.fits')
#plt.plot(cube.wave,cube.dat[17,21,:])
#plt.show()
#plt.imshow(np.log10(cube.dat[:,:,2871]))
#plt.imshow(np.log10(cube.wmap[:,:,2871]))


def Get_flex_template(nrow, ncol, name_out='miri_qsotemplate_flex.npy'):
    volume = '../../../MIRISIM/MIRI-ETC-SIM/'
    infits = volume + 'miri_etc_cube_quasar.fits'
    argsreadcube = {'fluxunit_in': 'Jy', 'waveunit_in': 'angstrom', 'waveunit_out': 'micron', 'wmapext': None} 
    cube_templ = readcube.CUBE(infile=infits, datext=1, varext=2, dqext=3, wavext=None,   **argsreadcube)
    #breakpoint()
    outpy = '../data/questfit_templates/' + name_out
    qsotemplate = {'wave':cube_templ.wave,'flux':cube_templ.dat[ncol-1, nrow-1, :],'dq':cube_templ.dq[ncol-1, nrow-1, :]}
    np.save(outpy,qsotemplate)

    infits2 = volume + 'miri_etc_cube_galaxy.fits'
    argsreadcube = {'fluxunit_in': 'Jy', 'waveunit_in': 'angstrom', 'waveunit_out': 'micron', 'wmapext': None} 
    cube_templ = readcube.CUBE(infile=infits2, datext=1, varext=2, dqext=3, wavext=None,   **argsreadcube)
    outpy = '../data/questfit_templates/' + 'miri_gal_spec.npy'
    galtemplate = {'wave':cube_templ.wave,'flux':cube_templ.dat[ncol-1, nrow-1, :],'dq':cube_templ.dq[ncol-1, nrow-1, :]}
    np.save(outpy,galtemplate)


def Get_central_template():
    # Make quasar template
    from q3dfit.common.makeqsotemplate import makeqsotemplate
    volume = '../../../MIRISIM/MIRI-ETC-SIM/'
    #outpy = volume + 'miri_qsotemplate_B.npy'
    outpy = '../data/questfit_templates/' + 'miri_qsotemplate_flex.npy'
    infits = volume + 'miri_etc_cube.fits'
    makeqsotemplate(infits, outpy, datext=1, varext=2, dqext=3, wmapext=None, wavext=None)


def init_guess_ampl(nrow, ncol, templ_name):
    header = b'\x00'
    datext = 1
    varext = 2
    dqext = 3
    quiet = True
    iuse = ncol-1
    juse = nrow-1
    from q3dfit.common import readcube
    argsreadcube_dict = {'fluxunit_in': 'Jy', 'waveunit_in': 'angstrom', 'waveunit_out': 'micron', 'wmapext': None} 
    cube2 = CUBE(infile='../../../MIRISIM/MIRI-ETC-SIM/miri_etc_cube_galaxy.fits', quiet=quiet,
        header=header, datext=datext, varext=varext, dqext=dqext,  **argsreadcube_dict)
    cube3 = CUBE(infile='../../../MIRISIM/MIRI-ETC-SIM/miri_etc_cube_quasar.fits', quiet=quiet,
        header=header, datext=datext, varext=varext, dqext=dqext, **argsreadcube_dict)
    cube4 = CUBE(infile='../../../MIRISIM/MIRI-ETC-SIM/miri_etc_cube.fits',   quiet=quiet, header=header, \
        datext=datext, varext=varext, dqext=dqext, **argsreadcube_dict)

    # cube4.dat[iuse, juse, :].flatten() / (cube2.dat[iuse, juse, :].flatten()+cube3.dat[iuse, juse, :].flatten())
    # f_ini_QSO = np.median(cube3.dat[iuse, juse, :].flatten()/cube4.dat[iuse, juse, :].flatten())
    # f_ini_gal = np.median(cube2.dat[iuse, juse, :].flatten()/cube4.dat[iuse, juse, :].flatten())
    c_scale =  constants.c * u.Unit('m').to('micron') /(cube3.wave)**2 *1e-23  *1e10

    templ = np.load('../data/questfit_templates/' + templ_name, allow_pickle='TRUE').item()
    y1 = templ['flux']*c_scale / max(templ['flux']*c_scale)
    y2 = cube3.dat[iuse, juse, :].flatten()*c_scale

    from scipy.optimize import curve_fit
    def func(y1, a):
        return a * y1
    popt, pcov = curve_fit(func, y1, y2)
    #breakpoint()

    # ampl_QSO = max(cube3.dat[iuse, juse, :].flatten()*c_scale)
    ampl_gal = max(cube2.dat[iuse, juse, :].flatten()*c_scale)
    ampl_QSO = popt[0]
    # ampl_gal = np.median(cube2.dat[iuse, juse, :].flatten()*c_scale)

    cffilename = '../test/test_questfit/miritest.cf'
    #from q3dfit.common import questfit_readcf
    #config_file = questfit_readcf.readcf(cffilename)
    cf = np.loadtxt(cffilename, dtype = 'str')
    count_gal_comp = 0
    for i in range(cf.shape[0]):
        if ('template' in cf[i]) and (not 'qsotempl' in cf[i]):
            count_gal_comp += 1

    s_out = ''
    with open(cffilename) as myfile:
        for n, row in enumerate(myfile):
            if ('template' in row) and (not 'qsotempl' in row):
                # count_gal_comp += 1
                ampl_now = cf[n][2]
                ampl_new = ampl_gal/count_gal_comp
                row_new = row.split(ampl_now)[0] + '{:.3f}'.format(ampl_new) + row.replace( row.split(ampl_now)[0]+ampl_now, '' )
            elif 'qsotempl' in row:
                ampl_now = cf[n][2]
                row_new = row.split(ampl_now)[0] + '{:.3f}'.format(ampl_QSO) + row.replace( row.split(ampl_now)[0]+ampl_now, '' )
            else:
                row_new = row
            s_out = s_out + row_new
    with open(cffilename, 'w') as new_file:
        new_file.write(s_out)
    


if __name__ == '__main__':
    template_exists = True
    if not template_exists:
        Get_central_template()

    flextempl = False
    give_init_guess_ampl = True


    run_all_spaxels = False
    if run_all_spaxels:
        ncol_FitFailed = np.array([])
        nrow_FitFailed = np.array([])
        for ncol_i in range(16):        # There are 16 cols, 25 rows in total
            for nrow_j in range(25):
                try:
                    if flextempl:
                        Get_flex_template(nrow_j, ncol_i)
                        Get_flex_template(nrow, ncol, name_out='miri_qsotemplate_flexB.npy')
                    elif ncol_i==0 and nrow_j==0:
                        Get_central_template()
                        Get_flex_template(nrow, ncol, name_out='miri_qsotemplate_flexB.npy')
                    if give_init_guess_ampl:
                        init_guess_ampl(nrow_j, ncol_i, 'miri_qsotemplate_flex.npy')
                    q3df('miritest', cols=ncol_i, rows=nrow_j, quiet=False)
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
        ncol = 6
        nrow = 13
        if flextempl:
            Get_flex_template(nrow, ncol)
            Get_flex_template(nrow, ncol, name_out='miri_qsotemplate_flexB.npy')
        else:
            Get_central_template()
            Get_flex_template(nrow, ncol, name_out='miri_qsotemplate_flexB.npy')
        if give_init_guess_ampl:
            init_guess_ampl(nrow, ncol, 'miri_qsotemplate_flex.npy')
        q3df('miritest', cols=ncol, rows=nrow, quiet=False)     # There are 16 cols, 25 rows in total
        q3da('miritest', cols=ncol, rows=nrow, quiet=False)




# Test creation of Gonzalez-Delgado templates
# from q3dfit.common.gdtemp import gdtemp
# gdtemp('/Users/drupke/Documents/stellar_models/gonzalezdelgado/SSPGeneva.z020',
#        '/Users/drupke/Documents/stellar_models/gonzalezdelgado/SSPGeneva_z020.npy')
