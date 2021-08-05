#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
; docformat = 'rst'
;
;+
;
; This function initializes the fitting parameters for PG1411+442
;
; :Categories:
;    IFSF
;
; :Returns:
;    A structure with tags specified in INITTAGS.txt.
;
; :Params:
;
; :Keywords:
;    initmaps: out, optional, type=structure
;      Parameters for map making.
;    initnad: out, optional, type=structure
;      Parameters for NaD fitting.
;
; :Author:
;    David S. N. Rupke::
;      Rhodes College
;      Department of Physics
;      2000 N. Parkway
;      Memphis, TN 38104
;      drupke@gmail.com
;
; :History:
;    ChangeHistory::
;      2015aug25, DSNR, created
;
; :Copyright:
;    Copyright (C) 2015 David S. N. Rupke
;
;    This program is free software: you can redistribute it and/or
;    modify it under the terms of the GNU General Public License as
;    published by the Free Software Foundation, either version 3 of
;    the License or any later version.
;
;    This program is distributed in the hope that it will be useful,
;    but WITHOUT ANY WARRANTY; without even the implied warranty of
;    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU
;    General Public License for more details.
;
;    You should have received a copy of the GNU General Public License
;    along with this program.  If not, see
;    http://www.gnu.org/licenses/.
;
;-
"""
import os.path
import numpy as np
from q3dfit.common import questfit_readcf

def miritest():

    # bad=1e99
    gal = 'miritest'
    outstr = ''
    ncols = 37
    nrows = 43
    # centcol = 9.002
    # centrow = 14.002
    platescale = 0.3
    fitrange = [11.53050463,13.47485667]

#   These are unique to the user
    #volume = '/Users/Endeavour/Projects/Q3D_dev/MIRI_ETC_sim/'
    volume = '../../../MIRISIM/MIRI-ETC-SIM/'
    infile = volume+'miri_etc_cube.fits'
    #infile = volume+'cube_reconstructed.fits'
    mapdir = volume+'maps/'
    outdir = volume+'outputs/'
    qsotemplate = volume+'miri_qsotemplate_B.npy'
    #stellartemplates = \
    #    '/Users/caroline/Documents/ARI-Heidelberg/Q3D/Q3DFIT/q3dfit/Test_GMOS_DATA/pg1411/'+'pg1411hosttemplate.npy'
    logfile = outdir+gal+'_fitlog.txt'
    batchfile = '../common/fitloop.pro'
    batchdir = '/Users/drupke/src/idl/batch/'
#
# Required pars
#

    if not os.path.isfile(infile): print('Data cube not found.')


    ### more MIR settings
    global_ice_model = 'ice_hc'
    global_ext_model = 'CHIAR06'
    #cffilename = '../test/test_questfit/IRAS21219m1757_dlw_qst.cf'
    cffilename = '../test/test_questfit/miritest.cf'
    config_file = questfit_readcf.readcf(cffilename)


# Lines to fit.
    lines = ['[NeII]12.81']
#    nlines = len(lines)

# Max no. of components.
    maxncomp = 1

# Initialize line ties, n_comps, z_inits, and sig_inits.
    linetie = dict()
    ncomp = dict()
    zinit_gas = dict()
    siginit_gas = dict()
    for i in lines:
        linetie[i] = '[NeII]12.81'
        ncomp[i] = np.full((ncols,nrows),maxncomp)
        zinit_gas[i] = np.full((ncols,nrows,maxncomp),0.)
        siginit_gas[i] = np.full(maxncomp, 500.) #0.1) #1000.)
        zinit_stars=np.full((ncols,nrows),0.0)

#
# Optional pars
#

# Tweaked regions are around HeII,Hb/[OIII],HeI5876/NaD,[OI],Halpha, and [SII]
# Lower and upper wavelength for re-fit
    tw_lo = [4600,5200,6300,6800,7000,7275]
    tw_hi = [4800,5500,6500,7000,7275,7375]
# Number of wavelength regions to re-fit
    tw_n = len(tw_lo)
# Fitting orders
    deford = 1
    tw_ord = np.full(tw_n,deford)
# Parameters for continuum fit
# In third dimension:
#   first element is lower wavelength limit
#   second element is upper
#   third is fit order
    tweakcntfit = np.full((ncols,nrows,3,tw_n),0)
    tweakcntfit[:,:,0,:] = tw_lo
    tweakcntfit[:,:,1,:] = tw_hi
    tweakcntfit[:,:,2,:] = tw_ord

    # Parameters for emission line plotting
    linoth = np.full((1, 1), '', dtype=object)
    linoth[0, 0] = '[[NeII]12.81]'
    argspltlin1 = {'nx': 1,
                    'ny': 1,
                    'label': ['test-MIRLINE'],
                        'wave': [128130.0],
                        'off': [[-120,90]],
                        'linoth': linoth}

    # Velocity dispersion limits and fixed values
    siglim_gas = np.ndarray(2)
    siglim_gas[:] = [5, 5000]
    # lratfix = {'[NI]5200/5198': [1.5]}

    #
    # Output structure
    #

    init = { \
            # Required pars
            'fcninitpar': 'parinit',#gmos
            'fitran': fitrange,
            'fluxunits': 1,  # erg/s/cm^2/sr
            'infile': infile,
            'label': gal,
            'lines': lines,
            'linetie': linetie,
            'maxncomp': maxncomp,
            'name': 'PG1411+442',
            'ncomp': ncomp,
            'mapdir': mapdir,
            'outdir': outdir,
            'platescale': platescale,
            'positionangle': 335,
            'minoraxispa': 75,
            'zinit_stars': zinit_stars,
            'zinit_gas': zinit_gas,
            'zsys_gas': 0.0,
            # Optional pars
#            'argscheckcomp': {'sigcut': 1,
#                              'ignore': ['[OI]6300', '[OI]6364',
#                                         '[SII]6716', '[SII]6731']},
            'argscontfit': {'qsoxdr': qsotemplate,
                            'siginit_stars': 50,
                            'uselog': 1,
                            'refit':'questfit',
                            'args_questfit': {'config_file': cffilename,
                                'global_ice_model': global_ice_model,
                                'global_ext_model': global_ext_model,
                                'models_dictionary': {},
                                'template_dictionary': {}} 
                            },
            'argscontplot': {'xstyle':'log',
                             'ystyle':'log',
                             'waveunit_in': 'Angstrom',
                             'waveunit_out': 'Angstrom',
                             'fluxunit_in':'flambda',
                             'fluxunit_out':'flambda',
                             'mode':'dark'},

            'argslinelist': {'vacuum': False},
            #'startempfile': stellartemplates,
            'argspltlin1': argspltlin1,
            # 'donad': 1,
            'decompose_qso_fit': 1,
            # 'remove_scattered': 1,
            'fcncheckcomp': 'checkcomp',
            'fcncontfit': 'fitqsohost',
            #'fcncontfit': 'ppxf',
            'maskwidths_def': 2000,
#            'tweakcntfit': tweakcntfit,
            'emlsigcut': 2,
            'logfile': logfile,
            'batchfile': batchfile,
            'batchdir': batchdir,
            'siglim_gas': siglim_gas,
            'siginit_gas': siginit_gas,
            'siginit_stars': 50,
                #            'cutrange': np.array([14133, 14743]),
            'nocvdf': 1,
            # 'cvdf_vlimits': [-3e3,3e3],
            # 'cvdf_vstep': 10d,
            # 'host': {'dat_fits': volume+'ifs/gmos/cubes/'+gal+'/'+\
            #         gal+outstr+'_host_dat_2.fits'} \
            'plotMIR': True,
            'qsoonly':1,
            'argsreadcube': {'fluxunit_in': 'Jy',
                            'waveunit_in': 'angstrom',
                            'waveunit_out': 'micron'}        
            }

    return(init)
