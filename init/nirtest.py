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


def nirtest():

    # bad=1e99
    gal = 'nirtest'
    outstr = ''
    ncols = 37
    nrows = 43
    # centcol = 9.002
    # centrow = 14.002
    platescale = 0.1
    fitrange = [9800,18000]

#   These are unique to the user
    volume = '/jwst0nb/lwz/jwst_q3d_data/'
    infile = volume+'NRS00001-QG-F100LP-G140H_comb_1234_g140h-f100lp_s3d.fits'
    mapdir = volume+'maps/'
    outdir = volume+'outputs/'
    qsotemplate = '/jwst0nb/lwz/jwst_q3d_data/nirtest_qsotemplate.npy'
    stellartemplates = \
        volume+'pg1411/pg1411hosttemplate.npy'
    logfile = outdir+gal+'_fitlog.txt'
    batchfile = '/jwst0nb/lwz/q3dfit/common/fitloop.pro'
    batchdir = '/Users/drupke/src/idl/batch/'
#
# Required pars
#

    if not os.path.isfile(infile): print('Data cube not found.')

# Lines to fit.
    lines = ['Halpha','Hbeta',
             '[OI]6300','[OI]6364','[OIII]4959','[OIII]5007',
             '[NII]6548','[NII]6583','[SII]6716','[SII]6731']
#    nlines = len(lines)

# Max no. of components.
    maxncomp = 1

# Initialize line ties, n_comps, z_inits, and sig_inits.
    linetie = dict()
    ncomp = dict()
    zinit_gas = dict()
    siginit_gas = dict()
    for i in lines:
        linetie[i] = 'Halpha'
        ncomp[i] = np.full((ncols,nrows),maxncomp)
        ncomp[i][8,13] = 0
        zinit_gas[i] = np.full((ncols,nrows,maxncomp),2.5)
        siginit_gas[i] = np.full(maxncomp,50)
        #zinit_gas[i][2,18,:]=0.091
        #zinit_gas[i][4,21:22,:]=0.091
        #zinit_gas[i][5,21,:]=0.091
        #zinit_gas[i][6:8,23:25,:]=0.091
        #zinit_gas[i][1:4,1:7,:]=0.0894
        #zinit_gas[i][6:16,0:7,:]=0.089
        #zinit_gas[i][11,8:9,:]=0.089
        zinit_stars=np.full((ncols,nrows),2.5)

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
    linoth = np.full((2, 6), '', dtype=object)
    linoth[0, 2] = '[OIII]4959'
    linoth[0, 3] = '[OI]6364'
    linoth[:, 4] = ['[NII]6548', '[NII]6583']
    linoth[0, 5] = '[SII]6716'
    argspltlin1 = {'nx': 3,
                   'ny': 2,
                   'label': ['', 'Hbeta', '[OIII]5007',
                             '[OI]6300', 'Halpha', '[SII]6731'],
                   'wave': [0,4861,5007,6300,6563,6731],
                   'off': [[-120,90],[-80,50],[-130,50],
                         [-80,120],[-95,70],[-95,50]],
                   'linoth': linoth}

    # Velocity dispersion limits and fixed values
    siglim_gas = np.ndarray(2)
    siglim_gas[:] = [5, 500]
    # lratfix = {'[NI]5200/5198': [1.5]}

    #
    # Output structure
    #

    init = { \
            # Required pars
            'fcninitpar': 'gmos',
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
            'zsys_gas': 0.0898,
            # Optional pars
            'argscheckcomp': {'sigcut': 3,
                              'ignore': ['[OI]6300', '[OI]6364',
                                         '[SII]6716', '[SII]6731']},
            'argscontfit': {'blrpar': [0, 7150, 5000/299792*7150,
                                       0, 5300, 5000/299792*5300],
                            'qsoxdr': qsotemplate,
                            'siginit_stars': 50,
                            'uselog': 1,
                            'refit': 1},
            'argslinelist': {'vacuum': False},
            'startempfile': stellartemplates,
            'argspltlin1': argspltlin1,
            # 'donad': 1,
            'decompose_qso_fit': 1,
            # 'remove_scattered': 1,
            'fcncheckcomp': 'checkcomp',
            'fcncontfit': 'fitqsohost',
            #'fcncontfit': 'ppxf',
            'maskwidths_def': 500,
            'tweakcntfit': tweakcntfit,
            'emlsigcut': 2,
            'logfile': logfile,
            'batchfile': batchfile,
            'batchdir': batchdir,
            'siglim_gas': siglim_gas,
            'siginit_gas': siginit_gas,
            'siginit_stars': 50,
            'cutrange': np.array([6410, 6430]),
            'nocvdf': 1,
            # 'cvdf_vlimits': [-3e3,3e3],
            # 'cvdf_vstep': 10d,
            # 'host': {'dat_fits': volume+'ifs/gmos/cubes/'+gal+'/'+\
            #         gal+outstr+'_host_dat_2.fits'} \
        }

    return(init)
