#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jun  8 11:14:14 2020

@author: drupke

Take outputs from Q3DF and perform fitting loop.

:Returns:
   None.

:Params:
   ispax: in, required, type=int
     Value of index over which to loop
   colarr: in, required,type=intarr(2)
     Column # of spaxel (0-offset)
   rowarr: in, required, type=intarr(2)
     Row # of spaxel (0-offset)
   cube: in, required, type=structure
     Output from READCUBE, containing data
   initdat: in, required, type=structure
     Output from initialization routine, containing fit parameters
   oned: in, required, type=byte
     Whether data is in a cube or in one dimension (longslit)
   onefit: in, required, type=byte
     If set, ignore second fit
   quiet: in, required, type=byte
     verbosity switch from Q3DF

:Keywords:
   logfile: in, optional, type=strarr
     Names of log filesone per spaxel.


:History:
   ChangeHistory::
     2016sep18, DSNR, copied from IFSF into standalone procedure
     2016sep26, DSNR, small change in masking for new treatment of spec. res.
     2016oct20, DSNR, fixed treatment of SIGINIT_GAS
     2016nov17, DSNR, added flux calibration
     2018jun25, DSNR, added MC error calculation on stellar parameters
     2021jan20, DSNR, finished translating to Python
     2021feb22, DSNR, updated logfile treatment

"""

from q3dfit.exceptions import InitializationError
from q3dfit.common.fitspec import fitspec
from q3dfit.common.sepfitpars import sepfitpars

import importlib
import numpy as np
import pdb


def fitloop(ispax, colarr, rowarr, cube, initdat, listlines, oned, onefit,
            quiet=True, logfile=None):

    if logfile is None:
        from sys import stdout
        logfile = stdout

    # When computing masking half-widths before second fit, sigmas from first
    # fit are multiplied by this number.
    masksig_secondfit_def = 2.
    # colind = ispax % cube.ncols
    # rowind = int(ispax / cube.ncols)
    # pdb.set_trace()
    i = colarr[ispax]  # colind, rowind]
    j = rowarr[ispax]  # colind, rowind]

    print(f'[col,row]=[{i+1},{j+1}] out of [{cube.ncols},{cube.nrows}]',
          file=logfile)

    if oned:
        flux = cube.dat[:, i]
        err = abs(cube.var[:, i])**0.5
        dq = cube.dq[:, i]
    else:
        flux = cube.dat[i, j, :]
        err = abs(cube.var[i, j, :])**0.5
        dq = cube.dq[i, j, :]

    errmax = max(err)

    if initdat.__contains__('vormap'):
        tmpi = cube.vorcoords[i, 0]
        tmpj = cube.vorcoords[i, 1]
        i = tmpi
        j = tmpj
        print(f'Reference coordinate: [col, row]=[{i+1}, {j+1}]', file=logfile)

    if oned:
        outlab = '{[outdir]}{[label]}_{:04d}'.format(initdat, initdat, i+1)
    else:
        outlab = '{[outdir]}{[label]}_{:04d}_{:04d}'.format(initdat,
                                                            initdat, i+1, j+1)
        
#   Apply DQ plane
    indx_bad = np.nonzero(dq > 0)
    if indx_bad[0].size > 0:
        flux[indx_bad] = 0.
        err[indx_bad] = errmax*100.

#   Check that the flux is not filled with 0s, infs, or nans
    somedata = ((flux != 0.).any() and
                (flux != np.inf).any() and
                (flux != np.nan).any())
    if somedata:

        ncomp = dict()
        if 'noemlinfit' not in initdat:

            # Extract # of components specific to this spaxel and
            # write as dict
            # Each dict key (line) will have one value (# comp)
            for line in initdat['lines']:
                if oned:
                    ncomp[line] = initdat['ncomp'][line][i]
                else:
                    ncomp[line] = initdat['ncomp'][line][i, j]

        # First fit

        dofit = True
        abortfit = False
        while(dofit):

            # Make sure ncomp > 0 for at least one line
            ct_comp_emlist = 0
            if not initdat.__contains__('noemlinfit'):
                for k in ncomp.values():
                    if k > 0:
                        ct_comp_emlist += 1

#           initialize gas sigma limit array
            if initdat.__contains__('siglim_gas'):
                if initdat['siglim_gas'].ndim == 1:
                    siglim_gas = initdat['siglim_gas']
                else:
                    if oned:
                        siglim_gas = initdat['siglim_gas'][i, ]
                    else:
                        siglim_gas = initdat['siglim_gas'][i, j, ]
            else:
                siglim_gas = False

#           initialize gas sigma initial guess array
            if initdat.__contains__('siginit_gas'):
                if initdat['siginit_gas'][initdat['lines'][0]].ndim == 1:
                    siginit_gas = initdat['siginit_gas']
                else:
                    siginit_gas = dict()
                    if oned:
                        for k in initdat['lines']:
                            siginit_gas[k] = initdat['siginit_gas'][k][i, ]
                    else:
                        for k in initdat['lines']:
                            siginit_gas[k] = initdat['siginit_gas'][k][i, j, ]
            else:
                siginit_gas = False

#           initialize stellar redshift initial guess
            if oned:
                zstar = initdat['zinit_stars'][i]
            else:
                zstar = initdat['zinit_stars'][i, j]

#           regions to ignore in fitting. Set to max(err)
            if initdat.__contains__('cutrange'):
                if initdat['cutrange'].ndim == 1:
                    indx_cut = \
                        np.intersect1d((cube.wave >=
                                        initdat['cutrange'][0]).nonzero(),
                                       (cube.wave <=
                                        initdat['cutrange'][1]).nonzero())
                    if indx_cut.size != 0:
                        dq[indx_cut] = 1
                        err[indx_cut] = errmax*100.
                elif initdat['cutrange'].ndim == 2:
                    for k in range(initdat['cutrange'].shape[0]):
                        indx_cut = \
                            np.intersect1d((cube.wave >=
                                            initdat['cutrange']
                                            [k, 0]).nonzero(),
                                           (cube.wave <=
                                            initdat['cutrange']
                                            [k, 1]).nonzero())
                        if indx_cut.size != 0:
                            dq[indx_cut] = 1
                            err[indx_cut] = errmax*100.
                else:
                    raise InitializationError('CUTRANGE not' +
                                              ' properly specified')

            # option to tweak continuum fit
            tweakcntfit = False
            if initdat.__contains__('tweakcntfit'):
                tweakcntfit = initdat['tweakcntfit'][i, j, :, :]

            # initialize starting wavelengths
            # should this be astropy table? dict of numpy arrays?
            # u['line'][(u['name']=='Halpha')]
            listlinesz = dict()
            if not initdat.__contains__('noemlinfit') and ct_comp_emlist > 0:
                for line in initdat['lines']:
                    if oned:
                        listlinesz[line] = \
                            np.array(listlines['lines']
                                     [(listlines['name'] == line)]) * \
                            (1. + initdat['zinit_gas'][line][i, ])
                    else:
                        listlinesz[line] = \
                            np.array(listlines['lines']
                                     [(listlines['name'] == line)]) * \
                            (1. + initdat['zinit_gas'][line][i, j, ])

            if not quiet:
                print('FITLOOP: First call to FITSPEC')

            structinit = fitspec(cube.wave, flux, err, dq, zstar, listlines,
                               listlinesz, ncomp, initdat, quiet=quiet,
                                 siglim_gas=siglim_gas,
                                 siginit_gas=siginit_gas,
                                 tweakcntfit=tweakcntfit, col=i+1, row=j+1)

            # if not quiet:
            #    print('FIT STATUS: '+structinit['fitstatus'])
            # To-do: Need to add a check on fit status here.

            # Second fit

            if not onefit and not abortfit:

                if 'noemlinfit' not in initdat and ct_comp_emlist > 0:

                    # set emission line mask pa rameters
                    linepars = sepfitpars(listlines, structinit['param'],
                                          structinit['perror'],
                                          initdat['maxncomp'])
                    listlinesz = linepars['wave']
                    # Multiply sigmas from first fit by MASKSIG_SECONDFIT_DEF
                    # to get half-widths for masking
                    if 'masksig_secondfit' in initdat:
                        masksig_secondfit = initdat['masksig_secondfit']
                    else:
                        masksig_secondfit = masksig_secondfit_def
                    maskwidths = linepars['sigma_obs']
                    for col in maskwidths.columns:
                        maskwidths[col] *= masksig_secondfit
                    maskwidths_tmp = maskwidths
                    peakinit_tmp = linepars['fluxpk_obs']
                    siginit_gas_tmp = linepars['sigma']

                else:

                    maskwidths_tmp = None
                    peakinit_tmp = None
                    siginit_gas_tmp = None

                if not quiet:
                    print('FITLOOP: Second call to FITSPEC')
                struct = fitspec(cube.wave, flux, err, dq, structinit['zstar'],
                                 listlines, listlinesz, ncomp, initdat,
                                 quiet=quiet, maskwidths=maskwidths_tmp,
                                 peakinit=peakinit_tmp,
                                 siginit_gas=siginit_gas_tmp,
                                 siglim_gas=siglim_gas,
                                 tweakcntfit=tweakcntfit, col=i+1, row=j+1)

                # if not quiet:
                #    print('FIT STATUS: '+structinit['fitstatus'])
                # To-do: Need to add a check on fit status here.

            else:

                struct = structinit

            # Check components

            if 'fcncheckcomp' in initdat and \
                'noemlinfit' not in initdat and \
                    not onefit and not abortfit and \
                    ct_comp_emlist > 0:

                siglim_gas = struct['siglim']

                linepars = sepfitpars(listlines, struct['param'],
                                      struct['perror'], initdat['maxncomp'])
                ccModule = \
                    importlib.import_module('q3dfit.common.' +
                                            initdat['fcncheckcomp'])
                fcncheckcomp = getattr(ccModule, initdat['fcncheckcomp'])
                # Note that this modifies the value of ncomp if necessary
                if 'argscheckcomp' in initdat:
                    newncomp = \
                        fcncheckcomp(linepars, initdat['linetie'],
                                     ncomp, siglim_gas,
                                     **initdat['argscheckcomp'])
                else:
                    newncomp = \
                        fcncheckcomp(linepars, initdat['linetie'],
                                     ncomp, siglim_gas)

                if len(newncomp) > 0:
                    for line, nc in newncomp.items():
                        print(f'FITLOOP: Repeating the fit of {line} with ' +
                              f'{nc} components.', file=logfile)
                else:
                    dofit = False
            else:
                dofit = False

        # save struct to be used by q3da later
        np.save(outlab, struct)
