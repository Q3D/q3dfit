#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from q3dfit.exceptions import InitializationError
from q3dfit.fitspec import fitspec
from q3dfit.sepfitpars import sepfitpars

import importlib
import numpy as np
import pdb


def fitloop(ispax, colarr, rowarr, cube, initdat, listlines, specConv, onefit,
            quiet=True, logfile=None):
    """
Created on Mon Jun  8 11:14:14 2020

Take outputs from Q3DF and perform fitting loop.


Parameters
----------

ispax : int
  Value of index over which to loop
colarr : intarr(2)
  Column # of spaxel (0-offset)
rowarr : intarr(2)
  Row # of spaxel (0-offset)
cube : object
  instance of Cube class
initdat : structure
  Output from initialization routine, containing fit parameters
onefit : byte
  If set, ignore second fit
quiet : byte, default=True
  verbosity switch from Q3DF
logfile : strarr, optional, default=None
     Names of log filesone per spaxel.

    """


    if logfile is None:
        from sys import stdout
        logfile = stdout

    # When computing masking half-widths before second fit, sigmas from first
    # fit are multiplied by this number.
    masksig_secondfit_def = 2.
    # colind = ispax % cube.ncols
    # rowind = int(ispax / cube.ncols)
    i = colarr[ispax]  # colind, rowind]
    j = rowarr[ispax]  # colind, rowind]
    print(i,j)
    if cube.dat.ndim == 1:
        print('[spec]=[1] out of [1]', file=logfile)
        if not quiet:
            print('[spec]=[1] out of [1]')
        flux = cube.dat
        err = cube.err
        dq = cube.dq
    elif cube.dat.ndim == 2:
        print(f'[spec]=[{i+1}] out of [{cube.ncols}]', file=logfile)
        if not quiet:
            print(f'[spec]=[{i+1}] out of [{cube.ncols}]')
        flux = cube.dat[:, i]
        err = cube.err[:, i]
        dq = cube.dq[:, i]
    else:
        print(f'[col,row]=[{i+1},{j+1}] out of [{cube.ncols},{cube.nrows}]',
              file=logfile)
        if not quiet:
            print(f'[col,row]=[{i+1},{j+1}] out of [{cube.ncols},{cube.nrows}]')
        flux = cube.dat[i, j, :]
        err = cube.err[i, j, :]
        dq = cube.dq[i, j, :]

    errmax = max(err)

    if initdat.__contains__('vormap'):
        tmpi = cube.vorcoords[i, 0]
        tmpj = cube.vorcoords[i, 1]
        i = tmpi
        j = tmpj
        print(f'Reference coordinate: [col, row]=[{i+1}, {j+1}]', file=logfile)
        if not quiet:
            print(f'Reference coordinate: [col, row]=[{i+1}, {j+1}]')

    if cube.dat.ndim == 1:
        outlab = '{[outdir]}{[label]}'.format(initdat, initdat)
    elif cube.dat.ndim == 2:
        outlab = '{[outdir]}{[label]}_{:04d}'.format(initdat, initdat, i+1)
    else:
        outlab = '{[outdir]}{[label]}_{:04d}_{:04d}'.format(initdat,
                                                            initdat, i+1, j+1)

#   Apply DQ plane
    if dq.ndim>0:
        indx_bad = np.nonzero(dq > 0)
    else:
        indx_bad= (dq!=0)
        if indx_bad==True:
            indx_bad=0
        else:
            indx_bad=dq
   #the oned keyword was having trouble because np.nonzero doesn't work for zero arrays. I think this fixes the issue

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
                    siglim_gas = initdat['siglim_gas'][i, j, ]
            else:
                siglim_gas = None

#           initialize gas sigma initial guess array
            if initdat.__contains__('siginit_gas'):
                if initdat['siginit_gas'][initdat['lines'][0]].ndim == 1:
                    siginit_gas = initdat['siginit_gas']
                else:
                    siginit_gas = dict()
                    for k in initdat['lines']:
                        siginit_gas[k] = initdat['siginit_gas'][k][i, j, ]
            else:
                siginit_gas = False

#           initialize stellar redshift initial guess
            if 'zinit_stars' in initdat:
                zstar = initdat['zinit_stars'][i, j]
            else:
                zstar = np.nan

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
                    listlinesz[line] = \
                        np.array(listlines['lines']
                                 [(listlines['name'] == line)]) * \
                        (1. + initdat['zinit_gas'][line][i, j, ])

            if not quiet:
                print('FITLOOP: First call to FITSPEC')

            q3dout_ij, structinit = fitspec(cube.wave, flux, err, dq, zstar, listlines,
                                 listlinesz, ncomp, specConv, initdat, quiet=quiet,
                                 siglim_gas=siglim_gas,
                                 siginit_gas=siginit_gas,
                                 tweakcntfit=tweakcntfit, col=i, row=j, logfile=logfile)
            print('FIT STATUS: '+str(structinit['fitstatus']), file=logfile)
            if not quiet:
                print('FIT STATUS: '+str(structinit['fitstatus']))

            # Second fit

            if not onefit and not abortfit:

                if 'noemlinfit' not in initdat and ct_comp_emlist > 0:

                    # set emission line mask parameters
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

                q3dout_ij, struct = fitspec(cube.wave, flux, err, dq, structinit['zstar'],
                                 listlines, listlinesz, ncomp, specConv, initdat,
                                 quiet=quiet, maskwidths=maskwidths_tmp,
                                 peakinit=peakinit_tmp,
                                 siginit_gas=siginit_gas_tmp,
                                 siglim_gas=siglim_gas,
                                 tweakcntfit=tweakcntfit, col=i, row=j, logfile=logfile)
                print('FIT STATUS: '+str(structinit['fitstatus']), file=logfile)
                if not quiet:
                    print('FIT STATUS: '+str(structinit['fitstatus']))

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
                    importlib.import_module('q3dfit.' +
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
                        if nc==0:   # CB: is there a meaning to fitting with zero components?
                            dofit = False
                            break
                        print(f'FITLOOP: Repeating the fit of {line} with ' +
                              f'{nc} components.', file=logfile)
                        if not quiet:
                            print(f'FITLOOP: Repeating the fit of {line} with ' +
                              f'{nc} components.')
                else:
                    dofit = False
            else:
                dofit = False

        # save struct to be used by q3da later
        np.save(outlab, struct)
