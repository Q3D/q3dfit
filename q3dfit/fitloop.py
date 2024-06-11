#!/usr/bin/env python3
# -*- coding: utf-8 -*-
'''
Created on Mon Jun  8 11:14:14 2020
'''
from q3dfit.exceptions import InitializationError
from q3dfit.fitspec import fitspec

import importlib
import numpy as np


def fitloop(ispax, colarr, rowarr, cube, q3di, listlines, specConv,
            onefit=False, quiet=True, logfile=None):
    """
    Take outputs from Q3DF and perform fitting loop.

    Parameters
    ----------

    ispax : int
        Value of index over which to loop
    colarr : array
        Column # of spaxel (0-offset)
    rowarr : array
        Row # of spaxel (0-offset)
    cube : object
        instance of Cube class
    q3di : object
        containing fit parameters
    onefit : bool, default=False
        If set, ignore second fit
    quiet : bool, default=True
        verbosity switch from Q3DF
    logfile : str, optional, default=None
         Name of log file

    Returns
    -------
    Nothing.
    """

    if logfile is None:
        from sys import stdout
        logfile = stdout

    i = colarr[ispax]
    j = rowarr[ispax]
    # print(i,j)
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

    if q3di.vormap is not None:
        tmpi = cube.vorcoords[i, 0]
        tmpj = cube.vorcoords[i, 1]
        i = tmpi
        j = tmpj
        print(f'Reference coordinate: [col, row]=[{i+1}, {j+1}]', file=logfile)
        if not quiet:
            print(f'Reference coordinate: [col, row]=[{i+1}, {j+1}]')

    outlab = '{0.outdir}{0.label}'.format(q3di)
    if cube.dat.ndim > 1:
        outlab += '_{:04d}'.format(i+1)
    if cube.dat.ndim > 2:
        outlab += '_{:04d}'.format(j+1)

    # Apply DQ plane
    # The oned keyword was having trouble because np.nonzero doesn't work for
    # zero arrays. I think this fixes the issue
    if dq.ndim > 0:
        indx_bad = np.nonzero(dq > 0)
    else:
        indx_bad = (dq != 0)
        if indx_bad:
            indx_bad = 0
        else:
            indx_bad = dq
    flux[indx_bad] = 0.
    err[indx_bad] = errmax*100.

#   Check that the flux is not filled with 0s, infs, or nans
    somedata = ((flux != 0.).any() and
                (flux != np.inf).any() and
                (flux != np.nan).any())
    if somedata:

        # Set up N_comp and fix/free (linevary) dictionaries for this spaxel
        ncomp = dict()
        if q3di.linevary is not None and q3di.dolinefit:
            linevary = dict()
        else:
            linevary = None
        if q3di.dolinefit:
            # Extract # of components and line fix/free specific to this spaxel
            # and write as dict. Do this outside of while loop because ncomp
            # may change with another iteration. Not sure how linevary should
            # change with iteration yet [TODO]
            # For # comp, each dict key (line) will have one value.
            # For linevary, each line will have a dict of arrays:
            #    dict has keys {'flx', 'cwv', 'sig'} with a value for each comp
            for line in q3di.lines:
                ncomp[line] = q3di.ncomp[line][i, j]
                if linevary is not None:
                    linevary[line] = dict()
                    linevary[line]['flx'] = q3di.linevary[line]['flx'][i, j, :]
                    linevary[line]['cwv'] = q3di.linevary[line]['cwv'][i, j, :]
                    linevary[line]['sig'] = q3di.linevary[line]['sig'][i, j, :]

        # First fit
        dofit = True
        abortfit = False
        while dofit:

            # Default values
            listlinesz = None
            siglim_gas = None
            siginit_gas = None
            siginit_stars = 50.
            tweakcntfit = None
            zstar = None

            # Are we doing a line fit?
            if q3di.dolinefit:

                # initialize gas sigma limit array
                if q3di.siglim_gas is not None:
                    if q3di.siglim_gas.ndim == 1:
                        siglim_gas = q3di.siglim_gas
                    else:
                        siglim_gas = q3di.siglim_gas[i, j, ]

                #  initialize gas sigma initial guess array
                if q3di.siginit_gas is not None:
                    if q3di.siginit_gas[q3di.lines[0]].ndim == 1:
                        siginit_gas = q3di.siginit_gas
                    else:
                        siginit_gas = dict()
                        for k in q3di.lines:
                            siginit_gas[k] = q3di.siginit_gas[k][i, j, ]

                # initialize starting wavelengths for lines
                # u['line'][(u['name']=='Halpha')]
                listlinesz = dict()
                if q3di.dolinefit:
                    for line in q3di.lines:
                        listlinesz[line] = \
                            np.array(listlines['lines']
                                     [(listlines['name'] == line)],
                                     dtype='float64') * \
                                (1. + q3di.zinit_gas[line][i, j, ])

            if q3di.docontfit:

                # initialize stellar redshift initial guess
                if q3di.zinit_stars is not None:
                    zstar = q3di.zinit_stars[i, j]

                #  initialize stellar sigma initial guess array
                if q3di.siginit_stars is not None:
                    siginit_stars = q3di.siginit_stars[i, j]

                # option to tweak continuum fit
                if q3di.tweakcntfit is not None:
                    tweakcntfit = q3di.tweakcntfit[i, j, :, :]

            # regions to ignore in fitting. Set to max(err)
            if q3di.cutrange is not None:
                if q3di.cutrange.ndim == 1:
                    indx_cut = \
                        np.intersect1d((cube.wave >=
                                        q3di.cutrange[0]).nonzero(),
                                       (cube.wave <=
                                        q3di.cutrange[1]).nonzero())
                    if indx_cut.size != 0:
                        dq[indx_cut] = 1
                        err[indx_cut] = errmax*100.
                elif q3di.cutrange.ndim == 2:
                    for k in range(q3di.cutrange.shape[0]):
                        indx_cut = \
                            np.intersect1d((cube.wave >=
                                            q3di.cutrange
                                            [k, 0]).nonzero(),
                                           (cube.wave <=
                                            q3di.cutrange
                                            [k, 1]).nonzero())
                        if indx_cut.size != 0:
                            dq[indx_cut] = 1
                            err[indx_cut] = errmax*100.
                else:
                    raise InitializationError('CUTRANGE not' +
                                              ' properly specified')

            if not quiet:
                print('FITLOOP: First call to FITSPEC')
            q3do_init = fitspec(cube.wave, flux, err, dq, zstar, listlines,
                                listlinesz, ncomp, specConv, q3di, quiet=quiet,
                                linevary=linevary,
                                siglim_gas=siglim_gas,
                                siginit_gas=siginit_gas,
                                siginit_stars=siginit_stars,
                                tweakcntfit=tweakcntfit, logfile=logfile)
            # Abort if no good data
            if q3do_init.nogood:
                abortfit = True
                print('FITLOOP: Aborting fit; no good data to fit.',
                      file=logfile)
                if not quiet:
                    print('FITLOOP: Aborting fit; no good data to fit.')
            else:
                print('FIT STATUS: '+str(q3do_init.fitstatus), file=logfile)
                if not quiet:
                    print('FIT STATUS: '+str(q3do_init.fitstatus))

            # Second fit

            if not onefit and not abortfit:

                maskwidths_tmp = None
                peakinit_tmp = None
                siginit_gas_tmp = None

                if q3do_init.dolinefit:
                    # set emission line mask parameters
                    q3do_init.sepfitpars()
                    listlinesz = q3do_init.line_fitpars['wave']
                    maskwidths = q3do_init.line_fitpars['sigma_obs']
                    # Multiply sigmas from first fit by MASKSIG_SECONDFIT
                    # to get half-widths for masking
                    for col in maskwidths.columns:
                        maskwidths[col] *= q3di.masksig_secondfit
                    maskwidths_tmp = maskwidths
                    peakinit_tmp = q3do_init.line_fitpars['fluxpk_obs']
                    siginit_gas_tmp = q3do_init.line_fitpars['sigma']

                if not quiet:
                    print('FITLOOP: Second call to FITSPEC')
                if q3di.fcncontfit == 'questfit' and \
                    hasattr(q3di,'argscontfit'):
                    q3di.argscontfit['rows'] = j+1
                    q3di.argscontfit['cols'] = i+1
                q3do = fitspec(cube.wave, flux, err, dq, q3do_init.zstar,
                               listlines, listlinesz, ncomp, specConv, q3di,
                               quiet=quiet,
                               linevary=linevary,
                               maskwidths=maskwidths_tmp,
                               peakinit=peakinit_tmp,
                               siginit_gas=siginit_gas_tmp,
                               siginit_stars=siginit_stars,
                               siglim_gas=siglim_gas,
                               tweakcntfit=tweakcntfit, logfile=logfile)
                print('FIT STATUS: '+str(q3do.fitstatus), file=logfile)
                if not quiet:
                    print('FIT STATUS: '+str(q3do.fitstatus))

            elif onefit and not abortfit:

                if q3do_init.dolinefit:
                    q3do_init.sepfitpars()
                q3do = q3do_init

            else:

                q3do = q3do_init

            # Check components

            if q3di.checkcomp and q3do.dolinefit and \
                not onefit and not abortfit:

                siglim_gas = q3do.siglim

                q3do.sepfitpars()

                ccModule = \
                    importlib.import_module('q3dfit.' +
                                            q3di.fcncheckcomp)
                fcncheckcomp = getattr(ccModule, q3di.fcncheckcomp)
                # Note that this modifies the value of ncomp if necessary
                newncomp = fcncheckcomp(q3do.line_fitpars, q3di.linetie, ncomp,
                                        siglim_gas, **q3di.argscheckcomp)

                if len(newncomp) > 0:
                    for line, nc in newncomp.items():
                        print(f'FITLOOP: Repeating the fit of {line} with ' +
                              f'{nc} components.', file=logfile)
                        if not quiet:
                            print(f'FITLOOP: Repeating the fit of {line} ' +
                                  f'with {nc} components.')
                else:
                    dofit = False
            else:
                dofit = False

        if not abortfit:
            # save q3do
            q3do.col = i+1
            q3do.row = j+1

            # update units, etc.
            q3do.fluxunit = cube.fluxunit_out
            q3do.waveunit = cube.waveunit_out
            q3do.fluxnorm = cube.fluxnorm
            q3do.pixarea_sqas = cube.pixarea_sqas
            #
            np.save(outlab, q3do, allow_pickle=True)
