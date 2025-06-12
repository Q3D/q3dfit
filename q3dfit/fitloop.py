#!/usr/bin/env python3
# -*- coding: utf-8 -*-
from q3dfit.exceptions import InitializationError
from q3dfit.fitspec import fitspec

import importlib
from typing import Optional

import os

import numpy as np
from astropy.table import Table

from q3dfit.q3din import q3din
from q3dfit.q3dutil import write_msg
from q3dfit.readcube import Cube
from q3dfit.spectConvol import spectConvol


def fitloop(ispax: int,
            colarr: np.ndarray,
            rowarr: np.ndarray,
            cube: Cube,
            q3di: q3din,
            listlines: Table,
            specConv: Optional[spectConvol],
            onefit: bool=False,
            quiet: bool=True):
    """
    Perform fitting loops over spaxels. This function calls
    :py:func:`~q3dfit.fitspec.fitspec` to fit the data.

    Parameters
    ----------

    ispax
        Value of index over which to loop.
    colarr
        Array of column #s of spaxels to be fit (0-offset).
    rowarr
        Array of row #s of spaxels to be fit (0-offset).
    cube
        :py:class:`~q3dfit.readcube.Cube` object containing data to be fit.
    q3di
        :py:class:`~q3dfit.q3din.q3din` object containing fitting parameters.
    listlines
        Emission line labels and rest frame wavelengths, as part of an astropy Table
        output by :py:func:`~q3dfit.linelist.linelist`.
    specConv
        Instance of :py:class:`~q3dfit.spectConvol.spectConvol` specifying 
        the instrumental spectral resolution convolution. If None, no convolution
        is performed.
    onefit
        If set, only one fit is performed. Default is False.
    quiet
        Optional. Suppress progress messages. Default is False.
    """

    i = colarr[ispax]
    j = rowarr[ispax]
    # print(i,j)
    if cube.dat.ndim == 1:
        write_msg('[spec]=[1] out of [1]', file=q3di.logfile, quiet=quiet)
        flux = cube.dat
        err = cube.err
        dq = cube.dq
    elif cube.dat.ndim == 2:
        write_msg(f'[spec]=[{i+1}] out of [{cube.ncols}]', file=q3di.logfile, quiet=quiet)
        flux = cube.dat[:, i]
        err = cube.err[:, i]
        dq = cube.dq[:, i]
    else:
        write_msg(f'[col,row]=[{i+1},{j+1}] out of [{cube.ncols},{cube.nrows}]',
            file=q3di.logfile, quiet=quiet)
        flux = cube.dat[i, j, :]
        err = cube.err[i, j, :]
        dq = cube.dq[i, j, :]

    errmax = max(err)

    '''
    if q3di.vormap is not None:
        tmpi = cube.vorcoords[i, 0]
        tmpj = cube.vorcoords[i, 1]
        i = tmpi
        j = tmpj
        print(f'Reference coordinate: [col, row]=[{i+1}, {j+1}]', file=logfile)
        if not quiet:
            print(f'Reference coordinate: [col, row]=[{i+1}, {j+1}]')
    '''

    outlab = os.path.join(q3di.outdir, q3di.label)
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

    # Indexing to wavelength array for regions to ignore in fitting
    indx_cut = []
    if q3di.cutrange is not None:
        if not isinstance(q3di.cutrange, np.ndarray):
            q3di.cutrange = np.array(q3di.cutrange)
        if q3di.cutrange.ndim == 1:
            ncut = 1
        else:
            ncut = q3di.cutrange.shape[0]
        try:
            for icut in range(ncut):
                indx_cut.append(
                    np.intersect1d(
                        (cube.wave >= q3di.cutrange.ravel()[0+icut*2]).nonzero(),
                        (cube.wave <= q3di.cutrange.ravel()[1+icut*2]).nonzero()))
        except:
           raise InitializationError('CUTRANGE not properly specified')

#   Check that the flux is not filled with 0s, infs, or nans
    somedata = ((flux != 0.).any() and
                (flux != np.inf).any() and
                (flux != np.nan).any())
    if somedata:

        # Set up N_comp and fix/free (linevary) dictionaries for this spaxel
        # These are set up for the first fit and don't need to change
        # after checkcomp() is called
        if q3di.dolinefit:
            ncomp = dict()
            linevary = dict()
            # Extract # of components and line fix/free specific to this spaxel
            # and write as dict. Do this outside of while loop because ncomp
            # may change with another iteration. Not sure how linevary should
            # change with iteration yet [TODO]
            # For # comp, each dict key (line) will have one value.
            # For linevary, each line will have a dict of arrays:
            #    dict has keys {'flx', 'cwv', 'sig'} with a value for each comp
            for line in q3di.lines:
                ncomp[line] = q3di.ncomp[line][i, j]
                linevary[line] = dict()
                linevary[line]['flx'] = q3di.linevary[line]['flx'][i, j, :]
                linevary[line]['cwv'] = q3di.linevary[line]['cwv'][i, j, :]
                linevary[line]['sig'] = q3di.linevary[line]['sig'][i, j, :]
        
        else:
            ncomp = None
            linevary = None

        # First fit
        dofit = True
        abortfit = False
        while dofit:

            # Default values
            # These are parameters that are reset after checkcomp is called.
            # These Nones apply if no line fit is done.
            listlinesz = None
            siglim_gas = None
            siginit_gas = None
            #tweakcntfit = None
            # This is if no continuum fit is done
            zstar = None
            siginit_stars = None

            # Are we doing a line fit?
            if q3di.dolinefit:
                # initialize for this spaxel:
                # gas sigma initial guess array
                # gas sigma limit array
                # redshifted starting wavelengths for lines
                # peak flux initial guess array
                siginit_gas = dict()
                siglim_gas = dict()
                listlinesz = dict()
                for k in q3di.lines:
                    siginit_gas[k] = q3di.siginit_gas[k][i, j, ]
                    siglim_gas[k] = q3di.siglim_gas[k][i, j, ]
                    listlinesz[k] = \
                        np.array(listlines['lines'][(listlines['name'] == k)], 
                            dtype=np.float64) * (1. + q3di.zinit_gas[k][i, j, ])

            if q3di.docontfit:

                # initialize for this spaxel:
                # stellar redshift initial guess
                # stellar sigma initial guess
                if q3di.zinit_stars is not None:
                    zstar = q3di.zinit_stars[i, j]
                    siginit_stars = q3di.siginit_stars[i, j]

                # option to tweak continuum fit
                #if q3di.tweakcntfit is not None:
                #    tweakcntfit = q3di.tweakcntfit[i, j, :, :]

            # regions to ignore in fitting: set to max(err)
            for indx in indx_cut:
                if indx.size != 0:
                    dq[indx] = 1
                    err[indx] = errmax*100.

            write_msg('FITLOOP: First call to FITSPEC', file=q3di.logfile, quiet=quiet)
            q3do_init = fitspec(cube.wave, flux, err, dq, q3di, 
                                zstar=zstar, 
                                siginit_stars=siginit_stars,
                                listlines=listlines,
                                listlinesz=listlinesz,
                                ncomp=ncomp,
                                linevary=linevary,
                                siginit_gas=siginit_gas,
                                siglim_gas=siglim_gas,
                                specConv=specConv, 
                                quiet=quiet,
                                #weakcntfit=tweakcntfit, 
                                fluxunit=cube.fluxunit_out,
                                waveunit=cube.waveunit_out)
            # Abort if no good data
            if q3do_init.nogood:
                abortfit = True
                write_msg('FITLOOP: Aborting fit; no good data to fit.',
                    file=q3di.logfile, quiet=quiet)
            else:
                if q3di.dolinefit:
                    write_msg('FIT STATUS: '+str(q3do_init.fitstatus), 
                              file=q3di.logfile, quiet=quiet)

            # Second fit

            if not onefit and not abortfit:

                # default values for no line fit
                maskwidths = None
                peakinit = None
                siginit_gas = None

                if q3do_init.dolinefit:
                    # get the line fit parameters from the first fit
                    q3do_init.sepfitpars()
                    # initialize for this spaxel:
                    # initial guess for redshifted wavelengths for lines
                    # maskwidths for lines
                    # initial guess for peak fluxes
                    # initial guess for gas sigmas
                    listlinesz = q3do_init.line_fitpars['wave']
                    maskwidths = q3do_init.line_fitpars['sigma_obs']
                    # Multiply sigmas from first fit by MASKSIG_SECONDFIT
                    # to get half-widths for masking
                    for col in maskwidths.columns:
                        maskwidths[col] *= q3di.masksig_secondfit
                    peakinit = q3do_init.line_fitpars['fluxpk_obs']
                    siginit_gas = q3do_init.line_fitpars['sigma']

                write_msg('FITLOOP: Second call to FITSPEC', file=q3di.logfile, quiet=quiet)
                if q3di.fcncontfit == 'questfit' and \
                    hasattr(q3di,'argscontfit'):
                    q3di.argscontfit['rows'] = j+1
                    q3di.argscontfit['cols'] = i+1
                q3do = fitspec(cube.wave, flux, err, dq,  q3di,
                               zstar=q3do_init.zstar,
                               siginit_stars=siginit_stars,
                               listlines=listlines,
                               listlinesz=listlinesz,
                               ncomp=ncomp,
                               linevary=linevary,
                               maskwidths=maskwidths,
                               peakinit=peakinit,
                               siginit_gas=siginit_gas,
                               siglim_gas=siglim_gas,
                               #tweakcntfit=tweakcntfit, 
                               specConv=specConv,
                               quiet=quiet,
                               fluxunit=cube.fluxunit_out,
                               waveunit=cube.waveunit_out)
                if q3do_init.dolinefit:
                    write_msg('FIT STATUS: '+str(q3do.fitstatus),
                              file=q3di.logfile, quiet=quiet)

            elif onefit and not abortfit:

                if q3do_init.dolinefit:
                    q3do_init.sepfitpars()
                q3do = q3do_init

            else:

                q3do = q3do_init

            # Check components
            if q3di.dolinefit:
                if q3di.checkcomp and not onefit and not abortfit:

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
                            write_msg(f'FITLOOP: Repeating the fit of {line} with ' +
                                    f'{nc} components.', file=q3di.logfile, quiet=quiet)
                    else:
                        dofit = False
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
