# -*- coding: utf-8 -*-
import copy as copy
import importlib.resources as pkg_resources
import numpy as np
import os
from typing import Literal, Optional

from astropy.table import Table

from q3dfit.data import linelists
import q3dfit.q3dutil as q3dutil
from . import q3din, q3dout


def q3dcollect(q3di: str | q3din.q3din,
               cols: Optional[int | list]=None,
               rows: Optional[int | list]=None,
               quiet: bool=True,
               compsortpar: Literal['flux','sigma','wave']='sigma',
               compsortdir: Literal['down','up']='up',
               ignoreres: bool=False):
    
    """
    Collate spaxel information together. As input, it requires a
    :py:class:`~q3dfit.q3din.q3din' object and the :py:class:`~q3dfit.q3dout.q3dout` 
    objects output by the fit.

    Nothing is returned, but the result is saved to the location :py:attr:`~q3dfit.q3di.q3di.outdir'
    with the filenames :py:attr:`~q3dfit.q3di.q3di.label`.line.npz and 
    :py:attr:`~q3dfit.q3di.q3di.label`.cont.npy.
    The :py:attr:`~q3dfit.q3di.q3di.label`.line.npz file contains the following
    dictionaries: emlwav, emlwaverr, emlsig, emlsigerr, emlflx, emlflxerr, emlweq, and emlncomp; 
    and the values ncols, nrows. The :py:attr:`~q3dfit.q3di.q3di.label`.cont.npy file contains
    the contcube dictionary.

    Parameters
    ----------
    q3di
        :py:class:`~q3dfit.q3din.q3din` object or file name.
    cols
        Optional. Column values for spaxels to be collated. Default is None, which means
        all columns will be collated. If a scalar, only that column will be collated. If a
        2-element list, the elements are the starting and ending columns to be
        collated. Unity-offset values assumed.
    rows
        Optional. Column values for spaxels to be collated. Default is None, which means
        all columns will be collated. If a scalar, only that column will be collated. If a
        2-element list, the elements are the starting and ending columns to be
        collated. Unity-offset values assumed.
    quiet
        Optional. If True, suppresses output to stdout. Default is True.
    compsortpar
        Optional. Parameter by which to sort components. Options are 'sigma', 'flux',
        'wave'. Default is 'sigma'.
    compsortdir
        Optional. Direction in which to sort components. Options are 'up' and 'down'.
        Default is 'up'.
    ignoreres
        Optional. Parameter passed to :py:meth:`~q3dfit.q3dout.q3dout.sepfitpars`. 
        If True, ignore spectral resolution in computing observed sigmas and peak fluxes. 
        This is mainly for backward compatibility with old versions, which did not store the 
        spectral resolution in an easily accessible way in the specConv object. 
        Default is False.
    """

    #load initialization object
    q3dii: q3din.q3din = q3dutil.get_q3dio(q3di)

    # set up linelist
    if q3dii.dolinefit:

        q3dutil.write_msg(f'Sorting components by {compsortpar} in the {compsortdir}ward direction.',
                          q3dii.logfile, quiet)

        linelist = q3dii.get_linelist()

        # table with doublets to combine
        with pkg_resources.path(linelists, 'doublets.tbl') as p:
            doublets = Table.read(p, format='ipac')
        # make a copy of singlet list
        lines_with_doublets = copy.deepcopy(q3dii.lines)
        # append doublet names to singlet list
        for (name1, name2) in zip(doublets['line1'], doublets['line2']):
            if name1 in linelist['name'] and name2 in linelist['name']:
                lines_with_doublets.append(name1+'+'+name2)

    # READ DATA
    cube = q3dii.load_cube(quiet=quiet)

    # process col, row specifications
    nspax, colarr, rowarr = \
        q3dutil.get_spaxels(cube.ncols, cube.nrows, cols=cols, rows=rows)
    # TODO
    # if q3dii.vormap is not None:
    #     vormap = q3dii.vromap
    #     nvorcols = max(vormap)
    #     vorcoords = np.zeros(nvorcols, 2)
    #     for i in range(0, nvorcols):
    #         xyvor = np.where(vormap == i).nonzero()
    #         vorcoords[:, i] = xyvor


    # Create output line dictionaries
    if q3dii.dolinefit:

        shape2d = (int(cube.ncols), int(cube.nrows))
        shape3d = (int(cube.ncols), int(cube.nrows), int(cube.nwave))
        emlwav = dict()
        emlwaverr = dict()
        emlsig = dict()
        emlsigerr = dict()
        emlweq = dict()
        emlflx = dict()
        emlflxerr = dict()
        emlncomp = dict()
        emlweq['ftot'] = dict()
        emlflx['ftot'] = dict()
        emlflxerr['ftot'] = dict()
        for k in range(0, q3dii.maxncomp):
            cstr = 'c' + str(k + 1)
            emlwav[cstr] = dict()
            emlwaverr[cstr] = dict()
            emlsig[cstr] = dict()
            emlsigerr[cstr] = dict()
            emlweq['f' + cstr] = dict()
            emlflx['f' + cstr] = dict()
            emlflxerr['f' + cstr] = dict()
            emlflx['f' + cstr + 'pk'] = dict()
            emlflxerr['f' + cstr + 'pk'] = dict()
        for line in lines_with_doublets:
            emlncomp[line] = np.zeros(shape2d, dtype=int)
            emlweq['ftot'][line] = np.zeros(shape2d, dtype=float) + np.nan
            emlflx['ftot'][line] = np.zeros(shape2d, dtype=float) + np.nan
            emlflxerr['ftot'][line] = np.zeros(shape2d, dtype=float) + np.nan
            for k in range(0, q3dii.maxncomp):
                cstr = 'c' + str(k + 1)
                emlwav[cstr][line] = np.zeros(shape2d, dtype=float) + np.nan
                emlwaverr[cstr][line] = np.zeros(shape2d, dtype=float) + np.nan
                emlsig[cstr][line] = np.zeros(shape2d, dtype=float) + np.nan
                emlsigerr[cstr][line] = np.zeros(shape2d, dtype=float) + np.nan
                emlweq['f'+cstr][line] = np.zeros(shape2d, dtype=float) + np.nan
                emlflx['f'+cstr][line] = np.zeros(shape2d, dtype=float) + np.nan
                emlflxerr['f'+cstr][line] = np.zeros(shape2d, dtype=float) + np.nan
                emlflx['f'+cstr+'pk'][line] = \
                    np.zeros(shape2d, dtype=float) + np.nan
                emlflxerr['f'+cstr+'pk'][line] = \
                    np.zeros(shape2d, dtype=float) + np.nan

    # create output cubes
    if q3dii.docontfit:
        hostcube = {'dat': np.zeros(shape3d),
                    'err': np.zeros(shape3d),
                    'dq':  np.zeros(shape3d),
                    'norm_div': np.zeros(shape3d),
                    'norm_sub': np.zeros(shape3d)}
        contcube = {'npts': np.zeros(shape2d) + np.nan,
                    'stel_rchisq': np.zeros(shape2d) + np.nan,
                    'stel_z': np.zeros(shape2d) + np.nan,
                    'stel_z_err': np.zeros((cube.ncols, cube.nrows, 2)) + np.nan,
                    'stel_av': np.zeros(shape2d) + np.nan,
                    'stel_av_err':
                        np.zeros((cube.ncols, cube.nrows, 2)) + np.nan,
                    'stel_sigma': np.zeros(shape2d) + np.nan,
                    'stel_sigma_err':
                        np.zeros((cube.ncols, cube.nrows, 2)) + np.nan}

        if q3dii.fcncontfit == 'ppxf':
            contcube['all_mod'] = np.zeros(shape3d)
            contcube['stel_mod'] = np.zeros(shape3d)
            contcube['poly_mod'] = np.zeros(shape3d)
        elif q3dii.fcncontfit == 'fitqsohost':
            contcube['qso_mod'] = np.zeros(shape3d)
            contcube['host_mod'] = np.zeros(shape3d)
            contcube['stel_mod'] = np.zeros(shape3d)
            contcube['poly_mod'] = np.zeros(shape3d)
            contcube['all_mod'] = np.zeros(shape3d)
        else:
            contcube['all_mod'] = np.zeros(shape3d)

    # LOOP THROUGH SPAXELS

    # track first continuum fit
    firstcontfit = True

    for ispax in range(0, nspax):

        i = colarr[ispax]
        j = rowarr[ispax]

        q3dutil.write_msg(f'Column {i+1} of {cube.ncols}', q3dii.logfile, quiet)

        # set up labeling

        # set this to false unless we're using Voronoi binning
        # and the tiling is missing
        # vortile = True
        labin = '{0.outdir}{0.label}'.format(q3dii)
        if cube.dat.ndim == 1:
            flux = cube.dat
            err = cube.err
            dq = cube.dq
            labout = labin
        elif cube.dat.ndim == 2:
            flux = cube.dat[:, i]
            err = cube.err[:, i]
            dq = cube.dq[:, i]
            labin += '_{:04d}'.format(i+1)
            labout = labin
        else:
            q3dutil.write_msg(f'    Row {j+1} of {cube.nrows}', q3dii.logfile, quiet)

            # TODO
            # if q3dii.vormap is not None:
            #    if np.isfinite(q3dii.vormap[i][j]) and \
            #            q3dii.vormap[i][j] is not np.nan:
            #        iuse = vorcoords[q3dii.vormap[i][j] - 1, 0]
            #        juse = vorcoords[q3dii.vormap[i][j] - 1, 1]
            #    else:
            #        vortile = False
            #else:
            iuse = i
            juse = j

            #if vortile:
            flux = cube.dat[iuse, juse, :].flatten()
            err = cube.err[iuse, juse, :].flatten()
            dq = cube.dq[iuse, juse, :].flatten()
            labin = '{0.outdir}{0.label}_{1:04d}_{2:04d}'.\
                format(q3dii, iuse+1, juse+1)
#            labout = '{0.outdir}{0.label}_{1:04d}_{2:04d}'.\
#                format(q3dii, i+1, j+1)

        # Restore fit after a couple of sanity checks
        # if vortile:
        infile = labin + '.npy'
        nodata = flux.nonzero()
        ct = len(nodata[0])
        # else:
        #    ct = 0

        if not os.path.isfile(infile):
            q3dutil.write_msg(f'No data for [{i+1}, {j+1}]', q3dii.logfile, quiet)

        else:

            # load fit object
            q3do: q3dout.q3dout = q3dutil.get_q3dio(infile)

            # Restore original error.
            q3do.spec_err = err[q3do.fitran_indx]

            if q3do.dolinefit:

                # process line fit parameters
                q3do.sepfitpars(doublets=doublets, ignoreres=ignoreres)

                # get correct number of components in this spaxel
                thisncomp = 0
                thisncompline = ''

                for line in lines_with_doublets:
                    sigtmp = q3do.line_fitpars['sigma'][line]
                    fluxtmp = q3do.line_fitpars['flux'][line]
                    # TODO
                    igd = [idx for idx in range(len(sigtmp)) if
                           (sigtmp[idx] != 0 and
                            not np.isnan(sigtmp[idx]) and
                            fluxtmp[idx] != 0 and
                            not np.isnan(fluxtmp[idx]))]
                    ctgd = len(igd)

                    if ctgd > thisncomp:
                        thisncomp = ctgd
                        thisncompline = line

                    # assign total fluxes
                    if ctgd > 0:
                        emlflx['ftot'][line][i, j] = \
                            q3do.line_fitpars['tflux'][line]
                        emlflxerr['ftot'][line][i, j] = \
                            q3do.line_fitpars['tfluxerr'][line]

                    # assign to output dictionary
                    emlncomp[line][i, j] = ctgd

                if thisncomp == 1:
                    isort = [0]
                elif thisncomp >= 2:
                    igd = np.arange(thisncomp)
                    sortpars = q3do.line_fitpars[compsortpar][thisncompline]
                    isort = np.argsort(sortpars[igd])
                    if compsortdir == 'down':
                        isort = np.flip(isort)
                if thisncomp > 0:
                    for line in lines_with_doublets:
                        kcomp = 1
                        for sindex in isort:
                            cstr = 'c' + str(kcomp)
                            emlwav[cstr][line][i, j] \
                                = q3do.line_fitpars['wave'][line].data[sindex]
                            emlwaverr[cstr][line][i, j] \
                                = q3do.line_fitpars['waveerr'][line].data[sindex]
                            emlsig[cstr][line][i, j] \
                                = q3do.line_fitpars['sigma'][line].data[sindex]
                            emlsigerr[cstr][line][i, j] \
                                = q3do.line_fitpars['sigmaerr'][line].data[sindex]
#                            emlweq['f' + cstr][line][i, j] \
#                                = lineweqs['comp'][line].data[sindex]
                            emlflx['f' + cstr][line][i, j] \
                                = q3do.line_fitpars['flux'][line].data[sindex]
                            emlflxerr['f' + cstr][line][i, j] \
                                = q3do.line_fitpars['fluxerr'][line].data[sindex]
                            emlflx['f' + cstr + 'pk'][line][i, j] \
                                = q3do.line_fitpars['fluxpk'][line].data[sindex]
                            emlflxerr['f' + cstr + 'pk'][line][i, j] \
                                = q3do.line_fitpars['fluxpkerr'][line].data[sindex]
                            kcomp += 1

            # Collate continuum data
            if q3do.docontfit:


                # Add wavelength to output cubes
                if firstcontfit:
                    contcube['wave'] = q3do.wave
                    firstcontfit = False

                contcube['npts'][i, j] = len(q3do.fitran_indx)

                # process continuum parameters
                q3do.sepcontpars(q3dii)

                hostcube['dat'][i, j, q3do.fitran_indx] = q3do.cont_dat
                hostcube['err'][i, j, q3do.fitran_indx] = err[q3do.fitran_indx]
                hostcube['dq'][i, j, q3do.fitran_indx] = dq[q3do.fitran_indx]
                hostcube['norm_div'][i, j, q3do.fitran_indx] \
                    = np.divide(q3do.cont_dat, q3do.cont_fit)
                hostcube['norm_sub'][i, j, q3do.fitran_indx] \
                    = np.subtract(q3do.cont_dat, q3do.cont_fit)

                if q3dii.fcncontfit == 'ppxf':
                    # Total flux from different components
                    contcube['all_mod'][i, j, q3do.fitran_indx] = q3do.cont_fit
                    contcube['stel_mod'][i, j, q3do.fitran_indx] = \
                        q3do.stelmod
                    contcube['poly_mod'][i, j, q3do.fitran_indx] = \
                        q3do.polymod
                    contcube['stel_sigma'][i, j] = q3do.ct_coeff['sigma']
                    contcube['stel_z'][i, j] = q3do.zstar
                    contcube['stel_sigma_err'][i, j, :] = \
                            np.full(2, q3do.ct_coeff['sigma_err'], dtype='float64')
                    contcube['stel_z_err'][i, j, :] = \
                            np.full(2, q3do.zstar_err, dtype='float64')
                    if q3dii.av_star is not None:
                        contcube['stel_av'][i, j] = q3do.ct_coeff['av']

                elif q3dii.fcncontfit in ['fitqsohost', 'questfit']:
                    if q3dii.fcncontfit == 'fitqsohost':
                        if 'refit' in q3dii.argscontfit:
                            if q3dii.argscontfit['refit'] == 'ppxf':
                                contcube['stel_mod'][i, j, q3do.fitran_indx] = \
                                    q3do.stelmod
                                contcube['poly_mod'][i, j, q3do.fitran_indx] = \
                                    q3do.polymod_refit
                                contcube['stel_sigma'][i, j] = \
                                    q3do.ct_coeff['ppxf']['sigma']
                                contcube['stel_z'][i, j] = q3do.zstar
                                contcube['stel_sigma_err'][i, j, :] \
                                    = np.full(2, q3do.ct_coeff['ppxf']['sigma_err'], 
                                              dtype='float64')
                                contcube['stel_z_err'][i, j, :] \
                                    = np.full(2, q3do.zstar_err, 
                                              dtype='float64')
                                if q3dii.av_star is not None:
                                    contcube['stel_av'][i, j] = \
                                        q3do.ct_coeff['ppxf']['av']

                        hostcube['dat'][i, j, q3do.fitran_indx] = \
                            q3do.cont_dat - q3do.qsomod
                        contcube['all_mod'][i, j, q3do.fitran_indx] = \
                            q3do.cont_fit
                        contcube['qso_mod'][i, j, q3do.fitran_indx] = \
                            q3do.qsomod
                        contcube['host_mod'][i, j, q3do.fitran_indx] = \
                            q3do.hostmod

                    elif q3dii.fcncontfit == 'questfit':

                        contcube['all_mod'][i, j, q3do.fitran_indx] = \
                            q3do.cont_fit
                        if q3do.zstar is not None:
                            contcube['stel_z'][i, j] = q3do.zstar
                        if q3do.zstar_err is not None:
                            contcube['stel_z_err'][i, j, :] = q3do.zstar_err

    # Save emission line and continuum dictionaries
    if q3dii.dolinefit:
        outfile = '{0.outdir}{0.label}'.format(q3dii)+'.line.npz'
        np.savez(outfile,
                 emlwav=emlwav, emlwaverr=emlwaverr,
                 emlsig=emlsig, emlsigerr=emlsigerr,
                 emlflx=emlflx, emlflxerr=emlflxerr,
                 emlweq=emlweq, emlncomp=emlncomp)
        q3dutil.write_msg(f'Saving emission-line fit results into {outfile}', q3dii.logfile, quiet)

    if q3dii.docontfit:
        outfile = '{0.outdir}{0.label}'.format(q3dii)+'.cont.npy'
        np.save(outfile, contcube)
        q3dutil.write_msg(f'Saving continuum fit results into {outfile}', q3dii.logfile, quiet)
