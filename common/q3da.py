# -*- coding: utf-8 -*-
"""
Routine to plot the continuum and emission lines fits to a spectrum.

As input, it requires a dictionary of initialization parameters and the output
dictionary, struct.npy, from Q3DF. The tags for the initialization structure
can be found in INITTAGS.txt.
----------
Returns: Nothing
----------
Parameters
----------
initproc: in, required, type=string
    Name of procedure to initialize the fit.
cols: in, optional, type=intarr, default=all
    Columns to fit, in 1-offset format. Either a scalar or a
    two-element vector listing the first and last columns to fit.
rows: in, optional, type=intarr, default=all
    Rows to fit, in 1-offset format. Either a scalar or a
    two-element vector listing the first and last rows to fit.
noplots: in, optional, type=byte
    Disable plotting.
quiet: in, optional, type=boolean
    Print error and progress messages. Propagates to most/all subroutines.

Created: 7/9/2020

@author: hadley

"""
import copy as copy
import importlib
import numpy as np
import os
import pdb

from astropy.table import Table
from ppxf.ppxf_util import log_rebin
from q3dfit.common.linelist import linelist
from q3dfit.common.readcube import CUBE
from q3dfit.common.sepfitpars import sepfitpars
# from q3dfit.common.cmpweq import cmpweq
from q3dfit.common import qsohostfcn
from scipy.special import legendre
from scipy import interpolate
# from timeit import default_timer as timer


def q3da(initproc, cols=None, rows=None, noplots=False, quiet=True):

    bad = 1.0 * 10**99

    if isinstance(initproc, str):
        from q3dfit.common.q3df_helperFunctions import __get_initdat
        initdat = __get_initdat(initproc)
    else:
        initdat = initproc

    if 'noemlinfit' not in initdat:
        # get linelist
        if 'argslinelist' in initdat:
            listlines = linelist(initdat['lines'], **initdat['argslinelist'])
        else:
            listlines = linelist(initdat['lines'])

        # table with doublets to combine
        doublets = Table.read('../data/linelists/doublets.tbl', format='ipac')
        # make a copy of singlet list
        lines_with_doublets = copy.deepcopy(initdat['lines'])
        # append doublet names to singlet list
        for (name1, name2) in zip(doublets['line1'], doublets['line2']):
            if name1 in listlines['name'] and name2 in listlines['name']:
                lines_with_doublets.append(name1+'+'+name2)

        # if 'argslinelist' in initdat:
        #     listlines_with_doublets = linelist(lines_with_doublets,
        #                                        **initdat['argslinelist'])
        # else:
        #     listlines_with_doublets = linelist(lines_with_doublets)

    if 'fcnpltcont' in initdat:
        fcnpltcont = initdat['fcnpltcont']
    else:
        #fcnpltcont = 'pltcont'
        fcnpltcont = 'plot_cont'

    # READ DATA

    if not ('datext' in initdat):
        datext = 1
    else:
        datext = initdat['datext']

    if not ('varext' in initdat):
        varext = 2
    else:
        varext = initdat['varext']

    if not ('dqext' in initdat):
        dqext = 3
    else:
        dqext = initdat['dqext']

    if not ('wmapext' in initdat):
        wmapext = 4
    else:
        wmapext = initdat['wmapext']

    header = bytes(1)

    if 'argsreadcube' in initdat:
        if initdat.__contains__('wavext'):
            cube = CUBE(infile=initdat['infile'], datext=datext, dqext=dqext,
                    quiet=quiet, varext=varext,
                    wavext=initdat['wavext'], **initdat['argsreadcube'])
        else:
            cube = CUBE(infile=initdat['infile'], quiet=quiet,
                    header=header, datext=datext, varext=varext,
                    dqext=dqext, **initdat['argsreadcube'])

    else:
        if initdat.__contains__('wavext'):
            cube = CUBE(infile=initdat['infile'], quiet=quiet,
                    header=header, datext=datext, varext=varext, wmapext=wmapext,
                    wavext=initdat['wavext'], dqext=dqext)
        else:
            cube = CUBE(infile=initdat['infile'], quiet=quiet,
                    header=header, datext=datext, varext=varext, wmapext=wmapext,
                    dqext=dqext)

    if 'vormap' in initdat:
        vormap = initdat['vormap']
        nvorcols = max(vormap)
        vorcoords = np.zeros(nvorcols, 2)
        for i in range(0, nvorcols):
            xyvor = np.where(vormap == i).nonzero()
            vorcoords[:, i] = xyvor  # TODO


# INITIALIZE OUTPUT FILES, need to write helper functions (printlinpar,
# printfitpar) later

# INITIALIZE LINE HASH
    if not('noemlinfit' in initdat):
        emlwav = dict()
        emlwaverr = dict()
        emlsig = dict()
        emlsigerr = dict()
        emlweq = dict()
        emlflx = dict()
        emlflxerr = dict()
        emlweq['ftot'] = dict()
        emlflx['ftot'] = dict()
        emlflxerr['ftot'] = dict()
        for k in range(0, initdat['maxncomp']):
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
            emlweq['ftot'][line] = np.zeros((cube.ncols, cube.nrows),
                                            dtype=float) + bad
            emlflx['ftot'][line] = np.zeros((cube.ncols, cube.nrows),
                                            dtype=float) + bad
            emlflxerr['ftot'][line] = np.zeros((cube.ncols, cube.nrows),
                                               dtype=float) + bad
            for k in range(0, initdat['maxncomp']):
                cstr = 'c' + str(k + 1)
                emlwav[cstr][line] = np.zeros((cube.ncols, cube.nrows),
                                              dtype=float) + bad
                emlwaverr[cstr][line] = np.zeros((cube.ncols, cube.nrows),
                                                 dtype=float) + bad
                emlsig[cstr][line] = np.zeros((cube.ncols, cube.nrows),
                                              dtype=float) + bad
                emlsigerr[cstr][line] = np.zeros((cube.ncols, cube.nrows),
                                                 dtype=float) + bad
                emlweq['f'+cstr][line] = np.zeros((cube.ncols, cube.nrows),
                                                  dtype=float) + bad
                emlflx['f'+cstr][line] = np.zeros((cube.ncols, cube.nrows),
                                                  dtype=float) + bad
                emlflxerr['f'+cstr][line] = np.zeros((cube.ncols, cube.nrows),
                                                     dtype=float) + bad
                emlflx['f'+cstr+'pk'][line] = \
                    np.zeros((cube.ncols, cube.nrows),
                             dtype=float) + bad
                emlflxerr['f'+cstr+'pk'][line] = \
                    np.zeros((cube.ncols, cube.nrows),
                             dtype=float) + bad
    # basically dictionaries of dictionaries of 2D arrays
    if 'flipsort' in initdat:
        flipsort = np.zeros(cube.ncols, cube.nrows)
        sizefs = len(initdat['flipsort'])
        for i in range(0, len(sizefs[0])):
            icol = initdat['flipsort'][0][i]-1
            irow = initdat['flipsort'][1][i]-1
            flipsort[icol, irow] = bytes(1)  # b

    # LOOP THROUGH SPAXELS

    # switch to track when first continuum processed
    firstcontproc = True

#   case: cols not set
    if cols is None:
        cols = np.array([1, cube.ncols], dtype=int)
#   case: cols is a scalar
    elif not isinstance(cols, (list, np.ndarray)):
        cols = np.array([cols, cols], dtype=int)
#   case: cols is a 1-element list
    elif len(cols) == 1:
        cols = np.array([cols[0], cols[0]], dtype=int)
#   case: cols is a 2-element list
    else:
        cols = np.array(cols, dtype=int)

    for i in range(cols[0] - 1, cols[1]):
        if not quiet:
            print(f'Column {i+1} of {cube.ncols}')

        if rows is None:
            rows = np.array([1, cube.nrows], dtype=int)
        elif not isinstance(rows, (list, np.ndarray)):
            rows = np.array([rows, rows], dtype=int)
        elif len(rows) == 1:
            rows = np.array([rows[0], rows[0]], dtype=int)
        else:
            rows = np.array(rows, dtype=int)

        for j in range(rows[0]-1, rows[1]):

            # set this to true if we're using Voronoi binning
            # and the tiling is missing
            novortile = False
            if cube.dat.ndim == 1:
                flux = cube.dat
                err = cube.err
                dq = cube.dq
                labin = '{[outdir]}{[label]}'.format(initdat, initdat)
                labout = labin
            elif cube.dat.ndim == 2:
                flux = cube.dat[:, i]
                err = cube.err[:, i]
                dq = cube.dq[:, i]
                labin = '{[outdir]}{[label]}_{:04d}'.\
                    format(initdat, initdat, i+1)
                labout = labin
            else:
                if not quiet:
                    print(f'    Row {j+1} of {cube.nrows}')

                if 'vormap' in initdat:
                    if np.isfinite(initdat['vormap'][i][j]) and \
                            (initdat['vormap'][i][j] is not bad):
                        iuse = vorcoords[initdat['vormap'][i][j] - 1, 0]
                        juse = vorcoords[initdat['vormap'][i][j] - 1, 1]
                    else:
                        novortile = True
                else:
                    iuse = i
                    juse = j

                if not novortile:
                    flux = cube.dat[iuse, juse, :].flatten()
                    err = cube.err[iuse, juse, :].flatten()
                    dq = cube.dq[iuse, juse, :].flatten()
                    labin = '{[outdir]}{[label]}_{:04d}_{:04d}'.\
                        format(initdat, initdat, iuse+1, juse+1)
                    labout = '{[outdir]}{[label]}_{:04d}_{:04d}'.\
                        format(initdat, initdat, i+1, j+1)

            # Restore fit after a couple of sanity checks
            # these sanity checks are wearing down my sanity
            if not novortile:
                infile = labin + '.npy'
                outfile = labout
                nodata = flux.nonzero()
                ct = len(nodata[0])
                filepresent = os.path.isfile(infile)  # check file
            else:
                # missing Voronoi bin for this spaxel
                filepresent = False
                ct = 0

            if not filepresent or ct == 0:

                badmessage = f'        No data for [{i+1}, {j+1}]'
                print(badmessage)

            else:
                struct = (np.load(infile, allow_pickle='TRUE')).item()

                # Restore original error.
                struct['spec_err'] = err[struct['fitran_indx']]

                if not struct['noemlinfit']:
                    # get line fit params
                    linepars, tflux = \
                        sepfitpars(listlines, struct['param'],
                                   struct['perror'],
                                   initdat['maxncomp'], tflux=True,
                                   doublets=doublets)
#                lineweqs = cmpweq(struct, listlines, doublets = emldoublets)

                # plot emission line data, print data to a file
                if not noplots:

                    # plot emission lines
                    if not struct['noemlinfit']:
                        if 'nolines' not in linepars:
                            if 'fcnpltlin' in initdat:
                                fcnpltlin = initdat['fcnpltlin']
                            else:
                                fcnpltlin = 'pltlin'
                            module = \
                                importlib.import_module('q3dfit.common.' +
                                                        fcnpltlin)
                            pltlinfcn = getattr(module, fcnpltlin)
                            if 'argspltlin1' in initdat:
                                pltlinfcn(struct, initdat['argspltlin1'],
                                          outfile + '_lin1')
                            if 'argspltlin2' in initdat:
                                pltlinfcn(struct, initdat['argspltlin2'],
                                          outfile + '_lin2')

                # Possibly add later: print fit parameters to a text file

                if not struct['noemlinfit']:
                    # get correct number of components in this spaxel
                    thisncomp = 0
                    thisncompline = ''

                    for line in lines_with_doublets:
                        sigtmp = linepars['sigma'][line]
                        fluxtmp = linepars['flux'][line]
                        # TODO
                        igd = [idx for idx in range(len(sigtmp)) if
                               (sigtmp[idx] != 0 and
                                sigtmp[idx] != bad and
                                fluxtmp[idx] != 0 and
                                fluxtmp[idx] != bad)]
                        ctgd = len(igd)

                        if ctgd > thisncomp:
                            thisncomp = ctgd
                            thisncompline = line

                        # assign total fluxes
                        if ctgd > 0:
    #                        emlweq['ftot', line, j, i] = lineweqs['tot'][line]
                            emlflx['ftot'][line][i, j] = tflux['tflux'][line]
                            emlflxerr['ftot'][line][i, j] = tflux['tfluxerr'][line]

                    if thisncomp == 1:
                        isort = [0]
                        if 'flipsort' in initdat:
                            if flipsort[i, j]:
                                print('Flipsort set for spaxel [' + str(i+1)
                                      + ',' + str(j + 1) + '] but ' +
                                      'only 1 component. Setting to 2 components' +
                                      ' and flipping anyway.')
                                isort = [0, 1]  # flipped
                    elif thisncomp >= 2:
                        # sort components
                        igd = np.arange(thisncomp)
                        # indices = np.arange(initdat['maxncomp'])
                        sigtmp = linepars['sigma'][thisncompline]
                        fluxtmp = linepars['flux'][thisncompline]
                        if 'sorttype' not in initdat:
                            isort = np.argsort(sigtmp[igd])
                        elif initdat['sorttype'] == 'wave':
                            isort = np.argsort(linepars['wave'][line, igd])
                        elif initdat['sorttype'] == 'reversewave':
                            isort = np.argsort(linepars['wave'][line, igd])[::-1]

                        if 'flipsort' in initdat:
                            if flipsort[i, j] is not None:
                                isort = isort[::-1]
                    if thisncomp > 0:
                        for line in lines_with_doublets:
                            kcomp = 1
                            for sindex in isort:
                                cstr = 'c' + str(kcomp)
                                emlwav[cstr][line][i, j] \
                                    = linepars['wave'][line].data[sindex]
                                emlwaverr[cstr][line][i, j] \
                                    = linepars['waveerr'][line].data[sindex]
                                emlsig[cstr][line][i, j] \
                                    = linepars['sigma'][line].data[sindex]
                                emlsigerr[cstr][line][i, j] \
                                    = linepars['sigmaerr'][line].data[sindex]
    #                            emlweq['f' + cstr][line][i, j] \
    #                                = lineweqs['comp'][line].data[sindex]
                                emlflx['f' + cstr][line][i, j] \
                                    = linepars['flux'][line].data[sindex]
                                emlflxerr['f' + cstr][line][i, j] \
                                    = linepars['fluxerr'][line].data[sindex]
                                emlflx['f' + cstr + 'pk'][line][i, j] \
                                    = linepars['fluxpk'][line].data[sindex]
                                emlflxerr['f' + cstr + 'pk'][line][i, j] \
                                    = linepars['fluxpkerr'][line].data[sindex]
                                kcomp += 1
                        # Possibly do later, print line fluxes to text file
                        # printlinpar, ~line 474

    # Process and plot continuum data
                # make and populate output data cubes
                if firstcontproc is True:
                    hostcube = \
                       {'dat': np.zeros((cube.ncols, cube.nrows, cube.nw)),
                        'err': np.zeros((cube.ncols, cube.nrows, cube.nw)),
                        'dq':  np.zeros((cube.ncols, cube.nrows, cube.nw)),
                        'norm_div': np.zeros((cube.ncols, cube.nrows, cube.nw)),
                        'norm_sub': np.zeros((cube.ncols, cube.nrows, cube.nw))}

                    if 'decompose_ppxf_fit' in initdat:
                        contcube = \
                            {'wave': struct['wave'],
                             'all_mod': np.zeros((cube.ncols, cube.nrows, cube.nw)),
                             'stel_mod': np.zeros((cube.ncols, cube.nrows, cube.nw)),
                             'poly_mod': np.zeros((cube.ncols, cube.nrows, cube.nw)),
                             'stel_mod_tot': np.zeros((cube.ncols, cube.nrows))
                             + bad,
                             'poly_mod_tot': np.zeros((cube.ncols, cube.nrows))
                             + bad,
                             'poly_mod_tot_pct': np.zeros((cube.ncols, cube.nrows))
                             + bad,
                             'stel_sigma': np.zeros((cube.ncols, cube.nrows))
                             + bad,
                             'stel_sigma_err': np.zeros((cube.ncols, cube.nrows, 2))
                             + bad,
                             'stel_z': np.zeros((cube.ncols, cube.nrows)) + bad,
                             'stel_z_err': np.zeros((cube.ncols, cube.nrows, 2))
                             + bad,
                             'stel_rchisq': np.zeros((cube.ncols, cube.nrows))
                             + bad,
                             'stel_ebv': np.zeros((cube.ncols, cube.nrows))
                             + bad,
                             'stel_ebv_err': np.zeros((cube.ncols, cube.nrows, 2))
                             + bad}

                    elif 'decompose_qso_fit' in initdat:
                        contcube = \
                            {'wave': struct['wave'],
                             'qso_mod':
                                 np.zeros((cube.ncols, cube.nrows, cube.nw)),
                             'qso_poly_mod':
                                 np.zeros((cube.ncols, cube.nrows, cube.nw)),
                             'host_mod':
                                 np.zeros((cube.ncols, cube.nrows, cube.nw)),
                             'poly_mod':
                                 np.zeros((cube.ncols, cube.nrows, cube.nw)),
                             'npts':
                                 np.zeros((cube.ncols, cube.nrows)) + bad,
                             'stel_sigma':
                                 np.zeros((cube.ncols, cube.nrows)) + bad,
                             'stel_sigma_err':
                                 np.zeros((cube.ncols, cube.nrows, 2)) + bad,
                             'stel_z':
                                 np.zeros((cube.ncols, cube.nrows)) + bad,
                             'stel_z_err':
                                 np.zeros((cube.ncols, cube.nrows, 2)) + bad,
                             'stel_rchisq':
                                 np.zeros((cube.ncols, cube.nrows)) + bad,
                             'stel_ebv':
                                 np.zeros((cube.ncols, cube.nrows)) + bad,
                             'stel_ebv_err':
                                 np.zeros((cube.ncols, cube.nrows, 2)) + bad}
                    else:
                        contcube = \
                            {'all_mod':
                             np.zeros((cube.ncols, cube.nrows, cube.nw)),
                             'stel_z':
                                 np.zeros((cube.ncols, cube.nrows)) + bad,
                             'stel_z_err':
                                 np.zeros((cube.ncols, cube.nrows, 2)) + bad,
                             'stel_rchisq':
                                 np.zeros((cube.ncols, cube.nrows)) + bad,
                             'stel_ebv':
                                 np.zeros((cube.ncols, cube.nrows)) + bad,
                             'stel_ebv_err':
                                 np.zeros((cube.ncols, cube.nrows, 2)) + bad}
                    firstcontproc = False

                hostcube['dat'][i, j, struct['fitran_indx']] = struct['cont_dat']
                hostcube['err'][i, j, struct['fitran_indx']] = \
                    err[struct['fitran_indx']]
                hostcube['dq'][i, j, struct['fitran_indx']] = \
                    dq[struct['fitran_indx']]
                hostcube['norm_div'][i, j, struct['fitran_indx']] \
                    = np.divide(struct['cont_dat'], struct['cont_fit'])
                hostcube['norm_sub'][i, j, struct['fitran_indx']] \
                    = np.subtract(struct['cont_dat'], struct['cont_fit'])

                if 'decompose_ppxf_fit' in initdat:
                    add_poly_degree = 4  # should match fitspec
                    if 'argscontfit' in initdat:
                        if 'add_poly_degree' in initdat['argscontfit']:
                            add_poly_degree = \
                                initdat['argscontfit']['add_poly_degree']
                    # Compute polynomial
                    dumy_log, wave_log,_ = \
                        log_rebin([struct['wave'][0],
                                   struct['wave'][len(struct['wave'])-1]],
                                  struct['spec'])
                    xnorm = cap_range(-1.0, 1.0, len(wave_log))
                    cont_fit_poly_log = 0.0
                    for k in range(0, add_poly_degree):
                        cfpllegfun = legendre(k)
                        cont_fit_poly_log += \
                            cfpllegfun(xnorm) * struct['ct_add_poly_weights'][k]
                    interpfunction = \
                        interpolate.interp1d(cont_fit_poly_log, wave_log,
                                             kind='linear', fill_value="extrapolate")
                    cont_fit_poly = interpfunction(np.log(struct['wave']))
                    # Compute stellar continuum
                    cont_fit_stel = np.subtract(struct['cont_fit'], cont_fit_poly)
                    # Total flux fromd ifferent components
                    cont_fit_tot = np.sum(struct['cont_fit'])
                    contcube['all_mod'][i, j, struct['fitran_indx']] = \
                        struct['cont_fit']
                    contcube['stel_mod'][i, j, struct['fitran_indx']] = \
                        cont_fit_stel
                    contcube['poly_mod'][i, j, struct['fitran_indx']] = \
                        cont_fit_poly
                    contcube['stel_mod_tot'][i, j] = np.sum(cont_fit_stel)
                    contcube['poly_mod_tot'][i, j] = np.sum(cont_fit_poly)
                    contcube['poly_mod_tot_pct'][i, j] \
                        = np.divide(contcube['poly_mod_tot'][i, j], cont_fit_tot)
                    contcube['stel_sigma'][i, j] = struct['ct_ppxf_sigma']
                    contcube['stel_z'][i, j] = struct['zstar']

                    if 'ct_errors' in struct:
                        contcube['stel_sigma_err'][i, j, :] \
                            = struct['ct_errors']['ct_ppxf_sigma']
                    # assuming that ct_errors is a dictionary
                    else:  # makes an array with two arrays
                        contcube['stel_sigma_err'][i, j, :] \
                            = [struct['ct_ppxf_sigma_err'],
                               struct['ct_ppxf_sigma_err']]

                    if 'ct_errors' in struct:
                        contcube['stel_z_err'][i, j, :] = \
                            struct['ct_errors']['zstar']
                    else:
                        contcube['stel_z_err'][i, j, :] \
                            = [struct['zstar_err'], struct['zstar_err']]

                elif 'decompose_qso_fit' in initdat:
                    if initdat['fcncontfit'] == 'fitqsohost':
                        if 'qsoord' in initdat['argscontfit']:
                            qsoord = initdat['argscontfit']['qsoord']
                        else:
                            qsoord = None  # ?

                        if 'hostord' in initdat['argscontfit']:
                            hostord = initdat['argscontfit']['hostord']
                        else:
                            hostord = None  # ?

                        if 'blrpar' in initdat['argscontfit']:
                            blrterms = len(initdat['argscontfit']['blrpar'])
                            # blrpar a 1D array
                        else:
                            blrterms = 0  # ?
                        # default here must be same as in IFSF_FITQSOHOST
                        if 'add_poly_degree' in initdat['argscontfit']:
                            add_poly_degree = \
                                initdat['argscontfit']['add_poly_degree']
                        else:
                            add_poly_degree = 0#30

                        # These lines mirror ones in IFSF_FITQSOHOST
                        struct_tmp = struct

                        # Get and renormalize template
                        qsotemplate = \
                            np.load(initdat['argscontfit']['qsoxdr'],
                                    allow_pickle='TRUE').item()
                        try:
                            qsowave = qsotemplate['wave']
                            qsoflux_full = qsotemplate['flux']
                        except:
                            qsotemplate = \
                                np.load(initdat['argscontfit']['qsoxdr'],
                                    allow_pickle='TRUE')
                            qsowave = qsotemplate['wave'][0]
                            qsoflux_full = qsotemplate['flux'][0]

                        # non zero could be uncessesary
    #                    iqsoflux = \
    #                        np.flatnonzero(np.where((
    #                            qsowave > struct_tmp['fitran'][0]*0.99999) & (
    #                                qsowave < struct_tmp['fitran'][1]*1.00001)))
                        iqsoflux = np.where((qsowave >= struct_tmp['fitran'][0]) & (qsowave <= struct_tmp['fitran'][1]))
                        # line 611
                        qsoflux = qsoflux_full[iqsoflux]
                        qsoflux /= np.median(qsoflux)
                        struct = struct_tmp
                        #If polynomial residual is re-fit with PPXF, separate out best-fit
                        #parameter structure created in IFSF_FITQSOHOST and compute polynomial
                        #and stellar components
                        if 'refit' in initdat['argscontfit'] and 'args_questfit' not in initdat['argscontfit']:
                            par_qsohost = struct['ct_coeff']['qso_host']
                            par_stel = struct['ct_coeff']['stel']
                            dumy_log, wave_rebin,_ = log_rebin([struct['wave'][0],
                                struct['wave'][len(struct['wave'])-1]],
                                struct['spec'])
                            xnorm = cap_range(-1.0, 1.0, len(wave_rebin)) #1D?
                            if add_poly_degree > 0:
                                par_poly = struct['ct_coeff']#['poly']
                                polymod_log = 0.0 # Additive polynomial
                                for k in range(0, add_poly_degree):
                                    cfpllegfun = legendre(k)
                                    polymod_log += cfpllegfun(xnorm) * par_poly[k]
                                interpfunct = interpolate.interp1d(wave_rebin, polymod_log, kind='cubic',fill_value="extrapolate")
                                polymod_refit = interpfunct(np.log(struct['wave']))
                            else:
                                polymod_refit = np.zeros(len(struct['wave']), dtype=float)
                            contcube['stel_sigma'][i, j] = struct['ct_coeff']#['ppxf_sigma']
                            contcube['stel_z'][i, j] = struct['zstar']

                            #Don't know ct_error's type
                            if 'ct_errors' in struct:
                                contcube['stel_sigma_err'][i, j, :] \
                                    = struct['ct_errors']['ct_ppxf_sigma']
                            else:
                                contcube['stel_sigma_err'][i, j, :] \
                                    = [struct['ct_ppxf_sigma_err'], struct['ct_ppxf_sigma_err']]
                            if 'ct_errors' in struct:
                                contcube['stel_z_err'][i, j, :] \
                                    = struct['ct_errors']['zstar']
                            else:
                                contcube['stel_z_err'][i, j, :] \
                                    = [struct['zstar_err'], struct['zstar_err']]
                            #again why aren't those two if statements combined
                        elif 'refit' in initdat['argscontfit'] and initdat['argscontfit']['refit']=='questfit': # Refitting with questfit in the MIR
                            par_qsohost = struct['ct_coeff']['qso_host']
                            dumy_log, wave_rebin,_ = log_rebin([struct['wave'][0],
                                struct['wave'][len(struct['wave'])-1]],
                                struct['spec'])
                            xnorm = cap_range(-1.0, 1.0, len(wave_rebin)) #1D?
                            polymod_refit = np.zeros(len(struct['wave']), dtype=float)  # Double-check

                        else:
                            par_qsohost = struct['ct_coeff']
                            polymod_refit = 0.0

                        #produce fit with template only and with template + host. Also
                        #output QSO multiplicative polynomial
                        qsomod_polynorm = 0.0
    #                    qsomod = qsohostfcn.qsohostfcn(struct['wave'], params_fit=par_qsohost, qsoflux = qsoflux,
    #                                      blrpar = initdat['argscontfit']['blrpar'],qsoonly=True,hostonly=True,qsoord=qsoord,hostord=hostord)

                        qsomod = qsohostfcn.qsohostfcn(struct['wave'], params_fit=par_qsohost, qsoflux = qsoflux,
                                          qsoonly=True, blrterms = blrterms,
                                          qsoscl = qsomod_polynorm, qsoord = qsoord,
                                          hostord = hostord)

    #                    qsomod = qsohostfcn.qsohostfcn(struct['wave'], params_fit=par_qsohost, qsoflux = qsoflux
    #                                                   ,blrterms = blrterms)

                        hostmod = struct['cont_fit_pretweak'] - qsomod

                        #if continuum is tweaked in any region, subide resulting residual
                        #proportionality @ each wavelength btwn qso and host components
                        qsomod_notweak = qsomod
                        if 'tweakcntfit' in initdat:
                            modresid = struct['cont_fit'] - struct['cont_fit_pretweak']
                            inz = np.where((qsomod != 0) & (hostmod != 0))[0]
                            qsofrac = np.zeros(len(qsomod))
                            for ind in inz:
                                qsofrac[ind] = qsomod[ind] / (qsomod[ind] + hostmod[ind])
                            qsomod += modresid * qsofrac
                            hostmod += modresid * (1.0 - qsofrac)
                        #components of qso fit for plotting
                        qsomod_normonly = qsoflux
                        if 'blrpar' in initdat['argscontfit']:
                            qsomod_blronly = qsohostfcn.qsohostfcn(struct['wave'], par_qsohost,
                                             qsoflux = qsoflux, blronly=True,
                                             blrpar = initdat['argscontfit']['blrpar'], qsoord = qsoord,
                                             hostord = hostord)
                    elif  initdat['fcncontfit'] == 'questfit':      # CB: adding option to plot decomposed QSO fit if questfit is used
                        from q3dfit.common.questfit import quest_extract_QSO_contrib
                        qsomod, hostmod, qsomod_intr, hostmod_intr = quest_extract_QSO_contrib(struct['ct_coeff'], initdat)
                        qsomod_polynorm = 1.
                        qsomod_notweak = qsomod
                        qsoflux = qsomod.copy()/np.median(qsomod)
                        qsomod_normonly = qsoflux
                        polymod_refit = 0.



                elif initdat['fcncontfit'] == 'ppxf' and 'qsotempfile' in initdat:
                    qsotempfile = np.load(initdat['qsotempfile'], allow_pickle='TRUE').item()
                    struct_qso = qsotempfile
                    qsomod = struct_qso['cont_fit'] * struct['ct_coeff'][len(struct['ct_coeff']) - 1]
                    hostmod = struct['cont_fit'] - qsomod
                elif initdat['fcncontfit'] == 'questfit':
                #else:
                    contcube['all_mod'][i, j, struct['fitran_indx']] = struct['cont_fit']
                    contcube['stel_z'][i, j] = struct['zstar']
                    if 'ct_errors' in struct:
                        contcube['stel_z_err'][i, j, :] = struct['ct_errors']['zstar']
                    else:
                        contcube['stel_z_err'][i, j, :] = [0, 0]

                contcube['stel_ebv'][i, j] = struct['ct_ebv']
                if 'ct_errors' in struct:
                    contcube['stel_ebv_err'][i, j,:]=struct['ct_errors']['ct_ebv']
                else:
                    contcube['stel_rchisq'][i, j]=0.0

                # Print ppxf results to stdout
                if ('decompose_ppxf_fit' in initdat) or \
                        ('decompose_qso_fit' in initdat):
                    if 'argscontfit' in initdat:
                        if 'print_output' in initdat['argscontfit']:
                            print("PPXF results: ")
                            if 'decompose_ppxf_fit' in initdat:
                                ct_coeff_tmp = struct['ct_coeff']
                                poly_tmp_pct = contcube['poly_mod_tot_pct'][i, j]
                            else:
                                ct_coeff_tmp = struct['ct_coeff']['stel']
                                poly_tmp_pct = \
                                    np.sum(polymod_refit) / np.sum(hostmod)
                            inz = np.where(ct_coeff_tmp != 0.0)
                            ctnz = len(inz)
                            if ctnz > 0:
                                coeffgd = ct_coeff_tmp[inz]
                                # normalize coefficients to % of total stellar coeffs.
                                totcoeffgd = np.sum(coeffgd)
                                coeffgd /= totcoeffgd
                                # re-normalize to % of total flux
                                coeffgd *= (1.0 - poly_tmp_pct)
                                # TODO: xdr file
                                startempfile = \
                                    np.load(initdat['startempfile']+".npy",
                                            allow_pickle='TRUE').item()
                                agesgd = startempfile['ages'][inz]  # check
                                # sum coefficients over age ranges
                                iyoung = np.where(agesgd < 1e7)
                                ctyoung = len(iyoung)
                                iinter1 = np.where(agesgd > 1e7 and agesgd < 1e8)
                                ctinter1 = len(iinter1)
                                iinter2 = np.where(agesgd > 1e8 and agesgd < 1e9)
                                ctinter2 = len(iinter2)
                                iold = np.where(agesgd > 1e9)
                                ctold = len(iold)
                                if ctyoung > 0:
                                    coeffyoung = np.sum(coeffgd[iyoung]) * 100.0
                                else:
                                    coeffyoung = 0.0
                                if ctinter1 > 0:
                                    coeffinter1 = np.sum(coeffgd[iinter1]) * 100.0
                                else:
                                    coeffinter1 = 0.0
                                if ctinter2 > 0:
                                    coeffinter2 = np.sum(coeffgd[iinter2]) * 100.0
                                else:
                                    coeffinter2 = 0.0
                                if ctold > 0:
                                    coeffold = np.sum(coeffgd[iold]) * 100.0
                                else:
                                    coeffold = 0.0
                                print(str(round(coeffyoung)) +
                                      ' contribution from ages <= 10 Myr.')
                                print(str(round(coeffinter1)) +
                                      ' contribution from 10 Myr < age <= 100 Myr.')
                                print(str(round(coeffinter2)) +
                                      ' contribution from 100 Myr < age <= 1 Gyr.')
                                print(str(round(coeffold)) +
                                      ' contribution from ages > 1 Gyr.')
                                print(' Stellar template convolved with sigma = ' +
                                      str(struct['ct_ppxf_sigma']) + 'km/s')

    #            Plot QSO and host only continuum fit
                if 'decompose_qso_fit' in initdat:

                    struct_host = copy.deepcopy(struct)
                    struct_qso = copy.deepcopy(struct_host)

                    struct_host['spec'] -= qsomod
                    struct_host['cont_dat'] -= qsomod
                    struct_host['cont_fit'] -= qsomod

                    struct_qso['spec'] -= hostmod
                    struct_qso['cont_dat'] -= hostmod
                    struct_qso['cont_fit'] -= hostmod
                    contcube['qso_mod'][i, j, struct['fitran_indx']] = qsomod
                    contcube['qso_poly_mod'][i, j, struct['fitran_indx']] = \
                        qsomod_polynorm
                    contcube['host_mod'][i, j, struct['fitran_indx']] = hostmod
                    contcube['poly_mod'][i, j, struct['fitran_indx']] = \
                        polymod_refit
                    contcube['npts'][i, j] = len(struct['fitran_indx'])
                    if 'remove_scattered' in initdat:
                        contcube['host_mod'][i, j, struct['fitran_indx']] -= \
                            polymod_refit
                    # Update hostcube.dat to remove tweakcnt mods
                    # Data minus (emission line model + QSO model,
                    # tweakcnt mods not included in QSO model)
                    hostcube['dat'][i, j, struct['fitran_indx']] \
                        = struct['cont_dat'] - qsomod_notweak

                    if not noplots and np.sum(struct_host['cont_fit']) != 0.0:
                        if 'refit' in initdat['argscontfit']:
                            compspec = np.array([polymod_refit,
                                                 hostmod-polymod_refit])
                            compfit = ['ord. ' + str(add_poly_degree) +
                                       ' Leg. poly.', 'stel. temp.']
                        else:
                            compspec = hostmod
                            if initdat['fcncontfit'] == 'questfit':     ##  CB: Work-around - think about this more later
                                compspec = [hostmod]
                            compfit = ['exponential terms']
                        module = importlib.import_module('q3dfit.common.' +
                                                         fcnpltcont)
                        pltcontfcn = getattr(module, fcnpltcont)
                        if 'argspltcont' in initdat:
                            pltcontfcn(struct_host, outfile + '_cnt_host',
                                       compspec=compspec, compfit=compfit,
                                       title='Host', fitran=initdat['fitran'],
                                       **initdat['argspltcont'],
                                       initdat=initdat)
                        else:
                            pltcontfcn(struct_host, outfile + '_cnt_host',
                                       compspec=[compspec],
                                       title='Host', fitran=initdat['fitran'],
                                       initdat=initdat)
                        if 'blrpar' in initdat['argscontfit']:
                            qsomod_blrnorm = np.median(qsomod) / \
                                max(qsomod_blronly)
                            compspec = np.array([qsomod_normonly,
                                                 qsomod_blronly *
                                                 qsomod_blrnorm])
                            compfit = ['raw template', 'scattered\times' +
                                       str(qsomod_blrnorm)]
                        else:
                            compspec = [[qsomod_normonly]]
                            compfit = ['raw template']
                            if initdat['fcncontfit'] == 'questfit':     ##  CB: Work-around - think about this more later
                                compspec = [qsomod_normonly]

                        if 'argspltcont' in initdat:
                            pltcontfcn(struct_qso, str(outfile) + '_cnt_qso',
                                       compspec=compspec, compfit=compfit,
                                       title='QSO', fitran=initdat['fitran'],
                                       **initdat['argspltcont'],
                                       initdat=initdat)
                        else:
                            pltcontfcn(struct_qso, outfile + '_cnt_qso',
                                       compspec=compspec, comptitles=compfit,
                                       title='QSO', fitran=initdat['fitran'],
                                       initdat=initdat)
                # Plot continuum
                # Make sure fit doesn't indicate no continuum; avoids
                # plot range error in continuum fitting routine,
                # as well as a blank plot!
                if not noplots and sum(struct['cont_fit']) != 0.0:

                    module = importlib.import_module('q3dfit.common.' +
                                                     fcnpltcont)
                    pltcontfcn = getattr(module, fcnpltcont)
                    if 'decompose_qso_fit' in initdat:
                        if 'argspltcont' in initdat:
                            pltcontfcn(struct, outfile + '_cnt',
                                       compspec=np.array([qsomod, hostmod]),
                                       title='Total',
                                       comptitles=['QSO', 'host'],
                                       fitran=initdat.fitran,
                                       **initdat['argspltcont'],
                                       initdat=initdat)
                        else:
                            pltcontfcn(struct, outfile + '_cnt',
                                       compspec=np.array([qsomod, hostmod]),
                                       title='Total',
                                       comptitles=['QSO', 'host'],
                                       fitran=initdat['fitran'],
                                       initdat=initdat)

                        if 'compare_to_real_decomp' in initdat:     # CB: in the case of the MIR mock ETC cube, compare the recovered QSO/host contribution from the combined cube to the real ones from the QSO/host only simulations
                            if initdat['compare_to_real_decomp']['on']:
                                from q3dfit.common import readcube
                                argsreadcube_dict = {'fluxunit_in': 'Jy',
                                                    'waveunit_in': 'angstrom',
                                                    'waveunit_out': 'micron'} 
                                file_host = initdat['compare_to_real_decomp']['file_host']
                                file_qso = initdat['compare_to_real_decomp']['file_qso']

                                if 'argsreadcube' in initdat:
                                    if initdat.__contains__('wavext'):
                                        cube2 = CUBE(infile=file_host, datext=datext, dqext=dqext,
                                                quiet=quiet, varext=varext,
                                                wavext=initdat['wavext'], **initdat['argsreadcube'])
                                        cube3 = CUBE(infile=file_qso, datext=datext, dqext=dqext,
                                                quiet=quiet, varext=varext,
                                                wavext=initdat['wavext'], **initdat['argsreadcube'])
                                    else:
                                        cube2 = CUBE(infile=file_host, quiet=quiet,
                                                header=header, datext=datext, varext=varext,
                                                dqext=dqext, **initdat['argsreadcube'])
                                        cube3 = CUBE(infile=file_qso, quiet=quiet,
                                                header=header, datext=datext, varext=varext,
                                                dqext=dqext, **initdat['argsreadcube'])
                                else:
                                    if initdat.__contains__('wavext'):
                                        cube2 = CUBE(infile=file_host, quiet=quiet, 
                                                header=header, datext=datext, varext=varext,
                                                wavext=initdat['wavext'], dqext=dqext)
                                        cube3 = CUBE(infile=file_qso, quiet=quiet, 
                                                header=header, datext=datext, varext=varext,
                                                wavext=initdat['wavext'], dqext=dqext)
                                    else:
                                        cube2 = CUBE(infile=file_host, quiet=quiet,
                                                header=header, datext=datext, varext=varext,
                                                dqext=dqext)
                                        cube3 = CUBE(infile=file_qso, quiet=quiet,
                                                header=header, datext=datext, varext=varext,
                                                dqext=dqext)


                                lam_exclude = sorted(set(cube2.wave.tolist()) - set(struct['wave'].tolist())) # exclude wavelength that are in cube2.wave but not in struct['wave']
                                okwave = np.ones(len(cube2.wave)).astype(bool)
                                for i,lam_i in enumerate(cube2.wave):
                                    if lam_i in lam_exclude:
                                        okwave[i] = False

                                # from scipy import constants
                                # from astropy import units as u
                                # c_scale =  constants.c * u.Unit('m').to('micron') /(cube2.wave[okwave])**2 *1e-23  *1e10      # [1e-10 erg/s/cm^2/um/sr]]

                                hostspec_real = cube2.dat[iuse, juse, :].flatten()[okwave] # * c_scale
                                qsospec_real = cube3.dat[iuse, juse, :].flatten()[okwave] # * c_scale


                                struct_overpredict = struct.copy()
                                struct_overpredict['cont_dat'] = 1.*struct['cont_fit']/struct['cont_dat']
                                struct_overpredict['cont_fit'] = 1.*struct['cont_fit']/struct['cont_dat']

                                print('Check: host-only cube + QSO-only cube = combined-cube spectrum?: ', (qsospec_real + hostspec_real)/struct['cont_dat'], '\n(check only works if we are actually running on the combined cube)' )

                                if 'argspltcont' in initdat:
                                    pltcontfcn(struct, outfile + '_cnt_decomp',
                                               #compspec=np.array([qsomod, hostmod, qsomod_intr, hostmod_intr, qsospec_real, hostspec_real]),
                                               compspec=np.array([qsomod, hostmod, qsospec_real, hostspec_real]),
                                               title='Total',
                                               #comptitles=['QSO model ext. & abs.', 'host model ext. & abs.', 'QSO model intrinsic', 'host model intrinsic', 'QSO intrinsic real', 'host intrinsic real'],
                                               comptitles=['QSO model', 'host model', 'QSO real', 'host real'],
                                               #compcols=['lightgrey', 'lightgrey' , 'c', 'plum', 'mediumblue', 'darkviolet'],
                                               compcols=['c', 'plum', 'mediumblue', 'darkviolet'],
                                               fitran=initdat.fitran,
                                               **initdat['argspltcont'],

                                               initdat=initdat)
                                    pltcontfcn(struct_overpredict, outfile + '_cnt_overpredict',
                                               compspec=np.array([1.*qsomod/qsospec_real, 1.*hostmod/hostspec_real]),
                                               title='Total',
                                               comptitles=['QSO_model / QSO_real = {:.3f}'.format(np.median(1.*qsomod/qsospec_real)), '<host_model / host_real> = {:.3f}'.format(np.median(1.*hostmod/hostspec_real))],

                                               compcols=['c', 'plum'],
                                               fitran=initdat.fitran,
                                               **initdat['argspltcont'],
                                               initdat=initdat)

                                else:
                                    pltcontfcn(struct, outfile + '_cnt_decomp',
                                               compspec=np.array([qsomod, hostmod, qsospec_real, hostspec_real]),
                                               title='Total',
                                               comptitles=['QSO model', 'host model', 'QSO real', 'host real'],
                                               #compcols=['paleturquoise', 'thistle' , 'c', 'plum', 'mediumblue', 'darkviolet'],
                                               compcols=['c', 'plum', 'mediumblue', 'darkviolet'],
                                               fitran=initdat['fitran'],
                                               initdat=initdat)
                                    pltcontfcn(struct_overpredict, outfile + '_cnt_overpredict',
                                               compspec=np.array([1.*qsomod/qsospec_real, 1.*hostmod/hostspec_real]),
                                               title='Total',
                                               #comptitles=['QSO_model / QSO_real', 'host_model / host_real'],
                                               comptitles=['QSO_model / QSO_real = {:.3f}'.format(np.median(1.*qsomod/qsospec_real)), '<host_model / host_real> = {:.3f}'.format(np.median(1.*hostmod/hostspec_real))],
                                               compcols=['c', 'plum'],
                                               fitran=initdat['fitran'],
                                               initdat=initdat)




                    elif 'decompose_ppxf_fit' in initdat:
                        if 'argspltcont' in initdat:
                            pltcontfcn(struct, outfile + '_cnt',
                                       compspec=np.array([cont_fit_stel,
                                                          cont_fit_poly]),
                                       title='Total',
                                       comptitless=['stel. temp.', 'ord. ' +
                                                    str(add_poly_degree) +
                                                    'Leg.poly'],
                                       fitran=initdat['fitran'],
                                       **initdat['argspltcont'],
                                       initdat=initdat)
                        else:
                            pltcontfcn(struct, outfile + '_cnt',
                                       compspec=np.array([cont_fit_stel,
                                                          cont_fit_poly]),
                                       title='Total',
                                       comptitles=['stel. temp.', 'ord. ' +
                                                   str(add_poly_degree) +
                                                   ' Leg. poly'],
                                       fitran=initdat['fitran'],
                                       initdat=initdat)
                    else:
                        if 'argspltcont' in initdat:
                            pltcontfcn(struct, outfile + '_cnt',
                                       fitran=initdat['fitran'],
                                       initdat=initdat,
                                       ct_coeff=struct['ct_coeff'],
                                       title='Total', **initdat['argspltcont'])
                        else:
                            pltcontfcn(struct, outfile + '_cnt',
                                       fitran=initdat['fitran'],
                                       initdat=initdat,
                                       ct_coeff=struct['ct_coeff'],
                                       title='Total')

    # Save emission line and continuum dictionaries
    np.savez('{[outdir]}{[label]}'.format(initdat, initdat)+'.lin.npz',
             emlwav=emlwav, emlwaverr=emlwaverr,
             emlsig=emlsig, emlsigerr=emlsigerr,
             emlflx=emlflx, emlflxerr=emlflxerr,
             emlweq=emlweq)
    np.save('{[outdir]}{[label]}'.format(initdat, initdat)+'.cont.npy',
            contcube)


def cap_range(x1, x2, n):
    a = np.zeros(1, dtype=float)
    interval = (x2 - x1) / (n - 1)
    #    print(interval)
    num = x1
    for i in range(0, n):
        a = np.append(a, num)
        num += interval
    a = a[1:]
    return a


def array_indices(array, index):
    height = len(array[0])
    x = index // height
    y = index % height
    return x, y
