# -*- coding: utf-8 -*-
import copy as copy
import importlib
import importlib.resources as pkg_resources
import matplotlib as mpl
import numpy as np
import os
from q3dfit.data import linelists

from astropy.table import Table
from ppxf.ppxf_util import log_rebin
from q3dfit.readcube import Cube
from q3dfit.sepfitpars import sepfitpars
from q3dfit.q3df_helperFunctions import __get_Cube, __get_linelist, \
    __get_q3di, __get_spaxels
from q3dfit.qsohostfcn import qsohostfcn
from numpy.polynomial import legendre
from scipy.interpolate import interp1d


def q3da(initobj, cols=None, rows=None, noplots=False, quiet=True,
         inline=True):
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
initobj: in, required, type=string
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
    bad = 1e99

    if inline is False:
        mpl.use('agg')

    q3di = __get_q3di(initobj)
    if q3di.dolinefit:

        linelist = __get_linelist(q3di)

        # table with doublets to combine
        with pkg_resources.path(linelists, 'doublets.tbl') as p:
            doublets = Table.read(p, format='ipac')
        # make a copy of singlet list
        lines_with_doublets = copy.deepcopy(q3di.lines)
        # append doublet names to singlet list
        for (name1, name2) in zip(doublets['line1'], doublets['line2']):
            if name1 in linelist['name'] and name2 in linelist['name']:
                lines_with_doublets.append(name1+'+'+name2)

    if q3di.docontfit:
        # get continuum plotting function
        module = importlib.import_module('q3dfit.contplot')
        fcncontplot = getattr(module, q3di.fcncontplot)

    # READ DATA
    cube, vormap = __get_Cube(q3di, quiet)
    nspax, colarr, rowarr = __get_spaxels(cube, cols=cols, rows=rows)
    # TODO
    # if q3di.vormap is not None:
    #     vormap = q3di.vromap
    #     nvorcols = max(vormap)
    #     vorcoords = np.zeros(nvorcols, 2)
    #     for i in range(0, nvorcols):
    #         xyvor = np.where(vormap == i).nonzero()
    #         vorcoords[:, i] = xyvor

# INITIALIZE OUTPUT FILES, need to write helper functions (printlinpar,
# printfitpar) later

# INITIALIZE LINE HASH
    if q3di.dolinefit:
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
        for k in range(0, q3di.maxncomp):
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
            emlncomp[line] = np.zeros((cube.ncols, cube.nrows), dtype=int)
            emlweq['ftot'][line] = np.zeros((cube.ncols, cube.nrows),
                                            dtype=float) + bad
            emlflx['ftot'][line] = np.zeros((cube.ncols, cube.nrows),
                                            dtype=float) + bad
            emlflxerr['ftot'][line] = np.zeros((cube.ncols, cube.nrows),
                                               dtype=float) + bad
            for k in range(0, q3di.maxncomp):
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

    # LOOP THROUGH SPAXELS
    # switch to track when first continuum processed
    firstcontproc = True

    for ispax in range(0, nspax):
        i = colarr[ispax]
        j = rowarr[ispax]

        if not quiet:
            print(f'Column {i+1} of {cube.ncols}')

        # set this to false unless we're using Voronoi binning
        # and the tiling is missing
        vortile = True
        labin = '{0.outdir}{0.label}'.format(q3di)
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
            if not quiet:
                print(f'    Row {j+1} of {cube.nrows}')

            #TODO
            # if q3di.vormap is not None:
            #    if np.isfinite(q3di.vormap[i][j]) and \
            #            q3di.vormap[i][j] is not bad:
            #        iuse = vorcoords[q3di.vormap[i][j] - 1, 0]
            #        juse = vorcoords[q3di.vormap[i][j] - 1, 1]
            #    else:
            #        vortile = False
            #else:
            iuse = i
            juse = j

            if vortile:
                flux = cube.dat[iuse, juse, :].flatten()
                err = cube.err[iuse, juse, :].flatten()
                dq = cube.dq[iuse, juse, :].flatten()
                labin = '{0.outdir}{0.label}_{1:04d}_{2:04d}'.\
                    format(q3di, iuse+1, juse+1)
                labout = '{0.outdir}{0.label}_{1:04d}_{2:04d}'.\
                    format(q3di, i+1, j+1)

        # Restore fit after a couple of sanity checks
        if vortile:
            infile = labin + '.npy'
            outfile = labout
            nodata = flux.nonzero()
            ct = len(nodata[0])
        else:
            # missing data for this spaxel
            filepresent = False
            ct = 0

        if ct == 0 or not os.path.isfile(infile):

            badmessage = f'        No data for [{i+1}, {j+1}]'
            print(badmessage)

        else:

            fitdict = (np.load(infile, allow_pickle='TRUE')).item()

            # Restore original error.
            fitdict['spec_err'] = err[fitdict['fitran_indx']]

            if not fitdict['noemlinfit']:
                # get line fit params
                linepars, tflux = \
                    sepfitpars(linelist, fitdict['param'],
                               fitdict['perror'],
                               q3di['maxncomp'], tflux=True,
                               doublets=doublets)
#                lineweqs = cmpweq(fitdict, linelist, doublets = emldoublets)

            # plot emission line data, print data to a file
            if not noplots:

                # plot emission lines
                if not fitdict['noemlinfit']:
                    if 'nolines' not in linepars:
                        if 'fcnpltlin' in q3di:
                            fcnpltlin = q3di['fcnpltlin']
                        else:
                            fcnpltlin = 'pltlin'
                        module = \
                            importlib.import_module('q3dfit.' +
                                                    fcnpltlin)
                        pltlinfcn = getattr(module, fcnpltlin)
                        if 'argspltlin1' in q3di:
                            pltlinfcn(fitdict, q3di['argspltlin1'],
                                      outfile + '_lin1')
                        if 'argspltlin2' in q3di:
                            pltlinfcn(fitdict, q3di['argspltlin2'],
                                      outfile + '_lin2')

            # Possibly add later: print fit parameters to a text file

            if not fitdict['noemlinfit']:
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

                    # assign to output dictionary
                    emlncomp[line][i,j] = ctgd

                if thisncomp == 1:
                    isort = [0]
                    if 'flipsort' in q3di:
                        if flipsort[i, j]:
                            print('Flipsort set for spaxel [' + str(i+1)
                                  + ',' + str(j + 1) + '] but ' +
                                  'only 1 component. Setting to 2 components' +
                                  ' and flipping anyway.')
                            isort = [0, 1]  # flipped
                elif thisncomp >= 2:
                    # sort components
                    igd = np.arange(thisncomp)
                    # indices = np.arange(q3di['maxncomp'])
                    sigtmp = linepars['sigma'][thisncompline]
                    fluxtmp = linepars['flux'][thisncompline]
                    if 'sorttype' not in q3di:
                        isort = np.argsort(sigtmp[igd])
                    elif q3di['sorttype'] == 'wave':
                        isort = np.argsort(linepars['wave'][line, igd])
                    elif q3di['sorttype'] == 'reversewave':
                        isort = np.argsort(linepars['wave'][line, igd])[::-1]

                    if 'flipsort' in q3di:
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
                   {'dat': np.zeros((cube.ncols, cube.nrows, cube.nwave)),
                    'err': np.zeros((cube.ncols, cube.nrows, cube.nwave)),
                    'dq':  np.zeros((cube.ncols, cube.nrows, cube.nwave)),
                    'norm_div': np.zeros((cube.ncols, cube.nrows,
                                          cube.nwave)),
                    'norm_sub': np.zeros((cube.ncols, cube.nrows,
                                          cube.nwave))}

                if 'decompose_ppxf_fit' in q3di:
                    contcube = \
                        {'wave': fitdict['wave'],
                         'all_mod': np.zeros((cube.ncols, cube.nrows,
                                              cube.nwave)),
                         'stel_mod': np.zeros((cube.ncols, cube.nrows,
                                               cube.nwave)),
                         'poly_mod': np.zeros((cube.ncols, cube.nrows,
                                               cube.nwave)),
                         'stel_mod_tot': np.zeros((cube.ncols, cube.nrows))
                         + bad,
                         'poly_mod_tot': np.zeros((cube.ncols, cube.nrows))
                         + bad,
                         'poly_mod_tot_pct': np.zeros((cube.ncols,
                                                       cube.nrows))
                         + bad,
                         'stel_sigma': np.zeros((cube.ncols, cube.nrows))
                         + bad,
                         'stel_sigma_err': np.zeros((cube.ncols,
                                                     cube.nrows, 2))
                         + bad,
                         'stel_z': np.zeros((cube.ncols, cube.nrows))
                         + bad,
                         'stel_z_err': np.zeros((cube.ncols, cube.nrows,
                                                 2)) + bad,
                         'stel_rchisq': np.zeros((cube.ncols, cube.nrows))
                         + bad,
                         'stel_ebv': np.zeros((cube.ncols, cube.nrows))
                         + bad,
                         'stel_ebv_err': np.zeros((cube.ncols, cube.nrows,
                                                   2)) + bad}

                elif 'decompose_qso_fit' in q3di:
                    contcube = \
                        {'wave': fitdict['wave'],
                         'qso_mod':
                             np.zeros((cube.ncols, cube.nrows,
                                       cube.nwave)),
                         'qso_poly_mod':
                             np.zeros((cube.ncols, cube.nrows,
                                       cube.nwave)),
                         'host_mod':
                             np.zeros((cube.ncols, cube.nrows,
                                       cube.nwave)),
                         'poly_mod':
                             np.zeros((cube.ncols, cube.nrows,
                                       cube.nwave)),
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
                         np.zeros((cube.ncols, cube.nrows, cube.nwave)),
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

            hostcube['dat'][i, j, fitdict['fitran_indx']] = \
                fitdict['cont_dat']
            hostcube['err'][i, j, fitdict['fitran_indx']] = \
                err[fitdict['fitran_indx']]
            hostcube['dq'][i, j, fitdict['fitran_indx']] = \
                dq[fitdict['fitran_indx']]
            hostcube['norm_div'][i, j, fitdict['fitran_indx']] \
                = np.divide(fitdict['cont_dat'], fitdict['cont_fit'])
            hostcube['norm_sub'][i, j, fitdict['fitran_indx']] \
                = np.subtract(fitdict['cont_dat'], fitdict['cont_fit'])

            if 'decompose_ppxf_fit' in q3di:
                add_poly_degree = 4  # should match fitspec
                if 'argscontfit' in q3di:
                    if 'add_poly_degree' in q3di['argscontfit']:
                        add_poly_degree = \
                            q3di['argscontfit']['add_poly_degree']
                # Compute polynomial
                dumy_log, wave_log, _ = \
                    log_rebin([fitdict['wave'][0],
                               fitdict['wave'][len(fitdict['wave'])-1]],
                              fitdict['spec'])
                xnorm = np.linspace(-1., 1., len(wave_log))
                cont_fit_poly_log = 0.0
                for k in range(0, add_poly_degree):
                    cfpllegfun = legendre(k)
                    cont_fit_poly_log += cfpllegfun(xnorm) * \
                        fitdict['ct_add_poly_weights'][k]
                interpfunction = \
                    interp1d(cont_fit_poly_log, wave_log, kind='linear',
                             fill_value="extrapolate")
                cont_fit_poly = interpfunction(np.log(fitdict['wave']))
                # Compute stellar continuum
                cont_fit_stel = np.subtract(fitdict['cont_fit'],
                                            cont_fit_poly)
                # Total flux fromd ifferent components
                cont_fit_tot = np.sum(fitdict['cont_fit'])
                contcube['all_mod'][i, j, fitdict['fitran_indx']] = \
                    fitdict['cont_fit']
                contcube['stel_mod'][i, j, fitdict['fitran_indx']] = \
                    cont_fit_stel
                contcube['poly_mod'][i, j, fitdict['fitran_indx']] = \
                    cont_fit_poly
                contcube['stel_mod_tot'][i, j] = np.sum(cont_fit_stel)
                contcube['poly_mod_tot'][i, j] = np.sum(cont_fit_poly)
                contcube['poly_mod_tot_pct'][i, j] \
                    = np.divide(contcube['poly_mod_tot'][i, j], cont_fit_tot)
                contcube['stel_sigma'][i, j] = fitdict['ct_ppxf_sigma']
                contcube['stel_z'][i, j] = fitdict['zstar']

                if 'ct_errors' in fitdict:
                    contcube['stel_sigma_err'][i, j, :] \
                        = fitdict['ct_errors']['ct_ppxf_sigma']
                # assuming that ct_errors is a dictionary
                else:  # makes an array with two arrays
                    contcube['stel_sigma_err'][i, j, :] \
                        = [fitdict['ct_ppxf_sigma_err'],
                           fitdict['ct_ppxf_sigma_err']]

                if 'ct_errors' in fitdict:
                    contcube['stel_z_err'][i, j, :] = \
                        fitdict['ct_errors']['zstar']
                else:
                    contcube['stel_z_err'][i, j, :] \
                        = [fitdict['zstar_err'], fitdict['zstar_err']]

            elif 'decompose_qso_fit' in q3di:
                if q3di['fcncontfit'] == 'fitqsohost':
                    if 'qsoord' in q3di['argscontfit']:
                        qsoord = q3di['argscontfit']['qsoord']
                    else:
                        qsoord = None

                    if 'hostord' in q3di['argscontfit']:
                        hostord = q3di['argscontfit']['hostord']
                    else:
                        hostord = None

                    if 'blrpar' in q3di['argscontfit']:
                        blrpar = q3di['argscontfit']['blrpar']
                    else:
                        blrpar = None
                    # default here must be same as in IFSF_FITQSOHOST
                    if 'add_poly_degree' in q3di['argscontfit']:
                        add_poly_degree = \
                            q3di['argscontfit']['add_poly_degree']
                    else:
                        add_poly_degree = 30

                    # Get and renormalize template
                    qsotemplate = \
                        np.load(q3di['argscontfit']['qsoxdr'],
                                allow_pickle='TRUE').item()
                    qsowave = qsotemplate['wave']
                    qsoflux_full = qsotemplate['flux']

                    iqsoflux = \
                        np.where((qsowave >= fitdict['fitran'][0]) &
                                 (qsowave <= fitdict['fitran'][1]))
                    qsoflux = qsoflux_full[iqsoflux]

                    # If polynomial residual is re-fit with PPXF,
                    # compute polynomial component
                    if 'refit' in q3di['argscontfit'] and \
                        'args_questfit' not in q3di['argscontfit']:

                        par_qsohost = fitdict['ct_coeff']['qso_host']
                        # par_stel = fitdict['ct_coeff']['stel']
                        dumy_log, wave_rebin, _ = \
                            log_rebin([fitdict['wave'][0],
                                       fitdict['wave']
                                      [len(fitdict['wave'])-1]],
                                      fitdict['spec'])
                        xnorm = np.linspace(-1., 1., len(wave_rebin))
                        if add_poly_degree > 0:
                            par_poly = fitdict['ct_coeff']['poly']
                            polymod_log = \
                                legendre.legval(xnorm, par_poly)
                            interpfunct = \
                                interp1d(wave_rebin, polymod_log,
                                         kind='cubic',
                                         fill_value="extrapolate")
                            polymod_refit = \
                                interpfunct(np.log(fitdict['wave']))
                        else:
                            polymod_refit = np.zeros(len(fitdict['wave']),
                                                     dtype=float)
                        contcube['stel_sigma'][i, j] = \
                            fitdict['ct_coeff']['ppxf_sigma']
                        contcube['stel_z'][i, j] = fitdict['zstar']

                        # MC? errors in stellar sigma and redshift
                        if 'ct_errors' in fitdict:
                            contcube['stel_sigma_err'][i, j, :] \
                                = fitdict['ct_errors']['ct_ppxf_sigma']
                        else:
                            contcube['stel_sigma_err'][i, j, :] \
                                = [fitdict['ct_ppxf_sigma_err'],
                                   fitdict['ct_ppxf_sigma_err']]
                        if 'ct_errors' in fitdict:
                            contcube['stel_z_err'][i, j, :] \
                                = fitdict['ct_errors']['zstar']
                        else:
                            contcube['stel_z_err'][i, j, :] \
                                = [fitdict['zstar_err'],
                                   fitdict['zstar_err']]

                    # Refitting with questfit in the MIR
                    elif 'refit' in q3di['argscontfit'] and \
                        q3di['argscontfit']['refit'] == 'questfit':

                        par_qsohost = fitdict['ct_coeff']['qso_host']
                        polymod_refit = np.zeros(len(fitdict['wave']),
                                                 dtype=float)

                    else:
                        par_qsohost = fitdict['ct_coeff']
                        polymod_refit = 0.0

                    #produce fit with template only and with template + host. Also
                    #output QSO multiplicative polynomial
                    qsomod_polynorm = 0.
                    qsomod = \
                        qsohostfcn(fitdict['wave'], params_fit=par_qsohost,
                                   qsoflux=qsoflux, qsoonly=True,
                                   blrpar=blrpar, qsoord=qsoord,
                                   hostord=hostord)
                    hostmod = np.array(fitdict['cont_fit_pretweak'] - qsomod)

                    #if continuum is tweaked in any region, subide resulting residual
                    #proportionality @ each wavelength btwn qso and host components
                    qsomod_notweak = qsomod
                    if 'tweakcntfit' in q3di:
                        modresid = fitdict['cont_fit'] - fitdict['cont_fit_pretweak']
                        inz = np.where((qsomod != 0) & (hostmod != 0))[0]
                        qsofrac = np.zeros(len(qsomod))
                        for ind in inz:
                            qsofrac[ind] = qsomod[ind] / (qsomod[ind] + hostmod[ind])
                        qsomod += modresid * qsofrac
                        hostmod += modresid * (1.0 - qsofrac)
                    #components of qso fit for plotting
                    qsomod_normonly = qsoflux
                    if blrpar is not None:
                        qsomod_blronly = \
                            qsohostfcn(fitdict['wave'],
                                       params_fit=par_qsohost,
                                       qsoflux=qsoflux, blronly=True,
                                       blrpar=blrpar, qsoord=qsoord,
                                       hostord=hostord)
                    else:
                        qsomod_blronly = 0.

                # CB: adding option to plot decomposed QSO fit if questfit is used
                elif q3di['fcncontfit'] == 'questfit':
                    from q3dfit.questfit import quest_extract_QSO_contrib
                    qsomod, hostmod, qsomod_intr, hostmod_intr = \
                        quest_extract_QSO_contrib(fitdict['ct_coeff'], q3di)
                    qsomod_polynorm = 1.
                    qsomod_notweak = qsomod
                    qsoflux = qsomod.copy()/np.median(qsomod)
                    qsomod_normonly = qsoflux
                    polymod_refit = 0.
                    blrpar = None
                    qsomod_blronly = 0.

            # Case of PPXF fit with quasar template
            elif q3di['fcncontfit'] == 'ppxf' and \
                'qsotempfile' in q3di:
                qsotempfile = np.load(q3di['qsotempfile'],
                                      allow_pickle='TRUE').item()
                struct_qso = qsotempfile
                qsomod = struct_qso['cont_fit'] * \
                    fitdict['ct_coeff'][len(fitdict['ct_coeff']) - 1]
                hostmod = fitdict['cont_fit'] - qsomod

            elif q3di['fcncontfit'] == 'questfit':

                contcube['all_mod'][i, j, fitdict['fitran_indx']] = \
                    fitdict['cont_fit']
                contcube['stel_z'][i, j] = fitdict['zstar']
                if 'ct_errors' in fitdict:
                    contcube['stel_z_err'][i, j, :] = \
                        fitdict['ct_errors']['zstar']
                else:
                    contcube['stel_z_err'][i, j, :] = [0, 0]

            # continuum attenuation and errors
            contcube['stel_ebv'][i, j] = fitdict['ct_ebv']
            if 'ct_errors' in fitdict:
                contcube['stel_ebv_err'][i, j, :] = \
                    fitdict['ct_errors']['ct_ebv']
            else:
                contcube['stel_rchisq'][i, j] = 0.

            # Print ppxf results to stdout
            if 'decompose_ppxf_fit' in q3di or \
                'decompose_qso_fit' in q3di:
                if 'argscontfit' in q3di:
                    if 'print_output' in q3di['argscontfit']:
                        print("PPXF results: ")
                        if 'decompose_ppxf_fit' in q3di:
                            ct_coeff_tmp = fitdict['ct_coeff']
                            poly_tmp_pct = contcube['poly_mod_tot_pct'][i, j]
                        else:
                            ct_coeff_tmp = fitdict['ct_coeff']['stel']
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
                                np.load(q3di['startempfile']+".npy",
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
                                  str(fitdict['ct_ppxf_sigma']) + 'km/s')

            # Plot QSO and host only continuum fit
            if q3di['decompose_qso_fit']:

                fitdict_host = copy.deepcopy(fitdict)
                fitdict_qso = copy.deepcopy(fitdict)

                fitdict_host['spec'] -= qsomod
                fitdict_host['cont_dat'] -= qsomod
                fitdict_host['cont_fit'] -= qsomod

                fitdict_qso['spec'] -= hostmod
                fitdict_qso['cont_dat'] -= hostmod
                fitdict_qso['cont_fit'] -= hostmod

                contcube['qso_mod'][i, j, fitdict['fitran_indx']] = \
                    qsomod.copy()
                contcube['qso_poly_mod'][i, j, fitdict['fitran_indx']] = \
                    qsomod_polynorm
                contcube['host_mod'][i, j, fitdict['fitran_indx']] = \
                    hostmod.copy()
                if isinstance(polymod_refit, float):
                    contcube['poly_mod'][i, j, fitdict['fitran_indx']] = 0.
                else:
                    contcube['poly_mod'][i, j, fitdict['fitran_indx']] = \
                        polymod_refit.copy()
                contcube['npts'][i, j] = len(fitdict['fitran_indx'])

                if 'remove_scattered' in q3di:
                    contcube['host_mod'][i, j, fitdict['fitran_indx']] -= \
                        polymod_refit

                # Update hostcube.dat to remove tweakcnt mods
                # Data minus (emission line model + QSO model,
                # tweakcnt mods not included in QSO model)

                hostcube['dat'][i, j, fitdict['fitran_indx']] \
                    = fitdict['cont_dat'] - qsomod_notweak

                if not noplots and np.sum(fitdict_host['cont_fit']) != 0.0:
                    if 'refit' in q3di['argscontfit']:
                        compspec = np.array([polymod_refit,
                                             hostmod - polymod_refit])
                        comptitles = ['ord. ' + str(add_poly_degree) +
                                      ' Leg. poly.', 'stel. temp.']
                    else:
                        compspec = [hostmod.copy()]
                        # CB: Work-around - think about this more later
                        if q3di['fcncontfit'] == 'questfit':
                            compspec = [hostmod.copy()]
                        comptitles = ['exponential terms']

                    fcncontplot(fitdict_host, outfile + '_cnt_host',
                               compspec=compspec, comptitles=comptitles,
                               title='Host', fitran=q3di['fitran'],
                               q3di=q3di, **argspltcont)
                    if blrpar is not None and max(qsomod_blronly) != 0.:
                        qsomod_blrnorm = np.median(qsomod) / \
                            max(qsomod_blronly)
                        compspec = np.array([qsomod_normonly,
                                             qsomod_blronly *
                                             qsomod_blrnorm])
                        comptitles = ['raw template', 'scattered*' +
                                   str(qsomod_blrnorm)]
                    else:
                        compspec = [qsomod_normonly.copy()]
                        comptitles = ['raw template']

                    if q3di['fcncontfit'] != 'questfit':
                        fcncontplot(fitdict_qso, str(outfile) + '_cnt_qso',
                                   compspec=compspec, comptitles=comptitles,
                                   title='QSO', fitran=q3di['fitran'],
                                   q3di=q3di, **argspltcont)
                    else:
                        fcncontplot(fitdict_qso, str(outfile) + '_cnt_qso',
                                   compspec=[fitdict_qso['cont_fit']],
                                   title='QSO', fitran=q3di['fitran'],
                                   comptitles=['QSO'], q3di=q3di,
                                   **argspltcont)

            # Plot continuum
            # Make sure fit doesn't indicate no continuum; avoids
            # plot range error in continuum fitting routine,
            # as well as a blank plot!
            if not noplots and sum(fitdict['cont_fit']) != 0.0:

                if q3di['decompose_qso_fit']:
                    fcncontplot(fitdict, outfile + '_cnt',
                               compspec=np.array([qsomod, hostmod]),
                               title='Total', comptitles=['QSO', 'host'],
                               fitran=q3di['fitran'], q3di=q3di,
                               **argspltcont)

                    if 'compare_to_real_decomp' in q3di:     # CB: in the case of the MIR mock ETC cube, compare the recovered QSO/host contribution from the combined cube to the real ones from the QSO/host only simulations
                        if q3di['compare_to_real_decomp']['on']:
                            from q3dfit import readcube
                            argsreadcube_dict = {'fluxunit_in': 'Jy',
                                                'waveunit_in': 'angstrom',
                                                'waveunit_out': 'micron'}

                            file_host = q3di['compare_to_real_decomp']['file_host']
                            file_qso = q3di['compare_to_real_decomp']['file_qso']

                            if 'argsreadcube' in q3di:
                                cube2 = Cube(file_host, quiet=quiet,
                                             header=header, datext=datext, varext=varext,
                                             dqext=dqext, **q3di['argsreadcube'])
                                cube3 = Cube(file_qso, quiet=quiet,
                                             datext=datext, varext=varext,
                                             dqext=dqext, **q3di['argsreadcube'])
                            else:
                                cube2 = Cube(file_host, quiet=quiet,
                                             datext=datext, varext=varext,
                                             dqext=dqext)
                                cube3 = Cube(file_qso, quiet=quiet,
                                             datext=datext, varext=varext,
                                             dqext=dqext)

                            lam_exclude = sorted(set(cube2.wave.tolist()) - set(fitdict['wave'].tolist())) # exclude wavelength that are in cube2.wave but not in fitdict['wave']
                            okwave = np.ones(len(cube2.wave)).astype(bool)
                            for i,lam_i in enumerate(cube2.wave):
                                if lam_i in lam_exclude:
                                    okwave[i] = False

                            # from scipy import constants
                            # from astropy import units as u
                            # c_scale =  constants.c * u.Unit('m').to('micron') /(cube2.wave[okwave])**2 *1e-23  *1e10      # [1e-10 erg/s/cm^2/um/sr]]

                            hostspec_real = cube2.dat[iuse, juse, :].flatten()[okwave] # * c_scale
                            qsospec_real = cube3.dat[iuse, juse, :].flatten()[okwave] # * c_scale

                            fitdict_overpredict = fitdict.copy()
                            fitdict_overpredict['cont_dat'] = 1.*fitdict['cont_fit']/fitdict['cont_dat']
                            fitdict_overpredict['cont_fit'] = 1.*fitdict['cont_fit']/fitdict['cont_dat']

                            print('Check: host-only cube + QSO-only cube = combined-cube spectrum?: ', (qsospec_real + hostspec_real)/fitdict['cont_dat'], '\n(check only works if we are actually running on the combined cube)' )

                            fcncontplot(fitdict, outfile + '_cnt_decomp',
                                       compspec=np.array([qsomod, hostmod, qsospec_real, hostspec_real]),
                                       title='Total', comptitles=['QSO model', 'host model', 'QSO real', 'host real'],
                                       compcols=['c', 'plum', 'mediumblue', 'darkviolet'],
                                       fitran=q3di.fitran, q3di=q3di,
                                       **argspltcont)

                            fcncontplot(fitdict_overpredict, outfile + '_cnt_overpredict',
                                       compspec=np.array([1.*qsomod/qsospec_real, 1.*hostmod/hostspec_real]),
                                       title='Total',
                                       comptitles=['QSO_model / QSO_real = {:.3f}'.format(np.median(1.*qsomod/qsospec_real)), '<host_model / host_real> = {:.3f}'.format(np.median(1.*hostmod/hostspec_real))],
                                       compcols=['c', 'plum'],
                                       fitran=q3di.fitran, q3di=q3di,
                                       **argspltcont)




                elif 'decompose_ppxf_fit' in q3di:
                    fcncontplot(fitdict, outfile + '_cnt',
                               compspec=np.array([cont_fit_stel,
                                                  cont_fit_poly]),
                               title='Total',
                               comptitles=['stel. temp.', 'ord. ' +
                                           str(add_poly_degree) +
                                           'Leg.poly'],
                               fitran=q3di['fitran'], q3di=q3di,
                               **argspltcont)
                else:
                    fcncontplot(fitdict, outfile + '_cnt',
                               fitran=q3di['fitran'],
                               q3di=q3di,
                               ct_coeff=fitdict['ct_coeff'],
                               title='Total', **argspltcont)

            # Plot continuum
            # Make sure fit doesn't indicate no continuum; avoids
            # plot range error in continuum fitting routine,
            # as well as a blank plot!
            if not noplots and 'argscontfit' in q3di.keys():
                if 'plot_decomp' in q3di['argscontfit'].keys():
                    if q3di['argscontfit']['plot_decomp']:
                        from q3dfit.plot_quest import plot_quest
                        if not fitdict['noemlinfit']:
                            lam_lines = \
                                fitdict['linelist']['lines'].tolist()
                        else:
                            lam_lines = []
                        plot_quest(fitdict['wave'],
                                   fitdict['cont_dat']+fitdict['emlin_dat'],
                                   fitdict['cont_fit']+fitdict['emlin_fit'],
                                   fitdict['ct_coeff'], q3di,
                                   lines=lam_lines,
                                   linespec=fitdict['emlin_fit'])

    if filepresent and ct != 0:
        # Save emission line and continuum dictionaries
        np.savez('{[outdir]}{[label]}'.format(q3di, q3di)+'.lin.npz',
                 emlwav=emlwav, emlwaverr=emlwaverr,
                 emlsig=emlsig, emlsigerr=emlsigerr,
                 emlflx=emlflx, emlflxerr=emlflxerr,
                 emlweq=emlweq, emlncomp=emlncomp,
                 ncols=cube.ncols, nrows=cube.nrows)
        np.save('{[outdir]}{[label]}'.format(q3di, q3di)+'.cont.npy',
                contcube)

    # Output to fits files -- test
    #from astropy.io import fits
    #hdu = fits.PrimaryHDU(emlflx['ftot']['[OIII]5007'][:,:])
    #hdu.writeto('{[outdir]}{[label]}'.format(q3di, q3di)+'_OIII5007flx.fits')


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
