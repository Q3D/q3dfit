#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Functions called by :py:mod:`~q3dfit.q3df.q3dfit` for single- and multi-threaded
execution.

If called externally (i.e., if we run q3df_helperFunctions.py from the command line),
this script will execute q3df_multiCore on multiple cores using MPI. This script
is called by :py:mod:`~q3dfit.q3df.q3dfit` when running in a multi-core environment.
Functions called by :py:mod:`~q3dfit.q3df.q3dfit` for single- and multi-threaded
execution.

If called externally (i.e., if we run q3df_helperFunctions.py from the command line),
this script will execute q3df_multiCore on multiple cores using MPI. This script
is called by :py:mod:`~q3dfit.q3df.q3dfit` when running in a multi-core environment.
"""
import time
from typing import Optional

import numpy as np
from astropy.table import Table

from q3dfit import q3dutil
from q3dfit.fitloop import fitloop
from q3dfit.q3din import q3din
from q3dfit.readcube import Cube
from q3dfit.spectConvol import spectConvol


def execute_fitloop(nspax: int,
                    colarr: np.ndarray[int],
                    rowarr: np.ndarray[int],
                    cube: Cube,
                    q3di: q3din,
                    linelist: Table,
                    specConv: spectConvol,
                    onefit: bool,
                    quiet: bool,
                    core: int=1,
                    nocrash: bool=False):
    '''
    Handle the FITLOOP execution. In its own function due to commonality between 
    single- and multi-threaded execution

    Parameters
    ----------
    nspax
        Number of spaxels to be fitted.
    colarr
        Array of column #s of spaxels to be fit (0-offset).
    rowarr
        Array of row #s of spaxels to be fit (0-offset).
    cube
        :py:class:`~q3dfit.readcube.Cube` object containing data to be fit.
    q3di
        :py:class:`~q3dfit.q3din.q3din` object containing fitting parameters.
    linelist
        Emission line labels and rest frame wavelengths, as part of an astropy Table
        output by :py:func:`~q3dfit.linelist.linelist`.
    specConv
        Instance of :py:class:`~q3dfit.spectConvol.spectConvol` specifying 
        the instrumental spectral resolution convolution.
    quiet
        If False, progress messages are written to stdout.
    onefit
        If True, only one fit is performed.
    core
        Optional. Core number for multi-threaded execution. Default is 1.
    nocrash
        Optional. If set, the routine will continue with the next spaxel in case
        of a crash. Default is False.
    '''
    q3dutil.write_msg('Core '+str(core)+': # spaxels fit='+str(nspax), 
        file=q3di.logfile, quiet=quiet)
    for ispax in range(0, nspax):
        if nocrash:     # In case of crash, just continue with the next spaxel
            try:
                fitloop(ispax, colarr, rowarr, cube, q3di, linelist, specConv,
                            onefit=onefit, quiet=quiet)
            except:
                continue
        else:           # Regular run; no continuation in case of crash
            fitloop(ispax, colarr, rowarr, cube, q3di, linelist, specConv,
                    onefit=onefit, quiet=quiet)


def q3df_oneCore(inobj: str | q3din,
                 cols: Optional[int | list | np.ndarray]=None,
                 rows: Optional[int | list | np.ndarray]=None,
                 onefit: bool=False,
                 quiet: bool=True,
                 nocrash: bool=False):
    '''
    :py:func:`~q3dfit.q3df.q3dfit` calls this function for single-threaded execution.
    The parameters are passed through from :py:func:`~q3dfit.q3df.q3dfit` and on
    to :py:func:`~q3dfit.fitloop.fitloop`.

    Parameters
    ----------
    inobj
        Either a string with the path to the numpy save file containing the
        input initialization object :py:class:`~q3dfit.q3din.q3din`, 
        or the object itself.
    cols
        Optional. Column values for spaxels to be fitted. Default is None, which means
        all columns will be fitted. If a scalar, only that column will be fitted. If a
        2-element list or array, the elements are the starting and ending columns to be
        fitted. Unity-offset values assumed.
    rows
        Optional. Row values for spaxels to be fitted. Default is None, which means
        all rows will be fitted. If a scalar, only that row will be fitted. If a
        2-element list or array, the elements are the starting and ending rows to be
        fitted. Unity-offset values assumed.
    onefit
        Optional. If set, only one fit is performed in :py:func:`~q3dfit.fitloop.fitloop`. 
        Default is False.
    quiet
        Optional. If False, progress messages are written to stdout. 
        Default is True. 
    nocrash
        Optional. If set, the routine will continue with the next spaxel in case 
        of a crash. Default is False.
    '''
    starttime = time.time()
    q3di = q3dutil.get_q3dio(inobj)
    linelist = q3di.get_linelist()
    cube = q3di.load_cube(quiet=quiet)
    specConv = q3di.get_dispersion()
    #if cols and rows and vormap:
    #    cols = q3dutil.get_voronoi(cols, rows, vormap)
    #    rows = 1
    nspax, colarr, rowarr = q3dutil.get_spaxels(cube.ncols, cube.nrows, cols, rows)
    # execute FITLOOP
    execute_fitloop(nspax, colarr, rowarr, cube, q3di,
                    linelist, specConv, onefit, quiet,
                    nocrash=nocrash)
    timediff = time.time()-starttime
    q3dutil.write_msg(f'q3df: Total time for calculation: {timediff:.2f} s.',
        file=q3di.logfile, quiet=quiet)


def q3df_multiCore(rank: int,
                   size: int,
                   inobj: str | q3din,
                   cols: Optional[list[int]]=None,
                   rows: Optional[list[int]]=None,
                   onefit: bool=False,
                   quiet: bool=True, 
                   nocrash: bool=False):

    '''
    Run :py:mod:`~q3dfit.fitloop.fitloop` on a single core in a multi-core environment. 
    This function is called by :py:mod:`~q3dfit.q3df_helperFunctions` when run externally.
    :py:func:`~q3dfit.q3df.q3dfit` calls :py:mod:`~q3dfit.q3df_helperFunctions` 
    externally when running in a multi-core environment.

    Parameters
    ----------
    rank
        Rank of this core for MPI execution.
    size
        Number of cores available for MPI execution.
    inobj
        Either a string with the path to the numpy save file containing the
        input initialization object :py:class:`~q3dfit.q3din.q3din`,
        or the object itself.
    cols
        Optional. Column values for spaxels to be fitted. If a single-element list, only that
        column will be fitted. If a 2-element list, the elements are the starting and
        ending columns to be fitted. Unity-offset values assumed. Default is None, which
        means all columns will be fitted.
    rows
        Optional. Row values for spaxels to be fitted. If a single-element list, only that row
        will be fitted. If a 2-element list, the elements are the starting and ending
        rows to be fitted. Unity-offset values assumed. Default is None, which means
        all rows will be fitted.
    onefit
        Optional. If True, only one fit is performed in :py:func:`~q3dfit.fitloop.fitloop`,
        instead of two. Default is False.
    quiet
        Optional. If False, progress messages are written to stdout. Default is True. 
    '''
    starttime = time.time()
    q3di = q3dutil.get_q3dio(inobj)
    linelist = q3di.get_linelist()

    # add core number to logfile name
    if q3di.logfile is not None:
        q3di.logfile = q3di.logfile + '_core'+str(rank+1)

    cube = q3di.load_cube(quiet=quiet) # ,vormap
    specConv = q3di.get_dispersion()
    
    #if cols and rows and vormap:
    #    cols = q3dutil.get_voronoi(cols, rows, vormap)
    #    rows = 1
    nspax, colarr, rowarr = q3dutil.get_spaxels(cube.ncols, cube.nrows, cols, rows)
    # get the range of spaxels this core is responsible for
    start = int(np.floor(nspax * rank / size))
    stop = int(np.floor(nspax * (rank+1) / size))
    colarr = colarr[start:stop]
    rowarr = rowarr[start:stop]
    # number of spaxels THIS CORE is responsible for
    nspax_thisCore = stop-start
    # execute FITLOOP
    execute_fitloop(nspax_thisCore, colarr, rowarr, cube, q3di,
                    linelist, specConv, onefit, quiet, 
                    core=rank+1, nocrash=nocrash)
    if q3di.logfile is None:
        from sys import stdout
        #logtmp = stdout
    #else:
        #logtmp = q3di.logfile
    timediff = time.time()-starttime
    q3dutil.write_msg(f'Q3DF: Total time for calculation: {timediff:.2f} s.',
                      file=q3di.logfile, quiet=quiet)


def string_to_intList(strArray: str) -> Optional[list[int]]:
    '''
    Convert a string of the form "[1, 2, 3, ...]" to a list of integers.

    Parameters
    ----------
    strArray
        The string to be converted. The string is expected to be
        a list of integers in square brackets, separated by commas; a 
        single integer; or a value starting with 'N' to indicate None.
        
    Returns
    -------
    intList
        A list of integers, or None if the input string started with 'N'.
    '''
    if strArray.startswith("N"):
        return None
    else:
        # strip leading and trailing brackets
        if strArray.startswith("["):
            strArray = strArray[1:-1]
        # form a list by splitting on commas
        intList = strArray.split(",")
        for i in range(len(intList)):
            # remove whitespace
            intList[i] = intList[i].strip()
            # remove leading and trailing quotes and cast to int
            intList[i] = int(intList[i].strip("'"))
        return intList


# if called externally, default to MPI behavior
# i.e., if we run q3df_helperFunctions.py from the command line
if __name__ == "__main__":
    from sys import argv
    from mpi4py import MPI
    # get multiprocessor data: number of tasks and which one this is
    comm = MPI.COMM_WORLD
    size = comm.Get_size()
    rank = comm.Get_rank()

    inobj = argv[1]
    # convert command-line argument strings to Python types
    cols = string_to_intList(argv[2])
    rows = string_to_intList(argv[3])
    if argv[4].startswith("T"):
        onefit = True
    else:
        onefit = False
    if argv[5].startswith("T"):
        quiet = True
    else:
        quiet = False
    # nocrash option
    if argv[6].startswith("T"):
        nocrash = True
    else:
        nocrash = False

        
    q3df_multiCore(rank, size, inobj, cols, rows, onefit, quiet, nocrash=nocrash)
