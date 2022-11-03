#!/usr/bin/env python
# -*- coding: utf-8 -*-
import numpy as np
import time

from mpi4py import MPI
from q3dfit import spectConvol
from q3dfit.exceptions import InitializationError
from q3dfit.fitloop import fitloop
from q3dfit.linelist import linelist
from q3dfit.readcube import Cube
from sys import argv, path

"""
Helper functions for q3df.py.

Created on Tue May 26 13:37:58 2020

@author: drupke
@author: canicetti
"""

def __get_linelist(q3di):
    '''
    Get linelist

    Parameters
    ----------
    q3di : TYPE
        DESCRIPTION.

    Returns
    -------
    listlines : TYPE
        DESCRIPTION.

    '''
    listlines = linelist(q3di.lines, **q3di.argslinelist)
    return listlines


def __get_dispersion(q3di, cube, quiet=True):
    '''
    read in the dispersion list and save to memory
    default return value is None (no convolution)

    Parameters
    ----------
    q3di : TYPE
        DESCRIPTION.
    cube : TYPE
        DESCRIPTION.
    quiet : TYPE, optional
        DESCRIPTION. The default is True.

    Returns
    -------
    TYPE
        DESCRIPTION.

    '''
    if not q3di.spect_convol:
        return None
    elif q3di.spect_convol:
        return spectConvol.spectConvol(q3di, cube, quiet=quiet)


def __get_Cube(q3di, quiet, logfile=None):
    '''
    initialize Cube object


    Parameters
    ----------
    q3di : TYPE
        DESCRIPTION.
    quiet : TYPE
        DESCRIPTION.
    logfile : TYPE, optional
        DESCRIPTION. The default is None.

    Returns
    -------
    cube : TYPE
        DESCRIPTION.
    TYPE
        DESCRIPTION.

    '''
    if logfile is None:
        from sys import stdout
        logfile = stdout
    #if q3di.__contains__('argsreadcube'):
        # datext, varext, and dqext can be specified through
        # argsreadcube or through q3di directly. Default to the value
        # in argsreadcube. Else set the value based on the default or from
        # q3di
    #    argsreadcubeuse = q3di['argsreadcube'].copy()
    #    if not argsreadcubeuse.__contains__('datext'):
    #        argsreadcubeuse['datext'] = datext
    #    if not argsreadcubeuse.__contains__('varext'):
    #        argsreadcubeuse['varext'] = varext
    #    if not argsreadcubeuse.__contains__('dqext'):
    #        argsreadcubeuse['dqext'] = dqext
    #else:
    #    argsreadcubeuse = {'datext': datext, 'varext': varext, 'dqext': dqext}

    cube = Cube(q3di.infile, quiet=quiet,
                logfile=logfile, datext=q3di.datext, varext=q3di.varext,
                dqext=q3di.dqext, vormap=q3di.vormap, **q3di.argsreadcube)

    return cube, q3di.vormap


def __get_q3di(initobj):
    '''
    Load initialization object. Determine whether it's already an object,
    or needs to be loaded from file.

    Parameters
    ----------
    q3di : string or object

    Raises
    ------
    InitializationError
        DESCRIPTION.

    Returns
    -------
    q3di object

    '''

    # If it's a string, assume it's an input .npy file
    if type(initobj) == str:
        q3diarr = np.load(initobj, allow_pickle=True)
        q3di = q3diarr[()]
    # If it's an ndarray, assume the file's been loaded but not stripped
    # to dict{}
    elif isinstance(initobj, np.ndarray):
        q3di = initobj[()]
    # If it's an object, assume all is well
    elif isinstance(initobj, object):
        q3di = initobj
    else:
        raise InitializationError('q3di not in expected format')

    return(q3di)


def __get_voronoi(cols, rows, vormap):
    '''
    construct voronoi map

    Parameters
    ----------
    cols : TYPE
        DESCRIPTION.
    rows : TYPE
        DESCRIPTION.
    vormap : TYPE
        DESCRIPTION.

    Raises
    ------
    ValueError
        DESCRIPTION.

    Returns
    -------
    cols : TYPE
        DESCRIPTION.

    '''
    if len(cols) == 1 and len(rows) == 1:
        cols = vormap[cols[0]-1, rows[0]-1]
        return cols
    else:
        raise ValueError('Q3DF: ERROR: Can only specify 1 spaxel, \
                         or all spaxels, in Voronoi mode.')


def __get_spaxels(cube, cols=None, rows=None):
    '''
    Set up 1D arrays specifying column value and row value at each point to be
    fitted. These are zero-offset for indexing other arrays.

    Parameters
    ----------
    cube : TYPE
        DESCRIPTION.
    cols : TYPE, optional
        DESCRIPTION. The default is None.
    rows : TYPE, optional
        DESCRIPTION. The default is None.

    Returns
    -------
    None.

    '''
    # Set up 2-element arrays with starting and ending columns/rows
    # These are unity-offset to reflect pixel labels
    if not cols:
        cols = [1, cube.ncols]
        ncols = cube.ncols
#   case: cols is a scalar
    elif not isinstance(cols, (list, np.ndarray)):
        cols = [cols, cols]
        ncols = 1
    elif len(cols) == 1:
        cols = [cols[0], cols[0]]
        ncols = 1
    else:
        ncols = cols[1]-cols[0]+1
    if not rows:
        rows = [1, cube.nrows]
        nrows = cube.nrows
    elif not isinstance(rows, (list, np.ndarray)):
        rows = [rows, rows]
        nrows = 1
    elif len(rows) == 1:
        rows = [rows[0], rows[0]]
        nrows = 1
    else:
        nrows = rows[1]-rows[0]+1

    if len(cols) or len(rows) <= 2:
        colarr = np.empty((ncols, nrows), dtype=np.int32)
        rowarr = np.empty((ncols, nrows), dtype=np.int32)
        for i in range(nrows):
            colarr[:, i] = np.arange(cols[0]-1, cols[1], dtype=np.int32)
        for i in range(ncols):
            rowarr[i, :] = np.arange(rows[0]-1, rows[1], dtype=np.int32)

        # Flatten from 2D to 1D arrays to preserve indexing using only ispax
        # currently not needed. fitloop expects 2D lists.
        colarr = colarr.flatten()
        rowarr = rowarr.flatten()
        nspax = ncols * nrows

    if len(cols)>2 or len(rows) >2:
        colarr = cols
        rowarr = rows
        nspax = len(cols)

    return nspax, colarr, rowarr


# handle the FITLOOP execution.
# In its own function due to commonality between single- and
# multi-threaded execution
def execute_fitloop(nspax, colarr, rowarr, cube, q3di, linelist, specConv,
                    onefit, quiet, logfile=None):
    print(nspax)
    for ispax in range(0, nspax):
        fitloop(ispax, colarr, rowarr, cube, q3di, linelist, specConv,
                onefit, quiet, logfile=logfile)


def q3df_oneCore(initobj, cols=None, rows=None, onefit=False,
                 quiet=True):
    '''
    q3df setup for multi-threaded execution

    Parameters
    ----------
    initobj : TYPE
        DESCRIPTION.
    cols : TYPE, optional
        DESCRIPTION. The default is None.
    rows : TYPE, optional
        DESCRIPTION. The default is None.
    onefit : TYPE, optional
        DESCRIPTION. The default is False.
    quiet : TYPE, optional
        DESCRIPTION. The default is True.

    Returns
    -------
    None.

    '''
    # add common subdirectory to Python PATH for ease of importing
    path.append("common/")
    starttime = time.time()

    q3di = __get_q3di(initobj)
    linelist = __get_linelist(q3di)

    if q3di.logfile is not None:
        logfile = open(q3di.logfile, 'w+')
    else:
        logfile = None

    cube, vormap = __get_Cube(q3di, quiet, logfile=logfile)
    specConv = __get_dispersion(q3di, cube, quiet=quiet)

    if cols and rows and vormap:
        cols = __get_voronoi(cols, rows, vormap)
        rows = 1
    nspax, colarr, rowarr = __get_spaxels(cube, cols, rows)

    # execute FITLOOP

    execute_fitloop(nspax, colarr, rowarr, cube, q3di, linelist, specConv,
                    onefit, quiet, logfile=logfile)

    if logfile is None:
        from sys import stdout
        logtmp = stdout
    else:
        logtmp = logfile
    timediff = time.time()-starttime
    print(f'Q3DF: Total time for calculation: {timediff:.2f} s.',
          file=logtmp)
    if logfile is not None:
        logfile.close()


def q3df_multiCore(rank, initobj, cols=None, rows=None,
                   onefit=False, ncores=1, quiet=True):
    '''
    q3df setup for multi-threaded execution

    Parameters
    ----------
    rank : TYPE
        DESCRIPTION.
    initobj : TYPE
        DESCRIPTION.
    cols : TYPE, optional
        DESCRIPTION. The default is None.
    rows : TYPE, optional
        DESCRIPTION. The default is None.
    onefit : TYPE, optional
        DESCRIPTION. The default is False.
    ncores : TYPE, optional
        DESCRIPTION. The default is 1.
    quiet : TYPE, optional
        DESCRIPTION. The default is True.

    Returns
    -------
    None.

    '''
    starttime = time.time()
    # If it's a string, assume it's an input .npy file
    if type(initobj) == str:
        q3diarr = np.load(initobj, allow_pickle=True)
        q3di = q3diarr[()]
    # If it's an ndarray, assume the file's been loaded but not stripped
    # to dict{}
    elif isinstance(initobj, np.ndarray):
        q3di = initobj[()]
    # If it's a dictionary, assume all is well
    elif isinstance(initobj, dict):
        q3di = initobj
    linelist = __get_linelist(q3di)

    if 'logfile' in q3di:
        logfile = open(q3di['logfile'] + '_core'+str(rank+1), 'w+')
    else:
        logfile = None

    cube, vormap = __get_Cube(q3di, quiet, logfile=logfile)
    specConv = __get_dispersion(q3di, cube, quiet=quiet)
    if cols and rows and vormap:
        cols = __get_voronoi(cols, rows, vormap)
        rows = 1
    nspax, colarr, rowarr = __get_spaxels(cube, cols, rows)
    # get the range of spaxels this core is responsible for
    start = int(np.floor(nspax * rank / size))
    stop = int(np.floor(nspax * (rank+1) / size))
    colarr = colarr[start:stop]
    rowarr = rowarr[start:stop]
    # number of spaxels THIS CORE is responsible for
    nspax_thisCore = stop-start
    # execute FITLOOP
    execute_fitloop(nspax_thisCore, colarr, rowarr, cube, q3di,
                    linelist, specConv, onefit, quiet, logfile=logfile)
    if logfile is None:
        from sys import stdout
        logtmp = stdout
    else:
        logtmp = logfile
    timediff = time.time()-starttime
    print(f'Q3DF: Total time for calculation: {timediff:.2f} s.',
          file=logtmp)
    if logfile is not None:
        logfile.close()


# if called externally, default to MPI behavior
if __name__ == "__main__":
    # get multiprocessor data: number of tasks and which one this is
    comm = MPI.COMM_WORLD
    size = comm.Get_size()
    rank = comm.Get_rank()
    # helper function: convert a string representing a list of integers
    # of form [1, 2, 3...] or [1,2,3...] to an actual list of integers

    def string_to_intArray(strArray):
        if strArray.startswith("N"):
            return None
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

    # convert strings from command-line arguments to usable Python data
    initobj = argv[1]
    cols = string_to_intArray(argv[2])
    rows = string_to_intArray(argv[3])
    if argv[4].startswith("T"):
        onefit = True
    else:
        onefit = False
    if argv[5].startswith("T"):
        quiet = True
    else:
        quiet = False
    q3df_multiCore(rank, initobj, cols, rows, onefit, size, quiet)
