#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
This procedure contains the helper functions for the file q3df.py. Its purpose is
to fit the continuum and emission lines of a spectrum.
As input, it requires a structure of initialization parameters.
The tags for this structure can be found in INITTAGS.txt.

Returns
-------
IDL save file (.xdr)

Parameters
----------
initproc: in, required, type=string
    Name of procedure to initialize the fit.
cols: in, optional, type=intarr, default=all
       Columns to fit, in 1-offset format. Either a scalar or a
       two-element vector listing the first and last columns to fit.
     ncores: in, optional, type=int, default=1
       Number of cores to split processing over.
     rows: in, optional, type=intarr, default=all
       Rows to fit, in 1-offset format. Either a scalar or a
       two-element vector listing the first and last rows to fit.
     onefit: in, optional, type=byte
       Option to skip second fit  primarily for testing.
     verbose: in, optional, type=byte
       Print error and progress messages. Propagates to most/all
       subroutines.

History
    2020may21, DSNR, copied header from IFSF.pro
    2020jun29, canicetti, refoactored into helper functions and renamed q3df_helperfunctons.py
    2022feb01, YI, added additional functions for wavelength convolution initialization

Created on Tue May 26 13:37:58 2020

@author: drupke
@author: canicetti
"""

__author__ = 'Q3D Team'
__credits__ = ['David S. N. Rupke', "Carlos Anicetti"]
__created__ = '2020 May 26'
__last_modified__ = '2020 Jun 29'


#   Get linelist
def __get_linelist(initdat):
    from q3dfit.linelist import linelist
    if initdat.__contains__('lines'):
        if initdat.__contains__('argslinelist'):
            linelist = linelist(initdat['lines'], **initdat['argslinelist'])
        else:
            linelist = linelist(initdat['lines'])
    else:
        linelist = linelist()
    return linelist


# read in the dispersion list and save to memory
# default return value is None (no convolution)
def __get_dispersion(initdat, cube, quiet=True):
    from q3dfit import spectConvol
    if 'spect_convol' not in initdat:
        return None
    elif initdat['spect_convol'] is not None:
        return spectConvol.spectConvol(initdat, cube, quiet=quiet)
    else:
        return None


# initialize Cube object
def __get_Cube(initdat, quiet, logfile=None):
    from q3dfit.readcube import Cube

#   Read data
#   Set default extensions
    if not initdat.__contains__('datext'):
        datext = 1
    else:
        datext = initdat['datext']
    if not initdat.__contains__('varext'):
        varext = 2
    else:
        varext = initdat['varext']
    if not initdat.__contains__('dqext'):
        dqext = 3
    else:
        dqext = initdat['dqext']
#   Check for additional arguments
    if not initdat.__contains__('vormap'):
        vormap = False
    else:
        vormap = initdat['vormap']
    if logfile is None:
        from sys import stdout
        logfile = stdout
    if initdat.__contains__('argsreadcube'):
        # datext, varext, and dqext can be specified through
        # argsreadcube or through initdat directly. Default to the value
        # in argsreadcube. Else set the value based on the default or from
        # initdat
        argsreadcubeuse = initdat['argsreadcube'].copy()
        if not argsreadcubeuse.__contains__('datext'):
            argsreadcubeuse['datext'] = datext
        if not argsreadcubeuse.__contains__('varext'):
            argsreadcubeuse['varext'] = varext
        if not argsreadcubeuse.__contains__('dqext'):
            argsreadcubeuse['dqext'] = dqext
    else:
        argsreadcubeuse = {'datext': datext, 'varext': varext, 'dqext': dqext}

    cube = Cube(initdat['infile'], quiet=quiet, vormap=vormap,
                logfile=logfile, **argsreadcubeuse)

    return cube, vormap


# construct voronoi map
def __get_voronoi(cols, rows, vormap):
    # Voronoi binned case
    if len(cols) == 1 and len(rows) == 1:
        cols = vormap[cols[0]-1, rows[0]-1]
        return cols
    else:
        raise ValueError('Q3DF: ERROR: Can only specify 1 spaxel, \
                         or all spaxels, in Voronoi mode.')


# Set up 1D arrays specifying column value and row value at each point to be
# fitted. These are zero-offset for indexing other arrays.
def __get_spaxels(cube, cols=None, rows=None):
    import numpy as np
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
def execute_fitloop(nspax, colarr, rowarr, cube, initdat, linelist, specConv,
                    onefit, quiet, logfile=None):
    from q3dfit.fitloop import fitloop
    print(nspax)
    for ispax in range(0, nspax):
        fitloop(ispax, colarr, rowarr, cube, initdat, linelist, specConv,
                onefit, quiet, logfile=logfile)


# q3df setup for single-threaded execution
def q3df_oneCore(initproc, cols=None, rows=None, onefit=False,
                 quiet=True):
    import numpy as np
    import time
    from sys import path
    from q3dfit.exceptions import InitializationError
    # add common subdirectory to Python PATH for ease of importing
    path.append("common/")
    starttime = time.time()

    # If it's a string, assume it's an input .npy file
    if type(initproc) == str:
        initdatarr = np.load(initproc, allow_pickle=True)
        initdat = initdatarr[()]
    # If it's an ndarray, assume the file's been loaded but not stripped
    # to dict{}
    elif isinstance(initproc, np.ndarray):
        initdat = initproc[()]
    # If it's a dictionary, assume all is well
    elif isinstance(initproc, dict):
        initdat = initproc
    else:
        raise InitializationError('initproc not in expected format')
    linelist = __get_linelist(initdat)

    if 'logfile' in initdat:
        logfile = open(initdat['logfile'], 'w+')
    else:
        logfile = None

    cube, vormap = __get_Cube(initdat, quiet, logfile=logfile)
    specConv = __get_dispersion(initdat, cube, quiet=quiet)

    if cols and rows and vormap:
        cols = __get_voronoi(cols, rows, vormap)
        rows = 1
    nspax, colarr, rowarr = __get_spaxels(cube, cols, rows)

    # execute FITLOOP
    execute_fitloop(nspax, colarr, rowarr, cube, initdat, linelist, specConv,
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



# q3df setup for multi-threaded execution
def q3df_multiCore(rank, initproc, cols=None, rows=None,
                   onefit=False, ncores=1, quiet=True):

    import numpy as np
    import time
    starttime = time.time()
    # If it's a string, assume it's an input .npy file
    if type(initproc) == str:
        initdatarr = np.load(initproc, allow_pickle=True)
        initdat = initdatarr[()]
    # If it's an ndarray, assume the file's been loaded but not stripped
    # to dict{}
    elif isinstance(initproc, np.ndarray):
        initdat = initproc[()]
    # If it's a dictionary, assume all is well
    elif isinstance(initproc, dict):
        initdat = initproc
    linelist = __get_linelist(initdat)

    if 'logfile' in initdat:
        logfile = open(initdat['logfile'] + '_core'+str(rank+1), 'w+')
    else:
        logfile = None

    cube, vormap = __get_Cube(initdat, quiet, logfile=logfile)
    specConv = __get_dispersion(initdat, cube, quiet=quiet)
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
    execute_fitloop(nspax_thisCore, colarr, rowarr, cube, initdat,
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
    from sys import argv
    from mpi4py import MPI
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
    initproc = argv[1]
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
    q3df_multiCore(rank, initproc, cols, rows, onefit, size, quiet)
