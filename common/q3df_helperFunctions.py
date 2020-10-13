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
     oned: in, optional, type=byte
       Data is assumed to be in a 2d array  choose this switch to
       input data as a 1d array.
     onefit: in, optional, type=byte
       Option to skip second fit  primarily for testing.
     verbose: in, optional, type=byte
       Print error and progress messages. Propagates to most/all
       subroutines.

History
    2020may21, DSNR, copied header from IFSF.pro
    2020jun29, canicetti, refoactored into helper functions and renamed q3df_helperfunctons.py 

Created on Tue May 26 13:37:58 2020

@author: drupke
@author: canicetti
"""

__author__ = 'Q3D Team'
__credits__ = ['David S. N. Rupke', "Carlos Anicetti"]
__created__ = '2020 May 26'
__last_modified__ = '2020 Jun 29'


#   This functon reads in the dictionary from the master initialization file. 
#   The multi-step process is because initproc is a string variable. The
#   initialization file must be in the init subdirectory of the Q3DFIT
#   distribution for this to work. There may be a better way with an 
#   arbitrary path.
def __get_initdat(initproc):
#   add folder "init" to python PATH variable, to avoid importing beyond top-level package
    from sys import path
    import importlib
    path.append("init")
    module = importlib.import_module("q3dfit.init." + initproc)
    fcninitproc = getattr(module,initproc)    
    return fcninitproc()


#   Get linelist
def __get_linelist(initdat):
    from q3dfit.common.linelist import linelist
    if initdat.__contains__('lines'):
        if initdat.__contains__('argslinelist'):
            linelist=linelist(initdat['lines'],**initdat['argslinelist'])
        else: linelist=linelist(initdat['lines'])
    else: linelist=linelist()
    return linelist


# initialize CUBE object
def __get_CUBE(initdat, oned, quiet):
    from q3dfit.common.readcube import CUBE
#   Read data
#   Set default extensions
    if not initdat.__contains__('datext'): datext=1
    else: datext=initdat['datext']
    if not initdat.__contains__('varext'): varext=2
    else: varext=initdat['varext']
    if not initdat.__contains__('dqext'): dqext=3
    else: dqext=initdat['dqext']
#   Check for additional arguments
    if not initdat.__contains__('vormap'): vormap=False
    else: vormap=initdat['vormap']
    if initdat.__contains__('argsreadcube'):
        cube = CUBE(infile=initdat['infile'],datext=datext,dqext=dqext,\
                    oned=oned,quiet=quiet,varext=varext,vormap=vormap,\
                    **initdat['argsreadcube'])
    else:
        cube = CUBE(infile=initdat['infile'],datext=datext,dqext=dqext,\
                    oned=oned,quiet=quiet,varext=varext,vormap=vormap)
    return cube, vormap
        

# construct voronoi map
def __get_voronoi(cols, rows, vormap):
#   Voronoi binned case
    if len(cols) == 1 and len(rows) == 1:
        cols=vormap[cols[0]-1,rows[0]-1]
        return cols
    else:
        raise ValueError('Q3DF: ERROR: Can only specify 1 spaxel, or all spaxels, \
              in Voronoi mode.')


# Set up 1D arrays specifying column value and row value at each point to be
# fitted. These are zero-offset for indexing other arrays.
def __get_spaxels(cube, cols=None, rows=None):
    import numpy as np
    # Set up 2-element arrays with starting and ending columns/rows
    # These are unity-offset to reflect pixel labels
    if not cols:
        cols=[1,cube.ncols]
        ncols = cube.ncols
    elif len(cols) == 1:
        cols = [cols[0], cols[0]]
        ncols = 1
    else:
        ncols = cols[1]-cols[0]+1
    if not rows:
        rows=[1,cube.nrows]
        nrows = cube.nrows
    elif len(rows) == 1:
        rows = [rows[0], rows[0]]
        nrows = 1
    else:
        nrows = rows[1]-rows[0]+1
    colarr = np.empty((ncols,nrows), dtype=np.int32)
    rowarr = np.empty((ncols,nrows), dtype=np.int32)
    for i in range(nrows): colarr[:,i] = np.arange(cols[0]-1,cols[1], dtype=np.int32)
    for i in range(ncols): rowarr[i,:] = np.arange(rows[0]-1,rows[1], dtype=np.int32)
    # Flatten from 2D to 1D arrays to preserve indexing using only ispax
    # currently not needed. fitloop expects 2D lists.
    #colarr = colarr.flatten()
    #rowarr = rowarr.flatten()
    nspax = ncols * nrows
    return nspax, colarr, rowarr
    

# handle the FITLOOP execution. 
# In its own function due to commonality between single- and multi-threaded execution
def execute_fitloop( nspax, colarr, rowarr, cube, initdat, linelist, oned, onefit,\
        quiet):
    from q3dfit.common.fitloop import fitloop
    #from fitloop import fitloop
    dolog = 0
    if "logfile" in initdat:
        dolog = 1
        logloop = []
        for i in range(nspax): logloop.append(initdat["logfile"])
    for ispax in range(0, nspax):
        if ispax == 0 and dolog: 
            from os import remove
            # delete log file, if it exists
            try: remove(initdat["logfile"])
            except FileNotFoundError: pass
        # TODO: delete next line when fitloop has been written
        print(ispax, "["+str(rowarr[ispax])+", " + str(colarr[ispax]) + "]")
        # TODO: Uncomment this command once fitloop has been written
        fitloop(ispax, colarr, rowarr, cube, initdat, linelist,\
                            oned, onefit, quiet, logfile=logloop)

# q3df setup for single-threaded execution
def q3df_oneCore( initproc, cols=None, rows=None, oned=False, onefit=False, quiet=True ):
    import pdb
    import time
    from sys import path
    # add common subdirectory to Python PATH for ease of importing
    path.append("common/")
    starttime = time.time()
    initdat = __get_initdat(initproc)
    linelist = __get_linelist(initdat)
    cube, vormap = __get_CUBE(initdat, oned, quiet)
    if cols and rows and vormap:
        cols = __get_voronoi(cols, rows, vormap)
        rows=1
    nspax, colarr, rowarr = __get_spaxels(cube, cols, rows)
    # execute FITLOOP
    execute_fitloop(nspax, colarr, rowarr, cube, initdat, linelist, oned, onefit, quiet)
    print('Q3DF: Total time for calculation: '+ str(time.time()-starttime) + ' s.')


# q3df setup for multi-threaded execution
def q3df_multiCore( rank, initproc, cols=None, rows=None, oned=False, onefit=False,\
          quiet=True ):
    import pdb
    import time
    from numpy import floor
    # TODO: modify import statements on __get functions to account for invoking from a different folder in multicore case
    multicore=True
    if rank == 0: starttime = time.time()
    initdat = __get_initdat(initproc)
    linelist = __get_linelist(initdat)
    cube, vormap = __get_CUBE(initdat, oned, quiet)
    if cols and rows and vormap:
        cols = __get_voronoi(cols, rows, vormap)
        rows=1
    nspax, colarr, rowarr = __get_spaxels(cube, cols, rows)
    # get the range of spaxels this core is responsible for
    start = int(floor(nspax * rank / size))
    stop = int(floor(nspax * (rank+1) / size))
    colarr = colarr[start:stop]
    rowarr = rowarr[start:stop]
    print("RANK " + str(rank) + ":")
    print(start, stop)
    print(colarr)
    print(rowarr)
    # number of spaxels THIS CORE is responsible for
    nspax_thisCore = stop-start
    # execute FITLOOP
    execute_fitloop(nspax_thisCore, colarr, rowarr, cube, initdat, linelist, oned, onefit, quiet)
    if rank == 0: print('Q3DF: Total time for calculation: '+ str(time.time()-starttime) + ' s.')


# if called externally, default to MPI behavior
if __name__ == "__main__":
    from sys import argv
    from mpi4py import MPI
    from sys import path
    path.append("../..")
    import q3dfit 
    # get multiprocessor data: number of tasks and which one this is
    comm = MPI.COMM_WORLD
    size = comm.Get_size()
    rank = comm.Get_rank()
    # helper function: convert a string representing a list of integers 
    # of form [1, 2, 3...] or [1,2,3...] to an actual list of integers
    def string_to_intArray(strArray):
        if strArray.startswith("N"): return None
        # strip leading and trailing brackets
        strArray = strArray[1:-1]
        # form an list by splitting on commas
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
    if argv[4].startswith("T"): oned=True
    else: oned=False
    if argv[5].startswith("T"): onefit=True
    else: onefit=False
    if argv[6].startswith("T"): quiet=True
    else: quiet=False
    q3df_multiCore(rank, initproc, cols, rows, oned, onefit, quiet)
