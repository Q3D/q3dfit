#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Helper functions for q3df.py.

Created on Tue May 26 13:37:58 2020

@author: drupke
@author: canicetti
"""
import time
import numpy as np

from q3dfit.fitloop import fitloop
import q3dfit.q3dutil as q3dutil

def execute_fitloop(nspax, colarr, rowarr, cube, q3di, linelist, specConv,
                    onefit, quiet, logfile=None, core=1, nocrash=False):

    '''
    handle the FITLOOP execution.
    In its own function due to commonality between single- and
    multi-threaded execution


    Parameters
    ----------
    nspax : TYPE
        DESCRIPTION.
    colarr : TYPE
        DESCRIPTION.
    rowarr : TYPE
        DESCRIPTION.
    cube : TYPE
        DESCRIPTION.
    q3di : TYPE
        DESCRIPTION.
    linelist : TYPE
        DESCRIPTION.
    specConv : TYPE
        DESCRIPTION.
    onefit : TYPE
        DESCRIPTION.
    quiet : TYPE
        DESCRIPTION.
    logfile : TYPE, optional
        DESCRIPTION. The default is None.

    Returns
    -------
    None.

    '''
    print('Core '+str(core)+': # spaxels fit='+str(nspax))
    for ispax in range(0, nspax):
        if nocrash:     # In case of crash, just continue with the next spaxel
            try:
                fitloop(ispax, colarr, rowarr, cube, q3di, linelist, specConv,
                            onefit=onefit, quiet=quiet, logfile=logfile)
            except:
                continue
        else:           # Regular run; no continuation in case of crash
            fitloop(ispax, colarr, rowarr, cube, q3di, linelist, specConv,
                    onefit=onefit, quiet=quiet, logfile=logfile)

def q3df_oneCore(inobj, cols=None, rows=None, onefit=False,
                 quiet=True, nocrash=False):


    '''
    q3df setup for multi-threaded execution

    Parameters
    ----------
    inobj : TYPE
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
    starttime = time.time()
    q3di = q3dutil.get_q3dio(inobj)
    linelist = q3dutil.get_linelist(q3di)

    if q3di.logfile is not None:
        logfile = open(q3di.logfile, 'w+')
    else:
        logfile = None

    cube, vormap = q3dutil.get_Cube(q3di, quiet=quiet, logfile=logfile)
    specConv = q3dutil.get_dispersion(q3di)
    if cols and rows and vormap:
        cols = q3dutil.get_voronoi(cols, rows, vormap)
        rows = 1
    nspax, colarr, rowarr = q3dutil.get_spaxels(cube, cols, rows)
    # execute FITLOOP
    execute_fitloop(nspax, colarr, rowarr, cube, q3di,
                    linelist, specConv, onefit, quiet, logfile=logfile,
                    nocrash=nocrash)
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


def q3df_multiCore(rank, size, inobj, cols=None, rows=None,
                   onefit=False, ncores=1, quiet=True):

    '''
    q3df setup for multi-threaded execution

    Parameters
    ----------
    rank : TYPE
        DESCRIPTION.
    inobj : TYPE
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
    q3di = q3dutil.get_q3dio(inobj)
    linelist = q3dutil.get_linelist(q3di)

    if q3di.logfile is not None:
        logfile = open(q3di.logfile + '_core'+str(rank+1), 'w+')
    else:
        logfile = None

    cube, vormap = q3dutil.get_Cube(q3di, quiet=quiet, logfile=logfile)
    specConv = q3dutil.get_dispersion(q3di)
    if cols and rows and vormap:
        cols = q3dutil.get_voronoi(cols, rows, vormap)
        rows = 1
    nspax, colarr, rowarr = q3dutil.get_spaxels(cube, cols, rows)
    # get the range of spaxels this core is responsible for
    start = int(np.floor(nspax * rank / size))
    stop = int(np.floor(nspax * (rank+1) / size))
    colarr = colarr[start:stop]
    rowarr = rowarr[start:stop]
    # number of spaxels THIS CORE is responsible for
    nspax_thisCore = stop-start
    # execute FITLOOP
    execute_fitloop(nspax_thisCore, colarr, rowarr, cube, q3di,
                    linelist, specConv, onefit, quiet, logfile=logfile,
                    core=rank+1)
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
    inobj = argv[1]
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
    q3df_multiCore(rank, size, inobj, cols, rows, onefit, quiet)
