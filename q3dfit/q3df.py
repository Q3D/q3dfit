#!/usr/bin/env python
# -*- coding: utf-8 -*-

__credits__ = ['Carlos Anicetti', 'David Rupke']
__created__ = '2020 June 29'

from typing import Optional
import numpy as np

from . import q3din


def q3dfit(q3din: str | q3din.q3din,
           cols: Optional[int | list]=None,
           rows: Optional[int | list]=None,
           onefit: bool=False,
           ncores: int=1,
           quiet: bool=True,
           mpipath: Optional[str]=None,
           nocrash: bool=False):
    """
    This is the core routine to fit the continuum and emission lines of
    a spectrum. For single-threaded processing, the routine calls
    :py:mod:`~q3dfit.q3df_helperFunctions.q3df_oneCore`. For multi-threaded
    processing, the routine runs `:py:mod:`~q3dfit.q3df_helperFunctions`
    as a script through MPI. This in turn calls 
    :py:mod:`~q3dfit.q3df_helperFunctions.q3df_multiCore`.


    Parameters
    ----------
    q3din
        Either a string with the path to the numpy save file containing the
        input initialization object :py:class:`~q3dfit.q3din.q3din`, 
        or the object itself.
    cols
        Optional. Column values for spaxels to be fitted. Default is None, which means
        all columns will be fitted. If a scalar, only that column will be fitted. If a
        2-element list, the elements are the starting and ending columns to be
        fitted. Unity-offset values assumed.
    rows
        Optional. Row values for spaxels to be fitted. Default is None, which means
        all rows will be fitted. If a scalar, only that row will be fitted. If a
        2-element list, the elements are the starting and ending rows to be
        fitted. Unity-offset values assumed.
    onefit
        Optional. If set, only one fit is performed in :py:mod:`~q3dfit.fitloop.fitloop`. 
        Default is False.
    ncores
        Optional. Number of cores for parallel processing. Default is 1.
    quiet
        Optional. If False, some progress messages are written to stdout. 
        Default is True. If a logfile is specified in :py:class:`~q3dfit.q3din.q3din`,
        these messages are written to the logfile as well. Some messages appear only
        in the logfile.
    mpipath
        Optional. Path to the MPI executable. Default is None.
    nocrash
        Optional. If set, the routine will continue with the next spaxel in case 
        of a crash. Default is False.

    """

    # invoke the correct q3df helper function depending on whether this is to a
    # single or multi-threaded process
    if ncores == 1:
        from q3dfit.q3df_helperFunctions import q3df_oneCore
        q3df_oneCore(q3din, cols, rows, onefit, quiet, nocrash=nocrash)
    elif ncores > 1:
        from inspect import getfile
        from q3dfit import q3df_helperFunctions
        from subprocess import call
        # If a 2-element list, convert cols and rows to string of form "[1,2]".
        # Note no whitespace.
        strcols = str(cols)
        strcols = strcols.replace(" ", "")
        strrows = str(rows)
        strrows = strrows.replace(" ", "")
        filename = getfile(q3df_helperFunctions)
        # start a new MPI process since MPI cannot be started from within a
        # Python script
        mpistr = "mpiexec"
        if mpipath is not None:
            from os.path import join
            mpistr = join(mpipath, mpistr)
        import sys
        call([mpistr, "-n", str(ncores), "python", filename,
              q3din, strcols, strrows, str(onefit), str(quiet), str(nocrash)])
