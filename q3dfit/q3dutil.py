#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Utility functions that are used by multiple modules in the q3dfit package.
"""
from __future__ import annotations

from typing import Optional
import numpy as np
from q3dfit.exceptions import InitializationError


def get_q3dio(inobj: str | object) -> object:
    '''
    Load initialization or output object. Determine whether it's already an object,
    or needs to be loaded from file.

    Parameters
    ----------
    inobj
        Input object. If a string, assume it's an input .npy file. If an ndarray,
        assume the file's been loaded but not stripped to dict{}. If an object,
        assume all is well.

    Raises
    ------
    InitializationError

    Returns
    -------
    object
        :py:class:`~q3dfit.q3din.q3din` or :py:class:`~q3dfit.q3dout.q3dout` object.

    '''

    # If it's a string, assume it's an input .npy file
    if type(inobj) == str:
        q3dioarr = np.load(inobj, allow_pickle=True) # type: ignore
        q3dio = q3dioarr[()]
    # If it's an ndarray, assume the file's been loaded but not stripped
    # to dict{}
    elif isinstance(inobj, np.ndarray):
        q3dio = inobj[()]
    # If it's an object, assume all is well
    elif isinstance(inobj, object):
        q3dio = inobj
    else:
        raise InitializationError('q3di/o not in expected format')

    return q3dio


def get_spaxels(ncols: int,
                nrows: int,
                cols: Optional[int | list | np.ndarray]=None,
                rows: Optional[int | list | np.ndarray]=None) -> \
                tuple[int, np.ndarray[int], np.ndarray[int]]:
    '''
    Set up 1D arrays specifying column value and row value for each spaxel to be
    fitted. These are zero-offset for indexing other arrays.

    Parameters
    ----------
    ncols
        Total number of columns in the data cube.
    nrows
        Total number of rows in the data cube.
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

    Returns
    -------
    int
        Number of spaxels to be fitted.
    np.ndarray[int]
        1D array of column values for spaxels to be fitted.
    np.ndarray[int]
        1D array of row values for spaxels to be fitted.
    '''
    # Set up 2-element arrays with starting and ending columns/rows
    # These are unity-offset to reflect pixel labels
    if not cols:
        cols = [1, ncols]
        ncols = ncols
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
        rows = [1, nrows]
        nrows = nrows
    elif not isinstance(rows, (list, np.ndarray)):
        rows = [rows, rows]
        nrows = 1
    elif len(rows) == 1:
        rows = [rows[0], rows[0]]
        nrows = 1
    else:
        nrows = rows[1]-rows[0]+1

    if len(cols)<=2 or len(rows) <= 2:
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

    if len(cols) > 2 or len(rows) > 2:
        colarr = cols
        rowarr = rows
        nspax = len(cols)

    return nspax, colarr, rowarr


'''
def get_voronoi(cols, rows, vormap):
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

    if len(cols) == 1 and len(rows) == 1:
        cols = vormap[cols[0]-1, rows[0]-1]
        return cols
    else:
        raise ValueError('Q3DF: ERROR: Can only specify 1 spaxel, \
                         or all spaxels, in Voronoi mode.')
'''


def write_msg(message: str,
              file: Optional[str]=None,
              quiet: bool=True
              ):
    '''
    Write message to log file and/or :py:func:`~sys.stdout`.

    Parameters
    ----------
    message
        Message to be written
    logfile
        Optional. Filename for progress messages. Default is None, which
        means no message is written to a file.
    quiet
        Optional. Suppress progress messages to stdout. Default is True.
    '''
    if file is not None:
        with open(file, 'a') as log:
            log.write(message)
    if not quiet:
        print(message)


class lmlabel():
    """
    Remove characters from a label string that are incompatible with LMFIT's
    parser; or reverse the operation.

    All keys of a Parameters() instance must be strings and valid Python symbol
    names. This class replaces incompatible characters with valid symbols.

    https://lmfit.github.io/lmfit-py/parameters.html#lmfit.parameter.Parameters

    Parameters
    ----------
    label
        Input label string.
    reverse
        Optional. If set, reverse the operation. Default is False.

    Attributes
    ----------
    label
        Original label string.
    lmlabel
        Label string with incompatible characters replaced by valid symbols.
    """
    def __init__(self,
                 label: str,
                 reverse: bool=False):

        if reverse:
            lmlabel = label
            origlabel = label.replace('lb', '[').replace('rb', ']').\
                replace('pt', '.')
        else:
            origlabel = label
            lmlabel = label.replace('[', 'lb').replace(']', 'rb').\
                replace('.', 'pt')
        self.label = origlabel
        self.lmlabel = lmlabel

# class lmpar():
#
#    def __init__(self, lmlabel, comp, partype):
#        self.parname = f'{lmlabel}_c{comp}_g{partype}'
