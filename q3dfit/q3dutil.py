#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Nov 10 20:06:50 2022

@author: drupke
"""
import numpy as np

from q3dfit import spectConvol
from q3dfit.exceptions import InitializationError
from q3dfit.linelist import linelist
from q3dfit.readcube import Cube


def get_linelist(q3di):
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
    vacuum = q3di.vacuum
    if hasattr(q3di, 'lines'):
        listlines = linelist(q3di.lines, vacuum=vacuum, **q3di.argslinelist)
    else:
        listlines = [] # linelist(q3di.lines, vacuum=vacuum, **q3di.argslinelist)
    return listlines


def get_dispersion(q3di):
    '''
    Instantiate spectConvol object with dispersion information for selected
    gratings. Return value is None (no convolution) if q3di.spect_convol is
    empty.

    Parameters
    ----------
    q3di : object

    Returns
    -------
    spectConvol : object

    '''
    if not q3di.spect_convol:
        return None
    else:
        return spectConvol.spectConvol(q3di.spect_convol)


def get_Cube(q3di, quiet=True, logfile=None):
    '''
    instantiate Cube object


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

    cube = Cube(q3di.infile, quiet=quiet,
                logfile=logfile, datext=q3di.datext, varext=q3di.varext,
                dqext=q3di.dqext, vormap=q3di.vormap, **q3di.argsreadcube)

    return cube, q3di.vormap


def get_q3dio(inobj):
    '''
    Load initialization or output object. Determine whether it's already an object,
    or needs to be loaded from file.

    Parameters
    ----------
    q3dio : string or object

    Raises
    ------
    InitializationError
        DESCRIPTION.

    Returns
    -------
    q3di/o object

    '''

    # If it's a string, assume it's an input .npy file
    if type(inobj) == str:
        q3dioarr = np.load(inobj, allow_pickle=True)
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

    return(q3dio)


def get_voronoi(cols, rows, vormap):
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


def get_spaxels(cube, cols=None, rows=None):
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


class lmlabel():
    """
Created on Wed Aug 25 14:07:30 2021

@author: drupke

Remove characters from a label string that are incompatible with LMFIT's
parser; or reverse the operation.

"All keys of a Parameters() instance must be strings and valid Python symbol
names, so that the name must match [a-z_][a-z0-9_]* and cannot be a Python
reserved word."

https://lmfit.github.io/lmfit-py/parameters.html#lmfit.parameter.Parameters

    """
    def __init__(self, label, reverse=False):
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
