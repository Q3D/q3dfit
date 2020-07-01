#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
This procedure is the core routine to fit the continuum and emission lines of 
a spectrum. As input, it requires a structure of initialization parameters.
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

Created on Tue May 26 13:37:58 2020

@author: drupke
"""

__author__ = 'Q3D Team'
__credits__ = ['David S. N. Rupke']
__created__ = '2020 May 26'
__last_modified__ = '2020 Jun 01'

def q3df( initproc, cols=None, rows=None, oned=False, onefit=False, \
          quiet=True ):
    
    import importlib
    import numpy as np
    import pdb
    import time

    from q3dfit.common.fitloop import fitloop
    from q3dfit.common.linelist import linelist
    from q3dfit.common.readcube import CUBE
    from q3dfit.exceptions import InitializationError
    
    starttime = time.time()
    
#   This block reads in the dictionary from the master initialization file. 
#   The multi-step process is because initproc is a string variable. The
#   initialization file must be in the init subdirectory of the Q3DFIT
#   distribution for this to work. There may be a better way with an 
#   arbitrary path.
    module = importlib.import_module('q3dfit.init.'+initproc)
    fcninitproc = getattr(module,initproc)    
    initdat = fcninitproc()

#   Get linelist
    if initdat.__contains__('lines'):
        if initdat.__contains__('argslinelist'):
            linelist=linelist(initdat['lines'],**initdat['argslinelist'])
        else: linelist=linelist(initdat['lines'])
    else:
#   This deviates from IFSFIT, but really need to throw an error here
#   because fitloop expects one to specify lines in the initialization file
        raise InitializationError('No lines to fit specified')
    
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
        
#   Loop through spaxels

#   Voronoi binned case
    if cols and rows and vormap:
        if len(cols) == 1 and len(rows) == 1:
            cols=vormap[cols[0]-1,rows[0]-1]
            rows=1
        else:
            print('Q3DF: ERROR: Can only specify 1 spaxel, or all spaxels, \
                  in Voronoi mode.')

#   Set up 2-element arrays with starting and ending columns/rows
#   These are unity-offset to reflect pixel labels
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

#   Set up 2D arrays specifying column value and row value at each point to be
#   fitted. These are zero-offset for indexing other arrays.
    colarr = np.empty((ncols,nrows),dtype=int)
    rowarr = np.empty((ncols,nrows),dtype=int)
    for i in range(nrows):
        colarr[:,i] = list(range(cols[0]-1,cols[1]))
    for i in range(ncols):
        rowarr[i,] = list(range(rows[0]-1,rows[1]))
    nspax = ncols * nrows
    
#   Call to FITLOOP or parallelization goes here.
    
#   Test call
    fitloop(0,colarr,rowarr,cube,initdat,linelist,oned,onefit,quiet,\
            logfile=initdat['logfile'])

    print('Q3DF: Total time for calculation: '+str(time.time()-starttime)+' s.')