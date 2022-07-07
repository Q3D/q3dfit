#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jun 15 12:07:16 2022

@author: ryanmccrory
"""

import numpy as np
from astropy.io import ascii
from astropy.table import Table, vstack    
from astropy import units as u
from pathlib import Path

def manualline(z, gal, lamb_min, lamb_max, vacuum=True, waveunit='micron'):
    """
    Similar to jwstline(), manuallines() produces a table with emission lines in the provided range. 
    Unlike jwstline(), this function can solve for a user-specified range of wavelengths and can account for 
        air wavelength conversions.  
    
    Parameters
    ----------
    
    z : flt, required
        Galaxy redshift
    gal : str, required
        Galaxy name for filenaming
    lamb_min : flt, required
        minimum wavelength value of instrument, units determined by waveunit value
    lamb_max : flt, required
        maximum wavelength of instrument, units determined by waveunit value
    vacuum : bool, optional, default = True
        if false, enables conversion to air wavelengths
    waveunit : str, optional, default = 'micron'
        determines the unit of input wavelengths and output table
        
    Returns
    --------
        
    lines : astropy table
        An astropy table of emission lines with keywords 'name', 'lines', q'linelab', 'observed'
        Example: H2_43_Q6, 2.98412, H$_2$(4-3) Q(6), 4.282212 
        Interanlly, everything is processed in microns, so filename inclues range values in microns. 
            
    """
        

 
 
    wu = waveunit.lower()
    home = str(Path.home())
 
    if ((wu!='angstrom') & (wu!='micron')):
        print("possible waveunit inputs are 'micron' or 'angstrom'")
        print ('Wave unit ',waveunit,' not recognized, returning microns')
        sig = 7
    elif (wu == 'angstrom'):
        # converting A input to microns
        lamb_max = lamb_max / 1.e4
        lamb_min = lamb_min / 1.e4
        sig = 3
    else:
        sig = 7
     
        
    #add links to more tables here
    lines_H2 = Table.read(home + '/q3dfit/q3dfit/data/linelists/linelist_H2.tbl',format='ipac')
    lines_DSNR = Table.read(home + '/q3dfit/q3dfit/data/linelists/linelist_DSNR_micron.tbl',format='ipac')
    lines_fine_str = Table.read(home + '/q3dfit/q3dfit/data/linelists/linelist_fine_str.tbl',format='ipac')
    lines_TSB = Table.read(home + '/q3dfit/q3dfit/data/linelists/linelist_TSB.tbl',format='ipac')     

    lines = vstack([lines_H2, lines_DSNR, lines_fine_str, lines_TSB])
    

 
    # unit conversion from program standard microns to angstroms
    # FIX THIS
    if (wu=='angstrom'):
        print('')
        print('angstroms unit selected')

    
    tcol = lines.columns[1]    

    
    if ((vacuum!=True) & (vacuum!=False)):
        print('Incorrect input for vacuum vs air. "' + vacuum + '" not recognized.') 
        print('proceeding with default vacuum = True')

    elif (vacuum == False):
        if (wu == 'angstrom'):
            temp=1.e4/(tcol*z)
        else: 
            temp = 1/(tcol*z)
            lines_air_z=lines['lines']/(1.+6.4328e-5 + 2.94981e-2/(146.-temp**2)+2.5540e-4/(41.-temp**2))
        #
    else:
        lines_air_z = np.multiply(tcol,z+1)

        #redshifts the table
        #rounds the floats in list CHANGE IF MORE PRECISION NEEDED
        round_lines = lines_air_z.round(decimals = sig) * u.micron
        #adds column to lines table corresponding w/ redshifted wavelengths
        lines['observed'] = round_lines
 
 
     #helper function to identify lines in instrument range
    def inrange(table, key_colnames):
         colnames = [name for name in table.colnames if name  in key_colnames]
         for colname in colnames:
             if np.any(table[colname] < lamb_min) or np.any(table[colname] > lamb_max):
                 return False
             return True
 
 
    tg = lines.group_by('observed')
    lines_inrange = tg.groups.filter(inrange)
    
    if (wu == 'angstrom'):
        lines_inrange['lines'] = (lines_inrange['lines']*1.e4).round(decimals = sig)
        lines_inrange['lines'].unit = 'angstrom'     
        
        lines_inrange['observed'] = (lines_inrange['observed']*1.e4).round(decimals = sig)
        lines_inrange['observed'].unit = 'angstrom'        
    
    lmin = str(lamb_min)
    lmax = str(lamb_max)
    
    
    # If/else statement to determine length of list and stop the code of list_len = 0
    # Figure out how to get code to stop.
    
    list_len = len(lines_inrange['lines'])
    
    if (list_len == 0):
        print('')
        print('There are no emission lines in the provided sample')
        print('')
        print('Terminating table save...')
    else:
        print('')
        print('There are ' + str(list_len) + ' emission lines in the provided range.')
        
        ascii.write(lines_inrange, ('lines_' + gal + '_' + lmin + '_to_' + lmax + '.tbl'), format = 'ipac', overwrite=True)
        print('')
        print('File written as: "lines_' + gal + '_' + lmin + '_to_' + lmax + '.tbl"', sep='')
        print('under the directory : '+ home + 'stuff')
      
            