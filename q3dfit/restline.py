#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jun 15 12:07:16 2022

"""

import numpy as np
from astropy.io import ascii
from astropy.table import Table, vstack    
from astropy import units as u
from pathlib import Path
import shutil

def restline(gal, lamb_min, lamb_max, waveunit='micron'):
    """
    Similar to jwstlinez() and  observedlinez(), restline() produces a table 
    with emission lines in the provided range. 
    Unlike the other 2 linelist functions, this function has REST wavelength
    lamb_min and lamb_max inputs. 
    This function is useful for identifying various emissions in a rest
    spectrum for fitting and analysis.
    In this way, the funtion operates a lot like a search function.  
    
    All input wavelengths are assumed to be in the REST FRAME!
    
    References stored under q3dfit/linelists are .tbl of filenames:
            
        lines_H2.tbl
        lines_DSNR_micron.tbl   
        lines_TSB.tbl
        lines_ref.tbl
        
    More tables can be added manually if saved in the linelists folder and 
    called in this function definition.
    
    
    Parameters
    ----------
    
    gal : str, required
        Galaxy name for filenaming. 
        Can also be customized for any other desired filename designations
    lamb_min : flt, required
        minimum <REST> wavelength value of instrument, units determined by 
            waveunit value
    lamb_max : flt, required
        maximum <REST> wavelength of instrument, units determined by waveunit value
    waveunit : str, optional, default = 'micron'
        determines the unit of input wavelengths and output table
        acceptable inputs are 'micron' and 'angstrom'
        
    Returns
    --------
        
    lines_...tbl : astropy table
        An astropy table of emission lines with keywords 'name', 'lines', 
        'linelab'
        Example Row: H2_43_Q6, 2.98412, H$_2$(4-3) Q(6), 4.282212 
        Interanlly, everything is processed in microns, so filename inclues 
        range values in microns. 
        The units of the table can be angstroms or microns, depending on the 
        entered value of waveunit.
        Output table contains comments descring data sources

            
    """
 
    wu = waveunit.lower()
    home = str(Path.home())
 
    if ((wu!='angstrom') & (wu!='micron')):
        print("possible waveunit inputs are 'micron' or 'angstrom'")
        print ('Wave unit ',waveunit,' not recognized, returning microns\n')
        sig = 7
    elif (wu == 'angstrom'):
        # converting A input to microns
        lamb_max = lamb_max / 1.e4
        lamb_min = lamb_min / 1.e4
        sig = 3
        print('Angstroms unit selected\n')
    else:
        sig = 7
     
        
    #add directories to more tables here
    #--------------------------------------------------------------------------
    lines_H2 = Table.read(home + '/q3dfit/q3dfit/data/linelists/linelist_H2.tbl',format='ipac')
    lines_DSNR = Table.read(home + '/q3dfit/q3dfit/data/linelists/linelist_DSNR_micron.tbl',format='ipac')
    lines_fine_str = Table.read(home + '/q3dfit/q3dfit/data/linelists/linelist_fine_str.tbl',format='ipac')
    lines_TSB = Table.read(home + '/q3dfit/q3dfit/data/linelists/linelist_TSB.tbl',format='ipac')     
    lines_PAH = Table.read(home + '/q3dfit/q3dfit/data/linelists/linelist_PAH.tbl',format='ipac')     

    #add table name here for vstack
    lines = vstack([lines_H2, lines_DSNR, lines_fine_str, lines_TSB, lines_PAH])    
    #--------------------------------------------------------------------------

    lines_air_z = lines.columns[1]
    
    #rounds the floats in list CHANGE IF MORE PRECISION NEEDED
    round_lines = lines_air_z.round(decimals = sig)
        #adds column to lines table corresponding w/ redshifted wavelengths
    lines['lines'] = round_lines
 
 
    #helper function to identify lines in instrument range
    def inrange(table, key_colnames):
         colnames = [name for name in table.colnames if name  in key_colnames]
         for colname in colnames:
             if np.any(table[colname] < lamb_min) or np.any(table[colname] > lamb_max):
                 return False
             return True
 
    #grouping by wavelength then filtering by inrange()
    tg = lines.group_by('lines')
    lines_inrange = tg.groups.filter(inrange)
    
    
    # converting table to desired units
    if (wu == 'angstrom'):
        lines_inrange['lines'] = (lines_inrange['lines']*1.e4).round(decimals = sig)
        lines_inrange['lines'].unit = 'angstrom' 
        
        # filename var determined by units
        filename = 'lines_' + gal + '_' + str(lamb_min * 1.e4) + '_to_' + str(lamb_max * 1.e4) + \
            '_angstroms_rl.tbl'

    else: 
        # filename for microns default
        filename = 'lines_' + gal + '_' + str(lamb_min) + '_to_' + str(lamb_max) + \
            '_microns_rl.tbl'            
       
    
    # var used to determine if a list has >0 entries along with printing length
    list_len = len(lines_inrange['lines'])

    #comments for each generated table    
    lines_inrange.meta['comments'] = \
    ['Tables generated from reference tables created by Nadia Zakamska and Ryan',
    ' McCrory',
    'All wavelengths are assumed to be in VACUUM',
    '>LINELIST_TSB:',
    '   Data Source 1: Storchi-Bergmann et al. 2009, MNRAS, 394, 1148',
    '   Data Source 2: Glikman et al. 2006, ApJ, 640, 579 (but looked up on NIST)',
    '>LINELIST_H2:',
    '   Data Source 1: JWST H_2 lines between 1 and 2.5 microns from this link:', 
    '   https://github.com/spacetelescope/jdaviz/tree/main/jdaviz/data/linelists',
    '   H2_alt.csv file; one typo corrected (the line marked as 2-9 Q(1)',
    '   replaced with 2-0 Q(1))',
    '   Data Source 2: ISO H_2 lines from 2.5 microns onwards from this link:', 
    '   https://www.mpe.mpg.de/ir/ISO/linelists, file H2.html',
    '>LINELIST_FINE_STR',
    '   Data Source 1: ISO list of fine structure lines at 2-205 micron', 
    '   https://www.mpe.mpg.de/ir/ISO/linelists/FSlines.html',
    '>LINELIST_DSNR_MICRON:',
    '   Data Sources: line lists by David S.N. Rupke created both in vacuum',
    '   and in air and recomputed on the common vacuum grid using Morton 1991.',
    '   Morton is only accurate above 2000A, so the six lines with air',
    '   wavelengths under 2000A are explicitly fixed based on NIST database.',
    '   A handful of previousy missing Latex labels were added by hand to the',
    '   original two tables before combining.',
    '   Original table converted to microns to align with program standard',
    '   measurements',
    '>LINELIST_PAH:',
    '   Data Source 1: data from the link',
    '   https://github.com/spacetelescope/jdaviz/blob/main/jdaviz/data/linelists',
    '',]

 
    if (list_len == 0):
        print('There are no emission lines in the provided sample\nTerminating table save...')

    else:
        print('There are ' + str(list_len) + ' emission lines in the provided range.\n')
        
        
        # writing and moving the table to the linelists folder
        ascii.write(lines_inrange, filename, format = 'ipac', overwrite=True)
        shutil.move((home + '/q3dfit/q3dfit/'+ filename), (home + '/q3dfit/q3dfit/data/linelists'))
        
        print('File written as: ' + filename, sep='')
        print('Under the directory : ' + home + '/q3dfit/data/linelists/\n')
      
            