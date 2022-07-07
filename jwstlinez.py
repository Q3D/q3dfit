#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import numpy as np
from astropy.io import ascii
from astropy.table import Table, vstack    
from astropy import units as u
from pathlib import Path

"""
Created on Tue Jun 14 15:51:39 2022

"""

def jwstline(z, gal, instrument, mode, grat_filt, waveunit = 'micron'):
    
    """
    Creates a table of emission lines expected to be found in a given instrument configuration for JWST. 
    References stored under linelists are .tbl of filenames:
    
    lines_H2
    lines_DSNR_micron   CHANGE
    lines_TSB
    lines_ref
    
    ADD MORE ONCE PAH COMES OUT
    String inputs are not case sensitive, but must be entered with exact spelling
    
    Parameters
    ----------
    
    z : flt, required
        Galaxy redshift
    gal : str, required
        Galaxy name for filenaming
    instrument : str, required
        JWST instrumnent. Inputs: NIRSpec, MIRI
    mode : str, required
        Instrument mode.
        NIRSpec Inputs: IFU, MOS, FSs, BOTS
        MIRI Inputs: MRS
    grat_filt : grating and filter combination for NIRSpec or channel for MIRI
        NIRSpec Inputs: G140M_F0701P, G140M_F1001P, G235M_F1701P, G395M_F2901P, G140H_F0701P, G140H_F1001P, 
            G235H_F1701P, G395H_F2901P, Prism_Clear
        MIRI Inputs: Ch1_A, Ch1_B, Ch1_C, Ch2_A, Ch2_B, Ch2_C, Ch3_A, Ch3_B, Ch3_C, Ch4_A, Ch4_B, Ch4_C
    waveunit : str, default='micron'
        Units desired for output tables. Inputs are 'micron' or 'angstrom'
            
    Returns
    --------
    
    lines : astropy table
        An astropy table of emission lines with keywords 'name', 'lines', q'linelab', 'observed'
        Example: H2_43_Q6, 2.98412, H$_2$(4-3) Q(6), 4.282212 
        
    
    
    """
    
    
    
    home = str(Path.home())
    inst = instrument.lower()
    md = mode.lower()
    gf = grat_filt.lower()
    wu = waveunit.lower() #test w/o to see if necessary
    
    if ((wu!='angstrom') & (wu!='micron')):
        print('Wave unit ',waveunit,' not recognized, proceeding with default, microns')
    
    
    #add links to more tables here
    lines_H2 = Table.read(home + '/q3dfit/q3dfit/data/linelists/linelist_H2.tbl',format='ipac')
    lines_DSNR = Table.read(home + '/q3dfit/q3dfit/data/linelists/linelist_DSNR_micron.tbl',format='ipac')
    lines_fine_str = Table.read(home + '/q3dfit/q3dfit/data/linelists/linelist_fine_str.tbl',format='ipac')
    lines_TSB = Table.read(home + '/q3dfit/q3dfit/data/linelists/linelist_TSB.tbl',format='ipac')
    inst_ref = Table.read('/Users/ryanmccrory/q3dfit/q3dfit/data/jwst_tables/' + instrument + '.tbl',format = 'ipac')

    lines = vstack([lines_H2, lines_DSNR, lines_fine_str, lines_TSB])
    tcol = lines.columns[1]
    #

    # mask to search for row of provided entries
    mask = (inst_ref['mode'] == md) & (inst_ref['grat_filt'] == gf) 
    print('')
    print('Instrument Configuration Wavelength Range:')
    print(' ')
    print(inst_ref[mask])

    lamb_min = inst_ref[mask]['lamb_min']
    lamb_max = inst_ref[mask]['lamb_max']
    #redshifts the table
    lines_air_z = np.multiply(tcol,z+1)
    #rounds the floats in list CHANGE IF MORE PRECISION NEEDED
    round_lines = lines_air_z.round(decimals = 7) * u.micron
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
    
    
   # unit conversion from program standard microns to angstroms
   # FIX THIS
    if (wu=='angstrom'):
        print('')
        print('angstroms unit selected')
        #lines_inrange['lines'] = (lines_inrange['lines']) .to (u.angstrom)
        #lines_inrange['observed'] = (lines_inrange['observed']) .to (u.angstrom)
        
        lines_inrange['lines'] = (lines_inrange['lines']*1.e4).round(decimals = 7)
        lines_inrange['lines'].unit = 'angstrom'
        lines_inrange['observed'] = (lines_inrange['observed']*1.e4).round(decimals = 7)
        lines_inrange['observed'].unit = 'angstrom'
        
    list_len = len(lines_inrange['lines'])
    
    if (list_len == 0):
        print('')
        print('There are no emission lines in the provided sample')
        print('')
        print('Terminating table save...')
    else:
        print('')
        print('There are ' + str(list_len) + ' emission lines in the provided range.')
        
        ascii.write(lines_inrange, ('lines_' + gal + '_' + inst + '_' + gf + '.tbl')\
            , format = 'ipac', overwrite=True)

        print('')
        print('File written as "lines_' + gal + '_' + inst + '_' + gf + '.tbl"', sep='')        
        print('Under the directory : ',home, '')
                
    