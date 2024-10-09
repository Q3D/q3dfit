#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import importlib.resources as pkg_resources
import numpy as np
from astropy.io import ascii
from astropy.table import Table, vstack
from astropy import units as u
from q3dfit.data import linelists
from q3dfit.data import jwst_tables


def jwstlinez(z, gal, instrument, mode, grat_filt, waveunit='micron',
              outdir=None):
    """
    Creates a table of emission lines expected to be found in a given
    instrument configuration for JWST.
    The 'observed' column displays a redshifted emission line center calculated
    using the input z value. Therefore, it is only as precise as the
    known z value.

    References stored under q3dfit/data/linelists are .tbl of filenames:

    lines_H2.tbl
    lines_DSNR_micron.tbl
    lines_TSB.tbl
    lines_ref.tbl
    lines_PAH.tbl

    Data for instrument configurations in microns can be found in
    q3dfit/data/jwst_tables in the files:

    miri.tbl
    nirspec.tbl

    String inputs are not case sensitive, but must be entered with exact
    spelling.
    Otherwise, the function will return errors.


    Parameters
    ----------

    z : flt, required
        Galaxy, systemic redshift.
        Used for redshifting emission lines for 'observed' column.
    gal : str, required
        Galaxy name for filenaming.
        Can also be customized for any other desired filename designations
    instrument : str, required
        JWST instrumnent. Inputs: NIRSpec, MIRI
        Used to grab the right table for each instrument.
    mode : str, required
        Instrument mode.
        NIRSpec Inputs: IFU, MOS, FSs, BOTS
        MIRI Inputs: MRS
    grat_filt : grating and filter combination for NIRSpec or channel for MIRI
        NIRSpec Inputs:
            G140M_F0701P, G140M_F1001P, G235M_F1701P, G395M_F2901P,
            G140H_F0701P, G140H_F1001P, G235H_F1701P, G395H_F2901P, Prism_Clear
        MIRI Inputs:
            Ch1_A, Ch1_B, Ch1_C, Ch2_A, Ch2_B, Ch2_C, Ch3_A, Ch3_B, Ch3_C,
            Ch4_A, Ch4_B, Ch4_C
    outdir : str, required
        output directory
    waveunit : str, default='micron'
        Units desired for output tables.
        Inputs are 'micron' or 'Angstrom'

    Returns
    --------

    lines : astropy table
        An astropy table of emission lines with keywords 'name', 'lines',
        'linelab', 'observed'
        Example Row: H2_43_Q6, 2.98412, H$_2$(4-3) Q(6), 4.282212

        Interanlly, everything is processed in microns, so filename inclues
        range values in microns.
        The units of the table can be Angstroms or microns, depending on the
        entered value of waveunit.
        Output table contains comments descring data sources


    """
    inst = instrument.lower()
    md = mode.lower()
    gf = grat_filt.lower()

    # get everything on the user-requested units:
    if ((waveunit != 'Angstrom') & (waveunit != 'micron')):
        print('Wave unit ', waveunit,
              ' not recognized, proceeding with default, microns\n')

    linetables = ['linelist_DSNR.tbl', 'linelist_H2.tbl', 'linelist_H1.tbl',
                  'linelist_fine_str.tbl', 'linelist_TSB.tbl',
                  'linelist_PAH.tbl']
    all_tables = []
    all_units = []
    for llist in linetables:
        with pkg_resources.path(linelists, llist) as p:
            newtable = Table.read(p, format='ipac')
            all_tables.append(newtable)
            all_units.append(newtable['lines'].unit)
    if (waveunit == 'Angstrom'):
        for i, un in enumerate(all_units):
            if (un == 'micron'):
                all_tables[i]['lines'] = all_tables[i]['lines']*1e4
                all_tables[i]['lines'].unit = 'Angstrom'
    else:
        for i, un in enumerate(all_units):
            if (un == 'Angstrom'):
                all_tables[i]['lines'] = all_tables[i]['lines']*1.e-4
                all_tables[i]['lines'].unit = 'micron'
    # now everything is on the same units, let's stack all the tables:
    lines = vstack(all_tables)
    tcol = lines.columns[1]

    if (inst == 'miri' or inst == 'nirspec'):
        with pkg_resources.path(jwst_tables, inst+'.tbl') as p:
            inst_ref = Table.read(p, format = 'ipac')
    else:
        print("\n--------------------------")
        print(inst + ' is not a valid instrument input')
        print("--------------------------\n")

    # mask to search for row of provided entries
    mask = (inst_ref['mode'] == md) & (inst_ref['grat_filt'] == gf)
    print('Instrument configuration rest wavelength range:\n')
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


    # converting table to desired units
    if (waveunit=='Angstrom'):
        print('Angstroms unit selected\n')

        lines_inrange['lines'] = (lines_inrange['lines']*1.e4).round(decimals = 7)
        lines_inrange['lines'].unit = 'Angstrom'

        lines_inrange['observed'] = (lines_inrange['observed']*1.e4).round(decimals = 7)
        lines_inrange['observed'].unit = 'Angstrom'

        # filename variable determined by units
        filename = 'lines_' + gal + '_' + str(lamb_min * 1.e4) + '_to_' + str(lamb_max * 1.e4) + \
            '_Angstroms_jwst.tbl'

    else:
        # filename for microns default
        filename = 'lines_' + gal + '_' + str(lamb_min) + '_to_' + str(lamb_max) + \
            '_microns_jwst.tbl'

    # var used to determine if a list has >0 entries along with printing length
    list_len = len(lines_inrange['lines'])

    #comments for each generated table
    lines_inrange.meta['comments'] = \
    ['Tables generated from reference tables created by Nadia Zakamska and Ryan McCrory',
    'All wavelengths are assumed to be in VACUUM',
    '>LINELIST_TSB:',
    '   Data Source 1: Storchi-Bergmann et al. 2009, MNRAS, 394, 1148',
    '   Data Source 2: Glikman et al. 2006, ApJ, 640, 579 (but looked up on NIST)',
    '>LINELIST_H2:',
    '   Data Source 1: JWST H_2 lines between 1 and 2.5 microns from this link:',
    '   https://github.com/spacetelescope/jdaviz/tree/main/jdaviz/data/linelists',
    '   H2_alt.csv file; one typo corrected (the line marked as 2-9 Q(1) replaced with 2-0 Q(1))',
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
    '   Original table converted to microns to align with program standard measurements',
    '>LINELIST_PAH:',
    '   Data Source 1: data from the link',
    '   https://github.com/spacetelescope/jdaviz/blob/main/jdaviz/data/linelists',
    '',]

    print('There are ' + str(list_len) +
          ' emission lines visible with this instrument configuration.\n')

    if outdir is not None:
        if (list_len == 0):
            print('There are no emission lines in the provided sampling range\n')
            print('Terminating table save...\n')
        else:
            ascii.write(lines_inrange, outdir+filename, format='ipac',
                        overwrite=True)

        print('File written as ' + filename, sep='')
        print('Under the directory : ', outdir)

    return lines_inrange



def observedlinez(z, gal, lamb_min, lamb_max, vacuum=True, waveunit='micron',
                  outdir=None):
    """
    Similar to jwstline(), observedlinez() produces a table with emission lines 
    in the provided range. 
    Unlike jwstline(), this function can solve for a user-specified range of 
    wavelengths and can account for air wavelength conversions.
    The column 'observed' accounts for both redshifting and air refraction. 
    It reports an expected center for an emission distribution and can be used
    to identify unknown emission lines for fitting. 
    
    All input wavelengths are assumed to be REDSHIFTED!
    
    References stored under q3dfit/data/linelists are .tbl of filenames:
            
        lines_H2.tbl
        lines_DSNR_micron.tbl   
        lines_TSB.tbl
        lines_ref.tbl
        lines_PAH.tbl
        
    More tables can be added manually if saved in the linelists folder and 
    called in this function definition.
    
    
    Parameters
    ----------
    
    z : flt, required
        Galaxy redshift
    gal : str, required
        Galaxy name for filenaming. 
        Can also be customized for any other desired filename designations
    lamb_min : flt, required
        minimum <OBSERVED> wavelength value of instrument, units determined by 
        waveunit value
    lamb_max : flt, required
        maximum <OBSERVED> wavelength of instrument, units determined by waveunit
        value
    outdir : str, required
        output directory
    vacuum : bool, optional, default = True
        if false, enables conversion to air wavelengths
    waveunit : str, optional, default = 'micron'
        determines the unit of input wavelengths and output table
        acceptable inputs are 'micron' and 'Angstrom'
        
    Returns
    --------
        
    lines : astropy table
        An astropy table of emission lines with keywords 'name', 'lines', 
        'linelab', 'observed'
        Example Row: H2_43_Q6, 2.98412, H$_2$(4-3) Q(6), 4.282212 
        Interanlly, everything is processed in microns, so filename inclues 
        range values in microns. 
        The units of the table can be Angstroms or microns, depending on the 
        entered value of waveunit.
        
        Output table contains comments descring data sources

            
    """
 
    # sig is a rounding variable, so it must be dependent on unit
    if ((waveunit!='Angstrom') & (waveunit!='micron')):
        print("possible waveunit inputs are 'micron' or 'Angstrom'")
        print ('Wave unit ',waveunit,' not recognized, returning micron\n')
        sig = 7
        
    elif (waveunit == 'Angstrom'):
        # converting A input to microns b/c inrange() relies on micron input
        lamb_max = lamb_max / 1.e4
        lamb_min = lamb_min / 1.e4
        sig = 3
        print('Angstroms unit selected\n')
        
    else:
        sig = 7
             
    linetables = ['linelist_DSNR.tbl', 'linelist_H2.tbl', 'linelist_H1.tbl',
                  'linelist_fine_str.tbl', 'linelist_TSB.tbl',
                  'linelist_PAH.tbl']
    all_tables = []
    all_units = []
    for llist in linetables:
        with pkg_resources.path(linelists, llist) as p:
            newtable = Table.read(p, format='ipac')
            all_tables.append(newtable)
            all_units.append(newtable['lines'].unit)
    # get everything on the user-requested units:
    if (waveunit == 'Angstrom'):
        for i, un in enumerate(all_units):
            if (un == 'micron'):
                all_tables[i]['lines'] = all_tables[i]['lines']*1e4
                all_tables[i]['lines'].unit = 'Angstrom'
    else:
        for i, un in enumerate(all_units):
            if (un == 'Angstrom'):
                all_tables[i]['lines'] = all_tables[i]['lines']*1.e-4
                all_tables[i]['lines'].unit = 'micron'
    # now everything is on the same units, let's stack all the tables:
    lines = vstack(all_tables)
    tcol = lines.columns[1]
    # Redshifting each entry in tcol (REST wavelengths)
    tcolz = np.multiply(tcol, z+1)

    # air conversion logic and redshifting of tablie columns
    
    if ((vacuum!=True) & (vacuum!=False)):
        #sets vacuum default to True (does nothing)
        print('Incorrect input for vacuum vs air. "' + vacuum + '" not recognized.') 
        print('proceeding with default vacuum = True\n')
        lines_air_z = np.multiply(tcol, z+1)

    # Vacuumn to air conversion from eq. 3 from Morton et al. 1991 ApJSS 77 119
    # Uses same equation as q3dfit/airtovac.py
    elif (vacuum == False):
        # meaning observations need air to vac conversion
        tmp = 1/tcolz
        
        # Air transformation for tcolz (REDSHIFTED wavelengths ONLY)
        lines_air_z = tcolz/(1.+6.4328e-5 + 2.94981e-2/(146.-tmp**2)\
                                        + 2.5540e-4/(41.-tmp**2))

        
    elif (vacuum == True):
        lines_air_z = tcolz
            

    #redshifts the table
    #rounds the floats in list CHANGE IF MORE PRECISION NEEDED
    round_lines = lines_air_z.round(decimals = 8)
    #adds column to lines table corresponding w/ redshifted wavelengths
    lines['observed'] = round_lines
 
 
    #helper function to identify lines in instrument range
    def inrange(table, key_colnames):
         colnames = [name for name in table.colnames if name  in key_colnames]
         for colname in colnames:
             if np.any(table[colname] < lamb_min) or np.any(table[colname] > lamb_max):
                 return False
             return True
 
    
    #grouping by wavelength then filtering by inrange()
    tg = lines.group_by('observed')
    lines_inrange = tg.groups.filter(inrange)
    
    # converting table to desired units
    if (waveunit == 'Angstrom'):
        lines_inrange['lines'] = (lines_inrange['lines']*1.e4).round(decimals = sig)
        lines_inrange['lines'].unit = 'Angstrom'     
        
        lines_inrange['observed'] = (lines_inrange['observed']*1.e4).round(decimals = sig)
        lines_inrange['observed'].unit = 'Angstrom'   
         
        # filename variable determined by units
        filename = 'lines_' + gal + '_' + str(lamb_min * 1.e4) + '_to_' + str(lamb_max * 1.e4) + \
            '_Angstroms_ml.tbl'
    else:
        # filename for microns default
        filename = 'lines_' + gal + '_' + str(lamb_min) + '_to_' + str(lamb_max) + \
            '_microns_ml.tbl'
    
    # var used to determine if a list has >0 entries along with printing length
    list_len = len(lines_inrange['lines'])

    # comments for each generated table    
    lines_inrange.meta['comments'] = \
    ['Tables generated from reference tables created by Nadia Zakamska and Ryan',
    ' McCrory',
    'All wavelengths are assumed to be in VACUUM',
    'Air wavelength conversion is taken from eq. 3 from Morton et al. 1991 ',
    'ApJSS 77 119',
    '-----------------------',
    '>LINELIST_TSB:',
    '   Data Source 1: Storchi-Bergmann et al. 2009, MNRAS, 394, 1148',
    '   Data Source 2: Glikman et al. 2006, ApJ, 640, 579 (but looked up on NIST)',
    '>LINELIST_H2:',
    '   Data Source 1: JWST H_2 lines between 1 and 2.5 microns from this link:', 
    '   https://github.com/spacetelescope/jdaviz/tree/main/jdaviz/data/linelists',
    '   H2_alt.csv file; one typo corrected (the line marked as 2-9 Q(1) replaced',
    '   with 2-0 Q(1))',
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
            
    # writing the table
    if outdir is not None:

        if (list_len == 0):

            print('There are no emission lines in the provided sample\nTerminating table save...')

        else:
            print('There are ' + str(list_len) + ' emission lines in the provided range.\n')

            ascii.write(lines_inrange, outdir+filename, format = 'ipac',
                    overwrite=True)
        
            print('File written as: ' + filename, sep='')
            print('under the directory ' + outdir)


def restline(gal, lamb_min, lamb_max, waveunit='micron', outdir=None):
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
    outdir : str, required
        output directory
    waveunit : str, optional, default = 'micron'
        determines the unit of input wavelengths and output table
        acceptable inputs are 'micron' and 'Angstrom'
        
    Returns
    --------
        
    lines_...tbl : astropy table
        An astropy table of emission lines with keywords 'name', 'lines', 
        'linelab'
        Example Row: H2_43_Q6, 2.98412, H$_2$(4-3) Q(6), 4.282212 
        Interanlly, everything is processed in microns, so filename inclues 
        range values in microns. 
        The units of the table can be Angstroms or microns, depending on the 
        entered value of waveunit.
        Output table contains comments descring data sources

            
    """
 
    if ((waveunit!='Angstrom') & (waveunit!='micron')):
        print("possible waveunit inputs are 'micron' or 'Angstrom'")
        print ('Wave unit ',waveunit,' not recognized, returning microns\n')
        sig = 7
    elif (waveunit == 'Angstrom'):
        # converting A input to microns
        lamb_max = lamb_max / 1.e4
        lamb_min = lamb_min / 1.e4
        sig = 3
        print('Angstroms unit selected\n')
    else:
        sig = 7
     
        
    linetables = ['linelist_DSNR.tbl', 'linelist_H2.tbl', 'linelist_H1.tbl',
                  'linelist_fine_str.tbl', 'linelist_TSB.tbl',
                  'linelist_PAH.tbl']
    all_tables = []
    all_units = []
    for llist in linetables:
        with pkg_resources.path(linelists, llist) as p:
            newtable = Table.read(p, format='ipac')
            all_tables.append(newtable)
            all_units.append(newtable['lines'].unit)
    # get everything on the user-requested units:
    if (waveunit == 'Angstrom'):
        for i, un in enumerate(all_units):
            if (un == 'micron'):
                all_tables[i]['lines'] = all_tables[i]['lines']*1e4
                all_tables[i]['lines'].unit = 'Angstrom'
    else:
        for i, un in enumerate(all_units):
            if (un == 'Angstrom'):
                all_tables[i]['lines'] = all_tables[i]['lines']*1.e-4
                all_tables[i]['lines'].unit = 'micron'
    # now everything is on the same units, let's stack all the tables:
    lines = vstack(all_tables)
    # Redshifting each entry in tcol (REST wavelengths)
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
    if (waveunit == 'Angstrom'):
        lines_inrange['lines'] = (lines_inrange['lines']*1.e4).round(decimals = sig)
        lines_inrange['lines'].unit = 'Angstrom' 
        
        # filename var determined by units
        filename = 'lines_' + gal + '_' + str(lamb_min * 1.e4) + '_to_' + str(lamb_max * 1.e4) + \
            '_Angstroms_rl.tbl'

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

     # writing the table
    if outdir is not None:

        if (list_len == 0):

            print('There are no emission lines in the provided sample\nTerminating table save...')

        else:
            print('There are ' + str(list_len) + ' emission lines in the provided range.\n')

            ascii.write(lines_inrange, outdir+filename, format = 'ipac',
                    overwrite=True)
        
            print('File written as: ' + filename, sep='')
            print('under the directory ' + outdir)
