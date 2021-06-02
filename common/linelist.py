def linelist(inlines=None,linelab=True,waveunit='Angstrom',vacuum=True):
    """
    Generates an astropy Table of lines from the master files provided in
    /data/linelists
    
    Returns:
    
       An astropy Table of lines with keywords 'name','linelab','lines'
       Example: 'Lyalpha', 'Ly \$\alpha\$', 1215.67
    
    Parameters:
    
       No required parameters; by default the entire tables are read from the 
       linelist_air.tbl and linelist_vac.tbl, recomputed into vacuum and returned
    
    Optional parameters:
    
       inlines: a list of strings to match against the 'name' part of the 
           master list (default is all)
       linelab: boolean variable (True or False) -- whether to return the 
           Tex-friendly labels for the lines (default is True)
       vacuum: boolean variable (True or False) -- whether to return 
           wavelengths in vacuum, default is True (otherwise returns air 
           wavelengths); conversion is equation 3 from Morton et al. 1991
           ApJSS 77 119
       waveunit: a string variable, 'Angstrom' or 'micron', default is Angstrom
    
    Examples:
        
        1. 
        
        u=linelist()
        
        will return all lines in the catalog with labels and central wavelengths
        in vacuum in Angstroms.
        
        2.
        
        mylist=['Paa', 'Halpha', 'Paa']
        u=linelist(inlines=mylist)
        
        will produce a table of vacuum wavelengths in Angstroms for three lines
        in the same order as given, and a warning there is a duplicated line 
        in mylist
        
        3.
        
        mylist=['Paa', 'Halpha', 'junk', 'H2_00_S0']
        u=linelist(inlines=mylist,vacuum=False,waveunit='micron',linelab=False)
        
        will produce a table of air wavelengths in micron for three lines, report
        that line 'junk' is not in the line list, convert to air wavelengths and 
        to microns, complain that H2_00_S0 is outside the validity of the Morton's
        formula for vacuum to air conversion, and remove the line lables
        
        4.
        
        To get the central wavelength for an individual feature by name:
            
        wv=np.array(linelist(['Halpha'])['lines'])
        
        OR
        
        u=linelist()
        wv=np.array(u['lines'][(u['name']=='Halpha')])
        
    History:
    2020jun24 Created by Nadia Zakamska to work with two tables, one in air, one
        in vacuum, bringing them on the user-specified wavelength scale
    2021jun1 Updated by NLZ to include new line tables, print warning if Morton
        conversion from vacuum to air is not applicable    
        
    """
    import pdb
    from astropy.table import Table, vstack
    
    # I will have more files here, and hopefully the script will be able to
    # handle them through this list, but the default is they are all in vacuum
    lines_DSNR=Table.read('../data/linelists/linelist_DSNR.tbl',format='ipac')
    lines_H2=Table.read('../data/linelists/linelist_H2.tbl',format='ipac')
    lines_MIRtest=Table.read('../data/linelists/linelist_MIRtest.tbl',format='ipac')
    all_tables=[lines_DSNR,lines_H2,lines_MIRtest]
    all_units=[lines_DSNR['lines'].unit,lines_H2['lines'].unit,lines_MIRtest['lines'].unit]

    # get everything on the user-requested units:
    if ((waveunit!='Angstrom') & (waveunit!='micron')):
        print ('Wave unit ',waveunit,' not recognized, returning Angstroms')
    if (waveunit=='micron'):
        for i,u in enumerate(all_units):
            if (u=='Angstrom'): 
                all_tables[i]['lines']=all_tables[i]['lines']*1.e-4
                all_tables[i]['lines'].unit='micron'
    else:
        for i,u in enumerate(all_units):
            if (u=='micron'): 
                all_tables[i]['lines']=all_tables[i]['lines']*1.e4
                all_tables[i]['lines'].unit='Angstrom'
    # now everything is on the same units, let's stack all the tables:
    all_lines=vstack(all_tables)
    # and this will be my output variable for the user
    outlines=all_lines
        
    # a helper function that removes duplicate lines to check for duplicates
    # https://stackoverflow.com/questions/12897374/get-unique-values-from-a-list-in-python
    def uniqlist(inlist):
        outlist=[]
        for x in inlist:
            if x not in outlist:
                outlist.append(x)
        return outlist

    # check if there is an input list of lines, those are the only ones we 
    # are retaining:                               
    if inlines:
        # first of all, let's check to see if any duplicate values in inlines:
        # https://stackoverflow.com/questions/12897374/get-unique-values-from-a-list-in-python
        if (inlines!=uniqlist(inlines)):
            print ('Multiple occurrences of the same lines, proceeding as user requested')
        outlines=Table(names=all_lines.colnames,
                       dtype=[all_lines.dtype[0],all_lines.dtype[1],all_lines.dtype[2]])
        for ind,inline in enumerate(inlines):
            sub_table=all_lines[(all_lines['name']==inline)]
            if (len(sub_table) == 0):
                print ('No line with name ',inline,' was found')
            # this error would only occur if the data table has duplications:    
            if (len(sub_table) > 1):
                print ('Too many lines with name ', inline, ' were found in the database,')
                print ('Returning the first database occurrence')
                sub_table=sub_table[0]
            outlines=vstack([outlines, sub_table])    
        # the healthy outcome is that the number of entries in the outlines
        # is the same as the number of entries in the inlines
        if (len(inlines)!=len(outlines)):
            print('Input list size different from output table size, most likely')
            print('because some lines were not found in the database')

    # if the user doesn't want Latex labels, remove them
    if (linelab==False):
        outlines.keep_columns(['name','lines'])
        
    if ((vacuum!=True) & (vacuum!=False)):
        print ('Incorrect input for vacuum vs air, proceeding with default (vacuum)')
        vacuum=True
    if (vacuum==False):
        # meaning air is desired; first let's check whether we need to issue
        # a Morton warning:
        if (outlines['lines'].unit=="Angstrom"):    
            morton_fail=((outlines['lines']<2000) | (outlines['lines']>25000))
            if (sum(morton_fail)>0): print('Line(s) ',outlines['name'][morton_fail],
                                           ' outside of Morton validity for conversion to air')
        if (outlines['lines'].unit=="micron"):    
            morton_fail=((outlines['lines']<0.2) | (outlines['lines']>2.5))
            if (sum(morton_fail)>0): print('Line(s) ',outlines['name'][morton_fail],
                                           ' outside of Morton validity for conversion to air')
        if (outlines['lines'].unit=="Angstrom"): 
            temp=1.e4/outlines['lines']
        else:
            temp=1./outlines['lines']
        outlines['lines']=outlines['lines']/(1.+6.4328e-5+
                2.94981e-2/(146.-temp**2)+2.5540e-4/(41.-temp**2))

    return outlines
