def masklin(llambda, linelambda, halfwidth, specres, nomaskran=''):
    """
    Masks emission lines from the spectrum for continuum fitting
    
    Returns:
        
        Array of llambda-array indices indicating non-masked wavelengths.
        
    Params:
        
        llambda: in, required, type=dblarr
            Wavelengths of the spectrum
        linelambda: in, required, type=astropy Table
            lines to mask (at the minimum, has 'lines' and can have 'name' and 
            'linelab' as well)
        halfwidth: in, required, type=astropy Table (has 'names' and 'halfwidths')
            Half width (in km/s) of masking region around each line
            
    Optional parameters:
        
        nomaskran: in, optional, type=np.array[2,nreg] offloating point values
            Set of lower and upper wavelength limits of regions not to mask.
            
    Example:
        
        from masklin import masklin
        # generate lines to be masked
        from linelist import linelist
        mylist=['Halpha','Hbeta','Hgamma']
        u=linelist(inlines=mylist)
        # generate wavelength array
        import numpy as np
        llambda=np.arange(3000.,8000.,1.)
        # generate half widths for masking
        # start with making an astropy table from the line names
        from astropy.table import Table
        # https://docs.astropy.org/en/stable/table/construct_table.html
        halfwidth=Table([u['name']])
        # populate another column
        # https://docs.astropy.org/en/stable/table/modify_table.html
        halfwidth['halfwidths']=6000.0
        # I have chosen a huge masking width for better visual effect
        
        #now use the masking function
        ind=masklin(llambda, u, halfwidth, 60.)
        
        # plot the spectrum without and with the masking
        from matplotlib import pyplot as plt
        plt.interactive(True)
        fig=plt.figure(1, figsize=(8,8))
        plt.clf()
        spec=1.+(llambda-3000.0)*1e-3
        plt.plot(llambda, spec, color='grey',alpha=0.3)
        plt.scatter(llambda[ind],spec[ind]+0.05,color='red',alpha=0.1,s=2)
        plt.show()
        
        # now let's define a no-masking array
        dontmask=np.array([[3000,4000,5000],[4000,5000,6000]])
        ind1=masklin(llambda, u, halfwidth, 60., nomaskran=dontmask)
        plt.scatter(llambda[ind1],spec[ind1]-0.05,color='blue',alpha=0.03,s=2)
        plt.show()
    """

    import numpy as np
    print ('ALTHOUGH THE RESOLUTION IS PASSED AS AN ARGUMENT, IT IS NOT')
    print ('BEING USED IN THIS FUNCTION. THIS IS AS IN THE ORIGINAL IFSF')
    print ('FUNCTION. LET US DISCUSS IF WE NEED THIS.')
    
    c = 299792.458
    # we will return the ones that are not masked
    
    # start by retaining all elements -- mark them all True
    retain=np.array(np.ones(len(llambda)), dtype=bool)

    # line is the index in the linelambda array and cwv is the central wavelength
    # let's flag the indices that are masked
    for line,cwv in enumerate(linelambda['lines']):
        temp1 = np.array((llambda <= cwv*(1. - halfwidth['halfwidths'][line]/c)), dtype=bool)
        temp2 = np.array((llambda >= cwv*(1. + halfwidth['halfwidths'][line]/c)), dtype=bool)
        retain = (retain & (temp1 | temp2))

    # if the user has defined the regions not to be masked: 
    if (len(nomaskran)>0): 
        # set all to False
        nomask=np.array(np.zeros(len(llambda)),dtype=bool)
        for j,llim in enumerate(nomaskran[0]):
            # set to True between lower and upper limit for each
            temp1 = np.array((llambda >= llim), dtype=bool)
            temp2 = np.array((llambda <=nomaskran[1][j]), dtype=bool)
            nomask=(nomask | (temp1 & temp2))
        # combine retain and nomask
        retain = (retain | nomask)
    
    # return the indices to be retained
    # https://stackoverflow.com/questions/21448225/getting-indices-of-true-values-in-a-boolean-list    
    indgd=np.where(retain)[0]
    return indgd

