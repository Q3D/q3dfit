def cmpcvdf(emlwav,emlwaverr,emlsig,emlsigerr,emlflx,emlflxerr,
                      ncomp,linelist,zref,vlimits=[-1.e4,1.e4],vstep=1.):

    """
    Generates a dictionary of cumulative velocity distribution functions for all
    emission lines and for all spaxels.
    
    Returns:
        nested dictionary emlcvdf
        emlcvdf={'vel': 1D array of velocities,
                 'flux':{dictionary with columns corresponding to line names and
                         3D data with two dimensions of imaging plane and number of
                         Gaussina component},
                 'fluxerr':{same},
                 'cumfluxnorm':{same},
                 'cumfluxnormerr':{same}}
        
    Input: 
        
    Description of the assumed input data dictionaries: emlwav, emlwaverr, emlsig, emlsigerr,
    emlflx, emlflxerr: 
    
    1. List of fluxes for a list of lines:
        from linelist import linelist
        import numpy as np
        from astropy.table import Table
        linelist=linelist(['Halpha','Hbeta','Hgamma'])
        flux = Table(np.full(linelist['name'].T.data.shape, np.nan), names = linelist['name'])

    Then individual line fluxes can be called by line names:
        flux['Halpha']

    2. This functions cmpcvdf takes a much more complicated input. emlflx is a nested
    dictionary, with emlflx['ftot'], emlflx['fc1pk'], emlflx['fc2pk'] etc for the number
    of Gaussian components. Within each of these dictionary entries, we have emission
    lines called by name: emlflx['ftot']['Halpha'], emlflx['ftot']['Hbeta'] etc. 
    And each one of these is a 2D array of values in the image plane of the IFU cube.                   
    """

    import numpy as np
    from rebin import rebin
    
    # I am sure I need to do something else here for the 'bad' values... 
    bad=1e99
    c=299792.458
    # these are allegedly the smallest numbers recognized
    minexp=-310 # this is the experimentally determined limit for when 
    # I can take a log of a 1e-minexp 
    mymin=np.exp(minexp)
    # establish the velocity array from the inputs or from the defaults
    modvel=np.arange(vlimits[0],vlimits[1]+vstep,vstep)
    nmod=np.size(modvel)
    
    # ok, emlflx['ftot'] has string-like attributes for lines, but for each line
    # it's a 2D array corresponding to the image plane... How do we do this?
    # I think we need a dictionary? But the IDL one is more complex because 
    # it also has other attributes? Like, all of the Gaussian components for
    # each line are also stored in it... OK, this will be a nested distionary
    # https://www.programiz.com/python-programming/nested-dictionary
    
    # the output of the function is apparently also a dictionary with somewhat
    # of a same structure
    emlcvdf={'vel':modvel,'flux':{},'fluxerr':{},'cumfluxnorm':{},'cumfluxnormerr':{}}
    
    # the list of line names in the input flux dictionary
    outlines=list(emlflx['ftot'].keys())
    # the size of the image:
    # warning: IDL and python apparently go across the 2D arrays in different directions... 
    size_cube=np.shape(emlflx['ftot'][outlines[0]])
    for line in outlines:
        # adding empty arrays to the dictionary, even though this is probably not
        # a pythonic way to do this: 
        emlcvdf['flux'][line]=np.zeros((size_cube[0],size_cube[1],nmod))
        emlcvdf['fluxerr'][line]=np.zeros((size_cube[0],size_cube[1],nmod))
        emlcvdf['cumfluxnorm'][line]=np.zeros((size_cube[0],size_cube[1],nmod))
        emlcvdf['cumfluxnormerr'][line]=np.zeros((size_cube[0],size_cube[1],nmod))
        beta = modvel/c
        dz = np.sqrt((1. + beta)/(1. - beta)) - 1.
        # central wavelength of the line in question
        cwv=(linelist['lines'][(linelist['name']==line)])[0]
        modwaves = cwv*(1. + dz)*(1. + zref)
        for i in range(ncomp):
            cstr='c'+str(i+1)
            # OK, this function proved impossible to find, but clearly must be
            # either found or written... I have three failed versions so far
            rbpkfluxes = rebin(emlflx['f'+cstr+'pk',line],newdims=(size_cube[0],size_cube[1],nmod))
            rbpkfluxerrs = rebin(emlflxerr['f'+cstr+'pk',line],newdims=(size_cube[0],size_cube[1],nmod))
            rbpkwaves = rebin(emlwav[cstr][line],newdims=(size_cube[0],size_cube[1],nmod))
            rbpkwaveerrs = rebin(emlwaverr[cstr][line],newdims=(size_cube[0],size_cube[1],nmod))
            rbsigmas = rebin(emlsig[cstr][line],newdims=(size_cube[0],size_cube[1],nmod))
            rbsigmaerrs = rebin(emlsigerr[cstr][line],newdims=(size_cube[0],size_cube[1],nmod))
            # in the original code there was a reshaping of modwaves, but I don't understand why... 
            rbmodwaves = rebin(modwaves,newdims=(size_cube[0],size_cube[1],nmod))
            inz = ((rbsigmas > 0) & (rbsigmas != bad) & (rbpkwaves > 0) & (rbpkwaves != bad) &
                   (rbpkwaveerrs > 0) & (rbpkwaveerrs != bad) & (rbpkfluxes > 0) & 
                   (rbpkfluxes != bad) & (rbpkfluxerrs > 0) & (rbpkfluxerrs != bad))
            if (sum(inz)>0):
                exparg = np.zeros((size_cube[0],size_cube[1],nmod))-minexp
                exparg[inz] = ((rbmodwaves[inz]/rbpkwaves[inz] - 1.) / (rbsigmas[inz]/c))**2. / 2.
                i_no_under = (exparg < -minexp)
                if (sum(i_no_under)>0):
                    emlcvdf['flux'][line][i_no_under] += rbpkfluxes[i_no_under]*np.exp(-exparg[i_no_under])
                    df_norm = rbpkfluxerrs[i_no_under]*np.exp(-exparg[i_no_under])
                    term1=rbpkfluxes[i_no_under]*np.abs(rbmodwaves[i_no_under]-rbpkwaves[i_no_under])
                    term2=rbsigmas[i_no_under]/c*rbpkwaves[i_no_under]
                    df_wave = term1/(term2**2)*rbpkwaveerrs[i_no_under]*np.exp(-exparg[i_no_under])
                    term3=rbpkfluxes[i_no_under]*(rbmodwaves[i_no_under]-rbpkwaves[i_no_under])**2
                    term4=rbsigmas[i_no_under]/c*rbpkwaves[i_no_under]
                    df_sig = term3/term4**2*rbsigmaerrs[i_no_under]/rbsigmas[i_no_under]*np.exp(-exparg[i_no_under])
                    dfsq = np.zeros((size_cube[0],size_cube[1],nmod))
                    dfsq = dfsq[i_no_under]
                    i_no_under_2 = ((df_norm > mymin) & (df_wave > mymin) & (df_sig > mymin))
                    if (sum(i_no_under_2)>0):
                        dfsq[i_no_under_2] = (df_norm[i_no_under_2])**2+(df_wave[i_no_under_2])**2+(df_sig[i_no_under_2])**2
                    emlcvdf['fluxerr'][line][i_no_under] += dfsq 
                        
        inz = (emlcvdf['flux'][line] > 0)
        if (sum(inz)>0):
            emlcvdf['fluxerr'][line][inz] = np.sqrt(emlcvdf['fluxerr'][line][inz])
        # size of each model bin
        dmodwaves = modwaves[1:nmod] - modwaves[0:nmod-1]
        # supplement with the zeroth element to make the right length
        dmodwaves=np.append(dmodwaves[0],dmodwaves)
        # rebin to full cube; OK, I am sure there is a super-elegant way to do this, but
        # brute force is called for right now
        temp5=np.zeros((size_cube[0],size_cube[1],nmod))
        for i in range(size_cube[0]):
            for j in range(size_cube[1]):
                 temp5[i][j]=dmodwaves
        dmodwaves=temp5
        fluxnorm = emlcvdf['flux'][line]*dmodwaves
        fluxnormerr = emlcvdf['fluxerr'][line]*dmodwaves
        fluxint = rebin(sum(fluxnorm,3),newdims=(size_cube[0],size_cube[1],nmod))
        inz = (fluxint != 0)
        if (sum(inz)>0):
            fluxnorm[inz]/=fluxint[inz]
            fluxnormerr[inz]/=fluxint[inz]
                
        emlcvdf['cumfluxnorm'][line][:,:,0] = fluxnorm[:,:,0]
        for i in range(1,nmod):
            emlcvdf['cumfluxnorm'][line][:,:,i] = emlcvdf['cumfluxnorm'][line][:,:,i-1] + fluxnorm[:,:,i]
        emlcvdf['cumfluxnormerr'][line] = fluxnormerr
           
    return(emlcvdf)
        
            
