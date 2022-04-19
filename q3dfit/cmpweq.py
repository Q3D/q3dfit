def cmpweq(instr,linelist,doublets=None):
    """
    Compute equivalent widths for the specified emission lines.
    Uses models of emission lines and continuum, and integrates over both using
    the "rectangle rule."

    Parameters
    ----------
    instr : dict
        Contains output of IFSF_FITSPEC.
    linelist: astropy Table
        Contains the output from linelist.
    doublets : ndarray
        A 2D array of strings combining doublets in pairs if it's
        desirable to return the total eq. width,
        for example:
            doublets=[['[OIII]4959','[OIII]5007'],['[SII]6716','[SII]6731']]
            or
            doublets=['[OIII]4959','[OIII]5007']
        default: None

    Returns
    -------
    ndarray
        Array of equivalent widths.

    """

    import numpy as np
    from q3dfit.cmplin import cmplin

    ncomp=instr['param'][1]
    nlam=len(instr['wave'])
    lines=linelist['name']

    tot={}
    comp={}
    dwave=instr['wave'][1:nlam]-instr['wave'][0:nlam-1]
    for line in lines:
        tot[line]=0.
        comp[line]=np.zeros(ncomp)
        for j in range(1, ncomp+1):
            modlines=cmplin(instr,line,j,velsig=True)
            if (len(modlines)!=1):
                comp[line][j-1]=np.sum(-modlines[1:nlam]/instr['cont_fit'][1:nlam]*dwave)
            else: comp[line][j-1]=0.
            tot[line]+=comp[line][j-1]

    #Special doublet cases: combine fluxes from each line
    if (doublets!=None):
        # this shouldn't hurt and should make it easier
        doublets=np.array(doublets)
        sdoub=np.shape(doublets)
        # this should work regardless of whether a single doublet is surrounded by single or double square parentheses:
        if (len(sdoub)==1):
            ndoublets=1
            # and let's put this all into a 2D array shape for consistency so we are easily able to iterate
            doublets=[doublets]
        else:
            ndoublets=sdoub[0]
        for i in range(ndoublets):
            if ((doublets[i][0] in lines) and (doublets[i][1] in lines)):
                #new line label
                dkey = doublets[i][0]+'+'+doublets[i][1]
                #add fluxes
                tot[dkey] = tot[doublets[i][0]]+tot[doublets[i][1]]
                comp[dkey] = comp[doublets[i][0]]+comp[doublets[i][1]]

    return({'tot': tot,'comp': comp})

