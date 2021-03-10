import numpy as np


def readcf(filename):
    ''' Function defnied to read in the configuration file for fitting with questfit
        in:
        filename: str
        out:
        '''


    cf = np.loadtxt(filename, dtype = 'str')
    
    numberoftemplates=list(cf.T[0]).count('template')
    numberofBB=list(cf.T[0]).count('blackbody')
    numberofpl=list(cf.T[0]).count('powerlaw')
    numberofext=list(cf.T[0]).count('extinction')
    numberofabs=list(cf.T[0]).count('absorption')
    numberofsources=list(cf.T[0]).count('source')
    
    #Creating the initilization for the models:
    
    init_questfit = {}
    for i in cf:
        if i[0] == 'template' or i[0] == 'powerlaw' or i[0] == 'blackbody':
            #populate initilization dictionary with

            #col 0: filename (if nessesary; path hardcoded in readcf.pro)
            #col 1: lower wavelength limit or normalization factor
            #col 2: upper wavelength limit or fix/free parameter (1 or 0) for normalization
            #col 3: name of ext. curve or ice feature
            #col 4: initial guess for Av
            #col 5: fix/free parameter (1/0) for Av
            #col 6: S,M = screen or mixed extinction
            #col 7: initial guess for BB temperature or powerlaw index
            #col 8: fix/free parameter (1/0) for BB temperature or powerlaw index
            init_questfit[i[1]+' '+i[4]] = [i[1:]]

        if i[0] == 'absorption' or i[0] == 'extinction':
            init_questfit[i[1]+' '+i[4]] = [i[1:]]
             
             
             
             
             
    return init_questfit
