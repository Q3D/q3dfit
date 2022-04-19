import numpy as np
import pdb


def readcf(filename):
    ''' Function defnied to read in the configuration file for fitting with questfit
        in:
        filename: str
        out:
        '''

    cf = np.loadtxt(filename, dtype='str', comments="#")

    numberoftemplates=list(cf.T[0]).count('template')
    numberofBB=list(cf.T[0]).count('blackbody')
    numberofpl=list(cf.T[0]).count('powerlaw')
    numberofext=list(cf.T[0]).count('extinction')
    numberofabs=list(cf.T[0]).count('absorption')
    numberofsources=list(cf.T[0]).count('source')

    #Creating the initilization for the models:

    init_questfit = {}
    for i in cf:
        if i[0] == 'template' or i[0] == 'powerlaw' or i[0] == 'blackbody' or i[0] == 'template_poly':
            #populate initilization dictionary with

            #col 0: filename (if nessesary; path hardcoded)
            #col 1: lower wavelength limit or normalization factor
            #col 2: upper wavelength limit or fix/free parameter (0 or 1) for normalization
            #col 3: name of ext. curve or ice feature
            #col 4: initial guess for Av
            #col 5: fix/free parameter (0/1) for Av
            #col 6: S,M = screen or mixed extinction
            #col 7: initial guess for BB temperature or powerlaw index
            #col 8: fix/free parameter (0/1) for BB temperature or powerlaw index
            #col 9: ice name model
            #col 10: intial guess for ice absorption tau
            #col 11: fix/free parameter (0/1) for tau
            init_questfit[i[0]+'_'+i[1]+'_'+i[4]+'_'+i[10]] = i[1:]

        if i[0] == 'absorption' or i[0] == 'extinction':
            #init_questfit[i[0]+'_'+i[1]+'_'+i[4]+'_'+i[10]] = i[1:]
            init_questfit[i[4]] = [i[1], i[0]]


        if i[0] == 'source' :
            init_questfit['source'] = i[1:]


    return init_questfit
