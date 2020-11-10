# -*- coding: utf-8 -*-
"""
Routine to plot the continuum and emission lines fits to a spectrum, doesn't
return anything.

Created: 7/9/2020

@author: hadley
"""
import numpy as np
import math
import pdb
import importlib
from q3dfit.common.linelist import linelist
from q3dfit.common.readcube import CUBE
from scipy.special import legendre
from scipy import interpolate
from ppxf.ppxf_util import log_rebin


def q3da(initproc, cols = None, rows = None, noplots = None, oned = None, \
         verbose = None, _extra = None):
    bad = 1.0 * 10**99
    fwhmtosig = 2.0 * math.sqrt(2.0 * np.log(2.0))
    
    if verbose != None:
        quiet = 0
    else: quiet = 1
    
    if oned != None:
        oned = 1
    else: oned = 0
    
    #reads initdat from initialization file ie pg1411 (initproc is a string)
    module = importlib.import_module('q3dfit.init.' + initproc)                 
    fcninitproc = getattr(module, initproc)    
    initdat = fcninitproc()
   
    #if 'donad' in initdat: do later       

    if not ('noemlinfit' in initdat):
        #get linelist
        if 'argslinelist' in initdat:
            listlines = linelist(initdat['lines'], **initdat['argslinelist'])
        else: listlines = linelist(initdat['lines'])
        nlines = len(listlines)
        
        #linelist with doublets to combine
        emldoublets = [['[SII]6716','[SII]6731'], \
                       ['[OII]3726','[OII]3729'], \
                       ['[NI]5198','[NI]5200'], \
                       ['[NeIII]3869','[NeIII]3967'], \
                       ['[NeV]3345','[NeV]3426'], \
                       ['MgII2796','MgII2803']]
        
        if len(emldoublets) == 1:
            ndoublets = 1
        else: ndoublets = len(emldoublets)
        
        lines_with_doublets = initdat['lines']
        
        keys = listlines.keys() 
        
        for i in range (0, ndoublets):
            if (emldoublets[i][0] in keys) and \
                (emldoublets[i][1] in keys):
                dkey = emldoublets[i][0] + "+" + emldoublets[i][1]
                lines_with_doublets = [lines_with_doublets] + [dkey]
        
        if 'argslinelist' in initdat:
            linelist_with_doublets = linelist(lines_with_doublets, \
                                               **initdat['argslinelist'])
        else: linelist_with_doublets = linelist(lines_with_doublets)
    
    if 'fcnpltcont' in initdat:
        fcnpltcont = initdat['fcnpltcont']
    else: fcnpltcont = 'pltcont'

#READ DATA
    if not ('datext' in initdat): datext = 1
    else: datext = initdat['datext']
    
    if not ('varext' in initdat): varext = 2
    else: varext = initdat['varext']
    
    if not ('dqext' in initdat): dqext = 3
    else: dqext = initdat['dqext']
    
    header = bytes(1)
    
    if 'argsreadcube' in initdat:
        cube = CUBE(infile = initdat['infile'], quiet = quiet, oned = oned, \
                    header = header, datext = datext, varext = varext, \
                    dqext = dqext, **initdat['argsreadcube'])
    else:
        cube = CUBE(infile = initdat['infile'], quiet = quiet, oned = oned, \
                    header = header, datext = datext, varext = varext, \
                    dqext = dqext)
    
    if 'vormap' in initdat:
        vormap = initdat['vormap']
        nvorcols = max(vormap)
        #making vorcoords an np array for this one
        vorcoords  = np.zeros(nvorcols, 2)
        for i in range (1, nvorcols + 1):
            ivor = np.where(vormap == i)
            xyvor = array_indices(vormap, ivor[0])
            #ctivor = len(xyvor)
            vorcoords[i - 1, 0] = xyvor[0]
            vorcoords[i - 1, 1] = xyvor[1] #check what dimensions vorcoords etc. are


#INITIALIZE OUTPUT FILES, need to write helper functions (printlinpar, 
#printfitpar) later
        
#INITIALIZE LINE HASH
    if not('noemlinfit' in initdat):
        emlwav = dict()
        emlwaverr = dict()
        emlsig = dict()
        emlsigerr = dict()
        emlweq = dict()
        emlflx = dict()
        emlflxerr = dict()
        emlweq['ftot'] = dict()
        emlflx['ftot'] = dict()
        emlflxerr['ftot'] = dict()
        for k in range (0, initdat['maxncomp']):
            cstr = 'c' + str(k + 1)
            emlwav[cstr] = dict()
            emlwaverr[cstr] = dict()
            emlsig[cstr] = dict()
            emlsigerr[cstr] = dict()
            emlweq['f' + cstr] = dict()
            emlflx['f' + cstr] = dict()
            emlflxerr['f' + cstr] = dict()
            emlflx['f' + cstr + 'pk'] = dict()
            emlflxerr['f' + cstr + 'pk'] = dict()
        for line in lines_with_doublets:
            emlweq['ftot'][line] = np.zeros((cube.nrows, cube.ncols), \
                dtype = float) + bad
            emlflx['ftot'][line] = np.zeros((cube.nrows, cube.ncols), \
                dtype = float) + bad
            emlflxerr['ftot'][line] = np.zeros((cube.nrows, cube.ncols), \
                dtype = float) + bad
        for k in range (0, initdat['maxncomp']):
            cstr = 'c' + str(k + 1)
            emlwav[cstr][line] = np.zeros((cube.nrows, cube.ncols), \
                dtype = float) + bad
            emlwaverr[cstr][line] = np.zeros((cube.nrows, cube.ncols), \
                dtype = float) + bad
            emlsig[cstr][line] = np.zeros((cube.nrows, cube.ncols), \
                dtype = float) + bad
            emlsigerr[cstr][line] = np.zeros((cube.nrows, cube.ncols), \
                dtype = float) + bad
            emlweq['f' + cstr][line] = np.zeros((cube.nrows, cube.ncols), \
                dtype = float) + bad
            emlflx['f' + cstr][line] = np.zeros((cube.nrows, cube.ncols), \
                dtype = float) + bad
            emlflxerr['f' + cstr][line] = np.zeros((cube.nrows, cube.ncols), \
                dtype = float) + bad
            emlflx['f' + cstr + 'pk'][line] = np.zeros((cube.nrows, cube.ncols), \
                dtype = float) + bad
            emlflxerr['f' + cstr + 'pk'][line] = np.zeros((cube.nrows, cube.ncols), \
                dtype = float) + bad
        #basically dictionaries of dictionaries of 2D arrays 
    if 'flipsort' in initdat:
        flipsort = np.zeros(cube.nrows, cube.ncols)
        sizefs = len(initdat['flipsort'])
        for i in range (0, len(sizefs[0])): #probably right but check again
            flipsort[initdat['flipsort'][0][i] - 1] \
                [initdat['flipsort'][1][i] - 1] = bytes(1) #b
 
#LOOP THROUGH SPAXELS
    #switch to track when first NaD normalization done
    firstnadnorm = 1
    #switch to track when first continuum processed
    firstcontproc = 1
    
    if cols != None:
        cols = [1, cube.ncols]
    elif len(cols) == 1: cols = [cols, cols]
    np.array(cols).astype("int")
   
    for i in range (cols[0] - 1, cols[1]):
        if verbose != None:
            print('Column ' + (i + 1) + ' of ' + cube.ncols) 
        
        if rows != None: rows = [1, cube.nrows]
        elif len(rows) == 1: rows = [rows, rows]
        #fix(rows)
        for h in range(0, len(rows)):
            rows[h] = int(rows[h])        
        
        for j in range (rows[0] - 1, rows[1]):            
            novortile = 0 #bytes thing again
            
            if oned != None: #i think?
                flux = np.array(cube.dat)[:, i]
                err = []
                for a in cube.var[:, i]:
                    err.append(np.sqrt(abs(a)))
                dq = cube.dq[:, i]
                labin = str(i + 1).zfill(4)
                labout = labin
            else:
                if verbose != None:
                    print(' Row ' + str(j + 1) + ' of ' + str(cube.nrows))
                
                if 'vormap' in initdat:
                    if np.isfinite(initdat['vormap'][i][j]) and \
                            (initdat['vormap'][i][j] is not bad):
                        iuse = vorcoords[initdat['vormap'][i][j] - 1, 0]
                        juse = vorcoords[initdat['vormap'][i][j] - 1, 1]
                    else: 
                        novortile = 1 #this byte thing again
                else:
                    iuse = i
                    juse = j
                
                if novortile == 1:#?? or 0? Or none?
                    flux = np.array(cube.dat)[juse, iuse, :].flatten()
                    err = np.array(math.sqrt(abs(cube.var[juse, iuse, :]))).flatten()
                    dq = np.array(cube.dq)[juse, iuse, :].flatten()
                    labin = str(juse + 1) + '_' + str(iuse + 1) #swapped here too
                    labout = str(j + 1) + '_' + str(i + 1)
            
            #Line 344
            if novortile == 1: #??
                infile = str(initdat['outdir']) + str(initdat['label']) \
                    + '_' + labin + '.xdr'
                outfile = initdat['outdir'] + initdat['label'] + '_' + labout
                nodata = np.where(flux != 0.0) #maybe not 0.0?
                ct = len(nodata)
                #check if infile exists:
                filepresent = True
                try:
                   fo = open(infile, "r") #extension?
                   #process after opening file
                   pass
                   fo.close()
                except IOError:
                   print ("File doesn't exist")
                   filepresent = False
                   ct = 0
            
            nofit = False
            
            if filepresent == False or ct < 0:
                nofit = True
                badmessage = 'No data for ' + str(i + 1) + ', ' + \
                    str(j + 1) + '.'
                print(badmessage)
            
            
            struct = np.load("struct.npy", allow_pickle='TRUE').item()
            struct['noemlinfit'] = err[struct['fitran_indx']] #necessary?
            
            if not 'noemlinfit' in struct:
                #get line fit params
                tflux = True #or false?
                linepars = sepfitpars(linelist, struct['param,struct']['perror'], \
                                      struct['parinfo'], tflux = tflux, \
                                      doublets = emldoublets)
                lineweqs = cmpweq(struct, linelist, doublets = emldoublets)
#plot emission line data, print data to a file
            if noplots == None:
                #plot emission lines
                if not 'noemlinfit' in struct:
                    if not 'nolines' in linepars:                        
                        if 'fcnpltlin' in initdat:
                            fcnpltlin = initdat['fcnpltlin']
                        else: fcnpltlin = 'ifsf_pltlin'
                        if 'argspltlin1' in initdat:
                            module = importlib.import_module('q3dfit.common.' + fcnpltlin)                 
                            pltlinfcn = getattr(module, initproc)    
                            pltlinfcn(struct, initdat['argspltlin1'], \
                                   outfile + '_lin1')
                        if 'argspltlin2' in initdat:
                            module = importlib.import_module('q3dfit.common.' + fcnpltlin)                 
                            pltlinfcn = getattr(module, initproc)    
                            pltlinfcn(struct, initdat['argspltlin2'], \
                                   outfile + '_lin2')
            #printfirpar
            
            if not 'noemlinfit' in struct:
                thisncomp = 0
                thisncompline = ''
                
                for line in lines_with_doublets:
                    sigtmp = linepars['sigma'][:, line]
                    fluxtmp = linepars['flux'][:, line]
                    igd = np.where(sigtmp[i] != False and sigtmp[i] != bad and \
                            fluxtmp != False and fluxtmp != bad)
                    ctgd = len(igd)
                    
                    if ctgd > thisncomp:
                        thisncomp = ctgd
                        thisncompline = line
                    
                    if ctgd > 0:
                        emlweq['ftot', line, j, i] = lineweqs['tot'][line]
                        emlflx['ftot', line, j, i] = tflux['tflux'][line]
                        emlflxerr['ftot', line, i, j] = tflux['tfluxerr'][line]
                
                if thisncomp == 1:
                    isort = 0
                    if 'flipsort' in initdat:
                        if flipsort[j, i]:
                            print('Flipsort set for spaxel [' + str(i + 1) \
                                   + ',' + str(j + 1) + '] but ' + \
                                  'only 1 component. Setting to 2 components \
                                   and ' + 'flipping anyway.')
                            isort = [1, 0]
                elif thisncomp > 2:
                    #sort components
                    igd = np.arange(thisncomp)
                    indices = np.arange(initdat['maxncomp'])
                    sigtmp = linepars['sigma'][:, thisncompline]
                    fluxtmp = linepars['flux'][:, thisncompline]
                    if not 'sorttype' in initdat:
                        isort = sigtmp[igd].sort()
                    elif initdat['sorttype'] == 'wave': #strings
                        isort = linepars['wave'][igd, line].sort() #reversed?
                    elif initdat['sorttype'] == 'reversewave':
                        isort = linepars['wave'][igd, line].sort(reverse = True)
                    
                    if 'flipsort' in initdat:
                        if flipsort[j, i] != None:
                            isort = isort.sort(reverse = True)
                if thisncomp > 0:
                    for line in lines_with_doublets:
                        kcomp = 1
                        for sindex in isort:
                            cstr='c' + str(kcomp)
                            emlwav[cstr][line][i, j] \
                                = linepars['wave'].cell(line, sindex)
                            emlwaverr[cstr][line][i, j] \
                                = linepars['waveerr'].cell(line, sindex)
                            emlsig[cstr][line][i, j] \
                                = linepars['sigma'].cell(line, sindex)
                            emlsigerr[cstr][line][i, j] \
                                = linepars['sigmaerr'].cell(line, sindex)
                            emlweq['f' + cstr][line][i, j] \
                                = lineweqs['comp'].cell(line, sindex)
                            emlflx['f' + cstr][line][i, j] \
                                = linepars['flux'].cell(line,sindex)
                            emlflxerr['f' + cstr][line][i, j] \
                                = linepars['fluxerr'].cell(line,sindex)
                            emlflx['f' + cstr + 'pk'][line][i, j] \
                                = linepars['fluxpk'].cell(line, sindex)
                            emlflxerr['f' + cstr + 'pk'][line][i, j] \
                                = linepars['fluxpkerr'].cell(line, sindex)
                            kcomp+=1 
                    #print line fluxes to text file
                    #printlinpar, ~line 474
                    
#Process and plot continuum data
              #make and populate output data cubes          
            if firstcontproc != 0: #i think
                hostcube = \
                   {'dat': np.zeros(cube.nrows, cube.ncols, cube.nz), \
                    'err': np.zeros(cube.nrows, cube.ncols, cube.nz), \
                    'dq':  np.zeros(cube.nrows, cube.ncols, cube.nz), \
                    'norm_div': np.zeros(cube.nrows, cube.ncols, cube.nz), \
                    'norm_sub': np.zeros(cube.nrows, cube.ncols, cube.nz)}
              
                if 'decompose_ppxf_fit' in initdat:
                    contcube = \
                        {'wave': struct['wave'], \
                         'all_mod': np.zeros(cube.nrows, cube.ncols, cube.nz), \
                         'stel_mod': np.zeros(cube.nrows, cube.ncols, cube.nz), \
                         'poly_mod': np.zeros(cube.nrows, cube.ncols, cube.nz), \
                         'stel_mod_tot': np.zeros(cube.nrows, cube.ncols) + bad, \
                         'poly_mod_tot': np.zeros(cube.nrows, cube.ncols) + bad, \
                         'poly_mod_tot_pct': np.zeros(cube.nrows, cube.ncols) + bad, \
                         'stel_sigma': np.zeros(cube.nrows, cube.ncols) + bad, \
                         'stel_sigma_err': np.zeros(cube.nrows, cube.ncols) + bad, \
                         'stel_z': np.zeros(cube.nrows, cube.ncols) + bad, \
                         'stel_z_err': np.zeros(cube.nrows, cube.ncols) + bad, \
                         'stel_rchisq': np.zeros(cube.nrows, cube.ncols) + bad, \
                         'stel_ebv': np.zeros(cube.nrows, cube.ncols) + bad, \
                         'stel_ebv_err': np.zeros(cube.nrows, cube.ncols) + bad}
              
                elif 'decompose_qso_fit' in initdat:
                    contcube = \
                        {'wave': struct['wave'], \
                         'qso_mod': np.zeros(cube.nrows, cube.ncols, cube.nz), \
                         'qso_poly_mod': np.zeros(cube.nrows, cube.ncols, cube.nz), \
                         'host_mod': np.zeros(cube.nrows, cube.ncols, cube.nz), \
                         'poly_mod': np.zeros(cube.nrows, cube.ncols, cube.nz), \
                         'npts': np.zeros(cube.nrows, cube.ncols) + bad, \
                         'stel_sigma': np.zeros(cube.nrows, cube.ncols) + bad, \
                         'stel_sigma_err': np.zeros(cube.nrows, cube.ncols) + bad, \
                         'stel_z': np.zeros(cube.nrows, cube.ncols) + bad, \
                         'stel_z_err': np.zeros(cube.nrows, cube.ncols) + bad, \
                         'stel_rchisq': np.zeros(cube.nrows, cube.ncols) + bad, \
                         'stel_ebv': np.zeros(cube.nrows, cube.ncols) + bad, \
                         'stel_ebv_err': np.zeros(cube.nrows, cube.ncols) + bad}
                else:
                    contcube = \
                        {'all_mod': np.zeros(cube.nrows, cube.ncols, cube.nz), \
                         'stel_z': np.zeros(cube.nrows, cube.ncols) + bad, \
                         'stel_z_err': np.zeros(cube.nrows, cube.ncols) + bad, \
                         'stel_rchisq': np.zeros(cube.nrows, cube.ncols) + bad, \
                         'stel_ebv': np.zeros(cube.nrows, cube.ncols) + bad, \
                         'stel_ebv_err': np.zeros(cube.nrows, cube.ncols) + bad}
                firstcontproc = 0
              
            hostcube['dat'][j, i, struct['fitran_indx']] = struct['cont_dat']
            hostcube['err'][j, i, struct['fitran_indx']] = err[struct['fitran_indx']]
            hostcube['dq'][j, i, struct['fitran_indx']] = dq[struct['fitran_indx']]
            hostcube['norm_div'][j, i, struct['fitran_indx']] \
                = np.divide(struct['cont_dat'], struct['cont_fit'])
            hostcube['norm_sub'][j, i, struct['fitran_indx']] \
                = np.subtract(struct['cont_dat'], struct['cont_fit'])
              
            if 'decompose_ppxf_fit' in initdat:
                add_poly_degree = 4.0 #should match fitspec
                if 'argscontfit' in initdat:
                    if 'add_poly_degree' in initdat['argscontfit']:
                        add_poly_degree = initdat['argscontfit']['add_poly_degree']
                #Compute polynomial
                dumy_log, wave_log = log_rebin([struct['wave'][0], \
                      struct['wave'][len(struct['wave'])-1]], \
                      struct['spec'])
                xnorm = cap_range(-1.0, 1.0, len(wave_log))
                cont_fit_poly_log = 0.0
                for k in range (0, add_poly_degree):
                    cfpllegfun = legendre(k)
                    cont_fit_poly_log += cfpllegfun(xnorm) * struct['ct_add_poly_weight'][k]
                interpfunction = interpolate.interp1d(cont_fit_poly_log, wave_log, kind='linear')
                cont_fit_poly = interpfunction(np.log(struct['wave']))
                #Compute stellar continuum
                cont_fit_stel = np.subtract(struct['cont_fit'], cont_fit_poly)
                #Total flux fromd ifferent components
                cont_fit_tot = np.sum(struct['cont_fit'])
                contcube['all_mod'][j, i, struct['fitran_indx']] = struct['cont_fit']
                contcube['stel_mod'][j, i, struct['fitran_indx']] = cont_fit_stel
                contcube['poly_mod'][j, i, struct['fitran_indx']] = cont_fit_poly
                contcube['stel_mod_tot'][j, i] = np.sum(cont_fit_stel)
                contcube['poly_mod_tot'][j, i] = np.sum(cont_fit_poly)
                contcube['poly_mod_tot_pct'][j, j] \
                    = np.divide(contcube['poly_mod_tot'][j, i], cont_fit_tot)
                contcube['stel_sigma'][j, i] = struct['ct_ppxf_sigma']
                contcube['stel_z'][j, i] = struct['zstar']
                
                if 'ct_errors' in struct:
                    contcube['stel_sigma_err'][j, i, :] \
                        = struct['ct_errors']['ct_ppxf_sigma']
                    #assuming that ct_errors is a dictionary
                else: #makes an array with two arrays
                    contcube['stel_sigma_err'][j, i, :] \
                        = [struct['ct_ppxf_sigma_err'], struct['ct_ppxf_sigma_err']]
                
                if 'ct_errors' in struct:                    
                    contcube['stel_z_err'][j, i, :] = struct['ct_errors']['zstar']
                else: 
                    contcube['stel_z_err'][j, i, :] \
                        = [struct['zstar_err'], struct['zstar_err']]
            
            elif 'decompose_qso_fit' in initdat:
                if initdat['fcncontfit'] == 'fitqsohost':
                    if 'qsoord' in initdat['argscontfit']:
                        qsoord = initdat['argscontfit']['qsoord']
                    else: qsoord = False #?
                    
                    if 'hostord' in initdat['argscontfit']:
                        hostord = initdat['argscontfit']['hostord'] 
                    else: hostord = False #?
                    
                    if 'blrpar' in initdat['argscontfit']:
                        blrterms = len(initdat['argscontfit']['blrpar'])  #blrpar a 1D array
                    else: blrterms = 0 #?
                    #default here must be same as in IFSF_FITQSOHOST
                    if 'add_poly_degree' in initdat['argscontfit']:
                        add_poly_degree = initdat['argscontfit']['add_poly_degree']
                    else: add_poly_degree = 30
                    
                    #These lines mirror ones in IFSF_FITQSOHOST
                    struct_tmp = struct

                    # Get and renormalize template (check to see what name file is saved under - a variable?)
                    qsotemplate = np.load("pg1411nucleartemplate.npy", allow_pickle='TRUE').item()                    
                    qsowave = qsotemplate['wave']
                    qsoflux_full = qsotemplate['flux']
                    iqsoflux = np.where(qsowave > struct_tmp['fitran'][0]*0.99999 and \
                                   qsowave < struct_tmp['fitran'][1]*1.00001)
                    #line 611      
                    qsoflux = qsoflux_full[iqsoflux]
                    qsoflux /= np.median(qsoflux)
                    struct = struct_tmp
                    #If polynomial residual is re-fit with PPXF, separate out best-fit
                    #parameter structure created in IFSF_FITQSOHOST and compute polynomial
                    #and stellar components
                    if 'refit' in initdat['argscontfit']:
                        par_qsohost = struct['ct_coeff']['qso_host']
                        par_stel = struct['ct_coeff']['stel']
                        #line 622
                        dumy_log, wave_rebin = log_rebin([struct['wave'][0], \
                            struct['wave'][len(struct['wave'])-1]], \
                            struct['spec'])
                        xnorm = cap_range(-1.0, 1.0, len(wave_log)) #1D?
                        if add_poly_degree > 0:
                            par_poly = struct['ct_coeff']['poly']
                            polymod_log = 0.0 # Additive polynomial
                            for k in range (0, add_poly_degree):
                                cfpllegfun = legendre(k)
                                polymod_log += cfpllegfun(xnorm) * par_poly[k]
                            interpfunct = interpolate.interp1d(polymod_log, wave_log, kind='linear')
                            polymod_refit = interpfunct(np.log(struct['wave']))
                        else:
                            polymod_refit = np.zeros(struct['wave'], dtype = float)
                        contcube['stel_sigma'][j, i] = struct['ct_coeff']['ppxf_sigma']
                        contcube['stel_z'][j, i] = struct['zstar']
                        
                        #Don't know ct_error's type
                        if 'ct_errors' in struct:
                            contcube['stel_sigma_err'][j, i, :] \
                                = struct['ct_errors']['ct_ppxf_sigma']
                        else:
                            contcube['stel_sigma_err'][j, i, :] \
                                = [struct['ct_ppxf_sigma_err'], struct['ct_ppxf_sigma_err']]
                        if 'ct_errors' in struct:
                            contcube['stel_z_err'][j, i, :] \
                                = struct['ct_errors']['zstar']
                        else: 
                            contcube['stel_z_err'][j, i, :] \
                                = [struct['zstar_err'], struct['zstar_err']]
                        #again why aren't those two if statements combined
                    else:
                        par_qsohost = struct['ct_coeff']
                        polymod_refit = 0.0
            
                    #produce fit with template only and with template + host. Also
                    #output QSO multiplicative polynomial
                    qsomod_polynorm = 0.0
                    qsomod = qsohostfcn(struct['wave'], par_qsohost, qsoflux = qsoflux, \
                                      qsoonly = True, blrterms = blrterms, \
                                      qsoscl = qsomod_polynorm, qsoord = qsoord, \
                                      hostord = hostord)
                    hostmod = struct['cont_fit_pretweak'] - qsomod
                    
                    #if continuum is tweaked in any region, subide resulting residual 
                    #proportionality @ each wavelength btwn qso and host components
                    qsomod_notweak = qsomod
                    if 'tweakcntfit' in initdat:
                        modresid = struct['cont_fit'] - struct['cont_fit_pretweak']
                        inz = np.where(qsomod != 0 and hostmod != 0)
                        qsofrac = np.array(len(qsomod), dtype = float)
                        for ind in inz:
                            qsofrac[ind] = qsomod[ind] / (qsomod[ind] + hostmod[ind])
                        qsomod += modresid * qsofrac
                        hostmod += modresid * (1.0 - qsofrac)
                    #components of qso fit for plotting
                    qsomod_normonly = qsoflux
                    if 'blrpar' in initdat['argscontfit']:
                        qsomod_blronly = qsohostfcn(struct['wave'], par_qsohost, \
                                         qsoflux = qsoflux, blronly = True, \
                                         blrterms = blrterms, qsoord = qsoord, \
                                         hostord = hostord)
            elif initdat['fcncontfit'] == 'ppxf' and 'qsotempfile' in initdat:
                struct_star = struct
                #???
                qsotempfile = np.load("qsotempfile.npy", allow_pickle='TRUE').item()                    
                struct_qso = struct
                struct = struct_star
                qsomod = struct_qso['cont_fit'] * struct['ct_coeff'][len(struct['ct_coeff']) - 1]
                hostmod = struct['cont_fit'] - qsomod
            else:
                contcube['all_mod'][j, i, struct['fitran_indx']] = struct['cont_fit']
                contcube['stel_z'][j, i] = struct['zstar']
                if 'ct_errors' in struct:
                    contcube['stel_z_err'][j, i, :] = struct['ct_errors']['zstar']
                #for backwards compatibility
                elif 'wstar_err' in struct:
                    contcube['stel_z_err'][j, i, :] = [struct['zstar_err'], struct['zstar_err']]
                else: contcube['stel_z_err'][j, i, :] = [0, 0]
            
            contcube['stel_ebv'][j, i] = struct['ct_ebv']
            if 'ct_errors' in struct:
                contcube['stel_ebv_err'][j, i, :] = struct['ct_errors']['ct_ebv']
            #for backwards compatibility
            if 'stel_rchisq' in struct: 
                contcube['stel_rchisq'][j, i] = struct['ct_rchisq']
            else: contcube['stel_rchisq'][j, i] = 0.0

            #Print ppxf results to stdout
            if 'decompose_ppxf_fit' in initdat or 'decompose_qso_fit' in initdat:
                if 'argscontfit' in initdat:
                    if 'print_output' in initdat['argscontfit']:
                        print("PPXF results: ")
                        if 'decompose_ppxf_fit' in initdat:
                            ct_coeff_tmp = struct['ct_coeff']
                            poly_tmp_pct = contcube['poly_mod_tot_pct'][j, i]
                        else:
                            ct_coeff_tmp = struct['ct_coeff']['stel']
                            poly_tmp_pct = np.sum(polymod_refit) / np.sum(hostmod)
                        inz = np.where(ct_coeff_tmp != 0.0)
                        ctnz = len(inz)
                        if ctnz > 0:
                            coeffgd = ct_coeff_tmp[inz]
                            #normalize coefficients to % of total stellar coeffs.
                            totcoeffgd = np.sum(coeffgd)
                            coeffgd /= totcoeffgd
                            #re-normalize to % of total flux
                            coeffgd *= (1.0 - poly_tmp_pct)
                            starttempfile = np.load(initdat['starttempfile'] + ".npy", allow_pickle='TRUE').item()                    
                            agesgd = starttempfile['ages'][inz] #check
                            #sum coefficients over age ranges
                            iyoung = np.where(agesgd < 1e7)
                            ctyoung = len(iyoung)
                            iinter1 = np.where(agesgd > 1e7 and agesgd < 1e8)
                            ctinter1 = len(iinter1)
                            iinter2 = np.where(agesgd > 1e8 and agesgd < 1e9)
                            ctinter2 = len(iinter2)
                            iold = np.where(agesgd > 1e9)
                            ctold = len(iold)
                            if ctyoung > 0:
                                coeffyoung = np.sum(coeffgd[iyoung]) * 100.0
                            else: coeffyoung = 0.0
                            if ctinter1 > 0:
                                coeffinter1 = np.sum(coeffgd[iinter1]) * 100.0
                            else: coeffinter1 = 0.0
                            if ctinter2 > 0:
                                coeffinter2 = np.sum(coeffgd[iinter2]) * 100.0
                            else: coeffinter2 = 0.0
                            if ctold > 0:
                                coeffold = np.sum(coeffgd[iold]) * 100.0
                            else: coeffold = 0.0
                            print(str(round(coeffyoung)) + \
                                ' contribution from ages <= 10 Myr.')
                            print(str(round(coeffinter1)) + \
                                ' contribution from 10 Myr < age <= 100 Myr.')
                            print(str(round(coeffinter2)) + \
                                ' contribution from 100 Myr < age <= 1 Gyr.')
                            print(str (round(coeffold)) + \
                                ' contribution from ages > 1 Gyr.')
                        print(' Stellar template convolved with sigma = ' + \
                              str(struct['ct_ppxf_sigma']) + 'km/s')
           
            #Plot QSO and host only continuum fit
            if 'decompose_qso_fit' in initdat:
                struct_host = struct
                struct_host['spec'] -= qsomod
                struct_host['cont_dat'] -= qsomod
                struct_host['cont_fit'] -= qsomod
                struct_qso = struct
                struct_qso['spec'] -= hostmod
                struct_qso['cont_dat'] -= hostmod
                struct_qso['cont_fit'] -= hostmod
                contcube['qso_mod'][j, i, struct['fitran_indx']] = qsomod
                contcube['qso_poly_mod'][j, i, struct['fitran_indx']] = qsomod_polynorm
                contcube['host_mod'][j, i, struct['fitran_indx']] = hostmod
                contcube['poly_mod'][j, i, struct['fitran_indx']] = polymod_refit
                contcube['npts'][j, i] = len(struct['fitran_indx'])
                if 'remove_scattered' in initdat:
                    contcube['host_mod'][j, i, struct['fitran_indx']] -= polymod_refit
                #Update hostcube.dat to remove tweakcnt mods
                #Data minus (emission line model + QSO model, tweakcnt mods not 
                #included in QSO model)     
                hostcube['dat'][j, i, struct['fitran_indx']] \
                    = struct['cont_dat'] - qsomod_notweak
                if noplots != None and np.sum(struct_host['cont_fit']) != 0.0:
                    if 'refit' in initdat['argscontfot']:
                        compspec = [[polymod_refit], [hostmod-polymod_refit]]
                        comptit = ['ord. ' + str(add_poly_degree) + \
                            ' Leg. poly.', 'stel. temp.']
                    else:
                        compspec = hostmod
                        comptit = ['exponential terms']
                    if 'argspltcont' in initdat:
                        module = importlib.import_module('q3dfit.common.' + fcnpltcont)                 
                        pltcontfcn = getattr(module, fcnpltcont)    
                        pltcontfcn(struct_host, str(outfile) + '_cnt_host', \
                                compspec = compspec, comptit = comptit, \
                                title='Host', fitran = initdat['fitran'], \
                                **initdat['argspltcont']) #extra thing again
                    else:
                        module = importlib.import_module('q3dfit.common.' + fcnpltcont)                 
                        pltcontfcn = getattr(module, fcnpltcont)    
                        pltcontfcn(struct_host, str(outfile) + '_cnt_host', \
                                compspec = compspec, comptit = comptit, \
                                title='Host', fitran = initdat['fitran'])
                    if 'blrpar' in initdat['argscontfit']:
                        qsomod_blrnorm = np.median(qsomod) / max(qsomod_blronly)
                        compspec = [[qsomod_normonly], [qsomod_blronly * qsomod_blrnorm]]
                        comptit = ['raw template', 'scattered\times' + \
                            str(qsomod_blrnorm)]
                    else:
                        compspec = [[qsomod_normonly]]
                        comptit = ['raw template']
                    if 'argspltcont' in initdat:
                        module = importlib.import_module('q3dfit.common.' + fcnpltcont)                 
                        pltcontfcn = getattr(module, fcnpltcont)    
                        pltcontfcn(struct_qso, str(outfile) + '_cnt_qso', \
                                compspec = compspec, comptit = comptit, \
                                title = 'QSO', fitran = initdat['fitran'], \
                                **initdat['argspltcont'])
                    else:
                        module = importlib.import_module('q3dfit.common.' + fcnpltcont)                 
                        pltcontfcn = getattr(module, fcnpltcont)    
                        pltcontfcn(struct_qso, outfile + '_cnt_qso', \
                                compspec = compspec, comptit = comptit, \
                                title = 'QSO', fitran = initdat['fitran'])
            #Plot continuum
            #Make sure fit doesn't indicate no continuum; avoids
            #plot range error in continuum fitting routine, as well as a blank
            #plot!    
            if noplots != None and sum(struct['cont_fit']) != 0.0:
                if 'decompose_qso_fit' in initdat:
                    if 'argspltcont' in initdat:
                        module = importlib.import_module('q3dfit.common.' + fcnpltcont)                 
                        pltcontfcn = getattr(module, fcnpltcont)    
                        pltcontfcn(struct, outfile + '_cnt', \
                                compspec = [[qsomod],[hostmod]], \
                                title = 'Total', comptit = ['QSO','host'], \
                                fitran = initdat.fitran, **initdat['argspltcont'])
                    else:
                        module = importlib.import_module('q3dfit.common.' + fcnpltcont)                 
                        pltcontfcn = getattr(module, fcnpltcont)    
                        pltcontfcn(struct, outfile + '_cnt', \
                                compspec = [[qsomod],[hostmod]], \
                                title = 'Total', comptit = ['QSO', 'host'], \
                                fitran = initdat['fitran'])
                elif 'decompose_ppxf_fit' in initdat:
                    if 'argspltcont' in initdat:
                        module = importlib.import_module('q3dfit.common.' + fcnpltcont)                 
                        pltcontfcn = getattr(module, fcnpltcont)    
                        pltcontfcn(struct, outfile + '_cnt', \
                                compspec = [[cont_fit_stel], [cont_fit_poly]], \
                                title='Total', \
                                comptit = ['stel. temp.', \
                                           'ord. ' + str(add_poly_degree) + \
                                         ' Leg. poly'], \
                                fitran = initdat['fitran'], **initdat['argspltcont'])
                    else:
                        module = importlib.import_module('q3dfit.common.' + fcnpltcont)                 
                        pltcontfcn = getattr(module, fcnpltcont)    
                        pltcontfcn(struct, outfile + '_cnt', \
                                compspec = [[cont_fit_stel], [cont_fit_poly]], \
                                title = 'Total', \
                                comptit = ['stel. temp.', 'ord. ' + \
                                           str(add_poly_degree) + ' Leg. poly'], \
                                fitran = initdat['fitran'])
                else:
                    if 'argspltcont' in initdat:
                        module = importlib.import_module('q3dfit.common.' + fcnpltcont)                 
                        pltcontfcn = getattr(module, fcnpltcont)    
                        pltcontfcn(struct, outfile + '_cnt', \
                                   fitran = initdat['fitran'], **initdat['argspltcont'])
                    else:
                        module = importlib.import_module('q3dfit.common.' + fcnpltcont)                 
                        pltcontfcn = getattr(module, fcnpltcont)    
                        pltcontfcn(struct, outfile + '_cnt', \
                                   fitran = initdat['fitran'])

#;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;
#; Process NaD (normalize, compute quantities and save, plot)
#;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;


            #             if tag_exist(initdat,'donad') then begin
            # if tag_exist(initdat,'decompose_qso_fit'):
            #     if tag_exist(initdat,'remove_scattered'):
            #         hostmod_tmp = hostmod - polymod_refit
            #         qsomod_tmp = qsomod + polymod_refit
            #     else:
            #         hostmod_tmp = hostmod
            #         qsomod_tmp = qsomod
        #         nadnormcont = (struct.cont_dat - qsomod_tmp)/hostmod_tmp
        #         nadnormconterr = struct.spec_err/hostmod_tmp
        #         nadnormstel = hostmod_tmp
        #     else:
        #         nadnormcont = struct.cont_dat/struct.cont_fit
        #         nadnormconterr = struct.spec_err/struct.cont_fit
        #         nadnormstel = struct.cont_fit
        #     if tag_exist(initnad,'argsnormnad'):
        #         normnad = ifsf_normnad(struct.wave,\
        #                              nadnormcont,\
        #                              nadnormconterr,\
        #                              struct.zstar,fitpars_normnad,\
        #                              _extra=initnad.argsnormnad)
        #         normnadem = ifsf_normnad(struct.wave,\
        #                                struct.emlin_dat,\
        #                                struct.spec_err,\
        #                                struct.zstar,fitpars_normnadem,\
        #                                /nosncut,/subtract,\
        #                                _extra=initnad.argsnormnad)
        #         normnadstel = ifsf_normnad(struct.wave,\
        #                                  nadnormstel,\
        #                                  struct.spec_err,\
        #                                  struct.zstar,fitpars_normnadstel,\
        #                                  _extra=initnad.argsnormnad)
        #         if ~ tag_exist(initnad.argsnormnad,'fitranlo'): 
        #             fitranlo = (1d +struct.zstar)*[5810d,5865d] \
        #         else:
        #             fitranlo = initnad.argsnormnad.fitranlo
        #         if ~ tag_exist(initnad.argsnormnad,'fitranhi'): 
        #             fitranhi = (1d +struct.zstar)*[5905d,5960d] 
        #         else: 
        #             fitranhi = initnad.argsnormnad.fitranhi
        #     else:
        #         normnad = ifsf_normnad(struct.wave,\
        #                              nadnormcont,\
        #                              nadnormconterr,\
        #                              struct.zstar,fitpars_normnad)
        #         normnadem = ifsf_normnad(struct.wave,\
        #                                struct.emlin_dat,\
        #                                struct.spec_err,\
        #                                struct.zstar,fitpars_normnadem,\
        #                                /nosncut,/subtract)
        #         normnadstel = ifsf_normnad(struct.wave,\
        #                                  nadnormstel,\
        #                                  struct.spec_err,\
        #                                  struct.zstar,fitpars_normnadstel)
        #         fitranlo = (1d +struct.zstar)*[5810d,5865d]
        #         fitranhi = (1d +struct.zstar)*[5905d,5960d]
        #     #Check data quality
        #     if normnad != None:
        #         igd = np.where(normnad.nflux gt 0d,ctgd) \
        #     else: 
        #         ctgd = 0
        #     #Compute empirical equivalent widths and emission-line fluxes
        #     if ctgd gt 0:
        #     #Create output data cube
        #         if firstnadnorm:
        #             ilo = value_locate(cube.wave,fitranlo[0])+1
        #             ihi = value_locate(cube.wave,fitranhi[1])
        #             dat_normnadwave = cube.wave[ilo:ihi]
        #             nz = ihi-ilo+1
        #             nadcube = \
        #                {'wave': np.zeros(cube.ncols,cube.nrows,nz),\
        #                 'cont': np.zeros(cube.ncols,cube.nrows,nz),\
        #                 'dat:' np.zeros(cube.ncols,cube.nrows,nz),\
        #                 'err': np.zeros(cube.ncols,cube.nrows,nz)+bad,\
        #                 'weq': np.zeros(cube.ncols,cube.nrows,4)+bad,\
        #                 'stelweq': np.zeros(cube.ncols,cube.nrows,2)+bad,\
        #                 'iweq': np.zeros(cube.ncols,cube.nrows,4)+bad,\
        #                 'emflux': np.zeros(cube.ncols,cube.nrows,2)+bad,\
        #                 'emul': np.zeros(cube.ncols,cube.nrows,4)+bad,\
        #                 'vel': np.zeros(cube.ncols,cube.nrows,6)+bad}
        #             firstnadnorm = 0
        #     #Defaults
        #     emflux=np.zeros(2)
        #     emul=np.zeros(4)+bad
        #     vel = np.zeros(6)+bad
        #     if tag_exist(initnad,'argsnadweq'):
        #     weq = ifsf_cmpnadweq(normnad.wave,normnad.nflux,normnad.nerr,\
        #                                snflux=normnadem.nflux,unerr=normnadem.nerr,\
        #                                emflux=emflux,emul=emul,\
        #                                _extra=initnad.argsnadweq)
        
        #           #These need to be compatible with the IFSF_CMPNADWEQ defaults
        #     if tag_exist(initnad.argsnadweq,'emwid'):
        #         emwid=initnad.argsnadweq.emwid 
        #     else:
        #         emwid=20d
        #     if tag_exist(initnad.argsnadweq,'iabsoff'): 
        #         iabsoff=initnad.argsnadweq.iabsoff 
        #     else: 
        #         iabsoff=4
        #       endif else begin
        #          weq = ifsf_cmpnadweq(normnad.wave,normnad.nflux,normnad.nerr,\
        #                               snflux=normnadem.nflux,unerr=normnadem.nerr,\
        #                               emflux=emflux,emul=emul)
        #          These need to be compatible with the IFSF_CMPNADWEQ defaults
        #          emwid=20d
        #          iabsoff=4l
        #       endelse
        #       Compute stellar continuum NaD equivalent widths from fit
        #       stelweq = ifsf_cmpnadweq(normnadstel.wave,normnadstel.nflux,\
        #                                normnadstel.nerr,\
        #                                wavelim=[5883d*(1d +initdat.zsys_gas),\
        #                                         6003d*(1d +initdat.zsys_gas),\
        #                                         0d,0d])
        
        #       Compute empirical velocities
        #       size_weq = size(weq)
        #       if size_weq[0] eq 2 then begin
        #          if tag_exist(initnad,'argsnadvel') then \
        #             vel = ifsf_cmpnadvel(normnad.wave,normnad.nflux,normnad.nerr,\
        #                                  weq[*,1],initdat.zsys_gas,\
        #                                  _extra=initnad.argsnadvel) \
        #          else vel = ifsf_cmpnadvel(normnad.wave,normnad.nflux,normnad.nerr,\
        #                                    weq[*,1],initdat.zsys_gas)
        #       endif
              
        #       ilo = where(normnad.wave[0] eq dat_normnadwave)
        #       ihi = where(normnad.wave[n_elements(normnad.wave)-1] \
        #             eq dat_normnadwave)
        
        #       Assume that stellar fit is a good model but that the error spectrum
        #       may not be perfect. Correct using stellar reduced chi squared
        #       if tag_exist(initnad,'errcorr_ctrchisq') then begin
        #          normnad.nerr *= struct.ct_rchisq
        #          weq[1,0] *= struct.ct_rchisq
        #          weq[3,0] *= struct.ct_rchisq
        #       endif
        #       nadcube.wave[i,j,*] = dat_normnadwave
        #       nadcube.cont[i,j,ilo:ihi] = struct.cont_fit[normnad.ind]
        #       nadcube.dat[i,j,ilo:ihi] = normnad.nflux
        #       nadcube.err[i,j,ilo:ihi] = normnad.nerr
        #       nadcube.weq[i,j,*] = weq[*,0]
        #       nadcube.iweq[i,j,*] = weq[*,1]
        #       nadcube.stelweq[i,j,*] = stelweq[0:1]
        #       nadcube.emflux[i,j,*] = emflux
        #       nadcube.emul[i,j,*] = emul
        #       nadcube.vel[i,j,*] = vel
        #       Plot data
        #       if not keyword_set(noplots) then \
        #          if tag_exist(initnad,'argspltnormnad') then \
        #             ifsf_pltnaddat,normnad,fitpars_normnad,struct.zstar,\
        #                            outfile+'_nad_norm',autoindices=weq[*,1],\
        #                            emwid=emwid,iabsoff=iabsoff,\
        #                            _extra=initnad.argspltnormnad else \
        #             ifsf_pltnaddat,normnad,fitpars_normnad,struct.zstar,\
        #                            outfile+'_nad_norm',autoindices=weq[*,1],\
        #                            emwid=emwid,iabsoff=iabsoff
        #    endif
        
        # endif
        
        # endelse
        
        # for

def cap_range(x1, x2, n):
    a = np.zeros(1, dtype = float)
    interval = (x2 - x1) / (n - 1)
    print(interval)
    num = x1
    for i in range (0, n):
        print(num)        
        a = np.append(a, num)
        num += interval
    a = a[1:]
    return a

def array_indices(array, index):
    height = len(array[0])
    x = index // height
    y = index % height
    return x, y
