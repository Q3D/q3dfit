# -*- coding: utf-8 -*-
"""
Routine to plot the continuum and emission lines fits to a spectrum, doesn't
return anything.
@author: hadley
"""
import numpy as np
import math
import pdb
import importlib
from q3dfit.common.linelist import linelist
from q3dfit.common.readcube import CUBE
from scipy.special import legendre #?
from scipy import interpolate

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
        
        #don't know how tocheck if a column name exists in astropy table?
        keys = listlines.keys() 
        
        for i in range (0, ndoublets):
            if (emldoublets[i][0] in keys) and \
                (emldoublets[i][1] in keys):
                dkey = emldoublets[i][0] + "+" + emldoublets[i][1]
                lines_with_doublets = [lines_with_doublets] + [dkey]
        
        #Haven't tested with linelist() yet
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
    
    header = bytes(1) #come back to this bc idk how it works rn
    
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
            ctivor = len(xyvor)
            vorcoords[i - 1, 0] = xyvor[0]
            vorcoords[i - 1, 1] = xyvor[1] #i'm only guessing, need to see what 
                                            #dimensions vorcoords etc. are


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
        flipsort = np.zeros(cube['nrows'], cube['ncols'])
        sizefs = len(initdat['flipsort'])
        for i in range (0, len(sizefs[0])): #probably right but check again lol
            flipsort[initdat['flipsort'][0][i] - 1] \
                [initdat['flipsort'][1][i] - 1] = bytes(5) #also come back to this
 
#LOOP THROUGH SPAXELS
    #switch to track when first NaD normalization done
    firstnadnorm = 1
    #switch to track when first continuum processed
    firstcontproc = 1
    
    if cols != None:
        cols = [1, cube.ncols] #makes an array with two elements?
    elif len(cols) == 1: cols = [cols, cols]
    np.array(cols).astype("int")
   
    for i in range (cols[0] - 1, cols[1]):
        if verbose != None:
            print('Column ' + (i + 1) + ' of ' + cube.ncols) 
        
        if rows != None: rows = [1, cube.nrows]
        elif len(rows) == 1: rows = [rows, rows]
        #fix(rows) bc I don't think there's an equiv in python? :
        
        for h in range(0, len(rows)):
            rows[h] = int(rows[h])        
        
        for j in range (rows[0] - 1, rows[1]):            
            novortile = 0 #bytes thing again
            
            if oned != None: #i think?
                flux = np.array(cube.dat)[:, i]
                #err = math.sqrt(abs(cube.var[:, i]))
                err = [0]
                err.pop()
                for a in cube.var[:, i]:
                    err.append(abs(a))
                dq = cube.dq[:, i]
                labin = str(i + 1).zfill(4)
                labout = labin
            else:
                if verbose != None:
                    print(' Row ' + str(j + 1) + ' of ' + str(cube['nrows']))
                
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
                    #switching i and j use but uhh
                    flux = np.array(cube['dat'])[juse, iuse, :].flatten()
                    err = np.array(math.sqrt(abs(cube['var'][juse, iuse, :]))).flatten()
                    dq = np.array(cube['dq'])[juse, iuse, :].flatten()
                    labin = str(juse + 1) + '_' + str(iuse + 1) #swapped here too
                    labout = str(j + 1) + '_' + str(i + 1)
            
            #Line 344
            if novortile == 1: #??
                infile = str(initdat['outdir']) + str(initdat['label']) \
                    + '_' + labin + '.xdr'
                outfile = initdat['outdir'] + initdat['label'] + '_' + labout
                nodata = np.where(flux != 0.0) #maybe not 0.0?
                ct = len(nodata)
                #check if infile exists here but for now I'm just leaving it
                filepresent = True            
            else: 
                filepresent = False
                ct = 0
            
            nofit = False #or was it true?
            
            if filepresent == False or ct < 0:
                nofit = True
                badmessage = 'No data for ' + str(i + 1) + ', ' + \
                    str(j + 1) + '.'
                print(badmessage)
            #else:
                #is there nothing in here?
            
            struct = np.load("struct.npy", allow_pickle='TRUE').item()
            struct['noemlinfit'] = err[struct['fitran_indx']] #necessary?
            
            if not 'noemlinfit' in struct:
                #get line fit params
                tflux = True #or false?
                linepars = sepfitpars(linelist, struct['param,struct']['perror'], \
                                      struct['parinfo'], tflux = tflux, \
                                      doublets = emldoublets) #need to write sepfitpars
                lineweqs = cmpweq(struct, linelist, doublets = emldoublets)
#plot emission line data, print data to a file
            if noplots == None:
                #plot emission lines
                if not 'noemlinfit' in struct:
                    if not 'nolines' in linepars:                        
                        if 'fcnpltlin' in initdat:
                            fcnpltlin = initdat['fcnpltlin']
                        else: fcnpltlin = 'ifsf_pltlin'
                        #if 'argspltlin1' in initdat:
                        
                        #if 'argspltlin2' in initdat:
                            
                #line 386
            
            if not 'noemlinfit' in struct:
                thisncomp = 0
                thisncompline = ''
                
                for line in lines_with_doublets:
                    sigtmp = linepars['sigma'][:, line]
                    fluxtmp = linepars['flux'][:, line]
                    igd = [None]
                    igd.pop(0)
                    for i in range (0, len(flux) * len(flux[0])): #assuming they're the same size
                        if sigtmp[i] != False and sigtmp[i] != bad and \
                            fluxtmp != False and fluxtmp != bad:
                            igd.append(i)
                    ctgd = len(igd) #igd is a 1d array?
                    
                    if ctgd > thisncomp:
                        thisncomp = ctgd
                        thiscompline = line
                    
                    if ctgd > 0:
                        #what do you do here?
                        line = 419
                
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
                    elif initdat['sorttype'] == wave: #'wave'?? line 444
                        isort = linepars['wave'][igd, line].sort() #reversed this
                    elif initdat['sorttype'] == reversewave:
                        isort = linepars['wave'][igd, line].sort(reverse = True)
                    
                    if 'flipsort' in initdat:
                        if flipsort[j, i] != None: #????
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
                    #Need to write printlinpar, line 474
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
                add_poly_degree = 4.0 #shoudl match fitspec
                if 'argscontfit' in initdat:
                    if 'add_poly_degree' in initdat['argscontfit']:
                        add_poly_degree = initdat['argscontfit']['add_poly_degree']
                #Compute polynomial
                #log_rebin?
                xnorm = cap_range(-1.0, 1.0, len(wave_log)) #wave_log from log_rebin
                cont_fit_poly_log = 0.0
                for k in range (0, add_poly_degree):
                    cont_fit_poly_log += legendre(xnorm, k) * struct['ct_add_poly_weight'][k]
                    #legendre? xnorm? k?
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
                contcub['poly_mod_tot'][j, i] = np.sum(cont_fit_poly)
                contcube['poly_mod_tot_pct'][j, j] \
                    = np.divide(contcube['poly_mod_tot'][j, i], cont_fit_tot)
                contcube['stel_sigma'][j, i] = struct['ct_ppxf_sigma']
                contcube['stel_z'][j, i] = struct['zstar']
                
                if 'ct_errors' in struct:
                    contcube['stel_sigma_err'][j, i, :] \
                        = struct['ct_errors']['ct_ppxf_sigma']
                    #assuming that ct_errors is a dictionary
                else: #makes an array with those two arrays in it?
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

                    # Get and renormalize template (check to see what name file is saved under)
                    qsotemplate = np.load("qsotemplate.npy", allow_pickle='TRUE').item()                    
                    qsowave = qsotemplate['wave']
                    qsoflux_full = qsotemplate['flux']
                    qsoflux = np.where(qsowave > struct_tmp['fitran'][0]*0.99999 and \
                                   qsowave < struct_tmp['fitran'][1]*1.00001)
                    #I think. line 611              
                    qsoflux /= np.median(qsoflux)
                    struct = struct_tmp
                    #If polynomial residual is re-fit with PPXF, separate out best-fit
                    #parameter structure created in IFSF_FITQSOHOST and compute polynomial
                    #and stellar components
                    if 'refit' in initdat['argscontfit']:
                        par_qsohost = struct['ct_coeff']['qso_host']
                        par_stel = struct['ct_coeff']['stel']
                        #log rebin, line 622
                        xnorm = cap_range(-1.0, 1.0, len(wave_log)) #1D?
                        if add_poly_degree > 0:
                            par_poly = struct['ct_coeff']['poly']
                            polymod_log = 0.0 # Additive polynomial
                            for k in range (0, add_poly_degree):
                                polymod_log += legendre(xnorm,k)*par_poly[k]
                            interpfunct = interpolate.interp1d(polymod_log, wave_log, kind='linear')
                            polymod_refit = interpfunct(np.log(struct['wave']))
                        else:
                            polymod_refit = np.zeros(struct['wave'], dtype = float)
                        contcube['stel_sigma'][j, i] = struct['ct_coeff']['ppxf_sigma']
                        contcube['stel_z'][j, i] = struct['zstar']
                        
                        if 'ct_errors' in struct:
                            contcube['stel_sigma_err'][j, i, :] \
                                = struct['ct_errors']['ct_ppxf_sigma'] #i think?
                        else:
                            contcube['stel_sigma_err'][j, i, :] \
                                = [struct['ct_ppxf_sigma_err'], struct['ct_ppxf_sigma_err']]
                        if 'ct_errors' in struct:
                            contcube['stel_z_err'][j, i, :] \
                                = struct['ct_errors']['zstar'] #zstar?
                        else: 
                            contcube['stel_z_err'][j, i, :] \
                                = [struct['zstar_err'], struct['zstar_err']]
                        #again like why aren't those two if statements combined
                    else:
                        par_qsohost = struct['ct_coeff']
                        polymod_refit = 0.0
            
                    #produce fit with template only and with template + host. Also
                    #output QSO multiplicative polynomial
                    qsomod_polynorm = True #??
                    qsohostfcn(struct['wave'], par_qsohost, qsomod, qsoflux = qsoflux, \
                                      qsoonly = True, blrterms = blrterms, \
                                      qsoscl = qsomod_polynorm, qsoord = qsoord, \
                                      hostord = hostord)
                    hostmod = struct['cont_fit_pretweak'] - qsomod
                    
                    #if continuum is tweaked in any region, subide resulting residual 
                    #proportionality @ each wavelength btwn qso and host components
                    qsomod_notweat = qsomod
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
                        qsohostfcn(struct['wave'], par_qsohost, qsomod_blronly, \
                                         qsoflux = qsoflux, blronly = True, \
                                         blrterms = blrterms, qsoord = qsoord, \
                                         hostord = hostord)
            elif initdat['fcncontfit'] == ppxf and 'qsotempfile' in initdat:
                struct_star = struct
                #ask abt name of file loaded here


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
