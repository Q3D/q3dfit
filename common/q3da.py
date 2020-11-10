# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""
import numpy as np
import math
import pdb
import importlib
from q3dfit.common.linelist import linelist
from q3dfit.common.readcube import CUBE

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
                        header=header, datext = datext, varext = varext, \
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
            xyvor = np.where(vormap == i) #i think so?
            vorcoords[i - 1, :] = xyvor

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
         cstr = 'c' + str(k + 1) #come back to this
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
                #replacement for where
                nodata = [None]
                nodata.pop(0)
                for i in range(0, len(flux)):
                    if flux[i] != 0:
                        nodata.append(i)
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
            else:
                #is there nothing in here?
                huh = what
            
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
                        if flipsort[j, i]: #where did flipsort come from?
                            print('Flipsort set for spaxel [' + str(i + 1) \
                                  + ',' + str(j + 1) + '] but ' + \
                            'only 1 component. Setting to 2 components and ' \
                            + 'flipping anyway.')
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
                        isort = linepars['wave'][igd, line].sort(reverse = true)
                    
                    if 'flipsort' in initdat:
                        if flipsort[j, i] != None: #????
                            isort = isort.sort(reverse = true)
                if thisncomp > 0:
                    for line in lines_with_doublets:
                        kcomp = 1
                        for sindex in isort:
                            cstr='c'+string(kcomp,format='(I0)')
                            emlwav[cstr,line,i,j]=linepars.wave[line,sindex]
                            emlwaverr[cstr,line,i,j]=linepars.waveerr[line,sindex]
                            emlsig[cstr,line,i,j]=linepars.sigma[line,sindex]
                            emlsigerr[cstr,line,i,j]=linepars.sigmaerr[line,sindex]
                            emlweq['f'+cstr,line,i,j]=lineweqs.comp[line,sindex]
                            emlflx['f'+cstr,line,i,j]=linepars.flux[line,sindex]
                            emlflxerr['f'+cstr,line,i,j]=linepars.fluxerr[line,sindex]
                            emlflx['f'+cstr+'pk',line,i,j]=linepars.fluxpk[line,sindex]
                            emlflxerr['f'+cstr+'pk',line,i,j]=linepars.fluxpkerr[line,sindex]
                            kcomp+=1 
                    

#;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;
#; Process NaD (normalize, compute quantities and save, plot)
#;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;

        if tag_exist(initdat,'donad') then begin
            if tag_exist(initdat,'decompose_qso_fit'):
                if tag_exist(initdat,'remove_scattered'):
                    hostmod_tmp = hostmod - polymod_refit
                    qsomod_tmp = qsomod + polymod_refit
                else:
                    hostmod_tmp = hostmod
                    qsomod_tmp = qsomod
                nadnormcont = (struct.cont_dat - qsomod_tmp)/hostmod_tmp
                nadnormconterr = struct.spec_err/hostmod_tmp
                nadnormstel = hostmod_tmp
            else:
                nadnormcont = struct.cont_dat/struct.cont_fit
                nadnormconterr = struct.spec_err/struct.cont_fit
                nadnormstel = struct.cont_fit
            if tag_exist(initnad,'argsnormnad'):
                normnad = ifsf_normnad(struct.wave,\
                                     nadnormcont,\
                                     nadnormconterr,\
                                     struct.zstar,fitpars_normnad,\
                                     _extra=initnad.argsnormnad)
                normnadem = ifsf_normnad(struct.wave,\
                                       struct.emlin_dat,\
                                       struct.spec_err,\
                                       struct.zstar,fitpars_normnadem,\
                                       /nosncut,/subtract,\
                                       _extra=initnad.argsnormnad)
                normnadstel = ifsf_normnad(struct.wave,\
                                         nadnormstel,\
                                         struct.spec_err,\
                                         struct.zstar,fitpars_normnadstel,\
                                         _extra=initnad.argsnormnad)
                if ~ tag_exist(initnad.argsnormnad,'fitranlo'): 
                    fitranlo = (1d +struct.zstar)*[5810d,5865d] \
                else:
                    fitranlo = initnad.argsnormnad.fitranlo
                if ~ tag_exist(initnad.argsnormnad,'fitranhi'): 
                    fitranhi = (1d +struct.zstar)*[5905d,5960d] 
                else: 
                    fitranhi = initnad.argsnormnad.fitranhi
            else:
                normnad = ifsf_normnad(struct.wave,\
                                     nadnormcont,\
                                     nadnormconterr,\
                                     struct.zstar,fitpars_normnad)
                normnadem = ifsf_normnad(struct.wave,\
                                       struct.emlin_dat,\
                                       struct.spec_err,\
                                       struct.zstar,fitpars_normnadem,\
                                       /nosncut,/subtract)
                normnadstel = ifsf_normnad(struct.wave,\
                                         nadnormstel,\
                                         struct.spec_err,\
                                         struct.zstar,fitpars_normnadstel)
                fitranlo = (1d +struct.zstar)*[5810d,5865d]
                fitranhi = (1d +struct.zstar)*[5905d,5960d]
            #Check data quality
            if normnad != None:
                igd = np.where(normnad.nflux gt 0d,ctgd) \
            else: 
                ctgd = 0
            #Compute empirical equivalent widths and emission-line fluxes
            if ctgd gt 0:
            #Create output data cube
                if firstnadnorm:
                    ilo = value_locate(cube.wave,fitranlo[0])+1
                    ihi = value_locate(cube.wave,fitranhi[1])
                    dat_normnadwave = cube.wave[ilo:ihi]
                    nz = ihi-ilo+1
                    nadcube = \
                       {'wave': np.zeros(cube.ncols,cube.nrows,nz),\
                        'cont': np.zeros(cube.ncols,cube.nrows,nz),\
                        'dat:' np.zeros(cube.ncols,cube.nrows,nz),\
                        'err': np.zeros(cube.ncols,cube.nrows,nz)+bad,\
                        'weq': np.zeros(cube.ncols,cube.nrows,4)+bad,\
                        'stelweq': np.zeros(cube.ncols,cube.nrows,2)+bad,\
                        'iweq': np.zeros(cube.ncols,cube.nrows,4)+bad,\
                        'emflux': np.zeros(cube.ncols,cube.nrows,2)+bad,\
                        'emul': np.zeros(cube.ncols,cube.nrows,4)+bad,\
                        'vel': np.zeros(cube.ncols,cube.nrows,6)+bad}
                    firstnadnorm = 0
            #Defaults
            emflux=np.zeros(2)
            emul=np.zeros(4)+bad
            vel = np.zeros(6)+bad
            if tag_exist(initnad,'argsnadweq'):
            weq = ifsf_cmpnadweq(normnad.wave,normnad.nflux,normnad.nerr,\
                                       snflux=normnadem.nflux,unerr=normnadem.nerr,\
                                       emflux=emflux,emul=emul,\
                                       _extra=initnad.argsnadweq)
        
                  #These need to be compatible with the IFSF_CMPNADWEQ defaults
            if tag_exist(initnad.argsnadweq,'emwid'):
                emwid=initnad.argsnadweq.emwid 
            else:
                emwid=20d
            if tag_exist(initnad.argsnadweq,'iabsoff'): 
                iabsoff=initnad.argsnadweq.iabsoff 
            else: 
                iabsoff=4
              endif else begin
                 weq = ifsf_cmpnadweq(normnad.wave,normnad.nflux,normnad.nerr,\
                                      snflux=normnadem.nflux,unerr=normnadem.nerr,\
                                      emflux=emflux,emul=emul)
                 These need to be compatible with the IFSF_CMPNADWEQ defaults
                 emwid=20d
                 iabsoff=4l
              endelse
              Compute stellar continuum NaD equivalent widths from fit
              stelweq = ifsf_cmpnadweq(normnadstel.wave,normnadstel.nflux,\
                                       normnadstel.nerr,\
                                       wavelim=[5883d*(1d +initdat.zsys_gas),\
                                                6003d*(1d +initdat.zsys_gas),\
                                                0d,0d])
        
              Compute empirical velocities
              size_weq = size(weq)
              if size_weq[0] eq 2 then begin
                 if tag_exist(initnad,'argsnadvel') then \
                    vel = ifsf_cmpnadvel(normnad.wave,normnad.nflux,normnad.nerr,\
                                         weq[*,1],initdat.zsys_gas,\
                                         _extra=initnad.argsnadvel) \
                 else vel = ifsf_cmpnadvel(normnad.wave,normnad.nflux,normnad.nerr,\
                                           weq[*,1],initdat.zsys_gas)
              endif
              
              ilo = where(normnad.wave[0] eq dat_normnadwave)
              ihi = where(normnad.wave[n_elements(normnad.wave)-1] \
                    eq dat_normnadwave)
        
              Assume that stellar fit is a good model but that the error spectrum
              may not be perfect. Correct using stellar reduced chi squared
              if tag_exist(initnad,'errcorr_ctrchisq') then begin
                 normnad.nerr *= struct.ct_rchisq
                 weq[1,0] *= struct.ct_rchisq
                 weq[3,0] *= struct.ct_rchisq
              endif
              nadcube.wave[i,j,*] = dat_normnadwave
              nadcube.cont[i,j,ilo:ihi] = struct.cont_fit[normnad.ind]
              nadcube.dat[i,j,ilo:ihi] = normnad.nflux
              nadcube.err[i,j,ilo:ihi] = normnad.nerr
              nadcube.weq[i,j,*] = weq[*,0]
              nadcube.iweq[i,j,*] = weq[*,1]
              nadcube.stelweq[i,j,*] = stelweq[0:1]
              nadcube.emflux[i,j,*] = emflux
              nadcube.emul[i,j,*] = emul
              nadcube.vel[i,j,*] = vel
              Plot data
              if not keyword_set(noplots) then \
                 if tag_exist(initnad,'argspltnormnad') then \
                    ifsf_pltnaddat,normnad,fitpars_normnad,struct.zstar,\
                                   outfile+'_nad_norm',autoindices=weq[*,1],\
                                   emwid=emwid,iabsoff=iabsoff,\
                                   _extra=initnad.argspltnormnad else \
                    ifsf_pltnaddat,normnad,fitpars_normnad,struct.zstar,\
                                   outfile+'_nad_norm',autoindices=weq[*,1],\
                                   emwid=emwid,iabsoff=iabsoff
           endif
        
        endif
        
        endelse
        
        for
        
