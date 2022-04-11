# -*- coding: utf-8 -*-
"""
Created on Fri Jun  4 22:09:04 2021

@author: lily
"""
import numpy as np
from astropy.modeling import models, fitting

def fitpoly(lam,flux,weight,template_lambdaz, template_flux, index, zstar,
            fitord=3, quiet=False, refit=False):
   

    ilam=lam[index]
    iflux=flux[index]
    #the fitter I used puts weights in 1/sigma so I took the square root to make the data correct
    w=weight[index]
    iweight=np.sqrt(w)


    ilam = ilam.reshape(ilam.size)
    iflux = iflux.reshape(ilam.size)
    iweight = iweight.reshape(ilam.size)


    if fitord==0:
        deg1=len(ilam)-1
        deg2=fitord
    else:
        deg1=fitord
        deg2=fitord
# parinfo is start params, it's unnecessary unless wanted

  #  parinfo = np.full(fitord+1, {'value': 0.0})
    #array where every every element is the dictionary: {'value': 0.0}


    #making astropy fitter
    fitter = fitting.LevMarLSQFitter()
    #making polynomial model
    polymod1= models.Polynomial1D(deg1)
    polymod2= models.Polynomial1D(deg2)


    #creating fluxfit
    fluxfit = fitter(polymod1, ilam, iflux, weights=iweight)
    fluxfitparam=fluxfit.parameters
#this currently will give a broadcast issue in astropy (I have reached out about the issue). The way to fix this is in data.py in line 1231 and 1232.
#A parenthesis needs to be added in line 1231 to be (np.ravel(weights)*...
#and add  .T).T] to line 1232

    #flip for numpy.poly1d
    ct_coeff=np.flip(fluxfitparam)

    ct_poly = np.poly1d(ct_coeff, variable='lambda')
    continuum=ct_poly(lam)

   # np.save('ct_coeff.npy', ct_coeff)

    icontinuum = ct_poly(index)

    if refit==True:
        for i in range (0, np.size(refit['ord']) - 1):
            tmp_ind=np.where(lam >= refit['ran'][0,i] and
                             lam <= refit['ran'][1,i])
            tmp_iind=np.where(ilam >= refit['ran'][0,i] and
                              ilam <= refit['ran'][1,i])
            #  parinfo=np.full(refit['ord'][i]+1, {'value':0.0})

            #degree of polynomial fit defaults to len(x-variable)-1
            if deg2==0:
                deg2=len(ilam[tmp_iind])-1

            #creating tmp_pars
            tmp_pars=fitter(polymod2, ilam[tmp_iind],
                            (iflux[tmp_iind]-icontinuum[tmp_iind]),
                            z=None, weights=iweight[tmp_iind])
            tmp_parsptmp=tmp_pars.parameters
            tmp_parsparam=np.flip(tmp_parsptmp)

            #lam[tmp_ind] doesn't make sense as a variable???
            ct_poly[tmp_ind] += np.poly1d(tmp_parsparam, variable='lambda')

    return continuum, ct_coeff, zstar
