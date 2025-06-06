#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import annotations

from astropy.constants import c
import numpy as np
from scipy import interpolate
from numpy.typing import ArrayLike

#from . import linelist


def interptemp(spec_lam: ArrayLike,
               temp_lam: ArrayLike,
               template: ArrayLike) -> ArrayLike:
    """
    (Linearly) interpolate templates from template wavelength grid to data
    wavelength grid.

    Parameters
    ----------
    spec_lam
        Wavelengths of data arrays.
    temp_lam
        Wavelengths of template arrays.
    template
        Model fluxes from templates. Second dimension is the number of
        templates.

    Returns
    -------
    ArrayLike
        The interpolated templates.
    """

    if len(template.shape) == 2:
        ntemp = template.shape[1]
        new_temp = np.zeros((spec_lam.shape[0], ntemp))
    else:
        ntemp = 1

    if np.min(temp_lam) > np.min(spec_lam):
        print('IFSF_INTERPTEMP: WARNING -- Extrapolating template from ' +
              str(min(temp_lam)) + ' to ' + str(min(spec_lam)) + '.')
    if np.max(temp_lam) < np.max(spec_lam):
        print('IFSF_INTERPTEMP: WARNING -- Extrapolating template from ' +
              str(max(temp_lam)) + ' to ' + str(max(spec_lam)) + '.')

    if ntemp != 1:
        for i in range(ntemp - 1):
            interpfunc = \
                interpolate.interp1d(temp_lam, template[:, i], kind='linear')
            new_temp[:, i] = interpfunc(spec_lam)
    else:
        interpfunc = \
            interpolate.interp1d(temp_lam, template, kind='linear',
                                 bounds_error=False, fill_value=0)
        new_temp = interpfunc(spec_lam)
    return new_temp


def interp_lis(spec_lam: ArrayLike,
               temp_lam_lis: list[ArrayLike] | ArrayLike,
               template_lis: list[ArrayLike] | ArrayLike) -> np.ndarray:
    """
    This function samples a single template or list of templates onto
    the wavelength array of the input observed spectrum.  The output is
    padded with NaN values instead of extrapolating beyond any template's
    wavelength range.

    Parameters
    ----------
    spec_lam
        reference wavelength array onto which the templates(s) are interpolated
    temp_lam_lis
        Template wavelength array or list containing the wavelength arrays of the
        different templates (not required to be at same length). (Also accepts 1D list,
        list containing a 1D array, or list of lists)
    template_lis
        Template flux array or list containing the flux arrays of the different templates
        (not required to be at same length). (Also accepts 1D list, list containing a 1D
        array, or list of lists)


    Returns
    -------
    numpy.ndarray
        Array containing the flux of the templates interpolated onto the spec_lam input
        array.
        If 2D, the 2nd dimension corresponds to the flux arrays of individual templates,
        i.e. the newly sampled flux of template 0 can be accessed as new_temp[:,0], while
        for template 1 this is new_temp[:,1], etc. The new_temp array is padded with NaN
        values where the wavelength range of the respective template was more narrow than
        the observed spectum.

    """
     # -- Keyword to fill output array with NaNs beyond the interpolation range.
    bounds_error=False

    # (not triggered for a single 1D list)
    if isinstance(template_lis, list) and hasattr(template_lis[0], "__len__"):
        ntemp = len(template_lis)
        new_temp = np.zeros((spec_lam.shape[0],ntemp))
        if ntemp == 1: # If a list containing just a 1D array is entered
            template_lis = template_lis[0]; temp_lam_lis=temp_lam_lis[0]
    else:
        ntemp = 1

    if ntemp >1:
        for i in range(ntemp):
            interpfunc = \
                interpolate.interp1d(temp_lam_lis[i], template_lis[i], kind='cubic', 
                                     bounds_error=bounds_error, fill_value=0)
            new_temp[:, i] = interpfunc(spec_lam)
    else:
        interpfunc = \
            interpolate.interp1d(temp_lam_lis, template_lis, kind='linear', 
                                 bounds_error=bounds_error, fill_value=float(0))
        new_temp = interpfunc(spec_lam)
    return new_temp


def cutoff_NaNs(spec_lam: ArrayLike,
                new_temp: np.ndarray) -> tuple[ArrayLike, np.ndarray]:
    """
    This function removes any NaN values from the output of the interp_lis() function.

    Parameters
    ----------
    spec_lam
        reference wavelength array onto which the templates(s) are interpolated
    new_temp
        1D or 2D array with newly interpolated template fluxes, corresponding to the
        output of interp_lis()

    Returns
    -------
    new_temp_NoNan
        Same as new_temp, but without any NaN elements / rows containing at least one NaN
        value

    """
    OKrow = np.array([])
    for i in range(new_temp.shape[0]):
        if np.sum(np.isnan(new_temp[i]))==0:
            OKrow = np.append(OKrow, i)
    spec_lam_NoNaN = spec_lam[OKrow.astype(int)]
    if len(new_temp.shape)>1:
        new_temp_NoNan = new_temp[OKrow.astype(int), :]
    else:
        new_temp_NoNan = new_temp[OKrow.astype(int)]
    return spec_lam_NoNaN, new_temp_NoNan


# def example_interp():

#     source1 = np.load('../test/test_questfit/IRAS21219m1757_dlw_qst.npy', allow_pickle=True)
#     if source1.shape[0]==1:	source1=source1[0]
#     templ1 = np.load('../data/questfit_templates/smith_nftemp3.npy', allow_pickle=True)
#     templ2 = np.load('../data/questfit_templates/smith_nftemp4.npy', allow_pickle=True)

#     tpl_wave_lis = [templ1['WAVE'], templ2['WAVE']]
#     tpl_flux_lis = [templ1['FLUX'], templ2['FLUX']]
#     temp_flux = interp_lis(source1['WAVE'], tpl_wave_lis, tpl_flux_lis)

#     wave_noNaN, temp_flux_NoNaN = cutoff_NaNs(source1['WAVE'], temp_flux)
#     return wave_noNaN, temp_flux_NoNaN


'''
def cmpweq(instr, linelist, doublets=None):
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
'''

