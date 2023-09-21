#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import numpy as np
from scipy import interpolate


def interp_lis(spec_lam, temp_lam_lis, template_lis):
    """
    This function samples a single template or list of templates onto
    the wavelength array of the input observed spectrum.  The output is
    padded with NaN values instead of extrapolating beyond any template's
    wavelength range.

    Parameters
    ----------
    spec_lam : 1D numpy array/list
        reference wavelength array onto which the templates(s) are interpolated
    temp_lam_lis : 1D numpy array or list of arrays
        Template wavelength array or list containing the wavelength arrays of the
        different templates (not required to be at same length). (Also accepts 1D list,
        list containing a 1D array, or list of lists)
    template_lis : 1D numpy array or list of arrays
        Template flux array or list containing the flux arrays of the different templates
        (not required to be at same length). (Also accepts 1D list, list containing a 1D
        array, or list of lists)


    Returns
    -------
    new_temp : 1D or 2D numpy array
        Array containing the flux of the templates interpolated onto the spec_lam input
        array.
        If 2D, the 2nd dimension corresponds to the flux arrays of individual templates,
        i.e. the newly sampled flux of template 0 can be accessed as new_temp[:,0], while
        for template 1 this is new_temp[:,1], etc. The new_temp array is padded with NaN
        values where the wavelength range of the respective template was more narrow than
        the observed spectum.

    """


    bounds_error=False # -- Keyword to fill output array with NaNs beyond the interpolation range.


#    fill_value = 0

    if isinstance(template_lis, list) and hasattr(template_lis[0], "__len__"):   # (not triggered for a single 1D list)
        ntemp = len(template_lis)
        new_temp = np.zeros((spec_lam.shape[0],ntemp))
        if ntemp==1: # If a list containing just a 1D array is entered
            template_lis = template_lis[0]; temp_lam_lis=temp_lam_lis[0]
    else:
        ntemp = 1

    if ntemp >1:
        for i in range(ntemp):
            interpfunc = \
                interpolate.interp1d(temp_lam_lis[i], template_lis[i], kind='cubic', bounds_error=bounds_error, fill_value=0)
            new_temp[:, i] = interpfunc(spec_lam)
    else:
        interpfunc = \
            interpolate.interp1d(temp_lam_lis, template_lis, kind='linear', bounds_error=bounds_error, fill_value=float(0))
        new_temp = interpfunc(spec_lam)
    return new_temp


def cutoff_NaNs(spec_lam, new_temp):
    """
    This function removes any NaN values from the output of the interp_lis() function.

    Parameters
    ----------
    spec_lam: 1D numpy array
        reference wavelength array onto which the templates(s) are interpolated
    new_temp: 1D or 2D numpy array or list of arrays
        1D or 2D array with newly interpolated template fluxes, corresponding to the
        output of interp_lis()

    Returns
    -------
    new_temp_NoNan: 1D numpy array or list of arrays
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


def example_interp():

    source1 = np.load('../test/test_questfit/IRAS21219m1757_dlw_qst.npy', allow_pickle=True)
    if source1.shape[0]==1:	source1=source1[0]
    templ1 = np.load('../data/questfit_templates/smith_nftemp3.npy', allow_pickle=True)
    templ2 = np.load('../data/questfit_templates/smith_nftemp4.npy', allow_pickle=True)

    tpl_wave_lis = [templ1['WAVE'], templ2['WAVE']]
    tpl_flux_lis = [templ1['FLUX'], templ2['FLUX']]
    temp_flux = interp_lis(source1['WAVE'], tpl_wave_lis, tpl_flux_lis)

    wave_noNaN, temp_flux_NoNaN = cutoff_NaNs(source1['WAVE'], temp_flux)
