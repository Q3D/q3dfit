#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from scipy import interpolate
import numpy as np


def interptemp(spec_lam, temp_lam, template):
    """
    (Linearly) interpolate templates from template wavelength grid to data
    wavelength grid.

    Parameters
    ----------

    spec_lam : dblarr(nwave_spec)
        Wavelengths of data arrays.
    temp_lam : dblarr(nwave_temp)
        Wavelengths of template arrays.
    template : dblarr(nwave_temp, ntemplates)
        Model fluxes from templates.


    Returns
    -------

    The interpolated templates, of type dblarr(nwave_spec, ntemplates).
    """

    if len(template.shape) == 2:
        ntemp = template.shape(1)
        new_temp = np.zeros((spec_lam.shape[0], ntemp))
    else:
        ntemp = 1

    if np.min(temp_lam) > np.min(spec_lam):
        print('IFSF_INTERPTEMP: WARNING -- Extrapolating template from ' +
              str(min(temp_lam)) + ' to ' + str(min(spec_lam)) + '.')
    if np.max(temp_lam) < np.max(spec_lam):
        print('IFSF_INTERPTEMP: WARNING -- Extrapolating template from ' +
              str(max(temp_lam)) + ' to ' + str(max(spec_lam)) + '.')

    # Default interpolation for INTERPOL is linear
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
