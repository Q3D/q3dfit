#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Dec 16 13:54:17 2020

@author: drupke
"""

# docformat = 'rst'
#
#+
#
# (Linearly) interpolate templates from template wavelength grid to data
# wavelength grid.
#
# :Categories:
#    IFSFIT
#
# :Returns:
#    The interpolated templates, of type dblarr(nwave_spec, ntemplates).
#
# :Params:
#    spec_lam: in, required, type=dblarr(nwave_spec)
#      Wavelengths of data arrays.
#    temp_lam: in, required, type=dblarr(nwave_temp)
#      Wavelengths of template arrays.
#    template: in, required, type=dblarr(nwave_temp\, ntemplates)
#      Model fluxes from templates.
#
# :Author:
#    David S. N. Rupke::
#      Rhodes College
#      Department of Physics
#      2000 N. Parkway
#      Memphis, TN 38104
#      drupke@gmail.com
#
# :History:
#    Change History::
#      2009, DSNR, copied base code from Harus Jabran Zahid
#      2013oct17, DSNR, documented
#      2013nov13, DSNR, renamed, added license and copyright
#      2013dec11, DSNR, added a comment
#
# :Copyright:
#    Copyright (C) 2013 David S. N. Rupke
#
#    This program is free software: you can redistribute it and/or
#    modify it under the terms of the GNU General Public License as
#    published by the Free Software Foundation, either version 3 of
#    the License or any later version.
#
#    This program is distributed in the hope that it will be useful,
#    but WITHOUT ANY WARRANTY# without even the implied warranty of
#    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU
#    General Public License for more details.
#
#    You should have received a copy of the GNU General Public License
#    along with this program.  If not, see
#    http://www.gnu.org/licenses/.

from scipy import interpolate


def interptemp(spec_lam, temp_lam, template):

    if template.ndim() == 2:
        ntemp = template.shape(1)
    else:
        ntemp = 1
    new_temp = np.array([spec_lam.shape(), ntemp])

    if min(temp_lam) > min(spec_lam):
        print('IFSF_INTERPTEMP: WARNING -- Extrapolating template from ' +
              min(temp_lam) + ' to ' + min(spec_lam) + '.')
    if max(temp_lam) < max(spec_lam):
        print('IFSF_INTERPTEMP: WARNING -- Extrapolating template from ' +
              max(temp_lam) + ' to ' + max(spec_lam) + '.')

    # Default interpolation for INTERPOL is linear
    for i in range(ntemp - 1):
        interpfunc = \
            interpolate.interp1d(temp_lam, template[*, i], kind='cubic')
        new_temp[*, i] = interpfunc(spec_lam)

    return, new_temp
