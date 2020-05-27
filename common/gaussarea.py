# docformat = 'rst'
#
#+
#
# Evaluate the integral of a Gaussian function over all x:
#   f(x) = norm * exp(-ax^2)
#   Area = sqrt(Pi/a)
# If a = 0, area is set to 0.
#
# :Categories:
#    IFSFIT
#
# :Returns:
#    Dictionary with tags area and, optionally, area_err.
#
# :Params:
#    a: in, required, numpy array
#      Coefficient of exponential argument
#
# :Keywords:
#    aerr: in, optional, numpy array
#      Error in a
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
#    ChangeHistory::
#      2014apr10, DSNR, documented, added license and copyright# added treatment
#                       of a = 0 case# moved to IFSFIT
#      2014jul09, DSNR, now uses STRUCT_ADDTAGS instead of ADD_TAGS# the former
#                       is from IDLUTILS, the latter from the SSW
#                       library
#      2017apr04, DSNR, now uses CREATE_STRUCT instead of
#                       STRUCT_ADDTAGS# IDLUTILS no longer necessary
#      2020may27, DW, translated to Python 3
#
# :Copyright:
#    Copyright (C) 2014--2017 David S. N. Rupke
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
#
#-


import numpy as np
import math

def gaussarea(a, aerr = None):
    
    ngauss = a.shape
    sqrtpi = np.sqrt(math.pi)
    
    out = np.zeros(ngauss)
    igda = np.where(a > 0)
    ctgda = np.count_nonzero(a > 0)
    
    if ctgda > 0:
        sqrta = np.sqrt(a[igda])
        out[igda] = sqrtpi/sqrta
    
    outstr={'area':out}
    
    if aerr:
        outerr = np.zeros(ngauss)
        if ctgda > 0:
            outerr[igda] = out[igda]*0.5 / a[igda] * aerr[igda]
        outstr['area_err'] = outerr
        
    return outstr
    
    