# docformat = 'rst'
#
#+
#
# Evaluate the integral of a Gaussian function over all x:
#   f(x) = norm * exp(-x/2sigma^2)
# Calls IFSF_GAUSSAREA to do the evaluation.
#
# :Categories:
#    IFSFIT
#
# :Returns:
#    Structure with tags flux and flux_err.
#
# :Params:
#    norm: in, required, numpy array
#      Coefficient of exponential
#    sigma: in, required, numpy array
#      Standard deviation
#    
# :Keywords:
#    normerr: in, optional, numpy array
#      Error in norm.
#    sigerr: in, optional, numpy array
#      Error in sigma
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
#                       of sigma = 0, norm = 0 cases# moved to IFSFIT
#      2020may27, DW, translated to Python 3
#
# :Copyright:
#    Copyright (C) 2014 David S. N. Rupke
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
from q3dfit.common.gaussarea import gaussarea

def gaussflux(norm, sigma, normerr=None,sigerr=None):
    
    ngauss = norm.shape
    
    a = np.zeros(ngauss)
    aerr = np.zeros(ngauss)
    igda = np.where(sigma > 0)
    ctgda = np.count_nonzero(sigma > 0)
    
    if ctgda > 0:
        a[igda] = 0.5 / sigma[igda]**2.
        if sigerr:
            aerr[igda] = sigerr[igda]/sigma[igda]**3.
    
    gint = gaussarea(a, aerr=aerr)
    flux = norm*gint['area']
    
    if sigerr:
        ginterr = gint['area_err']
    else:
        ginterr = np.zeros(ngauss)
    if normerr == None:
        normerr = np.zeros(ngauss)
    
    fluxerr = np.zeros(ngauss)
    
    igdn = np.where((norm > 0) & (gint['area'] > 0))
    ctgdn = np.count_nonzero((norm > 0) & (gint['area'] > 0))
    
    if ctgdn > 0:
        fluxerr[igdn] = flux[igdn]*np.sqrt((normerr[igdn]/norm[igdn])**2. + (ginterr[igdn]/gint['area'][igdn])**2.)
        
    outstr = {'flux': flux, 'flux_err': fluxerr}
    
    return outstr


    