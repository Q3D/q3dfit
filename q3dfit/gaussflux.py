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
#

import numpy as np
from math import pi


def gaussflux(norm, sigma, normerr=None, sigerr=None):

    flux = norm * sigma * np.sqrt(2. * pi)
    fluxerr = 0.

    if normerr is not None and sigerr is not None:
        fluxerr = flux*np.sqrt((normerr/norm)**2. + (sigerr/sigma)**2.)

    outstr = {'flux': flux, 'flux_err': fluxerr}

    return outstr
