import numpy as np
import math
from pyfsf_gaussarea import pyfsf_gaussarea

def pyfsf_gaussflux(norm, sigma, normerr='',sigerr=''):
    
    ngauss = norm.shape
    
    a = np.zeros(ngauss)
    aerr = np.zeros(ngauss)
    igda = np.where(sigma > 0)
    ctgda = np.count_nonzero(sigma > 0)
    
    if ctgda > 0:
        a[igda] = 0.5 / sigma[igda]**2.
        if sigerr != '':
            aerr[igda] = sigerr[igda]/sigma[igda]**3.
    
    gint = pyfsf_gaussarea(a, aerr=aerr)
    flux = norm*gint['area']
    
    if sigerr != '':
        ginterr = gint['area_err']
    else:
        ginterr = np.zeros(ngauss)
    if normerr != '':
        normerr = np.zeros(ngauss)
    
    fluxerr = np.zeros(ngauss)
    
    igdn = np.where((norm > 0) & (gint['area'] > 0))
    ctgdn = np.count_nonzero((norm > 0) & (gint['area'] > 0))
    
    if ctgdn > 0:
        fluxerr[igdn] = flux[igdn]*np.sqrt((normerr[igdn]/norm[igdn])**2. + (ginterr[igdn]/gint['area'][igdn])**2.)
        
    outstr = {'flux': flux, 'flux_err': fluxerr}
    
    return outstr
    