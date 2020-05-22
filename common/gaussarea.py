import numpy as np
import math

def pyfsf_gaussarea(a, aerr = ''):
    
    ngauss = a.shape
    sqrtpi = np.sqrt(math.pi)
    
    out = np.zeros(ngauss)
    igda = np.where(a > 0)
    ctgda = np.count_nonzero(a > 0)
    
    if ctgda > 0:
        sqrta = np.sqrt(a[igda])
        out[igda] = sqrtpi/sqrta
    
    outstr={'area':out}
    
    if aerr != '':
        outerr = np.zeros(ngauss)
        if ctgda > 0:
            outerr[igda] = out[igda]*0.5 / a[igda] * aerr[igda]
        outstr['area_err'] = outerr
        
    return outstr
    
    