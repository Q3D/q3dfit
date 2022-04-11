import numpy as np
import math


def gaussarea(a, aerr=None):
    '''
    Evaluate the integral of a Gaussian function over all x:
      f(x) = norm * exp(-ax^2)
      Area = sqrt(Pi/a)
    If a = 0, area is set to 0.

    Parameters
    ----------
    a: ndarray
        Coefficient of exponential argument
    aerr: ndarray, optional
        Error in a

    Returns
    -------
    dict
        Dictionary with tags area and, optionally, area_err.
    '''

    sqrtpi = np.sqrt(math.pi)
    out = 0.
    if a > 0:
        sqrta = np.sqrt(a)
        out = sqrtpi/sqrta
    outstr = {'area': out}
    if aerr:
        outerr = 0.
        if a > 0:
            outerr = out*0.5 / a * aerr
        outstr['area_err'] = outerr

    return outstr
