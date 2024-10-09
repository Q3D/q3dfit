# -*- coding: utf-8 -*-
"""
Created on Thu Jul  9 14:29:06 2020

@author: lily
"""
# Translated into python by Lily Whitesell, July 14, 2020
# Rewritten using astropy.io.ascii, DSNR, 8 Dec 2020

import numpy as np
from astropy.io import ascii


def gdtemp(infile, outfile):

    data = ascii.read(infile)
    ages_str = data.meta['comments'][43].split()
    ages_strnp = np.array(ages_str[2:])
    ages = ages_strnp.astype(float)
    Lambda = np.array(data['col1'])
    flux = np.zeros((len(data), ages.size), dtype=float)
    cols = data.colnames[1::3]
    for i in range(len(ages)):
        flux[:, i] = np.array(data[cols[i]])
        flux[:, i] /= flux[:, i].max()

    template = {'lambda': Lambda, 'flux': flux, 'ages': ages}
    # this isn't formally allowed in np.save (template should be an array).
    # other options are pickle.dump or json.dump
    np.save(outfile, template)
