#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jul 26 12:05:26 2022

@author: drupke
"""

from os import chdir
from q3dfit.q3df import q3df
from q3dfit.q3da import q3da

chdir('../jnb/')
q3di = 'nirspec-j1652/q3di.npy'

q3df(q3di, cols=51, rows=43, quiet=False)
q3da(q3di, cols=51, rows=43)
