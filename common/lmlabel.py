#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Aug 25 14:07:30 2021

@author: drupke

Remove characters from a label string that are incompatible with LMFIT's
parser; or reverse the operation.

"All keys of a Parameters() instance must be strings and valid Python symbol
names, so that the name must match [a-z_][a-z0-9_]* and cannot be a Python
reserved word."

https://lmfit.github.io/lmfit-py/parameters.html#lmfit.parameter.Parameters

"""


class lmlabel():

    def __init__(self, label, reverse=False):
        if reverse:
            lmlabel = label
            origlabel = label.replace('lb', '[').replace('rb', ']').\
                replace('pt', '.')
        else:
            origlabel = label
            lmlabel = label.replace('[', 'lb').replace(']', 'rb').\
                replace('.', 'pt')
        self.label = origlabel
        self.lmlabel = lmlabel


# class lmpar():
#
#    def __init__(self, lmlabel, comp, partype):
#        self.parname = f'{lmlabel}_c{comp}_g{partype}'
