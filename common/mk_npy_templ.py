# -*- coding: utf-8 -*-
"""
This script reads in all IDEOS/QUESTFIT templates in .xdr format in a given directory, 
and stores them in a more easily python-readable .npy format. 
The header is stored as text in a separate .meta file.
Any of the resulting .npy files, say draine03.npy f.ex., can then be read in python as follows:
  import numpy as np;  wave = np.load('draine03.npy')['WAVE']
or analogously for the the 'FLUX', 'STDEV' and 'FLAG' columns.


 :Usage:
      python Convert_templ_to_npy.py (--path_tpl='../data/questfit_templates/' --path_out='../data/questfit_templates/')

 :Params:
     path_tpl: in, optional, type=str
        Path pointing to the input .xdr templates
     path_out: in, optional, type=str
        Path where the output .npy templates should be stored

"""

import numpy as np
from scipy.io.idl import readsav
import glob
import os
import argparse


def xdr_templ_to_npy(file_xdr='draine03.xdr', path_out='../templates_npy/'):
	'''
	This function takes an xdr template file from .../IDEOS/templates/ and converts it 
	to a more easily python-readable .npy file. The header is stored as text in a separate .meta file.

	In: xdr template file, f.ex. draine03.xdr
	'''
	dict1 = readsav(file_xdr, python_dict=True, verbose=False)
	data = dict1['memstor'][0][1]['DATA'][0]
	header = dict1['memstor'][0][1]['HEADER'][0].decode("utf-8")
	templ_name = file_xdr.split('/')[-1].replace('.xdr', '')
	np.save(path_out+templ_name+'.npy', data)
	np.savetxt(path_out+templ_name+'.meta', np.array([header]), fmt='%s')


''' ----------------------------------------------------------
	Read in / define paths
 ---------------------------------------------------------- '''

parser = argparse.ArgumentParser()
parser.add_argument('--path_tpl', default='../data/questfit_templates/', help='Path pointing to the input .xdr templates')
parser.add_argument('--path_out', default='../data/questfit_templates/', help='Path where the output .npy templates should be stored')

path_tpl = vars(parser.parse_args())['path_tpl']
path_out = vars(parser.parse_args())['path_out']
path_si = path_tpl+'silicatemodels/'

if not os.path.isdir(path_out):	os.mkdir(path_out)
if not os.path.isdir(path_out+'silicatemodels/'):	os.mkdir(path_out+'silicatemodels/')


''' ----------------------------------------------------------
	Convert all templates from .xdr to .npy format
 ---------------------------------------------------------- '''

for file_i in glob.glob(path_tpl+'*.xdr'):
	xdr_templ_to_npy(file_i, path_out=path_out)

for file_i in glob.glob(path_si+'*.xdr'):
	xdr_templ_to_npy(file_i, path_out=path_out+'silicatemodels/')






