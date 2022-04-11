# -*- coding: utf-8 -*-
"""
This script reads in all IDEOS/QUESTFIT templates in .xdr format in a given directory, 
and stores them in a more easily python-readable .npy format. 
The header is stored as text in a separate .meta file.
Any of the resulting .npy files, say draine03.npy f.ex., can then be read in python as follows:
  import numpy as np;  wave = np.load('draine03.npy', allow_pickle=True)['WAVE']
or analogously for the the 'FLUX', 'STDEV' and 'FLAG' columns. ( The column names
can be retrieved by printing np.load('draine03.npy', allow_pickle=True)[0].dtype.names ).

Optionally, two test sources 4978688_0.ideos.xdr and IRAS21219m1757_dlw_qst.xdr are also 
converted into .npy format (only the 'WAVE', 'FLUX' and 'EFLUX' entries for 4978688_0.ideos.xdr, 
and only the 'WAVE', 'FLUX', 'STDEV' and 'FLAG' entries for IRAS21219m1757_dlw_qst.xdr).

 :Usage:
      python mk_npy_templ.py (--path_tpl='../data/questfit_templates/' --path_out='../data/questfit_templates/' --test_sources=True)

 Parameters
 ----------
 
 path_tpl : optional, str
     Path pointing to the input .xdr templates
 path_out : optional, str
     Path where the output .npy templates should be stored
 test_sources: optional, bool
     True if test sources 4978688_0.ideos.xdr and IRAS21219m1757_dlw_qst.xdr should also be converted to
     .npy (in addition to the templates)

"""

import numpy as np
from scipy.io.idl import readsav
import glob
import os
import argparse


def xdr_templ_to_npy(file_xdr='draine03.xdr', path_out='../data/questfit_templates/'):
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


def example_sources_to_npy():
    '''
    This function takes two test sources 4978688_0.ideos.xdr and IRAS21219m1757_dlw_qst.xdr 
    from ../test/test_questfit/ and converts them to a more easily python-readable .npy file. 
    Only the 'WAVE', 'FLUX' and 'EFLUX' entries are translated for 4978688_0.ideos.xdr, 
    and only the 'WAVE', 'FLUX', 'STDEV' and 'FLAG' entries for IRAS21219m1757_dlw_qst.xdr.
    '''
    dir_in='../test/test_questfit/'
    path_out=dir_in

    dict1 = readsav(dir_in+'4978688_0.ideos.xdr', python_dict=True, verbose=False)
    dict2 = readsav(dir_in+'IRAS21219m1757_dlw_qst.xdr', python_dict=True, verbose=False)

    data1 = np.array(list(zip(dict1['sed'][0]['WAVE'], dict1['sed'][0]['FLUX']['JY'], dict1['sed'][0]['EFLUX']['JY'])), dtype=[ (('wave', 'WAVE'), 'O'), (('flux', 'FLUX'), 'O'), (('eflux', 'EFLUX'), 'O')] )
    np.save(path_out+'4978688_0.ideos.npy', data1.view(np.recarray))

    data2 = dict2['memstor']['AAR'][0]['DATA']
    np.save(path_out+'IRAS21219m1757_dlw_qst.npy', data2)



''' ----------------------------------------------------------
	Read in / define paths
 ---------------------------------------------------------- '''

parser = argparse.ArgumentParser()
parser.add_argument('--path_tpl', default='../data/questfit_templates/', help='Path pointing to the input .xdr templates')
parser.add_argument('--path_out', default='../data/questfit_templates/', help='Path where the output .npy templates should be stored')
parser.add_argument('--test_sources', default=False, help='True if test sources 4978688_0.ideos.xdr and IRAS21219m1757_dlw_qst.xdr should also be converted to .npy')

path_tpl = vars(parser.parse_args())['path_tpl']
path_out = vars(parser.parse_args())['path_out']
do_test_sources = vars(parser.parse_args())['test_sources']
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


if do_test_sources:
	example_sources_to_npy()



