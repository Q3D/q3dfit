How to properly select the dispersion files in the initproc q3dfit initialization

1. create a dictionary of dispersions to use 

spect_instrum = {'instrument_name':['list of dispersers']}
spectres_convolve = {'ws_instrum':spect_instrum,'ws_method':2}

2. how to select instrument and disperser? 

- check the fits files in '/q3dfit/data/dispersion_files' directory

name has syntax: 'instrument_disperser_disp.fits'

EXAMPLE: 
'jwst_nirspec_g140h_disp.fits' --> spect_instrum = {'jwst_nirspec':['g140h']}
'flat_r1500_disp.fits'         --> spect_instrum = {'flat':['r1500']}

3. make sure the initproc structure contains the key: 
initproc{...., 'spect_convol':spectres_convolve,...}

4. run q3df() as usual

* by default, if initproc['spect_convol'] does not exist or if spectres_convolve == None, wavelength dependent convolution will not happen