MIR Configuration
=================

.. warning::

   The configuration of the MIR fit is in active development. This
   page will be updated as MIRI data from the Q3D program become
   available.

**How to set up configuration file for MIR fitting**

Mid-infrared continuum of quasars and their host galaxies is affected by a multitude of absorption and emission components. In ``q3dfit``, these components are incorporated via templates. The ``q3dfit`` user can choose which templates are included and / or supply their own templates, and build an emission and absorption continuum component via .cf (configuration) file. 

Here is an example configuration file as it is read by ``q3dfit``:

.. code-block:: console

		source        miritest.npy              11.55  13.45   dummy    0.0 0.0  X   0.0  0.0   _   _   _  
		template_poly miri_qsotemplate_flex.npy 0.059    1.    _        _   _    S   0.0  0.0   _   _   _  
		template      smith_nftemp4.npy         0.175    1.    global   1.5 1.   S   0.0  0.0   _   _   _  
		blackbody     warm                      0.1      1.    CHIAR06  1.5 1.   S 250.0  1.0   _   _   _  
		extinction    chiar06_i0857.npy         0.0      0.    CHIAR06  0.0 1.   X   0.0  0.0   _   _   _  
		absorption    ice+hc_abs.npy            0.0      0.    ice_hc   0.0 1.   X   0.0  0.0   _   _   _  


Here is another, more complicated configuration file: 

.. code-block:: console

		source        miritest.npy              11.55    13.45 dummy    0.0 0.0  X   0.0  0.0   _   _   _
		#Below are the emission components: templates we are fitting along with a single blackbody and a powerlaw. 
		#The templates and the blackbody will be extincted by CHIAR06 and DRAIN03 models and ice absorption.
                template      smith_nftemp3.npy         0.1      1.   CHIAR06   0.0 0.0  S   0.0  0.0 ice_hc 0.0  0.
		template      smith_nftemp4.npy         0.1      1.   CHIAR06   0.0 0.0  S   0.0  0.0 ice_hc 0.0  0.
		template  dens1e4:sifrom:run010002.npy  0.1      1.   DRAINE03  0.0 0.0  S   0.0  0.0  _   _   _
		blackbody     warm                      0.1      1.   CHIAR06   0.0 0.0  S 250.0  1.0 ice_hc 0.0  1.
		powerlaw      steep                     0.1      1.   DRAINE03  0.0 0.0  S    -1  1.0  _   _   _
		#The last rows of the configuration file deal with dust and ice extinction and absorption that are used to extinct the emission. 
		extinction    draine03.npy              0.0      0.   DRAINE03  0.0 0.0  X   0.0  0.0  _   _   _
		extinction    chiar06_i0857.npy         0.0      0.   CHIAR06   0.0 0.0  X   0.0  0.0  _   _   _
		absorption    ice+hc_abs.npy            0.0      0.   ice_hc    0.0 0.0  X   0.0  0.0  _   _   _


The .cf file consists of 13 space-separated text columns of any width. The hash symbol ``#`` indicates rows that are comments and are ignored by the code. Below is the detailed explanation for all the rows and columns, taking the first file as an example. 

.. list-table:: MIR configuration table
   :widths: 15 20 10 10 15 10 10 10 10 10 10 10 10
   :header-rows: 1

   * - A
     - B
     - C
     - D
     - E
     - F
     - G
     - H 
     - I
     - J
     - K
     - L
     - M
   * - source
     - miritest.npy     
     - -1  
     - -1   
     - n/a     
     - 0.0  
     - 0   
     - -
     - 0.0
     - 0 
     - _
     - _
     - _
   * - template_poly
     - miri_qsotemplate_flex.npy
     - 0.059
     - 1   
     - _
     - _
     - _
     - S
     - 0.0
     - 0 
     - _
     - _
     - _
   * - template
     - miri_qsotemplate_flex.npy
     - 0.175
     - 1   
     - global
     - 1.5
     - 1
     - S
     - 0.0
     - 0 
     - _
     - _
     - _
   * - blackbody
     - warm
     - 0.1
     - 1   
     - CHIAR06
     - 1.5
     - 1
     - S
     - 250.0
     - 0 
     - _
     - _
     - _
   * - extinction
     - chiar06_i0857.npy
     - 0.0
     - 0  
     - CHIAR06
     - 0.0
     - 1
     - -
     - 0.0
     - 0 
     - _
     - _
     - _
   * - absorption
     - ice+hc_abs.npy
     - 0.0
     - 0.  
     - ice_hc
     - 0.0
     - 1.
     - -
     - 0
     - 0 
     - _
     - _
     - _

A: The type of data (template, blackbody, powerlaw, absorption, extinction, ...). Put 'source' for the data to be fitted.

B: This is the filename to read in. It will be ignored for types 'blackbody' or 'powerlaw' as these are generated in the code itself. Some popular templates are pre-loaded in ``q3dfit/data/questfit_templates``. These include silicate opacity curves from `Chiar and Tielens (2006) <https://ui.adsabs.harvard.edu/abs/2006ApJ...637..774C/abstract>`_, astronomical dust opacity curves from `Draine (2003) <https://ui.adsabs.harvard.edu/abs/2003ApJ...598.1017D/abstract>`_, PAH templates from `Smith et al. (2007) <https://ui.adsabs.harvard.edu/abs/2007ApJ...656..770S/abstract>`_, and absorption curves of ices and hydrocarbons from `Spoon et al. (2004) <https://ui.adsabs.harvard.edu/abs/2004ApJS..154..184S/abstract>`_. 

C: For source: lower wavelength limit. "-1" will use the lowest possible common wavelength. [Still in development.]
	For template, blackbody, powerlaw: normalization factor.
  
	For absorption: tau_peak.  

	For extinction: any float; this will be ignored.  

D: For source: upper wavelength limit. "-1" will use the largest possible common wavelength. [Still in development.] 
	For template, blackbody, powerlaw: fix/free parameter for the normalization. 0=fixed, 1=free.  

	For absorption: fix/free parameter for tau_peak. 1=fixed, 0=free. [Note: not clear this is correct; may do nothing]

E: For extinction: shorthand name for the extinction curve  
	For absorption:  shorthand name for the ice absorption curve.
  
	For template, blackbody, powerlaw: In case of individual extinction applied to each component, set which exctinction curve should be applied via the shorthand defined for the extinction curve.
  
	**NOTE:** If this is set to 'global' for any row, the same global extinction and ice absorption will be applied to each fitting component (thus in the example above, the individual extinction settings will be ignored). If instead individual extinction is used and this is set to _ or -, then no extinction will be applied. 
 
	For source: any string; will be ignored

F: For template, blackbody, powerlaw: extinction value (A_V)  
	For source, extinction, absorption: any float; will be ignored  

G: For template, blackbody, powerlaw: fix/free parameter for A_V. 0=fixed, 1=free  
	For source, extinction, absorption: any float; will be ignored  

H: For template, blackbody, powerlaw: S=screen extinction, M=mixed extinction. [Still in development; only S mode has been tested.]
	For source, extinction, absorption: any string; will be ignored

I: For blackbody: temperature (in K), for powerlaw: index.
  
	For source, template, absorption, extinction: any float; will be ignored  

J: For blackbody: fix/free parameter for temperature. 0=fixed, 1=free  
	For powerlaw: fix/free parameter for powerlaw index. 0=fixed, 1=free.
  
	For source, template, absorption, extinction: any float; will be ignored  

K: For template, blackbody, powerlaw: In case of individual extinction/absorption applied to each component, set which absorption should be applied by the shorthand defined in column E.  
	For source, extinction, absorption: any string; will be ignored.
  
	**NOTE:** If this is set to _ or -, there will be no absorption applied to this curve (unless global is set for any component in column E which overrides this)  

L: For template, blackbody, powerlaw: initial guess for the amplitude of the absorption  
        For source, extinction, absorption: any float/string; will be ignored.  

M: For template, blackbody, powerlaw: fix/free parameter for absorption amplitude. 0=fixed, 1=free
        For source, extinction, absorption: any float/string; will be ignored
