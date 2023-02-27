Fitting
*******

Jupyter notebooks with fitting examples can be found `here
<https://github.com/Q3D/q3dfit/tree/main/jnb>`_. You can download and
run these examples on your own computer, either directly or by cloning
the ``q3dfit`` repository. Or, you can test the notebooks in binder:

.. image:: https://mybinder.org/badge_logo.svg
 :target: https://mybinder.org/v2/gh/Q3D/q3dfit/main

Released notebooks
==================

In the present release, three example notebooks are available.
	  
1. **Ground-based data, rest-frame optical, single emission-line
   component, quasar**

   Data cube is from Gemini/GMOS observations of `PG1411+442 <https://ned.ipac.caltech.edu/byname?objname=PG1411%2B442&hconst=67.8&omegam=0.308&omegav=0.692&wmap=4&corr_z=1>`_, as published in `Rupke et al. 2017 <https://ui.adsabs.harvard.edu/abs/2017ApJ...850...40R/abstract>`_.

   Jupyter notebook: `q3dfit_example_restframeopt_ground.ipynb <https://github.com/Q3D/q3dfit/blob/main/jnb/q3dfit_example_restframeopt_ground.ipynb>`_
	  
2. **JWST/NIRSpec-IFU data, rest-frame optical, multiple components,
   quasar**

   Data cube is from observations of `SDSSJ165202.64+172852.3 <https://ned.ipac.caltech.edu/byname?objname=SDSSJ165202.64%2B172852.3&hconst=67.8&omegam=0.308&omegav=0.692&wmap=4&corr_z=1>`_, as published in `Wylezalek et al. 2022 <https://ui.adsabs.harvard.edu/abs/2022arXiv221010074W/abstract>`_.

   Jupyter notebook: `q3dfit_example_restframeopt_nirspec.ipynb <https://github.com/Q3D/q3dfit/blob/main/jnb/q3dfit_example_restframeopt_nirspec.ipynb>`_

3. **Spitzer data, rest-frame mid-IR, single emission-line component, single spectrum, galaxy**
   
   Spectrum is a `low-resolution Spitzer IRS observation <https://cassis.sirtf.com/atlas/cgi/onespectrum.py?aorkey=22128896&ptg=0>`_ of `2MASX J15561599+3951374 <http://ned.ipac.caltech.edu/cgi-bin/objsearch?objname=2MASX%20J15561599%2B3951374&extend=no&hconst=73&omegam=0.27&omegav=0.73&corr_z=1&out_csys=Equatorial&out_equinox=J2000.0&obj_sort=RA+or+Longitude&of=pre_text&zv_breaker=30000.0&list_limit=5&img_stamp=YES#ObjNo1>`_.

   Jupyter notebook: `q3dfit_example_restframeMIR_Spitzer.ipynb <https://github.com/Q3D/q3dfit/blob/main/jnb/q3dfit_example_restframe
   MIR_Spitzer.ipynb>`_

   
Notebooks in progress
=====================

One MIR example notebook is also available from v0.1.0 of ``q3dfit``;
this will be updated as MIRI data from the Q3D program become
available. To run thos notebooks, first install v0.1.0 of ``q3dfit``:

    .. code-block:: console

        pip install q3dfit==0.1.0

4. **Mock ETC cube, rest-frame mid-IR, 1 PAH feature with [NeII]12.81
   line, quasar**

   Jupyter notebook: `run_q3dfit_MIRlines_mockETCcube.ipynb <https://github.com/Q3D/q3dfit/blob/main/jnb/run_q3dfit_MIRlines_mockETCcube.ipynb>`_

.. 
 SDSS spectrum, rest-frame optical, two emission-line components,
 galaxy + emission lines
 spectrum of Makani
 - Download necessary files ...
