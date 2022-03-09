Testing
=======

**Ground-based data, rest-frame optical, single emission-line component, single spaxel, quasar**

Data cube is from Gemini/GMOS observations of `PG1411+442
<https://ned.ipac.caltech.edu/byname?objname=PG1411%2B442&hconst=67.8&omegam=0.308&omegav=0.692&wmap=4&corr_z=1>`_,
as published in `Rupke et al. 2017
<https://ui.adsabs.harvard.edu/abs/2017ApJ...850...40R/abstract>`_.

#. Download necessary files from the Q3D `public Box folder <https://rhodes.box.com/s/2gxo68ffikvetl9f97qpnp6hqil3y128>`_: ``pg1411-gmos.fits``, ``pg1411-stellar-model.npy``

#. Edit the input/output files in the Jupyter notebook ``./jnb/run_q3dfit.ipynb``.

#. Run ``q3df`` and ``q3da`` from the notebook.

#. Compare the output plots to those in the Q3D public folder (subdirectory ``./pg1411-gmos-output/``).

**Spitzer data, rest-frame mid-IR, single emission-line component, single spectrum, galaxy**
   
Spectrum is a `low-resolution Spitzer IRS observation <https://cassis.sirtf.com/atlas/cgi/onespectrum.py?aorkey=22128896&ptg=0>`_ of `2MASX J15561599+3951374 <http://ned.ipac.caltech.edu/cgi-bin/objsearch?objname=2MASX%20J15561599%2B3951374&extend=no&hconst=73&omegam=0.27&omegav=0.73&corr_z=1&out_csys=Equatorial&out_equinox=J2000.0&obj_sort=RA+or+Longitude&of=pre_text&zv_breaker=30000.0&list_limit=5&img_stamp=YES#ObjNo1>`_.

#. Edit the output path in the Jupyter notebook ``./jnb/run_q3dfit_MIRlines_Spitzer.ipynb``.

#. Run ``q3df`` and ``q3da`` from the notebook.

**JWST/NIRSPEC ETC simulation, rest-frame optical, single emission-line component, single spaxel, quasar**

Input to simulation is same data cube as in the first test (PG1411+442).

#. Download necessary file from the Q3D `public Box folder <https://rhodes.box.com/s/2gxo68ffikvetl9f97qpnp6hqil3y128>`_: ``nirspec-etc-simulation.fits``

#. Edit the input/output files in the Jupyter notebook ``./jnb/run_q3dfit_nirspec-etc.ipynb``.

#. Run ``q3df`` and ``q3da`` from the notebook.

**Mock ETC cube, rest-frame mid-IR, 1 PAH feature with [NeII]12.81 line, quasar**

#. Download necessary file from the Q3D `public Box folder <https://rhodes.box.com/s/2gxo68ffikvetl9f97qpnp6hqil3y128>`_: ``miri-etc-simulation.fits``

#. Edit the input/output files in the Jupyter notebook ``./jnb/run_q3dfit_MIRlines_mockETCcube.ipynb``.

#. Run ``q3df`` and ``q3da`` from the notebook.

.. 
 SDSS spectrum, rest-frame optical, two emission-line components,
 galaxy + emission lines
 spectrum of Makani
 - Download necessary files ...
