q3dfit
======
	
``q3dfit`` is custom software for scientific analysis of integral
field unit (IFU) spectroscopy of quasars and their host galaxies,
specifically aimed at producing science-ready measurements from James
Webb Space Telescope (JWST) IFU spectrographs. ``q3dfit`` takes advantage
of the spectral differences between quasars and their host galaxies
for maximal-contrast subtraction of the quasar point-spread function
(PSF) to reveal and characterize the faint extended emission of the
host galaxy. Host galaxy emission is carefully fit with a combination
of stellar continuum, emission and absorption of dust and ices, and
ionic and molecular emission lines.

.. image:: Q3D_logo.png
  :width: 200
  :align: left
  :alt: Q3D logo

``q3dfit`` originated with the JWST Early Release Science program
`Imaging Spectroscopy of Quasar Hosts with JWST analyzed with a
powerful new PSF Decomposition and Spectral Analysis
Package <https://q3d.github.io/>`_, or *Q3D*. *Q3D* targeted three
quasars at redshifts 0.435, 1.593, 2.949 with both NIRSpec and
MIRI-MRS.

Descriptions of the use of ``q3dfit`` can be found in the *Q3D* papers:

- J1652 (z=2.949) NIRSpec: `Wylezalek et al. 2022 <https://ui.adsabs.harvard.edu/abs/2022ApJ...940L...7W/abstract>`_, `Vayner et al. 2023 <https://ui.adsabs.harvard.edu/abs/2023ApJ...955...92V/abstract>`_, and `Vayner et al. 2024 <https://ui.adsabs.harvard.edu/abs/2024ApJ...960..126V/abstract>`_

- J1652 MIRI: `Bertemes et al. 2024, submitted <https://ui.adsabs.harvard.edu/abs/2024arXiv240414475B/abstract>`_

- XID 2028 (z=1.593) NIRSpec: `Veilleux et al. 2023 <https://ui.adsabs.harvard.edu/abs/2023ApJ...953...56V/abstract>`_
  
- F2M1106 (z=0.435) MIRI: `Rupke et al. 2024 <https://ui.adsabs.harvard.edu/abs/2023ApJ...953L..26R/abstract>`_

The developers of ``q3dfit`` are:
* David Rupke (Rhodes College, software lead)
* Dominika Wylezalek (University of Heidelberg, PI)
* Nadia Zakamska (Johns Hopkins University, CoPI)
* Sylvain Veilleux (University of Maryland College Park, CoPI)
* Andrey Vayner (Johns Hopkins University, primary developer)
* Caroline Bertemes (Heidelberg, primary developer)
* Yuzo Ishikawa (Johns Hopkins University, primary developer)
* Weizhe Liu (University of Maryland College Park, primary developer)
* Carlos Anicetti (Johns Hopkins University, contributor)
* Grace Lim (Rhodes College, contributor)
* Ryan McCrory (Rhodes College, contributor)
* Anna Murphree (Rhodes College and University of Hawai'i, contributor)
* Lillian Whitesell (Rhodes College, contributor)
  
.. note:: Please use ``q3dfit`` and let us know if you find a bug or
   have a feature request. To do so, `submit an issue on GitHub
   <https://github.com/Q3D/q3dfit/issues>`_.

The ``q3dfit`` paper is in preparation. In the meantime, please cite
`Rupke et
al. 2023 <https://ui.adsabs.harvard.edu/abs/2023ascl.soft10004R/abstract>`_:

.. code-block:: none

   @software{2023ascl.soft10004R,
   author = {{Rupke}, David and {Wylezalek}, Dominika and {Zakamska}, Nadia and {Veilleux}, Sylvain and {Vayner}, Andrey and {Bertemes}, Caroline and {Ishikawa}, Yuzo and {Liu}, Weizhe and {Lim}, Hui Xian Grace and {Murphree}, Grey and {Whitesell}, Lillian and {McCrory}, Ryan and {Anicetti}, Carlos},
   title = "{q3dfit: PSF decomposition and spectral analysis for JWST-IFU spectroscopy}",
   howpublished = {Astrophysics Source Code Library, record ascl:2310.004},
   year = 2023,
   month = oct,
   eid = {ascl:2310.004},
   adsurl = {https://ui.adsabs.harvard.edu/abs/2023ascl.soft10004R},
   adsnote = {Provided by the SAO/NASA Astrophysics Data System}
   }

You may also wish to cite the IDL-based direct ancestors of
``q3dfit``:

.. code-block:: none

   @MISC{2014ascl.soft09005R,
   author = {{Rupke}, David S.~N.},
   title = "{IFSFIT: Spectral Fitting for Integral Field Spectrographs}",
   keywords = {Software},
   year = 2014,
   month = sep,
   eid = {ascl:1409.005},
   pages = {ascl:1409.005},
   archivePrefix = {ascl},
   eprint = {1409.005},
   adsurl = {https://ui.adsabs.harvard.edu/abs/2014ascl.soft09005R},
   adsnote = {Provided by the SAO/NASA Astrophysics Data System}
   }

   @MISC{2021ascl.soft12002R,
   author = {{Rupke}, D.~S.~N. and {Schweitzer}, M. and {Viola}, V. and {Lutz}, D. and {Sturm}, E. and {Spoon}, H. and {Veilleux}, S. and {Kim}, D. -C.},
   title = "{QUESTFIT: Fitter for mid-infrared galaxy spectra}",
   keywords = {Software},
   year = 2021,
   month = dec,
   eid = {ascl:2112.002},
   pages = {ascl:2112.002},
   archivePrefix = {ascl},
   eprint = {2112.002},
   adsurl = {https://ui.adsabs.harvard.edu/abs/2021ascl.soft12002R},
   adsnote = {Provided by the SAO/NASA Astrophysics Data System}
   }

We acknowledge funding provided by NASA through a contract issued by
Space Telescope Science Institute for support of Early Release Science
observations with JWST and in-kind contributions by contributors at
the University of Heidelberg, Rhodes College, Johns Hopkins
University, and University of Maryland College Park.

.. image:: https://img.shields.io/badge/license-GNU%20GPL%20v.3.0-blue
  :alt: license GNU GPL v.3.0

.. toctree::
   :maxdepth: 2
   :caption: Contents:
   
   self

   installation

   fitting

   MIR-configuration

   Module index <source/q3dfit>
   
..   examples/one_spaxel

Indices and tables
==================
   * :ref:`genindex`
   * :ref:`modindex`
   * :ref:`search`
