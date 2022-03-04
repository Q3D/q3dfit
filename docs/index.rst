Q3DFIT
===========

Q3DFIT is custom software for scientific analysis of integral field unit (IFU) spectroscopy of quasars and their host galaxies, specifically aimed at producing science-ready measurements from James Webb Space Telescope (JWST) IFU spectrographs. Q3DFIT takes advantage of the spectral differences between quasars and their host galaxies for maximal-contrast subtraction of the quasar point-spread function (PSF) to reveal and characterize the faint extended emission of the host galaxy. Host galaxy emission is carefully fit with a combination of stellar continuum, emission and absorption of dust and ices, and ionic and molecular emission lines. 

Q3DFIT has been tested on ground-based data where PSF is weakly wavelength-dependent. The update of Q3DFIT to the case of the strongly wavelength-dependent James Webb Space Telescope (JWST) PSF is currently in development. 

.. warning::

   This software has not yet been released to the public. You are viewing pre-release test documentation and content. 

Our papers describing Q3DFIT and all its functionalities are currently in preparation and the links will be posted here. If you use this package before these papers are published, kindly cite the following references:

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

The website for the project listing lead team members is `available here <https://wwwstaff.ari.uni-heidelberg.de/dwylezalek/q3d.html>`_. We gratefully acknowledge funding provided by NASA through a contract issues by Space Telescope Science Institute and in-kind contributions by leading and contributing members at the University of Heidelberg, Rhodes College, Johns Hopkins University, and University of Maryland College Park. 

.. toctree::
   :maxdepth: 1
   :caption: Contents:
   
   self

   installation

..   examples/one_spaxel

