1.2.0-beta (11 Oct 2023)
------------------------
- Cube.convolve now allows Gaussian and boxcar convolutions.
- Bugfix: Cube.convolve was not conserving flux.
- Bugfix: sphinx_rtd_theme now part of rtd requirements.

1.1.4 (11 Jul 2023)
-------------------
- Reverting lmfit method to least_squares for PG1411 test notebook.
- Added two least_squares arguments to PG1411 test notebook to improve convergence.
- Reverting lower bounds on polynomials in qsohostfcn.py back to 0 uniformly, as opposed to alternating no constraint and pos. def.
- Bugfix: np.float() -> np.float32() 

1.1.3rc2 (25 May 2023)
------------------
- Cleaning up jnb dir and .gitignore.

1.1.3rc1 (25 May 2023)
------------------
- Bugfix in continuum fit arguments, introduced by questfit tweaks
  
1.1.2 (24 May 2023)
------------------
- Bugfix: restline, observedlinez to remove hardcoded paths.
- Added flux, wavelength information to q3dout class.
- Optional line flux error computation from residual.
- Enhanced component sorting in q3dcollect.
- Hotfix: error in checking nan status of q3do.perror in fitspec.
- Various changes/fixes related to questfit and MIR plotting.

1.1.1 (6 Apr 2023)
------------------
- Changed checkcomp to check S/N on peak flux rather than total flux.
- Added option to change lmfit method to, e.g., leastsq, in line and cont. fitting.
- Changed radius = 0 case in spectral extraction to single spaxel.
- readcube.makeqsotemplate now invokes readcube.specextract for either a single-spaxel or a circular extraction. col and row can be specified.
- Bugfix: total flux in q3dout.sepfitpars()
- Execute_fitloop now logs core # explicitly
- Made plotquest() compatible with object architecture via new function plotdecomp().
- Bugfix: 'IR' option in plotcont() now works when some components (e.g. QSO templates) are not affected by extinction / absorption.
- Bugfix: redshift mistake within questfit()
- Made 'decompose_qso_fit' functionality work with updated architecture
- Enable white mode in plot_decomp(), include line fit in residuals if specified
- Added subone option to checkcomp.
- Bugfix: if all points rejected, abort fit.
- Add calculation of fluxpkerr based on error spectrum in case of np.nan due to bounds.
- Bugfix: flux calculation for lines (sigmaerr was wrong);
- Bugfix: output file for plots now defaults to output file specified in method call rather than file defined in load_q3dout()
- q3dout now contains parameter error dictionary with fluxpkerr computed from error spectrum.
- Bugfix: constrained_layout vs. tight_layout in q3dpro
- Bugfix: removed minima on even exponential terms in qsohostfcn to prevent qso and stellar templates from flipping sign
- Cleaned up treatment of input/output flux and wave units, including more verbose output for debugging.
- Relaxed version requirements in setup.
- Hotfix (10 Apr 2023): perror check in fitspec.

1.1.0 (27 Feb 2023)
-------------------

- MIR Spitzer notebook ported to new framework.

Patches:
- q3di now a required argument to q3do.line_fit()
- forcefloat64 now an option to q3di, for forcing 64-bit float inputs to
  continuum fitting routine
- Bugfix: q3dutil typo on line 209
- Linelist cleanups, additions

1.0.1 (7 Dec 2022)
------------------

Patches:
- Fixed bug in initialization of line ratio constraints. Added text
  better describing these constraints in notebooks.
- Fixed error in multicore processing due to conflicting
  filenames. math.py, utility.py, and q3dfit.py renamed to q3dmath.py,
  q3dutil.py, and q3df.py.
- All inputs to LMFIT now float32 to prevent numerical errors.
- Added sphinx processing for readthedocs.
- Fixed link errors in readthedocs.
- Bugfix: checkcomp now working properly.
- Misc. bugfixes.
  
1.0.0 (15 Nov 2022)
-------------------

Release for JWST Cycle 2 Call for Proposals. MIR fitting still in
progress due to lack of Q3D MIRI data, pending resolution of MIRI
grating issue.
- Software tested on NIRSpec data of J1652.
- Initialization dictionary converted to q3din class.
- Fit output now q3dout class.
- Plots of fit results moved to methods of q3dout class.
- Renaming / combining / clean-up of files.

0.1.0
-----

First release.
