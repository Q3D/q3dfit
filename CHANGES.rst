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
