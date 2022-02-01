# q3dfit
Python software for fitting integral field spectroscopic data

# Validation and testing
`q3dfit` can be parallelized across multiple processor cores using the
Message Passing Interface (MPI) standard. To enable this capability,
[install](https://www.mpich.org/downloads/) `mpich` on your hardware
and `pip install` the Python package `mpi4py.`

`mpich` install note for Macports: Run `sudo port select --set mpi
mpich-mp-fortran` to get default commands (like `mpiexec`) working.

These assume that Python is properly pathed via PYTHONPATH. In
multi-core processing, system path is used. Thus the tool you use to
run python (command line, Jupyter, Spyder) must inherit the system
path to be able to find, e.g., `mpiexec` and `q3dfit`. This can be
accomplished in the case of Jupyter or Spyder by running these
applications from the command line.

## ground-based data, rest-frame optical, single emission-line component, single spaxel, quasar

Data cube is from Gemini/GMOS observations of PG1411+442 

1. Download necessary files from the [Q3D Box folder](https://rhodes.box.com/s/q4zsp63ps01olkkh846k1nzbfw744gns):
   - `./testing/pg1411/pg1411rb3.fits`
   - `./testing/pg1411/pg1411qsotemplate.npy`
   - `./testing/pg1411/pg1411hosttemplate.npy`
2. Edit the input/output files in the Jupyter notebook `./jnb/run_q3dfit.ipynb`.
   - `infile` needs to point to `pg1411rb3.fits`
   - `qsotemplate` needs to point to `pg1411qsotemplate.npy`
   - `stellartemplates` needs to point to `pg1411hosttemplate.npy`
3. Run `q3df`/`q3da` from the notebook OR run `./test/test_pg1411c14r11.py` from a Python prompt.
4. Compare the output plots to those in the Q3D Box folder (subdirectory `./testing/pg1411/c14r11_output/`).

## Spitzer data (emission-line subtracted), rest-frame MIR, single emission-line component, single spaxel, quasar

Data "cube" is single spectrum of IRAS21219

--> This test no longer works because the fake emission line (`test-MIRLINE`) is in the wrong linelist.
A) Quasar-dominated example: Data "cube" is single spectrum of IRAS21219
1. Run `./test/test_f21219mir.py`.
2. Verify output plot in `./test/test_questfit/`.

B) Spectrum with strong MIR lines:  Data "cube" is single spectrum called 22128896 (shared by Erini / Nadia)

1. Edit the output path in the Jupyter notebook `./jnb/run_q3dfit_MIRlines_Spitzer.ipynb`.
2. Run `q3df`/`q3da` from the notebook OR run `./test/test_f22128896mir.py` from a Python prompt.
3. Compare outputs to files in `./test/test_questfit/`. Presently, two versions of the plots are made: a jpg file is created by `q3da`, and a png is created by `q3df` via `plot_quest`.

## NIRSPEC ETC simulation

1. Download necessary files from the [Q3D Box folder](https://rhodes.box.com/s/q4zsp63ps01olkkh846k1nzbfw744gns):
   - `./simulations/NIRSPEC-ETC/nirspec_etc_cube_both.fits`
   - `./simulations/NIRSPEC-ETC/nirspec_ETC_QSO.npy`
2. Edit the input/output files in the Jupyter notebook `./jnb/run_q3dfit_nirspec-etc.ipynb`.
3. Run `q3df`/`q3da` from the notebook OR run `./test/test_nirspec-etc.py` from a Python prompt.

## SDSS spectrum, rest-frame optical, two emission-line components, galaxy + emission lines

spectrum of Makani

1. Download necessary files from the [Q3D Box folder](https://rhodes.box.com/s/q4zsp63ps01olkkh846k1nzbfw744gns):
   - `./testing/makani/makanisdss.fits`
   - `./testing/makani/makani_stelmod.npy`
2. Edit the input/output files in the Jupyter notebook `./jnb/run_q3dfit_makani.ipynb`.
   - The variable `infile` needs to point to `makanisdss.fits`
   - The variable `stellartemplates` needs to point to `makani_stelmod.npy`
   - `outdir` is the location of the output files, including plots
   - `logfile` is an ASCII output file
3. Run `q3df`/`q3da` from the notebook OR run `./test/test_makanisdss.py` from a Python prompt.


## Mock ETC cube, rest-frame MIR, 1 PAH feature with [NeII]12.81 line + QSO background source (CB: need to double-check this with Andrey)

1. Download necessary file from the [Q3D Box folder](https://rhodes.box.com/s/q4zsp63ps01olkkh846k1nzbfw744gns):
   - `./simulations/MIRI-ETC-SIM/miri_etc_cube.fits`
2. Edit the input/output files in the Jupyter notebook `./jnb/run_q3dfit_MIRlines_mockETCcube.ipynb`.
   - The variable `infile` needs to point to 'miri_etc_cube.fits'
   - `outdir` is the location of the output files, including plots
   - `logfile` is an ASCII output file
   (- The config file 'cffilename' and QSO template 'qsotemplate' should be fine to leave as is, as these are relative paths. The jupyter notebook will generate the QSO template if it doesn't exist.) 
3. Run `q3df` from the notebook OR run `./test/test_miri.py` from a Python prompt.



## F05189-2524 UV STIS spectrum (test of polynomial fitter)

1. Download necessary files from the [Box folder](https://rhodes.box.com/s/8dshrdxl6b9ngg3wvdhvn79u8cicgmdp)
2. Edit the input/output files in the Jupyter notebook `./jnb/run_q3dfit_uvpoly.ipynb`.
3. Run `q3df`/`q3da` from the notebook OR run `./test/test_uvpoly.py` from a Python prompt.

--------------------

# Syntax of the MIR configuration file (.cf)  

The .cf file is placed in q3dfit/test/test_questfit/ and consists of 13 space-separated text columns of any width.
Below is an example:

|  A                                    |    B      |  C    |  D     |  E       |   F  |  G   |  H  |  I    |  J    | K  | L  | M  |
| -----------                            | --------- | ----- | ------ | -------- | ---- | ---- | --- | ---  | ----- |--- |--- |--- |
| source                            | miritest.npy |    11.55 |  13.45|   dummy  |   0.0|  0.0 |  X  |  0.0 |  0.0  | _  | _  | _  |
|template_poly       | miri_qsotemplate_flex.npy   |  0.059  |  1.    |      _   |     _ |    _ |   S|   0.0|  0.0  | _  | _  | _  |
|template             |        smith_nftemp4.npy   |  0.175  |  1.    | global   |    1.5|   1. |  S |   0.0|  0.0  | _  | _  | _  |
|blackbody            |                 warm       |  0.1    |  1.    | CHIAR06  |    1.5|   1. |  S | 250.0|  1.0  | _  | _  | _  |
|extinction           |        chiar06_i0857.npy   |  0.0    |  0.    | CHIAR06  |    0.0|   1. |  X |   0.0|  0.0  | _  | _  | _  |
|absorption           |        ice+hc_abs.npy      |  0.0    |  0.    | ice_hc   |   0.0 |  1.  | X  |  0.0 |  0.0  | _  | _  | _  |

- A: The type of data (template, blackbody, powerlaw, absorption, extinction, ...). Put 'source' for the data to be fitted.

- B: This is the filename to read in. It will be ignored for types 'blackbody' or 'powerlaw' as these are generated in the code itself.

- C: For source: lower wavelength limit. "-1" will use the lowest possible common wavelength. (CB: Double-check if -1 functionality is there)   
	For template, blackbody, powerlaw: normalization factor  
	For absorption: tau_peak  
	For extinction: any float; this will be ignored  

- D: For source: upper wavelength limit. "-1" will use the largest possible common wavelength. (CB: Double-check if -1 functionality is there)  
	For template, blackbody, powerlaw: fix/free parameter for the normalization. 1=fixed, 0=free.  
	For absorption: fix/free parameter for tau_peak. 1=fixed, 0=free  

- E: For extinction: shorthand name for the extinction curve  
	For absorption:  shorthand name for the ice absorption curve  
	For template, blackbody, powerlaw: In case of individual extinction applied to each component, set which exctinction curve should be applied via the shorthand defined for the extinction curve  
	**NOTE**: If this is set to 'global' for any row, the same global extinction and ice absorption will be applied to each fitting component (thus in the example above, the individual extinction settings are ignored). If instead individual extinction is used and this is set to _ or -, then no extinction will be applied.  
	For source: any string; will be ignored

- F: For template, blackbody, powerlaw: extinction value (A_V)  
	For source, extinction, absorption: any float; will be ignored  

- G: For template, bl, powerlaw: fix/free parameter for A_V. 1=fixed, 0=free  
	For source, extinction, absorption: any float; will be ignored  

- H: For template, blackbody, powerlaw: S=screen extinction, M=mixed extinction. (CB: Only screen has been tested so far)  
	For source, extinction, absorption: any string; will be ignored

- I: For blackbody: temperature (in K)  
	For powerlaw: index  
	For source, template, absorption, extinction: any float; will be ignored  

- J: For blackbody: fix/free parameter for temperature. 1=fixed, 0=free  
	For powerlaw: fix/free parameter for powerlaw index. 1=fixed, 0=free  
	For source, template, absorption, extinction: any float; will be ignored  

- K: For template, blackbody, powerlaw: In case of individual extinction/absorption applied to each component, set which absorption should be applied by the shorthand defined in column E.  
	For source, extinction, absorption: any string; will be ignored  
	**NOTE**: If this is set to _ or -, there will be no absorption applied to this curve (unless global is set for any component in column E which overrides this)  

- L: For template, blackbody, powerlaw: initial guess for the amplitude of the absorption  
        For source, extinction, absorption: any float/string; will be ignored  

- M: For template, blackbody, powerlaw: fix/free parameter for absorption amplitude. 1=fixed, 0=free
        For source, extinction, absorption: any float/string; will be ignored

