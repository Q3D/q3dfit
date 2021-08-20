# q3dfit
Python software for fitting integral field spectroscopic data

# Validation and testing
`q3dfit` can be parallelized across multiple processor cores using the
Message Passing Interface (MPI) standard. To enable this capability,
[install](https://www.mpich.org/downloads/) `mpich` on your hardware
and `pip install` the Python package `mpi4py.`

These assume that Python is properly pathed.

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

## NIRSPEC simulation data in the hei box

Data cube can be downloaded from the [heibox](https://heibox.uni-heidelberg.de/library/06eb022c-6252-40ea-aaa9-88af6d7d876d/Q3D/Simulations/May_2021)

1. Modify `./init/nirtest.py` accordingly and prepare the qso template to use (using `test_makeqsotemplate_nir.py` for now)
2. Run `./test/test_nirspec.py`

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
