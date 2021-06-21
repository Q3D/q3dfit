# q3dfit
Python software for fitting integral field spectroscopic data

# Validation and testing

These assume that Python is properly pathed.

## ground-based data, rest-frame optical, single emission-line component, single spaxel, quasar

Data cube is from Gemini/GMOS observations of PG1411+442 

1. Download necessary files from the [Q3D Box folder](https://rhodes.box.com/s/q4zsp63ps01olkkh846k1nzbfw744gns):
   - `./testing/pg1411/pg1411rb3.fits`
   - `./testing/pg1411/pg1411qsotemplate.npy`
   - `./testing/pg1411/pg1411hosttemplate.npy`
2. Edit the input/output files in the Jupyter notebook `run_q3dfit.ipynb`.
   - `infile` needs to point to `pg1411rb3.fits`
   - `qsotemplate` needs to point to `pg1411qsotemplate.npy`
   - `stellartemplates` needs to point to `pg1411hosttemplate.npy`
3. Run q3df/q3da from the notebook.
4. Compare the output plots to those in the Q3D Box folder (subdirectory `./testing/pg1411/c14r11_output/`).


## Spitzer data (emission-line subtracted), rest-frame MIR, single emission-line component, single spaxel, quasar

Data "cube" is single spectrum of IRAS21219

1. Run `./test/test_f21219mir.py`.
2. Verify output plot in `./test/test_questfit/`.


## NIRSPEC simulation data in the hei box

Data cube can be downloaded from the [heibox](https://heibox.uni-heidelberg.de/library/06eb022c-6252-40ea-aaa9-88af6d7d876d/Q3D/Simulations/May_2021)

1. Modify `./init/nirtest.py` accordingly and prepare the qso template to use (using `test_makeqsotemplate_nir.py` for now)
2. Run `./test/test_nirspec.py`
