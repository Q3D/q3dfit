import os.path
import numpy as np

# This may be unique to the user, insert your path to the q3dfit/ folder here
import sys
if '../../' not in sys.path:
    sys.path.append('../../')
from q3dfit.common import questfit_readcf


# This is unique to the user, name the function after your object.
def f21219mir():

    # These are unique to the user
    # bad=1e99
    ncols = 17
    nrows = 26
    fitrange = np.array([5.422479152679443, 29.980998992919922])*10000  # angstrom

    # These are unique to the user
    infile = '../test/test_questfit/IRAS21219m1757_dlw_qst_mock_cube.fits'
    mapdir = '../test/test_questfit/'
    outdir = mapdir
    logfile = outdir+'test_questfit_fitlog.txt'


    ### for our test object, pg1411, nothing needs to be changed here for now, make more flexible later


    ### more MIR settings
    #   These are unique to the user
    #  Include Spitzer source (independently of PG1411 for now for testing purposes)
    global_ice_model = 'ice_hc'
    global_ext_model = 'CHIAR06'
    cffilename = '../test/test_questfit/IRAS21219m1757_dlw_qst.cf'
    config_file = questfit_readcf.readcf(cffilename)

    # Required parameters

    if not os.path.isfile(infile):
        print('Data cube not found.')

    # Lines to fit.
    lines = ['test-MIRLINE']

    # Max no. of components.
    maxncomp = 1

    # Initialize line ties, n_comps, z_inits, and sig_inits.
    linetie = dict()
    ncomp = dict()
    zinit_gas = dict()
    siginit_gas = dict()
    for i in lines:
        linetie[i] = 'test-MIRLINE'
        ncomp[i] = np.full((ncols, nrows), maxncomp)
        zinit_gas[i] = np.full((ncols, nrows, maxncomp), 0.0898)
        siginit_gas[i] = np.full(maxncomp, 50)
        zinit_stars = np.full((ncols, nrows), 0.0898)

    #
    # Optional pars
    #

    # Parameters for emission line plotting
    linoth = np.full((1, 1), '', dtype=object)
    linoth[0, 0] = 'test-MIRLINE'
    argspltlin1 = {'nx': 1,
                   'ny': 1,
                   'label': ['test-MIRLINE'],
                   'wave': [168000.0],
                   'off': [[-500, 500]],
                   'linoth': linoth}

    # Velocity dispersion limits and fixed values
    siglim_gas = np.ndarray(2)
    siglim_gas[:] = [5, 1500]

    #
    # Output structure
    #

    init = { \
            # Required pars
            'fcninitpar': 'gmos',
            'fitran': fitrange,
            'fluxunits': 1,
            'infile': infile,
            'label':
                config_file['source'][0].replace('.ideos','').replace('.npy', ''),
            'lines': lines,
            'linetie': linetie,
            'maxncomp': maxncomp,
            'name': 'IRAS21219m1757_dlw_qst',
            'ncomp': ncomp,
            'mapdir': mapdir,
            'outdir': outdir,
            'zinit_stars': zinit_stars,
            'zinit_gas': zinit_gas,
            'zsys_gas': 0.0898,
            # Optional pars
            'argscontfit': {'config_file': cffilename,
                            'global_ice_model': global_ice_model,
                            'global_ext_model': global_ext_model,
                            'models_dictionary': {},
                            'template_dictionary': {}},
            'argslinelist': {'vacuum': False},
            'argspltlin1': argspltlin1,
            'fcncheckcomp': 'checkcomp',
            'fcncontfit': 'questfit',
            'maskwidths_def': 500,
            'emlsigcut': 2,
            'logfile': logfile,
            'siglim_gas': siglim_gas,
            'siginit_gas': siginit_gas,
            'siginit_stars': 50,
            'nocvdf': 1,
            'waveext': 4,
            'datext': 1,
            'varext': 2,
            'dqext': 3,
            'zerodq': True,
            'plotMIR': True,
        }

    return(init)
