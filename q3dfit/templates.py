from __future__ import annotations
from numpy.typing import ArrayLike

import numpy as np

def read_bpass(infile: str,
               outfile: str,
               waverange: ArrayLike = [1., 100000.],
               binary: bool = False,
               zs: ArrayLike = [0.001, 0.002, 0.003, 0.004, 0.006, 0.008, 
                                0.010, 0.014, 0.020, 0.030, 0.040]) -> None:
    '''
    Read the BPASS templates into a dictionary. Templates located
    here: https://bpass.auckland.ac.nz/14.html. Download the gzipped
    tarball of the desired version and alpha-enhancement. Unpack it
    into a directory, and pass the path to the directory to this function.

    From p 14 of the v2.3 manual:
    These files contain the primary output of BPASS, which is the stellar spectral
    energy distribution (SED). Flux values are given every 1 Angstrom in the 
    range 1 - 100,000 A. Most users will wish to resample to lower resolution, 
    depending on their use case. We caution that some of the stellar atmospheres 
    we use also have slightly lower spectral resolution.
    Each file has 52 columns and 105 rows. The first column lists a wavelength 
    in angstroms, and each remaining column n (n>1) holds the model flux for the 
    population at an age of 10^(6+0.1*(n-2)) years at that wavelength.
    The units of flux are Solar Luminosities per Angstrom, normalised for a 
    cluster of 1e6 Msun formed in a single instantaneous burst. The total 
    luminosity of the SED can be simply calculated by summing all the rows 
    together. Or the flux over the wavelength range from, for example, 2000 to 
    3000 Angstroms can be calculated by summing the 2000th to 3000th rows.

    Parameters
    ----------
    infile
        The path to the BPASS template files.
    outfile
        The path to the output file where the numpy save file
        will be written. Should have a .npy extension.
    waverange
        Optional. The wavelength range to use for the templates, expressed in 
        Angstroms. Defaults to [1., 100000.].
    binary
        Optional. If True, the models with binary star evolution will be used.
        Defaults to False.
    zs
        Optional. The metallicities to include. The options are
        [0.001,0.002,0.003,0.004,0.006,0.008,0.010,0.014,0.020,0.030,0.040].
        Defaults to all metallicities.
    '''
    # number of metallicities
    nz = len(zs)
    # ages for one metallicity
    nages = 51
    # log age in years
    ages = 10.**(6. + 0.1 * np.arange(0, nages))
    # Output spectra will loop through zs, and then the ages for each
    # metallicity. 
    # So first we'll run through the ages for every metallicity:
    # result will be an array with [age0, age1, age2, ..., agemax,
    # age0, age1, age2, ..., agemax, ...]
    agesall = np.tile(ages, nz)
    # Now we need to repeat the metallicities for each age.
    # The result will be an array with [z0, z0, ..., z0, z1, z1, ..., z1, ...]
    zall = np.repeat(zs, nages)

    # wavelength spacing is 1 Angstrom
    waveall = np.arange(waverange[0], waverange[1] + 1., dtype=float)
    # output array will have shape (nwave, nz * nages)
    fluxall = np.zeros((len(waveall), nz * nages), dtype=float)

    for iz, z in enumerate(zs):
        # read the file for this metallicity
        sinorbin = 'sin'
        if binary:
            sinorbin = 'bin'

        # convert the metallicity to a zero-padded integer string
        # with three digits
        zstr = f'{int(z * 1000):03d}'
        # strip the alpha enhancement value from the infile path
        alph = infile.split('.a')[1][0:3]
        filename = f'{infile}/spectra-{sinorbin}-imf135_300.a{alph}.z{zstr}.dat'
        # read the data from the file
        data = np.loadtxt(filename, usecols=range(nages + 1))
        # extract the wavelengths and fluxes
        wave = data[:, 0]
        flux = data[:, 1:]
        # find the indices of the wavelengths that are within the desired range
        indices = np.where((wave >= waverange[0]) & (wave <= waverange[1]))[0]
        # normalize fluxes over desired wavelength range
        for i in range(flux.shape[1]):
            flux[indices, i] /= np.mean(flux[indices, i])
        # write the fluxes to the output array
        fluxall[:, iz * int(nages):(iz + 1) * int(nages)] = flux[indices, :]

    # save the output array to a numpy file
    np.save(outfile, {'lambda': waveall,
                      'flux': fluxall,
                      'ages': agesall,
                      'zs': zall})

    print(f'BPASS templates saved to {outfile}')