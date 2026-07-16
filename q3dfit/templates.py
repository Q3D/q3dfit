from __future__ import annotations
from numpy.typing import ArrayLike

import numpy as np

def read_bpass(infile: str,
               starmassfile: str = None,
               outfile: str = '',
               waverange: ArrayLike = [1., 100000.],
               binary: bool = False,
               zs: ArrayLike = [0.001, 0.002, 0.003, 0.004, 0.006, 0.008, 
                                0.010, 0.014, 0.020, 0.030, 0.040],
               R: int = 10000,
               outdir: str = '',
               agerange: ArrayLike = [6., 11.]) -> None:
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
    Each file has 52 columns and 10^5 rows. The first column lists a wavelength 
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
    starmassfile
        Optional. The path to the BPASS starmass template files. If provided,
        the starmass templates will be read and saved to the output file.
    outfile
        The path to the output file where the numpy save file
        will be written. Should have a .npy extension.
    outdir
        Optional. Will generate an output filename based on the other parameters
        and save in this directory. If not provided, will save to the path 
        specified in outfile. If not provided, outfile must be provided.
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
    R
        Optional. The spectral resolution of the templates. Defaults to 10000.
    agerange
        Optional. Array containing the minimum and maximum log ages of the templates to be used.
        Must be a multiple of 0.1 between 6.0 and 11.0.
        Defaults to [6., 11.].
    '''
    # number of metallicities
    nz = len(zs)
    # ages for one metallicity
    nages = round(((agerange[1] - agerange[0]) / 0.1) + 1)
    # log age in years
    ages = 10.**(agerange[0] + 0.1 * np.arange(0, nages))
    # initial age data index
    ageindex = round((agerange[0] - 6) / 0.1)
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
    #initialize normalization factors array
    normall = np.zeros_like(agesall, dtype=float)
    #initialize starmasses array
    starmassesall = np.zeros(nages*nz, dtype=float)
    
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
        data = np.loadtxt(filename)
        # extract the wavelengths and fluxes
        wave = data[:, 0]
        flux = data[:, ageindex + 1: ageindex + nages + 1]
        norms = np.zeros(nages)
        # find the indices of the wavelengths that are within the desired range
        indices = np.where((wave >= waverange[0]) & (wave <= waverange[1]))[0]
        # normalize fluxes over desired wavelength range
        for i in range(flux.shape[1]):
            norm_factor = np.mean(flux[indices, i])
            flux[indices, i] /= norm_factor
            norms[i] = norm_factor
        # storing normalization factors in output array
        normall[iz * nages : (iz + 1) * nages] = norms[:]
        # write the fluxes to the output array
        fluxall[:, iz * nages : (iz + 1) * nages] = flux[indices, :]
        if starmassfile is not None:    
            # strip the alpha enhancement value from the infile path
            filename = f'{starmassfile}/starmass-{sinorbin}-imf135_300.z{zstr}.dat'
            # read the data from the file
            data = np.loadtxt(filename)
            # read data into starmasses array
            starmassesall[iz * nages: (iz + 1) * nages] = data[ageindex : ageindex + nages, 1]
        
    # calculating sigma array for templates based on spectral resolution
    sigma = np.array([ i / (2.35 * R) for i in waveall], dtype=float)

    if outdir != '':
        indir = infile.split('/')[-2]
        outfile = f'{outdir}/{indir}_z{min(zall) * 1000:03.0f}to{max(zall) * 1000:03.0f}_{sinorbin}_lam{waverange[0]:.0f}to{waverange[1]:.0f}.npy'

    # assenbling the output dictionary
    outarr = {'lambda': waveall,
              'flux': fluxall,
              'ages': agesall,
              'zs': zall,
              'unit': 'Angstrom',
              'sigma' : sigma,
              'norm' : normall,
            }
    
    if starmassfile is not None:
        outarr['starmass'] = starmassesall

    # save the output array to a numpy file
    np.save(outfile, outarr)

    print(f'BPASS templates saved to {outfile}')

def trim_templates(startempfile: str,
                   ages: ArrayLike = None,
                   agerange: ArrayLike = [6., 11.],
                   zs: ArrayLike = None,
                   zrange: ArrayLike = [0.001, 0.040],
                   waverange: ArrayLike = [1., 100000.],
                   outfile: str = None):
    '''
    Trim a BPASS template file to the desired age, metallicity, and wavelength ranges. 
    Returns a dictionary containing the trimmed templates, and saves to a new file, if specified.

    Parameters
    ----------
    startempfile
        Path to the BPASS template file to be trimmed.
    ages
        Optional. Array of ages to include in the trimmed templates. If None, will use agerange to select ages. Defaults to None.
    agerange
        Optional. Array containing the minimum and maximum log ages of the templates to be used.
        Must be a multiple of 0.1 between 6.0 and 11.
        Defaults to [6., 11.].
    zs
        Optional. Array of metallicities to include in the trimmed templates. If None, will use zrange to select metallicities. Defaults to None.
    zrange
        Optional. Array containing the minimum and maximum metallicities of the templates to be used.
        Defaults to [0.001, 0.040].
    waverange
        Optional. The wavelength range to use for the templates, expressed in Angstroms. Defaults to [1., 100000.].
    outfile
        Optional. Path to the file where the trimmed templates will be saved. 
        If None, the templates will not be saved to a file. Defaults to None.
    '''
    templates_orig = np.load(startempfile, allow_pickle=True)[()]

    # generating masks for ages and metallicities based on the provided ranges or arrays
    if ages is None:
        agemask = (templates_orig['ages'] >= 10**agerange[0]) & (templates_orig['ages'] <= 10**agerange[1])
    else:
        agemask = np.isin(templates_orig['ages'], ages)
    
    if zs is None:
        zmask = (templates_orig['zs'] >= zrange[0]) & (templates_orig['zs'] <= zrange[1])
    else:
        zmask = np.isin(templates_orig['zs'], zs)

    # combining the age and metallicity masks to get the final mask for selecting templates
    tempmask = agemask & zmask

    # generating a mask for the wavelength range
    wmask = (templates_orig['lambda'] >= waverange[0]) & (templates_orig['lambda'] <= waverange[1])

    #initializing the dictionary for the trimmed templates
    templates_trimmed = {}

    # populating the trimmed templates dictionary with the selected ages, metallicities, wavelengths, and corresponding fluxes and other properties
    templates_trimmed['ages'] = templates_orig['ages'][tempmask]
    templates_trimmed['zs'] = templates_orig['zs'][tempmask]
    templates_trimmed['lambda'] = templates_orig['lambda'][wmask]
    templates_trimmed['flux'] = templates_orig['flux'][wmask][:, tempmask]
    templates_trimmed['sigma'] = templates_orig['sigma'][wmask]
    templates_trimmed['norm'] = templates_orig['norm'][tempmask]
    if 'starmass' in templates_orig: templates_trimmed['starmass'] = templates_orig['starmass'][tempmask]
    templates_trimmed['unit'] = templates_orig['unit']

    if outfile is not None:
        np.save(outfile, templates_trimmed)
        print(f'Trimmed templates saved to {outfile}')

    return templates_trimmed

def read_bpass_starmass(infile: str,
                       outfile: str = '',
                       binary: bool = False,
                       zs: ArrayLike = [0.001, 0.002, 0.003, 0.004, 0.006, 0.008, 
                                        0.010, 0.014, 0.020, 0.030, 0.040],
                       agerange: ArrayLike = [6., 11.]) -> None:
    '''
    Read the BPASS starmass templates into a dictionary. Templates located
    here: https://bpass.auckland.ac.nz/14.html. Download the gzipped
    tarball of the desired version and alpha-enhancement. Unpack it
    into a directory, and pass the path to the directory to this function.

    From p 16 of the v2.3 manual:
    These files contain the total mass of the surviving stellar population as a function of age,
    for a population of 106 Msun formed at t=0. These do not include the mass in compact
    remnants.
    Each file has 51 rows (one for each age bin) and 2 columns. The first column holds the
    log(age/years) of the population while the second holds the total mass of surviving stars
    in solar masses.
    For the binary files we have included a third, untested, column that includes the mass in
    stellar remnants, i.e. white dwarfs, neutron stars and black holes. We will add this column
    to the single star population in future.
    Filenames: starmass-<opt>-<imf>.<z>.dat

    Parameters
    ----------
    infile
        The path to the BPASS template files.
    outfile
        The path to the output file where the numpy save file
        will be written. Should have a .npy extension.
    binary
        Optional. If True, the models with binary star evolution will be used.
        Defaults to False.
    zs
        Optional. The metallicities to include. The options are
        [0.001,0.002,0.003,0.004,0.006,0.008,0.010,0.014,0.020,0.030,0.040].
        Defaults to all metallicities.
    agerange
        Optional. Array containing the minimum and maximum log ages of the templates to be used.
        Must be a multiple of 0.1 between 6.0 and 11.0.
        Defaults to [6., 11.].
    '''

    # number of metallicities
    nz = len(zs)
    # ages for one metallicity
    nages = round(((agerange[1] - agerange[0]) / 0.1) + 1)
    # finding index for ages
    ageindex = round((agerange[0] - 6) / 0.1)
    # Output masses will loop through zs, and then the ages for each
    # metallicity.
    # Initializimg masses array
    starmassesall = np.zeros(nages*nz)


    for iz, z in enumerate(zs):
        # read file for single or binaries
        sinorbin = 'sin'
        if binary:
            sinorbin = 'bin'

        # convert the metallicity to a zero-padded integer string
        # with three digits
        zstr = f'{int(round(z * 1000)):03d}'
        # strip the alpha enhancement value from the infile path
        filename = f'{infile}/starmass-{sinorbin}-imf135_300.z{zstr}.dat'
        # read the data from the file
        data = np.loadtxt(filename)
        # read data into starmasses array
        starmassesall[iz * int(nages): (iz + 1) * int(nages)] = data[ageindex : ageindex + nages, 1]
    
    np.save(outfile, starmassesall)