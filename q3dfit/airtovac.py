def airtovac(wv, waveunit='Angstrom'):
    """
    Takes an array of wavelengths in air and converts them to vacuum
        using eq. 3 from Morton et al. 1991 ApJSS 77 119
        
    Parameters
    ----------
    wv : ndarray, shape (n,)
       Input wavelengths. Array of real elements with shape (n,)
       where 'n' is the number of wavelengths
    waveunit : str, optional
       Wavelength unit, could be 'Angstrom' or 'micron', 
       default is Angstrom
       
    Returns
    -------
    ndarray, shape (n,)
       An array of the same dimensions as the input and in the same units 
       as the input
    
    References
    ----------
    .. Morton et al. 1991 ApJSS 77 119
    
    Examples
    --------
    >>>wv=np.arange(3000,7000,1)
    >>>vac_wv=airtovac(wv)
    array([3000.87467224, 3001.87492143, 3002.87517064, ..., 6998.92971915,
       6999.92998844, 7000.93025774])
    
    """
    x=wv
    # get x to be in Angstroms for calculations if it isn't already
    if ((waveunit!='Angstrom') & (waveunit!='micron')):
        print ('Wave unit ',waveunit,' not recognized, returning Angstroms')
    if (waveunit=='micron'):
        x=wv*1.e4

    tmp=1.e4/x
    y=x*(1.+6.4328e-5+2.94981e-2/(146.-tmp**2)+2.5540e-4/(41.-tmp**2))
    # vacuum wavelengths are indeed slightly longer

    # get y to be in the same units as the input:
    if (waveunit=='micron'):
        y=y*1.e-4

    return(y)
