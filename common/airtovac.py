def airtovac(wv, waveunit='Angstrom'):
    """
    Takes an array of wavelengths in air and converts them to vacuum
        using eq. 3 from Morton et al. 1991 ApJSS 77 119

    Returns: 
       An array of the same dimensions as the input and in the same units 
       as the input

    Prameters:
       Input wavelength array wv

    Optional parameters: 
       waveunit: a string variable, 'Angstrom' or 'micron', default is Angstrom

    Examples:
       wv=np.arange(3000,7000,1)
       vac_wv=airtovac(wv)
       
    @author: Nadia Zakamska

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
