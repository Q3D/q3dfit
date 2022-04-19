#For images, there is a ton of stuff available...
#https://stackoverflow.com/questions/48121916/numpy-resize-rescale-image

def rebin(a,newdims,method='neighbour',centre=False,minusone=False):
    """
    Author: Nadia L. Zakamska, July 2020
    Inspired by https://github.com/mortcanty/CRCPython/blob/master/src/build/lib/auxil/congrid.py,
    but reworked to work on Python 3x

    Parameters
    ----------
    a : np.array, a multi-dimensional numpy array

    newdims : a tuple with the shape of the new array, must have the same 
        dimensionality as the original array a.
        If it doesn't, the code returns the original array. 

    method:
        neighbour - closest value from original data
        nearest and linear - uses n x 1-D interpolations using
                             scipy.interpolate.interp1d
        (see Numerical Recipes for validity of use of n 1-D interpolations)
        spline - uses ndimage.map_coordinates

    centre:
        True - interpolation points are at the centres of the bins
        False - points are at the front edge of the bin

    minusone:
        For example- inarray.shape = (i,j) & new dimensions = (x,y)
        False - inarray is resampled by factors of (i/x) * (j/y)
        True - inarray is resampled by(i-1)/(x-1) * (j-1)/(y-1)
        This prevents extrapolation one element beyond bounds of input array.

    Returns
    -------
    np.array with the newdims shape
    
    Examples:
        1.
        a=np.arange(10)
        newa=rebin(a,(100,))
        2.
        a=np.array([[1.,2.,3.],[2.,3.,4.]])
        newa1=rebin(a,(4,6))
        newa2=rebin(a,(2,3))
        3. 
        np.random.seed(seed=200)
        a=np.random.random((2,3,4))
        newa1=rebin(a,(2,3,4))
        newa2=rebin(a,(2,3,8))
        

    """
    import numpy as np
    if not a.dtype in [np.float64, np.float32]:
        a = np.cast[float](a)
    
    # I'll keep these offsets as they are
    m1 = np.cast[int](minusone) * 1.0
    ofs = np.cast[int](centre) * 0.5
    
    olddims=np.shape(a)
    nold=len(olddims)
    if (nold>5):
        print('not yet implemented for arrays 6D and up, returning the original')
        return(a)
    
    nnew=len(newdims) 
    if (nold!=nnew): 
        print("wrong output shape, returning original array")
        return(a)
            
    if method == 'neighbour':
        indold=np.indices(olddims)
        indnew=np.indices(newdims)
        for i,od in enumerate(olddims):
            indnew[i]=(indnew[i]+ofs)*(od*1.0-m1)/(newdims[i]*1.0-m1)-ofs
        indnew=np.array(indnew,dtype=int)
#        for i,od in enumerate(olddims):
#            if (i==0): ind=indnew[0]
#            else: ind=ind,indnew[i]
#        newa = a[ind]
        if (nold==1): newa=a[indnew[0]]
        if (nold==2): newa=a[indnew[0],indnew[1]]
        if (nold==3): newa=a[indnew[0],indnew[1],indnew[2]]
        if (nold==4): newa=a[indnew[0],indnew[1],indnew[2],indnew[3]]
        if (nold==5): newa=a[indnew[0],indnew[1],indnew[2],indnew[3],indnew[4]]
        
        return(newa)
    
    else:
        print('Other methods not yet implemented!')
        return(a)
    
