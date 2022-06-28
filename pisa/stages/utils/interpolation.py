import numpy as np

def logistic_function(a,b,c,x) :
    '''
    Logistic function as defined here: https://en.wikipedia.org/wiki/Logistic_function.
    Starts off slowly rising, before steeply rising, then plateaus.

    Parameters
    ----------
    a : float
        Normalisation (e.g. plateau height) 
    b : float
        Steepness of rise (larger value means steeper rise)
    c : float 
        x value at half-height of curve
    x : array
        The continuous parameter

    Returns
    -------
    f(x) : array
        The results of applying the logistic function to x
    '''
    return a / (1 + np.exp( -b * (x-c) ) )


    