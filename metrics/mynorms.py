# -*- coding: utf-8 -*-
"""

Author: Francesco Capponi <capponi.francesco87@gmail.com>
        
License: BSD 3 clause

"""

import numpy as np
from checks.mycheck import sanitycheck


def Euclidean(X):
        
    """
    Compute a collection of euclidean norms for a ndarray
    Parameters
    ----------
    X : numpy-like, shape = [n_data,n_features]
    """
    
    sanitycheck(X,np.ndarray)
    return np.sqrt(np.power(X, 2).sum(axis=1))



def L1(X):
    
    """
    Compute a collection of L1 norms for a ndarray
    Parameters
    ----------
    X : numpy-like, shape = [n_data,n_features]
    """
    
    sanitycheck(X,np.ndarray)
    return np.sqrt(np.power(X, 2)).sum(axis=1)
