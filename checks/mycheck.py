# -*- coding: utf-8 -*-
"""

Author: Francesco Capponi <capponi.francesco87@gmail.com>
        
License: BSD 3 clause

"""


def sanitycheck(X,types):
    """ 
    Sanity check: just to be sure that users are giving the right input parameters
    """
    
    if not isinstance(X, types):
        raise ValueError("Object has to be a ",types)

def shape_test(X, Y):
    """
    Check numpy arrays shapes
    """
    
    if(X.shape != Y.shape):
        raise ValueError("input data arrays must have the same shape")
