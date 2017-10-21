# -*- coding: utf-8 -*-
"""

Author: Francesco Capponi <capponi.francesco87@gmail.com>
        
License: BSD 3 clause

"""


import numpy as np

def R_squared(y_pred,y_test):
    """
    Compute the determination coefficient for regression analysis:

    Parameters
    ----------
    y_pred,y_test : numpy-like, shape = [n_test_sample,n_features]
    """
    
    if(y_pred.shape != y_test.shape):
        raise ValueError("prediction and validation data arrays must have the same shape")
    rat1=np.power(y_test-y_pred,2).sum()
    avg=np.mean(y_test,axis=0)
    rat2=np.power(y_test-avg,2).sum()
    return 1-rat1/rat2
       
def class_score(y_pred,y_test):
    """
    Compute the number of correctly predicted class codes:

    Parameters
    ----------
    y_pred,y_test : numpy-like, shape = [n_test_sample,n_features]
    """
    
    if(y_pred.shape != y_test.shape):
        raise ValueError("prediction and validation data arrays must have the same shape")
    return (y_pred==y_test).sum()/y_pred.shape[0]