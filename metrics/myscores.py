# -*- coding: utf-8 -*-
"""

Author: Francesco Capponi <capponi.francesco87@gmail.com>
        
License: BSD 3 clause

"""

import numpy as np
from checks.mycheck import shape_test

def R_squared(y_pred,y_test):
    """
    Compute the determination coefficient for regression analysis:

    Parameters
    ----------
    y_pred,y_test : numpy-like, shape = [n_test_sample,n_features]
    """
    
    shape_test(y_pred, y_test)

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
    
    shape_test(y_pred, y_test)

    return ((y_pred==y_test).astype(float)).sum()/y_pred.shape[0]
    
def cross_entropy(y_pred, y_test):
    """
    Compute the cross-entropy for binary classifications
    
    Parameters
    ----------
    
    y_pred,y_test : numpy-like, shape = [n_test_sample,n_features]
    """

    shape_test(y_pred, y_test)

    return -1.*np.mean(y_test*np.log(y_pred)+(1-y_test)*np.log(1-y_pred))
