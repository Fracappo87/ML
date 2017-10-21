# -*- coding: utf-8 -*-
"""

Author: Francesco Capponi <capponi.francesco87@gmail.com>

License: BSD 3 clause

"""

from metrics.myscores import R_squared,class_score

class MyRegressor(object):
    """ 
    Base class for regression learners. 
    Attributes
    ---------- 
    
    learner_type = str, type of learner (regressor)
    """

    def __init__(self):
        self.learner_type='regressor'

    def score(self,Y_pred,Y_test):
        """
        Compute the algorithm efficiency score by using the determination coefficient.        

        Parameters:
        ----------

        Y_pred : numpy-like, shape = [n_test_samples, n_output_features]
        Y_test : numpy-like, shape = [n_test_samples, n_output_features]
            
        """
        return R_squared(Y_pred,Y_test)
        
class MyClassifier(object):
    """                                                                                                               \
    Base class for classification learners.                                                                           \
                                                                                                                       
    Attributes                                                                                                        
    ----------                                                                                                        \
                                                                                                                       
    learner_type = str, type of learner (classifier)                                                                  \
    """

    def __init__(self):
        self.learner_type='classifier'
    
    def score(self,Y_pred,Y_test):
        """
        Compute the algorithm efficiency score by using the determination coefficient.        

        Parameters:
        ----------

        Y_pred : numpy-like, shape = [n_test_samples, n_output_features]
        Y_test : numpy-like, shape = [n_test_samples, n_output_features]
            
        """
        return class_score(Y_pred,Y_test)
