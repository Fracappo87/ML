# -*- coding: utf-8 -*-
"""

Author: Francesco Capponi <capponi.francesco87@gmail.com>

License: BSD 3 clause

"""

class MyRegressor(object):
    """ 
    Base class for regression learners. 
    Attributes
    ---------- 
    
    learner_type = str, type of learner (regressor)
    """

    def __init__(self):
        self.learner_type='regressor'


class MyClassifier(object):
    """                                                                                                               \
    Base class for classification learners.                                                                           \
                                                                                                                       
    Attributes                                                                                                        
    ----------                                                                                                        \
                                                                                                                       
    learner_type = str, type of learner (classifier)                                                                  \
    """

    def __init__(self):
        self.learner_type='classifier'
