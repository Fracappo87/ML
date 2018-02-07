# -*- coding: utf-8 -*-
"""
Created on Fri Oct 13 18:23:38 2017

Author: Francesco Capponi <capponi.francesco87@gmail.com>
     
License: BSD 3 clause
"""


import unittest
import numpy as np
import numpy.testing as npt
from ..mylogisticmodel import MyLogisticRegressionClassifier
        
class MyLogisticModelClassifierTest(unittest.TestCase):
    
    def test_init(self):
        """Testing class initialization"""
        
        print("Testing initialization of logistic model class")        
        
        test_dictionaries = [{'n_dimensions': 3, 'optimize': 'GadentDscent', 'start': 'random', 'offset': 0},
                             {'n_dimensions': 3, 'optimize': False, 'start': 'pabolo', 'offset': 0},
                             {'n_dimensions': 3, 'optimize': 'GradientDescent', 'start': 'pabolo', 'offset': 0},
                             {'n_dimensions': 3, 'optimize': 'GradientDescent', 'start': .9, 'offset': 0}]
        for dictionary in test_dictionaries:
            self.assertRaises(ValueError, MyLogisticRegressionClassifier, **dictionary)        
        
        logit_model = MyLogisticRegressionClassifier(n_dimensions=3)
        self.assertEqual(logit_model.n_dimensions,3,'Testing n_dimension attribute')
        self.assertEqual(logit_model.learning_type,'training_based')
        self.assertEqual(logit_model.optimize,'GradientDescent','Testing optimization flag')
        self.assertEqual(logit_model.start,'random','Testing parameters initialization flag')
        self.assertEqual(logit_model.offset,0.,'Testing offset attribute')
        
    def test_initialize(self):
        """Testing weights initialization"""
        
        print("Testing initialization of logistic model weights")
        
        offset = 0.00000345
        test_dictionaries = [{'n_dimensions': 3, 'optimize': 'GradientDescent', 'start': 'random', 'offset': 0},
                             {'n_dimensions': 7, 'optimize': 'GradientDescent', 'start': 'random', 'offset': 0},
                             {'n_dimensions': 13, 'optimize': 'GradientDescent', 'start': 'uniform', 'offset': 0},
                             {'n_dimensions': 30, 'optimize': 'GradientDescent', 'start': 'uniform', 'offset': offset}]
                             
        np.random.seed(1)
        w_check_values = [np.array([[4.170220e-01],[ 7.203245e-01],[ 1.143748e-04]]), np.array([[0.14675589],[ 0.09233859],[ 0.18626021],[ 0.34556073],[ 0.39676747],[ 0.53881673],[0.41919451]]), 0., offset]
        b_check_values = [0.302333, 0.68522, 0., offset]

        for dictionary, w_value, b_value in zip(test_dictionaries, w_check_values, b_check_values):
            logit_model = MyLogisticRegressionClassifier(**dictionary)
            logit_model._initialize()
            npt.assert_array_almost_equal(logit_model.w, w_value, err_msg='Testing weights array initialization')
            npt.assert_array_almost_equal(logit_model.b, b_value, err_msg='Testing bias initialization')