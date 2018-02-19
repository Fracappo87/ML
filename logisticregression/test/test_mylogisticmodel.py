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

        test_dictionaries = [{'minibatch_size': 3, 'optimization': 'GadentDscent', 'start': 'random', 'offset': 0},
                             {'minibatch_size': 3, 'optimization': False, 'start': 'pabolo', 'offset': 0},
                             {'minibatch_size': 3, 'optimization': 'adam', 'start': 'pabolo', 'offset': 0},
                             {'minibatch_size': 3, 'optimization': 'adam', 'start': .9, 'offset': 0}]
        for dictionary in test_dictionaries:
            self.assertRaises(ValueError, MyLogisticRegressionClassifier, **dictionary)        
        
        logit_model = MyLogisticRegressionClassifier(minibatch_size=3)
        self.assertEqual(logit_model.minibatch_size,3,'Testing minibatch_size attribute')
        self.assertEqual(logit_model.learning_type,'training_based')
        self.assertEqual(logit_model.optimization,None,'Testing optimization flag')
        self.assertEqual(logit_model.start,'random','Testing parameters initialization flag')
        self.assertEqual(logit_model.offset,0.,'Testing offset attribute')
        self.assertEqual(logit_model.n_iterations, 10, 'Testing n_iterations attribute')
        self.assertEqual(logit_model.learning_rate, .5, 'Testing learning_rate attribute')
        
    def test_initialize(self):
        """Testing weights initialization"""
        
        print("Testing initialization of logistic model weights")
        
        offset = 0.00000345
        test_dimensions = [3, 7]
        np.random.seed(1)
        w_check_values = [np.array([[4.170220e-01, 7.203245e-01, 1.143748e-04]]), np.array([[0.14675589, 0.09233859, 0.18626021, 0.34556073, 0.39676747, 0.53881673,0.41919451]])]
        b_check_values = [0.302333, 0.68522]

        for dimension, w_value, b_value in zip(test_dimensions, w_check_values, b_check_values):
            logit_model = MyLogisticRegressionClassifier()
            logit_model._initialize(dimension)
            self.assertSequenceEqual((1, dimension), logit_model.W.shape, 'Testing shape of weights array for random initialization')
            self.assertSequenceEqual((1, 1), logit_model.b.shape, 'Testing shape of bias array for random initialization')
            npt.assert_array_almost_equal(logit_model.W, w_value, err_msg='Testing weights array random initialization')
            npt.assert_array_almost_equal(logit_model.b, b_value, err_msg='Testing bias random initialization')
            
        logit_model = MyLogisticRegressionClassifier(start = 'uniform', offset = offset)
        logit_model._initialize(30)
        self.assertSequenceEqual((1, 30), logit_model.W.shape, 'Testing shape of weights array for uniform initialization')
        self.assertSequenceEqual((1, 1), logit_model.b.shape, 'Testing shape of bias array for uniform initialization')

        npt.assert_array_almost_equal(logit_model.W, offset, err_msg='Testing weights array uniform initialization')
        npt.assert_array_almost_equal(logit_model.b, offset, err_msg='Testing bias uniform initialization')
        
    def test_forward_prop(self):
        """Testing forward propagation"""
        
        print("Testing forward propagation")
        
        logit_model = MyLogisticRegressionClassifier(offset = 1., start = 'uniform')
        X_trains = [np.array([[.5, -.4, .7, -.1]]), np.array([[0.5, 1.],[-0.5, 0.4]]), np.array([[0.5, 0., 0.],[0.5, 0., 0.],[0.5, 0., 0.]])]
        Y_trains = [np.array([[0]]), np.array([[0],[1]]), np.array([[1],[1],[1]])]
        expected_activations = [np.array([[ 0.8455347]]), np.array([[0.9241418, 0.7109495]]), np.array([[0.8175744, 0.8175744, 0.8175744]])]
        expected_costs = [1.8677858, 1.4600217, 0.2014133]

        for X_train, Y_train, expected_activation, expected_cost in zip(X_trains, Y_trains, expected_activations, expected_costs):
            logit_model._initialize(X_train.shape[1])
            activation, cost = logit_model._forward_prop(X_train, Y_train)  
            np.testing.assert_array_almost_equal(activation, expected_activation, err_msg = 'Testing activation function values')
            np.testing.assert_array_almost_equal(cost, expected_cost, err_msg = 'Testing computation of the cost function')
            
    def test_backward_prop(self):
        """Testing backward propagation"""
        
        print("Testing backward propagation")
        
        # Trivial case, learning rate set to zero
        logit_model = MyLogisticRegressionClassifier(offset = 1., start = 'uniform', learning_rate=0.)
        X_train = np.array([[.5, -.4, .7, -.1]])
        Y_train = np.array([[0]])
        
        logit_model._initialize(X_train.shape[1])
        activation, cost = logit_model._forward_prop(X_train, Y_train)
        logit_model._back_prop(activation, X_train, Y_train)
        np.testing.assert_array_equal(logit_model.W, 1., "Testing weights array after backprop, zero learning rate")
        np.testing.assert_array_equal(logit_model.b, 1., "Testing bias array after backprop, zero learning rate")
        
        # Non trivial case, learning rate = 0.5, as default
        logit_model = MyLogisticRegressionClassifier(offset = 1., start = 'uniform')  
        X_trains = [np.array([[.5, -.4, .7, -.1]]), np.array([[0.5, 1.],[-0.5, 0.4]]), np.array([[0.5, 0., 0.],[0.5, 0., 0.],[0.5, 0., 0.]])]
        Y_trains = [np.array([[0]]), np.array([[0],[1]]), np.array([[1],[1],[1]])]
