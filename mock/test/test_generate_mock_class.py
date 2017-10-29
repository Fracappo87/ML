# -*- coding: utf-8 -*-
"""
Created on Tue Oct 24 19:09:14 2017

uthor: Francesco Capponi <capponi.francesco87@gmail.com>

License: BSD 3 clause
"""

import unittest
from ..generate_mock_class import MockClassRegressor,MockClassClassifier

class MyGenerateMockClassTest(unittest.TestCase):
    
    def test_mock_regressor_class(self):
        
        print("\n testing attributes and methods of mock regressor class")

        #TEST 1: checking correct attributes initialization        
        A=MockClassRegressor()
        self.assertFalse(A.check_fit,"a) Testing check_fit attribute")
        self.assertEqual(A.count_fit,0,"b) Testing count_fit attribute")
        self.assertFalse(A.check_predict,"c) Testing check_predict attribute")        
        self.assertEqual(A.learning_type,None,"d) Testing learning_type attribute")        
        self.assertEqual(A.count_predict,0,"e) Testing count_predict attribute")
        self.assertEqual(A.learner_type,"regressor","e) Testing learner_type attribute")
        
        #TEST 2: checking "fit" method
        N=3
        for i in range(N):
           A.fit('X','Y')
           
        self.assertTrue(A.check_fit,"f) Testing check_fit attribute after call of fit method")
        self.assertEqual(A.count_fit,N,"g) Testing count_fit attribute after call of fit method")
        A.__setattr__("learning_type","instance_based")
        A.fit('X','Y')
        self.assertEqual(A.count_fit,N+1,"h) Testing count_fit attribute after second call of fit method")
        self.assertEqual(A.prediction,'Y',"i) Testing count_fit attribute after second call of fit method")
        
        #TEST 3: checking "predict" method
        N=3
        for i in range(N):
           A.predict('X')
           
        self.assertTrue(A.check_predict,"l) Testing check_fit attribute after call of fit method")
        self.assertEqual(A.count_predict,N,"m) Testing count_fit attribute after call of fit method")
        A.__setattr__("learning_type","training_based")
        A.predict('X')
        self.assertEqual(A.count_predict,N+1,"n) Testing count_fit attribute after second call of fit method")
        self.assertEqual(A.prediction,'X',"o) Testing count_fit attribute after second call of fit method")

        
    def test_mock_classifier_class(self):
        
        print("\n testing attributes and methods of mock regressor class")

        #TEST 1: checking correct attributes initialization        
        A=MockClassClassifier()
        self.assertFalse(A.check_fit,"a) Testing check_fit attribute")
        self.assertEqual(A.count_fit,0,"b) Testing count_fit attribute")
        self.assertFalse(A.check_predict,"c) Testing check_predict attribute")        
        self.assertEqual(A.learning_type,None,"d) Testing learning_type attribute")        
        self.assertEqual(A.count_predict,0,"e) Testing count_predict attribute")
        self.assertEqual(A.learner_type,"classifier","e) Testing learner_type attribute")
        
        #TEST 2: checking "fit" method
        N=3
        for i in range(N):
           A.fit('X','Y')
           
        self.assertTrue(A.check_fit,"f) Testing check_fit attribute after call of fit method")
        self.assertEqual(A.count_fit,N,"g) Testing count_fit attribute after call of fit method")
        A.__setattr__("learning_type","instance_based")
        A.fit('X','Y')
        self.assertEqual(A.count_fit,N+1,"h) Testing count_fit attribute after second call of fit method")
        self.assertEqual(A.prediction,'Y',"i) Testing count_fit attribute after second call of fit method")
        
        #TEST 3: checking "predict" method
        N=3
        for i in range(N):
           A.predict('X')
           
        self.assertTrue(A.check_predict,"l) Testing check_fit attribute after call of fit method")
        self.assertEqual(A.count_predict,N,"m) Testing count_fit attribute after call of fit method")
        A.__setattr__("learning_type","training_based")
        A.predict('X')
        self.assertEqual(A.count_predict,N+1,"n) Testing count_fit attribute after second call of fit method")
        self.assertEqual(A.prediction,'X',"o) Testing count_fit attribute after second call of fit method")