# -*- coding: utf-8 -*-
"""
Created on Fri Oct 20 07:21:07 2017

Author: Francesco Capponi <capponi.francesco87@gmail.com>

License: BSD 3 clause
"""

import unittest
import numpy as np
from .. import generate_mock_data as gmd

class MyGenerateMockDataTest(unittest.TestCase):
    
    def test_generate_spatially_gaussian_arrays_single_pole(self):
        
        print("\n testing generation of spatially gaussian arrays, single pole")

        # TEST 1: arbitrary output space dimensions, real output values
        size_train=[10,100,1000]
        size_test=[5,50,500]
        n_input_feat=[1,3,4]
        n_output_feat=[1,3,6]
        y_val=-.6        
                
        for i in n_input_feat:
            for j in n_output_feat:
                for k in size_train:
                    for l in size_test:
                        A,C,B=gmd.generate_spatially_gaussian_arrays_single_pole(i,j,k,l,y_val)
                        self.assertEqual(C.all(),y_val,"a) Checking output numerical value.")
                        self.assertAlmostEqual(A.mean(axis=0).all(),0.,delta=1,msg="b) Checking mean of gaussianly distributed training data.")
                        self.assertAlmostEqual(B.mean(axis=0).all(),0.,delta=1,msg="c) Checking mean of gaussianly distributed test data.")
           
          
    def test_generate_spatially_gaussian_arrays_double_pole(self):
        
        print("\n testing generation of spatially gaussian arrays, double pole")
           
        size_train=[10,100,1000]
        size_test=[5,50,500]
        n_input_feat=[1,3,4]
        n_output_feat=[1,3,6]
        y_val=-.6        
           
         for i in n_input_feat:
                for j in n_output_feat:
                    A,C,B=gmd.generate_spatially_gaussian_arrays_double_pole(i,j,size_train,size_test,y_val)
                     