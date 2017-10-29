# -*- coding: utf-8 -*-
"""
Created on Fri Oct 20 07:21:07 2017

Author: Francesco Capponi <capponi.francesco87@gmail.com>

License: BSD 3 clause
"""

import unittest
import numpy as np
import numpy.testing as npt
from .. import generate_mock_data as gmd

class MyGenerateMockDataTest(unittest.TestCase):
    
    def test_generate_spatially_gaussian_arrays_single_pole(self):
        
        print("\n testing generation of spatially gaussian arrays, single pole")

        # TEST 1: arbitrary output space dimensions, real output values
        size_train=[20,100,1000]
        size_test=[10,50,500]
        n_input_feat=[1,3,4]
        n_output_feat=[1,3,6]
        y_val=-.6        
                
        for i in n_input_feat:
            for j in n_output_feat:
                for k in size_train:
                    for l in size_test:
                        A,C,B=gmd.generate_spatially_gaussian_arrays_single_pole(i,j,k,l,y_val)
                        npt.assert_allclose(C,y_val,rtol=0.,err_msg="a) Checking output numerical value.")
                        npt.assert_allclose(A.mean(axis=0)+1,1.,rtol=1.5,err_msg="b) Checking mean of gaussianly distributed training data.")
                        npt.assert_allclose(B.mean(axis=0)+1,1.,rtol=1.5,err_msg="c) Checking mean of gaussianly distributed test data.")
                        
                        
    def test_generate_spatially_gaussian_arrays_double_pole(self):
        
        print("\n testing generation of spatially gaussian arrays, double pole")
           
        size_train=[10,100,1000]
        size_test=[4,50,500]
        n_input_feat=[1,3,4]
        n_output_feat=[1,3,6]
        centres=np.array([10,100,1000])
        scales=.001*centres
        y_val=-.6        
           
        for i in n_input_feat:
            for j in n_output_feat:
                for k in size_train:
                    half_train=k//2
                    for l in size_test:
                        half_test=l//2
                        for m,n in zip(centres,scales):
                            # TEST 1: arbitrary output space dimensions, real output values, binary=False
                            A,C,B=gmd.generate_spatially_gaussian_arrays_double_pole(i,j,k,l,y_val,m,n)
                            npt.assert_allclose(C[:half_train],y_val,rtol=0.,err_msg="a) Checking output numerical value, first pole.")
                            npt.assert_allclose(C[half_train:],-y_val,rtol=0.,err_msg="b) Checking output numerical value, first pole.")                               
                            npt.assert_allclose(A[:half_train].mean(axis=0),m,rtol=n,err_msg="c) Checking mean of gaussianly distributed training data, first pole.")
                            npt.assert_allclose(A[half_train:].mean(axis=0),-m,rtol=n,err_msg="d) Checking mean of gaussianly distributed training data, second pole.")
                            npt.assert_allclose(B[:half_test].mean(axis=0),m,rtol=n,err_msg="e) Checking mean of gaussianly distributed test data, first pole.")
                            npt.assert_allclose(B[half_test:].mean(axis=0),-m,rtol=n,err_msg="f) Checking mean of gaussianly distributed test data, second pole.")
                        
                             # TEST 2: arbitrary output space dimensions, real output values, binary=True
                            A,C,B=gmd.generate_spatially_gaussian_arrays_double_pole(i,j,k,l,y_val,m,n,binary=True)
                            self.assertEqual(C.shape[1],1,"g) Checking correct transformation of output space dimension for class codes")
                            npt.assert_allclose(C[:half_train],y_val,rtol=0.,err_msg="h) Checking output numerical value, first pole, binary=True.")
                            npt.assert_allclose(C[half_train:],0,rtol=0.,err_msg="i) Checking output numerical value, first pole, binary=True.")                               
              
    def test_generate_grid_arrays_single_pole(self):
        
        print("\n testing generation of grid arrays, single pole")
        
        # TEST 1: checking correct error raising
        self.assertRaises(ValueError,gmd.generate_grid_arrays_single_pole,X_len=1,Y_len=1,n_output_features=1,size_train=2,y_val=0)        
        size_train=[10,100,1000]
        XY=[[4,4],[10,20],[500,20]]
        n_output_feat=[1,3,6]
        y_val=-.6
        
        for i,j in zip(size_train,XY):
            for k in n_output_feat:
                # TEST 2: arbitrary output space dimensions, real output values
                A,C,B=gmd.generate_grid_arrays_single_pole(j[0],j[1],k,i,y_val)
                npt.assert_allclose(C,y_val,rtol=0.,err_msg="a) Checking output numerical value.")
                self.assertLessEqual(A[:,0].max(),j[0]-1,"b) Checking correct generation of training data")
                self.assertLessEqual(A[:,1].max(),j[1]-1,"c) Checking correct generation of training data")
                self.assertGreaterEqual(A[:,0].min(),0,"d) Checking correct generation of training data")
                self.assertGreaterEqual(A[:,1].min(),0,"e) Checking correct generation of training data")
                self.assertLessEqual(B[:,0].max(),j[0]-1,"f) Checking correct generation of test data")
                self.assertLessEqual(B[:,1].max(),j[1]-1,"g) Checking correct generation of test data")
                self.assertGreaterEqual(B[:,0].min(),0,"h) Checking correct generation of test data")
                self.assertGreaterEqual(B[:,1].min(),0,"i) Checking correct generation of test data")
    
    def test_generate_grid_arrays_double_pole(self):
        
        print("\n testing generation of grid arrays, double pole")
        
        XY=[[4,4],[10,20],[500,20]]
        n_output_feat=[1,3,6]
        y_val=-.6
        
        for i,j in zip(n_output_feat,XY):
                # TEST 1: arbitrary output space dimensions, real output values, binary=False
                A,C,B=gmd.generate_grid_arrays_double_pole(j[0],j[1],i,y_val)
                npt.assert_allclose(C[:C.shape[0]//2],y_val,rtol=0.,err_msg="a) Checking output numerical value, first pole.")
                npt.assert_allclose(C[C.shape[0]//2:],-y_val,rtol=0.,err_msg="b) Checking output numerical value, second pole.")
                self.assertEqual(A[:,0].max(),2*j[0]-1,"c) Checking correct generation of training data")
                self.assertEqual(A[:,1].max(),2*j[1]-1,"d) Checking correct generation of training data")
                self.assertEqual(A[:,0].min(),0,"e) Checking correct generation of training data")
                self.assertEqual(A[:,1].min(),0,"f) Checking correct generation of training data")
                self.assertEqual(B[:,0].max(),2*j[0]-1,"g) Checking correct generation of training data")
                self.assertEqual(B[:,1].max(),2*j[1]-1,"h) Checking correct generation of training data")
                self.assertEqual(B[:,0].min(),0,"i) Checking correct generation of training data")
                self.assertEqual(B[:,1].min(),0,"l) Checking correct generation of training data")
               
                # TEST 2: arbitrary output space dimensions, real output values, binary=True
                A,C,B=gmd.generate_grid_arrays_double_pole(j[0],j[1],i,y_val,binary=True)
                self.assertEqual(C.shape[1],1,"m) Checking correct transformation of output space dimension for class codes")
                npt.assert_allclose(C[:C.shape[0]//2],y_val,rtol=0.,err_msg="n) Checking output numerical value, first pole.")
                npt.assert_allclose(C[C.shape[0]//2:],0,rtol=0.,err_msg="o) Checking output numerical value, second pole.")
           
    def test_generate_clustered_data(self):
        
        print("\n testing generation of clustered data around four poles")
        
         # TEST 1: checking correct error raising
        self.assertRaises(ValueError,gmd.generate_clustered_data,ndata_per_cluster=10,N=10,centres_value=[1,2,3])
        A,B,C,D=gmd.generate_clustered_data(ndata_per_cluster=100,N=10,centres_value=[1,2,3,4])
        self.assertEqual(A.shape[0],4*75,"a) Checking correct proportion of training input data")
        self.assertEqual(A.shape[1],2,"b) Checking correct dimension of training input space")
        self.assertEqual(B.shape[0],4*75,"c) Checking correct proportion of training output data")
        self.assertEqual(B.shape[1],1,"d) Checking correct dimension of training output space")
        self.assertEqual(C.shape[0],4*25,"e) Checking correct proportion of test input data")
        self.assertEqual(C.shape[1],2,"f) Checking correct dimension of test input space")
        self.assertEqual(D.shape[0],4*25,"g) Checking correct proportion of test output data")
        self.assertEqual(D.shape[1],1,"h) Checking correct dimension of test output space")