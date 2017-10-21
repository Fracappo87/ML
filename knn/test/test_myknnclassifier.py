# -*- coding: utf-8 -*-
"""
Created on Fri Oct 13 18:23:38 2017

Author: Francesco Capponi <capponi.francesco87@gmail.com>
     
License: BSD 3 clause
"""


import unittest
import numpy as np
from mock import generate_mock_data as gmd
from ..myknnclassifier import MyKnnClassifier
        
class MyKnnClassifierTest(unittest.TestCase):
    
    def test_myknnclassifier_attributes(self):
        """
        Testing the correcteness of the initialized attributes
        """
        
        # TEST 1: checking parameters initialization using correct parameters
        print("\n testing MyKnnClassifier attributes initialization")        
        my_knn = MyKnnClassifier(method="Euclidean",criterion="flat",n_neighbors=1,parallelize=False)
        self.assertEqual(my_knn.learner_type,"classifier","a) Checking knn attribute 'learner_type'.")
        self.assertEqual(my_knn.learning_type,"instance_based","b) Checking knn attribute 'learning_type'.")
        
        # TEST 2: checking correct exception raising when wrong input parameters are given
        self.assertRaises(ValueError,MyKnnClassifier,criterion=-5)          
        self.assertRaises(ValueError,MyKnnClassifier,criterion="agababu")

    def test_myknnClassifier_predict_with_fit_serial(self):
        
        """
        Testing the serial KNN
    
        TEST 1: knn classifier is probed against several types of scenarios regarding the number of input and output features.
                A certain number of training instances is generated using the function "gmd.generate_spatially_gaussian_arrays_single_pole", through a gaussian process centered around the origin.
                The number of degress of freedom (the dimensionality of the input space) may vary from 1 to N, where N can be set by the user. The same holds for the number of output features.
                Several number of scenarios for the number of input features can be decided, as shown in the following, using proper lists of integers.
                Whichever the scenario will be, "gmd.generate_spatially_gaussian_arrays_single_pole" set the class code of the output feature equal to an integer constant y_val, set by the user.
                Being all the training instances equal in output space, the knn classification must returns a constant series of output class codes, all equal to arrays containing y_val, regardless the chosen criterion for the imputation.
                TEST 1 ascertains that this is actually what happens.

        TEST 2: same philosophy of TEST 1 with a main difference: training instances are clustered around two different poles ([-center, -center,...] and [center, center, ...] in input space) using a gaussian process with a quite narrow variance.
                Each cluster has the same amount of elements : size_train/2            
                Each element belonging to one of the two clusters is assigned a specific output class code ([y_val] for pole 2, [0] for pole1).   
                Given the narrow spread of the two clusters, a properly functioning knn should work in this way:
                    
                    a) k<= size_train/2: k neighbors are all within a specific cluster, hence the result of the classification is the same as TEST 1
                    b) k>= size_train/2: k neighbors encompass members of the other cluster, hence the result of the classification is a number alpha, such that |alpha|=0 or y_val
        
                All this is checked in the second part of the TEST
        """        
        
        print("\n testing predict method using serial knn classifier")
        
        # TEST 1
        size_train=100
        size_test=5
        n_input_feat=[1,3,4]
        n_output_feat=[1]
        y_val=1        
                
        for i in range(1,size_train+1):
                    
            my_knn1 = MyKnnClassifier(method="Euclidean",criterion="flat",n_neighbors=i,parallelize=False)
            my_knn2 = MyKnnClassifier(method="Euclidean",criterion="weighted",n_neighbors=i,parallelize=False)
                    
            for i in n_input_feat:
                for j in n_output_feat:
                    A,C,B=gmd.generate_spatially_gaussian_arrays_single_pole(i,j,size_train,size_test,y_val)
                    my_knn1.fit(A,B)
                    my_knn1.predict(C)
                    
                    my_knn2.fit(A,B)
                    my_knn2.predict(C)
                    test_vec=C[:size_test]
                    
                    self.assertEqual(abs(test_vec-my_knn1.prediction).sum(),0,"a) Checking knn classification with flat criterion.")
                    self.assertEqual(abs(test_vec-my_knn2.prediction).sum(),0,"b) Checking knn classification with weighted criterion.")
        
        
         # TEST 2
        
        size_train=100
        size_test=50
        y_val=2  
        
        for i in range(1,size_train//2+1):
                    
            my_knn1 = MyKnnClassifier(method="Euclidean",criterion="flat",n_neighbors=i,parallelize=False)
            my_knn2 = MyKnnClassifier(method="Euclidean",criterion="weighted",n_neighbors=i,parallelize=False)
                    
            for i in n_input_feat:
                for j in n_output_feat:
                    A,C,B=gmd.generate_spatially_gaussian_arrays_double_pole(i,j,size_train,size_test,y_val,binary=True)
                    my_knn1.fit(A,B)
                    my_knn1.predict(C)
                                        
                    
                    my_knn2.fit(A,B)
                    my_knn2.predict(C)
                    test_vec=C[:size_test//2]
                    test_vec=np.concatenate((test_vec,test_vec-y_val),axis=0)
                        
                    self.assertTrue(abs(test_vec-my_knn1.prediction).sum()<1e-12,"c) Checking knn classification with flat criterion, 1<k<n_train_samples//2.")
                    self.assertTrue(abs(test_vec-my_knn2.prediction).sum()<1e-12,"d) Checking knn classification with weighted criterion, 1<k<n_train_samples//2.")
        
        for i in range(size_train//2+2,size_train+1):
                    
            my_knn1 = MyKnnClassifier(method="Euclidean",criterion="flat",n_neighbors=i,parallelize=False)
            my_knn2 = MyKnnClassifier(method="Euclidean",criterion="weighted",n_neighbors=i,parallelize=False)
                    
            for i in n_input_feat:
                for j in n_output_feat:
                    A,C,B=gmd.generate_spatially_gaussian_arrays_double_pole(i,j,size_train,size_test,y_val,binary=True)
                    my_knn1.fit(A,B)
                    my_knn1.predict(C)
                    
                    my_knn2.fit(A,B)
                    my_knn2.predict(C)
                    
                    for k,l in enumerate(zip(my_knn1.prediction,my_knn2.prediction)):
                        self.assertTrue((l[0] == y_val) or (l[0] == 0),"e) Checking knn classification with flat criterion, n_train_samples//2 < k <= n_train_samples.")
                        self.assertTrue((l[1] == y_val) or (l[1] == 0),"f) Checking knn classification with weighted criterion, n_train_samples//2 <= k < n_train_samples.")
     
    def test_myknnclassifier_predict_with_fit_grid(self):
        
        """
            Testing the 2-d grid KNN
        
            TEST 1a/1b: testing the correctness of grid KNN, using the same approach as in "test_myknnclassifier_predict_with_fit_serial", with few important differences.
                        1) The number of input features is constrained to 2
                        2) In TEST 1a the train and test set are obtained by randomly extracting the points of a 2D grid, using the function "gmd.generate_spatially_gaussian_arrays_single_pole"
                        3) In TEST 1b only two instances belong to the test set X_test: [0,0] and [Nx-1,Ny-1], Nx and Ny being  the 2d grid extensions.
                        These two points belong to two different regions
                        
                        
                          ---------|----(Nx-1,Ny-1)
                          |        |        |
                          |    +   |    -   |
                          |    +   |    -   |
                          |    +   |    -   |
                          |    +   |    -   |
                        (0,0)------|--------|
    
                        where output values of instances are equal in magnitude, but differ by a sign.
                        These data arrays are produced by "gmd.generate_grid_arrays_double_pole".
                        
                        The result of knn classification for these two points should be
                        
                         KNN[(0,0)]_k = yval, 
                         KNN[(Nx-1,Ny-1)]_k = 0, if k<= min(Nx,Ny)/2
        
        """
        
        print("\n testing predict method using grid knn classifier")
        
        
        sizes=[[5,5],[3,18],[10,10]]
        n_output_feat=[1]
        y_val=3        

        size_train=[]
        for i in sizes:
            size_train.append(i[0]*i[1]*3//4)
                
        for a,j in zip(size_train,sizes):
                
            for i in range(1,5):
                extensions=np.array(j)
                my_knn1 = MyKnnClassifier(method="grid",criterion="flat",n_neighbors=i,grid_size=extensions,parallelize=False)
                my_knn2 = MyKnnClassifier(method="grid",criterion="weighted",n_neighbors=i,grid_size=extensions,parallelize=False)
   
                for k in n_output_feat:
                    A,C,B=gmd.generate_grid_arrays_single_pole(j[0],j[1],k,a,y_val)
                    my_knn1.fit(A,B)
                    my_knn1.predict(C)
                        
                    my_knn2.fit(A,B)
                    my_knn2.predict(C)
                    test_vec=C[:(j[0]*j[1]-a)]
                    self.assertTrue(abs(test_vec-my_knn1.prediction).sum()<1e-12,"a) Checking knn grid classification with flat criterion.")
                    self.assertTrue(abs(test_vec-my_knn2.prediction).sum()<1e-12,"b) Checking knn grid classification with weighted criterion.")
        
        sizes=[[5,5],[3,10],[7,10]]
        n_output_feat=[1]
        y_val=3        
        for i in sizes:
            extensions=np.array([i[0]*2,i[1]*2],dtype=int)
            
            for j in [1,min(i)]:
                my_knn1 = MyKnnClassifier(method="grid",criterion="flat",n_neighbors=j,grid_size=extensions,parallelize=False)
                my_knn2 = MyKnnClassifier(method="grid",criterion="weighted",n_neighbors=j,grid_size=extensions,parallelize=False)
        
                for k in n_output_feat:
                    A,C,B=gmd.generate_grid_arrays_double_pole(i[0],i[1],k,y_val,binary=True)
            
                    my_knn1.fit(A,B)
                    my_knn1.predict(C)

                    my_knn2.fit(A,B)
                    my_knn2.predict(C)   
                    test_vec=C[:1]
                    test_vec=np.concatenate((test_vec,y_val-test_vec),axis=0)
                    self.assertTrue(abs(test_vec-my_knn1.prediction).sum()<=1e-12,"c) Checking knn classification with flat criterion, 1<k<=min(Nx,Ny)//2.")
                    self.assertTrue(abs(test_vec-my_knn2.prediction).sum()<=1e-12,"d) Checking knn classification with weighted criterion, 1<k<=min(Nx,Ny)//2.")

        
        #To be added: test for the classification with kdtree
        #test_myknnclassifier_predict_with_fit_kd_tree