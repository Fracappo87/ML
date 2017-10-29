"""
Author: Francesco Capponi <capponi.francesco87@gmail.com>
        David Preti       <preti.david@gmail.com>

License: BSD 3 clause
"""

import unittest
import numpy as np
import numpy.testing as npt
from mock import generate_mock_data as gmd
from ..myknnregressor import MyKnnRegressor

class MyKnnRegressorTest(unittest.TestCase):
    
    def test_myknnregressor_attributes(self):
        """
        Testing the correcteness of the initialized attributes
        """
        
        # TEST 1: checking parameters initialization using correct parameters
        print("\n testing MyKnnRegressor attributes initialization")        
        my_knn = MyKnnRegressor(method="classic",criterion="flat",n_neighbors=1,parallelize=False)
        self.assertEqual(my_knn.learner_type,"regressor","a) Checking knn attribute 'learner_type'.")
        self.assertEqual(my_knn.learning_type,"instance_based","b) Checking knn attribute 'learning_type'.")
        
        # TEST 2: checking correct exception raising when wrong input parameters are given
        self.assertRaises(ValueError,MyKnnRegressor,criterion=-5)          
        self.assertRaises(ValueError,MyKnnRegressor,criterion="agababu")

        
    def test_myknnregressor_predict_with_fit_serial(self):
        
        """
    
        Testing the serial KNN
    
        TEST 1: knn regressor is probed against several types of scenarios regarding the number of input and output features.
                A certain number of training instances is generated using the function "generate_spatially_gaussian_arrays_single_pole", through a gaussian process centered around the origin.
                The number of degress of freedom (the dimensionality of the input space) may vary from 1 to N, where N can be set by the user. The same holds for the number of output features.
                Several number of scenarios for the number of input/output features can be decided, as shown in the following, using proper lists of integers.
                Whichever the scenario will be, "generate_spatially_gaussian_arrays_single_pole" set the value of the output features equal to a constant y_val, set by the user.
                Being all the training instances equal in output space, the knn regression must returns a constant series of output values, all equal to arrays containing y_val, regardless the chosen criterion for the imputation.
                TEST 1 ascertains that this is actually what happens.

        TEST 2: same philosophy of TEST 1 with a main difference: training instances are clustered around two different poles ([-center, -center,...] and [center, center, ...] in input space) using a gaussian process with a quite narrow variance.
                Each cluster has the same amount of elements : size_train/2            
                Each element belonging to one of the two clusters is assigned a specific output value ([-y_val,-y_val,...] for pole 2, [y_val,y_val,...] for pole1).   
                Given the narrow spread of the two clusters, a properly functioning knn should work in this way:
                    
                    a) k<= size_train/2: k neighbors are all within a specific cluster, hence the result of the regression is the same as TEST 1
                    b) k>= size_train/2: k neighbors encompass members of the other cluster, hence the result of the regression is a number alpha, such that |alpha|<|y_val|
        
                All this is checked in the second part of the TEST
        """        
        
        print("\n testing predict method using serial knn regressor")
        
        # TEST 1
        size_train=100
        size_test=5
        n_input_feat=[1,3,4]
        n_output_feat=[1,3,6]
        y_val=-.6        
                
        for i in range(1,size_train+1):
                    
            my_knn1 = MyKnnRegressor(method="classic",criterion="flat",n_neighbors=i,parallelize=False)
            my_knn2 = MyKnnRegressor(method="classic",criterion="weighted",n_neighbors=i,parallelize=False)
                    
            for i in n_input_feat:
                for j in n_output_feat:
                    A,C,B=gmd.generate_spatially_gaussian_arrays_single_pole(i,j,size_train,size_test,y_val)
                    my_knn1.fit(A,B)
                    my_knn1.predict(C)
                    
                    my_knn2.fit(A,B)
                    my_knn2.predict(C)
                    test_vec=C[:size_test]
                    npt.assert_array_almost_equal(test_vec,my_knn1.prediction,decimal=10,err_msg="a) Checking knn regression with flat criterion.")
                    npt.assert_array_almost_equal(test_vec,my_knn2.prediction,decimal=10,err_msg="b) Checking knn regression with weighted criterion.")
        
        
         # TEST 2
        
        size_train=100
        size_test=50
        y_val=-.6  
        
        for i in range(1,size_train//2+1):
                    
            my_knn1 = MyKnnRegressor(method="classic",criterion="flat",n_neighbors=i,parallelize=False)
            my_knn2 = MyKnnRegressor(method="classic",criterion="weighted",n_neighbors=i,parallelize=False)
                    
            for i in n_input_feat:
                for j in n_output_feat:
                    A,C,B=gmd.generate_spatially_gaussian_arrays_double_pole(i,j,size_train,size_test,y_val)
                    my_knn1.fit(A,B)
                    my_knn1.predict(C)
                                        
                    
                    my_knn2.fit(A,B)
                    my_knn2.predict(C)
                    test_vec=C[:size_test//2]
                    test_vec=np.concatenate((test_vec,-test_vec),axis=0)
                        
                    npt.assert_array_almost_equal(test_vec,my_knn1.prediction,decimal=10,err_msg="c) Checking knn regression with flat criterion, 1<k<n_train_samples//2.")
                    npt.assert_array_almost_equal(test_vec,my_knn2.prediction,decimal=10,err_msg="d) Checking knn regression with weighted criterion, 1<k<n_train_samples//2.")
        
        for i in range(size_train//2+2,size_train+1):
                    
            my_knn1 = MyKnnRegressor(method="classic",criterion="flat",n_neighbors=i,parallelize=False)
            my_knn2 = MyKnnRegressor(method="classic",criterion="weighted",n_neighbors=i,parallelize=False)
                    
            for i in n_input_feat:
                for j in n_output_feat:
                    A,C,B=gmd.generate_spatially_gaussian_arrays_double_pole(i,j,size_train,size_test,y_val)
                    my_knn1.fit(A,B)
                    my_knn1.predict(C)
                    
                    my_knn2.fit(A,B)
                    my_knn2.predict(C)
                    
                    npt.assert_array_less(abs(my_knn1.prediction),abs(y_val),err_msg="e) Checking knn regression with flat criterion, n_train_samples//2 < k <= n_train_samples.")
                    npt.assert_array_less(abs(my_knn2.prediction),abs(y_val),err_msg="f) Checking knn regression with weighted criterion, n_train_samples//2 <= k < n_train_samples.")
        
        
    def test_myknnregressor_predict_with_fit_grid(self):
        
        """
        Testing the 2-d grid KNN
    
        TEST 1a/1b: testing the correctness of grid KNN, using the same approach as in "test_myknnregressor_predict_with_fit_serial", with few important differences.
                    1) The number of input features is constrained to 2
                    2) In TEST 1a the train and test set are obtained by randomly extracting the points of a 2D grid, using the function "generate_spatially_gaussian_arrays_single_pole"
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
                    These data arrays are produced by "generate_spatially_gaussian_arrays_double_pole".
                    
                    The result of knn regression for these two points should be
                    
                    a) KNN[(0,0)]_k = +val, if k<= min(Nx,Ny)^2/4
                       KNN[(Nx-1,Ny-1)]_k = -val, if k<= min(Nx,Ny)^2/4
    
                        where min(Nx,Ny)=Nx if Nx<Ny, Ny otherwise
                        
                    b) |KNN[(0,0)]_k| <val, if   M <= k<= Nx*Ny-2, with large enough M
                    Here, a less rigorous approach has to be adopted, since the prediction outcome
                    strictly depends on the linear dimensions of the system and how the different shells 
                    of neighbors get created.
                       
        """
        
        print("\n testing predict method using grid knn regressor")
        
        
        sizes=[[5,5],[3,18],[10,10]]
        n_output_feat=[1,3,6]
        y_val=-.6        

        size_train=[]
        for i in sizes:
            size_train.append(i[0]*i[1]*3//4)
                
        for a,j in zip(size_train,sizes):
                
            for i in range(1,5):
                extensions=np.array(j)
                my_knn1 = MyKnnRegressor(method="grid",criterion="flat",n_neighbors=i,grid_size=extensions,parallelize=False)
                my_knn2 = MyKnnRegressor(method="grid",criterion="weighted",n_neighbors=i,grid_size=extensions,parallelize=False)
   
                for k in n_output_feat:
                    A,C,B=gmd.generate_grid_arrays_single_pole(j[0],j[1],k,a,y_val)
                    my_knn1.fit(A,B)
                    my_knn1.predict(C)
                        
                    my_knn2.fit(A,B)
                    my_knn2.predict(C)
                    test_vec=C[:(j[0]*j[1]-a)]
                    npt.assert_array_almost_equal(test_vec,my_knn1.prediction,decimal=10,err_msg="a) Checking knn grid regression with flat criterion.")
                    npt.assert_array_almost_equal(test_vec,my_knn2.prediction,decimal=10,err_msg="b) Checking knn grid regression with weighted criterion.")
        
        sizes=[[3,2],[5,4],[10,3]]
        n_output_feat=[1,3,6]
        y_val=-.6        
        for i in sizes:
            extensions=np.array([i[0]*2,i[1]*2],dtype=int)
            
            for j in [1,int(min(extensions)//2*min(extensions)//2)]:
                my_knn1 = MyKnnRegressor(method="grid",criterion="flat",n_neighbors=j,grid_size=extensions,parallelize=False)
                my_knn2 = MyKnnRegressor(method="grid",criterion="weighted",n_neighbors=j,grid_size=extensions,parallelize=False)
        
                for k in n_output_feat:
                    A,C,B=gmd.generate_grid_arrays_double_pole(i[0],i[1],k,y_val)
                    
                    my_knn1.fit(A,B)
                    my_knn1.predict(C)
                    my_knn2.fit(A,B)
                    my_knn2.predict(C) 
                    npt.assert_array_almost_equal(abs(my_knn1.prediction),abs(y_val),decimal=10,err_msg="c) Checking knn regression with flat criterion, 1<k<=min(Nx,Ny)//2.")
                    npt.assert_array_almost_equal(abs(my_knn2.prediction),abs(y_val),decimal=10,err_msg="d) Checking knn regression with weighted criterion, 1<k<=min(Nx,Ny)//2.")
           
            for j in [int(extensions[0]*extensions[1]-5),int(extensions[0]*extensions[1]-1)]:
                
                my_knn1 = MyKnnRegressor(method="grid",criterion="flat",n_neighbors=j,grid_size=extensions,parallelize=False)
                my_knn2 = MyKnnRegressor(method="grid",criterion="weighted",n_neighbors=j,grid_size=extensions,parallelize=False)
        
                for k in n_output_feat:
                    A,C,B=gmd.generate_grid_arrays_double_pole(i[0],i[1],k,y_val)
            
                    my_knn1.fit(A,B)
                    my_knn1.predict(C)
                    my_knn2.fit(A,B)
                    my_knn2.predict(C)   
                    
                    npt.assert_array_less(abs(my_knn1.prediction),abs(y_val),err_msg="e) Checking knn regression with flat criterion, min(Nx,Ny)//2 < k <= Nx*Ny-2.")
                    npt.assert_array_less(abs(my_knn2.prediction),abs(y_val),err_msg="f) Checking knn regression with weighted criterion, M < k <= Nx*Ny-2.")
    
if __name__ == '__main__':
    unittest.main()

    #To be added: test for the regression with kdtree
        #test_myknnregressor_predict_with_fit_kd_tree
        