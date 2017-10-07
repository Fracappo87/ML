"""

Author: Francesco Capponi <capponi.francesco87@gmail.com>
        David Preti       <preti.david@gmail.com>

License: BSD 3 clause

"""

import unittest
import numpy as np
from ..myknnregressor import MyKnnRegressor



"""    
    Test arrays
    
    X_train_1 = array-like, shape = [size_train,n_input_features]
    Y_train_1 = array-like, shape = [size_train,n_output_features]
    X_test_1 = array-like, shape = [size_test,n_input_features]
    
    Y_train_1 has all elements set equal to a given user value, as shown in create_mock_arrays_1.
"""



def create_mock_arrays_1(n_input_features,n_output_features,size_train,size_test,y_val):
        
        centre=0.
        X_test=np.random.normal(centre,size=(size_test,n_input_features))
        X_train=np.random.normal(centre,size=(size_train,n_input_features))
        Y_train=np.array([y_val]*size_train*n_output_features).reshape(size_train,n_output_features)
        
        return X_train, Y_train, X_test



def create_mock_arrays_2(n_input_features,n_output_features,size_train,size_test,y_val):
        
        centre=100.
        sizo=size_test//2
        pole_1 = np.random.normal(centre,scale=.001,size=(sizo,n_input_features))
        pole_2 = np.random.normal(-centre,scale=.001,size=(sizo,n_input_features))
        X_test=np.concatenate((pole_1,pole_2))
        
        sizo=size_train//2
        pole_1 = np.random.normal(centre,scale=.001,size=(sizo,n_input_features))
        pole_2 = np.random.normal(-centre,scale=.001,size=(sizo,n_input_features))
        X_train=np.concatenate((pole_1,pole_2))
        
        y_pole_1=np.array([y_val]*sizo*n_output_features).reshape(sizo,n_output_features)
        y_pol_2=-1*y_pole_1
        Y_train = np.concatenate((y_pole_1,y_pol_2))
        return X_train, Y_train, X_test



""" 
    
    Test arrays: crated using a 2-d grid, whose extensions are given as input by the user
    
    X_train = array-like, shape = [size_train,2]
    Y_train = array-like, shape = [size_train,n_output_features]
    X_test = array-like, shape = [grid_size-size_train,size_test,2]
    
    Y_train has all elements set equal to a given user value, as shown in create_mock_arrays_1.

"""



def create_mock_grid_1(X_len,Y_len,n_output_features,size_train,y_val):
                
        a=X_len
        b=Y_len
        if size_train > a*b:
            raise ValueError("size_train can be at most equal to X_len*Y_len.")
            
        perc=a*b-size_train
        grid=np.zeros([a*b,2],dtype=int)

        sequence=[(x,y) for y in range(b) for x in range(a)]
        k=0
        for i,j in sequence:
            grid[k]=np.array([i,j])
            k+=1
            
        rand_idx=np.random.choice(np.arange(a*b),perc,replace=False)
        X_test=grid[rand_idx]
        X_train=grid[[x for x in range(a*b) if x not in rand_idx]]
        Y_train=np.array([y_val]*size_train*n_output_features).reshape(size_train,n_output_features)
        
        return X_train, Y_train, X_test
        
        

def create_mock_grid_2(X_len,Y_len,n_output_features,y_val):
                
        a=X_len
        b=Y_len
        
        grid=np.zeros([4*a*b,2],dtype=int)

        sequence=[(x,y) for y in range(2*b) for x in range(2*a)]
        k=0
        for i,j in sequence:
            grid[k]=np.array([i,j])
            k+=1
        idx=[0,4*a*b-1]
        X_test=grid[idx]
        X_train=grid[[x for x in range(1,4*a*b-1)]]
        
        y_pole_1=np.array([y_val]*(len(X_train)//2)*n_output_features).reshape((len(X_train)//2),n_output_features)
        y_pol_2=-1*y_pole_1
        Y_train = np.concatenate((y_pole_1,y_pol_2))
        
        return X_train, Y_train, X_test 
        
        
        
class MyKnnTest(unittest.TestCase):
    
    def test_myknnregressor_attributes(self):
        """
            Testing the correcteness of the initialized attributes
        """
        
        # TEST 1: checking parameters initialization using correct parameters
        print("\n testing attributes initialization")        
        my_knn = MyKnnRegressor(distance="Euclidean",criterion="flat",n_neighbors=1,parallelize=False)
        self.assertEqual(my_knn.learner_type,"regressor","a) Checking knn attribute 'learner_type'.")
        self.assertEqual(my_knn.learning_type,"instance_based","b) Checking knn attribute 'learning_type'.")
        
        # TEST 2: checking correct exception raising when wrong input parameters are given
        self.assertRaises(ValueError,MyKnnRegressor,criterion=-5)          
        self.assertRaises(ValueError,MyKnnRegressor,criterion="agababu")

        
    def test_myknnregressor_predict_with_fit_serial(self):
        
        """
        
            Testing the serial KNN
        
            TEST 1: knn regressor is probed against several types of scenarios regarding the number of input and output features.
                    A certain number of training instances is generated using the function "create_mock_arrays_1", through a gaussian process centered around the origin.
                    The number of degress of freedom (the dimensionality of the input space) may vary from 1 to N, where N can be set by the user. The same holds for the number of output features.
                    Several number of scenarios for the number of input/output features can be decided, as shown in the following, using proper lists of integers.
                    Whichever the scenario will be, "create_mock_arrays_1" set the value of the output features equal to a constant y_val, set by the user.
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
        
        print("\n testing predict method using serial knn")
        
        # TEST 1
        size_train=100
        size_test=5
        n_input_feat=[1,3,4]
        n_output_feat=[1,3,6]
        y_val=-.6        
                
        for i in range(1,size_train+1):
                    
            my_knn1 = MyKnnRegressor(distance="Euclidean",criterion="flat",n_neighbors=i,parallelize=False)
            my_knn2 = MyKnnRegressor(distance="Euclidean",criterion="weighted",n_neighbors=i,parallelize=False)
                    
            for i in n_input_feat:
                for j in n_output_feat:
                    A,C,B=create_mock_arrays_1(i,j,size_train,size_test,y_val)
                    my_knn1.fit(A,B)
                    my_knn1.predict(C)
                    
                    my_knn2.fit(A,B)
                    my_knn2.predict(C)
                    test_vec=C[:size_test]
                    
                    self.assertTrue(abs(test_vec-my_knn1.prediction).sum()<1e-12,"a) Checking knn regression with flat criterion.")
                    self.assertTrue(abs(test_vec-my_knn2.prediction).sum()<1e-12,"b) Checking knn regression with weighted criterion.")
        
        
         # TEST 2
        
        size_train=100
        size_test=50
        y_val=-.6  
        
        for i in range(1,size_train//2+1):
                    
            my_knn1 = MyKnnRegressor(distance="Euclidean",criterion="flat",n_neighbors=i,parallelize=False)
            my_knn2 = MyKnnRegressor(distance="Euclidean",criterion="weighted",n_neighbors=i,parallelize=False)
                    
            for i in n_input_feat:
                for j in n_output_feat:
                    A,C,B=create_mock_arrays_2(i,j,size_train,size_test,y_val)
                    my_knn1.fit(A,B)
                    my_knn1.predict(C)
                                        
                    
                    my_knn2.fit(A,B)
                    my_knn2.predict(C)
                    test_vec=C[:size_test//2]
                    test_vec=np.concatenate((test_vec,-test_vec),axis=0)
                        
                    self.assertTrue(abs(test_vec-my_knn1.prediction).sum()<1e-12,"c) Checking knn regression with flat criterion, 1<k<n_train_samples//2.")
                    self.assertTrue(abs(test_vec-my_knn2.prediction).sum()<1e-12,"d) Checking knn regression with weighted criterion, 1<k<n_train_samples//2.")
        
        for i in range(size_train//2+2,size_train+1):
                    
            my_knn1 = MyKnnRegressor(distance="Euclidean",criterion="flat",n_neighbors=i,parallelize=False)
            my_knn2 = MyKnnRegressor(distance="Euclidean",criterion="weighted",n_neighbors=i,parallelize=False)
                    
            for i in n_input_feat:
                for j in n_output_feat:
                    A,C,B=create_mock_arrays_2(i,j,size_train,size_test,y_val)
                    my_knn1.fit(A,B)
                    my_knn1.predict(C)
                    
                    my_knn2.fit(A,B)
                    my_knn2.predict(C)
                    
                    self.assertTrue((abs(my_knn1.prediction)<abs(y_val)).all(),"e) Checking knn regression with flat criterion, n_train_samples//2 < k <= n_train_samples.")
                    self.assertTrue((abs(my_knn2.prediction)<abs(y_val)).all(),"f) Checking knn regression with weighted criterion, n_train_samples//2 <= k < n_train_samples.")
        
        
    def test_myknnregressor_predict_with_fit_grid(self):
        
        """
            Testing the 2-d grid KNN
        
            TEST 1a/1b: testing the correctness of grid KNN, using the same approach as in "test_myknnregressor_predict_with_fit_serial", with few important differences.
                        1) The number of input features is constrained to 2
                        2) In TEST 1a the train and test set are obtained by randomly extracting the points of a 2D grid, using the function "create_mock_arrays_1"
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
                        These data arrays are produced by "create_mock_arrays_2".
                        
                        The result of knn regression for these two points should be
                        
                        a) KNN[(0,0)]_k = +val, if k<= min(Nx,Ny)/2
                           KNN[(Nx-1,Ny-1)]_k = -val, if k<= min(Nx,Ny)/2
        
                            where min(Nx,Ny)=Nx if Nx<Ny, Ny otherwise
                            
                        b) |KNN[(0,0)]_k| <=val, if k> min(Nx,Ny)/2
        """
        
        print("\n testing predict method using grid knn")
        
        
        sizes=[[5,5],[3,18],[10,10]]
        n_output_feat=[1,3,6]
        y_val=-.6        

        size_train=[]
        for i in sizes:
            size_train.append(i[0]*i[1]*3//4)
                
        for a,j in zip(size_train,sizes):
                
            for i in range(1,5):
                extensions=np.array(j)
                my_knn1 = MyKnnRegressor(distance="grid",criterion="flat",n_neighbors=i,grid_size=extensions,parallelize=False)
                my_knn2 = MyKnnRegressor(distance="grid",criterion="weighted",n_neighbors=i,grid_size=extensions,parallelize=False)
   
                for k in n_output_feat:
                    A,C,B=create_mock_grid_1(j[0],j[1],k,a,y_val)
                    my_knn1.fit(A,B)
                    my_knn1.predict(C)
                        
                    my_knn2.fit(A,B)
                    my_knn2.predict(C)
                    test_vec=C[:(j[0]*j[1]-a)]
                    self.assertTrue(abs(test_vec-my_knn1.prediction).sum()<1e-12,"a) Checking knn grid regression with flat criterion.")
                    self.assertTrue(abs(test_vec-my_knn2.prediction).sum()<1e-12,"b) Checking knn grid regression with weighted criterion.")
        
        sizes=[[5,5],[3,10],[7,10]]
        n_output_feat=[1,3,6]
        y_val=-.6        
        for i in sizes:
            extensions=np.array([i[0]*2,i[1]*2],dtype=int)
            
            for j in [1,min(i)]:
                my_knn1 = MyKnnRegressor(distance="grid",criterion="flat",n_neighbors=j,grid_size=extensions,parallelize=False)
                my_knn2 = MyKnnRegressor(distance="grid",criterion="weighted",n_neighbors=j,grid_size=extensions,parallelize=False)
        
                for k in n_output_feat:
                    A,C,B=create_mock_grid_2(i[0],i[1],k,y_val)
            
                    my_knn1.fit(A,B)
                    my_knn1.predict(C)

                    my_knn2.fit(A,B)
                    my_knn2.predict(C)   
                    test_vec=C[:1]
                    test_vec=np.concatenate((test_vec,-test_vec),axis=0)
                    self.assertTrue(abs(test_vec-my_knn1.prediction).sum()<1e-12,"c) Checking knn regression with flat criterion, 1<k<=min(Nx,Ny)//2.")
                    self.assertTrue(abs(test_vec-my_knn2.prediction).sum()<1e-12,"d) Checking knn regression with weighted criterion, 1<k<=min(Nx,Ny)//2.")

        
        #To be added: test for the regression with kdtree
        #test_myknnregressor_predict_with_fit_kd_tree
if __name__ == '__main__':
    unittest.main()
