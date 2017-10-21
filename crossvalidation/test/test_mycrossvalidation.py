"""
Author: Francesco Capponi <capponi.francesco87@gmail.com>

License: BSD 3 clause
"""

import unittest
import numpy as np
from mock import generate_mock_data as gmd
from mock.generate_mock_class import MockClassRegressor
from ..mycrossvalidation import MyCrossValidation
from knn.myknnregressor import MyKnnRegressor
        
class MyCrossValidationTest(unittest.TestCase):
    
    def test_mycrossvalidation_attributes(self):
        
        print("\n testing MyCrossValidation attributes initialization")
        
        # TEST 1: checking parameters initialization using correct parameters
        my_cv = MyCrossValidation(kfolds=5,reshuffle=True)
        self.assertEqual(5,my_cv.nfolds,"a) Checking the correct number of folds.")
        self.assertTrue(my_cv.first_folding,"b) Checking the correct initialization of 'first_folding' attribute.")
        self.assertTrue(my_cv.shuf,"c) Checking the correct initialization of 'shuf' attribute.")
        
        # TEST 2: checking correct exception raising when wrong input parameters are given
        self.assertRaises(ValueError,MyCrossValidation,kfolds=-5,reshuffle=True)          
        self.assertRaises(ValueError,MyCrossValidation,kfolds=5,reshuffle='a')
                
    def test_mycrossvalidation_cross_val_regress_no_data(self):
        
        print("\n testing (regression) crossvalidation procedure using mock learner class, no data")
        
        my_cv = MyCrossValidation(kfolds=5,reshuffle=True)
                        
        N=200 
        shapes=list([2,10,20])
        
        #TEST 1: setting learner type to regression, training based            
        for i in shapes:
            X=np.arange(N).reshape(N//i,i)
            Y=X

            mock_training_based=MockClassRegressor()
            mock_training_based.__setattr__('learning_type','training_based')
                        
            my_cv.cross_val(X,Y,mock_training_based)
            self.assertFalse(my_cv.first_folding,"a) Checking crossvalidation: modification of 'first_folding attribute'.")
            my_cv.__setattr__('first_folding',True)             
            
            self.assertEqual(mock_training_based.count_fit,my_cv.nfolds,"b) Checking crossvalidation: call of learner 'fit' method.")
            self.assertTrue(mock_training_based.check_fit,"c) Checking crossvalidation: call of learner 'fit' method.")
            self.assertEqual(mock_training_based.count_predict,my_cv.nfolds,"d) Checking crossvalidation: call of learner 'predict' method.")
            self.assertTrue(mock_training_based.check_predict,"e) Checking crossvalidation: call of learner 'predict' method.")
            self.assertTrue(my_cv.first_folding,"f) Checking crossvalidation: modification of 'first_folding attribute'.")
            
        #TEST 2: setting learner type to regression, instance based            
        for i in shapes:
            X=np.arange(N).reshape(N//i,i)
            Y=X

            mock_training_based=MockClassRegressor()
            mock_training_based.__setattr__('learning_type','instance_based')
                        
            my_cv.cross_val(X,Y,mock_training_based)
            self.assertFalse(my_cv.first_folding,"a) Checking crossvalidation: modification of 'first_folding attribute'.")
            my_cv.__setattr__('first_folding',True)             
            
            self.assertEqual(mock_training_based.count_fit,my_cv.nfolds,"b) Checking crossvalidation: call of learner 'fit' method.")
            self.assertTrue(mock_training_based.check_fit,"c) Checking crossvalidation: call of learner 'fit' method.")
            self.assertEqual(mock_training_based.count_predict,my_cv.nfolds,"d) Checking crossvalidation: call of learner 'predict' method.")
            self.assertTrue(mock_training_based.check_predict,"e) Checking crossvalidation: call of learner 'predict' method.")
            self.assertTrue(my_cv.first_folding,"f) Checking crossvalidation: modification of 'first_folding attribute'.")

    def test_mycrossvalidation_cross_val_regress_with_data_instance_based(self):
        
        print("\n testing (regression) crossvalidation procedure using knn regressor class, mock clustered data with noise")
        
        ndata_per_cluster=50
        pole_value=.05
        X_train,y_train,X_test,y_test=gmd.generate_clustered_data(ndata_per_cluster,pole_value)
        
        
        # Tuning optimal number of neighbors by using cross validation
        nfolds=5
        mycv = MyCrossValidation(kfolds=5,reshuffle=True)
        test_values=range(1,ndata_per_cluster+5)
        Rsquares=np.zeros((len(test_values),nfolds))
        
        for k in test_values:
            my_knn_fold=MyKnnRegressor(method="Euclidean",criterion="weighted",n_neighbors=k)
            mycv.cross_val(X_train,y_train,my_knn_fold)  
            Rsquares[k-1,:]=mycv.R_squared_collection
        
        # Averaging across the folds: looking for the optimal value of k
        avg_across_folds=(1-Rsquares).mean(axis=1)
        opt_val=np.argwhere(avg_across_folds==avg_across_folds.min())
        optimal_k=test_values[opt_val[0,0]]
 
        # Applying the algorithm with optimal number of neighbors

        my_knn=MyKnnRegressor(method="Euclidean",criterion="weighted",n_neighbors=optimal_k)
        my_knn.fit(X_train,X_test)
        my_knn.predict(y_train)
        
        self.assertEqual(optimal_k,1,"a) testing optimal number of neighbors from cross validation analysis of knn")
        self.assertEqual(my_knn.score(my_knn.prediction,y_test),1.,"b) testing determination coefficient")
       
# STILL TO DO: check cv using real datasets and two different typer of algorithms.        
if __name__ == '__main__':
    unittest.main()
