"""

Author: Francesco Capponi <capponi.francesco87@gmail.com>

License: BSD 3 clause

"""

import unittest
import numpy as np
from ..mycrossvalidation import MyCrossValidation
from knn.myknnregressor import MyKnnRegressor


class MockClass(object):
    
    def __init__(self):
        self.check_fit=False
        self.count_fit=0
        self.check_predict=False
        self.count_predict=0
        self.learner_type=None
        self.learning_type=None
        
        
    def fit(self,X,Y):
        self.count_fit+=1
        self.check_fit=True
        if (self.learning_type=='instance_based'):
            self.prediction=Y
        
    def predict(self,X):
        self.count_predict+=1
        self.check_predict=True
        if (self.learning_type=='training_based'):
            self.prediction=X

        
def generate_mock_data(ndata_per_cluster=10,N=10):
    """
    Generate a mock dataset for an instance-based learning algorithm, used to test
    crossvalidation class
    The dataset is a collection of points in 2D clustered around 4 different poles, with a specific value for each pole.
    Data are produced by using gaussian distributio for computing the 2D coordinates.
    If data are tightly clustered around their pole, the optimal value of neighbors should correspond to one.
    Indeed, for spatially separated clusters of data, we should have no difference in prediction if 1 or Ndata per cluster elements are used.
    In this case, the determination coefficients shoudl be exactly equal to 1.    
    The test probes both the computation of the optimal k value, and the final score.
    """
    
    np.random.seed(1) # fix the random seed to reproduce always the same result
    
    centres=[np.array([N,N]),np.array([-N,N]),np.array([N,-N]),np.array([-N,-N])]        
    centres_value=[30.3,-70.3,5.3,-4.4]    
    ndata=ndata_per_cluster*len(centres)
    
    X=np.zeros([ndata,2])
    y=np.zeros(ndata)
    
    for i,centres_info in enumerate(zip(centres,centres_value)):
        X[i*ndata_per_cluster:(i+1)*ndata_per_cluster] = np.random.normal(centres_info[0],0.01,size=(ndata_per_cluster,2))
        y[i*ndata_per_cluster:(i+1)*ndata_per_cluster] += centres_info[1]
        
    index=np.arange(ndata)
    np.random.shuffle(index)
        
    fraction=.25
    delimiter=int(ndata*fraction)
    train_index=index[delimiter:]
    test_index=index[:delimiter]
        
    X_train=X[train_index]
    y_train=y[train_index]
    
    X_test=X[test_index]
    y_test=y[test_index]
    
    y_train=y_train.reshape(-1,1)
    y_test=y_test.reshape(-1,1)
    return X_train,y_train,X_test,y_test
        
        
class MyKnnTest(unittest.TestCase):
    
    def test_mycrossvalidation_attributes(self):
        
        print("\n testing attributes initialization")
        
        # TEST 1: checking parameters initialization using correct parameters
        my_cv = MyCrossValidation(kfolds=5,reshuffle=True)
        self.assertEqual(5,my_cv.nfolds,"a) Checking the correct number of folds.")
        self.assertTrue(my_cv.first_folding,"b) Checking the correct initialization of 'first_folding' attribute.")
        self.assertTrue(my_cv.shuf,"c) Checking the correct initialization of 'shuf' attribute.")
        
        # TEST 2: checking correct exception raising when wrong input parameters are given
        self.assertRaises(ValueError,MyCrossValidation,kfolds=-5,reshuffle=True)          
        self.assertRaises(ValueError,MyCrossValidation,kfolds=5,reshuffle='a')
        
    def test_mycrossvalidation_R_squared(self):
        
        print("\n testing computation of determination coefficient")
    
        my_cv = MyCrossValidation(kfolds=5,reshuffle=True)
        
        N=20
        np.random.seed(1)
        for i in list([2,4,5]):        
            # TEST 1: checking computation of determination coefficient when target value and predictions are equal            
            y_pred=np.arange(N).reshape(i,N//i)
            y_test=y_pred
            self.assertEqual(my_cv.R_squared(y_pred,y_test),1.,"a) Checking R_squared for y_pred=y_test.")
        
            # TEST 2: checking computation of determination coefficient when target value and predictions are almost equal 
            y_test=y_pred+np.random.normal(0,.0001,y_pred.shape)        
            self.assertAlmostEqual(my_cv.R_squared(y_pred,y_test),1.,delta=10**(-9))#"b) Checking R_squared for y_test=ypred+gaussian noise.")
        
    def test_mycrossvalidation_cross_val_no_data(self):
        
        print("\n testing crossvalidation procedure using mock learner class, no data")
        
        my_cv = MyCrossValidation(kfolds=5,reshuffle=True)
                        
        N=200 
        shapes=list([2,10,20])
        
        #TEST 1: setting learner type to regression, training based            
        for i in shapes:
            X=np.arange(N).reshape(N//i,i)
            Y=X

            mock_training_based=MockClass()
            mock_training_based.__setattr__('learner_type','regressor')
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

            mock_training_based=MockClass()
            mock_training_based.__setattr__('learner_type','regressor')
            mock_training_based.__setattr__('learning_type','instance_based')
                        
            my_cv.cross_val(X,Y,mock_training_based)
            self.assertFalse(my_cv.first_folding,"a) Checking crossvalidation: modification of 'first_folding attribute'.")
            my_cv.__setattr__('first_folding',True)             
            
            self.assertEqual(mock_training_based.count_fit,my_cv.nfolds,"b) Checking crossvalidation: call of learner 'fit' method.")
            self.assertTrue(mock_training_based.check_fit,"c) Checking crossvalidation: call of learner 'fit' method.")
            self.assertEqual(mock_training_based.count_predict,my_cv.nfolds,"d) Checking crossvalidation: call of learner 'predict' method.")
            self.assertTrue(mock_training_based.check_predict,"e) Checking crossvalidation: call of learner 'predict' method.")
            self.assertTrue(my_cv.first_folding,"f) Checking crossvalidation: modification of 'first_folding attribute'.")

    def test_mycrossvalidation_cross_val_with_data_instance_based(self):
        
        print("\n testing crossvalidation procedure using knn regressor class, mock linear data with noise")
        
        ndata_per_cluster=50
        pole_value=.05
        X_train,y_train,X_test,y_test=generate_mock_data(ndata_per_cluster,pole_value)
        
        
        # Tuning optimal number of neighbors by using cross validation
        nfolds=5
        mycv = MyCrossValidation(kfolds=5,reshuffle=True)
        test_values=range(1,ndata_per_cluster+5)
        Rsquares=np.zeros((len(test_values),nfolds))
        
        for k in test_values:
            my_knn_fold=MyKnnRegressor(distance="Euclidean",criterion="weighted",n_neighbors=k)
            mycv.cross_val(X_train,y_train,my_knn_fold)  
            Rsquares[k-1,:]=mycv.R_squared_collection
        
        # Averaging across the folds: looking for the optimal value of k
        avg_across_folds=(1-Rsquares).mean(axis=1)
        opt_val=np.argwhere(avg_across_folds==avg_across_folds.min())
        optimal_k=test_values[opt_val[0,0]]
 
        # Applying the algorithm with optimal number of neighbors

        my_knn=MyKnnRegressor(distance="Euclidean",criterion="weighted",n_neighbors=optimal_k)
        my_knn.fit(X_train,X_test)
        my_knn.predict(y_train)
        
        self.assertEqual(optimal_k,1,"a) testing optimal number of neighbors from cross validation analysis of knn")
        self.assertEqual(mycv.R_squared(my_knn.prediction,y_test),1.,"b) testing determination coefficient")
       
# STILL TO DO: check cv using real datasets and two different typer of algorithms.        
if __name__ == '__main__':
    unittest.main()
