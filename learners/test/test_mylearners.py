"""

Author: Francesco Capponi <capponi.francesco87@gmail.com>
       
License: BSD 3 clause

"""

import unittest
import numpy as np
from ..mylearners import MyRegressor,MyClassifier

                
class MyLearnersTest(unittest.TestCase):
    
    def test_MyRegressor_attributes(self):
        """
        Testing the correcteness of the attributes
        """
        print("\n testing MyRegressor attributes initialization")
        
        regressor=MyRegressor()
        self.assertEqual(regressor.learner_type,"regressor","a) Checking the correct learner type definition.")
    
    def test_MyRegressor_score(self):
        """
        Testing the correcteness of the score method
        """
        print("\n testing computation of determination coefficient for regressors")
        
        regressor=MyRegressor()
        N=200
        np.random.seed(1)
        for i in list([2,4,5]):        
            # TEST 1: checking computation of determination coefficient when target value and predictions are equal   
            y_pred=np.arange(N).reshape(i,N//i)
            y_test=y_pred
            self.assertEqual(regressor.score(y_pred,y_test),1.,"a) Checking R_squared for y_pred=y_test.")
            
            # TEST 2: checking computation of determination coefficient when target value and predictions are almost equal 
            y_test=y_pred+np.random.normal(0,.0001,y_pred.shape)        
            self.assertAlmostEqual(regressor.score(y_pred,y_test),1.,delta=10**(-9))#"b) Checking R_squared for y_test=ypred+gaussian noise.")

    
    def test_MyClassifier_attributes(self):
        """                                                                                                            
        Testing the correcteness of the attributes                                                                     
        """

        print("\n testing MyClassifier attributes initialization")
        classifier=MyClassifier()
        self.assertEqual(classifier.learner_type,"classifier","a) Checking the correct learner type definition.")
    
    def test_MyClassifier_class_score(self):
        """
        Testing the correcteness of the classification score
        """
        print("\n testing computation of classification score for classifiers")
        
        classifier=MyClassifier() 
        
        # TEST 1: checking computation of classification scores for different scenarios 
        A=np.array(['A','b','A','c'])
        B=np.array(['A','B','A','c'])
        C=np.array(['C','C','A','f'])

        self.assertEqual(classifier.score(A,A),1.,"a) Checking class_score for y_pred=y_test.")
        self.assertEqual(classifier.score(A,B),3./4.,"b) Checking class_score for y_pred!=y_test.")
        self.assertEqual(classifier.score(A,C),1/4.,"c) Checking class_score for y_pred!=y_test.")
        
        A=np.array([1,2,3,4])
        B=np.array([1,1,2,1])
        C=np.array([1,2,5,6])

        self.assertEqual(classifier.score(A,A),1.,"d) Checking class_score for y_pred=y_test.")
        self.assertEqual(classifier.score(A,B),1./4.,"e) Checking class_score for y_pred!=y_test.")
        self.assertEqual(classifier.score(A,C),1./2.,"f) Checking class_score for y_pred!=y_test.")

        # TEST 3: testing correct error raising
        y_pred=np.array([1,2,3])
        y_test=np.array([[1,2],[1,2]])
        self.assertRaises(ValueError,classifier.score,y_pred,y_test)        

if __name__ == '__main__':
    unittest.main()
            
