"""

Author: Francesco Capponi <capponi.francesco87@gmail.com>
       
License: BSD 3 clause

"""

import unittest
import numpy as np
from ..myscores import R_squared
from ..myscores import class_score
from ..myscores import cross_entropy

class MyScores(unittest.TestCase):
    
    def test_R_squared(self):
        """
        Testing the correcteness of the R_squared function
        """
        
        print("\n testing computation of determination coefficient")
    
        N=20
        np.random.seed(1)
        for i in list([2,4,5]):        
            # TEST 1: checking computation of determination coefficient when target value and predictions are equal   
            y_pred=np.arange(N).reshape(i,N//i)
            y_test=y_pred
            self.assertEqual(R_squared(y_pred,y_test),1.,"a) Checking R_squared for y_pred=y_test.")
            
            # TEST 2: checking computation of determination coefficient when target value and predictions are almost equal 
            y_test=y_pred+np.random.normal(0,.0001,y_pred.shape)        
            self.assertAlmostEqual(R_squared(y_pred,y_test),1.,delta=10**(-9))#"b) Checking R_squared for y_test=ypred+gaussian noise.")
            
        # TEST 3: testing correct error raising
        y_pred=np.array([1,2,3])
        y_test=np.array([[1,2],[1,2]])
        self.assertRaises(ValueError,R_squared,y_pred,y_test)        
        
    def test_class_score(self):
        """
        Testing the correcteness of the classification score
        """
        
        print("\n testing computation of classification score")
        
        # TEST 1: checking computation of classification scores for different scenarios 
        A=np.array(['A','A','A'])
        B=np.array(['A','B','A'])
        C=np.array(['C','C','A'])

        self.assertEqual(class_score(A,A),1.,"a) Checking class_score for y_pred=y_test.")
        self.assertEqual(class_score(A,B),2./3.,"b) Checking class_score for y_pred!=y_test.")
        self.assertEqual(class_score(A,C),1./3.,"c) Checking class_score for y_pred!=y_test.")
        
        A=np.array([1,1,1])
        B=np.array([1,1,2])
        C=np.array([1,2,3])

        self.assertEqual(class_score(A,A),1.,"d) Checking class_score for y_pred=y_test.")
        self.assertEqual(class_score(A,B),2./3.,"e) Checking class_score for y_pred!=y_test.")
        self.assertEqual(class_score(A,C),1/3.,"f) Checking class_score for y_pred!=y_test.")

        # TEST 3: testing correct error raising
        y_pred=np.array([1,2,3])
        y_test=np.array([[1,2],[1,2]])
        self.assertRaises(ValueError,R_squared,y_pred,y_test)        

            
    def test_cross_entropy(self):
        """
        Testing the correctness of cross entropy score
        """
        
        print("\n testing computation of cross entropy score") 
        
        y_pred = np.array([.1, .2, .9, .8, .7])
        y_test = np.array([1, 0, 1, 1 ,0])
        self.assertAlmostEqual(cross_entropy(y_pred, y_test), 0.81164, places=5, msg="a) Testing cross entropy score")        
      
        y_pred = np.array([[.1, .2, .9, .8, .7]])
        y_test = np.array([[1, 0, 1, 1 ,1]])
        self.assertAlmostEqual(cross_entropy(y_pred, y_test), 0.64218, places=5, msg="b) Testing cross entropy score")
        
        y_pred = np.array([.1, .2, .9, .8, .7])
        y_test = np.array([1, 0, 0, 1 ,0])
        self.assertAlmostEqual(cross_entropy(y_pred, y_test), 1.25109
        , places=5, msg="c) Testing cross entropy score")
        
        y_pred = np.array([[.1, .2, .9, .8, .7]])
        y_test = np.array([[0, 0, 0, 1 ,0]])
        self.assertAlmostEqual(cross_entropy(y_pred, y_test), 0.81164, places=5, msg="d) Testing cross entropy score")


if __name__ == '__main__':
    unittest.main()
            
