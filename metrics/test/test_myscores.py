"""

Author: Francesco Capponi <capponi.francesco87@gmail.com>
       
License: BSD 3 clause

"""

import unittest
import numpy as np
from ..myscores import R_squared

                
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
            self.assertEqual(my_cv.R_squared(y_pred,y_test),1.,"a) Checking R_squared for y_pred=y_test.")
            
            # TEST 2: checking computation of determination coefficient when target value and predictions are almost equal 
            y_test=y_pred+np.random.normal(0,.0001,y_pred.shape)        
            self.assertAlmostEqual(my_cv.R_squared(y_pred,y_test),1.,delta=10**(-9))#"b) Checking R_squared for y_test=ypred+gaussian noise.")
            
            
if __name__ == '__main__':
    unittest.main()
            
