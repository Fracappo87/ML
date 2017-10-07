"""

Author: Francesco Capponi <capponi.francesco87@gmail.com>
       
License: BSD 3 clause

"""

import unittest
import numpy as np
from ..mynorms import Euclidean,L1

                
class MyNorms(unittest.TestCase):
    
    def test_Euclidan(self):
        """
        Testing the correcteness of the Euclidean norm function
        """
        print("\n testing Euclidean function")
        X=np.array([[-1,1],[1,-1],[-1,1]])
        self.assertTrue((Euclidean(X)==np.sqrt(2)).all(),"a) testing euclidean norm with 2D array, all norms equal")
        X=np.array([[3,0,4],[4,3,0]])
        self.assertTrue((Euclidean(X)==5.).all(),"b) testing euclidean norm with 3D array, all norms equal")
        X=np.array([[-3,2,-2],[-1,1,0]])
        self.assertEqual((Euclidean(X)-np.array([np.sqrt(17),np.sqrt(2)])).sum(),0,"c) testing euclidean norm with 3D array, different norms component")        
        
    def test_L1(self):
        """
        Testing the correcteness of the L1 norm function
        """
        print("\n testing L1 function")
        X=np.array([[-1,1],[1,-1],[-1,1]])
        self.assertTrue((L1(X)==2.).all(),"a) testing euclidean norm with 2D array, all norms equal")
        X=np.array([[3,0,4],[4,3,0]])
        self.assertTrue((L1(X)==7.).all(),"b) testing euclidean norm with 3D array, all norms equal")
        X=np.array([[-3,2,-2],[-1,1,0]])
        self.assertEqual((L1(X)-np.array([7.,2.])).sum(),0,"c) testing euclidean norm with 3D array, different norms component")        

if __name__ == '__main__':
    unittest.main()
            
