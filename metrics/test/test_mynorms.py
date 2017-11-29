"""
Author: Francesco Capponi <capponi.francesco87@gmail.com>       
License: BSD 3 clause
"""

import unittest
import numpy as np
import numpy.testing as npt
from ..mynorms import Euclidean, L1
                
class MyNorms(unittest.TestCase):    
    def test_Euclidan(self):
        """
        Testing the correcteness of the Euclidean norm function
        """
        print("\n testing Euclidean function")
        X = np.array([[-1,1],[1,-1],[-1,1]])
        npt.assert_array_equal(Euclidean(X),np.array([np.sqrt(2),np.sqrt(2),np.sqrt(2)]),err_msg = "a) testing euclidean norm with 2D array, all norms equal")
        X = np.array([[3,0,4],[4,3,0]])
        npt.assert_array_equal(Euclidean(X),np.array([5.,5.]),err_msg = "b) testing euclidean norm with 3D array, all norms equal")
        X = np.array([[-3,2,-2],[-1,1,0]])
        npt.assert_array_equal(Euclidean(X),np.array([np.sqrt(17),np.sqrt(2)]),err_msg = "c) testing euclidean norm with 3D array, different norms component")        
        
    def test_L1(self):
        """
        Testing the correcteness of the L1 norm function
        """
        print("\n testing L1 function")
        X = np.array([[-1,1],[1,-1],[-1,1]])
        npt.assert_array_equal(L1(X),np.array([2.,2.,2.]),err_msg = "a) testing euclidean norm with 2D array, all norms equal")
        X = np.array([[3,0,4],[4,3,0]])
        npt.assert_array_equal(L1(X),np.array([7.,7.]),err_msg = "b) testing euclidean norm with 3D array, all norms equal")
        X = np.array([[-3,2,-2],[-1,1,0]])
        npt.assert_array_equal(L1(X),np.array([7.,2.]),err_msg = "c) testing euclidean norm with 3D array, different norms component")        

if __name__ == '__main__':
    unittest.main()
            
