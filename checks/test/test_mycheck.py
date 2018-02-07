"""

Author: Francesco Capponi <capponi.francesco87@gmail.com>
       
License: BSD 3 clause

"""

import unittest
import numpy as np
from ..mycheck import sanitycheck
from ..mycheck import shape_test

                
class MyCheckTest(unittest.TestCase):
    
    def test_sanitycheck(self):
        """
        Testing the correcteness of the sanitycheck function
        """
        
        print("\n testing sanitycheck function")
        X=np.array([1,2,3])
        A="A"
        self.assertRaises(ValueError,sanitycheck,X=X,types=str)
        self.assertRaises(ValueError,sanitycheck,X=X,types=int)
        self.assertRaises(ValueError,sanitycheck,X=A,types=np.ndarray)
        
    def test_shape(self):
        """
        Testing the correcteness of the shape_test function
        """

        print("\n testing shape_test function")
        X=np.array([1,2,3])
        Y=np.array([[1,2],[3,4]])
        
        self.assertRaises(ValueError, shape_test, X=X, Y=Y)
                
if __name__ == '__main__':
    unittest.main()
            
