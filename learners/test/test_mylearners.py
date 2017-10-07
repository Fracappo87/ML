"""

Author: Francesco Capponi <capponi.francesco87@gmail.com>
       
License: BSD 3 clause

"""

import unittest
from ..mylearners import MyRegressor,MyClassifier

                
class MyLearnersTest(unittest.TestCase):
    
    def test_MyRegressor_attributes(self):
        """
        Testing the correcteness of the attributes
        """
        
        print("\n testing regressor attributes initialization")
        regressor=MyRegressor()
        self.assertEqual(regressor.learner_type,"regressor","a) Checking the correct learner type definition.")
        
    def test_MyClassifier_attributes(self):
        """                                                                                                            
        Testing the correcteness of the attributes                                                                     
        """

        print("\n testing classifier attributes initialization")
        classifier=MyClassifier()
        self.assertEqual(classifier.learner_type,"classifier","a) Checking the correct learner type definition.")

if __name__ == '__main__':
    unittest.main()
            
