# -*- coding: utf-8 -*-
"""
Created on Fri Oct 20 07:03:16 2017

Author: Francesco Capponi <capponi.francesco87@gmail.com>
     
License: BSD 3 clause
"""

from learners.mylearners import MyRegressor, MyClassifier

class MockClassRegressor(MyRegressor):
    
    def __init__(self):
        self.check_fit=False
        self.count_fit=0
        self.check_predict=False
        self.count_predict=0
        self.learning_type=None
        super().__init__()
        
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

class MockClassClassifier(MyClassifier):
    
    def __init__(self):
        self.check_fit=False
        self.count_fit=0
        self.check_predict=False
        self.count_predict=0
        self.learning_type=None
        super().__init__()
        
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
