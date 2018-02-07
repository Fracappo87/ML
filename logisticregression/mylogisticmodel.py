"""

Author: Francesco Capponi <capponi.francesco87@gmail.com>

License: BSD 3 clause

"""

import numpy as np
from checks.mycheck import sanitycheck
from learners.mylearners import MyClassifier
from metrics.myscores import cross_entropy
from functions.functions import sigmoid

class MyLogisticRegressionClassifier(MyClassifier):
    """
    Class for implementing the logistic regression classifier
    """

    def __init__(self, n_dimensions, optimize = 'GradientDescent', start = 'random', offset = 0,):
        sanitycheck(start, str)
        sanitycheck(optimize, str)        
        
        if (optimize != 'GradientDescent') and (optimize != 'StochasticGradientDescent'):
            raise ValueError('\'optimize\' attribute can only be \'GradientDescent\' or \'StochasticGradientDescent\'')
            
        if (start != 'random') and (start != 'uniform'):
            raise ValueError('\'start\' attribute can only be \'random\' or \'uniform\'')

        MyClassifier.__init__(self)
        self.learning_type='training_based'
        self.n_dimensions = n_dimensions
        self.optimize = optimize
        self.start = start
        self.offset = offset

##############################################################################
  
    def fit(self,X,Y):
        """
        Compute the value of logistic model parameters, by using gradient descent.
        More details can be found in any good machine learning / statistics book
            
        Parameters
        ----------
        X : numpy-like, shape = [n_samples,n_input_features]        
        Y : numpy-like, shape = [n_samples,n_output_features]
        """
        return self._fit(X,Y)
    

    def predict(self,X):
        """
        Compute the predicted classes for the input values given by X
             
        Parameters
        ----------
        X : numpy-like, shape = [n_samples,n_input_features]
        """ 
        return self._predict(X)
    
 ##############################################################################
    
    def _initialize(self):
        """Initialize model weights according to user given flag"""
        
        
        if self.start == 'random':
            self.w = np.random.uniform(size=self.n_dimensions).reshape([self.n_dimensions,1])  
            self.b = np.random.uniform()
        elif self.start == 'uniform':
            self.w = np.zeros([self.n_dimensions, 1])+self.offset
            self.b = self.offset
            
    def _propagate(self, X_train, Y_train):
        """
        Implement forward and backward propagation
        """
        
        m = X_train.shape[0]
        
        # FORWARD PROPAGATION (FROM X TO COST)
        # need to work with transpose arrays: it facilitates computation
        
        A = sigmoid(np.dot(self.w.T,X_train.T)+self.b)                                     
        cost = cross_entropy(A,Y_train.T)
        
        # BACKWARD PROPAGATION (TO FIND GRAD)
        dw = 1./m*np.dot(X_train.T,(A.T-Y_train))
        db = 1./m*np.sum(A-Y_train.T)
        ### END CODE HERE ###
    
        assert(dw.shape == self.w.shape)
        assert(db.dtype == float)
        assert(cost.shape == ())
        
        grads = {"dw": dw,
                 "db": db}
        
        return grads, cost
                
    def _fit(self,X,Y):
        self.__sanitycheck(X,np.ndarray)
        self.__sanitycheck(Y,np.ndarray)
        
        if(len(X.shape)==1):
            X=X.reshape(-1,1)
            
        
        
    def _predict(self,X):
        self.__sanitycheck(X,np.ndarray)

        if(len(X.shape)==1):
            X=X.reshape(-1,1)

        
