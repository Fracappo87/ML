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

    def __init__(self, minibatch_size = 2**6, start = 'random', n_iterations = 10, offset = 0., learning_rate = .5, optimization = None):
        sanitycheck(start, str)
        sanitycheck(minibatch_size, int)        
        sanitycheck(n_iterations, int)
        sanitycheck(offset, float)
        sanitycheck(learning_rate, float)        
            
        if (start != 'random') and (start != 'uniform'):
            raise ValueError('\'start\' attribute can only be \'random\' or \'uniform\'')
            
        if (optimization != None) and (optimization != 'momentum') and (optimization != 'RMS') and (optimization != 'adam'):
            raise ValueError('\'optimization\' attribute can only be \'momentum\' or \'RMS\' or \'adam\'')


        MyClassifier.__init__(self)
        self.learning_type = 'training_based'
        self.learning_rate = learning_rate
        self.minibatch_size = minibatch_size
        self.n_iterations = n_iterations
        self.offset = offset
        self.optimization = optimization
        self.start = start

##############################################################################
  
    def fit(self,X,Y):
        """
        Compute the value of logistic model parameters, by using gradient descent.
        Numerical optimizations are adopted, depending on the value assigned to the 'optimize' attribute
        More details can be found in any good machine learning / statistics book
            
        Parameters
        ----------
        X : numpy.ndarray, shape = [n_samples,n_input_features]        
        Y : numpy.ndarray, shape = [n_samples,n_output_features]
        """
        return self._fit(X,Y)
    

    def predict(self,X):
        """
        Compute the predicted classes for the input values given by X
             
        Parameters
        ----------
        X : numpy.ndarray, shape = [n_samples,n_input_features]        
        """ 
        return self._predict(X)
    
 ##############################################################################
    
    def _initialize(self, ndimensions):
        """Initialize model weights according to user given flag"""
            
        if self.start == 'random':
            self.W = np.random.uniform(size=ndimensions).reshape([1, ndimensions])  
            self.b = np.random.uniform(size = 1).reshape([1, 1])
        elif self.start == 'uniform':
            self.W = np.zeros([1,ndimensions])+self.offset
            self.b = np.zeros([1, 1])+self.offset
            
    def _forward_prop(self, X_train, Y_train):
        """
        Implement forward propagation
        """
                     
        activation = sigmoid(np.dot(self.W,X_train.T)+self.b)                                     
        cost = cross_entropy(activation,Y_train.T)
                            
        return activation, cost

    def _back_prop(self, activation, X_train, Y_train):
        """
        Implement backward propagation
        """
        m = X_train.shape[0]
        dW = 1./m*np.dot((activation-Y_train.T), X_train)
        db = 1./m*np.sum(activation-Y_train.T)
    
        self.W -= self.learning_rate*dW
        self.b -= self.learning_rate*db
                        
    def _fit(self,X_train,Y_train):
        sanitycheck(X_train,np.ndarray)
        sanitycheck(Y_train,np.ndarray)
        
        if(len(X_train.shape)==1):
            X_train=X_train.reshape(-1,1)
        
        self._initialize(X_train.shape[1])
        self.cost_list = []
        
        for step in range(self.n_iterations):
            activation, cost = self._forward_prop(X_train,Y_train)
            self.cost_list.append(cost)
            self._back_prop(activation, X_train, Y_train)
            
    def _predict(self,X_test):
        sanitycheck(X_test,np.ndarray)

        if(len(X_test.shape)==1):
            X_test=X_test.reshape(-1,1)
            
        self.prediction = sigmoid(np.dot(self.W,X_test.T)+self.b) >= .5

        
