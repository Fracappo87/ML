# -*- coding: utf-8 -*-
"""
Author: Francesco Capponi <capponi.francesco87@gmail.com>
        David Preti       <preti.david@gmail.com>

License: BSD 3 clause
"""

import numpy as np
# from metrics.myscores import R_squared
from checks.mycheck import sanitycheck

class MyCrossValidation(object):

    """
    My class for generation of the cross validation process.
    
    Methods:
            
    1) cross_val(X,Y,learner)
    
    Attributes:
    
    1) nfolds: int.
       Number of folds 
    2) first_folding: Bool.
       Bool variables that check if the index mask used for the folding has been initialized
    3) R_squared_collection: numpy.ndarray, shape = [nfolds]
       Collection of learning scores computed for each fold (regression learning only)
    """

    def __init__(self,kfolds=5,reshuffle=True):

        if (not isinstance(kfolds, int)) or (kfolds < 1):
            raise ValueError("number of neighbors has to be a positive integer number!.")
        elif not isinstance(reshuffle, bool):
            raise ValueError("parallelize has to be a bool variable, True or False!.")
        
        self.nfolds=kfolds
        self.shuf=reshuffle
        self.first_folding=True
        
##############################################################################

    """    
        Public methods.
    """

    def cross_val(self,X,Y,learner):
        """         
        Apply k-fold cross validation using the training data:

        Parameters
        ----------
        X:  numpy-like, shape = [n_train_sample,n_features]
        input features training data            
        
        Y : numpy-like, shape = [n_train_sample,n_features]
        output features training data 
        
        learner: any object referring to a learning algorithm (for classification or regression)
        a specific check is applied to ensure that such object has a "fit" and "predict 
        " method.
        
        Returns
        -------
        R_squared_collection: numpy-like, shape = [nfolds]
        Array of learning scores, whose dimension depends on the number of folds.
        """
        return self._cross_val(X,Y,learner)
        
##############################################################################        
        
    def _cross_val_regress(self,X,Y,learner,batch_length,idx):
        """
        Apply k-fold cross validation for the given regression learner
        """

        self.R_squared_collection = np.zeros(self.nfolds)
           
        for i in range(self.nfolds):
            test_index=self.index[i*batch_length:(i+1)*batch_length]
            train_index=np.concatenate((self.index[0:i*batch_length],self.index[(i+1)*batch_length:]),axis=0)    
            
            X_test_fold=X[test_index]
            Y_test_fold=Y[test_index]    
 
            X_train_fold=X[train_index]
            Y_train_fold=Y[train_index]
            
            args_fit=[[X_test_fold,Y_train_fold],[Y_train_fold,X_test_fold]]
            learner.fit(X_train_fold,args_fit[idx][0])
            learner.predict(args_fit[idx][1])
        
            self.R_squared_collection[i]=learner.score(learner.prediction,Y_test_fold)
            

    def _cross_val(self,X,Y,learner):
        #Initialization: check correctness of data format
        sanitycheck(X,np.ndarray)
        sanitycheck(Y,np.ndarray)
        
        learning=learner.learning_type
        if learning == 'instance_based':
            idx=0
        elif learning == 'training_based':             
            idx=1
        else:
            raise ValueError("Only two possible learning types are admitted: instance_based and training_based")            
        
        
        #Preparing the folds
        ndata=X.shape[0]
    
        if self.first_folding:
            self.index=np.arange(ndata,dtype=int)
            if self.shuf: np.random.shuffle(self.index)
            self.first_folding=False #You do not want to reshuffle when a hyper parameter changes
            
        batch_length=ndata//self.nfolds
             
        #Checking that the learner is either a regressor or a classifier
        learning = learner.learner_type
            
        if(learning == 'regressor'):
            return self._cross_val_regress(X,Y,learner,batch_length,idx)
        elif(learning =='classifier'):
            return self._cross_val_class(X,Y,learner)
        else:
            raise ValueError('The learner has to be either a regressor either a classifier!')
        
        
        
        
        
        
        
        
        
        
        
        
        