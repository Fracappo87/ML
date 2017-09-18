# -*- coding: utf-8 -*-
"""

Author: Francesco Capponi <capponi.francesco87@gmail.com>
        David Preti       <preti.david@gmail.com>

License: BSD 3 clause

"""

import numpy as np



class MyCrossValidation(object):

    """
        My class for generation of the cross validation process.
        
        Methods:
                
        1) R_squared(self,y_pred,y_test)
        2) cross_val(X,Y,learner)
        
        Attributes:
        
        1) nfolds: int.
           Number of folds 
        2) first_folding: Bool.
           Bool variables that check if the index mask used for the folding has been initialized
        3) R_squared_collection: numpy.ndarray, shape = [nfolds]
           Collection of determination coefficients computed for each fold (regression learning only)
    """

    def __init__(self,kfolds=5,reshuffle=True):

        if (not isinstance(kfolds, int)) or (kfolds < 1):
            raise ValueError("number of neighbors has to be a positive integer number!.")
        elif not isinstance(reshuffle, bool):
            raise ValueError("parallelize has to be a bool variable, True or False!.")
        
        self.nfolds=kfolds
        self.__shuf=reshuffle
        self.first_folding=True
        
##############################################################################

    """    
        Private methods.
        
        Just for internal use
    """

    def __sanitycheck(self,X,types):
        # sanity check: just to be sure the user is giving the right parameters
        if not isinstance(X, types):
            raise ValueError("Object has to be a ",types)

##############################################################################

    """    
        Public methods.
    """

    def R_squared(self,y_pred,y_test):
        """
            Compute the determination coefficient:

            Parameters
            ----------
            y_pred,y_test : numpy-like, shape = [n_test_sample,n_features]
        """
        return self._R_squared(y_pred,y_test)



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
            Array of determination coefficients, whose dimension depends on the number of folds.
        """
        return self._cross_val(X,Y,learner)
        
##############################################################################
      
    def _R_squared(self,y_pred,y_test):
        rat1=np.power(y_test-y_pred,2).sum()
        avg=np.mean(y_test,axis=0)
        rat2=np.power(y_test-avg,2).sum()
        return 1-rat1/rat2
        
        
    def _cross_val_regress(self,X,Y,learner):
        """
            Apply k-fold cross validation for the given learner
        """
        ndata=X.shape[0]
        
        if self.first_folding:
            self.index=np.arange(ndata,dtype=int)
            if self.__shuf: np.random.shuffle(self.index)
            self.first_folding=False
            
        batch_length=ndata//self.nfolds
        self.R_squared_collection = np.zeros(self.nfolds)
        
        learning=learner.learning_type
        if learning == 'instance_based':
            idx=0
        elif learning == 'training_based':             
            idx=1
        else:
            raise ValueError("Only two possible learning types are admitted: instance_based and training_based")            
            
        m=0
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
        
            self.R_squared_collection[m]=self._R_squared(learner.prediction,Y_test_fold)
            m+=1


    def _cross_val(self,X,Y,learner):
        #Initialization: check correctness of data format
        self.__sanitycheck(X,np.ndarray)
        self.__sanitycheck(Y,np.ndarray)
        
        #Checking that the learner has two fundamental methods: fit and predict
        learning = learner.learner_type
            
        if(learning == 'regressor'):
            return self._cross_val_regress(X,Y,learner)
        elif(learning =='classifier'):
            return self._cross_val_class(X,Y,learner)
        else:
            raise ValueError('The learner has to be either a regressor either a classifier!')
        
        
        
        
        
        
        
        
        
        
        
        
        