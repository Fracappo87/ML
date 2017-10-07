# -*- coding: utf-8 -*-
"""

Author: Francesco Capponi <capponi.francesco87@gmail.com>
        David Preti       <preti.david@gmail.com>

License: BSD 3 clause

"""

import numpy as np
from knn.myknn import MyKnn
from learners.mylearners import MyRegressor
from checks.mycheck import sanitycheck

class MyKnnRegressor(MyKnn,MyRegressor):

    """
    Our KNN regressor. It allows regression analysis the k-nearest-neighbors algorithms
    of the classe MyKnn
    
    Private parameters
    ----------
    
    Attributes
    ---------- 
    """


    def __init__(self,distance="Euclidean",criterion="flat",n_neighbors=5,leafsize=100,grid_size=np.zeros(1),parallelize=False):
        super().__init__(distance,n_neighbors,leafsize,grid_size,parallelize)
        
        if not isinstance(criterion, str):
            raise ValueError("criterion has to be a string!.")
        elif (criterion != "flat" and criterion != "weighted"):
            raise ValueError("criterion method can only be \"flat\" or \"weighted\"!.")

        self.__crit=criterion
        self.learning_type='instance_based'
        
##############################################################################

    """
        Public methods.
    """
    
    def predict(self, Y_train):
        """
            Predict the value of new input distances using the fitted KNN regressor.
            It has to be called after having used the "fit" method.

            Parameters:
            ----------

            Y_train : numpy-like, shape = [n_samples, n_output_features]

            The method compute takes information from the following attributes

            neighbors_idx : numpy-like, shape = [n_test_samples, n_first_k_neighbors]
            neighbors_dist :  numpy-like, shape = [n_test_samples, n_first_k_neighbors]
            
            and use them to predict the output values of the new instances.         
        """

        return self._predict(Y_train)


             
    def _predict(self, Y_train):
        
        sanitycheck(Y_train,np.ndarray)
        
        if( not len(self.neighbors_idx)) or ( not len(self.neighbors_dist)):
            raise ValueError("You need to call the \"fit\" method before!\n")
        
        # check for single or multiple output value        
        if(len(Y_train.shape)==1):
            length=1
        elif(len(Y_train.shape)==2):
            length=Y_train.shape[1]
        else:
            raise ValueError("Output vector must have the following shapes:\n [n_samples,], shape=(k,)\n or \n [n_samples, n_input_features] : shape=(k,l)\n")
        self.prediction=np.zeros([len(self.neighbors_idx),length])
        i=0
        if self.__crit=="flat":
            for j in self.neighbors_idx:
                self.prediction[i]=Y_train[j].mean(axis=0)
                i+=1
        elif self.__crit=="weighted":
            for j,k in zip(self.neighbors_idx,self.neighbors_dist):
                self.prediction[i]=np.average(Y_train[j],axis=0,weights=1/k)
                i+=1
        return self.prediction
            
            
            
            