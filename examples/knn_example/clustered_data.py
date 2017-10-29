# -*- coding: utf-8 -*-
"""
Created on Fri Oct  6 06:59:45 2017

Author: Francesco Capponi <capponi.francesco87@gmail.com>

This represent a simple example of application of knn regressor to a set
of mock data.
"""

import numpy as np
import matplotlib.pyplot as plt
from mock import generate_mock_data as gmd
from crossvalidation.mycrossvalidation import MyCrossValidation
from knn.myknnregressor import MyKnnRegressor

# Generating the data
ndata_per_cluster=500
pole_value=.01
X_train,y_train,X_test,y_test=gmd.generate_clustered_data(ndata_per_cluster,pole_value)
        
plt.figure()
plt.xlabel('X')
plt.ylabel('Y')
colors=['r','b','g','k']
plt.scatter(X_train[:,0],X_train[:,1],c=y_train.reshape(y_train.shape[0]))                        

# Tuning the number of neighbors by doing k-fold cross validation
nfolds=5
mycv = MyCrossValidation(kfolds=5,reshuffle=True)
test_values=range(1,ndata_per_cluster+10)
Rsquares=np.zeros((len(test_values),nfolds))


for k in test_values:
    my_knn_fold=MyKnnRegressor(method="classic",criterion="weighted",n_neighbors=k)
    mycv.cross_val(X_train,y_train,my_knn_fold)  
    Rsquares[k-1,:]=mycv.R_squared_collection

# Plotting the tuning procedure  
plt.figure()
plt.xlabel('Number of neighbors')
plt.ylabel('R^2')
plt.title('Determination coefficient across the fold')
   
for i in range(nfolds):
     plt.plot(test_values,1-Rsquares[:,i],)                          


avg_across_folds=(1-Rsquares).mean(axis=1)
plt.plot(test_values,avg_across_folds,'-',color='k',label='average across the folds')

opt_val=np.argwhere(avg_across_folds==avg_across_folds.min())        
plt.axvline(test_values[opt_val[0,0]],color='k',linestyle='--',label = 'optimal k')    
  
plt.legend()

optimal_k=test_values[opt_val[0,0]]

# Applying the algorithm with optimal number of neighbors
my_knn=MyKnnRegressor(method="classic",criterion="weighted",n_neighbors=optimal_k,leafsize=20)
rangesx,rangesy=np.linspace(X_train.min(),X_train.max(),200),np.linspace(X_train.min(),X_train.max(),200)
xv, yv = np.meshgrid(rangesx, rangesy)
X_test=np.column_stack((xv.flatten(),yv.flatten()))
my_knn.fit(X_train,X_test)
my_knn.predict(y_train)

# Plotting the result of the infilling
plt.figure()
plt.xlabel('X')
plt.ylabel('Y')
plt.scatter(X_test[:,0],X_test[:,1],c=my_knn.prediction.reshape(my_knn.prediction.shape[0]))    
plt.scatter(X_train[:,0],X_train[:,1],c=y_train.reshape(y_train.shape[0]))                      
     