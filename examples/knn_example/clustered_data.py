# -*- coding: utf-8 -*-
"""
Created on Fri Oct  6 06:59:45 2017

@author: Frank
"""

import numpy as np
import matplotlib.pyplot as plt
from crossvalidation.mycrossvalidation import MyCrossValidation
from knn.myknnclassifier import MyKnnClassifier


def generate_mock_data(ndata_per_cluster=10,N=10):
    """
    Generate a mock dataset for an instance-based learning algorithm, used to test
    crossvalidation class
    The dataset is a collection of points in 2D clustered around 4 different poles, with a specific value for each pole.
    Data are produced by using gaussian distributio for computing the 2D coordinates.
    If data are tightly clustered around their pole, the optimal value of neighbors should correspond (or be close) to the number
    of elements in each cluster (fixed by user).
    The test probes both the computation of the optimal k value, and the final score.
    """
    
    np.random.seed(1) # fix the random seed to reproduce always the same result
    
    centres=[np.array([N,N]),np.array([-N,N]),np.array([N,-N]),np.array([-N,-N])]        
    centres_value=[1,2,3,4]    
    ndata=ndata_per_cluster*len(centres)
    
    X=np.zeros([ndata,2])
    y=np.zeros(ndata)
    
    for i,centres_info in enumerate(zip(centres,centres_value)):
        X[i*ndata_per_cluster:(i+1)*ndata_per_cluster] = np.random.normal(centres_info[0],0.01,size=(ndata_per_cluster,2))
        y[i*ndata_per_cluster:(i+1)*ndata_per_cluster] += centres_info[1]
        
    index=np.arange(ndata)
    np.random.shuffle(index)
        
    fraction=.25
    delimiter=int(ndata*fraction)
    train_index=index[delimiter:]
    test_index=index[:delimiter]
        
    X_train=X[train_index]
    y_train=y[train_index]
    
    X_test=X[test_index]
    y_test=y[test_index]
    
    y_train=y_train.reshape(-1,1)
    y_test=y_test.reshape(-1,1)
    return X_train,y_train,X_test,y_test
        
        
        
ndata_per_cluster=500
pole_value=.02
X_train,y_train,X_test,y_test=generate_mock_data(ndata_per_cluster,pole_value)
        
plt.figure()
plt.xlabel('X')
plt.ylabel('Y')
colors=['r','b','g','k']
markers=['s','o','v','*']
for i in range(4):
     indexes=y_train[i*ndata_per_cluster:(i+1)*ndata_per_cluster]
     plt.scatter(X_train[i*ndata_per_cluster:(i+1)*ndata_per_cluster,0],X_train[i*ndata_per_cluster:(i+1)*ndata_per_cluster,1],c=indexes)                         


# Tuning the number of neighbors by doing k-fold cross validation

#nfolds=5
#mycv = MyCrossValidation(kfolds=5,reshuffle=True)
#test_values=range(1,ndata_per_cluster+10)
#Rsquares=np.zeros((len(test_values),nfolds))


#for k in test_values:
#    my_knn_fold=MyKnnRegressor(method="Euclidean",criterion="weighted",n_neighbors=k)
#    mycv.cross_val(X_train,y_train,my_knn_fold)  
#    Rsquares[k-1,:]=mycv.R_squared_collection
    
#plt.figure()
#plt.xlabel('Number of neighbors')
#plt.ylabel('R^2')
#plt.title('Determination coefficient across the fold')
   
#for i in range(nfolds):
#     plt.plot(test_values,1-Rsquares[:,i],)                          


#avg_across_folds=(1-Rsquares).mean(axis=1)
#plt.plot(test_values,avg_across_folds,'-',color='k',label='average across the folds')

#opt_val=np.argwhere(avg_across_folds==avg_across_folds.min())        
#plt.axvline(test_values[opt_val[0,0]],color='k',linestyle='--',label = 'optimal k')    
  
#plt.legend()

#optimal_k=test_values[opt_val[0,0]]
optimal_k=10
# Applying the algorithm with optimal number of neighbors

my_knn=MyKnnClassifier(method="Euclidean",criterion="weighted",n_neighbors=optimal_k,leafsize=20)
rangesx,rangesy=np.linspace(X_train.min(),X_train.max(),200),np.linspace(X_train.min(),X_train.max(),200)
xv, yv = np.meshgrid(rangesx, rangesy)
X_test=np.column_stack((xv.flatten(),yv.flatten()))
my_knn.fit(X_train,X_test)
my_knn.predict(y_train)



print(mycv.R_squared(my_knn.prediction,y_test))

plt.figure()
plt.xlabel('X')
plt.ylabel('Y')
plt.scatter(X_test[:,0],X_test[:,1],c=my_knn.prediction)                          
for i in range(4):
     indexes=y_train[i*ndata_per_cluster:(i+1)*ndata_per_cluster]
     plt.scatter(X_train[i*ndata_per_cluster:(i+1)*ndata_per_cluster,0],X_train[i*ndata_per_cluster:(i+1)*ndata_per_cluster,1],c=indexes)                         
