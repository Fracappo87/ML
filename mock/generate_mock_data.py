# -*- coding: utf-8 -*-
"""
Created on Thu Oct 19 07:18:15 2017

Author: Francesco Capponi <capponi.francesco87@gmail.com>
     
License: BSD 3 clause
"""

import numpy as np

def generate_spatially_gaussian_arrays_single_pole(n_input_features,n_output_features,size_train,size_test,y_val):
    """   
    It generates data arrays describing training input/output and test input.
    Data are gathered around the origin in feature space, and have all the same values in output space.
    This function is useful for doing simple tests of learning algorithms.    
    
    Test arrays
        
    X_train_1 = array-like, shape = [size_train,n_input_features]
    Y_train_1 = array-like, shape = [size_train,n_output_features]
    X_test_1 = array-like, shape = [size_test,n_input_features]
    """         
         
         
    centre=0.
    X_test=np.random.normal(centre,size=(size_test,n_input_features))
    X_train=np.random.normal(centre,size=(size_train,n_input_features))
    Y_train=np.array([y_val]*size_train*n_output_features).reshape(size_train,n_output_features)
        
    return X_train, Y_train, X_test

def generate_spatially_gaussian_arrays_double_pole(n_input_features,n_output_features,size_train,size_test,y_val,centre=100.,scale=.001,binary=False):
    """ 
    WARNING: UPDATE DOCUMENTATION ABOUT FINAL OUTPUT SIZES
    It generates data arrays describing training input/output and test input.
    Data are gathered around two opposite poles in feature space, and have the following values in output space:
    1) [y_val,y_val,...,y_val] for data gathered around the first pole
    2) -[y_val,y_val,...,y_val] for data gathered around the second pole
    This function is useful for doing simple non trivial tests of learning algorithms.
    The bool flag binary is used when mock data for classification algorithms have to be generated: when set to True it 
    set output space dimension to 1 and class values to y_val and 0 
    
    The position and spread of gaussian distributions can be set by assigning specific values (or array-values) to "centre" and "scale"    
    Test arrays
        
    X_train_1 = array-like, shape = [size_train,n_input_features]
    Y_train_1 = array-like, shape = [size_train,n_output_features]
    X_test_1 = array-like, shape = [size_test,n_input_features]   
    """         
    
    sizo=size_test//2
    pole_1 = np.random.normal(centre,scale,size=(sizo,n_input_features))
    pole_2 = np.random.normal(-centre,scale,size=(sizo,n_input_features))
    X_test=np.concatenate((pole_1,pole_2))
    
    sizo=size_train//2
    pole_1 = np.random.normal(centre,scale,size=(sizo,n_input_features))
    pole_2 = np.random.normal(-centre,scale,size=(sizo,n_input_features))
    X_train=np.concatenate((pole_1,pole_2))
    
    n_output_features=n_output_features*(not binary)+binary
    y_pole_1=np.array([y_val]*sizo*n_output_features).reshape(sizo,n_output_features)
    if binary: 
        y_pol_2=y_pole_1-y_val
    else:    
        y_pol_2=-1*y_pole_1
    Y_train = np.concatenate((y_pole_1,y_pol_2))
    
    return X_train, Y_train, X_test

def generate_grid_arrays_single_pole(X_len,Y_len,n_output_features,size_train,y_val):
        """ 
        It generates data arrays describing training input/output and test input.
        Data are distributes onto a 2-D regular grind in feature space, and have all the same values in output space.
        This function is useful for doing simple tests of learning algorithms.
 
        X_train = array-like, shape = [size_train,2]
        Y_train = array-like, shape = [size_train,n_output_features]
        X_test = array-like, shape =  [size_test,2]
        """
        
        a=X_len
        b=Y_len
        if size_train > a*b:
            raise ValueError("size_train can be at most equal to X_len*Y_len.")
            
        perc=a*b-size_train
        grid=np.zeros([a*b,2],dtype=int)

        sequence=[(x,y) for y in range(b) for x in range(a)]
        k=0
        for i,j in sequence:
            grid[k]=np.array([i,j])
            k+=1
            
        rand_idx=np.random.choice(np.arange(a*b),perc,replace=False)
        X_test=grid[rand_idx]
        X_train=grid[[x for x in range(a*b) if x not in rand_idx]]
        Y_train=np.array([y_val]*size_train*n_output_features).reshape(size_train,n_output_features)
        
        return X_train, Y_train, X_test

def generate_grid_arrays_double_pole(X_len,Y_len,n_output_features,y_val,binary=False):
        """   
        WARNING: UPDATE DOCUMENTATION ABOUT FINAL OUTPUT SIZES
        It generates data arrays describing training input/output and test input.
        Data are distributes onto a 2-D regular grind in feature space, and have the following values in output space:
        1) [y_val,y_val,...,y_val] for data belonging to the rectangle (0,0)-(Nx/2-1,Ny-1).
        2) -[y_val,y_val,...,y_val] for data belonging to the rectangle (Nx/2-1,Ny-1)-(Nx-1,Ny-1).
        X_test contains only two input points, given by (0,0) and (Nx-1,Ny-1)
        This function is useful for doing simple non trivial tests of learning algorithms.
        The bool flag binary is used when mock data for classification algorithms have to be generated: when set to True it 
        set output space dimension to 1 and class values to y_val and 0         
        
        Test arrays
            
        X_train = array-like, shape = [size_train,2]
        Y_train = array-like, shape = [size_train,n_output_features]
        X_test = array-like, shape =  [2,2]   
        """           
        a=X_len
        b=Y_len
        
        grid=np.zeros([4*a*b,2],dtype=int)

        sequence=[(x,y) for y in range(2*b) for x in range(2*a)]
        k=0
        for i,j in sequence:
            grid[k]=np.array([i,j])
            k+=1
        idx=[0,4*a*b-1]
        X_test=grid[idx]
        X_train=grid[[x for x in range(1,4*a*b-1)]]
        
        n_output_features=n_output_features*(not binary)+binary
        y_pole_1=np.array([y_val]*(len(X_train)//2)*n_output_features).reshape((len(X_train)//2),n_output_features)
        if binary: 
            y_pol_2=y_pole_1-y_val
        else:    
            y_pol_2=-1*y_pole_1
        Y_train = np.concatenate((y_pole_1,y_pol_2))
        
        return X_train, Y_train, X_test 
     
def generate_clustered_data(ndata_per_cluster=10,N=10,centres_value=[1,2,3,4]):
    """
    Generate a mock dataset for an instance-based learning algorithm, used to test
    crossvalidation class
    The dataset is a collection of points in 2D clustered around 4 different poles, with a specific value for each pole.
    Data are produced by using gaussian distributions for computing the 2D coordinates.
    If data are tightly clustered around their pole, the optimal value of neighbors should correspond to one.
    Indeed, for spatially separated clusters of data, we should have no difference in prediction if 1 or Ndata per cluster elements are used.
    In this case, the determination coefficients shoudl be exactly equal to 1.    
    The test probes both the computation of the optimal k value, and the final score.
    """
    
    if len(centres_value) !=4:
        raise ValueError("List of centers values must contain four elements.")
    np.random.seed(1) # fix the random seed to reproduce always the same result
    
    centres=[np.array([N,N]),np.array([-N,N]),np.array([N,-N]),np.array([-N,-N])]        
        
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
        
