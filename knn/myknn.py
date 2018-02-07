"""
Research of K-nearest neighbors. It provides the classical exact algorithm, along with approximated methods, useful when dealing with large datasets.
"""

import numpy as np
from metrics.mynorms import Euclidean
from checks.mycheck import sanitycheck

class MyKnn(object):

    """
    Class for K-neares-neighbors search.
    
    Parameters
    ----------  
    method : {'classic', 'grid', 'kd-tree'}
        Available methods provided by the class for computing K-nearest-neighbors.
             
    n_neighbors : int
        Number of neighbors to be determined for each data instance.
                  
    leafsize : int, optional, default = 100
        Parameter for tuning the leafsize when the 'kd-tree' method is adopted.
               
    grid_size : numpy.ndarray, optional, default = 0
        Size of the 2D grid. To be initialized when the 'grid' method is adopted.
                
    parallelize : bool, optional, default = True
        Flag for parallelizing the computation of K-nearest neighbors. Useful when dealing with large datasets: it implies the use of the 'kd-tree' method.
                  
    Attributes
    ----------  
    neighbors_dist: 
        numpy.ndarray, shape = [n_samples, n_neighbors]
        Array of indices referring to the K-nearest neighbors of a collection of data instances.
    
    neighbors_idx: 
        numpy.ndarray, shape = [n_samples, n_neighbors]
        Array of indices referring to the distances from a collection of data instances and their corresponding K-nearest neighbors.                  
    """


    def __init__(self,method="classic",n_neighbors=5,leafsize=100,grid_size=np.zeros(1),parallelize=False):
        if not isinstance(method, str):
            raise ValueError("method has to be a string!.")
        elif (not isinstance(n_neighbors, int)) or (n_neighbors <= 0):
            raise ValueError("number of neighbors has to be a positive integer number!.")
        elif (not isinstance(leafsize, int)) or (leafsize <= 0):
            raise ValueError("number of neighbors has to be a positive integer number!.")         
        elif (not isinstance(grid_size,np.ndarray)):
            raise ValueError("grid size has to be a bidimensional array!")
        elif not isinstance(parallelize, bool):
            raise ValueError("parallelize has to be a bool variable, True or False!.")
        elif method!="classic" and method != 'grid' and method != "kd-tree":
            raise ValueError("knn method can only be \" classic\",  \"grid\" or \"kd-tree\"!.")
        
        self.method=method
        self.__k=n_neighbors
        self.__paral=parallelize
        self.__leafsize = leafsize
        self.__grid_size = grid_size 
        self.__paral = parallelize
        
        super().__init__()
##############################################################################

    @staticmethod
    def _lexico(X,extensions):
        
        """
        Takes an array containing the d-dimensional coordiantes of n_samples points an return an array of the same length, with coordinates substituted by the corresponding lexicographic index.
        
        Parameters
        ----------
        
        X : numpy.ndarray,
            shape = [n_samples,n_input_features]
        
        Returns
        -------
        
        X_lex : numpy-like = [n_samples],
                lexicographic formula for a system with extensions [Nx,Ny,Nz...]: idx(x,y,z,...) = x+(Nx*y)+(Nx*Ny*z)+...            
        """
        
        X_lex=X
        for i in range(len(extensions)-1):
            extensions=np.roll(extensions,1)
            extensions[0]=1
            X_lex=(X_lex*extensions)
        
        return X_lex.sum(axis=len(X.shape)-1)
             
    @staticmethod    
    def _shell_neighbor(shell_grade,point):
        
        """        
        WARNING, WORKS ONLY FOR 2D flat discrete manifolds (usually images)!!! 
        This method computes the first shell-neighbors in 2-d, on a given shell of order shell_grade
        Examples:
        
        
            shell_grade=1  
            
                               A                  E-----F
                               |                  |  |  |  
                            D--x--B      +        |--x--|
                               |                  |  |  |
                               C                  H-----G

                           
                           
                           
            shel_grade=2
            
                               A                   o--I-----L--o           R-----------O
                               |                   |  |  |  |  |           |  |  |  |  |
                               o                   E--|--|--|--F           |--|--|--|--|
                               |                   |  |  |  |  |           |  |  |  |  |
                         D--o--x--o--B    +        |--|--x--|--|    +      |--|--x--|--|
                               |                   |  |  |  |  |           |  |  |  |  |
                               o                   H--|--|--|--G           |--|--|--|--|    
                               |                   |  |  |  |  |           |  |  |  |  |
                               C                   o--N-----M--o           Q-----------P

                        
      
        Parameters
        -----------
        shell_grade : int, shell level
        point : numpy-like, shape  = [1,2]
                    
        Returns
        -------
        res : numpy-like,shape = [n<=8*shell_grade,2]
        array of next neighbors in a given shell of order shell_order
        
        dist : numpy-like, shape = [n<=8*shell_grade]
        array of lattice distances
                    
        The function removes from res those elements with negative coordinates: this explains why n can be smaller than 8*shell_grade
        """
    
        incr_x = np.array([-shell_grade,0],int)
        incr_y = np.array([0,-shell_grade],int)
    
        res=np.array([incr_x,incr_y,-incr_x,-incr_y],int)
    
        for i in range(1,shell_grade):
            temp_x=np.array([incr_x+np.array([0,i]),incr_x+np.array([0,-i])],int)
            temp_y=np.array([incr_y+np.array([i,0]),incr_y+np.array([-i,0])],int)
        
            diag_x=np.append(temp_x,-temp_x,axis=0)
            diag_y=np.append(temp_y,-temp_y,axis=0)            
            diag_xy=np.append(diag_x,diag_y,axis=0)

            res=np.append(res,diag_xy,axis=0)

        
        diag=np.array([incr_x+incr_y,-incr_x+incr_y,-incr_x-incr_y,incr_x-incr_y],int)
        res=np.append(res,diag,axis=0)
        res+=point 
        res=np.delete(res,np.argwhere((res)<0),0)
        dist=np.sqrt(((res-point)*(res-point)).sum(axis=1))

        return res,dist
        
    @staticmethod    
    def _my_kdtree(data,leafsize):
        
        """
        Build a kd-tree for O(n log n) nearest neighbour search. It is employed when
        the option "kd_tree" is applied.
        It is really helpful when the number of missing instances is large, and the serial knn, or
        the exact grid method fail.

        Parameters
        ----------
        
        data: numpy-array-like, 
              shape = [ndata,ndim],
              array containing training instances features.

        leafsize: int,
                  max. number of data points to leave in a leaf (advisable to use always at least 5*k_neighbors)

        Returns
        -------
        
        kd-tree: list,
                 list of tuples representing tree nodes (hyperrectangles) and leaves.        
        """


        sanitycheck(data,np.ndarray)
    
        ndim = data.shape[1]
        ndata = data.shape[0]
        
        
        # find upper and lower bound in feature space
        hrect = np.zeros((2,ndim))
        hrect[0,:] = data.min(axis=0)
        hrect[1,:] = data.max(axis=0)

        # create the root of kd-tree
        idx = np.argsort(data[:,0])
        data = data[idx]
        splitval = data[ndata//2,0]

        left_hrect = hrect.copy()
        right_hrect = hrect.copy()
        left_hrect[1, 0] = splitval
        right_hrect[0, 0] = splitval
        
        tree = [(None, None, left_hrect, right_hrect, None, None)]

        stack = [(data[:ndata//2,:], idx[:ndata//2], 1, 0, True),
                 (data[ndata//2:,:], idx[ndata//2:], 1, 0, False)]
             
        # recursively split data in halves using hyper-rectangles:
        while stack:

            # pop data off stack
            data, didx, depth, parent, leftbranch = stack.pop()
            ndata = data.shape[0]
            nodeptr = len(tree)

            # update parent node

            _didx, _data, _left_hrect, _right_hrect, left, right = tree[parent]

            tree[parent] = (_didx, _data, _left_hrect, _right_hrect, nodeptr, right) if leftbranch \
                else (_didx, _data, _left_hrect, _right_hrect, left, nodeptr)

            # insert node in kd-tree

            # leaf node?
            if ndata <= leafsize:
                _didx = didx.copy()
                _data = data.copy()
                leaf = (_didx, _data, None, None, 0, 0)
                tree.append(leaf)

            # not a leaf, split the data in two      
            else:
                splitdim = depth % ndim
                idx = np.argsort(data[:,splitdim])
                data = data[idx]
                didx = didx[idx]
                nodeptr = len(tree)
                stack.append((data[:ndata//2,:], didx[:ndata//2], depth+1, nodeptr, True))
                stack.append((data[ndata//2:,:], didx[ndata//2:], depth+1, nodeptr, False))
                splitval = data[ndata//2,splitdim]
                if leftbranch:
                    left_hrect = _left_hrect.copy()
                    right_hrect = _left_hrect.copy()
                else:
                    left_hrect = _right_hrect.copy()
                    right_hrect = _right_hrect.copy()
                left_hrect[1, splitdim] = splitval
                right_hrect[0, splitdim] = splitval
                # append node to tree
                tree.append((None, None, left_hrect, right_hrect, None, None))

        return tree

    @staticmethod
    def _intersect(hrect,p):
        
        """
        Method for checking whether or not a given input instance belong to a specific hyper-rectangle.
    
        Parameters
        ----------

        hrect: numpy-array, shape=[2,ndim]
        array containing the opposite vertices of the hyper-rectangle   
        
        p: numpy-array, shape=[ndim]
        array definign the position of the input instance in feature space.
        
        
        Returns
        -------
        
        Bool Value: True if p is in the hyper-rectangle, False otherwise            
        """
        
        maxval = hrect[1,:]
        minval = hrect[0,:]
        return (np.prod((p>=minval))*np.prod((p<=maxval)))
    
    
    def _search_kdtree(self, tree, datapoint, K):   
        """ 
        Method for finding the k nearest neighbours of datapoint in a kdtree 
        It localizes the leaf of the kd-tree that contains the given instance, and then
        applies an exact knn algorithm for determining the k nearest neighbors inside the given leaf.

        
        Parameters
        ----------

        tree: list of tuples
        list of tuples representing tree nodes (hyperrectangles) and leaves.
        
        datapoint: numpy-array like, shape = [ndim]
        the input instance
        
        K: int
        the number of neighbors to be looked for
        """        
        
        stack = [tree[0]]
        #knn = [(numpy.inf, None)]*K
    
        while stack:

            leaf_idx, leaf_data, left_hrect, \
                      right_hrect, left, right = stack.pop()
    
        # leaf
            if leaf_idx is not None:
                sqd = np.sqrt(((leaf_data - datapoint)**2).sum(axis=1)) 
                idx = np.argsort(sqd)
                idx = idx[:K]
    
        # not a leaf
            else:

                # check left branch
                if self._intersect(left_hrect,datapoint):
                    stack.append(tree[left])

                # chech right branch
                if self._intersect(right_hrect, datapoint):
                    stack.append(tree[right])
        return sqd[idx], leaf_idx[idx]    
    
    
    
##############################################################################################
    
    def fit_parallel(self, X_train,X_test):
        """
        Fit the KNN regressor to the instances X_test, using the dataset X_train. This method works in parallel mode, hence it should be used on really big data sets

        Parameters
        ----------

        X_train : numpy.ndarray,
                  shape = [n_samples, n_input_features],
                  training dataset array.
                  
        X_test : numpy.ndarray,
                 shape = [n_test_samples, n_input_features],
                 test dataset array.
        """

        return self._fit_parallel(X_train,X_test)

    def fit(self, X_train,X_test):
        """
        Fit the KNN regressor to the instances X_test, using the dataset X_train, according to the given input flag
        The method works in a quite simple way, depending on the value assigned to the class' attributes.
        
        
        
        Parameters
        ----------
        
        X_train : numpy.ndarray,
                  shape = [n_samples, n_input_features],
                  training dataset array.
                  
        X_test : numpy.ndarray,
                 shape = [n_test_samples, n_input_features],
                 test dataset array.        
        """

        return self._fit(X_train,X_test)

    def predict(self, Y_train):
        """
        Base method for predicting new output values
        
        Parameters
        ----------
        
        Y_train : numpy.ndarray, 
                  shape = [n_samples, n_output_features]
                  array containing test set output values
        """

        return self._predict(Y_train)
  
    def _fit_serial(self, X_train,X_test):
         
        self.neighbors_idx=np.zeros([X_test.shape[0],self.__k],dtype=int)        
        self.neighbors_dist=np.zeros([X_test.shape[0],self.__k])   
    
        diff=(i-X_train for i in X_test)      
        i=0
        for j in diff:
            dist=Euclidean(j)
            mask=np.argsort(dist)
            self.neighbors_idx[i]=mask[:self.__k]
            self.neighbors_dist[i]=dist[mask][:self.__k]            
            i+=1 
            
            
            
    def _fit_grid(self,X_train,X_test,extensions):
        
        sanitycheck(X_train,np.ndarray)
        sanitycheck(X_test,np.ndarray)
        sanitycheck(extensions,np.ndarray)
        WARN=False
        
        if ((len(X_train[0]) != 2) and (len(X_train[-1]) != 2) and (len(X_test[1]) != 2) and (len(X_test[-2]) != 2)):
            raise ValueError("Grid methods actually works only for 2-d isotropic lattices!")
        
        
        X_train_lex=self._lexico(X_train,extensions)       

        self.neighbors_idx=np.zeros([X_test.shape[0],self.__k],dtype=int)        
        self.neighbors_dist=np.zeros([X_test.shape[0],self.__k])   
        nk_temp=np.zeros(self.__k,int)
        nk_dist_temp=np.zeros(self.__k)
        
        m=0
        for i in X_test:
        
            neib=self.__k        
            shell=1
            l=0
            
            while neib > 0:
    
                points,dist=self._shell_neighbor(shell,i)
                
                for j,k in zip(self._lexico(points,extensions),dist):
                
                    where=np.argwhere(X_train_lex==j)

                    if(len(where) and neib > 0):
                        nk_temp[l]=where
                        nk_dist_temp[l]=k
                        l+=1
                        neib-=1
                        
                shell+=1
            if shell==4:
                WARN=True
            self.neighbors_idx[m]=nk_temp
            self.neighbors_dist[m]=nk_dist_temp
            m+=1
        if WARN:    
            print("WARNING: grid method becomes not exact: exceeding the 3rd shell results in approximate definition of neighbors!")
            

    def _fit_kdtree(self,X_train, X_test,leafsize):
        """         
        Find the K nearest neighbours for data points in data,
        using an O(n log n) kd-tree.
        
        Parameters:
        ----------

        X_train : numpy-like, shape = [n_samples, n_input_features]
        X_test : numpy-like, shape = [n_test_samples, n_input_features]
        leafsize: int
        """
        
        K=self.__k
        if leafsize > len(X_train):
            print("Leafsize bigger than dataset size, setting leafsize=len(X_train)")
            leafsize=len(X_train)
            
        # build kdtree
        tree = self._my_kdtree(X_train,leafsize)

        # search kdtree
        self.neighbors_idx=np.zeros([X_test.shape[0],K],dtype=int)        
        self.neighbors_dist=np.zeros([X_test.shape[0],K])   
        m=0
        for i in X_test:
            self.neighbors_dist[m],self.neighbors_idx[m]=self._search_kdtree(tree, i, K)
            m+=1
       
       
    
            
    def _fit(self, X_train,X_test):
         
        sanitycheck(X_train,np.ndarray)
        sanitycheck(X_test,np.ndarray)
       
        #TO_ADD: session fro pd.Series and pd.DataFrame conversion
        
        
        if len(X_train.shape)==1 and len(X_test.shape)==1:
            X=np.column_stack((X_train,np.zeros(len(X_train))))
            Y=np.column_stack((X_test,np.zeros(len(X_test))))
        elif (X_train.shape[1] != X_test.shape[1]):
            raise ValueError("X_train and X_test must have the same number of input features!\n")
        else:
            X=X_train
            Y=X_test
            
        
        if self.method == "classic":
            return self._fit_serial(X,Y)
        elif self.method=="grid":
            return self._fit_grid(X,Y,self.__grid_size)
        elif self.method=="kd-tree":
            return self._fit_kdtree(X,Y,self.__leafsize)
        elif self.__paral and self.method == "kd-tree":
            return self.fit_parallel(X,Y)

     
    