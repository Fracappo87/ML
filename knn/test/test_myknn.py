"""

Author: Francesco Capponi <capponi.francesco87@gmail.com>
        David Preti       <preti.david@gmail.com>

License: BSD 3 clause

"""

import unittest
import numpy as np
from ..myknn import MyKnn


class MyKnnTest(unittest.TestCase):
    
    def test_myknn_attributes(self):
        
        print("\n testing attributes initialization")
        
        # TEST 1: checking parameters initialization using correct parameters
        my_knn = MyKnn()
        self.assertEqual('Euclidean',my_knn._MyKnn__dist,"a) Checking the correct value for distance choice.")
        self.assertEqual(5,my_knn._MyKnn__k,"b) Checking the correct number of neighbors.")
        self.assertEqual(100,my_knn._MyKnn__leafsize,"c) Checking the correct leafsize value.")
        self.assertEqual(np.zeros(1),my_knn._MyKnn__grid_size,"d) Checking the correct grid value.")
        self.assertFalse(my_knn._MyKnn__paral,"e) Checking the correct parallelization value.")
        
        # TEST 2: checking correct exception raising when wrong input parameters are given
        self.assertRaises(ValueError,MyKnn,distance=4)          
        self.assertRaises(ValueError,MyKnn,distance='gababubu')
        self.assertRaises(ValueError,MyKnn,n_neighbors=.1)
        self.assertRaises(ValueError,MyKnn,n_neighbors=-11)
        self.assertRaises(ValueError,MyKnn,leafsize=.1)
        self.assertRaises(ValueError,MyKnn,leafsize=-11)    
        self.assertRaises(ValueError,MyKnn,grid_size='a')
        self.assertRaises(ValueError,MyKnn,parallelize='a')
        
        
    def test_myknn_lexico(self):
        
        print("\n testing the lexico method")
        
        my_knn = MyKnn(distance="grid",n_neighbors=1,parallelize=False)
          
        exts=[5,5]
        lex_index=np.arange(25,dtype=int)
        sequence=[(x, y) for x in range(exts[0]) for y in range(exts[1])]        
        extensions=np.array(exts,dtype=int)
        
        k=0
        for j,i in sequence:
            point=np.array([i,j],dtype=int)
            self.assertEqual(lex_index[k],my_knn._lexico(point,extensions),"a) Checking lexicographic function for 2D, [5,5]")
            k+=1
            
        exts=[2,4,2]
        lex_index=np.arange(16,dtype=int)
        sequence=[(x,y,z) for x in range(exts[0]) for y in range(exts[1]) for z in range(exts[2])]        
        extensions=np.array(exts,dtype=int)
        
        l=0
        for k,j,i in sequence:
            point=np.array([i,j,k],dtype=int)
            self.assertEqual(lex_index[l],my_knn._lexico(point,extensions),"b) Checking lexicographic function for 3D,[2,4,2]")
            l+=1
            
        exts=[5,4,3,3,3]
        lex_index=np.arange(540,dtype=int)
        sequence=[(s,w,z,y,x) for s in range(exts[0]) for w in range(exts[1]) for z in range(exts[2]) for y in range(exts[3]) for x in range(exts[4])]        
        exts=[3,3,3,4,5]        
        extensions=np.array(exts,dtype=int)
        n=0
        
        for m,l,k,j,i in sequence:
            point=np.array([i,j,k,l,m],dtype=int)
            self.assertEqual(lex_index[n],my_knn._lexico(point,extensions),"c) Checking lexicographic function for 5D,[3,3,3,4,5]")
            n+=1
        
        
    def test_myknn_shell_neighbor(self):
        
        print("\n testing the shell_neighbor method")
        my_knn = MyKnn(distance="grid",n_neighbors=1,parallelize=False)
 
        shell_grade=1
        shello=np.array([[shell_grade,0],[0,shell_grade],[shell_grade,shell_grade]],dtype=int)
        disto=np.array([1,1,np.sqrt(2)]) 
        
        neib,dists=my_knn._shell_neighbor(1,np.array([0,0], dtype=int))
        self.assertEqual(0,(neib-shello).sum(),"a) Checking shell neighbors, shell 1, point [0,0].")
        self.assertEqual(0,(dists-disto).sum(),"b) Checking shell neighbors distances,, shell 1, point [0,0].")
                         
        shell_grade=2
        shello=np.array([[shell_grade,0],[0,shell_grade],[shell_grade,1],[1,shell_grade],[shell_grade,shell_grade]],dtype=int)
        disto=np.array([shell_grade,shell_grade,np.sqrt(5),np.sqrt(5),np.sqrt(8)])
        
        neib,dists=my_knn._shell_neighbor(2,np.array([0,0], dtype=int))
        self.assertEqual(0,(neib-shello).sum(),"c) Checking shell neighbors, shell 2, point [0,0].")
        self.assertEqual(0,(dists-disto).sum(),"d) Checking shell neighbors distances,, shell 2, point [0,0].")
        
        shell_grade=3
        shello=np.array([[shell_grade,0],[0,shell_grade],[shell_grade,1],[1,shell_grade],[shell_grade,2],[2,shell_grade],[shell_grade,shell_grade]],dtype=int)        
        disto=np.array([shell_grade,shell_grade,np.sqrt(10),np.sqrt(10),np.sqrt(13),np.sqrt(13),np.sqrt(18)])        

        neib,dists=my_knn._shell_neighbor(3,np.array([0,0], dtype=int))
        self.assertEqual(0,(neib-shello).sum(),"e) Checking shell neighbors, shell 3, point [0,0]")
        self.assertEqual(0,(dists-disto).sum(),"f) Checking shell neighbors distances,, shell 3, point [0,0].")
            
        
    def test_myknn_intersect(self):
        
        print("\n testing the intersect method")
        my_knn = MyKnn(distance="kd-tree",n_neighbors=1,parallelize=False)
        
        rectangle=np.array([[0,0],[1,1]])
        p=np.array([.5,.5])
        q=np.array([0,2])
        self.assertTrue(my_knn._intersect(rectangle,p), "a) Checking correctness of the intersect method (expected output=TRUE)")
        self.assertFalse(my_knn._intersect(rectangle,q),"b) Checking correctness of the intersect method (expected output=FALSE)")
        
        
        
    def test_myknn_my_kdtree(self):

        print("\n testing my_kdtree method")
        X=np.array([[0,0],[0,1],[1,0],[1,1]])
        Y=np.array([1,2,3,4])
        
        my_knn = MyKnn(distance="kd-tree",n_neighbors=1,parallelize=False)
        tree=my_knn._my_kdtree(X,X.shape[1])
        for k in tree: 
            if k[0] is not None:
                self.assertEqual((Y[k[0]][1]-Y[k[0]][0]),1,"a) Checking kdtree generator")
        
        tree=my_knn._my_kdtree(X,X.shape[1]//2)
        test_vec=np.zeros(4)
        compare_list=[]
        m=0
        for k in tree: 
            if k[0] is not None:
                test_vec[m]=Y[k[0]]
                compare_list.append(k[0])
                m+=1
                
        np.sort(test_vec)
        self.assertEqual((Y-test_vec).sum(),0.,"b) Checking kdtree generator")
        for k in range(4):
            self.assertEqual(Y[compare_list[3-k]],Y[k],"c) Checking kdtree generator")
            
        
        
    def test_myknn_search_kdtree(self):   
        
        print("\n testing the search kd-tree method")
        X=np.array([[0,0],[0,1],[1,0],[1,1]])
        Y=np.array([1,2,3,4])
        
        my_knn = MyKnn(distance="kd-tree",n_neighbors=1,parallelize=False)
        tree=my_knn._my_kdtree(X,X.shape[1])
        sqd,idx=my_knn._search_kdtree(tree,X[0],3)
        self.assertEqual((Y[idx]-Y[:2]).sum(),0,"a) Checking kdtree search method")
        
    
    def test_myknn_fit_serial(self):
        
        print("\n testing the fit_serial method")
        X=np.arange(1,11)
        X_test=np.array([0])
        
        for i in range(1,X.shape[0]):
            my_knn = MyKnn(distance="Euclidean",n_neighbors=i,parallelize=False)
            my_knn.fit(X,X_test)
            neib,dist=np.arange(1,i+1),np.arange(1,i+1)
            self.assertEqual((neib-X[my_knn.neighbors_idx]).sum(),0,"a) Checking knn idx with 1D mock example.")
            self.assertEqual((dist-my_knn.neighbors_dist).sum(),0,"b) Checking knn distance vector with 1D mock example.")
        
        X=np.array([[1,0],[-1,2],[0,-2],[0,3]])
        X_test=np.array([[0,0]])
        
        # 1 neighbor
        my_knn = MyKnn(distance="Euclidean",n_neighbors=1,parallelize=False)
        my_knn.fit(X,X_test)
        self.assertEqual((X[0]-X[my_knn.neighbors_idx]).sum(),0,"c) Checking 1nn neighbors idx with 2D mock example.")
        self.assertEqual((1-my_knn.neighbors_dist).sum(),0,"d) Checking 1nn distance vector with 2D mock example.")
        
        # 2 neighbors
        my_knn = MyKnn(distance="Euclidean",n_neighbors=2,parallelize=False)
        my_knn.fit(X,X_test)
        self.assertEqual((X[0::2]-X[my_knn.neighbors_idx]).sum(),0,"e) Checking 2nn neighbors idx with 2D mock example.")
        self.assertEqual((np.array([1,2])-my_knn.neighbors_dist).sum(),0,"f) Checking 2nn distance vector with 2D mock example.")
        
        # 3 neighbors
        my_knn = MyKnn(distance="Euclidean",n_neighbors=3,parallelize=False)
        my_knn.fit(X,X_test)
        self.assertEqual((X[np.array([0,2,1])]-X[my_knn.neighbors_idx]).sum(),0,"g) Checking 3nn neighbors idx with 2D mock example.")
        self.assertEqual((np.array([1,2,np.sqrt(5)])-my_knn.neighbors_dist).sum(),0,"h) Checking 3nn distance vector with 2D mock example.")
        
        # 4 neighbors
        my_knn = MyKnn(distance="Euclidean",n_neighbors=4,parallelize=False)
        my_knn.fit(X,X_test)
        self.assertEqual((X[np.array([0,2,1,3])]-X[my_knn.neighbors_idx]).sum(),0,"i) Checking 3nn neighbors idx with 2D mock example.")
        self.assertEqual((np.array([1,2,np.sqrt(5),3])-my_knn.neighbors_dist).sum(),0,"j) Checking 3nn distance vector with 2D mock example.")
        
        X=np.array([[1,0,0,0],[-1,2,1,0],[0,1,3,-2],[0,-3,1,3]])
        X_test=np.array([[0,0,0,0]])
        
        # 1 neighbor
        my_knn = MyKnn(distance="Euclidean",n_neighbors=1,parallelize=False)
        my_knn.fit(X,X_test)
        self.assertEqual((X[0]-X[my_knn.neighbors_idx]).sum(),0,"k) Checking 1nn neighbors idx with 4D mock example.")
        self.assertEqual((1-my_knn.neighbors_dist).sum(),0,"l) Checking 1nn distance vector with 4D mock example.")
        
        # 2 neighbors
        my_knn = MyKnn(distance="Euclidean",n_neighbors=2,parallelize=False)
        my_knn.fit(X,X_test)
        self.assertEqual((X[:2]-X[my_knn.neighbors_idx]).sum(),0,"m) Checking 2nn neighbors idx with 4D mock example.")
        self.assertEqual((np.array([1,np.sqrt(6.)])-my_knn.neighbors_dist).sum(),0,"n) Checking 2nn distance vector with 4D mock example.")
        
        # 3 neighbors
        my_knn = MyKnn(distance="Euclidean",n_neighbors=3,parallelize=False)
        my_knn.fit(X,X_test)
        self.assertEqual((X[:3]-X[my_knn.neighbors_idx]).sum(),0,"o) Checking 3nn neighbors idx with 4D mock example.")
        self.assertEqual((np.array([1,np.sqrt(6.),np.sqrt(14.)])-my_knn.neighbors_dist).sum(),0,"p) Checking 3nn distance vector with 4D mock example.")
        
        # 4 neighbors
        my_knn = MyKnn(distance="Euclidean",n_neighbors=4,parallelize=False)
        my_knn.fit(X,X_test)
        self.assertEqual((X-X[my_knn.neighbors_idx]).sum(),0,"q) Checking 3nn neighbors idx with 4D mock example.")
        self.assertEqual((np.array([1,np.sqrt(6.),np.sqrt(14.),np.sqrt(19.)])-my_knn.neighbors_dist).sum(),0,"r) Checking 3nn distance vector with 4D mock example.")
        
        
    def test_myknn_fit_grid(self):
        
        print("\n testing the fit_grid method")
        
        X=np.array([[1,0],[1,2],[0,2],[0,3]])
        X_test=np.array([[0,0]])
        extensions=np.array([4,2])
        # 1 neighbor
        my_knn = MyKnn(distance="grid",n_neighbors=1,grid_size=extensions,parallelize=False)
        my_knn.fit(X,X_test)
        self.assertEqual((X[0]-X[my_knn.neighbors_idx]).sum(),0,"a) Checking 1nn neighbors idx with 2D mock example.")
        self.assertEqual((1-my_knn.neighbors_dist).sum(),0,"b) Checking 1nn distance vector with 2D mock example.")
        
        # 2 neighbors
        my_knn = MyKnn(distance="grid",n_neighbors=2,grid_size=extensions,parallelize=False)
        my_knn.fit(X,X_test)
        self.assertEqual((X[0::2]-X[my_knn.neighbors_idx]).sum(),0,"c) Checking 2nn neighbors idx with 2D mock example.")
        self.assertEqual((np.array([1,2])-my_knn.neighbors_dist).sum(),0,"d) Checking 2nn distance vector with 2D mock example.")
        
        # 3 neighbors
        my_knn = MyKnn(distance="grid",n_neighbors=3,grid_size=extensions,parallelize=False)
        my_knn.fit(X,X_test)
        self.assertEqual((X[np.array([0,2,1])]-X[my_knn.neighbors_idx]).sum(),0,"e) Checking 3nn neighbors idx with 2D mock example.")
        self.assertEqual((np.array([1,2,np.sqrt(5)])-my_knn.neighbors_dist).sum(),0,"f) Checking 3nn distance vector with 2D mock example.")
        
        # 4 neighbors
        my_knn = MyKnn(distance="grid",n_neighbors=4,grid_size=extensions,parallelize=False)
        my_knn.fit(X,X_test)
        self.assertEqual((X[np.array([0,2,1,3])]-X[my_knn.neighbors_idx]).sum(),0,"g) Checking 3nn neighbors idx with 2D mock example.")
        self.assertEqual((np.array([1,2,np.sqrt(5),3])-my_knn.neighbors_dist).sum(),0,"h) Checking 3nn distance vector with 2D mock example.")
        
    #To be added: test for the regression with kdtree
    #def test_myknn_fit_kd_tree(self):
 
        
if __name__ == '__main__':
    unittest.main()
