from scipy import misc
import matplotlib.pyplot as plt
import numpy as np
from knn.myknnregressor import MyKnnRegressor
from crossvalidation.mycrossvalidation import MyCrossValidation 

face = misc.imread('examples/knn_example/photo.jpg')   
plt.imshow(face)
plt.show()
facetemp= misc.imread('examples/knn_example/photo.jpg')
print(face.shape)



#number of random numbers to be extracted

perc=1.4
ndef=int((face.shape[0]-1)*(face.shape[1]-1)*perc)



#generate random 

randX=np.random.randint(0, face.shape[0]-1, size=ndef)
randY=np.random.randint(0, face.shape[1]-1, size=ndef)



#shutting down the pixels, plotting the new picture

for i,j in zip(randX,randY):
    face[i,j]=np.zeros(face.shape[2])
plt.imshow(face)
plt.show()



# This is a really useful function for rapidly eliminate duplicate rows
 
def row_cancel(A):     
    x = np.random.rand(A.shape[len(A.shape)-1])
    y = A.dot(x)
    unique, index = np.unique(y, return_index=True)
    return A[index]


    
# Separating shut down from switched on pixels
    
Ytemp=np.argwhere(face==[0,0,0])[:,[0,1]]
Xtemp=np.argwhere(face!=[0,0,0])[:,[0,1]]



# Creating training and test set

X_test=row_cancel(Ytemp)
X_train=row_cancel(Xtemp)

Y_train=np.zeros([len(X_train),3])

k=0
for i in X_train:
    Y_train[k]=face[i[0],i[1]]
    k+=1
        

# Tuning the number of neighbors by doing k-fold cross validation

nfolds=5
mycv = MyCrossValidation(kfolds=5,reshuffle=True)
test_values=range(1,10)
Rsquares=np.zeros((len(test_values),nfolds))


for k in test_values:
    my_knn_fold=MyKnnRegressor(method="kd-tree",criterion="weighted",n_neighbors=k,leafsize=100,parallelize=False)
    mycv.cross_val(X_train,Y_train,my_knn_fold)  
    Rsquares[k-1,:]=mycv.R_squared_collection
    
# Plotting the tuning procedure
    
import matplotlib.pyplot as plt

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

my_knn=MyKnnRegressor(method="kd-tree",criterion="weighted",n_neighbors=optimal_k,leafsize=100,parallelize=False)
my_knn.fit(X_train,X_test)
my_knn.predict(Y_train)



# Using predicted value for filling shut down pixels

proba1=np.zeros([len(X_test),3])
proba2=np.zeros([len(X_test),3]) 

k=0 
for i,j in zip(X_test,my_knn.prediction):
        face[i[0],i[1]]=j                
        proba2[k]=facetemp[i[0],i[1]]
        proba1[k]=proba2[k]-j
        k+=1
plt.imshow(face)
plt.show()



