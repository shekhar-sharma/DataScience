
import pandas as pd
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from scipy.spatial.distance import pdist, squareform
import math
from sklearn.neighbors import NearestNeighbors


class ALPCCA:
    def __init__(self,X,Y,k=5):
        self.X=X
        self.Y=Y
        self.k=k
    
    #Calculate mean squared distance
    def meansquareddistances(self,X):
        return squareform(pdist(X,'sqeuclidean'))
    
    #This function is mainly used to minimise the complexity of calculation
    def denom(self,mean_squared_distances):
        var=(mean_squared_distances.shape[0])*(mean_squared_distances.shape[0]-1)
        return mean_squared_distances.sum()*2/var
    
    def similarity_matrix(self,X,k):
        mean_squared_distances=self.meansquareddistances(X)
        t= self.denom(mean_squared_distances)
        k_nbrs = NearestNeighbors(n_neighbors=k).fit(X)
        k_distances, k_indices = k_nbrs.kneighbors(X) 
        similarity=np.zeros([X.shape[0],X.shape[0]])
        for i in range(X.shape[0]):
            for j in range(k):
                similarity[i][k_indices[i][j]]=math.exp(-mean_squared_distances[i][k_indices[i][j]]/t)
        return similarity
    
    def covariance_matrices(self,X,Y,k):
        
        similarity_x=self.similarity_matrix(X,k)
        similarity_y=self.similarity_matrix(Y,k)
        I=np.identity(X.shape[0], dtype = float)
        
        #P=Identity_matrix+similarity_matrix(X)+similarity_matrix(Y)
        P=I+similarity_x+similarity_y
        
        #Covariance(X,Y)=Transform(X).P.Y
        Covariance_xy=np.dot(np.dot(X.T,P),Y)
        
        #Covariance(X,X)=Transform(X).X
        Covariance_xx=np.dot(X.T,X)
        
        #Covariance(Y,Y)=Transform(Y).Y
        Covariance_yy=np.dot(Y.T,Y)
        
        return Covariance_xy,Covariance_xx,Covariance_yy
    
    def fit(self):
        
        #Calculating Covariance Matrix for XY,XX,YY
        Covariance_xy,Covariance_xx,Covariance_yy=self.covariance_matrices(self.X,self.Y,self.k)
        cov_xx_inv=np.linalg.inv(Covariance_xx)
        cov_yy_inv=np.linalg.inv(Covariance_yy)
        
        #Compute Matrix
        H=np.dot(np.dot(Covariance_xx**(-0.5),Covariance_xy),Covariance_yy**(-0.5))
        
        #SVD Decomposition
        U,D,V_T = np.linalg.svd(H, full_matrices=True)
        W_x=np.dot(Covariance_xx**(-0.5),U)
        W_y=np.dot(Covariance_yy**(-0.5),V_T.T)

        return W_x,W_y
    
    def fit_transform(self):
        W_x,W_y=self.fit()
        # Reduced Data
        X_new=np.dot(self.X,W_x)
        Y_new=np.dot(self.Y,W_y)
        return X_new,Y_new
    
    def transform(self,W_x,W_y):
        
        X_new=np.dot(self.X,W_x)
        Y_new=np.dot(self.Y,W_y)
        return X_new,Y_new
        
