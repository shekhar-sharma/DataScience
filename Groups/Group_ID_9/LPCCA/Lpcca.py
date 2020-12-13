import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from scipy.spatial.distance import pdist, squareform
import math
from sklearn.neighbors import NearestNeighbors
import cmath
class LPCCA:
    def __init__(self,X,Y,k=5):
        self.X=X
        self.Y=Y
        self.k=k
        
    #Calculate mean squared distance
    def meansquareddistances(self,X):
        return squareform(pdist(X,'sqeuclidean'))
    
    def demon(self,mean_squared_distances):
        var=(mean_squared_distances.shape[0])*(mean_squared_distances.shape[0]-1)
        return mean_squared_distances.sum()*2/var
    
    def similarity_matrix(self,X,k):
        mean_squared_distances=self.meansquareddistances(X)
        t= self.demon(mean_squared_distances)
        k_nbrs = NearestNeighbors(n_neighbors=k).fit(X)
        k_distances, k_indices = k_nbrs.kneighbors(X) 
        similarity=np.zeros([X.shape[0],X.shape[0]])
        for i in range(X.shape[0]):
            for j in range(k):
                similarity[i][k_indices[i][j]]=math.exp(-mean_squared_distances[i][k_indices[i][j]]/t)
        return similarity
    
    def SdotS(self):
        similarity_x=self.similarity_matrix(self.X,self.k)
        similarity_y=self.similarity_matrix(self.Y,self.k)
        Sx_dot_Sy=np.multiply(similarity_x,similarity_y)
        Sx_dot_Sx=np.multiply(similarity_x,similarity_x)
        Sy_dot_Sy=np.multiply(similarity_y,similarity_y)
        Sy_dot_Sx=np.multiply(similarity_y,similarity_x)
        N=self.X.shape[0]
        D_xy=np.identity(N, dtype = float)
        D_xx=np.identity(N, dtype = float)
        D_yy=np.identity(N, dtype = float)
        D_yx=np.identity(N, dtype = float)

        add_sx_dot_sy=np.sum(Sx_dot_Sy, axis = 1)
        for i in range(N):
            D_xy[i][i]=add_sx_dot_sy[i]


        add_sx_dot_sx=np.sum(Sx_dot_Sx, axis = 1)
        for i in range(N):
            D_xx[i][i]=add_sx_dot_sx[i]


        add_sy_dot_sy=np.sum(Sy_dot_Sy, axis = 1)
        for i in range(N):
            D_yy[i][i]=add_sy_dot_sy[i]


        add_sy_dot_sx=np.sum(Sy_dot_Sx, axis = 1)
        for i in range(N):
            D_yx[i][i]=add_sy_dot_sx[i]


        S_X_Y=D_xy-Sx_dot_Sy
        S_X_X=D_xx-Sx_dot_Sx
        S_Y_X=D_yx-Sy_dot_Sx
        S_Y_Y=D_yy-Sy_dot_Sy


        return S_X_Y,S_Y_X,S_Y_Y,S_X_X
    
    def lpcca_Covariance_matrices(self):
        S_X_Y,S_Y_X,S_Y_Y,S_X_X=self.SdotS()
        lpcca_C_xy=np.dot(np.dot(self.X.T,S_X_Y),self.Y)
        lpcca_C_xx=np.dot(np.dot(self.X.T,S_X_X),self.X)
        lpcca_C_yy=np.dot(np.dot(self.Y.T,S_Y_Y),self.Y)
        lpcca_C_yx=np.dot(np.dot(self.Y.T,S_Y_X),self.X)
        return lpcca_C_xy,lpcca_C_xx,lpcca_C_yy,lpcca_C_yx
    
    def fit(self):
        
        lpcca_C_xy,lpcca_C_xx,lpcca_C_yy,lpcca_C_yx=self.lpcca_Covariance_matrices()
        cov_xx_inv=np.linalg.inv(lpcca_C_xx)
        cov_yy_inv=np.linalg.inv(lpcca_C_yy)
        
        H=np.dot(np.dot(np.lib.scimath.sqrt(cov_xx_inv),lpcca_C_xy),(np.lib.scimath.sqrt(cov_yy_inv)))
        U,D,V_T = np.linalg.svd(H, full_matrices=True)
        
        W_x=np.dot(np.lib.scimath.sqrt(cov_xx_inv),U)
        W_y=np.dot(np.lib.scimath.sqrt(cov_yy_inv),V_T.T)
        W_x=np.array(W_x)
        W_y=np.array(W_y)
        n_x=W_x.shape
        n_y=W_y.shape
        return W_x.real,W_y.real
    
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