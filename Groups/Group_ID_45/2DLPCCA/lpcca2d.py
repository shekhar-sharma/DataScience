import pandas as pd
import numpy as np
from scipy.spatial.distance import pdist, squareform
import math
from sklearn.neighbors import NearestNeighbors
from numpy import linalg
from scipy.linalg import eigh

class LPCCA2D:
    def __init__(self,X,Y,k=5):
        self.X=X
        self.Y=Y
        self.k=k
        
    def cosineSimilarity(self,X):
        return squareform(pdist(X,'cosine'))
    
    def similarity_matrix(self,X,k):
        cosine_Similarity=self.cosineSimilarity(X)
        k_nearest_neighbours = NearestNeighbors(n_neighbors=k,metric='cosine').fit(X)
        k_distances, k_indices = k_nearest_neighbours.kneighbors(X) 
        similarity=np.zeros([X.shape[0],X.shape[0]])
        for i in range(X.shape[0]):
            for j in range(k):
                similarity[i][k_indices[i][j]]=cosine_Similarity[i][k_indices[i][j]]
        return similarity
    
    def SimilarityMatricestoLMatrices(self):
        similarity_matrix_x=self.similarity_matrix(self.X,self.k)
        similarity_matrix_y=self.similarity_matrix(self.Y,self.k)
        
        Sx_dot_Sy=np.multiply(similarity_matrix_x,similarity_matrix_y)
        Sx_dot_Sx=np.multiply(similarity_matrix_x,similarity_matrix_x)
        Sy_dot_Sy=np.multiply(similarity_matrix_y,similarity_matrix_y)
        Sy_dot_Sx=np.multiply(similarity_matrix_y,similarity_matrix_x)
        
        N=self.X.shape[0]
        
        D_xy=np.identity(N, dtype = float)
        D_xx=np.identity(N, dtype = float)
        D_yy=np.identity(N, dtype = float)
        D_yx=np.identity(N, dtype = float)
        
        add_sx_dot_sx=np.sum(Sx_dot_Sx, axis = 1)
        for i in range(N):
            D_xx[i][i]=add_sx_dot_sx[i]

        add_sy_dot_sy=np.sum(Sy_dot_Sy, axis = 1)
        for i in range(N):
            D_yy[i][i]=add_sy_dot_sy[i]

        add_sx_dot_sy=np.sum(Sx_dot_Sy, axis = 1)
        for i in range(N):
            D_xy[i][i]=add_sx_dot_sy[i]

        add_sy_dot_sx=np.sum(Sy_dot_Sx, axis = 1)
        for i in range(N):
            D_yx[i][i]=add_sy_dot_sx[i]

        L_X_Y=D_xy-Sx_dot_Sy
        L_X_X=D_xx-Sx_dot_Sx
        L_Y_X=D_yx-Sy_dot_Sx
        L_Y_Y=D_yy-Sy_dot_Sy

        return L_X_Y,L_Y_X,L_Y_Y,L_X_X
    
    def lpcca2d_Covariance_matrices(self):
        L_X_Y,L_Y_X,L_Y_Y,L_X_X=self.SimilarityMatricestoLMatrices()
        lpcca2d_C_xy=np.dot(np.dot(self.X.T,L_X_Y),self.Y)
        lpcca2d_C_xx=np.dot(np.dot(self.X.T,L_X_X),self.X)
        lpcca2d_C_yy=np.dot(np.dot(self.Y.T,L_Y_Y),self.Y)
        lpcca2d_C_yx=np.dot(np.dot(self.Y.T,L_Y_X),self.X)
        return lpcca2d_C_xy,lpcca2d_C_xx,lpcca2d_C_yy,lpcca2d_C_yx
    
    def fit(self):
        
        lpcca2d_C_xy,lpcca2d_C_xx,lpcca2d_C_yy,lpcca2d_C_yx=self.lpcca2d_Covariance_matrices()
        cov_xx_inv=np.linalg.inv(lpcca2d_C_xx)
        cov_yy_inv=np.linalg.inv(lpcca2d_C_yy)
        
        A1=np.dot(np.dot(np.dot(lpcca2d_C_xy,cov_yy_inv),lpcca2d_C_yx),lpcca2d_C_xx)
        
        A2=np.dot(np.dot(np.dot(lpcca2d_C_yx,cov_xx_inv),lpcca2d_C_xy),lpcca2d_C_yy)
        
        eigenvalsquare,eigenvectorA=eigh(A1,eigvals_only=False)
        eigenvalsquare,eigenvectorB=eigh(A2,eigvals_only=False)
        
        eigenvectorA=np.array(eigenvectorA)
        eigenvectorB=np.array(eigenvectorB)
        
       
        return eigenvectorA,eigenvectorB
    
    def fit_transform(self):
        eigenvectorA,eigenvectorB=self.fit()
        ProjectionVectorX=np.dot(self.X,eigenvectorA)
        ProjectionVectorY=np.dot(self.Y,eigenvectorB)
        return ProjectionVectorX,ProjectionVectorY
    