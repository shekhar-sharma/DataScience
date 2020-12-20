import pandas as pd
import numpy as np
import math
from sklearn import decomposition
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import NearestNeighbors

def _center_scale_xy(x, y, scale=True):
    
    #Normalizes the data using StandardScaler
    
    if scale == True:
        return StandardScaler().fit_transform(x),StandardScaler().fit_transform(y)
    return x,y

class SDCCA :

    """
    Similarity Distance based CCA (SDCCA)
    
    Parameters
    ----------
    scale (boolean) (default true) : whether to scale the data or not    
    k (integer) (default 2) : number of neighbours to consider in K-Means algorithm.

    Attributes
    ----------
    x : first data set of dimension n x p with n samples and p features.
    y : second data set of dimension n x q with n samples and q features.    
    wx , wy : final projection vectors of two views of dimensions p x p and q x q respectively.
    n : number of samples in datasets
    p : number of features in x dataset
    q : number of features in y dataset
    
    """
    
    def __init__(self,scale = True,k=2):
        self.scale = scale
        self.k = k
        
    def __make_dimensionlity_same(self,x,y,p,q) :
        
        #Uses PCA to make the number of features same in both the dataset
        
        pca = decomposition.PCA(n_components=min(p,q))
        if p < q : 
            pc = pca.fit_transform(y)
            return x,pc,p,p
        else : 
            pc = pca.fit_transform(x)
            return pc,y,q,q
        
    def __bhattacharya_similarity_coeff(self,x,y) : 
        
        #Calculates the Bhattacharya similarity Coefficient

        n = x.shape[0]
        p = x.shape[1]
        q = y.shape[1]

        if p != q :
            x,y,p,q = self.__make_dimensionlity_same(x,y,p,q)

        sxy = np.zeros((n,n))
        for i in range(n) :
            for j in range(n) : 
                sxy[i][j] = abs(np.dot(x[i],y[j]))**0.5
        return sxy
    
    def __nearest_neighbor(self,x,y) :
        
        # Uses K-Means Algorithm to find the K neighbours and correspnding distances for all the samples.
        
        neigh = NearestNeighbors(n_neighbors=self.k)
        neigh.fit(x)
        x_dist, x_neigh = neigh.kneighbors(x,return_distance = True)
        
        neigh.fit(y)
        y_dist, y_neigh = neigh.kneighbors(y,return_distance = True)
        
        return x_dist,x_neigh,y_dist,y_neigh
            
    def fit(self,x,y) :

        """
        Fit the model from the data in x and the labels in y and finds the projection vectors
        
        Parameters
        ----------
        x : numpy array-like, shape (n x p)
            where n is the number of samples, and p is the number of features.
        y : numpy array-like, shape (n x q)
            where n is the number of samples, and q is the number of features.
            
        Variables
        ----------
        x_neigh , y_neigh : indices of k neighbours of each of the n samples of x and y respectively computed using K-Means algorithm.
        x_dist, y_dist : dist from each of the k neighbours of all n samples of x and y respectively.

        sx, sy : local manifold information between samples.
        sxy : it is dissimilarity between samples using 1-s~xy.
        s~xy : it is similarity between views x and y calculated using Bhattacharya similarity coefficient.
        
        Returns
        -------
        wx , wy : Projection vectors of dimensions p x p and q x q respectively.
      """
        # n is the number of samples and p and q are features of x and y respectively.
        self.n = x.shape[0]
        self.p = x.shape[1]
        self.q = y.shape[1]
        
        self.x = x
        self.y = y
                    
        # normalize the data
        self.x, self.y = _center_scale_xy(self.x, self.y, self.scale)
        
        x_dist,x_neigh,y_dist,y_neigh = self.__nearest_neighbor(self.x,self.y)
        
        tx = 0
        ty = 0
        sx = np.zeros((self.n,self.n))
        sy = np.zeros((self.n,self.n))
        
        for i in range(self.n) : 
            for j in range(self.n) : 
                tx += (2*(np.linalg.norm(self.x[i]-self.x[j],2)**2))/(self.n*(self.n-1))
        
        for i in range(self.n) : 
            for j in range(self.n) : 
                ty += (2*(np.linalg.norm(self.y[i]-self.y[j],2)**2))/(self.n*(self.n-1))
                
        for i in range(self.n) : 
            for j in range(self.k) : 
                sx[i][x_neigh[i][j]] = np.exp(-(x_dist[i][j]**2)/tx)

        for i in range(self.n) : 
            for j in range(self.k) : 
                sy[i][y_neigh[i][j]] = np.exp(-(y_dist[i][j]**2)/ty)
                                    
        sxy = 1-self.__bhattacharya_similarity_coeff(self.x,self.y)
        
        s = np.identity(self.n)+sx+sy+sxy

        cxy = np.dot(np.dot(self.x.transpose(),s),self.y) 
        cxx = np.dot(self.x.transpose(),self.x)
        cyy = np.dot(self.y.transpose(),self.y,)
        
        temp = np.dot(cxx, cxy)
        h = np.dot(temp,cyy)

        # Use SVD(singular value decomposition) to find the U and V.T matrix
        u,d,vh = np.linalg.svd(h)

        #Calculate projection vectors
        self.wx = np.dot(cxx, u)
        self.wy = np.dot(cyy, vh.transpose())
        
        return self.wx,self.wy
    
    def fit_transform(self, x,y) :

        """
        Applies the projection vectors on the dataset
        ----------
        x : numpy array-like, shape (n x p)
            where n is the number of samples, and p is the number of features.
        y : numpy array-like, shape (n x q)
            where n is the number of samples, and q is the number of features.
        
        Returns
        -------
        x_new , y_new : Projected views
      """
        wx,wy = self.fit(x,y)
        x_new = np.dot(x,wx.T)
        y_new = np.dot(y,wy.T)
        return x_new,y_new