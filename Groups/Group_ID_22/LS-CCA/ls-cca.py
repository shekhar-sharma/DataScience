#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Dec 10 00:29:25 2020

@author: sashbros
"""
#imported all libraries
import numpy as np
from numpy import linalg as la
from numpy.lib import scimath as smath
from sklearn.preprocessing import StandardScaler
from sklearn import decomposition
import pandas as pd

class LSCCA:
    
    # constructor
    def __init__(self, views):
        self.views = views
    
    # normalizing the data for all M views
    def normalize(self, M):
        for i in range(self.views):
            M[i] = StandardScaler().fit_transform(M[i])

        return M
    
    # reducing the dimensions, i.e., features, so all M views have same number of features(columns)
    def reduce_dimensions(self, M, min_cols):
        for i in range(self.views):
            if M[i].shape[1] > min_cols:
                pca = decomposition.PCA(n_components = min_cols)
                reduced_mtx = pca.fit_transform(M[i])
                M[i] = reduced_mtx
        
        return M
    
    # function that does all the computations for retrieving weight matrices
    def fit(self, M):
        
        # knowing minimum features(columns) in all M views
        min_cols = 99999
        for i in range(self.views):
            if M[i].shape[1] < min_cols:
                min_cols = M[i].shape[1]
        
        M = self.reduce_dimensions(M, min_cols)
        
        M = self.normalize(M)
        
        # R matrix will have all the covariance matrices stored
        rmat = [[0 for _ in range(self.views)] for _ in range(self.views)]
        
        for i in range(self.views):
            for j in range(self.views):
                rmat[i][j] = (1/(self.views-1)) * np.dot(np.transpose(M[i]), M[j]) # 1/(M-1) is the formula where M = views
        
        # W list will have all the weight matrices stored
        W = [0 for _ in range(self.views)]
        
        for i in range(self.views):
            W[i] = rmat[i][i]
            
            # W[i] = rmat[i][i]**(-0.5)
        
        for i in range(self.views-1):
            for j in range(i+1, self.views):
                # T matrix is the matrix on which we have to perform the SVD
                tmat = np.dot(rmat[i][i], rmat[i][j])
                tmat = np.dot(tmat, rmat[j][j])
                
                # tmat = np.dot(rmat[i][i]**(-0.5), rmat[i][j])
                # tmat = np.dot(tmat, rmat[j][j]**(-0.5))
                
                u, s, vh = la.svd(tmat)
                
                # computing and updating the weight matrices
            
                W[i] = np.dot(W[i], u)
                W[j] = np.dot(W[j], vh.transpose())

        return W # returning the List of all weight matrices
        

# Declaring and Initializing all M views
m1 = pd.DataFrame([[0., 0., 1.], [1.,0.,0.], [2.,2.,2.], [3.,5.,4.]])
m2 = pd.DataFrame([[0.1, -0.2], [0.9, 1.1], [6.2, 5.9], [11.9, 12.3]])
m3 = pd.DataFrame([[0.3, -0.4], [1.9, 0.9], [4.3, 3.2], [8.4, 7.1]])

# taking all M views in a List
M = []
M.append(m1)
M.append(m2)
M.append(m3)

# Calling the constructor of LSCCA and generating all the weight matrices W
lscca = LSCCA(3) # 3 is the number of views
W = lscca.fit(M)

# Printing all the weight matrices
count = 1
for i in W:
    print("w" + str(count) + ":")
    print(i, end="\n\n")
    count+=1
    
    
