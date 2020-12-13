#!/usr/bin/env python
# coding: utf-8

# ## KCCA (kernel canonical coorelation analysis)

# In[1]:


#Importing All required packages
import numpy as np
import pandas as pd
from numpy import dot
from sklearn.preprocessing import StandardScaler
from scipy.spatial.distance import pdist, squareform
from sklearn.preprocessing import KernelCenterer
from sklearn.metrics.pairwise import pairwise_kernels

class KCCA:
    
    def normalisation(self, X):
        return StandardScaler().fit_transform(X)
    
    #manual kernel rbf computation
    def gaussianRbfKernel(self, X):
        pairwise_dists = pdist(X, 'seuclidean')
        distance_matrix=squareform(pairwise_dists)
        K = np.exp( -distance_matrix ** 2 /  2)
        return K

    #Computing hilbert transform
    def hilbertTransform(self, X):
        
        k = pairwise_kernels(X,metric = 'rbf')
        phi = KernelCenterer().fit_transform(k)
        return phi
    
    def fit(self, inputX, inputY):
    
        #normalizing data mean=0 standard deviation=1
        X = self.normalisation(inputX)
        Y = self.normalisation(inputY)

        #transforming data to higher dimension
        phiX=self.hilbertTransform(X)
        phiY=self.hilbertTransform(Y)
        
        #generating kernel metrices
        kx=self.gaussianRbfKernel(X)
        ky=self.gaussianRbfKernel(Y)
        
        #Generating Matrix paramters
        r=1e-5 #regularization Parameter
        kxy = dot(kx,ky)
        kxx = dot(kx,kx) + r*kx
        kyy = dot(ky,ky) + r*ky

        #Matrix computation for computing projections a,b by langrange's equation using SVD
        htemp = dot(kxx**-0.5, kxy)
        h = dot(htemp, kyy**-0.5)
        U,D,V = np.linalg.svd(h, full_matrices=True)

        #resultant eigen vectors
        a = dot(kxx**-0.5, U)
        b = dot(kyy**-0.5, V.T)

        #resultant projections
        wx = dot(phiX, a)
        wy = dot(phiY, b)
        
        return wx,wy
    
    #Computing transformed matrix
    def fit_transform(self,inputX, inputY):
        wx,wy=self.fit(inputX, inputY)
        Xnew = dot(wx.T, inputX)
        Ynew = dot(wy.T, inputY)
        return Xnew,Ynew


# In[ ]:




