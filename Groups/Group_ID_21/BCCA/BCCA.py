"""
Created on Sun Dec 13 23:43:01 2020

@author: Prajal Badjatya
"""

import numpy as np
from scipy.linalg import eig
import h5py
import math


class BCCA:
    
    def __init__(self):
        self.length = 0
        self.Caa = [[]]
        self.Cab = [[]]
        self.Cba = [[]]
        self.Cbb = [[]]
        self.u = [[]]
        self.s = [[]]
        self.v = [[]]
        
        
        
    # returns normal multivariate distribution
    def multivariate_normal(x, d, mean, covariance):
        x_m = x - mean
        return (1. / (np.sqrt((2 * np.pi)**d * np.linalg.det(covariance))) * 
            np.exp(-(np.linalg.solve(covariance, x_m).T.dot(x_m)) / 2))
    
    def fit_transform(self, *x_list):
        self.fit(x_list)
        Wa,Wb = self.transform(x_list)
        self.Caa = [[]]
        self.Cab = [[]]
        self.Cba = [[]]
        self.Cbb = [[]]
        self.u = [[]]
        self.s = [[]]
        self.v = [[]]
        
        print(Wa)
        print(Wb)
        
    def normalize(matrix):
        m = np.mean(matrix,axis=0)
        matrix = matrix-m
        return matrix
    
    
    def transform(self,*x_list):
        length = len(x_list)
        wx=np.dot(np.lib.scimath.sqrt(self.Caa**(-1)),self.u)
        wy=np.dot(np.lib.scimath.sqrt(self.Cbb**(-1)),self.v.T)
        
        return wx,wy
    

    def fit(self, *x_list):
        
        length = len(x_list)
        # normalizing
        normalised_list = [self.normalize(x) for x in x_list]
        
        Xa = normalised_list[0][0]
        Xb = normalised_list[0][1]
        

        if Xa.shape[0]==Xb.shape[0]:
            n = Xa.shape[0]
        else:
            n = min(Xa.shape[0],Xb.shape[0])
        
        
        # Sample covariance matrix Caa between variable column vectors in Xa and Xb is Cab
        Cab = np.dot(Xa,Xb)
        Cab = Cab/(n-1)
        Cba = np.dot(Xb,Xa)
        Cba = Cba/(n-1)
        Caa = np.dot(Xa,Xa)
        Caa = Caa/(n-1)
        Cbb = np.dot(Xb,Xb)
        Cbb = Cbb/(n-1)
        
        #The joint covariance matrix is
        covariance_matrix = np.array([[Caa,Cab],[Cba,Cbb]])
        
        P = np.dot(np.dot(np.lib.scimath.sqrt(Caa*(-1)),Cab),np.lib.scimath.sqrt(Cbb*(-1)))
        u,s,v = np.linalg.svd(P)
        
        self.length = length
        self.Caa = Caa
        self.Cbb = Cbb
        self.Cxy = Cxy
        self.u = u
        self.v = v
        self.s =s
        
        self.fit(x_list)
        Wa,Wb = self.transform(x_list)
        
        #d=n
        #Identity matrix of d*d
        Id = np.identity(n)
        
        # Zero matrix 
        Zero = np.zeros([n,n], dtype=int)
        
        #Calculating latent variable y by normal multivariate distribution
        # y = multivariate_normal()
        
        #Assuming Ma and Mb random matrices for now
        Ma = np.rand(n,n)
        Mb = np.rand(n,n)
        
        # Sa and Sb correspond to the transformations of the latent variables
        Sa = np.dot(Caa,Wa)
        Sa = np.dot(Sa,Ma)
        
        Sb = np.dot(Cbb,Wb)
        Sb = np.dot(Sb,Mb)
        
        # SIa and SIb denote noise covariance matrices
        SIa = Caa - np.dot(Sa,Sa)
        SIb = Cbb - np.dot(Sb,Sb)
        
        print(Wa)
        print(Wb)
        
        
