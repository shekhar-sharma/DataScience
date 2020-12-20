# -*- coding: utf-8 -*-
"""
Created on Sun Dec 20 19:00:51 2020

@author: Prajal Badjatya
"""

import numpy as np
from scipy import linalg as lin
import IWD.py
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
from sklearn import linear_model
from scipy.special import gamma, factorial

class BCCA:
    
    def __init__(self, n_components=2, reg_param=0.1):
        self.n_components=2
        self.reg_param = reg_param
        self.view_length=0
    
    #function for calculating covariace matrix
    def calculate_covariance_matrix(self, x_list):
        view_length = len(x_list)
        #calculating covariance matrix
        p = np.vstack([x.T for x in x_list])
        covariance_matrix =  np.cov(p)
        
        print(covariance_matrix)
        return covariance_matrix
    
    
     # returns normal multivariate distribution
    def multivariate_normal(x, d, mean, covariance):
        x_m = x - mean
        return (1. / (np.sqrt((2 * np.pi)**d * np.linalg.det(covariance))) * 
            np.exp(-(np.linalg.solve(covariance, x_m).T.dot(x_m)) / 2))
    
    
    #function for calculating eigen values and eigen vectors
    def calculate_eigen(self, left, right):
        #eigen dimension calculation
        eigen_dimension = min([np.linalg.matrix_rank(left), np.linalg.matrix_rank(right)])
        
        #calculating eigen values and eigen vector
        eigenValues , eigenVectors = eig(left,right)
        #sorting eigen values and eigen vector
        indices = np.argsort(eig_vals)[::-1]
        eigenValues = eigenValues[indices][:eigen_dimension].real
        eigenVectors = eigenVectors[:,indices][:,:eigen_dimension].real
        
        #returning sorted eigen values and eigen vectors
        return eigenValues , eigenVectors
    
    def fit(self, *x_list):
        view_length = len(x_list)
        #print(x_list)
        
        #normalizing
        normalized_list = list
#         for x in x_list:
#             normalized_list.append(self.normalize(x))
        
#         #print(normalized_list)
        
        
        Xa = normalized_list[0][0]
        Xb = normalized_list[0][1]
        n = Xa.shape[0]
        
        Cab = np.dot(Xa.T,Xb)
        Cab = Cab/n
        #Cba = np.dot(Xb,Xa)
        #Cba = Cba/n
        Caa = np.dot(Xa.T,Xa)
        Caa = Caa/n
        Cbb = np.dot(Xb.T,Xb)
        Cbb = Cbb/n
        
        
        A = np.dot(np.lib.scimath.sqrt(Caa**(-1)),Cab)
        B = np.lib.scimath.sqrt(Cbb**(-1))
        P = np.dot(A,B)
        
        #single value decomposition
        u,s,v = np.linalg.svd(P)
        
        
        #substituting local variables for member variables
        self.view_length = view_length
        self.Caa = Caa
        self.Cab = Cab
        self.Cbb = Cbb
        self.u = u
        self.v = v
        self.s = s
        
        #Interbattery factory Analysis
        
        zero = np.zeros(n)
        #print(zero)
        identitymatrix = np.identity(n)
        #print(identitymatrix)
        
        #latent variable z = multivariate distribution with mean mu and covariance sigma
        z = np.random.multivariate_normal(zero,identitymatrix,(n,n))
        print(z)
        
        #initializing hyperpriors alpha and beta
        alpha = 10**(-14)
        beta = 10**(-14)
        
        #Gamma function
        gama = gamma([alpha,beta])
#         print("gamma")
#         print(gama)

        #For the covmat using conjugate inverse wishart prior
    
    
    
    
    def fit_transform(self, *x_list):
        self.fit(x_list)
        view_len = len(x_list)
        
        Wa = np.dot(np.lib.scimath.sqrt(self.Caa**(-1)),self.u)
        Wb = np.dot(np.lib.scimath.sqrt(self.Cbb**(-1)),self.v.T)
        
        #printing final weights
        print(Wa)
        print(Wb)
        
        #returning final weights
        return Wa , Wb
    
    
    @staticmethod
     #function for normalization
    def normalize(matrix):
        m = np.mean(matrix,axis=0)
        matrix = matrix-m
        return matrix
        
        
        
    
# def main():

#     a = np.random.rand(3,3)
#     b = np.random.rand(3,3)


#     # create instance of BCCA
#     print(a)
#     print(b)
#     bcca = BCCA(reg_param=0.01)
#     # calculate BCCA
#     #bcca.fit(a, b)
#     bcca.fit_transform(a,b)

# if __name__=="__main__":
#     main()
        
