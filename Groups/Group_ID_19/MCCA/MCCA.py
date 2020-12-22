#MCCA (Multiview Canonical Correlation Analysis)

import numpy as np 
from scipy import linalg as lin 
from sklearn.preprocessing import StandardScaler

class MCCA:
    
    def __init__(self,n_components=2,reg_param=0.01):
        self.n_components = n_components
        self.reg_param = reg_param
        self.dimen = []
        self.C = [[]]  #covariance matix 
        self.weights = [[]]  # list of projections
      
        
    #To normalize data so that mean=0 and std dev=1
    def normalize(self,X):
        return StandardScaler().fit_transform(X)
    
    #for calculating dimentions of each view
    def dimentions(self,X_list):
        views = len(X_list)
        dimen=[0]*views
        for  i in range(views):
            dimen[i]=X_list[i].shape[1]
        return dimen
    
    #for adding regularization parameter
    def add_reg_param(self,c):
        I = np.identity(c.shape[0])
        R = np.dot(self.reg_param,I)
        c = c+R
        return c
    #for calculating covariance matrix 
    def cov_mat(self,X_list):
        views = len(X_list)
        N = len(X_list[0])
        C = [[np.array([]) for i in range(views)] for j in range(views)]
        
        for i in range(views):
            for j in range(views):
                C[i][j]=np.dot(X_list[i].T,X_list[j])
                C[i][j]=np.divide(C[i][j],float(N))
                if i==j:
                    C[i][j]=self.add_reg_param(C[i][j])
        
        self.C = C
        return C


    def fit(self,X_list):
        views = len(X_list)
        #normalize the data
        X_list = [self.normalize(x) for x in X_list]
        #calculating covariance 
        C = self.cov_mat(X_list)
        
        dimen = self.dimentions(X_list)
        #create the initial alpha
        alpha_initial = [np.array([[]]) for i in range(views)]
        for k in range(views):
            alpha_initial[k]=np.random.rand(dimen[k])

        #inialize alpha
        alpha = [[np.array([]) for i in range(views)] for j in range(self.n_components)]

        #Horst Algorithm 
        for i in range(self.n_components):
            for j in range(views):
                sum = np.zeros(dimen[j])
                if i==0:
                    for k in range(views):
                        sum = np.add(sum.T,np.dot(C[j][k],alpha_initial[k].T))
                else:
                    for k in range(views):
                        sum = np.add(sum.T,np.dot(C[j][k],alpha[i-1][k].T))
                alpha[i][j]=sum
                deno = (np.dot(alpha[i][j].T,alpha[i][j]))**(0.5)
                alpha[i][j]=np.divide(alpha[i][j],float(deno))
                
                
    #calculating weights
        weights = [[]]*views
        for i in range(views):
            weights[i] = alpha[:,i]
        for i in range(views):
            weights[i] = np.hstack(x.reshape(dimen[i],1) for x in weights[i])
#         weights = [[]]*views
#         for i in range(self.n_components):
#             if i==0:
#                 for j in range(views):
#                     weights[j]=alpha[i][j]
#             else:
#                 for j in range(views):
#                     weights[j]=np.vstack([alpha[i][j],alpha[i-1][j]])

        self.weights=weights
   
    def transform(self,X_list):
        views = len(X_list)
        X_list = [self.normalize(x) for x in X_list]
        X_reduced = [[]]*views
        for i in range(views): 
            for i in range(views):
                X_reduced[i]=np.dot(X_list[i],self.weights[i].T)

        return X_reduced
        
                       
    def fit_transform(self,X_list):
        self.fit(X_list)
        X_reduced.self.tranform(X_list)
        return X_reduced
