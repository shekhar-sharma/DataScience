import numpy as np 
from scipy import linalg as lin 
from sklearn.preprocessing import StandardScaler

class MCCA:
    
    def _init_(self,n_components=2,reg_param=0.01):
        self.n_components = n_components
        self.reg_param = reg_param
        
        self.views = 0 #number of views
        self.C = [[]]  #covariance matix 
        self.w_list = []  # list of projections
        self.score_list = []
        
    #To normalize data so that mean=0 and std dev=1
    def normalize(self,X):
        return StandardScale.fit_tranform(X)
    
    #To find the covariance matrix containing
    #both within view and between view covariance
    def cov_mat(self,X_list):
        
        views = len(X_list)
        X_list_stacked = np.vstack(X.T for X in X_list)
        cov = np.cov(X_list_stacked)
        
        #dimention of views
        dimen = [0]*views
        for i in range(views):
            dimen[i] = len(X[i].T)
            
        #sum of dimention till individual view
        sum_dimen = [0]*views
        for i in range(1,views):
            sum_dimen[i] = sum([x for x in dimen][:i+1])
            
        #cov_mat containing both within view and between view covariance
        C = [[np.array([]) for i in range(views)] for j in range(views)]
        
        #for calculating C11 C12...C21 C22 ....Cm1,Cm2....Cmm
        for i in range(views):
            for j in range(views):
                C[i][j] = cov[sum_dimen[i]:sum_dimen[i+1],sum_dimen[j]:sum_dimen[j+1]]
                
        return C
    
    def add_reg_param(self,C):
        
        for i in range(views):
            C[i][i] += self.reg_param * np.average(np.diag(cov_mat[i][i])) * np.eye(cov_mat[i][i].shape[0])
            
        return C
    
    def fit(self,X_list):
        
        views = len(X_list)
        x_normalize = [self.normalize(x) for x in X_list]
        C = cov_mat(x_normalize)
        #C=add_reg_param(C)
        
        #Constructing A(left) and B(right) matrix of GEP(generalized eigen value problem) 
        A_rows = [np.hstack([np.zeros_like(C[i][j]) if i == j else C[i][j] for j in range(views)])for i in range(views)]
        A = np.vstack(A_rows)
        B_rows = [np.hstack([np.zeros_like(C[i][j]) if i != j else C[i][j] for j in range(views)])for i in range(views)]
        B = np.vstack(B_rows)
        
        #calculating eigen value and eigen vector 
        eig_vals,eig_vecs = lin.eig(A,B)
        w_list = [eigvecs[start:end] for start, end in zip(dimen[0:-1], dimen[1:])]
        self.w_list = w_list
        self.views = views
        self.C = C
        
    def tranform(self,X_list):
        views = len(X_list)
        X_normalize = [self.normalize(x) for x in X_list]
        i=0
        for X,W in zip(X_normalize,self.w_list):
            score_list[i]=np.dot(X,W)
            i=i+1
        self.score_lits=score_list
        return score_list
                       
    def fit_tranform(self,X_list):
        self.fit(X_list)
        self.tranform(X_list)
                       
                       
        
def main():
    a = np.random.rand(5,5)
    b = np.random.rand(5,6)
    c = np.random.rand(5,7)
    d = np.random.rand(5,8)
    mcca = MCCA()
    mcca.fit(a,b,c,d)
    
