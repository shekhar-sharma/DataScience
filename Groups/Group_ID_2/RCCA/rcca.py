import numpy as np
from scipy import linalg as lin
from sklearn.preprocessing import StandardScaler

class RCCA:
    def __init__(self,n_comp=2,reg_param=0.1):
        self.n_comp = n_comp
        self.reg_param = reg_param
    
    def fit(self,data):
        if len(data) != 2: #RCCA works only for 2 views of data
            print('Use different variant of CCA.')
            print('For this variant use only 2 views of the data.')
        else:
            print('Training RCCA with regularization parameter = {} and {} components'.format(self.reg_param,self.n_comp))
            M = [d.T for d in data]
            D,F = 2,[m.shape[0] for m in M] # F stores number of features in both datasets
            n_comp = min([m.shape[1] for m in M])# stores minimum number of samples
            cross_cov = [np.dot(a,b.T)/n_comp for a in M for b in M] #Sxx,Syx,Sxy,Syy
            self.cross_cov = cross_cov

            #left = A, right = B for A*u = lambda*B*u <- generalized eigen problem
            left,right = np.zeros((sum(F),sum(F))), np.zeros((sum(F),sum(F)))
            for i in range(D):
                right[sum(F[:i]):sum(F[:i+1]), sum(F[:i]):sum(F[:i+1])] = (cross_cov[i*(D+1)] + self.reg_param*np.eye(F[i]))
                for j in range(D):
                    if i != j:
                        left[sum(F[:j]):sum(F[:j+1]), sum(F[:i]):sum(F[:i+1])] = cross_cov[D*j + i]

            left = (left + left.T)/2 # converted to symmetric matrix
            right = (right + right.T)/2 # converted to symmetric matrix

            e_value, e_vector = lin.eigh(left, right) #solved generalized eigenproblem
            e_value[np.isnan(e_value)] = 0
            eval_index = np.argsort(e_value)[::-1]
            self.weights = []
            e_vector = e_vector[:, eval_index]
            for i in range(D):
                self.weights.append(e_vector[sum(F[:i]):sum(F[:i+1]), :self.n_comp])
            self.weights = [w.real for w in self.weights]
            self.c_comp = [np.dot(i[0].T,i[1]) for i in zip(M,self.weights)] #canonical components
            
            #canonical variates
            k = len(self.c_comp)
            self.variates = np.zeros((self.c_comp[0].shape[1],k,k))
            for i in range(k):
                for j in range(k):
                    if j > i:
                        self.variates[:,i,j] = [np.nan_to_num(np.corrcoef(p,q)[0,1]) for (p,q) in zip(self.c_comp[i].T,self.c_comp[j].T)]
            self.variates = self.variates[np.nonzero(self.variates)]
            return self
    
    def transform(self,data):
        sc = StandardScaler()
        d_norm = [sc.fit_transform(d) for d in data]
        Ls = [np.dot(d,w) for d,w in zip(d_norm,self.weights)]
        return Ls
        
