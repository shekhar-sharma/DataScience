import numpy as np
from scipy.linalg import eigh
from sklearn.preprocessing import StandardScaler

class RGCCA:

    def __init__(self, n_components=2, reg_param=0.5):

        self.n_components = n_components
        self.reg_param = reg_param

    def cal_eigh(self, left, right):

        #calculating eigen dimension
        eig_dim = min([np.linalg.matrix_rank(left), np.linalg.matrix_rank(right)])

        #calculating eigenvalues & eigenvector
        eig_vals, eig_vecs = eigh(left, right)

        #sorting eigenvalues & eigenvector
        sort_indices = np.argsort(eig_vals)[::-1]
        eig_vals = eig_vals[sort_indices][:eig_dim].real
        eig_vecs = eig_vecs[:,sort_indices][:,:eig_dim].real

        return eig_vals, eig_vecs

    def fit(self, data):
        sc = StandardScaler()
        data = [sc.fit_transform(d) for d in data]
        M = [d.T for d in data]
        D,F = len(data),[m.shape[0] for m in M] # F stores number of features in all datasets
        n_components = min([m.shape[1] for m in M])# stores minimum number of samples
        cross_cov = [n_components*np.dot(a,b) for a in M for b in data] #C11,C12,...C21,C22,.. and so on
        self.cov_mat = cross_cov

        left,right = np.zeros((sum(F),sum(F))), np.zeros((sum(F),sum(F)))
        for i in range(D):
            # add regularization parameter to right matrix of GEvP
            right[sum(F[:i]):sum(F[:i+1]), sum(F[:i]):sum(F[:i+1])] = (cross_cov[i*(D+1)] + self.reg_param*np.eye(F[i]))
            for j in range(D):
                if i != j:
                    left[sum(F[:j]):sum(F[:j+1]), sum(F[:i]):sum(F[:i+1])] = cross_cov[D*j + i]

        left = (left + left.T)/2 # converted to symmetric matrix
        right = (right + right.T)/2 # converted to symmetric matrix
        #calc GEvP
        eigvals, eigvecs = self.cal_eigh(left, right)
        self.weights = []
        for i in range(D):
            self.weights.append(eigvecs[sum(F[:i]):sum(F[:i+1]), :self.n_components])
        self.weights = [w.real for w in self.weights]
        self.c_comp = [np.dot(i[0],i[1]) for i in zip(data,self.weights)] #canonical components
            
        #canonical variables
        k = len(self.c_comp)
        self.variates = np.zeros((self.c_comp[0].shape[1],k,k))
        for i in range(k):
            for j in range(k):
                if j > i:
                    self.variates[:,i,j] = [np.nan_to_num(np.corrcoef(p,q)[0,1]) for (p,q) in zip(self.c_comp[i].T,self.c_comp[j].T)]
        self.variates = self.variates[np.nonzero(self.variates)]
        return self

    def transform(self, data):

        # data size check
        view_len = len(data)

        #normalizing
        sc = StandardScaler()
        d_norm = [sc.fit_transform(x) for x in data]
        Ls = [np.dot(d,w) for d,w in zip(d_norm,self.weights)]
        return Ls

    def fit_transform(self, data):
        self.fit(data)
        Ls = self.transform(data)
        print(Ls)