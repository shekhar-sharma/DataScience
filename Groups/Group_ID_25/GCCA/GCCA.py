import numpy as np
from scipy.linalg import eig
import matplotlib.pyplot as plt
import math
from matplotlib import colors
import h5py

class GCCA:

    def __init__(self, n_components=2, reg_param=0.1):

        # GCCA params
        self.n_components = n_components
        self.reg_param = reg_param

        # result of fitting
        self.view_len = 0
        self.cov_mat = [[]]
        self.h_list = []
        self.eigvals = np.array([])
        self.Vxx = [[]]
        self.Vyy = [[]]
        self.vxy = [[]]
        self.u = [[]]
        self.s = [[]]
        self.v = [[]]
    

        # result of transformation
        self.z_list = []


    def cal_eig(self, left, right):

        #calculating eigen dimension
        eig_dim = min([np.linalg.matrix_rank(left), np.linalg.matrix_rank(right)])

        #calculating eigenvalues & eigenvector
        eig_vals, eig_vecs = eig(left, right)

        #sorting eigenvalues & eigenvector
        sort_indices = np.argsort(eig_vals)[::-1]
        eig_vals = eig_vals[sort_indices][:eig_dim].real
        eig_vecs = eig_vecs[:,sort_indices][:,:eig_dim].real

        return eig_vals, eig_vecs

    def calc_cov_mat(self, x_list):

        view_len = len(x_list)

        #calc variance & covariance matrix
        z = np.vstack([x.T for x in x_list])
        cov = np.cov(z)
        
        return cov


    def fit(self, *x_list):

        # data size check
        view_len = len(x_list)
        print(np.shape(x_list))

        #normalizing
        #x_norm_list = [ self.normalize(x) for x in x_list]
        #print(np.shape(x_norm_list))

        #cov_mat = self.calc_cov_mat(x_norm_list)
        
        X = x_list[0][0]
        Y = x_list[0][1]
        #print(X)
        n = X.shape[0]
        Vxx=(1/n)*np.dot(X.T, X)
        Vyy=(1/n)*np.dot(Y.T, Y)
        Vxy=(1/n)*np.dot(X.T, Y)
        P=np.dot(np.dot(np.lib.scimath.sqrt(Vxx**(-1)),Vxy),np.lib.scimath.sqrt(Vyy**(-1)))
        #print(P)
        u,s,v=np.linalg.svd(P)
        #print(Vxx.shape)
        #print(np.shape(v))
#         wx=np.dot(np.lib.scimath.sqrt(Vxx**(-1)),u)
#         wy=np.dot(np.lib.scimath.sqrt(Vyy**(-1)),v.T)
        
        print(Vxx)
        print(Vyy)
        print(Vxy)

        #calculating generalized eigenvalue problem ( A*u = (lambda)*B*u )")


        # calc GEV
        #igvals, eigvecs = self.solve_eigprob(left, right)

        # substitute local variables for member variables
        self.view_len = view_len
        #self.cov_mat = cov_mat
        self.Vxx = Vxx
        self.Vyy = Vyy
        self.Vxy = Vxy
        self.u = u
        self.v = v
        self.s = s
        #print(cov_mat)
        #seLf.eigvals = eigvals

    def transform(self, *x_list):

        # data size check
        view_len = len(x_list)

        #normalizing
        x_norm_list = [ self.normalize(x) for x in x_list]
        
        wx=np.dot(np.lib.scimath.sqrt(self.Vxx**(-1)),self.u)
        wy=np.dot(np.lib.scimath.sqrt(self.Vyy**(-1)),self.v.T)
        #print(np.lib.scimath.sqrt(self.Vxx**(-1)))

        #transform matrices by GCCA
        #_list = [np.dot(x, h_vec) for x, h_vec in zip(x_norm_list, self.h_list)]

        #self.z_list = z_list

        #return z_list
        return wx,wy

    def fit_transform(self, *x_list):
        print(np.shape(x_list))
        self.fit(x_list)
        wx, wy = self.transform(x_list)
        print(wx)
        print(wy)

    @staticmethod
    def normalize(mat):
        m = np.mean(mat, axis=0)
        mat = mat - m
        return mat

    def calc_correlations(self):
        for i, z_i in enumerate(self.x_list):
            for j, z_j in enumerate(self.x_list):
                if i < j:
                   print ("(%d, %d): %f" %(i, j, np.corrcoef(x_i[:,0], x_j[:,0])[0, 1]))
