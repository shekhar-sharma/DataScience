import pandas as pd
import numpy as np
class DescriminativeExtendedcca():
    def __init__(self):
        self.w = [None, None]
        self.m = [None, None]
    def fit_tranform(self,X,Y):
        """
#        An implementation of Discriminative Extended CCA
#         # Arguments:
#             X and Y: the matrices containing the data for view 1 and view 2. Each row is a sample.
#             outdim_size: specifies the number of new features
#         # Returns
#             A and B: the linear transformation matrices
#             mean1 and mean2: the means of data for both views
#         """
        outdim_size=X.shape[1]
        matrix_y=np.dot(Y.T,Y)
        matrix_x=np.dot(X.T,X)
        #print(matrix_x)
        #print(matrix_x)
        
        """Finding Eigen values and Eigen Vectors of both the matrices"""
        [D_x, V_x] = np.linalg.eigh(matrix_x)
        [D_y, V_y] = np.linalg.eigh(matrix_y)
        #print(V_x)
        #print(V_y)
        
        """Converting Eigen value set(D_x,D_y) to daigonal Matrix"""
        eigenval_diag_matrix_x=np.diag(D_x)
        eigenval_diag_matrix_y=np.diag(D_y)
        
        """Calculating Deviation for X """
        sub_dev=np.dot(V_x,np.sqrt(eigenval_diag_matrix_x))
        #print(sub_dev)
        dev_x=np.dot(sub_dev,V_x.T)
        #print(dev_x)
        
        """Calculating Deviation for Y """
        sub_dev=np.dot(V_y,np.sqrt(eigenval_diag_matrix_y))
        dev_y=np.dot(sub_dev,V_y.T)
        #print(dev_x)
        
        
        """Now we will calculate dev_x*dev_y which will be proportional to Degree of Agreement between X and Y"""
        res=np.dot(dev_x.T,dev_y)
       # print(res)
        [U, D, V] = np.linalg.svd(res)
        self.w[0] = np.dot(dev_x, U[:, 0:outdim_size])
        self.w[1] = np.dot(dev_y, V[:, 0:outdim_size])
        D = D[0:outdim_size]
        [D2, V2] = np.linalg.eigh(res)
        sum_num=np.sum(D2)
        sum_den=np.sum(D_x)
        return sum_num/sum_den
#     def _get_result(self, x, idx):
#          result = x.reshape([1, -1]).repeat(len(x), axis=0)
#          result = numpy.dot(result, self.w[idx])
#          return result


# 

from DECCA import DescriminativedExtendecca
model=DescriminativeExtendedcca()
model.fit_tranform(a,b)


