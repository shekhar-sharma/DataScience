import numpy as np
from sktensor import ktensor,dtensor, cp_als
import tensorly
import pandas as pd 


class TensorCCA:
  
  #Initialize variables
  def __init__(self, max_iter = 500):
    
    self.H = None
    self.number_of_views = None
    self.max_iter = max_iter

  #Fits data to model 
  def fit(self, Views, reduce_to_dim):

    self.number_of_views = len(Views)

    Views = self.views_hat(Views)

    var_matrix = self.cov_matrix(Views)

    cov_tensor = self.covariance_tensor(Views)

    self.H = self.tcca(Views, var_matrix, cov_tensor,reduce_to_dim)

    return self.H

  #Mean centering of data
  def views_hat(self, Views):
    for v in range(self.number_of_views):
      rows = Views[v].shape[0]
      mean = np.mean(Views[v], axis=0)
      Views[v] = Views[v] - np.tile(mean,(rows, 1))

    return Views


  #Calculate canonical vectors
  def tcca(self,Views, var_matrix, cov_ten, reduce_to_dim):

    var_matrix_inverse = list()
    for v in range(self.number_of_views):
      var_matrix_inverse.append(self.root_inverse(var_matrix[v]) + np.eye(var_matrix[v].shape[0]))

    M_ten = self.ttm(cov_ten, var_matrix_inverse)
    M_ten = dtensor(M_ten)
    P, fit, itr = cp_als(M_ten, reduce_to_dim,max_iter = self.max_iter)
    
    H = list()
    for v in range(self.number_of_views):
        H.append(np.dot(var_matrix_inverse[v] , P.U[v]))

    self.H = H

    return H
  
  #Calculate covariance matrix of each view
  def cov_matrix(self, Views):

    number_of_samples = Views[0].shape[0]
    var_matrix = []

    for v in range(self.number_of_views):

      var_v = np.matmul(np.transpose(Views[v]),Views[v])
      var_v = var_v / (number_of_samples-1)
      var_matrix.append(var_v)

    return var_matrix

  #Calculate covariance tensor of given views
  def covariance_tensor(self,Views):

    number_of_samples = Views[0].shape[0]

    for n in range(number_of_samples):
      u = []
      for v in range(self.number_of_views):
        u.append(np.array(Views[v][n]).reshape(-1,1))
      
      cov_x = ktensor(u).toarray()

      if n == 0:
        cov_ten = cov_x
      else:
        cov_ten = cov_ten + cov_x
          
    cov_ten = cov_ten / (number_of_samples-1)

    return cov_ten

  #Calculate root inverse of covariance matrix
  def root_inverse(self, Cpp):

    [D1, V1] = np.linalg.eigh(Cpp)
    D1 = D1+ abs(D1) + np.spacing(1)
    CppRootInv = np.dot(np.dot(V1, np.diag(D1 ** -0.5)), V1.T)

    return CppRootInv

  

  #Calculate tensor times matrix i.e. mode-i product
  def ttm(self, cov_ten, var_matrix_inverse):
    for i in range(len(var_matrix_inverse)):
      tt_matrix = tensorly.tenalg.mode_dot(cov_ten , var_matrix_inverse[i] , i)
    
    return tt_matrix


  #Reduce dimensions of each view
  def transform(self, Views):

    Z = []

    for v in range(self.number_of_views):
      Z.append(np.dot(Views[v],self.H[v]))

    return Z  #Each element of Z contains a view with reduced dimensions
