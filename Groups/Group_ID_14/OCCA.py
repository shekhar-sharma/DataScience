import numpy as np
import pandas as pd
from numpy.linalg import matrix_power

#Orthogonal Canonical Correlation Analysis
class OCCA:
  '''
  OCCA takes 2 optional arguments
  n_Components:no. of components to be extracted from OCCA, defualt is set to 2
  noramlize: default is True, signify whether to scale/normalize the data
  '''
  def __init__(self, n_components=2,normalize=True,complex_=False):

    #OCCA parameters
    self.__n_components=n_components
    self.__normalize=normalize
    self.__complex_=complex_

    self.__X=None
    self.__Y=None

    #OCCA attributes
    self.__size, self.__p,self.__q=None,None,None
    self.__mean_x,self.__mean_y,self.__std_x,self.__std_y=0,0,1,1
    self.x_weights_=None
    self.y_weights_=None
    self.eigval_=None
    self.covariance_xx=None
    self.covariance_xy=None
    self.covariance_yy=None
    self.x_tranform_=None
    self.y_transform_=None

  def fit(self,X,Y):
    '''
    Takes 2 positional arguments X and Y both with same 1st dimension
    fit the occa model corresponding to the input data
    '''
    if X is None or Y is None:
      raise TypeError("One positional argument missing")
    self.__X=np.asarray(X)
    self.__Y=np.asarray(Y)
    self.__X=self.__X.astype('float64')
    self.__Y=self.__Y.astype('float64')
    #cheking input dimension
    if self.__X.ndim!=2 or self.__Y.ndim!=2:
      raise TypeError("Reshape the input array required 2D objects")
    self.__size,self.__p,self.__q=self.__X.shape[0],self.__X.shape[1],self.__Y.shape[1]
    #validating no. of components
    if self.__n_components!=int(self.__n_components) or self.__n_components<1 or self.__n_components>min(self.__p,self.__q):
      raise ValueError("Invalid number of compenents specified max ")
    #checking dim_0 of X and Y
    if self.__X.shape[0]!=self.__Y.shape[0]:
      raise ValueError("n_dim 0 of X and Y must be equal")
    #scaling data, normalize if nomalize set True
    self.__X,self.__Y=self.__normalization(self.__X,self.__Y,self.__normalize)
    #covaraiance matrices
    sxx,syy,sxy=self.__covariance(self.__X,self.__Y,self.__size)
    #twin eigen decomposition for obtaining projection weights or canonical covariates
    alpha,beta,eig_val=self.__twin_eigen_decomposition(sxx,syy,sxy,self.__n_components)
    self.x_weights_=alpha
    self.y_weights_=beta
    self.eigval_=eig_val
    print("OCCA (n_components=",self.__n_components,"normalize=",self.__normalize,"complex_=",self.__complex_,")")

  def transform(self, X,Y=None):
    '''
    Fit trained model onto input data
    Takes atmost 2 argaument X and Y where X is must and Y optional
    '''
    if X is None:
      raise TypeError("Miising one positional argument X")
    if X is not None:
      X=np.asarray(X)
      X=X.astype('float64')
    if Y is not None:
      Y=np.asarray(Y)
      Y=Y.astype('float64')
    if (X is not None and X.ndim!=2) or (Y is not None and Y.ndim!=2):
      raise TypeError("Reshape the input array required 2D objects")
    if X is not None and Y is not None and X.shape[1]!=self.__p and Y.shape[1]!=self.__q:
      raise ValueError("Wrong input dimension for broadcasting")
    if X.shape[1]!=self.__p:
      raise ValueError("Wrong X dimension must be equal to",self.p)
    #scaling/normalizing dataaccorind to fitted data mean and standard deviation
    X-=self.__mean_x
    X=X/self.__std_x
    #transforming data based on learned canonical covariates
    X=np.dot(X,self.x_weights_)
    self.x_transform_=X
    self.covariance_xx=(1/X.shape[0])*np.dot(X,X.T)
    #checking is second argument is specified or not
    if Y is not None and Y.shape[1]!=self.__q:
      raise ValueError("Wrong Y dimension must be equal to",self.q)
    if Y is not None:
      Y-=self.__mean_y
      Y=Y/self.__std_y
      Y=np.dot(Y,self.y_weights_)
      self.y_transform_=Y
      self.covariance_yy=(1/Y.shape[0])*np.dot(Y,Y.T)
      self.covariance_xy=(1/X.shape[0])*np.dot(X,Y.T)
    if Y is not None:
      return X, Y
    else:
      return X

  def fit_transform(self,X,Y):
    '''
    directly fit the model on input data and tranform it
    '''
    if X is None or Y is None:
      raise TypeError("2 arguments asre must, missing  apositional argument")
    #fit the data
    self.fit(X,Y)
    #transform the data
    X,Y=self.transform(X,Y)
    return X,Y

  def __normalization(self,X,Y,normalize):
    '''
    Takes 3 positional arguments X,Y and normalize
    Scale the input data X and Y and normalize it on the basis of normalize flag
    '''
    #scale the data to mean 0
    self.__mean_x=X.mean(axis=0)
    self.__mean_y=Y.mean(axis=0)
    #print(X.dtypes,Y.dtypes,self.mean_x.dtypes,self.mean_y.dtypes)
    X-=self.__mean_x
    Y-=self.__mean_y
    #if normalize set True, normalize X and Y
    if normalize:
      self.__std_x,self.__std_y=X.std(axis=0,ddof=1),Y.std(axis=0,ddof=1)
      self.__std_x=[1 if i==0 else i for i in self.__std_x]
      self.__std_y=[1 if i==0 else i for i in self.__std_y]
      X=X/self.__std_x
      Y=Y/self.__std_y
    return X,Y

  def __covariance(self,X,Y,n):
    '''
    Computes covariance matrix betwwen X-X, Y-Y and X-Y
    '''
    sxx=(1/n)*np.dot(X.T, X)
    syy=(1/n)*np.dot(Y.T, Y)
    sxy=(1/n)*np.dot(X.T, Y)
    return sxx,syy,sxy

  def __twin_eigen_decomposition(self,sxx,syy,sxy,n_components):
    '''
    Performs twin eigen decomposition of X and Y
    to compute teh canaonical covaraites by
    separately computing 1st canaonical covaraite pair and 
    iteratively computing the rest of the commponets based on prior extracted covariates
    '''
    alpha,beta,eig_val=self.__first_component(sxx,syy,sxy)
    eig_val=[eig_val**2]
    #computing the remaining covariates based on previosly computed pair  
    for k in range(2,n_components+1):
      x=np.dot(np.dot(alpha.T,self.__inv(sxx)),alpha)
      gx=np.dot(np.dot(np.dot(self.__inv(sxx),alpha),self.__inv(x)),alpha.T)
      y=np.dot(np.dot(beta.T,self.__inv(syy)),beta)
      gy=np.dot(np.dot(np.dot(self.__inv(syy),beta),self.__inv(y)),beta.T)
      hx=np.dot(self.__inv(sxx),sxy)
      hy=np.dot(self.__inv(syy),sxy.T)
      nx=gx.shape[0]
      ny=gy.shape[0]
      ix=np.identity(nx)
      iy=np.identity(ny)
      a=np.dot(ix-gx,hx)
      b=np.dot(iy-gy,hy)
      if self.__complex_==True:
        t=np.lib.scimath.sqrt(np.dot(a,b))
        q=np.lib.scimath.sqrt(np.dot(b,a))
      else:
        t=np.dot(a,b)
        t=[[0 if i<0 else 0 for i in t_]for t_ in t]
        t=np.sqrt(t)
        q=np.dot(b,a)
        q=[[0 if i<0 else 0 for i in q_]for q_ in q]
        q=np.sqrt(q)
      u_alpha,s_alpha,v_alpha=np.linalg.svd(t)
      u_beta,s_beta,v_beta=np.linalg.svd(q)
      wx_i=np.dot(self.__inv_sq_root(sxx),(v_alpha**2).T[:,0].reshape((v_alpha.shape[0],1)))
      wy_i=np.dot(self.__inv_sq_root(syy),(v_beta**2).T[:,0].reshape((v_beta.shape[0],1)))
      alpha=np.append(alpha,wx_i,axis=1)
      beta=np.append(beta,wy_i,axis=1)
      eig_val.append(s_beta[0]**4)
    return alpha,beta,eig_val
    
  def __first_component(self,sxx,syy,sxy):
    '''
    Computing 1st canonical covariate pair using eigen decomposition of covaraince matrix
    '''
    P=np.dot(np.dot(self.__inv_sq_root(sxx),sxy),self.__inv_sq_root(syy))
    u,s,v=np.linalg.svd(P)
    eig_val=s[0]
    wx=np.dot(self.__inv_sq_root(sxx),u)
    wy=np.dot(self.__inv_sq_root(syy),v.T)
    alpha=wx[:,0]
    beta=wy[:,0]
    alpha=alpha.reshape((alpha.shape[0],1))
    beta=beta.reshape((beta.shape[0],1))
    return alpha,beta,eig_val
  

  def __inv(self,A):
    #function to find inverse using eigenvalue decomposition which can even handle singular matrices
    val,vec=np.linalg.eigh(A)
    return np.dot(np.dot(vec, np.diag(val**-1)),vec.T)

  def __inv_sq_root(self,A):
    #function to find inverse sqaure root of a matrix using eigen decomposition
    val,vec=np.linalg.eigh(A)
    if self.__complex_==True:
      return np.dot(np.dot(vec,np.diag(np.lib.scimath.sqrt(val))),vec.T)
    else:
      val=[0 if i<0 else i for i in val]
      return np.dot(np.dot(vec,np.diag(np.sqrt(val))),vec.T)
