from sklearn.neighbors import NearestNeighbors
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from scipy.linalg import eigh
from scipy.linalg import sqrtm
from scipy.linalg import inv

class LapMCCA:
  def __init__(self,n_components=2,p_neighbors=3,reg=0.001):
    self.h = n_components
    self.p = p_neighbors
    self.r = reg
  
  def create_affinity_graph(self,X,label,n):
    wm=[[0 for k in range(n)] for k in range(n)]
    Xt=X.T
    nbrs=NearestNeighbors(self.p,algorithm='ball_tree').fit(Xt)
    distances, indices = nbrs.kneighbors(Xt,Xt.shape[0])
    for i in range(len(indices)):
      j=indices[i][0]
      lab=label[j]
      wm[j][j]=1
      count=0
      for k in range(len(indices[i])):
        if(count<Xt.shape[0] or count<self.p):
          if(lab==label[k] and j!=k):
            count+=1
            wm[j][k]=cosine_similarity(np.array([X[:,j].T]),np.array([X[:,k]]))[0][0].round(2)
        else:
          break
    return wm

  def D_within_view(self,W,n):
    D=np.zeros([n,n])
    for j in range(n):
      D[j][j]=np.sum(W[j])
    return D 

  def L_within_view(self,D,W,n):
    L=np.subtract(D,W)
    return L
  
  def S_within_view(self,X,L,Xt,n):
    S=np.dot(np.array(X),np.array(L))
    S=np.dot(S,np.array(Xt))
    return (S/(n**2))

  def W_i_j(self,Wi,Wj,n):
    W=np.multiply(Wi,Wj)
    return W    
  
  def D_i_j(self,W,n):
    D=np.zeros([n,n])
    for j in range(n):
      D[j][j]=np.sum(W[j])
    return D

  def L_i_j(self,D,W,n):
    L=np.subtract(D,W)
    return L  

  def S_i_j(self,X,L,Yt,n):
    S=np.dot(X,L)
    S=np.dot(S,Yt)
    return (S/(n**2))  

  def fit(self,X,label):
    W=[]
    n=X[0].shape[1]
    m=X.shape[0]

    # creating p nearest neighbor affinity graph
    for i in range(m):
      w=self.create_affinity_graph(X[i],label[i],n)
      W.append(w)
    W=np.array(W)

    # calculating within view covariance matrix
    D_w_view=[]
    for i in range(m):
      D_w_view.append(self.D_within_view(W[i],n))
    D_w_view=np.array(D_w_view)

    L_w_view=[]  
    for i in range(m):
      L_w_view.append(self.L_within_view(D_w_view[i],W[i],n))
    L_w_view=np.array(L_w_view)

    S_w_view=[]
    for i in range(m):
      S_w_view_i=[]
      for j in range(m):
        if(i==j):
          S_w_view_i.append(self.S_within_view(X[i],L_w_view[i],X[i].T,n))
        else:
          S_w_view_i.append(0)
      S_w_view.append(S_w_view_i)
    S_w_view=np.array(S_w_view)

    # regularizing the within-view covariance matrix
    for i in range(m):
      S_w_view[i][i]=np.add(S_w_view[i][i],np.dot(self.r,np.identity(S_w_view[i][i].shape[0])))

    # calculating between view covariance matrix
    Wij=[]
    for i in range(m):
      Wi=[]
      for j in range(m):
        if(i!=j):
          Wi.append((self.W_i_j(W[i],W[j],n)))
        else:
          Wi.append(0)
      Wij.append(Wi)
    Wij=np.array(Wij)

    Dij=[]
    for i in range(m):
      Di=[]
      for j in range(m):
        if(i!=j):
          Di.append(self.D_i_j(Wij[i][j],n))
        else:
          Di.append(0)
      Dij.append(Di)
    Dij=np.array(Dij)

    Lij=[]
    for i in range(m):
      Li=[]
      for j in range(m):
        if(i!=j):
          Li.append(self.L_i_j(Dij[i][j],Wij[i][j],n))
        else:
          Li.append(0)
      Lij.append(Li)
    Lij=np.array(Lij)

    Sij=[]
    for i in range(m):
      Si=[]
      for j in range(m):
        if(i!=j):
          Si.append(np.array(self.S_i_j(X[i],Lij[i][j],X[j].T,n)))
        else:
          Si.append(0)
      Sij.append(Si)
    Sij=np.array(Sij)

    # finding the alpha corresponding to every view
    alpha=[]
    for i in range(m):
      alpha.append([np.random.rand(X[i].shape[0])])
    
    for i in range(self.h):
      for j in range(m):
        alp=np.dot(S_w_view[j][j],alpha[j][i].T)
        for k in range(m):
          if(j!=k):
            alp=np.add(alp,np.dot(Sij[j][k],alpha[k][i].T))
        deno=(np.dot(alp,alp.T))**(0.5)
        alp=(alp)/deno
        alpha[j]=np.vstack((alpha[j],alp))
    
    for i in range(m):
      alpha[i]=np.delete(alpha[i],0,0)
      alpha[i]=alpha[i].T
    return alpha
  
  def fit_transform(self,X,label):
    alpha=self.fit(X,label)
    Xnew=[]
    for i in range(X.shape[0]):
      Xnew.append(np.dot(alpha[i].T,X[i]))
    return Xnew

