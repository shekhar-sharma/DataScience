import numpy as np


def inverse(A):
    return np.linalg.inv(A)
    
def eigen_decom(A):
    return np.linalg.eig(A)

"""
Frobenius normalization
"""
def norml21(A):
    row, col= A.shape
    l=[0 for j in range(col)]
    for j in range(col):
        s=0
        for i in range(row):
            s+=A[i][j]**2
        l[j]=s**0.5
    return np.array(l)

def converge(L1,L2):
    return np.array_equal(L1,L2)

"""
R: 3D array
    R is a three dimensional list
    Each matrix in R represents observations from different view
    Column of each matrix represents attributes of vth view and
    Row of each matrix represents different objects

d: int
    Dimension of the common representation

neta: float
      Regularization parameter
      
maxIter1: int
            Maximum iteration number

maxIter2: int
            Maximum iteration number
"""

def l21_cca(R, d=1, neta=1, maxIter1=1, maxIter2=1):
    R=np.array(R)
    v=R.shape[0]
    A=[Xv/neta-np.identity(Xv.shape[0]) for Xv in R]
    D=[np.identity(Xv.shape[0]) for Xv in R]
    Q=[0 for i in range(v)]
    for i in range(v):
        T1=np.dot(inverse(D[i]),A[i].T)
        Q[i]=np.dot(T1, inverse(np.dot(A[i],T1)))
    T1=[0 for i in range(v)]
    for i in range(v):
        T2=np.dot(Q[i].T, D[i])
        T1[i]=np.dot(T2,Q[i])
    B=sum(T1)
    Uo=eigen_decom(B)[1]
    Fo=[0 for i in range(v)]
    F=[0 for i in range(v)]
    for p in range(maxIter1):
        for q in range(maxIter2):
            for i in range(v):
                T1=np.dot(inverse(D[i]),A[i].T)
                T2=inverse(np.dot(A[i],T1))
                T3=np.dot(T1,T2)
                F[i]=np.dot(T3,Uo)
                T1=2*norml21(F[i])
                T1=1/T1
                D[i]=np.diag(T1)
                #D[i]=inverse(np.diag(T1))
            if converge(F,Fo):
                break
        Fo=F
        T1=[0 for i in range(v)]
        for i in range(v):
            T2=np.dot(Q[i].T, D[i])
            T1[i]=np.dot(T2,Q[i])
        B=sum(T1)
        T1=eigen_decom(B)
        dic={j:i for i,j in enumerate(T1[0])}
        T1=T1[1]
        U=[0 for i in range(d)]
        count=0
        for i in sorted(dic.keys()):
            U[count]=T1[:,dic[i]]
            count+=1
            if count==d:
                break
        U=np.array(U)
        U=U.T
        if converge(U,Uo):
            break
        Uo=U
    return U
