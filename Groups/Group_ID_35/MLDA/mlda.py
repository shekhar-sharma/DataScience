# For Understanding the functionality perfomed by each function refer to .ipynb file it has everything explained in detail 

# Author 
# Kanishk Gupta(0801CS171031)
# Aadeesh Jain (0801CS171001)
# Harsh Pastaria(0801CS171027)

import numpy as np
import pandas as pd
import math
import os
class MLDA():

    def selfCovarianceX(self,X,n):
        Cxx=np.dot(X,X.T)
        Cxx=Cxx/n
        return Cxx
    
    def selfCovarianceY(self,Y,n):
        Cyy=np.dot(Y,Y.T)
        Cyy=Cyy/n
        return Cyy
    def covarianceAcrossXY(self,X,Y,n):
        Cxy=np.dot(X,Y.T)
        Cxy=Cxy/n
        return Cxy

    def diagMatrixW(self,row,col,n):
        row,col=(n,n)
        W=[]
        for i in range(col):
            c=[]
            for j in range(row):
                if i==j:
                    c.append(1/n)
                else:
                    c.append(0)
            W.append(c)
        return W

    def identityMatrixI(self,row,col):
        I=[]
        for i in range(col):
            c1=[]
            for j in range(row):
                if i==j:
                    c1.append(1)
                else:
                    c1.append(0)
            I.append(c1)
        return I
    
    def withinclassScattermatrixX(self,X,row,col,n):
        W=self.diagMatrixW(row,col,n)
        I=self.identityMatrixI(row,col)
        Diff=np.subtract(I,W)
        S=np.dot(X,Diff)
        Swx=np.dot(S,X.T)
        Swx=Swx/n
        return Swx
    
    def betweennclassScattermatrixX(self,X,row,col,n):
        W=self.diagMatrixW(row,col,n)
        t=np.dot(X,W)
        Sbx=np.dot(t,X.T)
        Sbx=Sbx/n
        return Sbx
    
    def totalScattermatrixX(self,X,n):
        Stx=np.dot(X,X.T)
        Stx=Stx/n
        return Stx

    def withinclassScattermatrixY(self,Y,row,col,n):
        W=self.diagMatrixW(row,col,n)
        I=self.identityMatrixI(row,col)
        Diff=np.subtract(I,W)
        S=np.dot(Y,Diff)
        Swy=np.dot(S,Y.T)
        Swy=Swy/n
        return Swy
    
    def betweennclassScattermatrixY(self,Y,row,col,n):
        W=self.diagMatrixW(row,col,n)
        t=np.dot(Y,W)
        Sby=np.dot(t,Y.T)
        Sby=Sby/n
        return Sby
    
    def totalScattermatrixY(self,Y,n):
        Sty=np.dot(Y,Y.T)
        Sty=Sty/n
        return Sty

    def fit_transform(self,X,Y,row,col,n):

        Cxx=self.selfCovarianceX(X,n)
        Cyy=self.selfCovarianceY(Y,n)
        Cxy=self.covarianceAcrossXY(X,Y,n)

        Sbx=self.betweennclassScattermatrixX(X,row,col,n)
        Stx=self.totalScattermatrixX(X,n)
        Sby=self.betweennclassScattermatrixY(Y,row,col,n)
      
        Sty=self.totalScattermatrixX(Y,n)

        Trans_Stx=np.linalg.inv(Stx)
        Trans_Sty=np.linalg.pinv(Sty)

        a=np.dot(Sbx,Trans_Stx)
        b=np.dot(Cxy,Trans_Sty)
        c=np.dot(Cxy,Trans_Stx)
        d=np.dot(Sby,Trans_Sty)

        Inv_a=np.linalg.pinv(a)
        Inv_d=np.linalg.pinv(d)

        F1=np.dot(Inv_a,b)
        F2=np.dot(F1,Inv_d)
        F=np.dot(F2,c)
        F[F<0]=0
        F_Final=np.sqrt(F)

        U,D,V=np.linalg.svd(F_Final)
        V=V.T

        Cxxi=np.linalg.inv(Cxx)
        Cxxi[Cxxi<0]=0
        Cxx_sqrt=np.sqrt(Cxxi)
        wx=np.dot(Cxx_sqrt,U)


        Cyyi=np.linalg.pinv(Cyy)
        Cyyi[Cyyi<0]=0
        Cyy_sqrt=np.sqrt(Cyyi)
        wy=np.dot(Cyy_sqrt,V)
        
        w_net=[wx,wy]
        return w_net
