# For Understanding the functionality perfomed by each function refer to .ipynb file it has everything explained in detail 

# Authors 
# Kanishk Gupta(0801CS171031)
# Aadeesh Jain (0801CS171001)
# Harsh Pastaria(0801CS171027
import numpy as np
import pandas as pd
import math
import os

class MULDA:    
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
    
    def calculatingSigma(self,X,Y,n):
        Stx = self.totalScattermatrixX(X,n)
        Sty = self.totalScattermatrixY(Y,n)
        sigma1= Stx.trace()
        sigma2= Sty.trace()
        sigma= sigma1/sigma2
        return sigma
    
    def fit(self,X,Y,n,row,col):
        Stx = self.totalScattermatrixX(X,n)
        Sty = self.totalScattermatrixY(Y,n)
        Sbx = self.betweennclassScattermatrixX(X,row,col,n)
        Cxy = self.covarianceAcrossXY(X,Y,n)
        Sby = self.betweennclassScattermatrixY(Y,row,col,n)
        Cxx = self.selfCovarianceX(X,n)
        Cyy = self.selfCovarianceY(Y,n)
        sigma = self.calculatingSigma(X,Y,n)
        I = self.identityMatrixI(row,col)
        d = Stx.shape[0]
        

        Dx = [ [ 0 for i in range(row) ] for j in range(col) ]
        Dy = [ [ 0 for i in range(row) ] for j in range(col) ]
            
        for u in range(0,d):
            #calculating Px
            p=np.dot(Stx,np.transpose(Dx))
            p1=np.dot((np.dot(Dx,Stx)),np.transpose(Dx))
            p1_inv=np.linalg.pinv(p1)

            Px=np.dot((np.dot(p,p1_inv)),Dx)
            Px = I - Px
            #Calculating Py
            P=np.dot(Sty,np.transpose(Dy))
            P1=np.dot((np.dot(Dy,Sty)),np.transpose(Dy))
            P1_inv=np.linalg.pinv(P1)

            Py=np.dot((np.dot(P,P1_inv)),Dy)
            Py= I-Py
            
            #Wx and Wy
            A = Sty*Px*Sbx
            A = sigma*A
            B = Sty*Px*Cxy
            B = sigma*B
            C = Stx*Py*Cxy
            D = Stx*Py*Sby
            
            Inv_a=np.linalg.pinv(A)
            Inv_d=np.linalg.pinv(D)

            F1=np.dot(Inv_a,B)
            F2=np.dot(F1,Inv_d)
            F=np.dot(F2,C)
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
            
            #rth vector pair
            wValues, vVectors = np.linalg.eig(wx)
            index = np.argmax(wValues, axis=0)
            
            l = [0]*row
            for i in range (0,row):
                l[i] = wx[i][index]

        
            for i in range(len(l)):
                Dx[i][u]=l[i]
                
            wValuesY, vVectorsY = np.linalg.eig(wy)
            indexY = np.argmax(wValuesY, axis=0)
            l1 = [0]*row
            for i in range (0,row):
                l1[i] = wy[i][indexY]

            for i in range(len(l1)):
                Dy[i][u]=l1[i] 

        d_net = [Dx,Dy]
        return d_net

    def combined_Features(self,X,Y,n,r,c):
        
        d_net = self.fit(X,Y,n,r,c)
        Dx = d_net[0]
        Dy = d_net[1]

        temp = np.array(Dx)
        temp.reshape(r,c)
        Wx = temp
        temp1 = np.array(Dy)
        temp1.reshape(r,c)
        Wy = temp1
        Wy.shape
        Res1 = Wy.dot(X)
        Res2 = Wx.dot(Y)
        Z= [[0]*1]*2
        Z[0] = Res1
        Z[1] = Res2
        return Z

