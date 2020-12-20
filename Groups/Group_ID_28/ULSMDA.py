
import numpy as np
import pandas as pd
import math

class ULSMDA():
    def affMatrix(self,X,Y,n):
        Ww=X
        Wb=Y

        for i in range(n):
            for j in range(n):
                if Ww[i][j]>=1:
                    Ww[i][j]=1
                else:
                    Ww[i][j]=0

        for i in range(n):
            for j in range(n):
                if Wb[i][j]>=1:
                    Wb[i][j]=1
                else:
                    Wb[i][j]=0

        return Ww,Wb



    def diagMatrix(self,row,col,n,Ww,Wb):
        row,col=(n,n)
        Dw=[]
        for i in range(col):
            ls=[]
            for j in range(row):
                if i==j:
                    ls.append(Ww[i][j])
                else:
                    ls.append(0)
            Dw.append(ls)
        Db=[]
        for i in range(col):
            ls=[]
            for j in range(row):
                if i==j:
                    ls.append(Wb[i][j])
                else:
                    ls.append(0)
            Db.append(ls)
        return Dw,Db


    def identityMatrix(self,row,col,n)
        row,col=(n,n)
        Id=[]
        for i in range(col):
            ls1=[]
            for j in range(row):
                if i==j:
                    ls1.append(1)
                else:
                    ls1.append(0)
            Id.append(ls1)
        return Id

    def laplacMatrix(self,Dw,Ww,Id,Lw,Lb):
        Lw=np.subtract(Dw,Ww)
        Lb=np.subtract(Db,Wb)
        sub=np.subtract(Lw,Lb)
        mul=np.dot((n-1),Id)
        A=np.add(sub,mul)
        return A      

    def fit(self,X,Y,row,col,n):
        Ww,Wb=self.affMatrix(X,Y,n)
        Dw,Db=self.diagMatrix(row,col,n,Ww,Wb)
        Id1=self.identityMatrix(row,col,n)
        A=self.laplacMatrix(Dw,Db,Lw,Lb,Id1)
        G=np.dot(np.dot(X,X.T),A)
        Hin=np.divide(G,4)
        V=np.dot(np.linalg.pinv(Hin),G)
        Zx=np.dot(V.T,X)
        Zy=np.dot(V.T,Y)
        return Zy




