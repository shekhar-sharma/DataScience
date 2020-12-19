import mlda
import numpy as np
import pandas as pd
import math
import os

if __name__ == '__main__':
    def main():

        X=pd.read_csv('mfeat-mor',delim_whitespace=True,header=None)
        Y=pd.read_csv('mfeat-pix',delim_whitespace=True,header=None) 
                
        Y=Y[:10]
        X=X[:10]
        Y.drop(Y.iloc[:, 7:], inplace = True, axis = 1) 

        Y.drop(Y.iloc[:, 6:], inplace = True, axis = 1)

        n=X.shape[1]
        row,col=(n,n)

        Mlda = mlda.MLDA()
        vTransforms = Mlda.fit_transform(X,Y,row,col,n)
        
        print("Wx -> ")

        print(vTransforms[0])
        print()
        print("Wy -> ")
        print(vTransforms[1])


main()