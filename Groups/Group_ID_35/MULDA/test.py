# For Understanding the functionality perfomed by each function refer to .ipynb file it has everything explained in detail 

# Authors 
# Kanishk Gupta(0801CS171031)
# Aadeesh Jain (0801CS171001)
# Harsh Pastaria(0801CS171027

import mulda
import pandas as pd
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
        row = X.shape[0] 
        col = row

        Mulda = mulda.MULDA()
        featureMatrix = Mulda.combined_Features(X,Y,n,row,col)
        
        print("Z -> ")
        print(featureMatrix)
        
main()
