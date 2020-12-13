import scipy.linalg as sci_lia
import numpy as np
import numpy.linalg as np_lia

class RandomizedCCA:
    
    #Calculation of random matrix generated after 'q' passes
    def RandomMatrix(self,X,Y,Q):
        Y = np.dot(X.T, Y)
        Y = np.dot(Y,Q)
        return Y
    
    #Optimizing Q to get a smaller matrix for reducing calculations
    def Optimization(self, Q, A):
        Ctemp1 = np.dot(Q.T, A.T)
        Ctemp2 = np.dot(Ctemp1, A)
        C = np.dot(Ctemp2, Q)
        return C
    
    #lambdax, lambday are scale-free parameterization 
    def LambdaCalculation(self,X):
        phi = 0.01
        A = np.dot(X.T,X)
        Trace = np.matrix.trace(A)
        lambdaA = phi*Trace / X.shape[1]
        return lambdaA
    
    def FinalMatrixCalculation(self, Qx, Qy, X, Y):
        
        lambdax = self.LambdaCalculation(X)
        lambday = self.LambdaCalculation(Y)

        #Final optimization over bases Qx, Qy
        Cx = self.Optimization(Qx, X)
        Cy = self.Optimization(Qy, Y)
        
        #cholesky matrix
        Lx = np.linalg.cholesky(Cx + (lambdax*np.dot(Qx.T, Qx)))
        Ly = np.linalg.cholesky(Cy + (lambday*np.dot(Qy.T, Qy)))
        
        #Final matrix calculation
        Fmatrix = np.dot(Qx.T, X.T)
        Fmatrix = np.dot(Fmatrix, Y)
        Fmatrix = np.dot(Fmatrix, Qy)
        Fmatrix = np.dot(np_lia.inv(Lx.T), Fmatrix)
        Fmatrix = np.dot(Fmatrix, np_lia.inv(Ly))
        
        return Fmatrix,Lx,Ly

    #k+p is number of features to be extracted in final output
    #q is number of passes [iteration]
    
    def fit(self, X, Y, n_features, n_passes):

        #Randomized range finder for A^T.B
        Qx = np.random.rand(X.shape[1],n_features)
        Qy = np.random.rand(Y.shape[1],n_features)
        
        for i in range(n_passes):
            Ya = self.RandomMatrix(X, Y, Qy)
            Yb = self.RandomMatrix(Y, X, Qx)
            
            Qx = sci_lia.orth(Ya)
            Qy = sci_lia.orth(Yb)      
        
        finalMatrix,Lx,Ly = self.FinalMatrixCalculation(Qx, Qy, X, Y)
        
        U,D,V = np_lia.svd(finalMatrix)
        Xx = ((X.shape[0])**(0.5)) * (np.dot(Qx, np.dot(np_lia.inv(Lx), U)))
        Xy = ((Y.shape[0])**(0.5)) * (np.dot(Qy, np.dot(np_lia.inv(Ly), V)))
    
        return Xx, Xy, D
    

