import numpy as np
from scipy.linalg import svd as scipy_svd
from collections import Counter
class DisCCA(): 
    def __init__(self):
        self.wy=None
        self.wy=None
        self.output_dimensions=None
        self.C_B=0
        self.C_W=0

    '''
    Preprocess X and Y to nd array and get all  tuples of same classes together
    '''
    def preprocessing(self, X, Y, target):

        target_copy=[[target[i],i] for i in range(len(target))]
        target_copy.sort()


        if(type(X).__module__!='numpy'):
            X=X.to_numpy()
        if(type(Y).__module__!='numpy'):
            Y=Y.to_numpy()

        
        X_copy=np.array([])
        for i in range(len(target_copy)):
            new_row=X[target_copy[i][1]]
            if(len(X_copy)==0):
                X_copy=[new_row]
            else:
                X_copy = np.vstack([X_copy, new_row])

        Y_copy=np.array([])
        for i in range(len(target_copy)):
            new_row=Y[target_copy[i][1]]
            if(len(Y_copy)==0):
                Y_copy=[new_row]
            else:
                Y_copy = np.vstack([Y_copy, new_row])        
    
        X=X_copy.T
        Y=Y_copy.T
        return X,Y

    '''
    Function to fit data to model
    '''

    def fit(self, X, Y, target, output_dimensions):

        
        self.output_dimensions=output_dimensions

        

        X,Y=self.preprocessing(X, Y, target)
        X_shape=X.shape
        Y_shape=Y.shape
    

        #Zero mean X and Y
        X_hat=X-X.mean(axis=1, keepdims=True)
        Y_hat=Y-Y.mean(axis=1, keepdims=True)

        class_freq=dict(Counter(target))  
        N=len(target)
        print(X.shape,Y.shape)

        '''
        Creating block diagonal matrix A
        A=[[1](n1*n1)
                    [1](n2*n2)
                            ...
                                ...
                                    ...

                                        [1](nc*nc) ]
        '''
        i=0
        A=np.array([])
        cumulative_co=0
        for c in class_freq:
            for j in range(class_freq[c]):
                new_row=np.concatenate((np.zeros(cumulative_co), np.ones(class_freq[c]), np.zeros(N-cumulative_co-class_freq[c])),axis=0)
                if(len(A)==0):
                    A=new_row
                else:
                    A = np.vstack([A, new_row])
            cumulative_co+=class_freq[c]
            i+=1
        
        self.C_W=np.matmul(np.matmul(X_hat,A),Y_hat.transpose()) #Within class similarity matrix
        self.C_B=-(self.C_W) #Between class similarity matrix

        Sigma_xy=self.C_W/N
        Sigma_yx=np.matmul(np.matmul(Y_hat,A),X_hat.T)/N


        '''
        regularizing Sigma_xx and Sigma_yy
        '''
        rx = 1e-4 #regulazisation coefficient for x 
        ry = 1e-4 #regulazisation coefficient for y
        Sigma_xx=np.matmul(X_hat,X_hat.T)/N+ rx * np.identity(X_shape[0])
        Sigma_yy=np.matmul(Y_hat,Y_hat.T)/N + ry* np.identity(Y_shape[0])



        '''
        Finding inverse square root of  Sigma_xx and Sigma_yy
        using A^(-1/2)= PΛ^(-1/2)P'
        where
        P is matrix containing Eigen vectors of A in row form
        Λ is diagonal matrix containing eigen values in diagonal
        '''
        [eigen_values_xx, eigen_vectors_matrix_xx] = np.linalg.eigh(Sigma_xx)
        [eigen_values_yy, eigen_vectors_matrix_yy]= np.linalg.eigh(Sigma_yy)
        Sigma_xx_root_inverse = np.dot(np.dot(eigen_vectors_matrix_xx, np.diag(eigen_values_xx ** -0.5)), eigen_vectors_matrix_xx.T)
        Sigma_yy_root_inverse = np.dot(np.dot(eigen_vectors_matrix_yy, np.diag(eigen_values_yy ** -0.5)), eigen_vectors_matrix_yy.T)

        T=np.matmul(np.matmul(Sigma_xx_root_inverse,Sigma_xy),Sigma_yy_root_inverse)

        U, S, V = scipy_svd(T)

        self.wx= np.dot(Sigma_xx_root_inverse, U[:, 0:self.output_dimensions])
        self.wy= np.dot(Sigma_yy_root_inverse, V[:, 0:self.output_dimensions])       
        
        return None

    '''
    transform data to new view
    '''
    def transform(self, X, Y):

        if(type(X).__module__!='numpy'):
            X=X.to_numpy()
        if(type(Y).__module__!='numpy'):
            Y=Y.to_numpy()
        
        X_transformed=np.matmul(X,self.wx)
        Y_transformed=np.matmul(Y,self.wy)
        
        return X_transformed, Y_transformed

    def get_within_class_similarity(self):
        return self.C_W

    def get_between_class_similarity(self):
        return self.C_B