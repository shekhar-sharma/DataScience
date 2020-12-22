#MCCA (Multiview Canonical Correlation Analysis)

import numpy as np 
from scipy import linalg as lin 
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier

class CWMVFE:
    
    def __init__(self,n_components=2,reg_param=0.01 , L):
        self.n_components = n_components
        self.reg_param = reg_param
        self.dimen = []
        self.C = [[]]  #covariance matix 
    
        
    #To normalize data so that mean=0 and std dev=1
    def normalize(self,X):
        return StandardScaler().fit_transform(X)

    
    #for adding regularization parameter
    def add_reg_param(self,c):
        I = np.identity(c.shape[0])
        R = np.dot(self.reg_param,I)
        c = c+R
        return c
    #for calculating covariance matrix 
    def cov_mat(self,X_list):
        views = len(X_list)
        N = len(X_list[0])
        C = [[np.array([]) for i in range(views)] for j in range(views)]
        
        for i in range(views):
            for j in range(views):
                C[i][j]=np.dot(X_list[i].T,X_list[j])
                C[i][j]=np.divide(C[i][j],float(N))
                if i==j:
                    C[i][j]=self.add_reg_param(C[i][j])
        
        self.C = C
        return C


    # it will find the k nearest element 
    def k_nearest(a_list , b_list , n_neighbors):
        knn = KNeighborsClassifier(n_neighbors)
        knn.fit(a_list , b_list)
        return (a_list , b_list)


# it will calculate ecludian distance for jenson shannon algorithm.
    def ecludian_distance(x_list):
        size = len(x_list)
        d = [[]]*size
        for i in range(size):
            for j in range(size):
                m=x[i] - x[j]
                if (m >= 0):
                    d[i][j]=m
                else:
                    d[i][j]= -m
        # while the i and j value is not specified is not specified in the algo so i am taking i=0 and j=size.
        upperis = pow((1+pow(d[i][j] , 2) , -1) 
        for i in range(size):
            for j in range(i):
                loweris = pow((1+pow(d[i][j] , 2) , -1)
        q= float(upperis / loweris)
        return q

    def jenson_shannon(self ,a_list , b_list , L):
        mid_q = ((ecludian_distance(a_list) + ecludian_distance(b_list))/2)
        num_a = 0
        num_b = 0
        for i in range(L):
            num_a = num_a + (ecludian_distance(a_list)log10(ecludian_distance(a_list)/mid_q))
            num_b = num_b + (ecludian_distance(b_list)log10(ecludian_distance(b_list)/mid_q))
        js=(0.5(num_a)+num_b)
        return js


    def sigh(a_list , b_list):
        old_a_list = a_list
        old_b_list = b_list
        sigh_a[] = 0*len(a_list)
        sigh_b[] = 0*len(b_list)
        k_nearest(a_list , b_list , 5)
        for i in range(len(a_list)):
            sigh_a
            sigh_b[i]= (old_b_list[i] - b_list[i])
            sigh_a[i] = (old_a_list[i] - a_list[i])
        return (sigh_a , sigh_b)

     
   
    def transform(self,X_list):
        views = len(X_list)
        X_list = [self.normalize(x) for x in X_list]
        X_reduced = [[]]*views
        for i in range(views): 
            for i in range(views):
                X_reduced[i]=np.dot(X_list[i],self.weights[i].T)

        return X_reduced
        
    def fit(self , a_list , b_list):
        view = len(a_list)
        #normalize the data
        a_list = [self.normalize(x) for x in a_list]
        b_list = [self.normalize(x) for x in b_list]

        for i in range(view):
            sigh(a_list[i] ,b_list[i])
            first_term =first_term + (jenson_shannon(a_list[i] , b_list[i] ,i)(sigh_a)(sigh_b.T)(np.dot(cov_mat(a_list[i]).T , cov_mat(b_list[i]))))
            second_term =second_term + (jenson_shannon(a_list[i] , b_list[i] ,i)(sigh_a)(sigh_a.T )(np.dot(cov_mat(a_list[i]).T ,cov_mat(a_list[i])))
        # n order to get more generalized flexibility, a parameter γ > 0 is introduced to balance the above two terms, so i assume lamda is 0.5
        lamda=0.5
        final_value = (first_term - (lamda)*(second_term))
        return final_value

    def fit_transform(self , a_list , b_list):
        self.fit(a_list , b_list)
        final_value=np.transpose(final_value)
        return final_value
    
    

    


