import numpy as np
from numpy import linalg as LA
from sklearn.preprocessing import StandardScaler,MinMaxScaler


class RMvFS:
    '''
    This class implements Augmented Lagrangian Multiplier method
    to minimize the proposed objective function for feature selection.
    It do feature selection for Classification under supervised learning.
    '''
    # lv = is dimension of each samples in view v, or no of features
    # N = total number of samples
    # c = total categories in classification task under supervised learning

    def __init__(self, **kwargs):
        # tolerance,max_iter,p,lam1,lam2,threshold
        self.epsilon = kwargs.get('tolerance', 0.001)    # tolerance
        self.T = kwargs.get('max_iter', 20)             # max iterations
        self.p = kwargs.get('p', 10)              # p > 1
        self.lambda_1 = kwargs.get('lam1', 0.01)       # > 0
        self.lambda_2 = kwargs.get('lam2', 0.1)       # > 0
        self.threshold = kwargs.get('threshold', 0.1)

        return None

    def constructY(self, Y):
        # creating sparse matrix of Y
        s = set(Y)
        index = [i for i in range(len(s))]
        d = dict(zip(s, index))
        new_Y = [[0 for j in range(len(s))] for i in range(len(Y))]
        for i in range(len(Y)):
            new_Y[i][d[Y[i]]] = 1

        return new_Y

    def l21_normalization(self,matrix,row,col):
        total = 0
        for i in range(row):
            temp = 0
            for j in range(col):
                temp += matrix[i][j]**2
            total += temp**(1/2)
        
        return total 


    def objective_fun_calc(self):
        obj_fun = 0
        term1 = 0
        for v in range(self.v):
            error = np.matmul( self.X[v].transpose(), self.W[v] ) - self.Y
            term1 += (self.theta[v]**self.p) * self.l21_normalization(error,self.N,self.c)

        term2 = 0
        for v in range(self.v):
            term2 += self.l21_normalization(self.W[v],self.lv[v],self.c)

        term3 = 0
        for v in range(self.v):
            term3 += LA.norm(self.W[v])

        obj_fun = term1 + self.lambda_1*term2 + self.lambda_2*term3

        return obj_fun

    def D1_calc(self,v):
        D1v = [[0 for j in range(self.lv[v])] for k in range(self.lv[v])]
        for j in range(self.lv[v]):
            s = 0
            for el in self.W[v][j]:
                s += el**2
            if s==0:
                D1v[j][j] = 0
            else:    
                D1v[j][j] = 1/(2*(s**(1/2)))

        return D1v

    def fit(self, X1, *Xn, Y):
        # output is nothing but it sets parameter W, theta and feature_importance
        if Xn:
            Xn = [X1] + list(Xn)
        else:
            Xn = [X1]

        self.v = len(Xn)

        # standardise X and Y first
  
        for v in range(self.v):
            scaler = MinMaxScaler()
            # relacing any string with 0 in the dataset
            Xn[v] = [[0 if type(Xn[v][i][j]) == str else Xn[v][i][j]
                      for j in range(len(Xn[v][0]))] for i in range(len(Xn[v]))]
            scaler.fit(Xn[v])
            Xn[v] = np.array(scaler.transform(Xn[v]),dtype=np.float64)

        # transpose X, as given Xv is (N x lv)
        for v in range(self.v):
            Xn[v] = Xn[v].transpose()

        self.X = Xn     # dim = (v x lv x N)
        self.Y = np.array(self.constructY(Y))  # dim = (N x c)
        self.N = self.Y.shape[0]
        self.lv = [len(Xi) for Xi in self.X]
        self.theta = [1/self.v for i in range(self.v)]
        self.c = self.Y.shape[1]
        self.W = np.array([[[1.0 for i in range(self.c)]
                            for j in range(self.lv[k])] for k in range(self.v)])  # dim = (v x l x c)
        self.E = np.array([[[1.0 for i in range(self.c)]
                            for j in range(self.N)] for k in range(self.v)])  # dim = (v x N x c)
        self.LAMBADA = self.E   # lagrange multiplier
        self.mu = 0.1           # panelty parameter
        self.zeta = 1.2

        obj_t = 0
        iter = 1

        while iter <= self.T:
        
            iter += 1
            
            D1 = [0 for i in range(self.v)]
            D2 = [0 for i in range(self.v)]  # collection of D2v
            G = [0 for i in range(self.v)]  # collection of Gv values

            # objective function calculation for t'th iteration
            if obj_t == 0:
                obj_t = self.objective_fun_calc()           

            #calculating denometer for view-weightage parameter theta
            denom = 0
            for v in range(self.v):
                l21norm = self.l21_normalization(self.E[v],self.N,self.c)
                if l21norm != 0:
                    denom += (1/l21norm)**(1/(self.p-1))
            
            # iterating over each view
            for v in range(self.v):
                
                #D1v calculation
                D1[v] = self.D1_calc(v)

                # D2v calculation
                D2v = [[0 for j in range(self.lv[v])] for k in range(self.lv[v])]
                for j in range(self.lv[v]):
                    D2v[j][j] = LA.norm(self.W[v])

                D2[v] = D2v

                # updating projection matrix
                self.W[v] = self.mu * np.matmul(
                                        LA.inv(
                                            self.lambda_1 * np.array(D1[v])
                                            + self.lambda_2 * np.array(D2[v])
                                            + self.mu*np.matmul( self.X[v], self.X[v].transpose()) ),
                                        np.matmul(self.X[v], self.Y + self.E[v] - ((1/self.mu)*self.LAMBADA[v])))

                # calculating Gv
                G[v] = np.matmul( self.X[v].transpose(), self.W[v] ) - self.Y + ( (1/self.mu)*self.LAMBADA[v] )

                # calculating slack variable Ev
                param = (self.theta[v] ** self.p)/self.mu

                # as Ev and Gv are nothing but as same dimension as Y, i is row no
                for i in range(self.N):
                    gvi = sum(G[v][i]**2)**(1/2) 
                    if gvi > param:
                        self.E[v][i] = (1 - (param/gvi)) * G[v][i]
                    else:
                        self.E[v][i] = 0  # row as 0 elements

                # update view weighted theta
                for v in range(self.v):
                    l21norm = self.l21_normalization(self.E[v],self.N,self.c)
                    if l21norm == 0 or denom == 0:
                        self.theta[v] = 0
                    else:
                        self.theta[v] = ((1/l21norm)**(1/(self.p-1)))/denom

                # UPDATE LAGRANGIAN multiplier matrix
                self.LAMBADA[v] += self.mu*( np.matmul(self.X[v].transpose(), self.W[v])
                                             - self.Y - self.E[v]
                                            )

            #updating penalty paramter
            self.mu = self.zeta*self.mu

            # objective function calculation for t+1'th iteration
            obj_t1 = self.objective_fun_calc()

            # calculating error
            er = abs(obj_t1 - obj_t)

            obj_t = obj_t1

            if er < self.epsilon:
                break
                
        self.feature_importance = []
        for v in range(self.v):
            for row in range(self.lv[v]):
                self.feature_importance.append(self.theta[v]*sum(self.W[v][row]))

        return None

    def transform(self, X1, *Xn):
        output = np.array(X1)

        if Xn:
            for mat in Xn:
                output = np.append(output,mat,1)

        mask = [True if el >= self.threshold else False for el in self.feature_importance]
        output = output.transpose()
        output = output[mask]
               
        return output.transpose()
