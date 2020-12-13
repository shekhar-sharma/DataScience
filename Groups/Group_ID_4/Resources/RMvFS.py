import numpy as np
from numpy import linalg as LA
from sklearn.preprocessing import StandardScaler

class RMvFS:
    '''
    This class implements Agmented Lagrangian multiplication method
    to minimize the proposed formula for feature selection.
    It do feature selection for Classification under supervised learning.
    '''
    # lv = is dimension of each samples in view v, or no of features
    # N = total number of samples
    # c = total categories in classification task under supervised learning

    def __init__(self, **kwargs):
        # tolerance,max_iter,p,lam1,lam2
        self.epsilon = kwargs.get('tolerance', 0.001)    # tolerance
        self.T = kwargs.get('max_iter', 20)             # max iterations
        self.p = kwargs.get('p', 10)              # p > 1
        self.lambda_1 = kwargs.get('lam1', 0.01)       # > 0
        self.lambda_2 = kwargs.get('lam2', 0.1)       # > 0
        self.threshold = kwargs.get('threshold',0.01) # select from model uses 1e-5

        return None

    def constructY(Y):
        #creating sparse matrix of Y
        s = set(Y)
        index = [i for i in range(len(s))]
        d = dict(zip(s,index))
        new_Y = [[0 for j in range(len(s)] for i in range(len(Y))]
        for i in range(len(Y)):
            new_Y[i][d[Y[i]]] = 1
                
        return new_Y

    def fit(self, X1, *Xn, Y):
        #output is nothing but it sets parameter W and theta

        if Xn:
            Xn = list(Xn)
            Xn.append(X1)
        else:
            Xn = [X1]   # Xn has to be 3 dmiensional

        self.v = len(Xn)

        # standardise X and Y first
        Xn = np.array(Xn)
        for v in range(self.v):
            scaler = StandardScaler()
            scaler.fit(X[v].fillna(0))

        # transpose X, as given Xv is (N x lv)
        for v in range(self.v):
            Xn[v] = Xn[v].transpose()

        self.X = Xn     # dim = (v x lv x N)
        self.Y = constructY(Y) # dim = (N x c)
        self.N = len(self.Y)
        self.lv = [len(Xi) for Xi in self.X]
        self.theta = [1/self.v for i in range(self.v)]
        self.c = len(Y[0])
        self.W = np.array([[[1 for i in range(self.c)]
                  for j in range(self.lv[k])] for k in range(len(self.v))])  # dim = (v x l x c)
        self.E = np.array([[[1 for i in range(self.c)]
                  for j in range(self.N)] for k in range(len(self.v))])  # dim = (v x N x c)
        self.LAMBADA = self.E   # lagrange multiplier
        self.mu = 0.1           # panelty parameter
        self.zeta = 1.2

        iter = 1

        while iter <= self.T:
            iter += 1
            D1 = [0 for i in range(self.v)]  # collection of D1v for this iteration
            D2 = [0 for i in range(self.v)]  # collection of D2v
            G = [0 for i in range(self.v)]  # collection of Gv values
    
            #objective function calculation for t'th iteration
            obj_t=0
            for v in range(self.v): 
                term2 = np.matmul(np.array(self.X[v]).transpose(), self.W[v]) - self.Y
                obj_t += (self.theta**self.p) * np.sum(np.sum(term2[row]**2 for row in range(self.N))**(1/2))
            
            temp = 0
            for v in range(self.v):
                temp += np.sum(np.sum(self.W[v][row]**2 for row in range(self.lv[v]))**(1/2))

            obj_t += self.lambda_1*temp

            temp = 0
            for v in range(self.v):
                temp += LA.norm(W[v])
            
            obj_t += self.lambda_2*temp
            
            # iterating over each view
            for v in range(self.v):
                
                # D1v calculation
                D1v = [[0 for j in range(self.lv[v])] for k in range(self.lv[v])]
                for j in range(self.lv[v]):
                    for k in range(self.lv[v]):
                        if j == k:
                            sum = 0
                            for el in self.W[v][j]:
                                sum += el**2
                            D1v[j][k] = 1/(2*(sum**(1/2))

                D1[v] = D1v

                # D2v calculation
                f_norm = LA.norm(W[v])
                D2v = [[0 for j in range(self.lv[v])] for k in range(self.lv[v])]
                for j in range(self.lv[v]):
                    for k in range(self.lv[v]):
                        if j == k:
                            D2v[i][k] = f_norm

                D2[v] = D2v

                #updating projection matrix
                W[v] = self.mu * np.matmul(LA.inv(self.lambda_1 * np.array(D1[v]) + self.lambda_2 * np.array(D2[v]) + self.mu*np.matmul(self.X[v],np.array(self.X[v]).transpose())),
                                            np.matmul(self.X[v],self.Y + self.E[v] - ((1/self.mu)*self.LAMBADA[v])))

                #calculating Gv
                G[v] = np.matmul(np.array(self.X[v]).transpose(), self.W[v]) - self.Y + ((1/self.mu)*self.LAMBADA[v])

                #update slack variable E
                param = (self.theta[v] ** self.p)/self.mu
                # as Ev and Gv are nothing but as same dimension as Y, i is row no        
                for i in range(self.N):
                    gvi = np.sum(G[v][i]**2)**(1/2) # G is np array
                    if gvi > param:
                        self.E[v][i] = (1 - (param/gvi)) * G[v][i]
                    else:
                        self.E[v][i] = 0 # row as 0 elements
                
                #UPDATE LAGRANGIAN multiplier matrix
                self.LAMBADA[v] = self.LAMBADA[v] + self.mu*(np.matmul(np.array(self.X[v]).transpose(), self.W[v]) - self.Y - self.E[v]) 

                
            # update view weighted theta
            denom = 0
            for v in range(self.v):
                l21norm = np.sum(np.sum(self.E[v][row]**2 for row in range(self.N)**(1/2))
                denom += (1/l21norm)**(1/(self.p-1))

            for v in range(self.v):
                l21norm = np.sum(np.sum(self.E[v][row]**2 for row in range(self.N))**(1/2))
                self.theta[v] = ((1/l21norm)**(1/(self.p-1)))/denom

            self.mu = self.zeta*self.mu

            #objective function calculation for t+1'th iteration
            obj_t1=0
            for v in range(self.v): 
                term2 = np.matmul(np.array(self.X[v]).transpose(), self.W[v]) - self.Y
                obj_t1 += (self.theta**self.p) * np.sum(np.sum(term2[row]**2 for row in range(self.N))**(1/2))
            
            temp = 0
            for v in range(self.v):
                temp += np.sum(np.sum(self.W[v][row]**2 for row in range(self.lv[v]))**(1/2))

            obj_t1 += self.lambda_1*temp

            temp = 0
            for v in range(self.v):
                temp += LA.norm(W[v])
            
            obj_t1 += self.lambda_2*temp

            #calculating error
            er = abs(obj_t1 - obj_t)

            if er < self.epsilon:
                break

        return None

    def transform(self, X1, *Xn):
        self.W = [self.theta[v]*self.W[v] for v in range(self.v)]

