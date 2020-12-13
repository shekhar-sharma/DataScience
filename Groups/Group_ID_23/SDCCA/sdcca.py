import pandas as pd
import numpy as np
import math
from sklearn import decomposition,preprocessing
from sklearn.neighbors import NearestNeighbors

def _center_scale_xy(x, y, scale=True):
    if scale == True:
        return StandardScaler().fit_transform(x),StandardScaler().fit_transform(y)
    return x,y

class SDCCA :

    def __init__(self,n_components = 2,scale = True,k=2):
        self.n_components = n_components
        self.scale = scale
        self.k = k

    def __make_dimensionlity_same(self,x,y,p,q) :
        pca = decomposition.PCA(n_components=min(p,q))
        if p < q :
            pc = pca.fit_transform(y)
            return x,pd.DataFrame(pc),p,p
        else :
            pc = pca.fit_transform(x)
            return pd.DataFrame(pc),y,q,q

    def __bhattacharya_similarity_coeff(self,x,y) :
        return 1-np.sum(np.sqrt(x*y),axis = 1)

    def __nearest_neighbor(self,x,y) :
        neigh = NearestNeighbors(n_neighbors=self.k)
        neigh.fit(x)
        x_neigh = neigh.kneighbors(x,return_distance = False)

        neigh.fit(y)
        y_neigh = neigh.kneighbors(y,return_distance = False)

        return x_neigh,y_neigh

    def fit(self,x,y) :
        n = x.shape[0]
        p = x.shape[1]
        q = y.shape[1]

        if p != q :
            x,y,p,q = self.__make_dimensionlity_same(x,y,p,q)

        # print("original data")
        # print(x)
        # print(y)

        x, y = _center_scale_xy(x, y, self.scale)

        # print("normalized data")
        # print(x)
        # print(y)

        x_neigh,y_neigh = self.__nearest_neighbor(x,y)

        # print("x neighbours : ")
        # print(x_neigh)
        # print("y neighbours : ")
        # print(y_neigh)

        tx = 0
        ty = 0
        sx = np.zeros((n,n))
        sy = np.zeros((n,n))

        for i in range(n) :
            for j in range(n) :
                tx += (2*(np.linalg.norm(x.iloc[i]-x.iloc[j],2)**2))/(n*(n-1))

        for i in range(n) :
            for j in range(n) :
                ty += (2*(np.linalg.norm(y.iloc[i]-y.iloc[j],2)**2))/(n*(n-1))

        # print("tx")
        # print(tx)
        # print("ty")
        # print(ty)

        for i in range(n) :
            for j in range(n) :
                if j in x_neigh[i] :
                    sx[i][j] = np.exp(-(np.linalg.norm(x.iloc[i]-x.iloc[j],2)**2)/tx)
                else :
                    sx[i][j] = 0

        for i in range(n) :
            for j in range(n) :
                if j in y_neigh[i] :
                    sy[i][j] = np.exp(-(np.linalg.norm(y.iloc[i]-y.iloc[j],2)**2)/ty)
                else :
                    sy[i][j] = 0

        sxy = self.__bhattacharya_similarity_coeff(x,y)

        # print("sx sy sxy")
        # print(sx)
        # print(sy)
        # print(sxy)

        s = np.identity(n)+sx+sy+sxy.values

        # print("s")
        # print(s)

        cxy = np.dot(np.dot(x,y.transpose()),s)
        cxx = np.dot(x,x.transpose())
        cyy = np.dot(y,y.transpose())

        # print("cxy")
        # print(cxy)
        # print("cxx")
        # print(cxx)
        # print("cyy")
        # print(cyy)

        # print("cxx**-0.5")
        # print(cxx**(-0.5))
        # print("cyy**-0.5")
        # print(cyy**(-0.5))

        temp = np.dot(cxx**(-0.5), cxy)
        h = np.dot(temp,cyy**(-0.5))

        # print("h")
        # print(h)

        u,d,vh = np.linalg.svd(h,full_matrices = True)

        wx = np.dot(cxx**(-0.5), u)
        wy = np.dot(cyy**(-0.5), vh.transpose())

        return wx,wy

x = pd.DataFrame([[0., 0., 1.], [1.,0.,0.], [2.,2.,2.], [3.,5.,4.]])
y = pd.DataFrame([[0.1, 0.2], [0.9, 1.1], [6.2, 5.9], [11.9, 12.3]])
# x = pd.DataFrame([[1,2,4],[4,8,16],[7,14,18]])
# y = pd.DataFrame([[1,3,9],[2,6,18],[3,9,27]])
# print(x.corr())
# print(y.corr())

sdcca = SDCCA(1,False)
print(sdcca.fit(x,y))
