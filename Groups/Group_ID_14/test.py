import numpy as np
import pandas as pd
from OCCA import OCCA

'''
Generating 2 random dataset each of
which containing 100 samples and have feature
dimensionality of 10 and 25 respectively
'''

def occa_reduced_features(X,Y):
    occa=OCCA(n_components=4, complex_=False)
    occa.fit(X,Y)
    X_reduced,Y_reduced=occa.transform(X,Y)
    print("Reduced X shape",X_reduced.shape,"Reduced Y shape",Y_reduced.shape)
    print(X_reduced)
    print(Y_reduced)
    '''
        Alternatively, instead of fit and then usingb transform,
        fit_transform could also be used which directly fit the
        model on data and transform it
    '''
    return


X=np.random.rand(100,10)
Y=np.random.rand(100,25)
occa_reduced_features(X,Y)
