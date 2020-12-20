# -*- coding: utf-8 -*-
"""
Created on Sun Dec 20 19:02:06 2020

@author: Prajal Badjatya
"""

import numpy as np
import bcca

#function for generating the input data
def data_generation(range_val = 100):
    a = np.random.randint(1,range_val,(3, 3))
    b = np.random.randint(1,range_val,(3, 3))
    
    return a,b

def main():

    #Data creation
    a,b = data_generation()
    print(a)
    print(b)

    # create instance of BCCA
    Bcca = bcca.BCCA()
    #calling method fit_transform
    Wa, Wb = Bcca.fit_transform(a,b)
    
    #printing final weights
    print(Wa)
    print(Wb)

if __name__=="__main__":
    main()