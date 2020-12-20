import numpy as np
import gcca

def generat_data(range_val = 100):
    a = np.random.randint(1,range_val,(3, 3))
    b = np.random.randint(1,range_val,(3, 3))
    
    return a,b

def main():

    #create data in advance
    a,b = generat_data()
    print(a)
    print(b)

    # create instance of GCCA
    Gcca = gcca.GCCA()
    # calculate GCCA
    #gcca.fit(a, b)
    # transform
    #gcca.transform(a, b)
    wx, wy = Gcca.fit_transform(a,b)
    
    print(wx)
    print(wy)

if __name__=="__main__":
    main()
