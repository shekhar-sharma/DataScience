#!/usr/bin/env python
# coding: utf-8

# <center><h1>Local Two-Dimensional Canonical Correlation Analysis</h1></center>

# In[67]:


from sklearn.cross_decomposition import CCA
import numpy as np
from tkinter import *
from tkinter import simpledialog
def mainFunction():
    root = Tk()
    row=simpledialog.askinteger("Input Rows","How many Rows you want in Matrix X")
    root.mainloop()
    
    X = []
    
    for i in range(row):
       
       row = list(map(int, input().split()))
       
       X.append(row)
    
    print(X)
    
    root = Tk()
    row1=simpledialog.askinteger("Input Rows","How many Rows you want in Matrix Y")
    root.mainloop()
    
    Y = []
    
    for i in range(row1):
       
       row = list(map(int, input().split()))
       
       Y.append(row)
    
    print(Y)
    Ax=X
    AT_x=np.transpose(X)  #Transform of matrix of A(x)
    Ax=(Ax+AT_x)/2                                             # step 14.1
    print(Ax)
    
    Ay=Y
    AT_y=np.transpose(Y)  #Transform of matrix of A(y)
    Ay=(Ay+AT_y)/2                                            # step 14.2
    print(Ay)
    
    
    Axy=Ax*Ay                                                  # step 14.3
    print(Axy)
    
    
    
    
    T_x=np.transpose(X)
    T_y=np.transpose(Y)  #transform of matrix of Y
    
    
    X_l = np.transpose(X)
    
    Y_l = np.transpose(Y)
    
    X_r= np.transpose(X)
    
    Y_r= np.transpose(Y)
    
    result1 = [[sum(a * b for a, b in zip(X_l_row, X_r_col))  
                            for X_r_col in zip(*X_r)] 
                                    for X_l_row in X_l] 
    
    
    
    finalr1 = [[sum(a * b for a, b in zip(X_row, result1_col))  
                            for result1_col in zip(*result1)]                       
                                    for X_row in X] 
       
        
    Finalr1_1=finalr1*Ax
    
    result2 = [[sum(a * b for a, b in zip(Y_l_row, Y_r_col))  
                            for Y_r_col in zip(*Y_r)] 
                                    for Y_l_row in Y_l] 
       
    
    finalr2 = [[sum(a * b for a, b in zip(Y_row, result2_col))  
                            for result2_col in zip(*result2)] 
                                    for Y_row in Y]
    
    
    Finalr2_2=finalr2*Ay
    
    
    
    mul1=Finalr1_1*Finalr2_2
    
    Coveriance=np.cov(mul1)
    
    
    value=(X_l*Coveriance*Y_l)                                     #step 15
    print(value)
    
    value2=2*(mul1)
    
    arg_max=value2
                                                  #step 16
    print(arg_max)
    
    
    
    
    
    value3=(X_r*Coveriance*Y_r)
                                                  #step 18
    print(value3)
    arg_max=value3
    print(arg_max)
    dup=[]
    for k in X_l:
        for i in k:
            dup.append(i)
            
    X_l=max(dup)
    
    dup=[]
    for k in Y_l:
        for i in k:
            dup.append(i)
            
    Y_l=max(dup)
     
    
    dup=[]
    for k in Finalr1_1:
        for i in k:
            dup.append(i)
            
    Finalr1_1=int(max(dup))
    
    
    dup=[]
    for k in Finalr2_2:
        for i in k:
            dup.append(i)
            
    Finalr2_2=int(max(dup))
    
    
    m1 = [[0,Finalr1_1], [Finalr2_2,0]]
    m2 = [[X_l], [Y_l]]
    
    final_Result = [[sum(a * b for a, b in zip(m1_row, m2_col))  
                            for m2_col in zip(*m2)] 
                                    for m1_row in m1] 
    
    print(final_Result)                                               #step 21


mainFunction()

# In[ ]:




