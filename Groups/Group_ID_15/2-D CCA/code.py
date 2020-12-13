# -*- coding: utf-8 -*-
"""
Created on Fri Dec 11 13:14:44 2020

@author: Lokesh
"""

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
    
    
    T_x=np.transpose(X)  #Transform of matrix of X
    
    
    
    T_y=np.transpose(Y)  #transform of matrix of Y
    
    
    M_x=np.mean(T_x)  #mean of matrix 
    
    
    M_y=np.mean(T_y)
    
    
    
    X_t=T_x-M_x                                    #finding correlation
    Y_t=T_y-M_y
    
    
    
    
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
       
    
    
    
    result2 = [[sum(a * b for a, b in zip(Y_l_row, Y_r_col))  
                            for Y_r_col in zip(*Y_r)] 
                                    for Y_l_row in Y_l] 
       
    
    
    
    
    
    finalr2 = [[sum(a * b for a, b in zip(Y_row, result2_col))  
                            for result2_col in zip(*result2)] 
                                    for Y_row in Y] 
       
                                                                
    
    cov_mat = np.stack((finalr1), axis = 0)  
      
    p=np.cov(cov_mat)
    print(p)
    
    cov_mat = np.stack((finalr2), axis = 0)  
      
    q=np.cov(cov_mat)
    print(q)
    finalr1=1                                    #print max covariance between left and right transform matrix w.r.t to X And Y 
    finalr2=1                                                             
    ry_t=np.transpose(Y_r)
    
    YT_t=np.transpose(Y_t)
    
    
    XT_t=np.transpose(X_t)
    
    
    rx_t=np.transpose(X_r)
    
    
    p1= [[sum(a * b for a, b in zip(X_t_row, X_r_col))  
                            for X_r_col in zip(*X_r)] 
                                    for X_t_row in X_t] 
       
    
    
    p2= [[sum(a * b for a, b in zip(ry_t_row, YT_t_col))  
                            for YT_t_col in zip(*YT_t)] 
                                    for ry_t_row in ry_t] 
    
    
    
    Fp3 = [[sum(a * b for a, b in zip(p1_row, p2_col))  
                            for p2_col in zip(*p2)] 
                                    for p1_row in p1] 
    
    q1= [[sum(a * b for a, b in zip(X_t_row, X_r_col))  
                            for X_r_col in zip(*X_r)] 
                                    for X_t_row in X_t] 
       
    
    
    q2= [[sum(a * b for a, b in zip(rx_t_row, XT_t_col))  
                            for XT_t_col in zip(*XT_t)] 
                                    for rx_t_row in rx_t] 
       
    
    
    Fq3 = [[sum(a * b for a, b in zip(q1_row, q2_col))  
                            for q2_col in zip(*q2)] 
                                    for q1_row in q1] 
    
    
    
    s1= [[sum(a * b for a, b in zip(Y_t_row, X_r_col))  
                            for X_r_col in zip(*X_r)] 
                                    for Y_t_row in Y_t] 
       
    
    
    s2= [[sum(a * b for a, b in zip(rx_t_row, XT_t_col))  
                            for XT_t_col in zip(*XT_t)] 
                                    for rx_t_row in rx_t] 
       
    
    Fs3 = [[sum(a * b for a, b in zip(q1_row, q2_col))  
                            for q2_col in zip(*q2)] 
                                    for q1_row in q1] 
    
    
    cov_mat = np.stack((Fp3), axis = 0)  
      
    R_xy=np.cov(cov_mat)
    print(R_xy)                                                   #print Cov(x,y) for right transform
    
    cov_mat = np.stack((Fq3), axis = 0)  
      
    R_xx=np.cov(cov_mat)
    print(R_xx)                                                   #print Cov(x,x) for right transform
    
    cov_mat = np.stack((Fs3), axis = 0)  
      
    R_yy=np.cov(cov_mat)
    print(R_yy)                                                  #print Cov(y,y) for right transform
    
    XL_t = np.transpose(X_l)
    
    YL_t = np.transpose(Y_l)
    
    
    l1= [[sum(a * b for a, b in zip(XL_t_row, Fp3_col))  
                            for Fp3_col in zip(*Fp3)] 
                                    for XL_t_row in XL_t] 
       
    
    
    Fl1 = [[sum(a * b for a, b in zip(l1_row, Y_l_col))  
                            for Y_l_col in zip(*Y_l)] 
                                    for l1_row in l1] 
    
    
    l2= [[sum(a * b for a, b in zip(XL_t_row, Fq3_col))  
                            for Fq3_col in zip(*Fq3)] 
                                    for XL_t_row in XL_t] 
       
    
    
    Fl2 = [[sum(a * b for a, b in zip(l2_row, X_l_col))  
                            for X_l_col in zip(*X_l)] 
                                    for l2_row in l2] 
    
    
    l3= [[sum(a * b for a, b in zip(YL_t_row, Fs3_col))  
                            for Fs3_col in zip(*Fs3)] 
                                    for YL_t_row in YL_t] 
       
    
    
    Fl3 = [[sum(a * b for a, b in zip(l3_row, Y_l_col))  
                            for Y_l_col in zip(*Y_l)] 
                                    for l3_row in l3] 
    
    
    Fl1=Fl2=Fl3=1                                                        #set value of all cov matrices of lest value to 1
    
    m1= [[sum(a * b for a, b in zip(XT_t_row, X_l_col))  
                            for X_l_col in zip(*X_l)] 
                                    for XT_t_row in XT_t] 
       
    
    
    m2= [[sum(a * b for a, b in zip(YL_t_row, Y_t_col))  
                            for Y_t_col in zip(*Y_t)] 
                                    for YL_t_row in YL_t] 
       
    
    
    Fm3 = [[sum(a * b for a, b in zip(m1_row, m2_col))  
                            for m2_col in zip(*m2)] 
                                    for m1_row in m1] 
    
    
    n1= [[sum(a * b for a, b in zip(XT_t_row, X_l_col))  
                            for X_l_col in zip(*X_l)] 
                                    for XT_t_row in XT_t] 
       
    
    
    n2= [[sum(a * b for a, b in zip(XL_t_row, X_t_col))  
                            for X_t_col in zip(*X_t)] 
                                    for XL_t_row in XL_t] 
       
    
    
    Fn3 = [[sum(a * b for a, b in zip(n1_row, n2_col))  
                            for n2_col in zip(*n2)] 
                                    for n1_row in n1] 
    
    
    o1= [[sum(a * b for a, b in zip(YT_t_row, Y_l_col))  
                            for Y_l_col in zip(*Y_l)] 
                                    for YT_t_row in YT_t] 
       
    
    
    o2= [[sum(a * b for a, b in zip(YL_t_row, Y_t_col))  
                            for Y_t_col in zip(*Y_t)] 
                                    for YL_t_row in YL_t] 
       
    
    
    Fo3 = [[sum(a * b for a, b in zip(o1_row, o2_col))  
                            for o2_col in zip(*o2)] 
                                    for o1_row in o1] 
    
    
    
    cov_mat = np.stack((Fm3), axis = 0)  
      
    L_xy=np.cov(cov_mat)
    print("Cov(x,y) for left transform")
    print(L_xy)                                                       #print Cov(x,y) for left transform                                                      
    
    cov_mat = np.stack((Fn3), axis = 0)  
      
    L_xx=np.cov(cov_mat)
    print(" Cov(x,y) for left transform")
    print(L_xx)                                                    #print Cov(x,y) for left transform
    
    cov_mat = np.stack((Fo3), axis = 0)  
      
    L_yy=np.cov(cov_mat)
    print("Cov(x,y) for left transform")
    print(L_yy)                                                #print Cov(x,y) for left transform
    
    XR_t = np.transpose(X_r)
    
    YR_t = np.transpose(Y_r)
    
    
    r1= [[sum(a * b for a, b in zip(XR_t_row, Fm3_col))  
                            for Fm3_col in zip(*Fm3)] 
                                    for XR_t_row in XR_t] 
       
    
    
    Fr1 = [[sum(a * b for a, b in zip(r1_row, Y_r_col))  
                            for Y_r_col in zip(*Y_r)] 
                                    for r1_row in r1] 
    
    
    r2= [[sum(a * b for a, b in zip(XR_t_row, Fn3_col))  
                            for Fn3_col in zip(*Fn3)] 
                                    for XR_t_row in XR_t] 
       
    
    
    Fr2 = [[sum(a * b for a, b in zip(r2_row, X_r_col))  
                            for X_r_col in zip(*X_r)] 
                                    for r2_row in r2] 
    
    
    r3= [[sum(a * b for a, b in zip(YR_t_row, Fo3_col))  
                            for Fo3_col in zip(*Fo3)] 
                                    for YR_t_row in YR_t] 
       
    
    
    Fr3 = [[sum(a * b for a, b in zip(r3_row, Y_r_col))  
                            for Y_r_col in zip(*Y_r)] 
                                    for r3_row in r3] 
    
    
    Fr1=Fr2=Fr3                                                                 #set value of all cov matrices of right transform value to 1
    #max_cov= max(p,q)
    #print(max_cov)


mainFunction()