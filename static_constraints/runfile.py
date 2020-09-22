#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jul 14 09:12:12 2020

@author: kali
"""

#Importing libraries
import numpy as np
import pandas as pd
from Hard import Hard
from Soft_M import Soft_M
from Soft_C import Soft_C
from Dynamic import Dynamic
from Coello import Coello
import statistics
import time
from IPython import get_ipython

#Class Problem
class Problem():
    def __init__(self,func, lb,ub, cons=False,cons_eq=False):
        self.f = func
        self.lb = lb
        self.ub = ub
        self.cons = cons
        self.cons_eq=cons_eq       

"---------------------------------------------------------------------------------------------------------------------------------------------------------"

"Objective Functions and PSO Implementations"
                  
"---------------------------------------------------------------------------------------------------------------------------------------------------------"
"f1"
"--------------------"
def f_f1(x):
    D=np.size(x)
    return np.sum(x[i] for i in range(D))

def c1_f1(x):
    return np.sum(x)-50

def c3_f1(x):
    return np.sum(x**2)-100

"--------------------"        
"f10"
"--------------------"        
def f_f10(x):
    D=np.size(x)
    return np.sum(np.sum(x[j] for j in range(i))**2 for i in range(D))

def c1_f10(x):
    return np.sum(x-100)

def ce1_f10(x):
    return np.sum(x)

"--------------------"        
"f12"
"--------------------" 
def f_f12(x):
    D=np.size(x)
    return 10*D+ np.sum(x[i]**2-10*np.cos(2*np.pi*x[i]) for i in range(D))

def ce1_f12(x):
    return np.sum(x)-5

"--------------------"        
"f22"
"--------------------" 
def f_f22(x):
    D=np.size(x)
    return np.sum(x[i]**2 for i in range(D))

def c1_f22(x):
    return np.sum(x)-20

"--------------------"        
"f24"
"--------------------" 

def f_f24(x):
    D=np.size(x)
    return -(1+np.sum(np.sin(10*np.sqrt(np.abs(x[i]))) for i in range(D)))

def ce1_f24(x):
    return np.sum(x) - 1
"---------------------------------------------------------------------------------------------------------------------------------------------------------"

"Objective Functions and PSO Implementations"
                  
"---------------------------------------------------------------------------------------------------------------------------------------------------------"
t=pd.DataFrame(columns=["Hard","Dynamic","Soft_M","Soft_c","Coello"], 
    index=["f1","f10","f12","f22","f24"])

keys=("Hard","Dynamic","Soft_M","Soft_c","Coello")
indexs=("f1","f10","f12","f22","f24")

ubranges=([100]*30,[100]*30,[5.12]*30,[5.12]*30,[10]*30)
lbranges=([-100]*30,[-100]*30,[-5.12]*30,[-5.12]*30,[0.25]*30)
probs=(f_f1,f_f10,f_f12,f_f22,f_f24)
con=([c1_f1,c3_f1],[c1_f10],False,[c1_f22],False)
coneq=(False,[ce1_f10],[ce1_f12],False,[ce1_f24])

K=[Hard,Dynamic,Soft_M,Soft_C,Coello]


"Test matrices"
Results=pd.DataFrame(columns=["Hard","Dynamic","Soft_M","Soft_c","Coello"], 
    index=["f1","f10","f12","f22","f24"])
"-------------------------------------------------------------------"
"Runtime"
"-------------------------------------------------------------------"

t0 = time.time()
Struct=np.zeros((50, 12))
#
for k in range(len(keys)):
    for j in range(len(indexs)):
        prob=(Problem(probs[j],lbranges[j],ubranges[j],cons=con[j],cons_eq=coneq[j]))
        for i in range(0,50):
            Struct[i,0],Struct[i,1],Struct[i,2],Struct[i,9],Struct[i,10],=K[k](prob,swarmsize=20)
            print("{:} {:} {:}% complete".format(keys[k],indexs[j],((i+1)/50)*100))
        #get_ipython().magic('clear')
        Results[keys[k]][indexs[j]]="{:.2e}".format(sum(Struct[:,0])/len(Struct[:,0]))+ ' +- ' +"{:.2e}".format(statistics.stdev(Struct[:,0]))

t1 = time.time()
print(t1-t0)   
    
Results.to_csv('f1.csv')
print(Results)



