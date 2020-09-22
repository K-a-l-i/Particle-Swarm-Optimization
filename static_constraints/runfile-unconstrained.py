



#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Sep  3 10:30:41 2020

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
from Static import Static
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
    return np.sum(np.abs(x[i]) for i in range(D))

"--------------------"        
"f10"
"--------------------"        
def f_f10(x):
    D=np.size(x)
    val=np.empty((1,D))
    for i in range(D):
        for j in range(i):
            val[0][i]=np.sum((x[j]**2))
    return np.sum(val)


"--------------------"        
"f12"
"--------------------" 
def f_f12(x):
    D=np.size(x)
    return 10*D+ np.sum(x[i]**2-10*np.cos(2*np.pi*x[i]) for i in range(D))


"--------------------"        
"f22"
"--------------------" 
def f_f22(x):
    return np.sum(x)**2


"--------------------"        
"f24"
"--------------------" 

def f_f24(x):
    D=np.size(x)
    return -(1+np.sum(np.sin(10*np.sqrt(np.abs(x[i]))) for i in range(D)))
  

"---------------------------------------------------------------------------------------------------------------------------------------------------------"

"Objective Functions and PSO Implementations"
                  
"---------------------------------------------------------------------------------------------------------------------------------------------------------"

keys=("Hard","Dynamic","Soft_M","Soft_c","Coello","Static")
lvls=("Mean","Min","% Feasibility","avg Count","avg Mag")
indexs=('f1','f10','f12','f22','f24')

arrays=[np.array(["Hard","Hard","Hard","Hard","Hard","Dynamic","Dynamic","Dynamic","Dynamic","Dynamic","Soft_M","Soft_M","Soft_M","Soft_M","Soft_M","Soft_c","Soft_c","Soft_c","Soft_c","Soft_c","Coello","Coello","Coello","Coello","Coello","Static","Static","Static","Static","Static",]),np.array(["Mean","Min","% Feasibility","avg Count","avg Mag","Mean","Min","% Feasibility","avg Count","avg Mag","Mean","Min","% Feasibility","avg Count","avg Mag","Mean","Min","% Feasibility","avg Count","avg Mag","Mean","Min","% Feasibility","avg Count","avg Mag","Mean","Min","% Feasibility","avg Count","avg Mag","Mean","Min","% Feasibility","avg Count","avg Mag"])]

tuples = list(zip(*arrays))
index = pd.MultiIndex.from_tuples(tuples, names=['Functions', 'Statistics'])


ubranges=([100]*30,[100]*30,[5.12]*30,[5.12]*30,[10]*30)
lbranges=([-100]*30,[-100]*30,[-5.12]*30,[-5.12]*30,[0.25]*30)
probs=(f_f1,f_f10,f_f12,f_f22,f_f24)


K=[Hard,Dynamic,Soft_M,Soft_C,Coello,Static]


"Test matrices"
Results=pd.DataFrame(columns=["Hard","Dynamic","Soft_M","Soft_c","Coello","Static"], index=["f1","f10","f12","f22","f24"])
Results = pd.DataFrame(index=['f1', 'f10', 'f12','f22','f24'], columns=index)
"-------------------------------------------------------------------"
"Runtime"
"-------------------------------------------------------------------"

t0 = time.time()
Struct=np.zeros((20,6),dtype=object)
Mean=[]
for k in range(len(keys)):
    for j in range(len(indexs)):
        prob=(Problem(probs[j],lbranges[j],ubranges[j]))
        for i in range(0,20):
            Struct[i,0],Struct[i,1],Struct[i,2],Struct[i,3],Struct[i,4],Struct[i,5],=K[k](prob,swarmsize=20)
            print("{:} {:} {:}% complete".format(keys[k],indexs[j],((i+1)/20)*100))
            if Struct[i,0]!=np.inf:
                Mean.append(Struct[i,0])
        if len(Mean)==0:
                Results.loc[indexs[j],(keys[k],lvls[0])]="no feasible soln."
        else:
            Results.loc[indexs[j],(keys[k],lvls[0])]="{:.2e}".format(statistics.mean(Mean))+ ' +- ' +"{:.2e}".format(statistics.stdev(Mean))
        Results.loc[indexs[j],(keys[k],lvls[2])]=statistics.mean(Struct[:,5])*100
        Results.loc[indexs[j],(keys[k],lvls[3])]=statistics.mean(Struct[:,1])
        Results.loc[indexs[j],(keys[k],lvls[4])]=statistics.mean(Struct[:,2])

t1 = time.time()
print(t1-t0)   
    
print(Results)

p=[0,0,0,0,0]
l=0
for i in indexs:
    for j in keys:
        Results.loc[i,(j,'Min')]=p[l]
    l=l+1


Results.to_csv('f1.csv')
















