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
from mpi4py import MPI

#Class Problem
class Problem():
    def __init__(self,func, lb,ub, cons=False,cons_eq=False):
        self.f = func
        self.lb = lb
        self.ub = ub
        self.cons = cons
        self.cons_eq=cons_eq  

comm=MPI.COMM_WORLD
my_rank=comm.Get_rank()
p=comm.Get_size()
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
    return np.sum(x+20) 

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
#t=pd.DataFrame(columns=["Hard","Dynamic","Soft_M","Soft_c","Coello"], 
#    index=["f1","f10","f12","f22","f24"])
#
#keys=("Hard","Dynamic","Soft_M","Soft_c","Coello")
#indexs=("f1","f10","f12","f22","f24")
#
#ubranges=([100]*30,[100]*30,[5.12]*30,[5.12]*30,[10]*30)
#lbranges=([-100]*30,[-100]*30,[-5.12]*30,[-5.12]*30,[0.25]*30)
#probs=(f_f1,f_f10,f_f12,f_f22,f_f24)
#con=([c1_f1,c3_f1],[c1_f10],False,[c1_f22],False)
#coneq=(False,[ce1_f10],[ce1_f12],False,[ce1_f24])
#
#K=[Hard,Dynamic,Soft_M,Soft_C,Coello]

"Test matrices"
Results=pd.DataFrame(columns=["Hard","Dynamic","Soft_M","Soft_C","Coello"], 
    index=["f1","f10","f12","f22","f24"])


"Variable Dictionary"
VarDict = {
    1: [Hard, f_f1,[-100]*30,[100]*30,[c1_f1,c3_f1],False ], 2: [Dynamic, f_f1,[-100]*30,[100]*30,[c1_f1,c3_f1],False ], 3: [Soft_C, f_f1,[-100]*30,[100]*30,[c1_f1,c3_f1],False ],4:[Soft_M, f_f1, [-100]*30,[100]*30,[c1_f1,c3_f1],False ], 5:[Coello, f_f1,[-100]*30,[100]*30,[c1_f1,c3_f1],False ],
    
    6:[Hard, f_f10,[-100]*30,[100]*30,[c1_f10],[ce1_f10]], 7:[Dynamic, f_f10,[-100]*30,[100]*30,[c1_f10],[ce1_f10]] ,8:[Soft_C, f_f10,[-100]*30,[100]*30,[c1_f10],[ce1_f10]] ,9: [Soft_M, f_f10, [-100]*30,[100]*30,[c1_f10],[ce1_f10]],10: [Coello, f_f10,[-100]*30,[100]*30,[c1_f10],[ce1_f10]],
    
    11:[Hard, f_f12,[-5.12]*30,[5.12]*30,False,[ce1_f12]] ,12:[Dynamic, f_f12,[-5.12]*30,[5.12]*30,False,[ce1_f12]],13:[Soft_C, f_f12,[-5.12]*30,[5.12]*30,False,[ce1_f12]],14:[Soft_M, f_f12,[-5.12]*30,[5.12]*30,False,[ce1_f12]],15:[Coello, f_f12,[-5.12]*30,[5.12]*30,False,[ce1_f12]] ,   
    
    16:[Hard, f_f22,[-5.12]*30,[5.12]*30,[c1_f22],False],17:[Dynamic, f_f22,[-5.12]*30,[5.12]*30,[c1_f22],False],18:[Soft_C, f_f22,[-5.12]*30,[5.12]*30,[c1_f22],False],19:[Soft_M, f_f22,[-5.12]*30,[5.12]*30,[c1_f22],False],20:[Coello, f_f22,[-5.12]*30,[5.12]*30,[c1_f22],False],
    
    21:[Hard, f_f24,[0.25]*30,[10]*30,False,[ce1_f24]] ,22:[Dynamic, f_f24,[0.25]*30,[10]*30,False,[ce1_f24]],23:[Soft_C, f_f24,[0.25]*30,[10]*30,False,[ce1_f24]],24:[Soft_M, f_f24,[0.25]*30,[10]*30,False,[ce1_f24]],25:[Coello, f_f24,[0.25]*30,[10]*30,False,[ce1_f24]]    ,   
    }

"-------------------------------------------------------------------"
"Runtime"
"-------------------------------------------------------------------"

T0 = time.time()
Struct=np.zeros((20, 12))
#
if my_rank!=0:
    prob=(Problem(VarDict[my_rank][1],VarDict[my_rank][2],VarDict[my_rank][3],cons=VarDict[my_rank][4],cons_eq=VarDict[my_rank][5]))
    for i in range(20):
        Struct[i,0],Struct[i,1],Struct[i,2],Struct[i,9],Struct[i,10],=VarDict[my_rank][0](prob,swarmsize=20)
    a="{:}".format(VarDict[my_rank][0].__name__)
    f="{:}".format(VarDict[my_rank][1].__name__.split('_')[1])        
    val="{:.2e}".format(sum(Struct[:,0])/len(Struct[:,0]))+ ' +- ' +"{:.2e}".format(statistics.stdev(Struct[:,0]))
    comm.send([a,f,val],dest=0)
    t1 = time.time()
    print("[Sent] {:} {:} ,rank {:}-> 0 \n Runtime: {:}".format(a,f,my_rank,t1-T0))
if my_rank==0:
   t0=time.time()
   for source in range(1,p):
       out=comm.recv(source=source)
       Results[out[0]][out[1]]=out[2]
       t1 = time.time()
       print("Received 0 <- {:} {:}, source {:}".format(out[0],out[1],source))
   T1=time.time()
   print(" -------------------------------------------------------------------- \n Jirre, task complete! \n Total runtime = {:}s \n --------------------------------------------------------------------".format((T1-T0)%60))   
   print(Results)
MPI.Finalize


