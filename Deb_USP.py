#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jul 17 10:10:56 2020

@author: kali
"""

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Apr  4 09:42:08 2020

    Perform a particle swarm optimization (PSO)
   
    Parameters
    ==========
    func : function
        The function to be minimized
    lb : array
        The lower bounds of the design variable(s)
    ub : array
        The upper bounds of the design variable(s)
   
    Optional
    ========
    swarmsize : int
        The number of particles in the swarm (Default: (4 + floor(3*log(D)), where D is parameter dimmension)
    omega : scalar
        Particle velocity scaling factor (Default: 0.5)
    phip : scalar
        Scaling factor to search away from the particle's best known position
        (Default: 0.5)
    phig : scalar
        Scaling factor to search away from the swarm's best known position
        (Default: 0.5)
    maxiter : int
        The maximum number of iterations for the swarm to search (Default: 100)
    minstep : scalar
        The minimum stepsize of swarm's best position before the search
        terminates (Default: 1e-8)
    minfunc : scalar
        The minimum change of swarm's best objective value before the search
        terminates (Default: 1e-8)
    
"""
#-----------------------------------------------------------------------------------------------------------------------------

#Loading dependancies

import numpy as np
import math
#-----------------------------------------------------------------------------------------------------------------------------

# Creating a few functions to be used within pso   
def mag_cons(constraints,x):
    mag=0
    for constraint in constraints:
        if constraint(x)>0:
            mag=mag + np.abs(constraint(x))
    return mag

def count_cons(constraints,x):
    count=0
    for constraint in constraints:
        if np.any(constraint(x)>0):
            count=count+1
    return count

def mag_cons_eq(constraints,x):
    mag=0
    for constraint in constraints:
        if constraint(x)!=0:
            mag=mag + np.abs(constraint(x))
    return mag

def count_cons_eq(constraints,x):
    count=0
    for constraint in constraints:
        if constraint(x)!=0:
            count=count+1
    return count
    
 
def is_feasible(constraints, x):
    temp=[]
    for constraint in constraints:
        temp.append(constraint(x)<=0)
    return np.all(temp)  

def is_feasible_eq(constraints,x):
    temp=[]
    for constraint in constraints:
        temp.append(constraint(x)==0)
    return np.all(temp) 

def usp(p,d,mode):
    
    normal_deviates = np.random.normal(size=(1,d))[0]
    radius = np.sqrt((normal_deviates**2).sum(axis=0)) 
    dist =  np.sqrt(np.sum((mode - p) ** 2))
               
    point = (normal_deviates/radius)*(dist/2) + mode
    
    return point

def C(constraints, x):
    c=[0,0,0]
    for constraint in constraints:
        if constraint(x)<1e-2:
            c[2]+=1
        elif constraint(x)<1e-0:
            c[2]+=1
            c[1]+=1
        else:
            c[2]+=1
            c[1]+=1
            c[0]+=1
    return c
# def ghost_usp(d,mode,mode_mag):
#     normal_deviates = np.random.normal(size=(1,d))[0]
#     radius = np.sqrt((normal_deviates**2).sum(axis=0)) 
               
#     point = (normal_deviates/radius)*(mode_mag/2) + mode
    
#     return point
    
   

#-----------------------------------------------------------------------------------------------------------------------------

def Deb_USP(func, swarmsize=-1, omega=0.7,phip=1.49445, phig=1.49445, maxiter=100, minstep=1e-12, minfunc=1e-12, debug=False, processes=1,particle_output=False):
    lb=[]
    ub=[]
    for l in func.lb:
        lb+=l
    lb=np.array(lb)
    for u in func.ub:
        ub+=u
    ub = np.array(ub)
    
    assert len(lb)==len(ub), 'Lower- and upper-bounds must be the same length'
    assert np.all(ub>lb), 'All upper-bound values must be greater than lower-bound values'

    if swarmsize<1:
        swarmsize= 4 + math.floor(3 * math.log(len(lb)))
    
    # Check for constraint function(s) #########################################
    
    cons=func.cons
    cons_eq = func.cons_eq
    
    #setting up count and magnitude functions based on constraints
    if np.all(cons) and np.all(cons_eq):
        def mag(x):
            return mag_cons(cons,x)+mag_cons_eq(cons_eq,x)
        def count(x):
            return count_cons(cons,x)+count_cons_eq(cons_eq,x)
        def feasible(x):
            return is_feasible(cons,x) and is_feasible_eq(cons_eq,x)

    elif np.all(cons):
        def mag(x):
            return mag_cons(cons,x)
        def count(x):
            return count_cons(cons,x)
        def feasible(x):
            return is_feasible(cons,x)
    elif np.all(cons_eq):
        def count(x):
            return count_cons_eq(cons_eq,x)
        def mag(x):
            return mag_cons_eq(cons_eq,x)
        def feasible(x):
            return is_feasible_eq(cons_eq,x)
    else:
        def mag(x):
            return 0
        def count(x):
            return 0
        def feasible(x):
            return True

     # Initialize the particle swarm ############################################
    S = swarmsize
    D = len(lb)  # the number of dimensions each particle has
    #x = np.random.rand(S, D)  # particle positions
    x=np.zeros((S,D))
    for i in range(S):
        for j in range(D):
            x[i,j] = np.random.uniform(lb[j],ub[j])
    v = np.zeros_like(x)  # particle velocities
    fx = np.zeros(S)  # current particle function values
    #fs = np.zeros(S, dtype=bool)  # feasibility of each particle
    fp = np.ones(S)*np.inf  # best particle function values
    g = []  # best swarm position
    fg = np.inf  # best swarm position starting value
    gmag =float(np.inf) #Magnitude of constraint violations by global best particle
    fxmag = np.ones(S)*np.inf #Magnitude of constraint violations by current particles
    fx_c=np.zeros(S) #number of constraints violated by each particle
    fc =np.inf #number of constraints violated by global best particle
    
    for i in range(S):
        print(i)
        fx[i] = func.f(x[i, :])
        fxmag[i]=mag(x[i,:])
        fx_c[i]=count(x[i,:])

    p=x
    fp=fx
    fpmag=fxmag
    fp_c=fx_c
    i_min=np.argmin(fpmag)
    # # Update swarm's best position
    # fease= np.where(fpmag==0)
    # if np.size(fease)!=0:
    #     dictionary ={i:fp[i] for i in fease[0]}
    #     i_min= list(dictionary.keys())[list(dictionary.values()).index(min(dictionary.values()))]
    # else:
    #     i_min=np.argmin(fpmag)
    
    fg = fp[i_min]
    g = p[i_min, :].copy()
    gmag=fpmag[i_min].copy()
    fc=fp_c[i_min].copy()
        
    # print(' {} {}'.format(fg,gmag))
    # Initialize the particle's velocity
#    v = vlow + np.random.rand(S, D)*(vhigh - vlow)
    # Iterate until termination criterion met ##################################
    it=1
    while it <= maxiter:
        rp = np.random.uniform(size=(S, D))
        rg = np.random.uniform(size=(S, D))
        phip = 2- it/maxiter
        phig = 1+ it/maxiter
        omega = 0.9 - 0.5*it/maxiter
        # Update the particles velocities
        v = omega*v + phip*rp*(p - x) + phig*rg*(g - x)
        
        # #indices of max personal best values based on magnitude of violations
        inds = np.argsort(fpmag)[-int(np.floor(S/2)):]
        
        # Update the particles' positions
        x = x + v
        
        # #initialize random points
        for i in inds:
                x[i,:]=usp( x[i,:],D,g)
        #Correct for bound violations
        maskl = x < lb
        masku = x > ub
        x = x*(~np.logical_or(maskl, masku)) + lb*maskl + ub*masku

        # Update objectives and constraints
        for i in range(S):
            fx[i] = func.f(x[i, :])
            fxmag[i]=mag(x[i,:])
            fx_c[i]=count(x[0,:])
        # Store particle's best position (if constraints are satisfied)
        # i_update = np.logical_and(fx <fp, np.any(fxmag <= fpmag and fx_c <= fp_c))
        i_update=np.full((1,S), False, dtype=bool)[0]
        
        for i, m in enumerate(fxmag):
               if  fxmag[i]<=fpmag[i]: #and fx_c[i]<=fp_c[i]
                   i_update[i]=True
                
        if any(i_update):
              p[i_update, :] = x[i_update, :].copy()
              fp[i_update] = fx[i_update]
              fpmag[i_update]=fxmag[i_update]
              fp_c[i_update]=fx_c[i_update]
              fease= np.where(fpmag==0)
              if np.size(fease)!=0:
                  dictionary ={i:fp[i] for i in fease[0]}
                  i_min= list(dictionary.keys())[list(dictionary.values()).index(min(dictionary.values()))]
              else:
                  i_min=np.argmin(fpmag)
       
        
        if fpmag[i_min] <=gmag: #and fp_c[i_min] <= fc 
                 gmag=fpmag[i_min]
                 g=p[i_min, :].copy()
                 fg = fx[i_min]
                    
        it+=1       
            
        
    avgmag=gmag/len(cons)  
#    print('Stopping search: maximum iterations reached --> {:}'.format(maxiter))
    if gmag>0:
        Fease=1
    else:
        Fease=0
    return fg, gmag,count(g),Fease,C(cons,g),avgmag
   
    

    
    


