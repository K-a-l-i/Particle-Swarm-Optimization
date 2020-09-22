#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jun 17 14:35:52 2020

@author: kali
"""

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
#
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
        if constraint(x)>0:
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
        temp.append(np.all(constraint(x)<=0))
    return np.all(temp)  

def is_feasible_eq(constraints,x):
    temp=[]
    for constraint in constraints:
        temp.append(np.all(constraint(x)==0))
    return np.all(temp) 
     
#-----------------------------------------------------------------------------------------------------------------------------

def Hard(func, swarmsize=-1, w=0.7, c1=1.4, c2=1.4, maxiter=2000, minstep=1e-12, minfunc=1e-12, debug=False, processes=1,particle_output=False):
    
    #handling of lower and upper bounds
    lb=np.array(func.lb)
    ub=np.array(func.ub)
    assert len(lb)==len(ub), 'Lower- and upper-bounds must be the same length'
    #assert hasattr(func, '__call__'), 'Invalid function handle'
    assert np.all(ub>lb), 'All upper-bound values must be greater than lower-bound values'  
    
     #Setting default swarmsize if one is not given
    if swarmsize<1:
        swarmsize= 4 + math.floor(3 * math.log(len(lb)))
        
    # Initialize the particle swarm ############################################
    S = swarmsize
    D = len(lb)  # the number of dimensions each particle has
    
    #x = np.random.rand(S, D)  # particle positions
    x=np.random.uniform(low=func.lb[0],high=func.ub[0],size=(S,D))
    v = np.zeros_like(x)  # particle velocities   
    fx = np.zeros(S)  # current particle function values
    fs = np.ones(S, dtype=bool)  # feasibility of each particle
    fp = np.ones(S)*np.inf  # best particle function values
    p = np.ones_like(x)*np.inf  # best particle values
    g = []  # best swarm position
    fg = np.inf  # best swarm position starting value
#    
    # Initialize the particle's position
    #x = lb + x*(ub - lb)
    #x = np.random.uniform(-1,1)*ub*x + 0.5*(lb+ub)
      
    # Check for constraint function(s) #########################################
    cons=func.cons
    cons_eq=func.cons_eq

    #setting up count, magnitude and feasibility functions based on constraints
    if cons and cons_eq:
        def mag(x):
            return mag_cons(cons,x)+mag_cons_eq(cons_eq,x)
        def count(x):
            return count_cons(cons,x)+count_cons_eq(cons_eq,x)
        def feasible(x):
            return is_feasible(cons,x) and is_feasible_eq(cons_eq,x)

    elif cons:
        def mag(x):
            return mag_cons(cons,x)
        def count(x):
            return count_cons(cons,x)
        def feasible(x):
            return is_feasible(cons,x)
        
    elif cons_eq:
        def count(x):
            return count_cons_eq(cons_eq,x)
        def mag(x):
            return mag_cons_eq(cons_eq,x)
        def feasible(x):
            return is_feasible_eq(cons_eq,x)
        
    else:
        def feasible(x):
            return True
        def mag(x):
            return 0 
        def count(x):
            return 0
    
    #calculating feasibility and function values for each particle
    for i in range(S):
        fx[i]=func.f(x[i,:])
        fs[i]=feasible(x[i,:])
     
    # Store particle's best position (if constraints are satisfied)
    i_update = np.logical_and((fx < fp), fs)
    p[i_update, :] = x[i_update, :].copy()
    fp[i_update] = fx[i_update]

    # Update swarm's best position
    i_min = np.argmin(fp)
    if fp[i_min] < fg:
        fg = fp[i_min]
        g = p[i_min, :].copy()
    else:
        # At the start, there may not be any feasible starting point, so just
        # give it a temporary "best" point since it's likely to change
        g = x[0, :].copy()
    
      
    #Setting initial velocities
    #vhigh = np.abs(ub - lb)
    #vlow = -vhigh
    # Initialize the particle's velocity
    #v = vlow + np.random.rand(S, D)*(vhigh - vlow)
       
    # Iterate until termination criterion met ##################################
    it = 1
    while it <= maxiter:
        rp = np.random.uniform(size=(S, D))
        rg = np.random.uniform(size=(S, D))
      
        # Update the particles velocities
        v = w*v + c1*rp*(p - x) + c2*rg*(g - x)
        # Update the particles' positions
        x = x + v
        
        # Update objectives and constraints
        for i in range(S):
            fx[i]=func.f(x[i,:])
            fs[i]=feasible(x[i,:])
        
           
        # Store particle's best position (if constraints are satisfied)
        i_update = np.logical_and((fx < fp), fs)
        p[i_update, :] = x[i_update, :].copy()
        fp[i_update] = fx[i_update]

        # Compare swarm's best position with global best position
        i_min = np.argmin(fp)
        p_min = p[i_min, :].copy()

        e,E=0,0
        if fp[i_min] < fg:
            stepsize = np.sqrt(np.sum((g - p_min)**2))

            if np.abs(fg - fp[i_min]) <= minfunc:
#                print('Stopping search: Swarm best objective change less than {:}'\
#                    .format(minfunc))
                e=1            
                return fp[i_min],count(p_min),mag(p_min),e,E,int(feasible(p_min))
            elif stepsize <= minstep:
                E=1
#                print('Stopping search: Swarm best position change less than {:}'\
#                    .format(minstep))                
                return fp[i_min],count(p_min),mag(p_min),e,E,int(feasible(p_min))
            else:
                g = p_min.copy()
                fg = fp[i_min]
        it += 1
    
#    print('Stopping search: maximum iterations reached --> {:}'.format(maxiter))
    
    if not feasible(g):
     
#        print("However, the optimization couldn't find a feasible design. Sorry")
        return fg,count(g),mag(g),e,E,int(feasible(g))
    else:
        return fg,count(g),mag(g),e,E,int(feasible(g))
    



    
    
    
