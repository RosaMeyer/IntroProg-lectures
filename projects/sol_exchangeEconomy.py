# imports
from types import SimpleNamespace

import numpy as np
from scipy import optimize

import pandas as pd 
import matplotlib.pyplot as plt

colors = plt.rcParams['axes.prop_cycle'].by_key()['color']

plt.rcParams.update({"axes.grid":True,"grid.color":"black","grid.alpha":"0.25","grid.linestyle":"--"})
plt.rcParams.update({'font.size': 14})

# a. initialize the model and set parameters
def initialize_model():
    par = SimpleNamespace()
    sol = SimpleNamespace()
    sim = SimpleNamespace()

    # i. preferences
    par.alpha = 1/3
    par.beta = 2/3

    # ii. endowments
    par.w1A = 0.8
    par.w2A = 0.3

    # iii. grids and allocation
    par.N = 75
    par.P1_grid = 0.5 + 2*np.linspace(0, 1, par.N+1)
    par.eps_grid = np.empty((2, par.N+1))

    # iv. solution parameters
    par.tol = 1e-8
    par.maxiter = 500
    par.adj = 0.5
    sol.solved = False

    return par, sol, sim

# b. define utility and demand functions 
def utility_A(par, x1A, x2A):
    return x1A**par.alpha * x2A**(1-par.alpha)

def utility_B(par, x1B, x2B):
    return x1B**par.beta * x2B**(1-par.beta)

def demand_A(par, p1):
    income = p1 * par.w1A + par.w2A
    
    x1A = par.alpha * (income / p1)
    x2A = (1 - par.alpha) * income 

    return x1A, x2A

def demand_B(par, p1):
    income = p1 * (1 - par.w1A) + (1 - par.w2A)

    x1B = par.beta * (income / p1)
    x2B = (1 - par.beta) * income

    return x1B, x2B

# c. define function for checking market clearing
def check_market_clearing(par, p1):
    x1A, x2A = demand_A(par, p1)
    x1B, x2B = demand_B(par, p1)

    eps1 = (x1A - par.w1A) + x1B - (1 - par.w1A)
    eps2 = (x2A - par.w2A) + x2B - (1 - par.w2A)

    return eps1, eps2

# d. define function for finding the Walras equilibrium
def walras_eq(par, sol, p_guess=1, print_output=True):
    p1 = p_guess
    t = 0 

    while True:
        eps = check_market_clearing(par, p1)

        if np.abs(eps[0]) < par.tol or t >= par.maxiter:
            if print_output:
                print('\nSolved!')
                print(f'{t:3d}: p1 = {p1:12.8f}')
                print(f'Excess demand of good 1: {eps[0]:14.8f}')
                print(f'Excess demand of good 2: {eps[1]:14.8f}')

            sol.p1 = p1
            sol.xA = demand_A(par, p1)
            sol.uA = utility_A(par, sol.xA[0], sol.xA[1])
            sol.uB = utility_B(par, 1 - sol.xA[0], 1 - sol.xA[1]) # utility_B(par, sol.xB[0], sol.xB[1])

            sol.solved = True
            sol.eps = eps
            break 

        p1 = p1 + par.adj * eps[0]

        if print_output:
            if t < 5 or t % 5 == 0:
                print(f'{t:3d}: p1 = {p1:12.8f} -> excess demand of good 1 -> {eps[0]:14.8f}')
            elif t == 5:
                print(' ...')
        
        t += 1

    if not sol.solved:
        print('Not solved!')

# e. identifies allocations in the economy where both consumers can be made better off compared to their initial endowments - C function
def improvement_set(par, sol):
    x_vec = np.linspace(0, 1, par.N+1)
    C_set = np.full((2, (par.N+1)**2), np.nan)

    sol.uA_w = utility_A(par, par.w1A, par.w2A)
    sol.uB_w = utility_B(par, 1 - par.w1A, 1 - par.w2A)

    i = 0
    for x1A in x_vec:
        for x2A in x_vec:
            if utility_A(par, x1A, x2A) >= sol.uA_w and utility_B(par, 1 - x1A, 1 - x2A) >= sol.uB_w:
                C_set[0, i] = x1A
                C_set[1, i] = x2A
            i += 1

    sol.C_set = C_set[:, ~np.isnan(C_set).any(axis=0)]
    return sol.C_set

# f. creates/draws the Edgeworth box
def create_edgeworth(par, figsize=(6,6)):
    """
    Creates the edgeworth box
    """

    # a. total endowment
    w1bar = 1.0
    w2bar = 1.0

    # b. figure set up
    fig = plt.figure(frameon=False,figsize=figsize, dpi=100)
    ax_A = fig.add_subplot(1, 1, 1)

    ax_A.set_xlabel("$x_1^A$")
    ax_A.set_ylabel("$x_2^A$")

    temp = ax_A.twinx()
    temp.set_ylabel("$x_2^B$")
    ax_B = temp.twiny()
    ax_B.set_xlabel("$x_1^B$")
    ax_B.invert_xaxis()
    ax_B.invert_yaxis()

    # limits
    ax_A.plot([0,w1bar],[0,0],lw=2,color='black')
    ax_A.plot([0,w1bar],[w2bar,w2bar],lw=2,color='black')
    ax_A.plot([0,0],[0,w2bar],lw=2,color='black')
    ax_A.plot([w1bar,w1bar],[0,w2bar],lw=2,color='black')

    ax_A.set_xlim([-0.1, w1bar + 0.1])
    ax_A.set_ylim([-0.1, w2bar + 0.1])    
    ax_B.set_xlim([w1bar + 0.1, -0.1])
    ax_B.set_ylim([w2bar + 0.1, -0.1])

    return fig,ax_A,ax_B

# g. plots the Edgeworth box
def plot_edgeworth(par, sol, pareto_set=True, question6b=False, indifference_curves=False, figsize=None):
    """
    Plots the edgeworth box with different features depending on the question
    """

    def plot_indifference_curves(x1_A,uA,uB,color='black'):

        x1_A_grid = np.linspace(0.7*x1_A,1.3*x1_A,100)
        x2_A_grid = (uA/x1_A_grid**par.alpha)**(1/(1-par.alpha))
        ax_A.plot(x1_A_grid,x2_A_grid,color=color)
        
        x1_B = 1-x1_A
        x1_B_grid = np.linspace(0.7*x1_B,1.3*x1_B,100)
        x2_B_grid = (uB/x1_B_grid**par.beta)**(1/(1-par.beta))
        ax_B.plot(x1_B_grid,x2_B_grid,color=color)
        
    fig, ax_A, ax_B = create_edgeworth(par, figsize=figsize)

    if pareto_set: # illustrate pareto improving set

        improvement_set(par, sol)
        ax_A.scatter(sol.C_set[0,:],sol.C_set[1,:],
                     color='lightblue',label='Pareto improving allocations')

    if question6b: # illustrate solution to questions 3-6a

        ax_A.scatter(*sol.xA,marker='x',color='black',label='market solution')

        x1_A = np.linspace(-1,2,1000)
        ax_A.plot(x1_A, par.w2A + sol.p1*(par.w1A-x1_A),ls='--',color='black')

        if indifference_curves:
            plot_indifference_curves(sol.xA[0],sol.uA,sol.uB)

        for q in ['4a','4b','5a','5b','6a']:
                
            if '4' in q:
                color = colors[0]
                label = f'{q} - A is price setter'
            elif '5' in q:
                color = colors[1]
                label = f'{q} - A is market maker'
            elif '6' in q:
                color = colors[2]
                label = '6a - social planner'

            if 'a' in q and not '6' in q:
                if indifference_curves: continue
                marker = 'x'
                label += ' (approx.)'
            else:
                marker = 'o'

            ax_A.scatter(*getattr(sol,f'xA_{q}'),label=label,color=color,marker=marker)
                
            if indifference_curves:

                x1_A = sol.__dict__[f'xA_{q}'][0]
                uA = sol.__dict__[f'uA_{q}']
                uB = sol.__dict__[f'uB_{q}']
                plot_indifference_curves(x1_A,uA,uB,color)

    # illustrate endowment
    ax_A.scatter(par.w1A,par.w2A,marker='s',color='black',label='endowment')

    ax_A.legend(frameon=True,loc='upper right',bbox_to_anchor=(2.0,1.0)); 

# h. plots and calculates excess demand and market clearing price
def plot_excess_demand(par, sol):
    """
    Plots the excess demand for good 1 and 2 as a function of p1
    Further, the market clearing price is found and plotted
    """

    for i, p1 in enumerate(par.P1_grid):
        eps = check_market_clearing(par, p1)
        par.eps_grid[:, i] = eps  # Update each column with the excess demand values

    fig, ax = plt.subplots(figsize=(6, 4))

    ax.scatter(par.P1_grid, par.eps_grid[0, :], s=0.75, label='$\epsilon_1$')
    ax.scatter(par.P1_grid, par.eps_grid[1, :], s=0.75, label='$\epsilon_2$')

    # plot solution
    walras_eq(par, sol, print_output=False)
    ax.scatter(sol.p1, sol.eps[0], label='market clearing')

    ax.legend()
    ax.set_xlabel('$p_1$')
    ax.set_ylabel('Excess demand')

    # print errors and market clearing price
    print(f'Error in market clearing conditions: {sol.eps[0]:.8f}')
    print(f'\nMarket clearing price: {sol.p1:.8f}')

# i. 
def question4(par, sol):

    # solve 4a
    sol.ua_grid = np.empty(par.N+1)

    for i,p1 in enumerate(par.P1_grid):
        x1B,x2B = demand_B(par, p1)
        if x1B >1 or x2B >1:
            sol.ua_grid[i] = 0
        else:
            sol.ua_grid[i] = utility_A(par, 1-x1B, 1-x2B)

    fig, ax = plt.subplots(figsize=(6,4))

    ax.scatter(par.P1_grid,sol.ua_grid,s=1,label='$u^{A}$')

    # plot solution 
    istar = np.argmax(sol.ua_grid)
    ax.scatter(par.P1_grid[istar], sol.ua_grid[istar], label='optimal point within P1 set')

    sol.p_4a = par.P1_grid[istar]
    sol.uA_4a = sol.ua_grid[istar]

    x1B,x2B = demand_B(par, sol.p_4a)
    sol.xA_4a = np.array([1-x1B,1-x2B])
    sol.uB_4a = utility_B(par, x1B, x2B)

    ax.legend()
    ax.set_xlabel('$p_1$')
    ax.set_ylabel('Utility')

    # Add title to the plot
    ax.set_title('Question 4a: Utility and Optimal Point within $P_1$ Set')

    print(f'Solution to question 4a:')
    print(f'p1 = {par.P1_grid[istar]:12.8f} with utility {sol.ua_grid[istar]:12.8f}')

    # solve 4b
    def obj(p1):
        x1B, x2B = demand_B(par, p1[0])
        if x1B >1 or x2B >1:
            return 0
        else:
            return -utility_A(par, 1-x1B, 1-x2B)
        
    opt = optimize.minimize(obj,x0=[par.P1_grid[istar]])

    sol.p_4b = opt.x[0]
    sol.uA_4b = -opt.fun

    x1B,x2B = demand_B(par, sol.p_4b)
    sol.xA_4b = np.array([1-x1B,1-x2B])
    sol.uB_4b = utility_B(par, x1B, x2B)

    print('\nSolution to question 4b:')
    print(f'p1 = {sol.p_4b:12.8f} with utility {sol.uA_4b:12.8f}')

def question5(par, sol):

    # a. solve 5a
    sol.ua_grid_5 = np.empty(sol.C_set.shape[1])

    for i, (x1A, x2A) in enumerate(zip(sol.C_set[0,:], sol.C_set[1,:])):
        sol.ua_grid_5[i] = utility_A(par, x1A, x2A)

    # plot solution 
    istar = np.argmax(sol.ua_grid_5)
     
    sol.xA_5a = sol.C_set[:,istar]
    sol.uA_5a = sol.ua_grid_5[istar]
    sol.uB_5a = utility_B(par, 1-sol.xA_5a[0], 1-sol.xA_5a[1])

    print(f'Solution to question 5a:')
    print(f'x1A = {sol.xA_5a[0]:12.8f}')
    print(f'x2A = {sol.xA_5a[1]:12.8f}')
    print(f'Utility {sol.uA_5a:12.8f}')

    # b. solve 5b
    def obj(xA):
        return -utility_A(par, xA[0], xA[1])

    constraints = ({'type': 'ineq', 'fun': lambda x: utility_B(par, 1-x[0], 1-x[1]) - sol.uB_w})
    bounds = ((0,1),(0,1))
        
    opt = optimize.minimize(obj,x0=sol.xA_5a,
                            method='SLSQP',
                            bounds=bounds,constraints=constraints)
    sol.xA_5b = opt.x
    sol.uA_5b = -opt.fun
    sol.uB_5b = utility_B(par, 1-sol.xA_5b[0], 1-sol.xA_5b[1])

    print(f'\nSolution to question 5b:')
    print(f'x1A = {sol.xA_5b[0]:12.8f}')
    print(f'x2A = {sol.xA_5b[1]:12.8f}')
    print(f'Utility {sol.uA_5b:12.8f}')

def question6a(par, sol):

    def obj(x):
        return -utility_A(par, x[0], x[1])-utility_B(par, 1-x[0], 1-x[1])
    
    bounds = ((0,1),(0,1))
    
    opt = optimize.minimize(obj,x0=[0.5,0.5],
                            method='SLSQP',
                            bounds=bounds)
    sol.xA_6a = opt.x
    sol.uA_6a = utility_A(par, sol.xA_6a[0], sol.xA_6a[1])
    sol.uB_6a = utility_B(par, 1-sol.xA_6a[0], 1-sol.xA_6a[1])

    print(f'Solution to question 6a:')
    print(f'x1A = {sol.xA_6a[0]:12.8f}')
    print(f'x2A = {sol.xA_6a[1]:12.8f}')
    print(f'Utility of A: {sol.uA_6a:12.8f}')
    print(f'Utility of B: {sol.uB_6a:12.8f}')

def question7(par, sim):

    sim.W = np.random.uniform(0,1,size=(2,50))
    fig, ax_A, ax_B = create_edgeworth(par)

    ax_A.scatter(sim.W[0,:],sim.W[1,:],marker='o',color='lightblue',label='endowments')
    ax_A.legend(frameon=True,loc='upper right',bbox_to_anchor=(1.6,1.0));

def question8(par, sol, sim):

    sim.xA = np.empty(sim.W.shape)

    for i in range(sim.W.shape[1]):
            
        par.w1A = sim.W[0,i]
        par.w2A = sim.W[1,i]
        walras_eq(par, sol, print_output=False)
        sim.xA[:,i] = sol.xA
        
    fig, ax_A, ax_B = create_edgeworth(par)

    ax_A.scatter(sim.W[0,:],sim.W[1,:],marker='o',color='grey',label='endowments')
    ax_A.scatter(sim.xA[0,:],sim.xA[1,:],marker='x',color='blue',label='market solutions')
    
    for i in range(sim.W.shape[1]):
        ax_A.arrow(sim.W[0,i],sim.W[1,i],sim.xA[0,i]-sim.W[0,i],sim.xA[1,i]-sim.W[1,i],color='grey')

    I = np.argsort(sim.xA[0,:])

    ax_A.plot(sim.xA[0,I],sim.xA[1,I],color='grey',lw=0.5,label='contract curve')
    ax_A.plot([0,sim.xA[0,I[0]]],[0,sim.xA[1,I[0]]],color='grey',lw=0.5)
    ax_A.plot([1,sim.xA[0,I[-1]]],[1,sim.xA[1,I[-1]]],color='grey',lw=0.5)

    ax_A.legend(frameon=True,loc='upper right',bbox_to_anchor=(1.65,1.0));
