from types import SimpleNamespace
import time
import numpy as np
from scipy import optimize
import pandas as pd 
import matplotlib.pyplot as plt
 
class OLGClass:

    def __init__(self,do_print=True):
        """ initialize the model """

        if do_print: print('initializing the model:')

        self.par = SimpleNamespace() # Defining namespace for parameters
        self.sim = SimpleNamespace() # Defining namespace for simulation variables

        if do_print: print('calling .baseline()')
        self.baseline()

        if do_print: print('calling .allocate()')
        self.allocate()
    

    def baseline(self):
        """ Setup baseline parameters """

        par = self.par

        # a. Households
        par.sigma = 2 # CRRA coefficient
        par.beta = 0.95 # Discount factor

        # b. Firms
        par.production_function = 'cobb-douglas'
        par.alpha = 0.5 # Capital weight
        par.theta = 0.0 # Substitution parameter
        par.delta = 0.5 # Depreciation rate

        # c. Government
        par.tau_w = 0.35 # Tax in labor income
        par.tau_r = 0.2 # Tax on capital income
        par.bal_budget = True # Making sure the budget is complied

        # d. Initial stocks
        par.K_lag_ini = 0.5 # Initial capital stock
        par.B_lag_ini = 0.0 # Initial government debt

        # e. Timeframe
        par.simT = 50 # Length of simulation

        # f. Population
        par.n = np.full(par.simT, 0.05) # Population growth rate before the policy
        par.surv = 0.98 # Probability of surviving (becoming old)
        par.p_ini = 1 # Initial population
        par.p_young = [par.p_ini] + [0] * (par.simT - 1) # The young initial population
        par.p_old = [par.p_ini*par.surv/(1+par.n[0])] + [0] * (par.simT - 1) # The old initial population


    def allocate(self):
        """ Allocate the variables for simulation """
        
        par = self.par 
        sim = self.sim

        # a. Defining the different variables
        household = ['C1','C2'] # C1 = The youngs consumption. C2 = The olds consumption
        firm = ['Y','K','K_lag', 'k', 'k_lag'] # Y = Total production. K = Total capital. K_lag = Total capital the period before. k = Capital pr. worker. k_lag = Capital pr. worker last period
        prices = ['w','rk','rb','r','rt'] # w = Wages. rk = Rental rate of capital. rb = Rate on bonds. r = after-depreciation return. rt = Return after tax
        government = ['G','T','B','balanced_budget','B_lag'] # G = Government spending. T = Tax income for government. B = Bond value. balanced_budget = Budget constraint. B_lag = Bond value last period
        population = ['L','L_lag'] # L = Total population. L_lag = Total population last period

        # b. Allocating the variables
        allvarnames = household + firm + prices + government + population # Making a list of all the varibles
        for varname in allvarnames: # loops trough all variables in the unique list
            sim.__dict__[varname] = np.nan*np.ones(par.simT) 


    def define_population(self, t):
        """ Define the population throughout the simulation """

        par = self.par

        # a. The young population in the new period is the young population the period before including the growth rate
        par.p_young[t] = par.p_young[t-1] * (1 + par.n[t-1])

        # b. The old population in the new period is the young population in the period before who survived
        par.p_old[t] = par.p_young[t-1] * par.surv


    def simulate(self,do_print=True):
        """ Simulate the model """

        t0 = time.time()

        par = self.par
        sim = self.sim

        # a. Defining the initial values
        sim.K_lag[0] = par.K_lag_ini
        sim.k_lag[0] = par.K_lag_ini/par.p_ini 
        sim.B_lag[0] = par.B_lag_ini
        sim.L_lag[0] = par.p_ini

        # b. Iterate over the periods
        for t in range(par.simT):

            # i. Define population
            if t > 0:
                self.define_population(t)

            # ii. Simulate the model before we find s
            simulate_before_s(par,sim,t)

            if t == par.simT-1: continue       

            # iii. Find bracket to search for the optimal savings rate
            s_min,s_max = find_s_bracket(par,sim,t)

            # iv. Find optimal s
            obj = lambda s: calc_euler_error(s,par,sim,t=t) # Objective function
            result = optimize.root_scalar(obj,bracket=(s_min,s_max),method='bisect') # Optimize w.r.t. s
            s = result.root # Finding the optimized value of s

            # v. Simulate the model after we find s
            simulate_after_s(par,sim,t,s)

        if do_print: print(f'simulation done in {time.time()-t0:.2f} secs') # Show how long the simulation takes

        print(f'\noptimal saving rate in period 49 = {s:3f}') # Show the result in the last period
    
    
def find_s_bracket(par,sim,t,maxiter=500,do_print=False):
    """ Find a bracket for s """

    # a. Maximum bracket
    s_min = 0.0 + 1e-8 # Save nothing (almost)
    s_max = 1.0 - 1e-8 # Save everything (almost)

    # b. Finding value and Euler error for maximum saving rate
    value = calc_euler_error(s_max,par,sim,t)
    sign_max = np.sign(value)
    if do_print: print(f'euler-error for s = {s_max:12.8f} = {value:12.8f}')

    # c. Find upper and lower bound for bracket      
    lower = s_min
    upper = s_max

    it = 0
    while it < maxiter:
                
        # i. Find the midpoint for bracket and value for Euler error
        s = (lower+upper)/2 # Midpoint
        value = calc_euler_error(s,par,sim,t) # Euler error

        if do_print: print(f'euler-error for s = {s:12.8f} = {value:12.8f}')

        # ii. See if conditions is complied
        valid = not np.isnan(value)
        correct_sign = np.sign(value)*sign_max < 0
        
        # iii. Making the midpoint the new minimum value
        if valid and correct_sign: 
            s_min = s
            s_max = upper
            if do_print: 
                print(f'bracket to search in with opposite signed errors:')
                print(f'[{s_min:12.8f}-{s_max:12.8f}]')
            return s_min,s_max
        elif not valid: # Too low s -> increase lower bound
            lower = s
        else: # Too high s -> increase upper bound
            upper = s

        # iv.
        it += 1

    raise Exception('cannot find bracket for s')


def calc_euler_error(s,par,sim,t):
    """ Finding s """

    # a. Simulate
    simulate_after_s(par,sim,t,s) # Simulate the first period
    simulate_before_s(par,sim,t+1) # Simulate the next period

    # c. The Euler equation 
    LHS = sim.C1[t]**(-par.sigma)
    RHS = (1+sim.rt[t+1])*par.beta * sim.C2[t+1]**(-par.sigma)

    return LHS-RHS # Euler error


def simulate_before_s(par,sim,t):
    """ Simulation """

    # a. Update the lag variables in each period
    if t > 0:
        sim.K_lag[t] = sim.K[t-1]
        sim.B_lag[t] = sim.B[t-1]
        sim.L_lag[t] = sim.L_lag[0]*(1+par.n[t])**t
        sim.k_lag[t] = sim.K_lag[t]/sim.L_lag[t]

    # b. Production function and factor prices
    if par.production_function == 'ces':

        # i. Production function
        sim.Y[t] = (par.alpha*sim.K_lag[t]**(-par.theta) + (1-par.alpha)*(sim.L_lag[t])**(-par.theta) )**(-1.0/par.theta)

        # ii. Factor prices
        sim.rk[t] = par.alpha*sim.K_lag[t]**(-par.theta-1) * sim.Y[t]**(1.0+par.theta)
        sim.w[t] = (1-par.alpha)*(sim.L_lag[t])**(-par.theta-1) * sim.Y[t]**(1.0+par.theta)

    elif par.production_function == 'cobb-douglas':

        # i. Production function
        sim.Y[t] = (sim.K_lag[t]**par.alpha) * ((sim.L_lag[t])**(1-par.alpha))

        # ii. Factor prices
        sim.rk[t] = par.alpha * (sim.K_lag[t]**(par.alpha-1)) * ((sim.L_lag[t])**(1-par.alpha))
        sim.w[t] = (1-par.alpha) * (sim.K_lag[t]**(par.alpha)) * ((sim.L_lag[t])**(-par.alpha))

    else:

        raise NotImplementedError('unknown type of production function')

    # b. No-arbitrage
    sim.r[t] = sim.rk[t]-par.delta # After-depreciation return
    sim.rb[t] = sim.r[t] # Bond rates is the same
    sim.rt[t] = (1-par.tau_r)*sim.r[t] # Return after tax

    # c. The olds consumption in period t
    sim.C2[t] = (1+sim.rt[t])*(sim.K_lag[t]+sim.B_lag[t]) 

    # d. Government activities
    sim.T[t] = par.tau_r*sim.r[t]*(sim.K_lag[t]+sim.B_lag[t]) + par.tau_w*sim.w[t]*sim.L_lag[t] # Tax revenue
    
    if par.bal_budget == True: # Makes sure the governments budget holds 
        sim.balanced_budget[:] = True 

    if sim.balanced_budget[t]: # Balanced budget condition if it is required
        sim.G[t] = sim.T[t] - sim.r[t]*sim.B_lag[t]

    sim.B[t] = (1+sim.r[t])*sim.B_lag[t] - sim.T[t] + sim.G[t] # Debt


def simulate_after_s(par,sim,t,s):
    """ Simulation """

    # a. The youngs consumption
    sim.C1[t] = (1 - par.tau_w) * sim.w[t] * (1.0 + par.n[t]) ** t * (1 - s)

    # b. Stocks at the end of the periods
    I = sim.Y[t] - sim.C1[t] - sim.C2[t] - sim.G[t]
    sim.K[t] = (1 - par.delta) * sim.K_lag[t] + I - sim.K_lag[t] * (1 - par.surv)


