from types import SimpleNamespace

import numpy as np
from scipy import optimize

import pandas as pd 
import matplotlib.pyplot as plt

class Koopman:

    def __init__(self):
        """ initialize the model """

        par = self.par = SimpleNamespace() # parameters
        sim = self.sim = SimpleNamespace() # simulation variables

        # a. externally given parameters
        par.a = 0.5 

        # b. parameters to be chosen (here guesses)
        par.e1 = 12
        par.e2 = 0

        # c. misc paramters
        par.simT = 10_000 # length of simulation

        # d. calculate compound paramters
        self.calc_compound_par()

        # e. simulation
        sim.u = np.zeros(par.simT)
        sim.z = np.zeros(par.simT)
        sim.y = np.zeros(par.simT)

        # f. data (numbers given in notebook)
        datamoms.std_y = 1.64
        datamoms.std_pi = 0.21
        datamoms.corr_y_pi = 0.31
        datamoms.autocorr_y = 0.84
        datamoms.autocorr_pi = 0.48

    def calc_compound_par(self):
        """ calculates compound parameters """

        par = self.par

        par.a = (1+par.alpha*par.gamma*par.phi)/(1+par.alpha*par.gamma)
        par.beta = 1/(1+par.alpha*par.gamma)

    def simulate(self):
        """ simulate the full model """

        np.random.seed(1917)

        par = self.par
        sim = self.sim

        # a. draw random  shock innovations
        sim.x = np.random.normal(loc=0.0,scale=par.sigma_x,size=par.simT)
        sim.c = np.random.normal(loc=0.0,scale=par.sigma_c,size=par.simT)

        # b. period-by-period
        for t in range(par.simT):

            # i. lagged
            if t == 0:
                z_lag = 0.0
                s_lag = 0.0
                y_hat_lag = 0.0
                pi_hat_lag = 0.0
            else:
                z_lag = sim.z[t-1]
                s_lag = sim.s[t-1]
                y_hat_lag = sim.y_hat[t-1]
                pi_hat_lag = sim.pi_hat[t-1]

            # ii. AR(1) shocks
            z = sim.z[t] = par.delta*z_lag + sim.x[t]
            s = sim.s[t] = par.omega*s_lag + sim.c[t]

            # iii. output and inflation
            sim.y_hat[t] = par.a*y_hat_lag + par.beta*(z-z_lag) \
                - par.alpha*par.beta*s + par.alpha*par.beta*par.phi*s_lag
            sim.pi_hat[t] = par.a*pi_hat_lag + par.gamma*par.beta*z \
                - par.gamma*par.beta*par.phi*z_lag + par.beta*s - par.beta*par.phi*s_lag

    def solve_ss(alpha, c):
        """ Example function. Solve for steady state k. 

        Args:
            c (float): costs
            alpha (float): parameter
        
        If lambda < 2 

        Returns:
            result (RootResults): the solution represented as a RootResults object.

        """ 
        
        # a. Objective function, depends on k (endogenous) and c (exogenous).
        f = lambda k: k**alpha - c
        obj = lambda kss: kss - f(kss)

        #. b. call root finder to find kss.
        result = optimize.root_scalar(obj,bracket=[0.1,100],method='bisect')
        
        return result



