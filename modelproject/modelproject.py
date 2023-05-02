from types import SimpleNamespace

import numpy as np
from scipy import optimize

import pandas as pd 
import matplotlib.pyplot as plt

class Koopman:

    def __init__(self):
            """ setup model """

        # a. create namespaces
        par = self.par = SimpleNamespace()

        # b. intial stock
        par.e1 = 12
        par.e2 = 0

        # c. consumer preferences
        par.alpha = 0.5

        # d. consuming (tror dette er forkert)
        par.x1 = 6
        par.x2 = 6

        # e. targets
        par.beta0_target = 0.4
        par.beta1_target = -0.1

        # f. solution
        sol.LM_vec = np.zeros(par.wF_vec.size)
        sol.HM_vec = np.zeros(par.wF_vec.size)
        sol.LF_vec = np.zeros(par.wF_vec.size)
        sol.HF_vec = np.zeros(par.wF_vec.size)

        sol.beta0 = np.nan
        sol.beta1 = np.nan

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



