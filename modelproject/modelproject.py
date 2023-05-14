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
        opt = SimpleNamespace()
        # a. externally given parameters
        par.a = 0.5 

        # b. parameters to be chosen 
        par.e1 = 12
        par.e2 = 0
        par.p2 = 1

        # c. misc paramters
        par.simT = 10_000 # length of simulation

        # d. calculate compound paramters
        #self.calc_compound_par()

        # e. simulation
        sim.u = np.zeros(par.simT)
        sim.z = np.zeros(par.simT)
        sim.y = np.zeros(par.simT)
        sim.p1 = np.zeros(par.simT)
        sim.x1 = np.zeros(par.simT)
        sim.x2 = np.zeros(par.simT)
        sim.m = np.zeros(par.simT)
        sim.pi = np.zeros(par.simT)


    def simulate(self):
        """ simulate the full model """

        np.random.seed(1917)

        par = self.par
        sim = self.sim


        # b. period-by-period
        # Output from company
        y = 1/(2*p1)

        # Profit
        pi = 1/(4*p1)

        # Utility maximization
        m = p1*par.e1+par.p2*par.e2+pi
        x1 = par.a*(m/p1)
        x2 = (1-par.a)*(m/par.p2)







