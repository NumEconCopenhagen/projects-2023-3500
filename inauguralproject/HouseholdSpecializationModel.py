
from types import SimpleNamespace

import numpy as np
from scipy import optimize

import pandas as pd 
import matplotlib.pyplot as plt

class HouseholdSpecializationModelClass:

    def __init__(self):
        """ setup model """

        # a. create namespaces
        par = self.par = SimpleNamespace()
        sol = self.sol = SimpleNamespace()

        # b. preferences
        par.rho = 2.0
        par.nu = 0.001
        par.epsilon = 1.0
        par.omega = 0.5 

        # c. household production
        par.alpha = 0.5
        par.sigma = 1.0

        # d. wages
        par.wM = 1.0
        par.wF = 1.0
        par.wF_vec = np.linspace(0.8,1.2,5)

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

    def calc_utility(self,LM,HM,LF,HF):
        """ calculate utility """

        par = self.par
        sol = self.sol

        # a. consumption of market goods
        C = par.wM*LM + par.wF*LF

        # b. home production
        if par.sigma == 1:
            H = HM**(1-par.alpha)*HF**par.alpha
        elif par.sigma == 0:
            H = np.argmin(HM, HF)
        else:
            H = ((1-par.alpha)*HM**((par.sigma-1)/par.sigma)+par.alpha*HF**((par.sigma-1)/par.sigma))**(par.sigma/(par.sigma-1)) 


        # c. total consumption utility
        Q = C**par.omega*H**(1-par.omega)
        utility = np.fmax(Q,1e-8)**(1-par.rho)/(1-par.rho)

        # d. disutlity of work
        epsilon_ = 1+1/par.epsilon
        TM = LM+HM
        TF = LF+HF
        disutility = par.nu*(TM**epsilon_/epsilon_+TF**epsilon_/epsilon_)
        
        return utility - disutility

    def solve_discrete(self,do_print=False):
        """ solve model discretely """
        
        par = self.par
        sol = self.sol
        opt = SimpleNamespace()
        
        # a. all possible choices
        x = np.linspace(0,24,49)
        LM,HM,LF,HF = np.meshgrid(x,x,x,x) # all combinations
    
        LM = LM.ravel() # vector
        HM = HM.ravel()
        LF = LF.ravel()
        HF = HF.ravel()

        # b. calculate utility
        u = self.calc_utility(LM,HM,LF,HF)
    
        # c. set to minus infinity if constraint is broken
        I = (LM+HM > 24) | (LF+HF > 24) # | is "or"
        u[I] = -np.inf
    
        # d. find maximizing argument
        j = np.argmax(u)
        
        opt.LM = LM[j]
        opt.HM = HM[j]
        opt.LF = LF[j]
        opt.HF = HF[j]

        # d1. finding HF/HM relationship
        opt.HFHM = HF[j]/HM[j]

        # e. print
        if do_print:
            for k,v in opt.__dict__.items():
                print(f'{k} = {v:6.4f}')

        return opt

    def solve(self,do_print=False):
        """ solve model continously """
        par = self.par
        sol = self.sol 
        opt = SimpleNamespace()
        
        # Making guesses for each parameter:
        #LM,HM,LF,HF
        LM_g=6
        HM_g=6
        LF_g=6
        HF_g=6
        All_g=[LM_g,HM_g,LF_g,HF_g]

        # Creating objective 
        objective_function = lambda x: -self.calc_utility(x[0],x[1],x[2],x[3])

        # Setting bounds
        bounds=[(0,24),(0,24),(0,24),(0,24)]
        
        # Finding result element and extracting values from it
        res = optimize.minimize(objective_function,All_g,method='Nelder-Mead',bounds=bounds) 
        opt.LM = res.x[0]
        opt.HM = res.x[1]
        opt.LF = res.x[2]
        opt.HF = res.x[3]

        # e. print
        if do_print:
            for k,v in opt.__dict__.items():
                print(f'{k} = {v:6.4f}')

        return opt  
   

    def solve_wF_vec(self,discrete=False):
        """ solve model for vector of female wages """

        par = self.par
        sol = self.sol

        for i,wF in enumerate(par.wF_vec):    
            par.wF = wF
            
            # Running the model and replacing the values in the vectors with the optimal values
            opt = self.solve()
            sol.HF_vec[i]=opt.HF
            sol.HM_vec[i]=opt.HM

        return sol.HF_vec, sol.HM_vec



    def run_regression(self):
        """ run regression """

        par = self.par
        sol = self.sol

        self.solve_wF_vec()

        x = np.log(par.wF_vec)
        y = np.log(sol.HF_vec/sol.HM_vec)
        A = np.vstack([np.ones(x.size),x]).T
        sol.beta0,sol.beta1 = np.linalg.lstsq(A,y,rcond=None)[0] 

        return sol.beta0,sol.beta1
        
    
    def estimate(self):
        """ estimate alpha and sigma """
        par = self.par
        sol = self.sol
        
        def error(x):
            par.alpha, par.sigma = x
            self.solve_wF_vec()
            self.run_regression()
            fejl = (par.beta0_target-sol.beta0)**2  + (par.beta1_target-sol.beta1)**2
            return fejl
        
        x0 = [0.5,0.5]
        bounds = [(0,1),(0,1)]
        solution = optimize.minimize(error, x0, method='Nelder-Mead', bounds=bounds)
        sol.alpha = solution.x[0]
        sol.sigma = solution.x[1]
        

                                        
