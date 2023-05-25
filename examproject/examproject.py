
from types import SimpleNamespace
from scipy import optimize
import numpy as np
import matplotlib.pyplot as plt
class examproject:

    def opgave5_6(self):
        sol = SimpleNamespace()
        # Preferences
        alfa = 0.5
        kappa = 1
        v = 1/(2*16**2)
        w = 1
        epsilon = 1
        rho = 1.001
        sigma = 1.001
        tau = 0.3406
        def utility(labour):
            con = kappa + (1 - tau) * w * labour
            inner = (alfa * con**((sigma - 1) / sigma) + (1 - alfa) * 1**((sigma - 1) / sigma))**(sigma / (sigma-1))
            return (((inner**(1 - rho)) - 1) / (1 - rho)) - v * ((labour**(1 + epsilon)) / (1 + epsilon))

        def govtfunc(labour):
            return tau * w * labour * ((1 - tau) * w)


        def solve5(sol):
            objective_function = lambda x: -utility(x[0])
            x0 = [10.01]
            bounds = [(0.001, 24.0)]
            solution = optimize.minimize(objective_function, x0, method='Nelder-Mead', bounds=bounds)
            sol.labour = solution.x[0]
            return sol

        sol.sol1 = solve5(sol.labour)
        sol.govtspen1 = govtfunc(sol.labour)
    
        epsilon = 1
        rho = 1.5
        sigma = 1.5

        sol.sol2 = solve5(sol.labour)
        sol.govtspen2 = govtfunc(sol.labour)