from types import SimpleNamespace

from operator import le
import numpy as np
from scipy import interpolate
from scipy import optimize


class Consumption():
    def __init__(self):
        """Set up the Model"""
        par =self.par =SimpleNamespace()


        # Baseline parameters
        par.alpha = 0.5
        par.kappa = 1.0
        par.nu = 1/(2*16**2)
        par.w = 1.0
        par.tau = 0.30
        par.w_tilde = (1-par.tau)*par.w

        
    G_values = [1.0, 2.0]

    # Define the utility function
    def utility(self,L, w_tilde, G):
        par =self.par
        #w_tilde = (1-par.tau)*par.w
        C = par.kappa + w_tilde*L
        utility = np.log(C**par.alpha * G**(1-par.alpha)) - par.nu*L**2/2
        return utility  

    # Define the given formula to compare 
    def L_star_formula(self,w_tilde):
        par = self.par
        L_star_formula = (-par.kappa + np.sqrt(par.kappa**2 + 4*par.alpha/par.nu*w_tilde**2)) / (2*w_tilde)
        return L_star_formula
    
    # def valuation(self):

    #     def objective(x):


    
    #     const = [{'type': 'ineq', 'fun': lambda x: 24-x[0] -x[1]}, 
    #                 {'type': 'ineq', 'fun': lambda x: 24-x[2] -x[3]}]



    # for G in G_values():
        
    #     # Calculate L_star using optimization
    #     L_star_opt = optimize.minimize(utility, x0=[12], args=(w_tilde, G), bounds=[(0, 24)]).x[0]
    #     # Calculate L_star using the formula
    #     L_star_formula_val = L_star_formula(w_tilde)
    #     print(f"For G = {G}, L_star (optimized) = {L_star_opt:.2f}, L_star (formula) = {L_star_formula_val:.2f}")