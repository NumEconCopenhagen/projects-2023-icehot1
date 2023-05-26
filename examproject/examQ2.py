import numpy as np
from types import SimpleNamespace
from scipy import optimize
from scipy import interpolate

class SalonModel():
    """ The model for analyzing the profitability of a hair salon business. """
        
    def __init__(self):
        """ setup model """
 
        np.random.seed(123)

        # a. create namespaces
        par = self.par = SimpleNamespace()
        sol = self.sol = SimpleNamespace()

        # b. parameters
        par.eta = 0.5
        par.w = 1.0
        par.rho = 0.90
        par.iota = 0.01
        par.sigma_epsilon = 0.10
        par.R = (1 + 0.01)**(1/12)
        par.K = 1000  

        # c. series
        sol.epsilon = np.random.normal(-0.5*par.sigma_epsilon**2, par.sigma_epsilon, (par.K, 120))
        sol.kappa = np.zeros((par.K, 120))
        sol.labor = np.zeros((par.K, 120))

    def profit(self, l, kappa):
        """ calculate profit """
        par = self.par
        if l < 0:
            return np.nan
        return kappa * l**(1 - par.eta) - par.w * l
    
    def optimal_labor(self, kappa):
        """ calculate labor that maximizes profits """
        par = self.par
        l = ((1 - par.eta) * kappa / par.w)**(1 / par.eta)
        return np.where(l >= 0, l, 0) #ensure that labor is never negative

    def labor_policy(self, l_star, l_prev, Delta):
        """ Compute labor for the current time period given policy """
        l = np.where(abs(l_prev - l_star) > Delta, l_star, l_prev)
        return np.maximum(l, 0) 

    def shock_series(self, Delta):
        """ generate the shock series """
        sol = self.sol
        par = self.par
        kappa_minus_1 = 1  # initial demand shock
        sol.kappa[:, 0] = np.exp(par.rho * np.log(kappa_minus_1) + sol.epsilon[:, 0])  # kappa for t = 0
        l_star = self.optimal_labor(sol.kappa[:, 0])
        sol.labor[:, 0] = self.labor_policy(l_star, 0, Delta)  # labor for t = 0
        for t in range(1, 120):
            sol.kappa[:, t] = np.exp(par.rho * np.log(sol.kappa[:, t-1]) + sol.epsilon[:, t])
            l_star = self.optimal_labor(sol.kappa[:, t])
            sol.labor[:, t] = self.labor_policy(l_star, sol.labor[:, t-1], Delta)
    
    def salon_value(self):
        """ calculate the salon value """
        par = self.par
        sol = self.sol
        # create array for time discounts
        discounts = par.R**(-np.arange(120))
        # create adjustment cost array
        adjustment_costs = np.roll(sol.labor, shift=1, axis=1) != sol.labor
        adjustment_costs[:, 0] = 0  # no adjustment cost at t=0
        # compute value
        h = discounts * (sol.kappa * sol.labor**(1 - par.eta) - par.w * sol.labor - par.iota * adjustment_costs)
        return np.mean(h.sum(axis=1))  # sum over time for each series, then average

    def calculate_H(self, Delta):
        """ calculate H """
        self.shock_series(Delta)
        return self.salon_value()
    
    def optimal_delta(self):
        """ find the optimal Delta that maximizes H """
    
        # a. define objective function
        objective = lambda Delta: -self.calculate_H(Delta)
    
        # b. choose bounds for Delta
        bounds = (0.0, 1.0)  
    
        # c. optimize
        result = optimize.minimize_scalar(objective, bounds=bounds, method='bounded')
    
        # d. save and return results
        optimal_delta = result.x
        self.optimal_delta = optimal_delta

        return optimal_delta
    
