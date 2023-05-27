import numpy as np
from types import SimpleNamespace
from scipy import optimize
from scipy import interpolate

class SalonModel():
    """ The model for analyzing the profitability of a hair salon business. """
        
    def __init__(self):
        """ setup model """

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
        par.t = 120 

        # c. series
        sol.epsilon = np.random.normal(-0.5*par.sigma_epsilon**2, par.sigma_epsilon, (par.K, par.t))
        sol.kappa = np.zeros((par.K, par.t)) 
        sol.labor = np.zeros((par.K, par.t)) 

    def profit(self, l, kappa):
        """ calculate profit """
        par = self.par
        if l < 0: #labor can not be negative
            return np.nan
        return kappa * l**(1 - par.eta) - par.w * l #profit formula given in problem. 
    
    def optimal_labor(self, kappa):
        """ calculate labor that maximizes profits """
        par = self.par 
        l = ((1 - par.eta) * kappa / par.w)**(1 / par.eta) #the formula given in the problem which we also utilize as a policy
        return np.where(l >= 0, l, 0) #ensure that labor is never negative

    def labor_policy(self, l_star, l_prev, Delta):
        """ Compute labor for the current time period given policy """        
        #where l_star is the optimal labor for the current period, l_prev is the optimal labor for the previous period
        # Delta is the threshold for determining whether to adjust the number of hairdressers
        l = np.where(abs(l_prev - l_star) > Delta, l_star, l_prev) #if Delta=0 then the l_star is used 
                                                                    # if |l_prev - l_star | > Delta then l_star is used.                                                      
        return np.maximum(l, 0) #ensuring that the maximum adjusted labor returned by the labor_policy function is always non-negative.

    def shock_series(self, Delta):
        """ generate the shock series """
        sol = self.sol
        par = self.par
        
        #setting for t=0
        kappa_minus_1 = 1  # initial demand shock
        sol.kappa[:, 0] = np.exp(par.rho * np.log(kappa_minus_1) + sol.epsilon[:, 0])  # kappa for t = 0 
        l_star = self.optimal_labor(sol.kappa[:, 0]) # optimal labor for the period t=0
        sol.labor[:, 0] = self.labor_policy(l_star, 0, Delta)  # labor for t = 0
        
        #setting t>0
        for t in range(1, par.t):
            sol.kappa[:, t] = np.exp(par.rho * np.log(sol.kappa[:, t-1]) + sol.epsilon[:, t]) # kappa for each period
            l_star = self.optimal_labor(sol.kappa[:, t]) # optimal labor for each period
            sol.labor[:, t] = self.labor_policy(l_star, sol.labor[:, t-1], Delta) # labor for each period
    
    def salon_value(self):
        """ calculate the salon value """
        par = self.par
        sol = self.sol

        # create array for time discounts
        discounts = par.R**(-np.arange(par.t)) #formula given in the problem

        # create adjustment cost array 
        adjustment_costs = np.roll(sol.labor, shift=1, axis=1) != sol.labor #formula given in the problem
        adjustment_costs[:, 0] = 0  # no adjustment cost at t=0

        # compute value
        h = discounts * (sol.kappa * sol.labor**(1 - par.eta) - par.w * sol.labor - par.iota * adjustment_costs) #ex post value for all the periods
        return np.mean(h.sum(axis=1))  # sum over time for each series, then average
    
    def calculate_H(self, Delta):
        """ calculate H """
        self.shock_series(Delta) #Generate a series of demand shocks. 
        return self.salon_value() #Calculate the value of the salon over all the generated shock series.
    
    def optimal_delta(self):
        """ find the optimal Delta that maximizes H """
    
        # a. define objective function
        objective = lambda Delta: -self.calculate_H(Delta) #maximizing H given Delta
    
        # b. choose bounds for Delta
        bounds = (0.0, 1.0)  
    
        # c. optimize
        result = optimize.minimize_scalar(objective, bounds=bounds, method='bounded')
    
        # d. save and return results
        optimal_delta = result.x
        self.optimal_delta = optimal_delta

        return optimal_delta 
    
    #DIFFERENT THRESHOLDS
    def adjustment_cost(self, l, l_prev):
        """ Calculate adjustment cost based on labor difference """
        par = self.par
        threshold = 0.5  
        adjustment = abs(l - l_prev)
    
        # Step function: cost is zero up to threshold, then jumps to fixed level
        cost = np.where(adjustment > threshold, par.iota, 0)

        return cost

    def salon_value_adjusted(self):
        """ calculate the salon value """
        par = self.par
        sol = self.sol

        # create array for time discounts
        discounts = par.R**(-np.arange(par.t)) #formula given in the problem

        # create adjustment cost array 
        adjustment_costs = self.adjustment_cost(sol.labor, np.roll(sol.labor, shift=1, axis=1)) #calculate adjustment cost
        adjustment_costs[:, 0] = 0  # no adjustment cost at t=0

        # compute value
        h = discounts * (sol.kappa * sol.labor**(1 - par.eta) - par.w * sol.labor - adjustment_costs) #ex post value for all the periods
        return np.mean(h.sum(axis=1))  # sum over time for each series, then average

    def calculate_H_adjusted(self, Delta):
        """ calculate H """
        self.shock_series(Delta) # Generate a series of demand shocks. 
        return self.salon_value_adjusted() # Calculate the value of the salon when adjustment cost is defined instead of a fixed cost over all the generated shock series.
    
    def optimal_delta_adjusted(self):
        """ find the optimal Delta that maximizes H """
    
        # a. define objective function
        objective = lambda Delta: -self.calculate_H_adjusted(Delta) #maximizing H given Delta
    
        # b. choose bounds for Delta
        bounds = (0.0, 1.0)  
    
        # c. optimize
        result = optimize.minimize_scalar(objective, bounds=bounds, method='bounded')
    
        # d. save and return results
        optimal_delta = result.x
        self.optimal_delta = optimal_delta

        return optimal_delta 