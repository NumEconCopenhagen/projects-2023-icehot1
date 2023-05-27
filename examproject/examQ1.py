from types import SimpleNamespace
import numpy as np
from scipy import optimize
import matplotlib.pyplot as plt
from sympy import symbols, diff

class WorkerUtilityModel:
    
    def __init__(self):
        """ Set up Model """
        
        # a. Create namespaces
        self.par = SimpleNamespace()
        self.sol = SimpleNamespace()

        # b. Model base parameters
        self.par.alpha = 0.5
        self.par.kappa = 1.0
        self.par.nu = 1 / (2 * 16 ** 2)
        self.par.w = 1.0
        self.par.tau = 0.3
        self.par.G = 1.0  # Starting Value 
        
        # c. Solution
        self.sol.L = None
        self.sol.L_tilde = {}  # Store optimal labour supply  from our optimization 
        self.sol.L_star = {}  # Store optimal labour supply from the equation given for comparison 
        self.sol.tau_star = None  # Store optimal tax rate
        
    def utility(self, L):
        """ Calculate utility based on given hours of labor """
        
        par = self.par

        # Calculate private consumption C
        C = par.kappa + (1 - par.tau) * par.w * L

        if C <= 0:
            return -np.inf

        # Calculate utility
        utility = np.log(C ** par.alpha * par.G ** (1 - par.alpha)) - par.nu * (L ** 2) / 2
        
        return utility
    
    def solve(self):
        """ Solve the model by maximizing the utility """

        # Objective function for scipy's optimizer
        objective = lambda L: -self.utility(L)
        
        # Bounds 
        bounds = [(0, 24)]
        
        # Initial guess 
        L0 = [12]
        
        # Find the results 
        result = optimize.minimize(objective, L0, method='Nelder-Mead', bounds=bounds)
        
        # Store the optimal solution
        self.sol.L = result.x[0]

        return result, self.sol.L
    
    def find_optimal_labour_supply(self, tau=None, wage=None):
        par = self.par

        if tau is not None:
            original_tau = self.par.tau
            self.par.tau = tau
        # Define G values
        G_values = [1.0, 2.0]

        # Wage 
        if wage is not None:
            original_wage = par.w  
            par.w = wage  

        for G in G_values:
            self.par.G = G  
            self.solve()  
            self.sol.L_tilde[G] = self.sol.L  # Store optimal labour supply

        # Reset G, wage and tax to default value
        par.G = 1.0
        if tau is not None: 
            self.par.tau = original_tau
        if wage is not None:
            self.par.w = original_wage

        return self.sol.L_tilde


    def verify_optimal_labour_supply(self):
        """ Verify the optimal labour supply choice """
        # Define G values
        G_values = [1.0, 2.0]

        for G in G_values:
            # We do not want the negative results, so we try our results 
            w_tilde = (1 - self.par.tau) * self.par.w
            try:
                L_star = (-self.par.kappa + np.sqrt(self.par.kappa ** 2 + 4 * (self.par.alpha / self.par.nu) * w_tilde ** 2)) / (2 * w_tilde)
            except ValueError:
                print(f'Error occurred')
                L_star = 0
            self.sol.L_star[G] = L_star

        return self.sol.L_star
    
    def plot_L_star(self):

        # Set range for w
        w_values = np.linspace(0.1, 2, 100)  # avoid w=0 to prevent division by zero

        # Calculate corresponding L_star values
        L_star_values = [(-self.par.kappa + np.sqrt(self.par.kappa ** 2 + 4 * (self.par.alpha / self.par.nu) * ((1 - self.par.tau) * w) ** 2)) / (2 * (1 - self.par.tau) * w) for w in w_values]

        # Calculate L_star values from optimization
        L_star_opt_values = []
        for w in w_values:
            self.find_optimal_labour_supply(self.par.tau, w)
            L_star_opt_values.append(self.sol.L_tilde[1.0]) 

        # Plot L_star vs w
        plt.figure(figsize=(10,6))
        plt.plot(w_values, L_star_values, label="$L^{\star}(\\tilde{w})$ (Theoretical)")
        plt.plot(w_values, L_star_opt_values, label="$L^{\star}(\\tilde{w})$ (Optimization)", linestyle='--')
        plt.xlabel('w')
        plt.ylabel('$L^{\star}(\\tilde{w})$')
        plt.title('Optimal Labour Supply as a Function of Wage')
        plt.legend()
        plt.grid(True)
        plt.show()

    def calculate_G(self, tau=None, wage=None):
    
        if tau is not None:
            original_tau = self.par.tau
            self.par.tau = tau

        if wage is not None:
            original_wage = self.par.w
            self.par.w = wage

        # Define G
        G_value = self.par.tau * self.par.w * self.find_optimal_labour_supply()[1]

        # Reset tau and wage to default value
        if tau is not None:
            self.par.tau = original_tau
        if wage is not None:
            self.par.w = original_wage

        return G_value
    
    
    def utility_tau(self, tau):
        par = self.par
        par.tau = tau
        self.solve()
        par.G = self.calculate_G(tau)
        utility = self.utility(self.sol.L)
        self.optimal_L = self.sol.L  # Store the optimal labour supply
        return utility


    def find_optimal_tax_rate(self):
        par = self.par

        # Objective function for scipy's optimizer
        objective = lambda tau: -self.utility_tau(tau)

        # Bounds
        bounds = [(0.01, 0.99)]  # avoid 0 and 1 to prevent division by zero

        # Initial guess
        tau0 = [0.5]

        # Optimize
        result = optimize.minimize(objective, tau0, method='SLSQP', bounds=bounds)

        # Store the optimal tax rate
        self.tau_star = result.x[0]

        return self.tau_star
    

    
    def plot_optimal_tax_rate(self):
        tau_values = np.linspace(0.01, 0.99, 100)  # range of tax rate values
        utility_values = []  # to store utility values corresponding to each tax rate

        # calculate utility for each tax rate in the range
        for tau in tau_values:
            utility = self.utility_tau(tau)
            utility_values.append(utility)

        # calculate the optimal tax rate
        optimal_tau = self.find_optimal_tax_rate()

        # plot utility as a function of tax rate
        plt.figure(figsize=(10, 6))
        plt.plot(tau_values, utility_values, label="Utility")
        plt.axvline(x=optimal_tau, color='r', linestyle='--', label=f"Optimal Tax Rate: {optimal_tau}")
        plt.xlabel('Tax Rate ($\\tau$)')
        plt.ylabel('Utility')
        plt.title('Utility as a function of Tax Rate')
        plt.legend()
        plt.grid(True)
        plt.show()

   
    
    def plot_implied_values(self):

        tau_values = np.linspace(0.0001, 0.9999, 100)
        L_values = []
        G_values = []
        utility_values = []

        for tau in tau_values:
            self.find_optimal_labour_supply(tau=tau)
            L_values.append(self.sol.L_tilde[1.0])
            G = tau * self.par.w * self.sol.L_tilde[1.0]
            G_values.append(G)
            self.par.G = G
            utility_values.append(self.utility(self.sol.L_tilde[1.0]))

        # Create the plots
        fig, ax1 = plt.subplots(figsize=(10, 6))

        ax1.plot(tau_values, L_values, label='Optimal labor $L^*$')
        ax1.plot(tau_values, G_values, label='Government consumption $G$')
        ax1.set_xlabel('Labor income tax rate $\\tau$')
        ax1.set_ylabel('$L^*$, $G$')
        ax1.legend()

        ax2 = ax1.twinx()
        ax2.plot(tau_values, utility_values, 'r', label='Utility')
        ax2.set_ylabel('Utility', color='r')
        ax2.tick_params('y', colors='r')

        fig.tight_layout()
        plt.title('Dependency of $L^*$, $G$ and Utility on $\\tau$')
        plt.grid(True)
        plt.show()

        # Create the plots
        plt.figure(figsize=(15, 10))

        plt.subplot(3, 1, 1)
        plt.plot(tau_values, L_values, label='Optimal labor $L^*$')
        plt.xlabel('Tax rate $\\tau$')
        plt.ylabel('Optimal labor $L^*$')
        plt.title('Dependency of optimal labor on tax rate')
        plt.legend()
        plt.grid(True)

        plt.subplot(3, 1, 2)
        plt.plot(tau_values, G_values, label='Government consumption $G$')
        plt.xlabel('Tax rate $\\tau$')
        plt.ylabel('Government consumption $G$')
        plt.title('Dependency of government consumption on tax rate')
        plt.legend()
        plt.grid(True)

        plt.subplot(3, 1, 3)
        plt.plot(tau_values, utility_values, label='Worker utility')
        plt.xlabel('Tax rate $\\tau$')
        plt.ylabel('Worker utility')
        plt.title('Dependency of worker utility on tax rate')
        plt.legend()
        plt.grid(True)

        plt.tight_layout()
        plt.show()
