

from types import SimpleNamespace 

import numpy as np
from numpy import array
import sympy as sm
from sympy import *
import scipy as sc 
from scipy import optimize
import matplotlib.pyplot as plt
from scipy.optimize import fsolve 
import ipywidgets as widgets
from ipywidgets import interact
from IPython.display import display


class CournotDuopoly(): 

    def __init__(self, mc1, mc2, a):
        """Create model"""
        par=self.par=SimpleNamespace()

        par.mc = [mc1, mc2]    # Marginal cost 
        par.a = a     #Total Demand when price equals zero     

    def inverse_demand(self, q1, q2): #Define the inverse demand function as marketdemand
        par=self.par
        marketdemand= (par.a - (q1 + q2))
        return marketdemand
    
    def cost(self, q, mc): #Defining the cost function 
        if q == 0:
            cost = 0
        else:
            cost = mc*q
        return cost
        

    #Profit function for firm 1  
    def profit1(self, q1, q2, c):
        

        profit1 = self.inverse_demand(q1,q2)*q1 - self.cost(q1,c) 
        return profit1

    #Profit function for firm 2 
    def profit2(self, q1, q2,c):

        profit2= self.inverse_demand(q1,q2)*q2 - self.cost(q2,c) 
        return profit2
    
    
    #Best Response functions for both firm 1 and firm 2 in the Cournot Model 

    def BR1(self, q2, c1):
        x1 = optimize.minimize(lambda x: -self.profit1(x, q2, c1), x0=0, method='SLSQP' )
        return x1.x[0]
    
    def BR2(self,q1, c2):
        x2 = optimize.minimize(lambda x: -self.profit2(q1, x, c2), x0=0, method='SLSQP' )
        return x2.x[0]
    
    #Finding the quantity produced in Nash Equilibrium when q1=q2=qNE
    
    def nash_equilibrium(self, q):
        par =self.par
        return np.array([q[0] - self.BR1(q[1], par.mc[0]),
                         q[1] - self.BR2(q[0], par.mc[1])])
    

    def equilibrium_values(self):
        initial_guess = np.array([0, 0])
        res = optimize.fsolve(lambda q: self.nash_equilibrium(q), initial_guess)
        return res 
    
    def calculate_profits(self):
        q_opt = self.equilibrium_values()
        profit1 = int(round(self.profit1(q_opt[0], q_opt[1], self.par.mc[0])))
        profit2 = int(round(self.profit2(q_opt[0], q_opt[1], self.par.mc[1])))
        return profit1, profit2
        
    
    # three plots for the results shown in the ipynb file 
    
    def plot_cournot_nashequilibrium(self):
        @interact(mc1=(5, 20, 1), mc2=(5, 20, 1), a=(20, 60, 5))
        def plot(mc1, mc2, a):
            # Create an instance of the model with the specific parameters
                # create an instance of the model with your specific parameters
            model = CournotDuopoly(mc1, mc2, a)

            # Get the optimal quantity for each firm
            q_opt = model.equilibrium_values()

            # Create a vector of q values that includes 0 and goes up to some value larger than the optimal quantities
            q_values = np.linspace(0, max(q_opt)*1.5, 100)

            # Calculate best response for each firm at each quantity
            br1_values = [model.BR1(q2, model.par.mc[0]) for q2 in q_values]
            br2_values = [model.BR2(q1, model.par.mc[1]) for q1 in q_values]

            # Create the figure and the axes
            fig, ax = plt.subplots()

            # Plot the best response functions
            ax.plot(q_values, br1_values, label='Firm 1 best response')
            ax.plot(br2_values, q_values, label='Firm 2 best response')

            # Highlight the equilibrium point and annotate it
            ax.plot(q_opt[1], q_opt[0], 'ro')  # plot the point as a red dot
            ax.annotate(f'Equilibrium ({q_opt[1]:.2f}, {q_opt[0]:.2f})', (q_opt[1], q_opt[0]), 
                        textcoords="offset points", xytext=(-15,-10), ha='center', fontsize=8, color='black')

            # Set labels and title
            ax.set_xlabel('Quantity for firm 2')
            ax.set_ylabel('Quantity for firm 1')
            ax.set_title('Cournot Duopoly Nash Equilibrium')
            ax.legend()


            # Show the plot
            plt.show()



    def plot_mc_changes_both_firms(self, mc1, mc2, a):
        # Initialize the parameters
        mc1 = mc1
        mc2 = mc2
        a = a  # Total demand when price equals zero 

        # Create a range of common marginal costs for both firms
        mc_values = np.linspace(5, 20, 100)

        # Calculate optimal quantities for each marginal cost
        q1_opt_values = []
        q2_opt_values = []
        for mc in mc_values:
            model = CournotDuopoly(mc1=mc, mc2=mc, a=a)
            q_opt = model.equilibrium_values()
            q1_opt_values.append(q_opt[0])
            q2_opt_values.append(q_opt[1])

        # Create the figure and the axes
        fig, ax = plt.subplots()

        # Plot the optimal quantities as a function of the common marginal cost
        ax.plot(mc_values, q1_opt_values, label='Quantity for firm 1')
        ax.plot(mc_values, q2_opt_values, label='Quantity for firm 2')

        # Set labels and title
        ax.set_xlabel('Common marginal cost for both firms')
        ax.set_ylabel('Optimal quantity')
        ax.set_title('Optimal quantities as a function of common marginal cost')
        ax.legend()

        # Show the plot
        plt.show()
    
    def plot_cournot_mc_changes_firm1(self,mc1, mc2, a):
        # Initialize the parameters
        mc1 = mc1
        mc2 = mc2  # Marginal cost for firm 2
        a = a  # Total demand when price equals zero 

        # Create a range of marginal costs for firm 1
        mc_values = np.linspace(5, 20, 100)

        # Calculate optimal quantities for each marginal cost
        q1_opt_values = []
        q2_opt_values = []
        for mc1 in mc_values:
            model = CournotDuopoly(mc1=mc1, mc2=mc2, a=a)
            q_opt = model.equilibrium_values()
            q1_opt_values.append(q_opt[0])
            q2_opt_values.append(q_opt[1])

        # Create the figure and the axes
        fig, ax = plt.subplots()

        # Plot the optimal quantities as a function of mc1
        ax.plot(mc_values, q1_opt_values, label='Quantity for firm 1')
        ax.plot(mc_values, q2_opt_values, label='Quantity for firm 2')

        # Set labels and title
        ax.set_xlabel('Marginal cost for firm 1')
        ax.set_ylabel('Optimal quantity')
        ax.set_title('Optimal quantities as a function of marginal cost for firm 1')
        ax.legend()

        # Show the plot
        plt.show()

    def plot_comparison(self):
        # Define the range of marginal costs
        mc_range = np.linspace(2, 20, 100)

        # Calculate the optimal quantities for Cournot model
        q_cournot = []
        for mc in mc_range:
            cournot_model = CournotDuopoly(mc, mc, a=40)
            q_opt = cournot_model.equilibrium_values()
            q_cournot.append(q_opt)

        # Calculate the optimal quantities for Stackelberg model
        q_stackelberg = []
        for mc in mc_range:
            stackelberg_model = StackelbergDuopoly(mc, mc, a=40)
            q1, q2 = stackelberg_model.get_optimal_values_Stackelberg()
            q_stackelberg.append([q1, q2])

        # Convert the lists to NumPy arrays
        q_cournot = np.array(q_cournot)
        q_stackelberg = np.array(q_stackelberg)

        # Create the plot
        fig, ax = plt.subplots()
        ax.plot(mc_range, q_cournot[:, 0], label="Cournot - Firms")
        ax.plot(mc_range, q_stackelberg[:, 0], label="Stackelberg - Firm 1")
        ax.plot(mc_range, q_stackelberg[:, 1], label="Stackelberg - Firm 2")
        ax.set_xlabel('Marginal Cost')
        ax.set_ylabel('Quantity')
        ax.set_title('Cournot vs Stackelberg Duopoly')
        ax.legend()

        # Show the plot
        plt.show()



#Defning a new class for the StackelbergDuopoly within the CournotDuopoly class

class StackelbergDuopoly(CournotDuopoly):

    #Find the optimal quantity produced in Stackelberg
    
    def get_optimal_values_Stackelberg(self):
        def follower_BestResponse(q1):  # The follower's best response is a function of q1
            return self.BR2(q1, self.par.mc[1])
        
        def profitleader(q1):
            q2 = follower_BestResponse(q1)
            return self.profit1(q1, q2, self.par.mc[0])

        # The leader chooses q1 to maximize their own profit, taking into account the follower's best response
        solution = optimize.minimize(lambda q1: -profitleader(q1), x0=0, bounds=[(0, np.inf)], method='SLSQP')
        q1optStackelberg = solution.x[0]

        # The follower then chooses q2 given the leader's chosen q1
        q2optStackelberg = follower_BestResponse(q1optStackelberg)

        return q1optStackelberg, q2optStackelberg
    
    def calculate_profits(self):
        q1_opt, q2_opt = self.get_optimal_values_Stackelberg()
        profit1 = int(round(self.profit1(q1_opt, q2_opt, self.par.mc[0])))
        profit2 = int(round(self.profit2(q1_opt, q2_opt, self.par.mc[1])))
        return profit1, profit2
    
    #Plot the results (note to us - maybe make the plot a little bit more prettier)

    def plot_Stackelberg(num_points=500):
        # Initialization
        mc_values = np.linspace(0.5, 10, 20,num_points)  # Create an array of different marginal cost values
        qleader_values = []  # Store equilibrium quantities for firm 1
        qfollower_values = []  # Store equilibrium quantities for firm 2

        # Loop over the different marginal cost values
        for mc in mc_values:
            Stackelberg_model = StackelbergDuopoly(mc, mc, a=40)
            q1, q2 = Stackelberg_model.get_optimal_values_Stackelberg()
            qleader_values.append(q1)
            qfollower_values.append(q2)

        # Create the plot
        fig, ax = plt.subplots()
        ax.plot(mc_values, qleader_values, label="Firm 1")
        ax.plot(mc_values, qfollower_values, label="Firm 2")
        ax.set_xlabel('Marginal Cost')
        ax.set_ylabel('Quantity')
        ax.set_title('Stackelberg Duopoly - Nash Equilibrium')
        ax.legend()
        plt.show()





