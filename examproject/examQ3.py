import numpy as np
from scipy.optimize import minimize
import matplotlib.pyplot as plt
import time

def refined_global_optimizer(func, num_parameters, bounds, tol, K_warmup, K_max):
    
    best_x = None 
    best_f = None
    initial_guesses = []

    for k in range(K_max):
        # A. Draw random initial guess, x^k
        xk = np.random.uniform(low=bounds[0], high=bounds[1], size=num_parameters) 
        #size is the number of parameters in the input function.  


        # B. If k < K_warmup, go to step E.
        if k >= K_warmup: 
        #setting the argument for if k >= K_warmup, then calculate chi_k and set x^k_0, else go to step E where x^k_0=x_k
            
            # C. Calculate chik
            chi_k = 0.5 * (2 / (1 + np.exp((k - K_warmup) / 100)))

            # D. Set xk0
            xk0 = chi_k * xk + (1 - chi_k) * best_x
        else:
            xk0 = xk

        initial_guesses.append(xk0)

        # E. Run optimizer 
        #Use a local optimization method (BFGS) to find a local minimum of the function from the initial guess xk0
        res = minimize(func, xk0, method='BFGS', tol=tol)

        # F. Update best result
        #If the function value at the new solution (res.fun) is better than the current best function value
        if best_f is None or res.fun < best_f:
            best_f = res.fun
            best_x = res.x

        # G. If f(x*) < tol, break loop
        if best_f < tol:
            break

    return best_x, best_f, initial_guesses

def run_optimization_and_measure_time(func, num_parameters, bounds, tol, K_warmup, K_max):
    
    start_time = time.time()
    best_x, best_f, initial_guesses = refined_global_optimizer(func, num_parameters, bounds, tol, K_warmup, K_max)
    end_time = time.time()

    execution_time = end_time - start_time
    print(f"Execution time with K_warmup = {K_warmup}: {execution_time}")
    print(f"Best solution with K_warmup = {K_warmup}: {best_x}")
    print(f"Function value at best solution: {best_f}")

    return best_x, best_f, execution_time


