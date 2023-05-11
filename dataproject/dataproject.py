import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import fsolve

def euler_equation(c_k, params):
    c, k = c_k
    alpha, rho, theta, g, n, G, k_lag = params

    eq1 = c - ((alpha*k_lag**(alpha-1) - rho - theta*g)/theta) * c
    eq2 = k - (k_lag**alpha - c - G - (n+g)*k_lag)

    return eq1, eq2

def ramsey_model_simulation(T=100, rho=0.03, G_multiplier=1, G_increase_duration=1, plot=True):
    # Parameters
    alpha = 1/3
    theta = 1
    g = 0.02
    n = 0.01

    # Steady-state values
    k_ss = (alpha / (rho + theta * g))**(1 / (1 - alpha))
    y_ss = k_ss**alpha
    G_ss = 0.5 * y_ss  # 50% of the initial steady-state GDP
    c_ss = y_ss - G_ss - (n + g) * k_ss

    # Initialize variables
    k = np.zeros(T + 1)
    c = np.zeros(T)
    r = np.zeros(T)
    w = np.zeros(T)
    y = np.zeros(T)
    G = np.zeros(T)

    k[0] = k_ss

    # Generate G path
    G[:G_increase_duration] = G_ss * G_multiplier
    G[G_increase_duration:] = G_ss

    # Model simulation
    for t in range(T):
        params = (alpha, rho, theta, g, n, G[t], k[t])
        initial_guess = (c_ss, k_ss)
        c_t, k_t1 = fsolve(euler_equation, initial_guess, args=(params,))

        c[t] = c_t
        k[t + 1] = k_t1

        r[t] = alpha * k[t]**(alpha - 1)
        w[t] = (1 - alpha) * k[t]**alpha
        y[t] = k[t]**alpha

    # Plot results
    if plot:
        time = np.arange(T)

        plt.figure()
        plt.plot(time, k[1:])
        plt.title("Capital Stock")
        plt.xlabel("Time")
        plt.ylabel("k")

        plt.figure()
        plt.plot(time, y)
        plt.title("Output")
        plt.xlabel("Time")
        plt.ylabel("y")

        plt.figure()
        plt.plot(time, c)
        plt.title("Consumption")
        plt.xlabel("Time")
        plt.ylabel("c")

        plt.figure()
        plt.plot(time, r)
        plt.title("Interest Rate")
        plt.xlabel("Time")
        plt.ylabel("r")

        plt.figure()
        plt.plot(time, w)
        plt.title("Wage")
        plt.xlabel("Time")
        plt.ylabel("w")

        plt.show()

    return k, y, c, r, w
