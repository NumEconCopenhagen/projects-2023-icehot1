import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import fsolve

def euler_equation(c_k, params):
    c, k = c_k
    alpha, rho, theta, g, n, G, k_lag = params

    eq1 = c - ((alpha*k_lag**(alpha-1) - rho - theta*g)/theta) * c
    eq2 = k - (k_lag**alpha - c - G - (n+g)*k_lag)

    return eq1, eq2

def ramsey_model_simulation(T=200, rho=0.1, G_shock=None, plot=True):
    # Parameters
    alpha = 0.3
    theta = 4/3
    g = 0.025
    n = 0.02

    # Steady-state values
    k_ss = (alpha / (rho + theta * g))**(1 / (1 - alpha))
    G_ss = 0
    c_ss = k_ss**alpha - G_ss - (n + g) * k_ss

    # Initialize variables
    k = np.zeros(T + 1)
    c = np.zeros(T)
    r = np.zeros(T)
    w = np.zeros(T)
    y = np.zeros(T)
    dy = np.zeros(T)
    G = np.zeros(T) + G_ss

    k[0] = k_ss

    # Apply G shock
    if G_shock is not None:
        for t, value in G_shock.items():
            G[t:t + len(value)] = G_ss + value

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

        if t > 0:
            dy[t] = (y[t] - y[t - 1]) / y[t - 1] + g
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

        plt.figure()
        plt.plot(time, dy)
        plt.title("Output Growth Rate")
        plt.xlabel("Time")
        plt.ylabel("dy")

        plt.show()

    return k, c, r, w, y, dy
