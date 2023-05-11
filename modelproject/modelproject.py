import numpy as np
import matplotlib.pyplot as plt

def ramsey_model_simulation(T=100, plot=True):
    # Parameters
    alpha = 1/3
    rho0 = 0.03
    rho1 = 0.06
    theta = 1
    g = 0.02
    n = 0.01
    G = 0

    # Steady-state values
    k_ss = (alpha/(rho0+theta*g))**(1/(1-alpha))
    c_ss = k_ss**alpha - G - (n+g)*k_ss

    # Initialize variables
    k = np.zeros(T+1)
    c = np.zeros(T)
    r = np.zeros(T)
    w = np.zeros(T)
    y = np.zeros(T)

    k[0] = k_ss

    # Model simulation
    for t in range(T):
        r[t] = alpha * k[t]**(alpha-1)
        w[t] = (1-alpha) * k[t]**alpha
        y[t] = k[t]**alpha
        c[t] = ((alpha*k[t]**(alpha-1) - rho1 - theta*g)/theta) * c_ss
        k[t+1] = k[t]**alpha - c[t] - G - (n+g)*k[t]

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

def ramsey_model_simulation_v2(T=100, G0=0, G1=0.5, plot=True):
    # Parameters
    alpha = 1/3
    rho = 0.03
    theta = 1
    g = 0.02
    n = 0.01

    # Steady-state values
    k_ss = (alpha/(rho+theta*g))**(1/(1-alpha))
    c_ss = k_ss**alpha - G0 - (n+g)*k_ss

    # Initialize variables
    k = np.zeros(T+1)
    c = np.zeros(T)
    r = np.zeros(T)
    w = np.zeros(T)
    y = np.zeros(T)

    k[0] = k_ss

    # Model simulation
    for t in range(T):
        r[t] = alpha * k[t]**(alpha-1)
        w[t] = (1-alpha) * k[t]**alpha
        y[t] = k[t]**alpha
        c[t] = ((alpha*k[t]**(alpha-1) - rho - theta*g)/theta) * c_ss
        k[t+1] = k[t]**alpha - c[t] - G1 - (n+g)*k[t]

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
