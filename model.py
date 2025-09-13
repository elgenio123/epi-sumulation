import numpy as np
from scipy.integrate import odeint
import matplotlib.pyplot as plt

# --- SIR Model (Susceptible, Infected, Recovered) ---

def sir_model(y, t, beta, gamma):
    """
    SIR model differential equations.

    Args:
        y (list): A list containing the current number of S, I, and R individuals.
        t (float): Time.
        beta (float): The infection rate.
        gamma (float): The recovery rate.

    Returns:
        list: The derivatives dS/dt, dI/dt, and dR/dt.
    """
    S, I, R = y
    N = S + I + R
    dSdt = -beta * S * I / N
    dIdt = beta * S * I / N - gamma * I
    dRdt = gamma * I
    return [dSdt, dIdt, dRdt]

def seir_model(y, t, beta, sigma, gamma):
    """
    SEIR model differential equations.

    Args:
        y (list): A list containing the current number of S, E, I, and R individuals.
        t (float): Time.
        beta (float): The infection rate.
        sigma (float): The rate at which exposed individuals become infectious (incubation period inverse).
        gamma (float): The recovery rate.

    Returns:
        list: The derivatives dS/dt, dE/dt, dI/dt, and dR/dt.
    """
    S, E, I, R = y
    N = S + E + I + R
    dSdt = -beta * S * I / N
    dEdt = beta * S * I / N - sigma * E
    dIdt = sigma * E - gamma * I
    dRdt = gamma * I
    return [dSdt, dEdt, dIdt, dRdt]