import numpy as np
import matplotlib.pyplot as plt
from src.black_scholes_greeks import *


def payoff_plot():
    S = np.linspace(0, 200, 500)
    K = 100

    call_payoff = np.maximum(S - K, 0)
    put_payoff = np.maximum(K - S, 0)

    plt.figure(figsize=(10, 6))
    plt.plot(S, call_payoff, label="Call Payoff")
    plt.plot(S, put_payoff, label="Put Payoff")
    plt.title("Option Payoff Diagram")
    plt.xlabel("Stock Price")
    plt.ylabel("Payoff")
    plt.legend()
    plt.grid()
    plt.show()


def greek_plot():
    S = np.linspace(50, 150, 500)
    K, T, r, sigma = 100, 1, 0.05, 0.2

    deltas = [delta_call(s, K, T, r, sigma) for s in S]
    gammas = [gamma(s, K, T, r, sigma) for s in S]
    vegas = [vega(s, K, T, r, sigma) for s in S]

    plt.figure(figsize=(10, 6))
    plt.plot(S, deltas, label="Delta")
    plt.plot(S, gammas, label="Gamma")
    plt.plot(S, vegas, label="Vega")
    plt.title("Greeks vs Stock Price")
    plt.xlabel("Stock Price")
    plt.legend()
    plt.grid()
    plt.show()


if __name__ == "__main__":
    payoff_plot()
    greek_plot()
