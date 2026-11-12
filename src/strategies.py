"""Multi-leg option strategies: payoff, value, Greeks and break-evens.

A strategy is a list of :class:`Leg` objects. Quantities are signed (positive = long,
negative = short) and expressed in contracts of one unit of the underlying.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Sequence

import numpy as np
from scipy.stats import norm

from src.black_scholes_greeks import (
    bs_call_price,
    bs_put_price,
    d1,
    d2,
    delta_call,
    delta_put,
    gamma,
    rho_call,
    theta_call,
    vega,
)


@dataclass(frozen=True)
class Leg:
    """One option position: signed quantity, strike and call/put type."""

    option_type: str
    strike: float
    quantity: float = 1.0

    def __post_init__(self) -> None:
        if self.option_type not in ("call", "put"):
            raise ValueError("option_type must be 'call' or 'put'")
        if self.strike <= 0:
            raise ValueError("strike must be positive")


Strategy = Sequence[Leg]


def theta_put(S, K, T, r, sigma):
    """Black-Scholes put theta (per year)."""
    term1 = -(S * norm.pdf(d1(S, K, T, r, sigma)) * sigma) / (2 * np.sqrt(T))
    return term1 + r * K * np.exp(-r * T) * norm.cdf(-d2(S, K, T, r, sigma))


def rho_put(S, K, T, r, sigma):
    """Black-Scholes put rho."""
    return -K * T * np.exp(-r * T) * norm.cdf(-d2(S, K, T, r, sigma))


_PRICE = {"call": bs_call_price, "put": bs_put_price}
_DELTA = {"call": delta_call, "put": delta_put}
_THETA = {"call": theta_call, "put": theta_put}
_RHO = {"call": rho_call, "put": rho_put}


def payoff(legs: Strategy, S_T) -> np.ndarray:
    """Total payoff at expiry (ignoring premium) for terminal price(s) ``S_T``."""
    S_T = np.asarray(S_T, dtype=float)
    total = np.zeros_like(S_T)
    for leg in legs:
        intrinsic = (
            np.maximum(S_T - leg.strike, 0.0)
            if leg.option_type == "call"
            else np.maximum(leg.strike - S_T, 0.0)
        )
        total = total + leg.quantity * intrinsic
    return total


def value(legs: Strategy, S, T, r, sigma) -> np.ndarray:
    """Black-Scholes mark-to-market value of the whole strategy."""
    return sum(leg.quantity * _PRICE[leg.option_type](S, leg.strike, T, r, sigma) for leg in legs)


def net_premium(legs: Strategy, S, T, r, sigma) -> float:
    """Net cost to open the strategy (positive = debit, negative = credit)."""
    return float(value(legs, S, T, r, sigma))


def profit(legs: Strategy, S_T, S0, T, r, sigma) -> np.ndarray:
    """Expiry P&L: payoff minus the premium paid today at spot ``S0``."""
    return payoff(legs, S_T) - net_premium(legs, S0, T, r, sigma)


def _aggregate(legs: Strategy, fn: Callable[[Leg], np.ndarray]) -> np.ndarray:
    return sum(leg.quantity * fn(leg) for leg in legs)


def greeks(legs: Strategy, S, T, r, sigma) -> dict[str, np.ndarray]:
    """Position-level delta, gamma, vega, theta and rho (sums of the legs' Greeks)."""
    return {
        "delta": _aggregate(legs, lambda l: _DELTA[l.option_type](S, l.strike, T, r, sigma)),
        "gamma": _aggregate(legs, lambda l: gamma(S, l.strike, T, r, sigma)),
        "vega": _aggregate(legs, lambda l: vega(S, l.strike, T, r, sigma)),
        "theta": _aggregate(legs, lambda l: _THETA[l.option_type](S, l.strike, T, r, sigma)),
        "rho": _aggregate(legs, lambda l: _RHO[l.option_type](S, l.strike, T, r, sigma)),
    }


def break_evens(legs: Strategy, S0, T, r, sigma, s_max: float | None = None) -> list[float]:
    """Expiry break-even prices, found exactly on the piecewise-linear P&L.

    P&L is linear between strikes and beyond the last strike, so zero crossings are located
    by linear interpolation on the strike grid (plus 0 and ``s_max``).
    """
    strikes = sorted({leg.strike for leg in legs})
    s_max = s_max if s_max is not None else 2.0 * strikes[-1] + 1.0
    grid = np.array([0.0, *strikes, s_max])
    pnl = profit(legs, grid, S0, T, r, sigma)
    roots: list[float] = []
    for x0, x1, y0, y1 in zip(grid[:-1], grid[1:], pnl[:-1], pnl[1:]):
        if y0 == 0.0:
            roots.append(float(x0))
        if y0 * y1 < 0:
            roots.append(float(x0 - y0 * (x1 - x0) / (y1 - y0)))
    if pnl[-1] == 0.0:
        roots.append(float(grid[-1]))
    return sorted(set(round(x, 10) for x in roots))


def max_profit_loss(legs: Strategy, S0, T, r, sigma) -> tuple[float, float]:
    """(max profit, max loss) over S_T in [0, inf); ``inf`` / ``-inf`` if unbounded.

    Slope beyond the top strike is the net call quantity; at 0 the P&L is finite.
    """
    strikes = sorted({leg.strike for leg in legs})
    grid = np.array([0.0, *strikes])
    pnl = profit(legs, grid, S0, T, r, sigma)
    net_calls = sum(l.quantity for l in legs if l.option_type == "call")
    hi = np.inf if net_calls > 0 else float(pnl.max())
    lo = -np.inf if net_calls < 0 else float(pnl.min())
    if net_calls == 0:
        hi, lo = float(pnl.max()), float(pnl.min())
    return hi, lo


# ---- Named strategies ----

def long_straddle(K: float) -> list[Leg]:
    return [Leg("call", K), Leg("put", K)]


def long_strangle(K_put: float, K_call: float) -> list[Leg]:
    return [Leg("put", K_put), Leg("call", K_call)]


def bull_call_spread(K_low: float, K_high: float) -> list[Leg]:
    return [Leg("call", K_low, 1), Leg("call", K_high, -1)]


def bear_put_spread(K_low: float, K_high: float) -> list[Leg]:
    return [Leg("put", K_high, 1), Leg("put", K_low, -1)]


def butterfly(K_low: float, K_mid: float, K_high: float) -> list[Leg]:
    """Long call butterfly: +1 low, -2 mid, +1 high."""
    return [Leg("call", K_low, 1), Leg("call", K_mid, -2), Leg("call", K_high, 1)]


def iron_condor(K1: float, K2: float, K3: float, K4: float) -> list[Leg]:
    """Short iron condor (K1<K2<K3<K4): long K1 put, short K2 put, short K3 call, long K4 call."""
    if not K1 < K2 < K3 < K4:
        raise ValueError("strikes must satisfy K1 < K2 < K3 < K4")
    return [Leg("put", K1, 1), Leg("put", K2, -1), Leg("call", K3, -1), Leg("call", K4, 1)]
