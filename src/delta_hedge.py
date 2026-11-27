"""Discrete delta-hedging simulation under geometric Brownian motion.

A trader sells one European option at the Black-Scholes price computed with the
*hedging* volatility ``sigma_hedge`` and replicates it by holding ``delta`` units of the
underlying, financing the position at the risk-free rate and rebalancing at ``n_steps``
equally spaced dates. The stock actually moves with volatility ``sigma_real``.

Two classical facts are reproduced by :func:`simulate_delta_hedge` and checked in the tests:

* With ``sigma_real == sigma_hedge`` the hedging error is zero in the continuous limit and
  its standard deviation shrinks like ``1 / sqrt(n_steps)``.
* With ``sigma_real != sigma_hedge`` the expected present-value P&L of the seller is
  ``0.5 * (sigma_hedge**2 - sigma_real**2) * integral(exp(-r t) S_t**2 Gamma_t dt)``,
  evaluated by :func:`expected_vol_mismatch_pnl`.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from src.black_scholes_greeks import (
    bs_call_price,
    bs_put_price,
    delta_call,
    delta_put,
    gamma,
)


@dataclass(frozen=True)
class HedgeResult:
    """Per-path present-value P&L of the option seller (premium minus hedging cost)."""

    pnl: np.ndarray
    n_steps: int

    @property
    def mean(self) -> float:
        return float(self.pnl.mean())

    @property
    def std(self) -> float:
        return float(self.pnl.std(ddof=1))

    @property
    def stderr(self) -> float:
        """Monte Carlo standard error of :attr:`mean`."""
        return self.std / np.sqrt(self.pnl.size)


def _price_and_delta(option_type: str):
    if option_type == "call":
        return bs_call_price, delta_call
    if option_type == "put":
        return bs_put_price, delta_put
    raise ValueError("option_type must be 'call' or 'put'")


def simulate_delta_hedge(
    S0: float,
    K: float,
    T: float,
    r: float,
    sigma_real: float,
    sigma_hedge: float | None = None,
    n_steps: int = 52,
    n_paths: int = 20_000,
    option_type: str = "call",
    seed: int | None = 0,
) -> HedgeResult:
    """Simulate hedging a short option position with discrete delta rebalancing.

    The stock follows GBM with drift ``r`` and volatility ``sigma_real``. The seller
    collects the Black-Scholes premium at ``sigma_hedge`` (default ``sigma_real``), holds
    ``delta(S_t, sigma_hedge)`` shares between rebalances, and settles the payoff at ``T``.

    Returns the present value (discounted at ``r``) of terminal wealth, one value per path.
    """
    if min(S0, K, T, sigma_real) <= 0 or n_steps < 1 or n_paths < 2:
        raise ValueError("S0, K, T, sigma_real must be positive; need n_steps >= 1, n_paths >= 2")
    sigma_hedge = sigma_real if sigma_hedge is None else sigma_hedge
    if sigma_hedge <= 0:
        raise ValueError("sigma_hedge must be positive")
    price, delta = _price_and_delta(option_type)

    rng = np.random.default_rng(seed)
    dt = T / n_steps
    growth = np.exp(r * dt)

    S = np.full(n_paths, float(S0))
    premium = float(price(S0, K, T, r, sigma_hedge))
    shares = delta(S, K, T, r, sigma_hedge)
    cash = premium - shares * S

    for i in range(1, n_steps + 1):
        z = rng.standard_normal(n_paths)
        S = S * np.exp((r - 0.5 * sigma_real**2) * dt + sigma_real * np.sqrt(dt) * z)
        cash = cash * growth
        if i < n_steps:
            new_shares = delta(S, K, T - i * dt, r, sigma_hedge)
            cash = cash - (new_shares - shares) * S
            shares = new_shares

    intrinsic = np.maximum(S - K, 0.0) if option_type == "call" else np.maximum(K - S, 0.0)
    terminal_wealth = shares * S + cash - intrinsic
    return HedgeResult(pnl=terminal_wealth * np.exp(-r * T), n_steps=n_steps)


def expected_vol_mismatch_pnl(
    S0: float,
    K: float,
    T: float,
    r: float,
    sigma_real: float,
    sigma_hedge: float,
    n_time: int = 400,
    n_z: int = 4001,
) -> float:
    """Continuous-hedging expected PV P&L of a short option hedged at ``sigma_hedge``.

    ``E = 0.5 (sigma_hedge^2 - sigma_real^2) * int_0^T e^{-r t} E[S_t^2 Gamma(S_t, t)] dt``
    with ``S_t`` lognormal under volatility ``sigma_real`` and drift ``r``. The gamma is
    evaluated at ``sigma_hedge`` because that is the delta actually held. Gamma is a narrow
    spike near the strike close to expiry, which Gauss-Hermite quadrature resolves poorly,
    so the inner expectation is a trapezoid sum over a dense standard-normal grid on
    [-8, 8]; the time integral uses the midpoint rule.
    """
    z = np.linspace(-8.0, 8.0, n_z)
    pdf = np.exp(-0.5 * z**2) / np.sqrt(2.0 * np.pi)
    dz = z[1] - z[0]
    t = (np.arange(n_time) + 0.5) * (T / n_time)
    total = 0.0
    for ti in t:
        St = S0 * np.exp((r - 0.5 * sigma_real**2) * ti + sigma_real * np.sqrt(ti) * z)
        g = gamma(St, K, T - ti, r, sigma_hedge)
        total += np.exp(-r * ti) * float(np.sum(pdf * St**2 * g) * dz)
    return 0.5 * (sigma_hedge**2 - sigma_real**2) * total * (T / n_time)
