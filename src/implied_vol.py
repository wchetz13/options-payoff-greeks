"""Implied volatility: the sigma at which Black–Scholes reproduces a market price."""

import numpy as np

from src.black_scholes_greeks import bs_call_price, bs_put_price, vega

VOL_LOW, VOL_HIGH = 1e-6, 5.0


def price_bounds(S, K, T, r, option_type="call"):
    """Model-free no-arbitrage bounds on a European option price."""
    pv_strike = K * np.exp(-r * T)
    if option_type == "call":
        return max(S - pv_strike, 0.0), S
    return max(pv_strike - S, 0.0), pv_strike


def implied_volatility(price, S, K, T, r, option_type="call", tol=1e-10, max_iter=100):
    """Solve for sigma with safeguarded Newton–Raphson.

    Newton steps use vega and converge quadratically near the root. Every iteration also
    tightens a [low, high] bracket (price is increasing in sigma), and any Newton step that
    would leave the bracket, or a vanishing vega, falls back to bisection, so the solver
    cannot diverge the way plain Newton does for deep in- or out-of-the-money options.
    """
    if option_type not in ("call", "put"):
        raise ValueError("option_type must be 'call' or 'put'")
    pricer = bs_call_price if option_type == "call" else bs_put_price
    lower, upper = price_bounds(S, K, T, r, option_type)
    if not lower < price < upper:
        raise ValueError(f"price {price} outside no-arbitrage bounds ({lower:.6f}, {upper:.6f})")
    if pricer(S, K, T, r, VOL_HIGH) < price:
        raise ValueError(f"implied volatility exceeds {VOL_HIGH:.0%}")

    low, high = VOL_LOW, VOL_HIGH
    # Brenner–Subrahmanyam approximation: a good start for near-the-money options.
    sigma = float(np.clip(np.sqrt(2 * np.pi / T) * price / S, 0.05, 1.0))
    for _ in range(max_iter):
        diff = pricer(S, K, T, r, sigma) - price
        if abs(diff) < tol:
            return sigma
        if diff > 0:
            high = sigma
        else:
            low = sigma
        v = vega(S, K, T, r, sigma)
        step = sigma - diff / v if v > 1e-12 else np.nan
        sigma = step if low < step < high else 0.5 * (low + high)
        if high - low < 1e-14:
            return sigma
    raise RuntimeError("implied volatility did not converge")
