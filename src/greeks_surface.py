"""Greeks evaluated on grids: spot x maturity and strike x maturity surfaces.

All functions are vectorised over NumPy arrays, so a whole surface is one call with no Python
loops. Delta, gamma, vega and theta reuse :mod:`src.black_scholes_greeks`; vanna and volga
(second-order cross/vol Greeks that the base module lacks) are defined here.

Conventions: vega, vanna and volga are per unit of volatility (1.00 = 100 vol points); theta is
per year. Use the ``scale`` arguments of :func:`plot_surface` to rescale for display.
"""

from __future__ import annotations

from typing import Callable

import numpy as np
from scipy.stats import norm

from src.black_scholes_greeks import (
    d1,
    d2,
    delta_call,
    delta_put,
    gamma,
    rho_call,
    theta_call,
    vega,
)

Array = np.ndarray
GreekFn = Callable[[Array, Array, Array, float, float], Array]


def vanna(S, K, T, r, sigma):
    r"""Vanna :math:`\partial^2 V/\partial S\,\partial\sigma = -\varphi(d_1)\,d_2/\sigma`."""
    return -norm.pdf(d1(S, K, T, r, sigma)) * d2(S, K, T, r, sigma) / sigma


def volga(S, K, T, r, sigma):
    r"""Volga (vomma) :math:`\partial^2 V/\partial\sigma^2 = \mathrm{vega}\cdot d_1 d_2/\sigma`."""
    return vega(S, K, T, r, sigma) * d1(S, K, T, r, sigma) * d2(S, K, T, r, sigma) / sigma


def theta_put(S, K, T, r, sigma):
    """Put theta (per year), from the call theta via put-call parity."""
    return theta_call(S, K, T, r, sigma) + r * K * np.exp(-r * T)


# Greeks that are identical for calls and puts, plus the call/put-specific ones.
_COMMON: dict[str, GreekFn] = {"gamma": gamma, "vega": vega, "vanna": vanna, "volga": volga}
_CALL: dict[str, GreekFn] = {"delta": delta_call, "theta": theta_call, "rho": rho_call}
_PUT: dict[str, GreekFn] = {"delta": delta_put, "theta": theta_put}


def available_greeks(option_type: str = "call") -> list[str]:
    """Names accepted by :func:`greek_grid` for the given option type."""
    table = _CALL if option_type == "call" else _PUT
    return sorted({**_COMMON, **table})


def greek_grid(
    greek: str,
    x: Array,
    T: Array,
    *,
    axis: str = "spot",
    S: float = 100.0,
    K: float = 100.0,
    r: float = 0.05,
    sigma: float = 0.2,
    option_type: str = "call",
) -> Array:
    """Evaluate a Greek on a (len(T), len(x)) grid.

    Args:
        greek: One of :func:`available_greeks`.
        x: Values of the varying quantity along columns: spot prices if ``axis == "spot"``,
            strikes if ``axis == "strike"``.
        T: Times to maturity (years, > 0) along rows.
        axis: Whether ``x`` is spot (strike fixed at ``K``) or strike (spot fixed at ``S``).
        S, K, r, sigma: Fixed model inputs. ``sigma`` is flat across the grid.
        option_type: ``"call"`` or ``"put"``.

    Returns:
        Array of shape ``(len(T), len(x))``; row i is maturity ``T[i]``.
    """
    if option_type not in ("call", "put"):
        raise ValueError("option_type must be 'call' or 'put'")
    if axis not in ("spot", "strike"):
        raise ValueError("axis must be 'spot' or 'strike'")
    table = {**_COMMON, **(_CALL if option_type == "call" else _PUT)}
    if greek not in table:
        raise ValueError(f"unknown greek {greek!r} for a {option_type}; choose from {sorted(table)}")
    x = np.asarray(x, dtype=float)
    T = np.asarray(T, dtype=float)
    if np.any(x <= 0) or np.any(T <= 0):
        raise ValueError("x and T must be strictly positive")
    xx, TT = np.meshgrid(x, T)
    spot, strike = (xx, K) if axis == "spot" else (S, xx)
    return table[greek](spot, strike, TT, r, sigma)


def plot_heatmaps(path: str | None = None):
    """Draw a 2x3 panel of call Greek heatmaps (spot x maturity) and optionally save it."""
    import matplotlib

    if path is not None:
        matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    spots = np.linspace(60, 140, 161)
    mats = np.linspace(0.05, 2.0, 120)
    panels = [
        ("delta", "Delta", "viridis", 1.0),
        ("gamma", "Gamma", "magma", 1.0),
        ("vega", "Vega (per 1 vol pt)", "magma", 0.01),
        ("theta", "Theta (per day)", "viridis", 1 / 365),
        ("vanna", "Vanna (per 1 vol pt)", "coolwarm", 0.01),
        ("volga", "Volga (per 1 vol pt$^2$)", "coolwarm", 1e-4),
    ]
    fig, axes = plt.subplots(2, 3, figsize=(14, 7.5), constrained_layout=True)
    for ax, (name, title, cmap, scale) in zip(axes.flat, panels):
        z = greek_grid(name, spots, mats, axis="spot") * scale
        kw = {}
        if cmap == "coolwarm":  # diverging: centre colour scale on zero
            lim = np.abs(z).max()
            kw = {"vmin": -lim, "vmax": lim}
        mesh = ax.pcolormesh(spots, mats, z, cmap=cmap, shading="auto", **kw)
        ax.axvline(100, color="white", lw=0.8, ls="--")
        ax.set(title=title, xlabel="Spot $S$ (K = 100)", ylabel="Maturity $T$ (years)")
        fig.colorbar(mesh, ax=ax)
    fig.suptitle("Black-Scholes call Greeks: spot x maturity (r = 5%, $\\sigma$ = 20%)")
    if path is not None:
        fig.savefig(path, dpi=130)
    return fig


def summary_table() -> list[tuple[str, float, float, float]]:
    """ATM call Greeks at S=K=100 for T = 0.1, 0.5, 1.0 (used in the docs)."""
    T = np.array([0.1, 0.5, 1.0])
    rows = []
    for g in ("delta", "gamma", "vega", "theta", "vanna", "volga"):
        v = greek_grid(g, np.array([100.0]), T)[:, 0]
        rows.append((g, *map(float, v)))
    return rows


if __name__ == "__main__":
    import os

    os.makedirs("images", exist_ok=True)
    plot_heatmaps("images/greeks_heatmaps.png")
    print("| Greek | T=0.1 | T=0.5 | T=1.0 |\n|---|---|---|---|")
    for name, *vals in summary_table():
        print(f"| {name} | " + " | ".join(f"{v:.5f}" for v in vals) + " |")
