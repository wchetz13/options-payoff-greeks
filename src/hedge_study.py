"""Delta-hedging study: error vs rebalancing frequency and P&L vs vol mismatch.

Run with ``python -m src.hedge_study``; prints the README tables and saves
``images/delta_hedging.png``.
"""

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from src.delta_hedge import expected_vol_mismatch_pnl, simulate_delta_hedge

PARAMS = dict(S0=100.0, K=100.0, T=0.5, r=0.04)
SIGMA = 0.25
N_PATHS = 20_000
STEPS = [5, 10, 21, 63, 126, 252, 504]
HEDGE_VOLS = [0.15, 0.20, 0.25, 0.30, 0.35]


def frequency_table() -> list[tuple[int, float, float]]:
    """(rebalances, std of PV P&L, std * sqrt(n))."""
    rows = []
    for n in STEPS:
        res = simulate_delta_hedge(**PARAMS, sigma_real=SIGMA, n_steps=n, n_paths=N_PATHS)
        rows.append((n, res.std, res.std * np.sqrt(n)))
    return rows


def mismatch_table() -> list[tuple[float, float, float, float]]:
    """(hedge vol, simulated mean, standard error, analytic mean) at weekly rebalancing."""
    rows = []
    for sh in HEDGE_VOLS:
        res = simulate_delta_hedge(**PARAMS, sigma_real=SIGMA, sigma_hedge=sh,
                                   n_steps=126, n_paths=N_PATHS)
        rows.append((sh, res.mean, res.stderr,
                     expected_vol_mismatch_pnl(**PARAMS, sigma_real=SIGMA, sigma_hedge=sh)))
    return rows


def plot(path: Path, freq, mism) -> None:
    fig, (a, b) = plt.subplots(1, 2, figsize=(11, 4.2))
    n, sd = np.array([f[0] for f in freq]), np.array([f[1] for f in freq])
    a.loglog(n, sd, "o-", label="Simulated std of P&L")
    a.loglog(n, sd[0] * np.sqrt(n[0] / n), "--", color="grey", label=r"$\propto 1/\sqrt{n}$")
    a.set_xlabel("Rebalances over the option's life")
    a.set_ylabel("Std of hedged P&L (PV)")
    a.set_title("Hedging error vs rebalancing frequency")
    a.legend()
    a.grid(alpha=0.3, which="both")
    sh = np.array([m[0] for m in mism])
    b.errorbar(sh, [m[1] for m in mism], yerr=[2 * m[2] for m in mism], fmt="o", capsize=3,
               label="Simulated mean (±2 s.e.)")
    b.plot(sh, [m[3] for m in mism], "--", label="Analytic")
    b.axhline(0, color="grey", lw=0.8)
    b.axvline(SIGMA, color="grey", lw=0.8, ls=":")
    b.set_xlabel("Hedging (implied) volatility; realised = 25%")
    b.set_ylabel("Mean PV P&L of option seller")
    b.set_title("Selling vol: P&L vs implied-realised gap")
    b.legend()
    b.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(path, dpi=150)


def main() -> None:
    freq, mism = frequency_table(), mismatch_table()
    print("| Rebalances | Std of P&L | Std x sqrt(n) |\n|---|---|---|")
    for n, sd, scaled in freq:
        print(f"| {n} | {sd:.3f} | {scaled:.3f} |")
    print("\n| Hedge vol | Simulated mean | Std. error | Analytic |\n|---|---|---|---|")
    for sh, m, se, ex in mism:
        print(f"| {sh:.0%} | {m:+.3f} | {se:.3f} | {ex:+.3f} |")
    out = Path(__file__).resolve().parents[1] / "images" / "delta_hedging.png"
    plot(out, freq, mism)
    print(f"\nsaved {out}")


if __name__ == "__main__":
    main()
