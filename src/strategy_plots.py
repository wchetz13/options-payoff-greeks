"""Generate strategy P&L and Greeks figures: ``python -m src.strategy_plots``."""

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from src import strategies as st

S0, T, R, SIG = 100.0, 0.5, 0.04, 0.25
STRATEGIES = {
    "Long straddle (100)": st.long_straddle(100),
    "Bull call spread (95/105)": st.bull_call_spread(95, 105),
    "Long butterfly (90/100/110)": st.butterfly(90, 100, 110),
    "Short iron condor (80/90/110/120)": st.iron_condor(80, 90, 110, 120),
}


def summary_table() -> list[tuple]:
    """Rows of (name, net premium, break-evens, max profit, max loss)."""
    rows = []
    for name, legs in STRATEGIES.items():
        mx, ml = st.max_profit_loss(legs, S0, T, R, SIG)
        rows.append((name, st.net_premium(legs, S0, T, R, SIG),
                     st.break_evens(legs, S0, T, R, SIG), mx, ml))
    return rows


def plot(path: Path) -> None:
    grid = np.linspace(60, 140, 400)
    fig, axes = plt.subplots(2, 2, figsize=(11, 7), sharex=True)
    for ax, (name, legs) in zip(axes.ravel(), STRATEGIES.items()):
        ax.plot(grid, st.profit(legs, grid, S0, T, R, SIG), label="At expiry")
        ax.plot(grid, st.value(legs, grid, T / 2, R, SIG) - st.net_premium(legs, S0, T, R, SIG),
                "--", label="Halfway to expiry")
        ax.axhline(0, color="grey", lw=0.8)
        ax.set_title(name, fontsize=10)
        ax.grid(alpha=0.3)
    axes[0, 0].legend(fontsize=8)
    for ax in axes[1]:
        ax.set_xlabel("Underlying price")
    for ax in axes[:, 0]:
        ax.set_ylabel("P&L")
    fig.tight_layout()
    fig.savefig(path, dpi=150)


if __name__ == "__main__":
    for row in summary_table():
        print(row)
    plot(Path("images/strategies_pnl.png"))
