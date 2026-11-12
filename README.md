# Options Payoff & Greeks Modeling (Black–Scholes)

This project models **European option payoffs** and **Black–Scholes Greeks** to show how option value and risk sensitivities change with the underlying price.

It’s designed as a compact, recruiter-friendly reference for:

- Payoff diagrams (call/put)
- Analytical Greeks under Black–Scholes (Delta, Gamma, Vega, Theta, Rho)
- Visual intuition: how risk changes across moneyness

---

## Model

Under Black–Scholes, the underlying follows Geometric Brownian Motion:

$$
dS_t = r S_t dt + \sigma S_t dW_t
$$

European option prices are computed using the closed-form Black–Scholes formula, and Greeks are computed analytically.

---

## Payoff Diagram

Call and put payoffs at expiration:

- Call payoff: max($S_T - K$, 0)
- Put payoff:  max($K - S_T$, 0)

![Payoff Diagram](./images/payoff_diagram.png)

---

## Greeks vs Underlying Price

The following plot shows how **Delta, Gamma, and Vega** vary as a function of the underlying price $S$.

Key intuition:

- **Delta** increases from ~0 to ~1 as the call moves from OTM → ITM  
- **Gamma** peaks near ATM and falls off in the tails  
- **Vega** is typically highest near ATM (longer-dated options increase Vega)

![Greeks vs Price](./images/greeks_vs_price.png)

---

## Implied Volatility

`src/implied_vol.py` inverts Black–Scholes. Given a market price, it finds $\sigma$ such that

$$
C_{BS}(S, K, T, r, \sigma) = C_{\text{market}}
$$

It uses **safeguarded Newton–Raphson**:

- Newton steps use vega and converge quadratically near the root.
- Every iteration also narrows a $[\sigma_{low}, \sigma_{high}]$ bracket. Price is increasing in $\sigma$, so each evaluation tells you which side of the root you're on.
- Any Newton step that would leave the bracket, or a near-zero vega, falls back to bisection. Plain Newton can diverge for deep in- or out-of-the-money options; this can't.
- It starts from the Brenner–Subrahmanyam approximation, $\sigma_0 \approx \sqrt{2\pi/T}\,C/S$.
- Prices outside the no-arbitrage bounds are rejected, as are prices with no time value, where no unique $\sigma$ exists.

Round trip: price a call at $\sigma = 25\%$ ($S = 100$, $T = 0.5$, $r = 4\%$), then solve back:

| Strike | Price at σ = 25% | Solved σ |
|---|---|---|
| 80 | 22.1882 | 0.25000000 |
| 90 | 14.1127 | 0.25000000 |
| 100 | 8.0080 | 0.25000000 |
| 110 | 4.0599 | 0.25000000 |
| 120 | 1.8575 | 0.25000000 |

```python
from src.implied_vol import implied_volatility

implied_volatility(8.0080, S=100, K=100, T=0.5, r=0.04)  # 0.2500001 (the input price is rounded to 4 dp)
```

---

## Multi-Leg Strategies

`src/strategies.py` prices and analyses option combinations built from signed `Leg(option_type, strike, quantity)` objects. For a strategy $\{(q_i, K_i, \text{type}_i)\}$:

$$
\text{Value} = \sum_i q_i\,V_i(S,K_i,T,r,\sigma), \qquad
\text{Payoff}(S_T)=\sum_i q_i\,\text{intrinsic}_i(S_T), \qquad
\text{P\&L}(S_T)=\text{Payoff}(S_T)-\text{Value}(S_0)
$$

Position Greeks are the quantity-weighted sums of leg Greeks (each checked against finite differences of the strategy value). Because expiry P&L is piecewise linear with kinks only at the strikes, break-evens are found exactly by interpolation, and the maximum profit and loss come from the strike grid plus the net call quantity (which fixes the slope above the top strike).

Built-ins: `long_straddle`, `long_strangle`, `bull_call_spread`, `bear_put_spread`, `butterfly`, `iron_condor`.

Parameters: $S_0=100$, $T=0.5$, $r=4\%$, $\sigma=25\%$.

| Strategy | Net premium | Break-evens | Max profit | Max loss |
|---|---|---|---|---|
| Long straddle (100) | 14.04 | 85.96, 114.04 | unbounded | 14.04 |
| Bull call spread (95/105) | 5.01 | 100.01 | 4.99 | 5.01 |
| Long butterfly (90/100/110) | 2.16 | 92.16, 107.84 | 7.84 | 2.16 |
| Short iron condor (80/90/110/120) | -3.93 (credit) | 86.07, 113.93 | 3.93 | 6.07 |

![Strategy P&L](./images/strategies_pnl.png)

```python
from src import strategies as st

legs = st.iron_condor(80, 90, 110, 120)
st.break_evens(legs, S0=100, T=0.5, r=0.04, sigma=0.25)   # [86.07, 113.93]
st.greeks(legs, S=100, T=0.5, r=0.04, sigma=0.25)["vega"]
```

Regenerate the table and figure with `python -m src.strategy_plots`.

---

## Tests

```bash
pip install -r requirements.txt pytest
pytest -q
```

149 tests cover:

- Reference prices from Hull and put–call parity.
- Every Greek checked against a finite-difference estimate.
- Implied volatility round trips across 120 combinations of strike, maturity and volatility, for calls and puts.
- Error handling at the no-arbitrage bounds.
- Multi-leg strategies: payoffs, additivity of value, convergence to payoff at expiry, position Greeks vs finite differences, break-evens and max profit/loss.

CI runs the suite on Python 3.11, 3.12 and 3.13.

---

## How to Run

From the repo root:

```bash
python -m venv .venv
.\.venv\Scripts\activate
pip install -r requirements.txt
python -m src.visualize
