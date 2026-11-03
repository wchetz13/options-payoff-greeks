import numpy as np
import pytest

from src.black_scholes_greeks import (
    bs_call_price,
    bs_put_price,
    delta_call,
    delta_put,
    gamma,
    rho_call,
    theta_call,
    vega,
)

S, K, T, r, SIGMA = 100.0, 100.0, 1.0, 0.05, 0.2
H = 1e-4


def test_reference_prices():
    # Hull, Options, Futures, and Other Derivatives: S = K = 100, r = 5%, sigma = 20%, T = 1.
    assert bs_call_price(S, K, T, r, SIGMA) == pytest.approx(10.450584, abs=1e-6)
    assert bs_put_price(S, K, T, r, SIGMA) == pytest.approx(5.573526, abs=1e-6)


@pytest.mark.parametrize("strike", [70.0, 100.0, 130.0])
def test_put_call_parity(strike):
    lhs = bs_call_price(S, strike, T, r, SIGMA) - bs_put_price(S, strike, T, r, SIGMA)
    assert lhs == pytest.approx(S - strike * np.exp(-r * T), abs=1e-10)


def test_greeks_match_finite_differences():
    def call(s=S, t=T, rate=r, sig=SIGMA):
        return bs_call_price(s, K, t, rate, sig)

    assert delta_call(S, K, T, r, SIGMA) == pytest.approx((call(s=S + H) - call(s=S - H)) / (2 * H))
    assert delta_put(S, K, T, r, SIGMA) == pytest.approx(delta_call(S, K, T, r, SIGMA) - 1)
    assert gamma(S, K, T, r, SIGMA) == pytest.approx(
        (call(s=S + H) - 2 * call() + call(s=S - H)) / H**2, rel=1e-4
    )
    assert vega(S, K, T, r, SIGMA) == pytest.approx(
        (call(sig=SIGMA + H) - call(sig=SIGMA - H)) / (2 * H)
    )
    assert rho_call(S, K, T, r, SIGMA) == pytest.approx(
        (call(rate=r + H) - call(rate=r - H)) / (2 * H)
    )
    # Theta is the change per year as calendar time passes, i.e. as T shrinks.
    assert theta_call(S, K, T, r, SIGMA) == pytest.approx(
        -(call(t=T + H) - call(t=T - H)) / (2 * H)
    )
