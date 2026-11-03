import numpy as np
import pytest

from src.black_scholes_greeks import bs_call_price, bs_put_price, vega
from src.implied_vol import implied_volatility, price_bounds

PRICERS = {"call": bs_call_price, "put": bs_put_price}


@pytest.mark.parametrize("option_type", ["call", "put"])
@pytest.mark.parametrize("K", [60, 80, 100, 120, 150])
@pytest.mark.parametrize("T", [0.1, 1.0, 3.0])
@pytest.mark.parametrize("sigma", [0.05, 0.2, 0.6, 1.5])
def test_round_trip(option_type, K, T, sigma):
    S, r = 100.0, 0.03
    price = PRICERS[option_type](S, K, T, r, sigma)
    lower, _ = price_bounds(S, K, T, r, option_type)
    if price <= lower:
        # No time value left (to float precision): every low sigma gives the same price, so
        # there is nothing to invert and the solver must refuse rather than guess.
        with pytest.raises(ValueError, match="no-arbitrage"):
            implied_volatility(price, S, K, T, r, option_type)
        return
    solved = implied_volatility(price, S, K, T, r, option_type)
    # The solver always reproduces the price; sigma itself is only identifiable when vega
    # is large enough for a price tolerance to pin it down.
    assert PRICERS[option_type](S, K, T, r, solved) == pytest.approx(price, abs=1e-9)
    if vega(S, K, T, r, sigma) > 1e-3:
        assert solved == pytest.approx(sigma, abs=1e-6)


def test_deep_otm_from_a_bad_starting_point_does_not_diverge():
    S, K, T, r, sigma = 100.0, 250.0, 0.25, 0.0, 0.9
    price = bs_call_price(S, K, T, r, sigma)
    assert implied_volatility(price, S, K, T, r) == pytest.approx(sigma, abs=1e-6)


@pytest.mark.parametrize("price", [0.0, -1.0, 100.0, 150.0])
def test_rejects_prices_outside_no_arbitrage_bounds(price):
    with pytest.raises(ValueError, match="no-arbitrage"):
        implied_volatility(price, 100.0, 100.0, 1.0, 0.05)


def test_rejects_unknown_option_type():
    with pytest.raises(ValueError, match="option_type"):
        implied_volatility(5.0, 100.0, 100.0, 1.0, 0.05, "straddle")


def test_rejects_volatility_above_solver_range():
    S, K, T, r = 100.0, 100.0, 1.0, 0.0
    price = 0.5 * (bs_call_price(S, K, T, r, 5.0) + S)
    with pytest.raises(ValueError, match="exceeds"):
        implied_volatility(price, S, K, T, r)


def test_bounds_satisfy_put_call_parity():
    S, K, T, r = 100.0, 90.0, 2.0, 0.04
    call_low, _ = price_bounds(S, K, T, r, "call")
    put_low, put_high = price_bounds(S, K, T, r, "put")
    assert call_low == pytest.approx(S - K * np.exp(-r * T))
    assert put_low == 0.0
    assert put_high == pytest.approx(K * np.exp(-r * T))
