import numpy as np
import pytest

from src import strategies as st
from src.black_scholes_greeks import bs_call_price, bs_put_price

S0, T, R, SIG = 100.0, 0.5, 0.04, 0.25
ALL = {
    "straddle": st.long_straddle(100),
    "strangle": st.long_strangle(90, 110),
    "bull": st.bull_call_spread(95, 105),
    "bear": st.bear_put_spread(95, 105),
    "fly": st.butterfly(90, 100, 110),
    "condor": st.iron_condor(80, 90, 110, 120),
}


def test_invalid_leg():
    with pytest.raises(ValueError):
        st.Leg("future", 100)
    with pytest.raises(ValueError):
        st.iron_condor(90, 80, 110, 120)


def test_straddle_payoff():
    s = np.array([60.0, 100.0, 140.0])
    assert np.allclose(st.payoff(st.long_straddle(100), s), [40, 0, 40])


def test_butterfly_payoff_shape():
    s = np.array([80.0, 90.0, 100.0, 110.0, 120.0])
    assert np.allclose(st.payoff(st.butterfly(90, 100, 110), s), [0, 0, 10, 0, 0])


def test_value_is_sum_of_legs():
    legs = st.iron_condor(80, 90, 110, 120)
    expected = (
        bs_put_price(S0, 80, T, R, SIG) - bs_put_price(S0, 90, T, R, SIG)
        - bs_call_price(S0, 110, T, R, SIG) + bs_call_price(S0, 120, T, R, SIG)
    )
    assert st.value(legs, S0, T, R, SIG) == pytest.approx(expected)


def test_value_converges_to_payoff_at_expiry():
    s = np.array([70.0, 97.0, 103.0, 125.0])
    for legs in ALL.values():
        assert np.allclose(st.value(legs, s, 1e-8, R, 0.2), st.payoff(legs, s), atol=1e-4)


@pytest.mark.parametrize("name", list(ALL))
def test_greeks_match_finite_differences(name):
    legs = ALL[name]
    g = st.greeks(legs, S0, T, R, SIG)
    h = 1e-3
    v = lambda S=S0, t=T, r=R, s=SIG: st.value(legs, S, t, r, s)
    assert g["delta"] == pytest.approx((v(S=S0 + h) - v(S=S0 - h)) / (2 * h), abs=1e-6)
    assert g["gamma"] == pytest.approx((v(S=S0 + h) - 2 * v() + v(S=S0 - h)) / h**2, abs=1e-4)
    assert g["vega"] == pytest.approx((v(s=SIG + 1e-5) - v(s=SIG - 1e-5)) / 2e-5, abs=1e-5)
    assert g["rho"] == pytest.approx((v(r=R + 1e-5) - v(r=R - 1e-5)) / 2e-5, abs=1e-4)
    assert g["theta"] == pytest.approx(-(v(t=T + 1e-5) - v(t=T - 1e-5)) / 2e-5, abs=1e-4)


def test_straddle_break_evens_symmetric_about_strike_plus_premium():
    legs = st.long_straddle(100)
    prem = st.net_premium(legs, S0, T, R, SIG)
    lo, hi = st.break_evens(legs, S0, T, R, SIG)
    assert lo == pytest.approx(100 - prem)
    assert hi == pytest.approx(100 + prem)


def test_bull_call_spread_bounds():
    legs = st.bull_call_spread(95, 105)
    prem = st.net_premium(legs, S0, T, R, SIG)
    mx, ml = st.max_profit_loss(legs, S0, T, R, SIG)
    assert mx == pytest.approx(10 - prem)
    assert ml == pytest.approx(-prem)
    assert st.break_evens(legs, S0, T, R, SIG) == [pytest.approx(95 + prem)]


def test_iron_condor_credit_and_bounds():
    legs = st.iron_condor(80, 90, 110, 120)
    prem = st.net_premium(legs, S0, T, R, SIG)
    assert prem < 0  # opened for a credit
    mx, ml = st.max_profit_loss(legs, S0, T, R, SIG)
    assert mx == pytest.approx(-prem)
    assert ml == pytest.approx(-10 - prem)
    assert len(st.break_evens(legs, S0, T, R, SIG)) == 2


def test_unbounded_straddle():
    mx, ml = st.max_profit_loss(st.long_straddle(100), S0, T, R, SIG)
    assert mx == np.inf
    assert ml == pytest.approx(-st.net_premium(st.long_straddle(100), S0, T, R, SIG))


def test_put_call_parity_via_strategies():
    # long call + short put == forward
    legs = [st.Leg("call", 100, 1), st.Leg("put", 100, -1)]
    fwd = S0 - 100 * np.exp(-R * T)
    assert st.value(legs, S0, T, R, SIG) == pytest.approx(fwd)
