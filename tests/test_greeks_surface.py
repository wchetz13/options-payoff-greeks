import numpy as np
import pytest

from src.black_scholes_greeks import bs_call_price, bs_put_price
from src.greeks_surface import available_greeks, greek_grid, vanna, volga

S, K, T, R, SIG = 100.0, 105.0, 0.75, 0.04, 0.25


def fd(f, x, h):
    return (f(x + h) - f(x - h)) / (2 * h)


def test_vanna_matches_cross_difference():
    from src.black_scholes_greeks import delta_call

    est = fd(lambda s: delta_call(S, K, T, R, s), SIG, 1e-5)
    assert vanna(S, K, T, R, SIG) == pytest.approx(est, rel=1e-6)


def test_volga_matches_second_difference():
    h = 1e-3
    est = (
        bs_call_price(S, K, T, R, SIG + h) - 2 * bs_call_price(S, K, T, R, SIG) + bs_call_price(S, K, T, R, SIG - h)
    ) / h**2
    assert volga(S, K, T, R, SIG) == pytest.approx(est, rel=1e-4)


def test_put_theta_matches_finite_difference():
    from src.greeks_surface import theta_put

    # theta = -dV/dT
    est = -fd(lambda t: bs_put_price(S, K, t, R, SIG), T, 1e-5)
    assert theta_put(S, K, T, R, SIG) == pytest.approx(est, rel=1e-6)


def test_grid_shape_and_orientation():
    xs, ts = np.linspace(80, 120, 7), np.array([0.25, 1.0])
    g = greek_grid("delta", xs, ts)
    assert g.shape == (2, 7)
    # Call delta rises with spot along each row.
    assert np.all(np.diff(g, axis=1) > 0)


def test_grid_matches_scalar_evaluation():
    from src.black_scholes_greeks import gamma

    g = greek_grid("gamma", np.array([90.0, 110.0]), np.array([0.5]), K=100.0, r=0.03, sigma=0.3)
    assert g[0, 1] == pytest.approx(gamma(110.0, 100.0, 0.5, 0.03, 0.3))


def test_strike_axis_fixes_spot():
    g = greek_grid("delta", np.array([80.0, 120.0]), np.array([1.0]), axis="strike", S=100.0)
    assert g[0, 0] > g[0, 1]  # call delta falls with strike


def test_call_put_delta_parity_on_grid():
    xs, ts = np.linspace(70, 130, 13), np.linspace(0.1, 2, 5)
    diff = greek_grid("delta", xs, ts) - greek_grid("delta", xs, ts, option_type="put")
    assert np.allclose(diff, 1.0)


def test_gamma_vega_same_for_calls_and_puts():
    xs, ts = np.linspace(70, 130, 9), np.array([0.3, 1.2])
    for g in ("gamma", "vega", "vanna", "volga"):
        assert np.allclose(greek_grid(g, xs, ts), greek_grid(g, xs, ts, option_type="put"))


def test_vanna_zero_where_d2_zero_and_sign_flips():
    # vanna ∝ -d2: positive deep OTM calls (d2<0), negative deep ITM.
    g = greek_grid("vanna", np.array([60.0, 150.0]), np.array([0.5]))
    assert g[0, 0] > 0 > g[0, 1]


def test_gamma_peaks_near_atm_and_decays_with_maturity_away_from_atm():
    xs = np.linspace(60, 140, 161)
    g = greek_grid("gamma", xs, np.array([0.25]))
    assert abs(xs[np.argmax(g[0])] - 100) < 5


@pytest.mark.parametrize("bad", [{"greek": "nope"}, {"option_type": "swap"}, {"axis": "z"}])
def test_invalid_arguments(bad):
    kw = {"greek": "delta", **bad}
    with pytest.raises(ValueError):
        greek_grid(kw.pop("greek"), np.array([100.0]), np.array([1.0]), **kw)


def test_nonpositive_inputs_rejected():
    with pytest.raises(ValueError):
        greek_grid("delta", np.array([100.0]), np.array([0.0]))


def test_available_greeks_put_excludes_rho():
    assert "rho" in available_greeks("call") and "rho" not in available_greeks("put")
