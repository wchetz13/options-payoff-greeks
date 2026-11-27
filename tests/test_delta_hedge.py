import numpy as np
import pytest

from src.delta_hedge import expected_vol_mismatch_pnl, simulate_delta_hedge

PARAMS = dict(S0=100.0, K=100.0, T=0.5, r=0.04)


def test_hedge_error_vanishes_with_frequent_rebalancing():
    res = simulate_delta_hedge(**PARAMS, sigma_real=0.25, n_steps=500, n_paths=5_000)
    assert abs(res.mean) < 4 * res.stderr + 1e-3
    assert res.std < 0.5


def test_error_std_scales_like_inverse_sqrt_steps():
    steps = np.array([10, 40, 160, 640])
    stds = [simulate_delta_hedge(**PARAMS, sigma_real=0.25, n_steps=int(n), n_paths=8_000).std
            for n in steps]
    slope = np.polyfit(np.log(steps), np.log(stds), 1)[0]
    assert slope == pytest.approx(-0.5, abs=0.07)


@pytest.mark.parametrize("option_type", ["call", "put"])
@pytest.mark.parametrize("sigma_hedge", [0.15, 0.35])
def test_vol_mismatch_mean_matches_analytic(option_type, sigma_hedge):
    kw = dict(PARAMS, sigma_real=0.25, sigma_hedge=sigma_hedge, option_type=option_type)
    res = simulate_delta_hedge(**kw, n_steps=400, n_paths=20_000)
    # Gamma is identical for calls and puts, so the analytic value is type-independent.
    expected = expected_vol_mismatch_pnl(**PARAMS, sigma_real=0.25, sigma_hedge=sigma_hedge)
    assert res.mean == pytest.approx(expected, abs=4 * res.stderr + 0.02)


def test_selling_rich_vol_is_profitable_on_average():
    # Sold at 35% implied, realised 25%: positive expected P&L; reverse is negative.
    assert expected_vol_mismatch_pnl(**PARAMS, sigma_real=0.25, sigma_hedge=0.35) > 0
    assert expected_vol_mismatch_pnl(**PARAMS, sigma_real=0.25, sigma_hedge=0.15) < 0
    assert expected_vol_mismatch_pnl(**PARAMS, sigma_real=0.25, sigma_hedge=0.25) == pytest.approx(0.0)


def test_reproducible_with_seed():
    a = simulate_delta_hedge(**PARAMS, sigma_real=0.2, n_steps=20, n_paths=100, seed=7)
    b = simulate_delta_hedge(**PARAMS, sigma_real=0.2, n_steps=20, n_paths=100, seed=7)
    np.testing.assert_array_equal(a.pnl, b.pnl)


def test_input_validation():
    with pytest.raises(ValueError):
        simulate_delta_hedge(**PARAMS, sigma_real=-0.1)
    with pytest.raises(ValueError):
        simulate_delta_hedge(**PARAMS, sigma_real=0.2, option_type="straddle")
    with pytest.raises(ValueError):
        simulate_delta_hedge(**PARAMS, sigma_real=0.2, n_steps=0)
    with pytest.raises(ValueError):
        simulate_delta_hedge(**PARAMS, sigma_real=0.2, sigma_hedge=0.0)
