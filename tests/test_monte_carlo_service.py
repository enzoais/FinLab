"""Tests Monte Carlo : GBM déterministe, reproductibilité, espérance E[S_T]=S0·exp(mu·T),
VaR/CVaR simulés, shortfall, max drawdown, cohérence du drift arithmétique."""
import numpy as np
import pytest

from services import monte_carlo_service as mc


def test_paths_shape_and_initial_value():
    paths, t = mc.simulate_gbm_paths(
        S0=100.0, mu=0.07, sigma=0.2, T_years=2.0,
        n_paths=50, n_steps_per_year=12, seed=1,
    )
    assert paths.shape == (50, 2 * 12 + 1)
    assert np.all(paths[:, 0] == 100.0)
    assert t[0] == 0.0 and t[-1] == pytest.approx(2.0)


def test_deterministic_when_sigma_zero():
    # sigma=0 => S_T = S0 * exp(mu * T) sur toutes les trajectoires
    paths, _ = mc.simulate_gbm_paths(
        S0=100.0, mu=0.05, sigma=0.0, T_years=3.0,
        n_paths=10, n_steps_per_year=252, seed=7,
    )
    expected = 100.0 * np.exp(0.05 * 3.0)
    assert np.allclose(paths[:, -1], expected, atol=1e-6)


def test_reproducible_with_seed():
    a, _ = mc.simulate_gbm_paths(100, 0.07, 0.2, 1.0, 100, 252, seed=123)
    b, _ = mc.simulate_gbm_paths(100, 0.07, 0.2, 1.0, 100, 252, seed=123)
    assert np.array_equal(a, b)


def test_expected_terminal_matches_arithmetic_drift():
    # Sous MBG, E[S_T] = S0 * exp(mu * T) où mu est le drift *arithmétique*.
    # Ce test garantit la convention utilisée par le simulateur.
    S0, mu, sigma, T = 100.0, 0.08, 0.2, 5.0
    paths, _ = mc.simulate_gbm_paths(S0, mu, sigma, T, n_paths=200_000, n_steps_per_year=52, seed=2024)
    expected = S0 * np.exp(mu * T)
    assert np.mean(paths[:, -1]) == pytest.approx(expected, rel=0.02)


def test_annual_withdrawal_reduces_wealth():
    base, _ = mc.simulate_gbm_paths(100_000, 0.05, 0.1, 5.0, 500, 12, seed=3, annual_flow=0.0)
    with_wd, _ = mc.simulate_gbm_paths(100_000, 0.05, 0.1, 5.0, 500, 12, seed=3, annual_flow=5_000.0)
    assert np.mean(with_wd[:, -1]) < np.mean(base[:, -1])


def test_terminal_wealth_stats_keys_and_order():
    paths, _ = mc.simulate_gbm_paths(100, 0.07, 0.2, 1.0, 1000, 52, seed=5)
    stats = mc.terminal_wealth_stats(paths)
    for k in ("mean", "std", "p5", "p25", "p50", "p75", "p95", "min", "max"):
        assert k in stats
    assert stats["p5"] <= stats["p50"] <= stats["p95"]
    assert stats["min"] <= stats["p5"]


def test_var_cvar_simulated_percentages():
    paths, _ = mc.simulate_gbm_paths(100, 0.0, 0.3, 1.0, 5000, 52, seed=9)
    vc = mc.var_cvar_simulated(paths[:, -1], S0=100.0, confidence=0.95)
    assert vc["cvar_level"] <= vc["var_level"]
    assert vc["cvar_pct"] <= vc["var_pct"]


def test_probability_shortfall_bounds():
    terminal = np.array([50.0, 80.0, 100.0, 120.0, 150.0])
    p = mc.probability_shortfall(terminal, threshold=100.0)
    assert p == pytest.approx(2 / 5)  # 50 et 80 sont sous 100


def test_max_drawdown_monotonic_path_is_zero():
    # Trajectoire strictement croissante => drawdown nul
    paths = np.array([[1.0, 2.0, 3.0, 4.0]])
    dd = mc.max_drawdown_per_path(paths)
    assert dd[0] == pytest.approx(0.0)


def test_max_drawdown_detects_decline():
    paths = np.array([[100.0, 120.0, 60.0, 90.0]])  # pic 120, creux 60 => -50%
    dd = mc.max_drawdown_per_path(paths)
    assert dd[0] == pytest.approx(-0.5, abs=1e-9)


def test_run_monte_carlo_analysis_pipeline():
    out = mc.run_monte_carlo_analysis(
        S0=100_000, mu=0.07, sigma=0.18, T_years=10.0,
        n_paths=2000, n_steps_per_year=52, seed=11, shortfall_threshold=100_000,
    )
    assert out["success"] is True
    assert 0.0 <= out["prob_shortfall"] <= 1.0
    assert "var_cvar" in out and "drawdown_stats" in out
