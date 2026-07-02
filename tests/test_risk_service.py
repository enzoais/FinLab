"""Tests risque : VaR (3 méthodes), concentration, contribution, stress, Kupiec, bêta."""
import numpy as np
import pandas as pd
import pytest
from scipy.stats import norm

from services import risk_service as rk


def _bdays(n):
    return pd.bdate_range("2020-01-01", periods=n)


def test_var_parametric_matches_normal_formula():
    rng = np.random.default_rng(0)
    r = 0.0002 + 0.01 * rng.standard_normal(5000)
    var, cvar = rk.var_parametric(r, confidence=0.95)
    mu, sigma = float(np.mean(r)), float(np.std(r))
    assert var == pytest.approx(mu + norm.ppf(0.05) * sigma, rel=1e-9)
    assert cvar < var  # CVaR (queue) plus sévère que la VaR


def test_var_methods_are_close_on_normal_data():
    rng = np.random.default_rng(1)
    r = 0.01 * rng.standard_normal(20000)
    vh, _ = rk.var_historical(r, 0.95)
    vp, _ = rk.var_parametric(r, 0.95)
    vm, _ = rk.var_monte_carlo(r, 0.95, seed=7)
    # Sur des données normales, les 3 méthodes doivent être proches
    assert vh == pytest.approx(vp, abs=0.001)
    assert vm == pytest.approx(vp, abs=0.001)
    assert vh < 0 and vp < 0 and vm < 0


def test_scale_to_horizon_root_time():
    assert rk.scale_to_horizon(-0.02, 10) == pytest.approx(-0.02 * np.sqrt(10), rel=1e-12)


def test_concentration_equal_weights():
    c = rk.concentration_metrics(np.array([0.25, 0.25, 0.25, 0.25]))
    assert c["hhi"] == pytest.approx(0.25, abs=1e-9)
    assert c["effective_n"] == pytest.approx(4.0, abs=1e-9)
    assert c["top3"] == pytest.approx(0.75, abs=1e-9)


def test_concentration_concentrated_portfolio():
    c = rk.concentration_metrics(np.array([0.9, 0.1]))
    assert c["hhi"] == pytest.approx(0.82, abs=1e-9)
    assert c["effective_n"] < 1.3  # très concentré → peu de positions effectives


def test_component_risk_sums_to_100():
    weights = np.array([0.5, 0.3, 0.2])
    cov = np.diag([0.04, 0.09, 0.16])
    contrib = rk.component_risk_pct(weights, cov)
    assert contrib.sum() == pytest.approx(100.0, abs=1e-6)
    assert np.all(contrib >= 0)


def test_stress_scenarios_scale_with_beta_and_aum():
    rows_b1 = rk.stress_scenarios(beta=1.0, aum=1_000_000, scenarios=[("krach", -0.40)])
    assert rows_b1[0]["pnl_eur"] == pytest.approx(-400_000.0, abs=1e-6)
    rows_b2 = rk.stress_scenarios(beta=2.0, aum=1_000_000, scenarios=[("krach", -0.40)])
    assert rows_b2[0]["pnl_eur"] == pytest.approx(-800_000.0, abs=1e-6)  # bêta 2 → double la perte


def test_portfolio_beta_of_scaled_series():
    rng = np.random.default_rng(2)
    bench = 0.01 * rng.standard_normal(500)
    port = 1.8 * bench  # portefeuille = 1,8× le marché
    assert rk.portfolio_beta(port, bench) == pytest.approx(1.8, rel=1e-6)


def test_kupiec_pass_when_exceptions_match_expected():
    # 100 jours, 95 % de confiance → 5 dépassements attendus ; on en met exactement 5
    returns = np.concatenate([np.zeros(95), np.full(5, -0.10)])
    res = rk.kupiec_test(returns, var_level=-0.05, confidence=0.95)
    assert res["observed"] == 5
    assert res["expected"] == pytest.approx(5.0)
    assert res["passed"] is True


def test_kupiec_fails_when_too_many_exceptions():
    # 25 dépassements pour 5 attendus → modèle rejeté
    returns = np.concatenate([np.zeros(75), np.full(25, -0.10)])
    res = rk.kupiec_test(returns, var_level=-0.05, confidence=0.95)
    assert res["observed"] == 25
    assert res["passed"] is False


def test_historical_worst_is_min_return():
    r = np.array([0.01, -0.03, 0.02, -0.08, 0.01])
    hw = rk.historical_worst(r, aum=1000.0)
    assert hw["worst_1d_pct"] == pytest.approx(-0.08)
    assert hw["worst_1d_eur"] == pytest.approx(-80.0)


def test_compute_risk_end_to_end():
    n = 300
    idx = _bdays(n)
    rng = np.random.default_rng(3)
    a = 100 * np.cumprod(1 + 0.01 * rng.standard_normal(n))
    b = 100 * np.cumprod(1 + 0.012 * rng.standard_normal(n))
    prices = pd.DataFrame({"A": a, "B": b}, index=idx)
    bench = pd.Series(100 * np.cumprod(1 + 0.009 * rng.standard_normal(n)), index=idx)
    out = rk.compute_risk(prices, weights=None, aum=1_000_000, confidence=0.95, horizon_days=1, benchmark_prices=bench)
    assert out["success"] is True
    assert out["var_methods"]["historique"]["var"] < 0
    assert out["var_methods"]["parametrique"]["var"] < 0
    assert len(out["stress_scenarios"]) == 4
    assert out["risk_contributions_pct"].sum() == pytest.approx(100.0, abs=1e-6)
    assert "passed" in out["kupiec"]
