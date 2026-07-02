"""Tests backtest : courbe de capital, max drawdown connu, signe du Sharpe, poids, rééquilibrage."""
import numpy as np
import pandas as pd
import pytest

from services import backtest_service as bt


def _bdays(n: int) -> pd.DatetimeIndex:
    """Index de n jours ouvrés à partir de 2020-01-01."""
    return pd.bdate_range("2020-01-01", periods=n)


def test_equal_and_normalize_weights():
    assert bt.equal_weights(4).tolist() == [0.25, 0.25, 0.25, 0.25]
    assert bt.normalize_weights([2.0, 2.0]).tolist() == [0.5, 0.5]
    # Somme nulle → repli sur équipondéré
    assert bt.normalize_weights([0.0, 0.0]).tolist() == [0.5, 0.5]


def test_capital_curve_constant_returns():
    # Deux actifs identiques croissant à taux constant r → capital = C0 * (1+r)^(n-1)
    r = 0.001
    n = 252
    idx = _bdays(n)
    growth = 100 * (1 + r) ** np.arange(n)
    prices = pd.DataFrame({"A": growth, "B": growth}, index=idx)
    res = bt.compute_backtest(prices, weights=None, initial_capital=1000.0, rebalance="none", risk_free_rate=0.0)
    assert res["success"] is True
    curve = res["capital_curve"]
    assert curve.iloc[0] == pytest.approx(1000.0, abs=1e-9)
    assert curve.iloc[-1] == pytest.approx(1000.0 * (1 + r) ** (n - 1), rel=1e-9)
    # Rendement total cohérent
    assert res["metrics"]["total_return"] == pytest.approx((1 + r) ** (n - 1) - 1, rel=1e-9)


def test_max_drawdown_known_value():
    # Actif unique : 100 -> 120 -> 60 -> 90. Drawdown pic(120)->creux(60) = -50 %.
    idx = _bdays(4)
    prices = pd.DataFrame({"A": [100.0, 120.0, 60.0, 90.0]}, index=idx)
    res = bt.compute_backtest(prices, weights=np.array([1.0]), initial_capital=1000.0, rebalance="none", risk_free_rate=0.0)
    assert res["success"] is True
    assert res["metrics"]["max_drawdown"] == pytest.approx(-0.5, abs=1e-9)


def test_sharpe_sign_follows_drift():
    n = 250
    idx = _bdays(n)
    rng = np.random.default_rng(0)
    noise = rng.standard_normal(n - 1)
    # Drift positif
    up = np.concatenate([[100.0], 100 * np.cumprod(1 + 0.001 + 0.01 * noise)])
    prices_up = pd.DataFrame({"A": up}, index=idx)
    sharpe_up = bt.compute_backtest(prices_up, np.array([1.0]), 1000.0, "none", 0.0)["metrics"]["sharpe_ratio"]
    assert sharpe_up > 0
    # Drift négatif (même bruit)
    down = np.concatenate([[100.0], 100 * np.cumprod(1 - 0.001 + 0.01 * noise)])
    prices_down = pd.DataFrame({"A": down}, index=idx)
    sharpe_down = bt.compute_backtest(prices_down, np.array([1.0]), 1000.0, "none", 0.0)["metrics"]["sharpe_ratio"]
    assert sharpe_down < 0


def test_equal_vs_weighted_allocation():
    # A monte fort, B reste plat. Surpondérer A doit donner un capital final plus élevé.
    n = 120
    idx = _bdays(n)
    a = 100 * (1.002) ** np.arange(n)
    b = np.full(n, 100.0)
    prices = pd.DataFrame({"A": a, "B": b}, index=idx)
    eq = bt.compute_backtest(prices, None, 1000.0, "none", 0.0)["capital_curve"]
    weighted = bt.compute_backtest(prices, np.array([0.9, 0.1]), 1000.0, "none", 0.0)["capital_curve"]
    assert weighted.iloc[-1] > eq.iloc[-1]
    assert weighted.iloc[-1] != pytest.approx(eq.iloc[-1])


def test_rebalancing_identical_assets_is_neutral():
    # Rééquilibrer entre deux actifs identiques ne change rien à la courbe de capital.
    n = 300  # couvre plusieurs mois → le rééquilibrage se déclenche réellement
    idx = _bdays(n)
    rng = np.random.default_rng(1)
    path = 100 * np.cumprod(1 + 0.01 * rng.standard_normal(n))
    prices = pd.DataFrame({"A": path, "B": path.copy()}, index=idx)
    bh = bt.compute_backtest(prices, None, 1000.0, "none", 0.0)["capital_curve"]
    monthly = bt.compute_backtest(prices, None, 1000.0, "monthly", 0.0)["capital_curve"]
    assert np.allclose(bh.values, monthly.values)


def test_rebalancing_differs_when_assets_diverge():
    # Quand les actifs divergent, buy & hold et rééquilibrage mensuel donnent des courbes différentes.
    n = 300
    idx = _bdays(n)
    a = 100 * (1.001) ** np.arange(n)
    b = 100 * (0.999) ** np.arange(n)
    prices = pd.DataFrame({"A": a, "B": b}, index=idx)
    bh = bt.compute_backtest(prices, None, 1000.0, "none", 0.0)["capital_curve"]
    monthly = bt.compute_backtest(prices, None, 1000.0, "monthly", 0.0)["capital_curve"]
    assert not np.allclose(bh.values, monthly.values)


def test_benchmark_identical_gives_zero_alpha():
    # Portefeuille mono-actif comparé au même actif : alpha ≈ 0, tracking error ≈ 0.
    n = 150
    idx = _bdays(n)
    rng = np.random.default_rng(2)
    path = 100 * np.cumprod(1 + 0.008 * rng.standard_normal(n))
    prices = pd.DataFrame({"A": path}, index=idx)
    bench = pd.Series(path, index=idx)
    res = bt.compute_backtest(prices, np.array([1.0]), 1000.0, "none", 0.0, benchmark_prices=bench)
    bm = res["benchmark_metrics"]
    assert bm is not None
    assert abs(bm["benchmark_alpha"]) < 1e-9
    assert bm["tracking_error"] < 1e-9
    assert "information_ratio" in bm


def test_invalid_rebalance_mode():
    idx = _bdays(3)
    prices = pd.DataFrame({"A": [100.0, 101.0, 102.0]}, index=idx)
    res = bt.compute_backtest(prices, None, 1000.0, "weekly", 0.0)
    assert res["success"] is False


def test_weights_length_mismatch():
    idx = _bdays(3)
    prices = pd.DataFrame({"A": [100.0, 101.0, 102.0], "B": [100.0, 100.0, 100.0]}, index=idx)
    res = bt.compute_backtest(prices, np.array([1.0]), 1000.0, "none", 0.0)
    assert res["success"] is False
