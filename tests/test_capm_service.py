"""Tests CAPM : log-returns, régression OLS synthétique, Kₑ, beta ajusté, détection ticker."""
import numpy as np
import pandas as pd
import pytest

from services import capm_service as capm


def test_log_returns_basic():
    prices = pd.Series([100.0, 110.0, 121.0])
    ret = capm.log_returns(prices)
    assert len(ret) == 2
    assert ret.iloc[0] == pytest.approx(np.log(1.1), abs=1e-12)


def test_ols_regression_recovers_beta():
    # Données synthétiques : rendements actif = 2 × rendements marché => beta = 2
    rng = np.random.default_rng(42)
    idx = pd.date_range("2020-01-01", periods=300, freq="D")
    market = pd.Series(rng.normal(0.0005, 0.01, 300), index=idx)
    asset = 2.0 * market
    reg = capm.ols_regression(asset, market, risk_free_rate=0.05)
    assert reg["success"] is True
    assert reg["beta"] == pytest.approx(2.0, abs=1e-6)
    assert reg["r_squared"] == pytest.approx(1.0, abs=1e-9)


def test_ols_regression_insufficient_data():
    idx = pd.date_range("2020-01-01", periods=5, freq="D")
    s = pd.Series(np.zeros(5), index=idx)
    reg = capm.ols_regression(s, s, risk_free_rate=0.05)
    assert reg["success"] is False


def test_cost_of_equity_formula():
    # Kₑ = Rf + β (E[Rm] − Rf) = 0.03 + 1.5*(0.10-0.03) = 0.135
    ke = capm.cost_of_equity(0.03, 1.5, 0.10)
    assert ke == pytest.approx(0.135, abs=1e-12)


def test_adjusted_beta_blume():
    # (2/3)β + 1/3 : β=1 => 1 ; β=2 => 5/3
    assert capm.adjusted_beta(1.0) == pytest.approx(1.0, abs=1e-12)
    assert capm.adjusted_beta(2.0) == pytest.approx(5.0 / 3.0, abs=1e-12)


@pytest.mark.parametrize("s,expected", [
    ("AAPL", True),
    ("BRK.B", True),
    ("Apple Inc", False),
    ("", False),
    ("TOOLONGTICKER", False),
])
def test_looks_like_ticker(s, expected):
    assert capm._looks_like_ticker(s) is expected


def test_rolling_beta_length():
    rng = np.random.default_rng(0)
    idx = pd.date_range("2020-01-01", periods=200, freq="D")
    market = pd.Series(rng.normal(0, 0.01, 200), index=idx)
    asset = 1.3 * market + pd.Series(rng.normal(0, 0.001, 200), index=idx)
    roll = capm.rolling_beta(asset, market, risk_free_rate=0.02, window=60)
    # De window..N inclus => N - window + 1 points
    assert len(roll) == 200 - 60 + 1
    assert roll.iloc[-1] == pytest.approx(1.3, abs=0.15)
