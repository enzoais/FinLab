"""Tests Markowitz : min variance analytique, frontière, VaR/CVaR, décomposition du risque, PSD."""
import numpy as np
import pandas as pd
import pytest

from services import markowitz_service as mk


def test_min_variance_two_uncorrelated_assets():
    # Deux actifs non corrélés, variances 0.04 et 0.16.
    # Poids min-variance analytiques : w1 = v2/(v1+v2) = 0.16/0.20 = 0.8
    mu = np.array([0.10, 0.10])
    cov = np.array([[0.04, 0.0], [0.0, 0.16]])
    res = mk.optimize_portfolio(mu, cov, risk_free_rate=0.02, objective="min_variance", target_return=None)
    assert res["success"] is True
    assert res["weights"][0] == pytest.approx(0.8, abs=1e-3)
    assert res["weights"][1] == pytest.approx(0.2, abs=1e-3)
    assert res["weights"].sum() == pytest.approx(1.0, abs=1e-9)


def test_max_sharpe_weights_sum_to_one_and_bounded():
    mu = np.array([0.12, 0.08, 0.15])
    cov = np.array([
        [0.040, 0.006, 0.010],
        [0.006, 0.030, 0.008],
        [0.010, 0.008, 0.050],
    ])
    res = mk.optimize_portfolio(mu, cov, 0.02, "max_sharpe", None, max_weight_per_asset=0.6)
    assert res["success"] is True
    assert res["weights"].sum() == pytest.approx(1.0, abs=1e-6)
    assert np.all(res["weights"] >= -1e-6)
    assert np.all(res["weights"] <= 0.6 + 1e-6)


def test_efficient_frontier_returns_points():
    mu = np.array([0.10, 0.14])
    cov = np.array([[0.04, 0.01], [0.01, 0.09]])
    curve, weights = mk.efficient_frontier(mu, cov, 0.02, n_points=20)
    assert curve.shape[1] == 2
    assert len(curve) > 0
    # Rendements de la frontière croissants avec la vol (globalement)
    assert curve[-1][1] >= curve[0][1]


def test_risk_contributions_sum_to_portfolio_vol():
    weights = np.array([0.8, 0.2])
    cov = np.array([[0.04, 0.0], [0.0, 0.16]])
    contrib = mk.risk_decomposition(weights, cov)
    port_vol = np.sqrt(weights @ cov @ weights)
    assert contrib.sum() == pytest.approx(port_vol, abs=1e-12)


def test_ensure_positive_semidefinite_lifts_eigenvalues():
    # Matrice non-PSD (valeur propre négative)
    bad = np.array([[1.0, 2.0], [2.0, 1.0]])  # valeurs propres 3 et -1
    fixed = mk._ensure_positive_semidefinite(bad, ridge=1e-8)
    min_eig = np.min(np.linalg.eigvalsh(fixed))
    assert min_eig >= 1e-8 - 1e-12


def test_compute_var_cvar_ordering():
    returns = np.array([-0.05, -0.03, -0.02, -0.01, 0.0, 0.01, 0.02, 0.03, 0.04, 0.05])
    var, cvar = mk.compute_var_cvar(returns, confidence=0.90)
    # CVaR (moyenne de la queue) est au plus égal au VaR (quantile)
    assert cvar <= var
    assert var <= 0


def test_compute_var_cvar_empty():
    var, cvar = mk.compute_var_cvar(np.array([]), confidence=0.95)
    assert var == 0.0 and cvar == 0.0


def test_target_return_objective_hits_target():
    mu = np.array([0.10, 0.14])
    cov = np.array([[0.04, 0.01], [0.01, 0.09]])
    res = mk.optimize_portfolio(mu, cov, 0.02, "target_return", target_return=0.12)
    assert res["success"] is True
    assert res["expected_return"] == pytest.approx(0.12, abs=1e-4)


def test_diversification_ratio_at_least_one():
    weights = np.array([0.5, 0.5])
    cov = np.array([[0.04, 0.0], [0.0, 0.04]])  # non corrélés => diversification > 1
    dr = mk.diversification_ratio(weights, cov)
    assert dr > 1.0
