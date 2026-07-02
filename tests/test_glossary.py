"""Tests d'intégrité du glossaire : entrées non vides et clés attendues présentes."""
import pytest

from utils.glossary import GLOSSARY, get_glossary_text

# Clés que l'app référence explicitement (une par notion clé de chaque onglet).
EXPECTED_KEYS = [
    # Général
    "benchmark", "risk_free_rate", "log_returns",
    # CAPM / Beta
    "capm", "beta", "cost_of_equity", "jensen_alpha", "r_squared", "rolling_beta",
    # Portefeuille
    "markowitz", "efficient_frontier", "min_variance", "max_sharpe", "sharpe_ratio",
    "portfolio_volatility", "expected_return", "diversification_ratio", "optimal_weights",
    "covariance", "correlation", "var", "cvar", "tracking_error", "information_ratio", "alpha",
    # Risque (fonds / mandat)
    "parametric_var", "stress_test", "concentration", "risk_contribution", "var_backtesting",
    # Obligations
    "bond_price", "ytm", "macaulay_duration", "modified_duration", "convexity",
    "coupon_rate", "spread_vs_rf", "accrued_interest", "clean_dirty_price", "price_yield_curve",
    "dv01", "cs01",
    # Options
    "black_scholes", "call", "put", "implied_volatility", "delta", "gamma", "theta", "vega",
    "rho", "spot", "strike", "time_to_expiry", "volatility", "intrinsic_value", "payoff",
    # Monte Carlo
    "monte_carlo", "gbm", "drift", "terminal_wealth", "fan_chart", "shortfall_probability",
    "max_drawdown",
    # Backtest
    "backtest", "capital_curve", "total_return", "cagr", "annualized_volatility",
    "var_historical", "cvar_historical", "rebalancing", "buy_and_hold", "equal_weight",
]


def test_all_entries_are_non_empty_strings():
    for key, text in GLOSSARY.items():
        assert isinstance(text, str), f"{key} n'est pas une chaîne"
        assert len(text.strip()) > 20, f"{key} est trop court"


@pytest.mark.parametrize("key", EXPECTED_KEYS)
def test_expected_key_present(key):
    assert key in GLOSSARY, f"Clé manquante dans le glossaire : {key}"


def test_entries_cover_the_three_pedagogical_parts():
    # Chaque entrée suit la grammaire : c'est quoi / à quoi ça sert / comment ça se calcule.
    for key, text in GLOSSARY.items():
        assert "C'est quoi" in text, f"{key} : volet « c'est quoi » manquant"
        assert "À quoi ça sert" in text, f"{key} : volet « à quoi ça sert » manquant"
        assert "Comment ça se calcule" in text, f"{key} : volet « comment ça se calcule » manquant"


def test_get_glossary_text_fallback():
    assert "non disponible" in get_glossary_text("cle_inexistante").lower()
    assert get_glossary_text("beta") == GLOSSARY["beta"]
