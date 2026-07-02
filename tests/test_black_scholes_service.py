"""Tests Black-Scholes : valeurs analytiques connues, parité, Greeks, IV."""
import math

import numpy as np
import pytest

from services import black_scholes_service as bs

# Cas de référence classique : S=K=100, T=1, r=5%, sigma=20%
# Valeurs analytiques connues : call ≈ 10.4506, put ≈ 5.5735.
REF = dict(S=100.0, K=100.0, T=1.0, r=0.05, sigma=0.20)


def test_call_price_reference_value():
    price = bs.bsm_call_price(REF["S"], REF["K"], REF["T"], REF["r"], REF["sigma"])
    assert price == pytest.approx(10.4506, abs=1e-3)


def test_put_price_reference_value():
    price = bs.bsm_put_price(REF["S"], REF["K"], REF["T"], REF["r"], REF["sigma"])
    assert price == pytest.approx(5.5735, abs=1e-3)


def test_put_call_parity_holds():
    call = bs.bsm_call_price(**REF)
    put = bs.bsm_put_price(**REF)
    lhs, rhs, diff = bs.put_call_parity_check(
        REF["S"], REF["K"], REF["T"], REF["r"], call, put
    )
    assert diff == pytest.approx(0.0, abs=1e-9)


def test_greeks_relations():
    g = bs.bsm_greeks(**REF)
    # Delta call de référence = N(d1) ≈ 0.6368
    assert g["delta_call"] == pytest.approx(0.6368, abs=1e-3)
    # Relation put/call sur le delta
    assert g["delta_put"] == pytest.approx(g["delta_call"] - 1.0, abs=1e-12)
    # Gamma et Vega strictement positifs
    assert g["gamma"] > 0
    assert g["vega"] > 0
    # Theta d'un call (sans dividende) est négatif ici (décroissance temporelle)
    assert g["theta_call"] < 0


def test_degenerate_inputs_return_nan():
    assert math.isnan(bs.bsm_call_price(100, 100, 0.0, 0.05, 0.2))  # T=0
    assert math.isnan(bs.bsm_call_price(100, 100, 1.0, 0.05, 0.0))  # sigma=0
    assert math.isnan(bs.bsm_put_price(-1, 100, 1.0, 0.05, 0.2))    # S<=0


def test_implied_volatility_round_trip():
    sigma_true = 0.25
    price = bs.bsm_call_price(REF["S"], REF["K"], REF["T"], REF["r"], sigma_true)
    iv = bs.implied_volatility(price, "call", REF["S"], REF["K"], REF["T"], REF["r"])
    assert iv is not None
    assert iv == pytest.approx(sigma_true, abs=1e-4)


def test_implied_volatility_below_intrinsic_returns_none():
    # Prix de call sous la valeur intrinsèque => pas d'IV
    iv = bs.implied_volatility(0.01, "call", 150.0, 100.0, 1.0, 0.05)
    assert iv is None


def test_run_bs_analysis_iv_error_message():
    # Prix aberrant : le pipeline doit renvoyer success=False avec un message
    out = bs.run_bs_analysis(
        spot=150.0, strike=100.0, time_to_expiry=1.0,
        risk_free_rate=0.05, market_price=1.0, option_type_for_iv="call",
    )
    assert out["success"] is False
    assert out["error"] is not None


def test_run_bs_analysis_success():
    out = bs.run_bs_analysis(
        spot=100.0, strike=100.0, time_to_expiry=1.0,
        risk_free_rate=0.05, volatility=0.20,
    )
    assert out["success"] is True
    assert out["call_price"] == pytest.approx(10.4506, abs=1e-3)
    assert out["put_call_diff"] == pytest.approx(0.0, abs=1e-9)


def test_sensitivity_spot_up_increases_call():
    rows = bs.sensitivity_scenarios(
        100, 100, 1.0, 0.05, 0.2, "call", spot_shocks_pct=[5], vol_shocks_pct=[],
    )
    spot_row = next(r for r in rows if r["shock"].startswith("Spot"))
    assert spot_row["delta_pct"] > 0  # un call gagne quand le spot monte
