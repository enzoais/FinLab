"""Tests obligations : obligation au pair, aller-retour YTM, durations, convexité, monotonie."""
import numpy as np
import pytest

from services import fixed_income_service as fi

BOND = dict(face_value=100.0, coupon_rate=0.05, maturity_years=10.0, frequency=2)


def test_par_bond_prices_at_face_value():
    # Quand coupon == YTM, l'obligation cote au pair (= face value)
    price = fi.bond_price_dcf(**BOND, ytm=0.05)
    assert price == pytest.approx(100.0, abs=1e-6)


def test_price_below_par_when_ytm_above_coupon():
    price = fi.bond_price_dcf(**BOND, ytm=0.08)
    assert price < 100.0


def test_price_above_par_when_ytm_below_coupon():
    price = fi.bond_price_dcf(**BOND, ytm=0.03)
    assert price > 100.0


def test_price_is_monotonic_decreasing_in_ytm():
    ytms = [0.01, 0.03, 0.05, 0.07, 0.10]
    prices = [fi.bond_price_dcf(**BOND, ytm=y) for y in ytms]
    assert all(prices[i] > prices[i + 1] for i in range(len(prices) - 1))


def test_ytm_from_price_round_trip():
    price = fi.bond_price_dcf(**BOND, ytm=0.062)
    ytm = fi.ytm_from_price(**BOND, price=price)
    assert ytm is not None
    assert ytm == pytest.approx(0.062, abs=1e-6)


def test_modified_duration_less_than_macaulay():
    mac = fi.macaulay_duration(**BOND, ytm=0.05)
    mod = fi.modified_duration(**BOND, ytm=0.05)
    assert mac > 0
    assert 0 < mod < mac  # D_mod = D_mac / (1 + y/f) < D_mac


def test_convexity_is_positive():
    conv = fi.convexity(**BOND, ytm=0.05)
    assert conv > 0


def test_zero_coupon_macaulay_equals_maturity():
    # Une obligation zéro-coupon a une duration de Macaulay = maturité
    zc = dict(face_value=100.0, coupon_rate=0.0, maturity_years=5.0, frequency=1)
    mac = fi.macaulay_duration(**zc, ytm=0.04)
    assert mac == pytest.approx(5.0, abs=1e-9)


def test_run_bond_analysis_from_ytm():
    out = fi.run_bond_analysis(**BOND, ytm=0.05)
    assert out["success"] is True
    assert out["price"] == pytest.approx(100.0, abs=1e-2)
    assert out["modified_duration"] < out["macaulay_duration"]
    assert len(out["cash_flows"]) == 20  # 10 ans * 2 par an


def test_run_bond_analysis_rejects_both_ytm_and_price():
    out = fi.run_bond_analysis(**BOND, ytm=0.05, price=100.0)
    assert out["success"] is False


def test_run_bond_analysis_rejects_bad_frequency():
    out = fi.run_bond_analysis(
        face_value=100.0, coupon_rate=0.05, maturity_years=10.0, frequency=3, ytm=0.05,
    )
    assert out["success"] is False


def test_dv01_positive_and_matches_modified_duration():
    # DV01 (variation de prix pour +1 bp) ≈ duration modifiée × prix × 0,0001
    price = fi.bond_price_dcf(**BOND, ytm=0.05)
    mod = fi.modified_duration(**BOND, ytm=0.05)
    dv = fi.dv01(**BOND, ytm=0.05)
    assert dv > 0
    assert dv == pytest.approx(mod * price * 1e-4, rel=1e-2)


def test_cs01_equals_dv01_for_vanilla_bond():
    # Pour une obligation vanille à taux fixe, CS01 = DV01 (même décalage du taux d'actualisation)
    dv = fi.dv01(**BOND, ytm=0.05)
    cs = fi.cs01(**BOND, ytm=0.05)
    assert cs == pytest.approx(dv, abs=1e-12)


def test_dv01_increases_with_maturity():
    short = fi.dv01(face_value=100.0, coupon_rate=0.05, maturity_years=2.0, frequency=2, ytm=0.05)
    long = fi.dv01(face_value=100.0, coupon_rate=0.05, maturity_years=20.0, frequency=2, ytm=0.05)
    assert long > short  # plus la maturité est longue, plus la sensibilité au taux est grande


def test_run_bond_analysis_exposes_dv01_cs01():
    out = fi.run_bond_analysis(**BOND, ytm=0.05)
    assert out["success"] is True
    assert out["dv01"] > 0
    assert out["cs01"] == pytest.approx(out["dv01"], abs=1e-4)


def test_rate_scenarios_direction():
    base_price = fi.bond_price_dcf(**BOND, ytm=0.05)
    rows = fi.rate_scenarios(**BOND, base_ytm=0.05, base_price=base_price, shifts_bp=[-100, 100])
    up = next(r for r in rows if r["shift_bp"] == 100)
    down = next(r for r in rows if r["shift_bp"] == -100)
    assert up["delta_pct"] < 0    # hausse des taux => baisse du prix
    assert down["delta_pct"] > 0  # baisse des taux => hausse du prix
