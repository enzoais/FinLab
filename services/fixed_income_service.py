"""
Fixed Income & Bond Pricing - Business logic.
DCF price, YTM (Newton/scipy), Macaulay/Modified duration, Convexity, Price-Yield curve.
Accrued interest (30/360), Dirty/Clean price, rate scenarios, bond comparison.
"""
import numpy as np
from scipy.optimize import brentq

try:
    import pandas as pd
except ImportError:
    pd = None


def _days_30_360(d1: "pd.Timestamp", d2: "pd.Timestamp") -> int:
    """
    30/360 day count: days between d1 and d2.
    Simple convention: day = min(day, 30), then 360*(y2-y1) + 30*(m2-m1) + (d2-d1).
    Returns non-negative value (0 if d2 <= d1).
    """
    if pd is None:
        return max(0, int((d2 - d1).days))
    day1, day2 = min(d1.day, 30), min(d2.day, 30)
    raw = 360 * (d2.year - d1.year) + 30 * (d2.month - d1.month) + (day2 - day1)
    return max(0, raw)


def bond_price_dcf(
    face_value: float,
    coupon_rate: float,
    maturity_years: float,
    frequency: int,
    ytm: float,
) -> float:
    """
    Bond price by DCF (discounted cash flows).
    Coupon per period = face_value * (coupon_rate / frequency).
    n = number of periods = round(maturity_years * frequency).
    Price = sum of PV(coupons) + PV(principal).
    """
    if frequency < 1 or maturity_years <= 0 or face_value <= 0:
        return np.nan
    n = int(round(maturity_years * frequency))
    if n <= 0:
        return np.nan
    r = ytm / frequency
    if r <= -1:
        return np.nan
    coupon_per_period = face_value * (coupon_rate / frequency)
    # PV of coupons: annuity (1 - (1+r)^{-n}) / r
    if abs(r) < 1e-10:
        pv_coupons = coupon_per_period * n
    else:
        pv_coupons = coupon_per_period * (1 - (1 + r) ** (-n)) / r
    pv_principal = face_value / (1 + r) ** n
    return pv_coupons + pv_principal


def ytm_from_price(
    face_value: float,
    coupon_rate: float,
    maturity_years: float,
    frequency: int,
    price: float,
    ytm_guess: float = 0.05,
) -> float | None:
    """
    Solve for YTM (annual decimal) such that bond_price_dcf(..., ytm) = price.
    Uses scipy.optimize.brentq. Returns None if no solution in (1e-6, 1.0).
    """
    if price <= 0 or face_value <= 0 or maturity_years <= 0 or frequency < 1:
        return None
    n = int(round(maturity_years * frequency))
    if n <= 0:
        return None

    def objective(ytm: float) -> float:
        return bond_price_dcf(face_value, coupon_rate, maturity_years, frequency, ytm) - price

    # YTM must be > -1/frequency for (1+y/f) > 0
    low = -0.49  # safe for f=1,2,4 so 1+y/f > 0
    high = 2.0
    try:
        if objective(low) * objective(high) > 0:
            return None
        ytm = brentq(objective, low, high, xtol=1e-10)
        return float(ytm)
    except (ValueError, ZeroDivisionError):
        return None


def macaulay_duration(
    face_value: float,
    coupon_rate: float,
    maturity_years: float,
    frequency: int,
    ytm: float,
    price: float | None = None,
) -> float:
    """
    Macaulay duration in years.
    D_Mac = (1/P) * sum (t/f) * PV(CF_t), t in periods, so result in years.
    """
    if price is None:
        price = bond_price_dcf(face_value, coupon_rate, maturity_years, frequency, ytm)
    if price <= 0 or np.isnan(price):
        return np.nan
    n = int(round(maturity_years * frequency))
    if n <= 0:
        return np.nan
    r = ytm / frequency
    if r <= -1:
        return np.nan
    coupon_per_period = face_value * (coupon_rate / frequency)
    weighted_sum = 0.0
    for t in range(1, n + 1):
        if t < n:
            cf = coupon_per_period
        else:
            cf = coupon_per_period + face_value
        pv = cf / (1 + r) ** t
        weighted_sum += (t / frequency) * pv
    return weighted_sum / price


def modified_duration(
    face_value: float,
    coupon_rate: float,
    maturity_years: float,
    frequency: int,
    ytm: float,
    price: float | None = None,
) -> float:
    """
    Modified duration (sensitivity to annual yield): D_Mod = D_Mac / (1 + y/f).
    Approximate % price change for 1% yield change: -D_Mod * 0.01.
    """
    d_mac = macaulay_duration(face_value, coupon_rate, maturity_years, frequency, ytm, price)
    if np.isnan(d_mac):
        return np.nan
    r = ytm / frequency
    if r <= -1:
        return np.nan
    return d_mac / (1 + r)


def convexity(
    face_value: float,
    coupon_rate: float,
    maturity_years: float,
    frequency: int,
    ytm: float,
    price: float | None = None,
) -> float:
    """
    Convexity (annual): second-order sensitivity so that
    dP/P ≈ -Modified_Duration*dy + 0.5*Convexity*(dy)^2.
    Formula: (1/(P*(1+r)^2*f^2)) * sum CF_t * t*(t+1) / (1+r)^t.
    """
    if price is None:
        price = bond_price_dcf(face_value, coupon_rate, maturity_years, frequency, ytm)
    if price <= 0 or np.isnan(price):
        return np.nan
    n = int(round(maturity_years * frequency))
    if n <= 0:
        return np.nan
    r = ytm / frequency
    if r <= -1:
        return np.nan
    coupon_per_period = face_value * (coupon_rate / frequency)
    factor = 1.0 / (price * (1 + r) ** 2 * frequency**2)
    weighted_sum = 0.0
    for t in range(1, n + 1):
        if t < n:
            cf = coupon_per_period
        else:
            cf = coupon_per_period + face_value
        pv = cf / (1 + r) ** t
        weighted_sum += t * (t + 1) * pv
    return factor * weighted_sum


def cash_flows_table(
    face_value: float,
    coupon_rate: float,
    maturity_years: float,
    frequency: int,
    ytm: float,
) -> list[dict]:
    """
    Return list of dicts: period (1-based), time_years, cash_flow, pv.
    For display in Statistics / Bond details.
    """
    n = int(round(maturity_years * frequency))
    if n <= 0:
        return []
    r = ytm / frequency
    if r <= -1:
        return []
    coupon_per_period = face_value * (coupon_rate / frequency)
    rows = []
    for t in range(1, n + 1):
        if t < n:
            cf = coupon_per_period
        else:
            cf = coupon_per_period + face_value
        pv = cf / (1 + r) ** t
        rows.append({
            "period": t,
            "time_years": t / frequency,
            "cash_flow": cf,
            "pv": pv,
        })
    return rows


def accrued_interest(
    face_value: float,
    coupon_rate: float,
    maturity_years: float,
    frequency: int,
    settlement_date: "pd.Timestamp",
    reference_date: "pd.Timestamp | None" = None,
) -> float:
    """
    Accrued interest (30/360). Clean price = DCF price. Dirty = Clean + Accrued.
    By default uses a fixed coupon grid (e.g. Jan 1 / Jul 1 for semi-annual) so that
    last_coupon and next_coupon change when the user changes the settlement date,
    and accrued interest moves (e.g. 2025-06-12 -> last=2025-01-01, next=2025-07-01).
    If reference_date is provided, coupon dates are built from ref + maturity_years
    (legacy behaviour for tests).
    Returns accrued amount in same units as face value.
    """
    if pd is None or frequency < 1 or maturity_years <= 0 or face_value <= 0:
        return 0.0
    n = int(round(maturity_years * frequency))
    if n <= 0:
        return 0.0
    settlement = settlement_date.normalize() if hasattr(settlement_date, "normalize") else settlement_date
    months_per_period = 12 // frequency
    try:
        maturity_date = settlement + pd.DateOffset(years=maturity_years)
        if settlement >= maturity_date:
            return 0.0  # bond already matured
        if reference_date is not None:
            # Legacy: build coupon dates from reference (e.g. today) for tests
            ref = reference_date.normalize() if hasattr(reference_date, "normalize") else reference_date
            maturity_date = ref + pd.DateOffset(years=maturity_years)
            if settlement >= maturity_date:
                return 0.0
            coupon_dates = [
                maturity_date - pd.DateOffset(months=months_per_period * k)
                for k in range(1, n + 1)
            ]
        else:
            # Fixed grid so accrued moves when user changes settlement (e.g. Jan 1, Jul 1)
            epoch = pd.Timestamp("2000-01-01").normalize()
            n_grid = 500  # enough to cover any settlement
            coupon_dates = [
                epoch + pd.DateOffset(months=months_per_period * k)
                for k in range(n_grid)
            ]
        past_or_on = [d for d in coupon_dates if d <= settlement]
        future = [d for d in coupon_dates if d > settlement]
        if not past_or_on or not future:
            return 0.0
        last_coupon = max(past_or_on)
        next_coupon = min(future)
    except (ValueError, TypeError):
        return 0.0
    days_accrued = _days_30_360(last_coupon, settlement)
    days_in_period = _days_30_360(last_coupon, next_coupon)
    if days_in_period <= 0:
        return 0.0
    coupon_per_period = face_value * (coupon_rate / frequency)
    accrued = (days_accrued / days_in_period) * coupon_per_period
    return min(accrued, coupon_per_period)  # cap at one period


def rate_scenarios(
    face_value: float,
    coupon_rate: float,
    maturity_years: float,
    frequency: int,
    base_ytm: float,
    base_price: float,
    shifts_bp: list[int],
) -> list[dict]:
    """
    Price and delta P/P for given yield shifts (in basis points).
    Returns list of dicts: shift_bp, new_ytm, new_price, delta_pct.
    """
    results = []
    for shift_bp in shifts_bp:
        dy = shift_bp / 10_000.0
        new_ytm = base_ytm + dy
        new_price = bond_price_dcf(face_value, coupon_rate, maturity_years, frequency, new_ytm)
        if np.isnan(new_price) or new_price <= 0:
            continue
        delta_pct = (new_price - base_price) / base_price * 100 if base_price > 0 else 0
        results.append({
            "shift_bp": shift_bp,
            "new_ytm": new_ytm,
            "new_price": new_price,
            "delta_pct": delta_pct,
        })
    return results


def price_yield_curve(
    face_value: float,
    coupon_rate: float,
    maturity_years: float,
    frequency: int,
    ytm_min: float = 0.005,
    ytm_max: float = 0.15,
    n_points: int = 50,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Return (yield_grid, price_grid) for plotting Price-Yield curve.
    yield_grid in decimal (e.g. 0.05 for 5%); price_grid in currency.
    """
    yield_grid = np.linspace(ytm_min, ytm_max, n_points)
    price_grid = np.array([
        bond_price_dcf(face_value, coupon_rate, maturity_years, frequency, y)
        for y in yield_grid
    ])
    return yield_grid, price_grid


def run_bond_analysis(
    face_value: float,
    coupon_rate: float,
    maturity_years: float,
    frequency: int,
    ytm: float | None = None,
    price: float | None = None,
    ytm_min: float = 0.005,
    ytm_max: float = 0.15,
    n_curve_points: int = 50,
) -> dict:
    """
    Run full bond analysis: price, YTM, Macaulay/Modified duration, convexity,
    cash flows, and Price-Yield curve.
    Either provide ytm (then price is computed) or price (then YTM is solved).
    Returns dict with success, error (if fail), and on success: price, ytm,
    macaulay_duration, modified_duration, convexity, cash_flows, yield_grid, price_grid.
    """
    if face_value <= 0:
        return {"success": False, "error": "Face value must be positive."}
    if frequency not in (1, 2, 4):
        return {"success": False, "error": "Frequency must be 1, 2, or 4."}
    if maturity_years <= 0 or maturity_years > 100:
        return {"success": False, "error": "Maturity must be positive and not exceed 100 years."}
    n = int(round(maturity_years * frequency))
    if n <= 0:
        return {"success": False, "error": "Number of periods must be at least 1 (increase maturity or frequency)."}

    if ytm is not None and price is not None:
        return {"success": False, "error": "Provide either YTM or price, not both."}
    if ytm is None and price is None:
        return {"success": False, "error": "Provide either YTM or price."}

    if ytm is not None:
        # Price from YTM
        price_val = bond_price_dcf(face_value, coupon_rate, maturity_years, frequency, ytm)
        if np.isnan(price_val) or price_val <= 0:
            return {"success": False, "error": "Could not compute price for the given YTM."}
        ytm_val = ytm
    else:
        # YTM from price
        ytm_val = ytm_from_price(face_value, coupon_rate, maturity_years, frequency, price)
        if ytm_val is None:
            return {"success": False, "error": "Could not solve for YTM from the given price (check price is positive and reasonable)."}
        price_val = bond_price_dcf(face_value, coupon_rate, maturity_years, frequency, ytm_val)

    mac_dur = macaulay_duration(face_value, coupon_rate, maturity_years, frequency, ytm_val, price_val)
    mod_dur = modified_duration(face_value, coupon_rate, maturity_years, frequency, ytm_val, price_val)
    conv = convexity(face_value, coupon_rate, maturity_years, frequency, ytm_val, price_val)

    cf_rows = cash_flows_table(face_value, coupon_rate, maturity_years, frequency, ytm_val)
    y_grid, p_grid = price_yield_curve(
        face_value, coupon_rate, maturity_years, frequency,
        ytm_min=ytm_min, ytm_max=ytm_max, n_points=n_curve_points,
    )

    # Round outputs for consistent display (2 decimals for price, 2 for durations/convexity)
    return {
        "success": True,
        "price": round(price_val, 2),
        "ytm": ytm_val,
        "macaulay_duration": round(mac_dur, 2),
        "modified_duration": round(mod_dur, 2),
        "convexity": round(conv, 2),
        "cash_flows": cf_rows,
        "yield_grid": y_grid,
        "price_grid": p_grid,
    }
