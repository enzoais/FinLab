"""
Black-Scholes option pricing - Business logic.
European options (no dividends): BSM price, Greeks, implied volatility, put-call parity.
Optional: fetch spot and historical volatility from yfinance.
"""
import math
from typing import Literal

import numpy as np
from scipy.stats import norm
from scipy.optimize import brentq

try:
    import pandas as pd
    import yfinance as yf
    HAS_YFINANCE = True
except ImportError:
    HAS_YFINANCE = False
    pd = None


def _d1_d2(S: float, K: float, T: float, r: float, sigma: float) -> tuple[float, float]:
    """Compute d1 and d2 for Black-Scholes (European, no dividend)."""
    if T <= 0 or sigma <= 0 or S <= 0 or K <= 0:
        return (np.nan, np.nan)
    sqrt_T = math.sqrt(T)
    d1 = (math.log(S / K) + (r + 0.5 * sigma ** 2) * T) / (sigma * sqrt_T)
    d2 = d1 - sigma * sqrt_T
    return (d1, d2)


def bsm_call_price(S: float, K: float, T: float, r: float, sigma: float) -> float:
    """European call price (no dividend)."""
    if T <= 0 or sigma <= 0 or S <= 0 or K <= 0:
        return np.nan
    d1, d2 = _d1_d2(S, K, T, r, sigma)
    return S * norm.cdf(d1) - K * math.exp(-r * T) * norm.cdf(d2)


def bsm_put_price(S: float, K: float, T: float, r: float, sigma: float) -> float:
    """European put price (no dividend)."""
    if T <= 0 or sigma <= 0 or S <= 0 or K <= 0:
        return np.nan
    d1, d2 = _d1_d2(S, K, T, r, sigma)
    return K * math.exp(-r * T) * norm.cdf(-d2) - S * norm.cdf(-d1)


def bsm_greeks(
    S: float, K: float, T: float, r: float, sigma: float
) -> dict[str, float]:
    """
    Compute all Greeks. Theta per day, Vega per 1% vol move, Rho per 1% rate move.
    Returns dict: d1, d2, delta_call, delta_put, gamma, theta_call, theta_put, vega, rho_call, rho_put.
    """
    d1, d2 = _d1_d2(S, K, T, r, sigma)
    if np.isnan(d1):
        return {
            "d1": np.nan, "d2": np.nan,
            "delta_call": np.nan, "delta_put": np.nan,
            "gamma": np.nan, "theta_call": np.nan, "theta_put": np.nan,
            "vega": np.nan, "rho_call": np.nan, "rho_put": np.nan,
        }
    sqrt_T = math.sqrt(T)
    n_d1 = norm.pdf(d1)
    exp_rt = math.exp(-r * T)

    delta_call = norm.cdf(d1)
    delta_put = delta_call - 1.0

    # Gamma (same for call and put)
    if S * sigma * sqrt_T > 1e-15:
        gamma = n_d1 / (S * sigma * sqrt_T)
    else:
        gamma = np.nan

    # Theta: per calendar day (negative = time decay)
    # dV/dT in BSM: -S*n(d1)*sigma/(2*sqrt(T)) - r*K*exp(-r*T)*N(d2) for call
    theta_call_annual = -S * n_d1 * sigma / (2 * sqrt_T) - r * K * exp_rt * norm.cdf(d2)
    theta_put_annual = -S * n_d1 * sigma / (2 * sqrt_T) + r * K * exp_rt * norm.cdf(-d2)
    theta_call = theta_call_annual / 365.0
    theta_put = theta_put_annual / 365.0

    # Vega: sensitivity to 1% (0.01) move in volatility. V = S*n(d1)*sqrt(T) per unit sigma.
    vega = S * n_d1 * sqrt_T * 0.01

    # Rho: sensitivity to 1% (0.01) move in rate.
    rho_call = K * T * exp_rt * norm.cdf(d2) * 0.01
    rho_put = -K * T * exp_rt * norm.cdf(-d2) * 0.01

    return {
        "d1": d1,
        "d2": d2,
        "delta_call": delta_call,
        "delta_put": delta_put,
        "gamma": gamma,
        "theta_call": theta_call,
        "theta_put": theta_put,
        "vega": vega,
        "rho_call": rho_call,
        "rho_put": rho_put,
    }


def _no_arbitrage_bounds(
    S: float, K: float, T: float, r: float
) -> tuple[tuple[float, float], tuple[float, float]]:
    """
    No-arbitrage bounds for European call and put (no dividend).
    Returns (call_min, call_max), (put_min, put_max).
    """
    if T <= 0 or S <= 0 or K <= 0:
        return (0.0, S), (0.0, K)
    disc = math.exp(-r * T)
    call_min = max(0.0, S - K * disc)
    call_max = S
    put_min = max(0.0, K * disc - S)
    put_max = K * disc
    return (call_min, call_max), (put_min, put_max)


def implied_volatility(
    market_price: float,
    option_type: Literal["call", "put"],
    S: float,
    K: float,
    T: float,
    r: float,
    sigma_low: float = 0.001,
    sigma_high: float = 3.0,
) -> float | None:
    """
    Solve for sigma such that BSM price equals market_price.
    Returns annual volatility (decimal) or None if no solution.
    """
    if market_price <= 0 or T <= 0 or S <= 0 or K <= 0:
        return None

    def objective(sig: float) -> float:
        if option_type == "call":
            return bsm_call_price(S, K, T, r, sig) - market_price
        return bsm_put_price(S, K, T, r, sig) - market_price

    try:
        plow = objective(sigma_low)
        phigh = objective(sigma_high)
        if plow * phigh > 0:
            return None
        sigma = brentq(objective, sigma_low, sigma_high, xtol=1e-10)
        return float(sigma)
    except (ValueError, ZeroDivisionError):
        return None


def put_call_parity_check(
    S: float, K: float, T: float, r: float, call_price: float, put_price: float
) -> tuple[float, float, float]:
    """
    C - P should equal S - K*exp(-r*T). Returns (lhs, rhs, abs_diff).
    """
    lhs = call_price - put_price
    rhs = S - K * math.exp(-r * T)
    return (lhs, rhs, abs(lhs - rhs))


def fetch_spot_from_ticker(ticker: str) -> float | None:
    """
    Fetch latest close price for ticker from yfinance. Returns None on failure.
    """
    if not HAS_YFINANCE or not ticker or not ticker.strip():
        return None
    t = ticker.strip().upper()
    try:
        hist = yf.Ticker(t).history(period="5d", interval="1d", auto_adjust=True)
        if hist is None or hist.empty or "Close" not in hist.columns:
            return None
        close = hist["Close"].iloc[-1]
        return float(close)
    except Exception:
        return None


def historical_volatility(ticker: str, window_days: int = 30) -> float | None:
    """
    Annualized volatility from log returns over the last window_days.
    Returns None on failure or insufficient data.
    """
    if not HAS_YFINANCE or not ticker or not ticker.strip() or window_days < 2:
        return None
    t = ticker.strip().upper()
    try:
        # Request a bit more than window_days to have enough after dropna
        period = "1mo" if window_days <= 30 else "3mo" if window_days <= 90 else "1y"
        hist = yf.Ticker(t).history(period=period, interval="1d", auto_adjust=True)
        if hist is None or len(hist) < 2 or "Close" not in hist.columns:
            return None
        close = hist["Close"].iloc[-window_days - 1 :].dropna()
        if len(close) < 2:
            return None
        log_ret = np.log(close / close.shift(1)).dropna()
        if len(log_ret) < 2:
            return None
        sigma_daily = float(np.std(log_ret))
        sigma_annual = sigma_daily * math.sqrt(252)
        return sigma_annual
    except Exception:
        return None


def run_bs_analysis(
    spot: float,
    strike: float,
    time_to_expiry: float,
    risk_free_rate: float,
    volatility: float | None = None,
    market_price: float | None = None,
    option_type_for_iv: Literal["call", "put"] = "call",
) -> dict:
    """
    Full Black-Scholes pipeline: prices, Greeks, optional IV from market price.
    Returns dict with success, call_price, put_price, greeks, implied_vol (if solved), put_call_parity, etc.
    """
    out = {
        "success": False,
        "error": None,
        "spot": spot,
        "strike": strike,
        "T": time_to_expiry,
        "r": risk_free_rate,
        "sigma": volatility,
        "call_price": None,
        "put_price": None,
        "implied_vol": None,
        "delta_call": None,
        "delta_put": None,
        "gamma": None,
        "theta_call": None,
        "theta_put": None,
        "vega": None,
        "rho_call": None,
        "rho_put": None,
        "d1": None,
        "d2": None,
        "put_call_lhs": None,
        "put_call_rhs": None,
        "put_call_diff": None,
    }

    if spot <= 0 or strike <= 0:
        out["error"] = "Spot and strike must be positive."
        return out
    if time_to_expiry <= 0:
        out["error"] = "Time to expiry must be positive (T > 0)."
        return out

    sigma = volatility
    if market_price is not None and market_price > 0:
        # Solve for IV
        sigma = implied_volatility(
            market_price, option_type_for_iv, spot, strike, time_to_expiry, risk_free_rate
        )
        if sigma is None:
            (call_min, call_max), (put_min, put_max) = _no_arbitrage_bounds(
                spot, strike, time_to_expiry, risk_free_rate
            )
            if option_type_for_iv == "call":
                low, high = call_min, call_max
                name = "Call"
            else:
                low, high = put_min, put_max
                name = "Put"
            if market_price < low:
                out["error"] = (
                    f"Could not find implied volatility: market price {market_price:.2f} is "
                    f"below the minimum {name} value ({low:.2f}) for Spot={spot:.2f}, Strike={strike:.2f}. "
                    f"A {name} cannot be worth less than its intrinsic value."
                )
            elif market_price > high:
                out["error"] = (
                    f"Could not find implied volatility: market price {market_price:.2f} is "
                    f"above the maximum {name} value ({high:.2f}) for Spot={spot:.2f}, Strike={strike:.2f}. "
                    f"Check no-arbitrage bounds."
                )
            else:
                out["error"] = (
                    "Could not find implied volatility for the given market price. "
                    f"For {option_type_for_iv} with Spot={spot:.2f}, Strike={strike:.2f}: "
                    f"price must be between {low:.2f} and {high:.2f}."
                )
            return out
        out["implied_vol"] = sigma
    elif volatility is None or volatility <= 0:
        out["error"] = "Provide either volatility or market price to solve for IV."
        return out

    out["sigma"] = sigma
    call_price = bsm_call_price(spot, strike, time_to_expiry, risk_free_rate, sigma)
    put_price = bsm_put_price(spot, strike, time_to_expiry, risk_free_rate, sigma)
    out["call_price"] = call_price
    out["put_price"] = put_price

    greeks = bsm_greeks(spot, strike, time_to_expiry, risk_free_rate, sigma)
    out["d1"] = greeks["d1"]
    out["d2"] = greeks["d2"]
    out["delta_call"] = greeks["delta_call"]
    out["delta_put"] = greeks["delta_put"]
    out["gamma"] = greeks["gamma"]
    out["theta_call"] = greeks["theta_call"]
    out["theta_put"] = greeks["theta_put"]
    out["vega"] = greeks["vega"]
    out["rho_call"] = greeks["rho_call"]
    out["rho_put"] = greeks["rho_put"]

    lhs, rhs, diff = put_call_parity_check(
        spot, strike, time_to_expiry, risk_free_rate, call_price, put_price
    )
    out["put_call_lhs"] = lhs
    out["put_call_rhs"] = rhs
    out["put_call_diff"] = diff

    out["success"] = True
    return out


def sensitivity_scenarios(
    spot: float,
    strike: float,
    time_to_expiry: float,
    risk_free_rate: float,
    sigma: float,
    option_type: Literal["call", "put"],
    spot_shocks_pct: list[float] | None = None,
    vol_shocks_pct: list[float] | None = None,
    time_shock_days: float | None = None,
) -> list[dict]:
    """
    Price changes for small shocks in S, sigma, T. Each row: shock description, new price, delta_pct.
    """
    if option_type == "call":
        base_price = bsm_call_price(spot, strike, time_to_expiry, risk_free_rate, sigma)
    else:
        base_price = bsm_put_price(spot, strike, time_to_expiry, risk_free_rate, sigma)

    rows = []
    if spot_shocks_pct is None:
        spot_shocks_pct = [-5, -1, 1, 5]
    for pct in spot_shocks_pct:
        S_new = spot * (1 + pct / 100.0)
        if option_type == "call":
            p_new = bsm_call_price(S_new, strike, time_to_expiry, risk_free_rate, sigma)
        else:
            p_new = bsm_put_price(S_new, strike, time_to_expiry, risk_free_rate, sigma)
        delta_pct = (p_new - base_price) / base_price * 100.0 if base_price else 0.0
        rows.append({
            "shock": f"Spot {pct:+.0f}%",
            "new_price": p_new,
            "delta_pct": delta_pct,
        })
    if vol_shocks_pct is None:
        vol_shocks_pct = [-1, 1]
    for pct in vol_shocks_pct:
        sigma_new = sigma * (1 + pct / 100.0)
        if option_type == "call":
            p_new = bsm_call_price(spot, strike, time_to_expiry, risk_free_rate, sigma_new)
        else:
            p_new = bsm_put_price(spot, strike, time_to_expiry, risk_free_rate, sigma_new)
        delta_pct = (p_new - base_price) / base_price * 100.0 if base_price else 0.0
        rows.append({
            "shock": f"Vol {pct:+.0f}%",
            "new_price": p_new,
            "delta_pct": delta_pct,
        })
    if time_shock_days is not None and time_to_expiry > 0:
        T_new = max(1e-6, time_to_expiry - time_shock_days / 365.0)
        if option_type == "call":
            p_new = bsm_call_price(spot, strike, T_new, risk_free_rate, sigma)
        else:
            p_new = bsm_put_price(spot, strike, T_new, risk_free_rate, sigma)
        delta_pct = (p_new - base_price) / base_price * 100.0 if base_price else 0.0
        rows.append({
            "shock": f"Time -{time_shock_days} day(s)",
            "new_price": p_new,
            "delta_pct": delta_pct,
        })
    return rows
