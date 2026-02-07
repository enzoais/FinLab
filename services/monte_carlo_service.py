"""
Monte Carlo wealth simulation - Business logic.
GBM paths, percentile curves, terminal wealth stats, VaR/CVaR (simulated), shortfall probability.
Optional: estimate drift and volatility from ticker; annual withdrawals/contributions; max drawdown.
"""
import numpy as np

try:
    import pandas as pd
    import yfinance as yf
    HAS_YFINANCE = True
except ImportError:
    HAS_YFINANCE = False
    pd = None

ANNUALIZE = 252
# Wealth floor after withdrawals (avoid negative); paths at or below this are treated as "ruin" for shortfall.
WEALTH_FLOOR = 1e-10


def simulate_gbm_paths(
    S0: float,
    mu: float,
    sigma: float,
    T_years: float,
    n_paths: int,
    n_steps_per_year: int,
    seed: int | None = None,
    annual_flow: float = 0.0,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Simulate GBM paths: S_t = S_0 * exp((mu - 0.5*sigma^2)*t + sigma*W_t).
    Optionally apply annual_flow at the end of each year (positive = withdrawal, negative = contribution).
    Returns (paths, time_axis) where paths is (n_paths, n_steps+1), time_axis is (n_steps+1,).
    """
    if seed is not None:
        np.random.seed(seed)
    n_steps = int(round(T_years * n_steps_per_year))
    if n_steps < 1:
        n_steps = 1
    dt = T_years / n_steps
    drift = (mu - 0.5 * sigma ** 2) * dt
    vol = sigma * np.sqrt(dt)
    Z = np.random.standard_normal((n_paths, n_steps))
    log_returns = drift + vol * Z

    paths = np.zeros((n_paths, n_steps + 1))
    paths[:, 0] = S0
    steps_per_year = max(1, int(n_steps_per_year))
    for i in range(n_steps):
        paths[:, i + 1] = paths[:, i] * np.exp(log_returns[:, i])
        # at end of each year (after step), apply flow
        step_index = i + 1
        if annual_flow != 0 and step_index % steps_per_year == 0 and step_index < n_steps + 1:
            paths[:, step_index] = np.maximum(paths[:, step_index] - annual_flow, WEALTH_FLOOR)
    time_axis = np.linspace(0, T_years, n_steps + 1)
    return paths, time_axis


def percentile_curves(
    paths: np.ndarray,
    percentiles: list[float] | None = None,
) -> dict[float, np.ndarray]:
    """
    For each time step, compute percentiles across paths.
    paths: (n_paths, n_steps+1). Returns dict percentile -> array of length n_steps+1.
    """
    if percentiles is None:
        percentiles = [5, 25, 50, 75, 95]
    out = {}
    for p in percentiles:
        out[p] = np.percentile(paths, p, axis=0)
    return out


def terminal_wealth_stats(paths: np.ndarray) -> dict[str, float]:
    """Mean, std, and percentiles (5, 25, 50, 75, 95) of terminal wealth."""
    terminal = paths[:, -1]
    return {
        "mean": float(np.mean(terminal)),
        "std": float(np.std(terminal)),
        "p5": float(np.percentile(terminal, 5)),
        "p25": float(np.percentile(terminal, 25)),
        "p50": float(np.percentile(terminal, 50)),
        "p75": float(np.percentile(terminal, 75)),
        "p95": float(np.percentile(terminal, 95)),
        "min": float(np.min(terminal)),
        "max": float(np.max(terminal)),
    }


def var_cvar_simulated(
    terminal_wealths: np.ndarray,
    S0: float,
    confidence: float = 0.95,
) -> dict[str, float]:
    """
    VaR and CVaR (Expected Shortfall) on simulated terminal wealth.
    - VaR at confidence c: level such that P(terminal_wealth <= VaR) = 1 - c (left-tail quantile).
    - CVaR: E[terminal_wealth | terminal_wealth <= VaR] (average of worst (1-c)% outcomes).
    - var_pct / cvar_pct: (level - S0) / S0 (negative = loss in % of initial wealth).
    Returns dict: var_level, var_pct, cvar_level, cvar_pct.
    """
    terminal_wealths = np.asarray(terminal_wealths).ravel()
    q = (1 - confidence) * 100  # e.g. 5 for 95% VaR => 5th percentile
    var_level = float(np.percentile(terminal_wealths, q))
    tail = terminal_wealths[terminal_wealths <= var_level]
    cvar_level = float(np.mean(tail)) if len(tail) > 0 else var_level
    # as percentage of initial: (W - S0) / S0, so loss in % = (S0 - W)/S0
    var_pct = (var_level - S0) / S0 if S0 != 0 else 0.0
    cvar_pct = (cvar_level - S0) / S0 if S0 != 0 else 0.0
    return {
        "var_level": var_level,
        "var_pct": var_pct,
        "cvar_level": cvar_level,
        "cvar_pct": cvar_pct,
    }


def probability_shortfall(terminal_wealths: np.ndarray, threshold: float) -> float:
    """
    Proportion of paths where terminal wealth is below the threshold (shortfall).
    When threshold is 0 (ruin), paths that hit the wealth floor (e.g. after large withdrawals)
    are counted as shortfall, since we cap wealth at WEALTH_FLOOR and never store negative values.
    """
    terminal_wealths = np.asarray(terminal_wealths).ravel()
    below = terminal_wealths < threshold
    if threshold <= WEALTH_FLOOR:
        below = below | (terminal_wealths <= WEALTH_FLOOR)
    return float(np.mean(below))


def max_drawdown_per_path(paths: np.ndarray) -> np.ndarray:
    """
    For each path, compute maximum drawdown (peak-to-trough).
    Drawdown at time t: (S_t - max_{s<=t} S_s) / max_{s<=t} S_s in [-1, 0].
    Returns (n_paths,) array of most negative drawdown per path (e.g. -0.25 = 25% max drawdown).
    """
    running_max = np.maximum.accumulate(paths, axis=1)
    drawdowns = (paths - running_max) / np.where(running_max > 0, running_max, 1.0)
    return np.min(drawdowns, axis=1)


def run_monte_carlo_analysis(
    S0: float,
    mu: float,
    sigma: float,
    T_years: float,
    n_paths: int,
    n_steps_per_year: int,
    confidence: float = 0.95,
    seed: int | None = None,
    annual_flow: float = 0.0,
    shortfall_threshold: float | None = None,
) -> dict:
    """
    Full Monte Carlo pipeline: GBM paths, percentile curves, terminal stats, VaR/CVaR, shortfall, max drawdown.
    Returns dict with paths, time_axis, percentile_curves, terminal_stats, var_cvar, prob_shortfall,
    max_drawdowns (array), drawdown_stats (dict).
    """
    paths, time_axis = simulate_gbm_paths(
        S0, mu, sigma, T_years, n_paths, n_steps_per_year, seed=seed, annual_flow=annual_flow
    )
    pcurves = percentile_curves(paths)
    tstats = terminal_wealth_stats(paths)
    terminal = paths[:, -1]
    var_cvar = var_cvar_simulated(terminal, S0, confidence=confidence)
    prob_shortfall = (
        probability_shortfall(terminal, shortfall_threshold)
        if shortfall_threshold is not None
        else None
    )
    max_dd = max_drawdown_per_path(paths)
    dd_stats = {
        "mean": float(np.mean(max_dd)),
        "p5": float(np.percentile(max_dd, 5)),
        "p50": float(np.percentile(max_dd, 50)),
        "p95": float(np.percentile(max_dd, 95)),
    }
    return {
        "success": True,
        "paths": paths,
        "time_axis": time_axis,
        "percentile_curves": pcurves,
        "terminal_stats": tstats,
        "var_cvar": var_cvar,
        "prob_shortfall": prob_shortfall,
        "shortfall_threshold": shortfall_threshold,
        "max_drawdowns": max_dd,
        "drawdown_stats": dd_stats,
    }


def estimate_mu_sigma_from_ticker(
    ticker: str,
    start_date,
    end_date,
) -> tuple[float | None, float | None]:
    """
    Estimate annualized drift (mu) and volatility (sigma) from historical prices.
    Uses log returns and annualization factor 252.
    Returns (mu, sigma) or (None, None) on failure.
    """
    if not HAS_YFINANCE or not ticker or not ticker.strip():
        return (None, None)
    t = ticker.strip().upper()
    try:
        hist = yf.download(
            t,
            start=start_date,
            end=end_date,
            progress=False,
            auto_adjust=True,
            threads=False,
        )
        if hist is None or len(hist) < 2:
            return (None, None)
        close = hist["Close"] if "Close" in hist.columns else hist.iloc[:, 3]
        if isinstance(close, pd.DataFrame):
            close = close.squeeze()
        close = close.dropna()
        if len(close) < 2:
            return (None, None)
        log_ret = np.log(close / close.shift(1)).dropna()
        if len(log_ret) < 2:
            return (None, None)
        mu_daily = float(np.mean(log_ret))
        sigma_daily = float(np.std(log_ret))
        mu_annual = mu_daily * ANNUALIZE
        sigma_annual = sigma_daily * np.sqrt(ANNUALIZE)
        return (mu_annual, sigma_annual)
    except Exception:
        return (None, None)
