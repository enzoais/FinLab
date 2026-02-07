"""
Modern Portfolio Theory (Markowitz) - Business logic.
Multi-asset price download, log returns, covariance, optimization (max Sharpe, min variance, target return),
efficient frontier, random portfolios, risk decomposition.
"""
import numpy as np
import pandas as pd
import yfinance as yf
from scipy.optimize import minimize

# Minimum overlapping observations for valid analysis
MIN_OBS = 60
# Annualization factor for daily data
ANNUALIZE = 252


def _close_series_from_df(df: pd.DataFrame, ticker: str) -> pd.Series | None:
    """Extract Close (or Adj Close) as 1D Series from yfinance multi-ticker DataFrame."""
    if not isinstance(df.columns, pd.MultiIndex):
        return None
    if ticker not in df.columns.get_level_values(0):
        return None
    try:
        s = df[ticker]["Close"] if "Close" in df[ticker].columns else df[ticker].iloc[:, 3]
    except Exception:
        return None
    if isinstance(s, pd.DataFrame):
        s = s.squeeze()
    vals = np.asarray(s)
    if vals.ndim > 1:
        vals = vals.ravel()
    idx = s.index if hasattr(s, "index") else df.index
    out = pd.Series(vals, index=idx).dropna()
    return out if len(out) >= 2 else None


def download_prices_multi(
    tickers: list[str],
    start_date: pd.Timestamp,
    end_date: pd.Timestamp,
) -> dict:
    """
    Download adjusted close prices for multiple tickers and align on common dates.
    Returns dict with keys: success, prices (DataFrame, columns=tickers), dates, error (if success is False).
    """
    tickers = [t.strip().upper() for t in tickers if t and str(t).strip()]
    if len(tickers) < 2:
        return {"success": False, "error": "At least 2 tickers are required."}

    try:
        df = yf.download(
            tickers,
            start=start_date,
            end=end_date,
            progress=False,
            auto_adjust=True,
            threads=False,
            group_by="ticker",
        )
    except Exception as e:
        return {"success": False, "error": f"Download failed: {str(e)}"}

    if df.empty or len(df) < 2:
        return {"success": False, "error": "No or insufficient price data in the selected period."}

    # Single-ticker case: yfinance returns DataFrame with columns [Open, High, Low, Close, ...]
    if len(tickers) == 1:
        t = tickers[0]
        s = df["Close"] if "Close" in df.columns else df.iloc[:, 3]
        if isinstance(s, pd.DataFrame):
            s = s.squeeze()
        prices = pd.DataFrame({t: pd.Series(np.asarray(s).ravel(), index=df.index).dropna()})
    else:
        # Multi-ticker: columns are (Ticker, OHLCV)
        series_list = []
        for t in tickers:
            ser = _close_series_from_df(df, t)
            if ser is not None:
                series_list.append((t, ser))
        if len(series_list) < 2:
            return {"success": False, "error": "Insufficient valid series after download (need at least 2)."}
        prices = pd.DataFrame({t: s for t, s in series_list}).dropna(how="any")

    if len(prices) < MIN_OBS:
        return {"success": False, "error": f"Insufficient overlapping data (need at least {MIN_OBS} observations)."}

    return {
        "success": True,
        "prices": prices,
        "dates": prices.index,
        "tickers": list(prices.columns),
    }


def log_returns(prices: pd.Series | pd.DataFrame) -> pd.Series | pd.DataFrame:
    """Compute daily log returns from price series or DataFrame (column-wise)."""
    if isinstance(prices, pd.Series):
        return np.log(prices / prices.shift(1)).dropna()
    return np.log(prices / prices.shift(1)).dropna()


def annualize_mean_and_cov(daily_returns: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
    """From daily returns DataFrame, return (mean_annual vector, cov_annual matrix)."""
    mu_daily = daily_returns.mean(axis=0).values
    cov_daily = daily_returns.cov().values
    mu_annual = mu_daily * ANNUALIZE
    cov_annual = cov_daily * ANNUALIZE
    return mu_annual, cov_annual


def _ensure_positive_semidefinite(cov: np.ndarray, ridge: float = 1e-8) -> np.ndarray:
    """Add small ridge to diagonal if needed so optimization is stable."""
    n = cov.shape[0]
    cov = np.asarray(cov, dtype=float)
    if not np.allclose(cov, cov.T):
        cov = (cov + cov.T) / 2
    min_eig = np.min(np.linalg.eigvalsh(cov))
    if min_eig < ridge:
        cov = cov + (ridge - min_eig) * np.eye(n)
    return cov


def optimize_portfolio(
    mu: np.ndarray,
    cov: np.ndarray,
    risk_free_rate: float,
    objective: str,
    target_return: float | None,
    long_only: bool = True,
    max_weight_per_asset: float = 1.0,
    max_gross_exposure: float | None = None,
) -> dict:
    """
    Find optimal portfolio weights.
    objective: "max_sharpe" | "min_variance" | "target_return"
    target_return: annualized decimal (e.g. 0.10 for 10%), required if objective == "target_return".
    max_weight_per_asset: cap per asset (e.g. 0.4 for 40%), 1.0 = no cap.
    max_gross_exposure: cap on sum(abs(weights)) (e.g. 2.0 for 200%), None = no cap. Only used when long_only is False.
    Returns dict: success, weights, expected_return, volatility, sharpe_ratio, error.
    """
    n = len(mu)
    cov = _ensure_positive_semidefinite(cov)
    max_w = min(1.0, max(0.0, float(max_weight_per_asset)))

    def port_return(w):
        return np.dot(w, mu)

    def port_vol(w):
        return np.sqrt(np.dot(w, np.dot(cov, w)))

    def neg_sharpe(w):
        r = port_return(w)
        v = port_vol(w)
        if v <= 0:
            return 1e10
        return -(r - risk_free_rate) / v

    bounds = [(0.0, max_w)] * n if long_only else [(None, None)] * n
    constraints = [{"type": "eq", "fun": lambda w: np.sum(w) - 1}]
    if not long_only and max_gross_exposure is not None and max_gross_exposure >= 1.0:
        constraints.append({"type": "ineq", "fun": lambda w: float(max_gross_exposure) - np.sum(np.abs(w))})

    if objective == "max_sharpe":
        res = minimize(neg_sharpe, np.ones(n) / n, method="SLSQP", bounds=bounds, constraints=constraints)
    elif objective == "min_variance":
        res = minimize(lambda w: port_vol(w) ** 2, np.ones(n) / n, method="SLSQP", bounds=bounds, constraints=constraints)
    elif objective == "target_return":
        if target_return is None:
            return {"success": False, "error": "Target return is required for target_return objective."}
        constraints_target = constraints + [{"type": "eq", "fun": lambda w: port_return(w) - target_return}]
        res = minimize(
            lambda w: port_vol(w) ** 2,
            np.ones(n) / n,
            method="SLSQP",
            bounds=bounds,
            constraints=constraints_target,
        )
    else:
        return {"success": False, "error": f"Unknown objective: {objective}."}

    if not res.success:
        return {"success": False, "error": res.message or "Optimization did not converge."}

    w = res.x
    w = w / w.sum()  # re-normalize
    ret = port_return(w)
    vol = port_vol(w)
    sharpe = (ret - risk_free_rate) / vol if vol > 0 else 0.0

    if objective == "target_return" and abs(ret - target_return) > 1e-4:
        return {"success": False, "error": "Target return not achievable with given constraints."}

    return {
        "success": True,
        "weights": w,
        "expected_return": ret,
        "volatility": vol,
        "sharpe_ratio": sharpe,
    }


def efficient_frontier(
    mu: np.ndarray,
    cov: np.ndarray,
    risk_free_rate: float,
    n_points: int = 50,
    long_only: bool = True,
    max_weight_per_asset: float = 1.0,
    max_gross_exposure: float | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Compute efficient frontier (volatility, return) for a grid of target returns.
    Returns (frontier_curve, frontier_weights): curve (n_points, 2) [vol, ret], weights (n_points, n).
    """
    cov = _ensure_positive_semidefinite(cov)
    n = len(mu)
    max_w = min(1.0, max(0.0, float(max_weight_per_asset)))
    min_ret = mu.min()
    max_ret = mu.max()
    if max_ret <= min_ret:
        target_returns = np.linspace(min_ret, min_ret + 0.01, n_points)
    else:
        target_returns = np.linspace(min_ret, max_ret, n_points)

    frontier = []
    weights_list = []
    bounds = [(0.0, max_w)] * n if long_only else [(None, None)] * n
    constraints_base = [{"type": "eq", "fun": lambda w: np.sum(w) - 1}]
    if not long_only and max_gross_exposure is not None and max_gross_exposure >= 1.0:
        constraints_base.append({"type": "ineq", "fun": lambda w: float(max_gross_exposure) - np.sum(np.abs(w))})

    for target in target_returns:
        constraints = constraints_base + [{"type": "eq", "fun": lambda w, t=target: np.dot(w, mu) - t}]
        res = minimize(
            lambda w: np.dot(w, np.dot(cov, w)),
            np.ones(n) / n,
            method="SLSQP",
            bounds=bounds,
            constraints=constraints,
        )
        if res.success:
            w = res.x / res.x.sum()
            vol = np.sqrt(np.dot(w, np.dot(cov, w)))
            ret = np.dot(w, mu)
            frontier.append([vol, ret])
            weights_list.append(w)
    if not frontier:
        return np.array([]).reshape(0, 2), np.array([]).reshape(0, n)
    return np.array(frontier), np.array(weights_list)


def random_portfolios(
    mu: np.ndarray,
    cov: np.ndarray,
    risk_free_rate: float,
    n_portfolios: int = 500,
    long_only: bool = True,
    max_weight_per_asset: float = 1.0,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Generate random portfolios (volatility, return, sharpe) for scatter cloud.
    Returns (cloud, weights): cloud shape (n_portfolios, 3) [vol, ret, sharpe], weights shape (n_portfolios, n).
    """
    n = len(mu)
    cov = _ensure_positive_semidefinite(cov)
    max_w = min(1.0, max(0.0, float(max_weight_per_asset)))
    out = []
    weights_list = []
    for _ in range(n_portfolios):
        if long_only:
            w = np.random.exponential(1, n)
        else:
            w = np.random.randn(n)
        w = w / w.sum()
        if long_only and max_w < 1.0:
            # Cap weights at max_w (simple scaling; may not sum to 1, so renormalize)
            w = np.minimum(w, max_w)
            s = w.sum()
            if s > 0:
                w = w / s
        ret = np.dot(w, mu)
        vol = np.sqrt(np.dot(w, np.dot(cov, w)))
        sharpe = (ret - risk_free_rate) / vol if vol > 0 else 0
        out.append([vol, ret, sharpe])
        weights_list.append(w)
    return np.array(out), np.array(weights_list)


def _download_benchmark_returns(
    benchmark_ticker: str,
    start_date: pd.Timestamp,
    end_date: pd.Timestamp,
    align_index: pd.Index,
) -> pd.Series | None:
    """Download benchmark prices and return daily log returns aligned to align_index. Returns None on failure."""
    if not benchmark_ticker or not benchmark_ticker.strip():
        return None
    try:
        df = yf.download(
            benchmark_ticker.strip().upper(),
            start=start_date,
            end=end_date,
            progress=False,
            auto_adjust=True,
            threads=False,
        )
    except Exception:
        return None
    if df.empty or len(df) < 2:
        return None
    close = df["Close"] if "Close" in df.columns else df.iloc[:, 3]
    if isinstance(close, pd.DataFrame):
        close = close.squeeze()
    prices = pd.Series(np.asarray(close).ravel(), index=df.index).dropna()
    ret = log_returns(prices)
    if ret is None or len(ret) < 2:
        return None
    # Align to portfolio dates (keep only dates where both benchmark and portfolio have data)
    common = ret.index.intersection(align_index)
    if len(common) < MIN_OBS:
        return None
    return ret.loc[common].dropna()


def compute_var_cvar(returns_daily: np.ndarray, confidence: float = 0.95) -> tuple[float, float]:
    """
    Historical VaR and CVaR (Expected Shortfall) from daily returns.
    returns_daily: 1D array of daily log returns.
    confidence: e.g. 0.95 for 95% VaR (5th percentile).
    Returns (var_daily, cvar_daily) as decimals (e.g. -0.012 for -1.2% per day).
    """
    if returns_daily is None or len(returns_daily) < 2:
        return (0.0, 0.0)
    r = np.asarray(returns_daily).ravel()
    q = (1 - confidence) * 100  # 5 for 95% VaR
    var = np.percentile(r, q)
    cvar = np.mean(r[r <= var]) if np.any(r <= var) else var
    return (float(var), float(cvar))


def risk_decomposition(weights: np.ndarray, cov: np.ndarray) -> np.ndarray:
    """
    Marginal contribution to risk: (cov @ w) / port_vol.
    Contribution to risk = weight * marginal_contribution (not normalized to sum to 1).
    Returns array of length n with marginal contributions.
    """
    vol = np.sqrt(np.dot(weights, np.dot(cov, weights)))
    if vol <= 0:
        return np.zeros_like(weights)
    marginal = np.dot(cov, weights) / vol
    return weights * marginal  # contribution per asset


def run_markowitz_analysis(
    tickers: list[str],
    start_date: pd.Timestamp,
    end_date: pd.Timestamp,
    risk_free_rate: float,
    objective: str,
    target_return: float | None,
    long_only: bool = True,
    max_weight_per_asset: float = 1.0,
    max_gross_exposure: float | None = None,
    benchmark_ticker: str | None = None,
    n_frontier_points: int = 50,
    n_random_portfolios: int = 500,
) -> dict:
    """
    Full Markowitz pipeline: download, returns, covariance, optimize, frontier, random cloud,
    optional benchmark comparison, VaR/CVaR.
    Returns dict with success, tickers, weights, expected_return, volatility, sharpe_ratio,
    frontier_curve, random_cloud, mean_returns, cov_matrix, corr_matrix, n_obs, risk_contributions,
    condition_number, var_95_daily, cvar_95_daily, benchmark_* (if benchmark_ticker), error.
    """
    out = {
        "success": False,
        "tickers": tickers,
        "weights": None,
        "expected_return": None,
        "volatility": None,
        "sharpe_ratio": None,
        "frontier_curve": None,
        "frontier_weights": None,
        "random_cloud": None,
        "random_weights": None,
        "mean_returns": None,
        "cov_matrix": None,
        "corr_matrix": None,
        "n_obs": None,
        "risk_contributions": None,
        "condition_number": None,
        "var_95_daily": None,
        "cvar_95_daily": None,
        "benchmark_return_annual": None,
        "benchmark_vol_annual": None,
        "alpha_vs_benchmark": None,
        "tracking_error_annual": None,
        "information_ratio": None,
        "error": None,
    }

    dl = download_prices_multi(tickers, start_date, end_date)
    if not dl["success"]:
        out["error"] = dl["error"]
        return out

    prices = dl["prices"]
    tickers = dl["tickers"]
    returns_daily = log_returns(prices)
    if returns_daily.isna().any().any():
        returns_daily = returns_daily.dropna()
    if len(returns_daily) < MIN_OBS:
        out["error"] = f"Insufficient data after dropna (need at least {MIN_OBS} observations)."
        return out

    mu, cov = annualize_mean_and_cov(returns_daily)
    cov = _ensure_positive_semidefinite(cov)
    try:
        cond = np.linalg.cond(cov)
    except Exception:
        cond = None
    out["condition_number"] = cond
    out["mean_returns"] = mu
    out["cov_matrix"] = cov
    out["corr_matrix"] = np.corrcoef(returns_daily.T) if len(tickers) > 1 else np.array([[1.0]])
    out["n_obs"] = len(returns_daily)

    opt = optimize_portfolio(mu, cov, risk_free_rate, objective, target_return, long_only, max_weight_per_asset, max_gross_exposure)
    if not opt["success"]:
        out["error"] = opt["error"]
        return out

    out["weights"] = opt["weights"]
    out["expected_return"] = opt["expected_return"]
    out["volatility"] = opt["volatility"]
    out["sharpe_ratio"] = opt["sharpe_ratio"]
    out["risk_contributions"] = risk_decomposition(opt["weights"], cov)

    # Portfolio daily returns (for VaR/CVaR and benchmark comparison)
    port_returns_daily = np.dot(returns_daily.values, opt["weights"])
    var_95, cvar_95 = compute_var_cvar(port_returns_daily, confidence=0.95)
    out["var_95_daily"] = var_95
    out["cvar_95_daily"] = cvar_95

    # Benchmark comparison (optional)
    if benchmark_ticker and benchmark_ticker.strip():
        bench_ret = _download_benchmark_returns(benchmark_ticker, start_date, end_date, returns_daily.index)
        if bench_ret is not None and len(bench_ret) >= MIN_OBS:
            common_idx = returns_daily.index.intersection(bench_ret.index)
            if len(common_idx) >= MIN_OBS:
                port_aligned = np.dot(returns_daily.loc[common_idx].values, opt["weights"])
                bench_aligned = bench_ret.loc[common_idx].values
                bench_return_annual = np.mean(bench_aligned) * ANNUALIZE
                bench_vol_annual = np.std(bench_aligned) * np.sqrt(ANNUALIZE)
                diff = port_aligned - bench_aligned
                tracking_error_annual = np.std(diff) * np.sqrt(ANNUALIZE)
                alpha_annual = out["expected_return"] - bench_return_annual
                information_ratio = alpha_annual / tracking_error_annual if tracking_error_annual > 0 else 0.0
                out["benchmark_return_annual"] = bench_return_annual
                out["benchmark_vol_annual"] = bench_vol_annual
                out["alpha_vs_benchmark"] = alpha_annual
                out["tracking_error_annual"] = tracking_error_annual
                out["information_ratio"] = information_ratio

    frontier_curve, frontier_weights = efficient_frontier(
        mu, cov, risk_free_rate, n_points=n_frontier_points, long_only=long_only, max_weight_per_asset=max_weight_per_asset, max_gross_exposure=max_gross_exposure
    )
    out["frontier_curve"] = frontier_curve
    out["frontier_weights"] = frontier_weights

    random_cloud, random_weights = random_portfolios(
        mu, cov, risk_free_rate, n_portfolios=n_random_portfolios, long_only=long_only, max_weight_per_asset=max_weight_per_asset
    )
    out["random_cloud"] = random_cloud
    out["random_weights"] = random_weights

    out["success"] = True
    out["error"] = None
    return out


def diversification_ratio(weights: np.ndarray, cov: np.ndarray) -> float:
    """Ratio of weighted asset volatilities to portfolio volatility. Higher = more diversification benefit."""
    vol_assets = np.sqrt(np.diag(cov))
    weighted_vol = np.dot(weights, vol_assets)
    port_vol = np.sqrt(np.dot(weights, np.dot(cov, weights)))
    if port_vol <= 0:
        return 0.0
    return weighted_vol / port_vol
