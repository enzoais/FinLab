"""
CAPM & Beta Analysis - Business logic.
Download prices (yfinance), log returns, OLS regression, Kₑ, Rolling Beta, Adjusted Beta.
"""
import numpy as np
import pandas as pd
import yfinance as yf

# Optional: use statsmodels for OLS (t-stat, p-value, CI). Fallback to numpy if not available.
try:
    import statsmodels.api as sm
    HAS_STATSMODELS = True
except ImportError:
    HAS_STATSMODELS = False


# Benchmark ticker mapping (label -> yfinance ticker)
# US, then Europe, then Japan — order preserved for the selectbox
BENCHMARK_OPTIONS = {
    # US broad market
    "S&P 500": "^GSPC",
    "SPY (ETF)": "SPY",
    "NASDAQ Composite": "^IXIC",
    "QQQ (ETF, Nasdaq-100)": "QQQ",
    "Russell 2000": "^RUT",
    "IWM (ETF)": "IWM",
    # Europe
    "Euro Stoxx 50": "^STOXX50E",
    "DAX (Germany)": "^GDAXI",
    "CAC 40 (France)": "^FCHI",
    "FTSE 100 (UK)": "^FTSE",
    "SMI (Switzerland)": "^SSMI",
    # Japan
    "Nikkei 225": "^N225",
}

ROLLING_WINDOWS = [60, 126, 252]


def _looks_like_ticker(s: str) -> bool:
    """True if string looks like a ticker (e.g. AAPL, BRK.B), not a company name."""
    s = s.strip().upper()
    if not s or len(s) > 10:
        return False
    # Allow letters and at most one dot (e.g. BRK.B)
    cleaned = s.replace(".", "")
    return cleaned.isalpha() and s.count(".") <= 1


def resolve_asset_ticker(query: str) -> tuple[str, str | None, str | None]:
    """
    Resolve user input to a yfinance ticker. Accepts ticker (AAPL) or company name (Apple).
    Returns (ticker_symbol, display_name_or_none, exchange_display_or_none).
    """
    if not query or not query.strip():
        return ("", None, None)
    q = query.strip()
    # Always try search first so "Apple" -> AAPL (otherwise "Apple".upper() = "APPLE" is wrong)
    try:
        search = yf.Search(q, max_results=5)
        if search.quotes and len(search.quotes) > 0:
            first = search.quotes[0]
            symbol = first.get("symbol")
            longname = first.get("longname") or first.get("shortname") or None
            exch_disp = first.get("exchDisp") or None
            if symbol:
                return (str(symbol).strip(), longname, exch_disp)
    except Exception:
        pass
    # Fallback: use as ticker only if it looks like one (e.g. AAPL, MSFT, BRK.B)
    if _looks_like_ticker(q):
        return (q.upper(), None, None)
    return (q.upper(), None, None)


def get_asset_display_info(ticker: str) -> tuple[str | None, str | None]:
    """
    Get company name and exchange for a ticker (e.g. for display when user typed ticker directly).
    Returns (long_name_or_none, exchange_display_or_none). Uses yfinance Ticker.info.
    """
    if not ticker or not ticker.strip():
        return (None, None)
    try:
        t = yf.Ticker(ticker.strip().upper())
        info = t.info or {}
        name = info.get("longName") or info.get("shortName")
        exchange = info.get("fullExchangeName") or info.get("exchange")
        return (name, exchange)
    except Exception:
        return (None, None)


def download_prices(
    asset_ticker: str,
    benchmark_ticker: str,
    start_date: pd.Timestamp,
    end_date: pd.Timestamp,
) -> dict:
    """
    Download adjusted close prices for asset and benchmark (two separate yfinance calls).
    Returns dict with keys: success, asset, benchmark, error (if success is False).
    """
    if not asset_ticker or not asset_ticker.strip():
        return {"success": False, "error": "Asset ticker is required."}
    if not benchmark_ticker or not benchmark_ticker.strip():
        return {"success": False, "error": "Benchmark ticker is required."}

    a = asset_ticker.strip().upper()
    b = benchmark_ticker.strip().upper()

    try:
        df_asset = yf.download(
            a,
            start=start_date,
            end=end_date,
            progress=False,
            auto_adjust=True,
            threads=False,
        )
        df_bench = yf.download(
            b,
            start=start_date,
            end=end_date,
            progress=False,
            auto_adjust=True,
            threads=False,
        )
    except Exception as e:
        return {"success": False, "error": f"Download failed: {str(e)}"}

    if df_asset.empty or len(df_asset) < 2:
        return {"success": False, "error": f"No or insufficient data for asset '{a}' in the selected period."}
    if df_bench.empty or len(df_bench) < 2:
        return {"success": False, "error": f"No or insufficient data for benchmark '{b}' in the selected period."}

    # Use Close (or Adj Close if present); ensure we get Series with index
    def _close_series(df):
        if "Close" in df.columns:
            s = df["Close"]
        else:
            s = df.iloc[:, 3]
        if isinstance(s, pd.DataFrame):
            s = s.squeeze()
        # Force 1D: yfinance can yield (n, 1) ndarray; avoid pd.Series(2D)
        vals = np.asarray(s)
        if vals.ndim > 1:
            vals = vals.ravel()
        idx = s.index if isinstance(s, (pd.Series, pd.DataFrame)) else df.index
        return pd.Series(vals, index=idx).dropna()

    asset_series = _close_series(df_asset)
    bench_series = _close_series(df_bench)
    if len(asset_series) < 2 or len(bench_series) < 2:
        return {"success": False, "error": "Insufficient price data after dropping NaN."}

    aligned = pd.DataFrame({"asset": asset_series, "benchmark": bench_series}).dropna()
    if len(aligned) < 30:
        return {"success": False, "error": "Insufficient overlapping data (need at least 30 observations)."}

    return {
        "success": True,
        "asset": aligned["asset"],
        "benchmark": aligned["benchmark"],
        "dates": aligned.index,
    }


def log_returns(prices: pd.Series) -> pd.Series:
    """Compute daily log returns from price series."""
    return np.log(prices / prices.shift(1)).dropna()


def ols_regression(
    asset_returns: pd.Series,
    market_returns: pd.Series,
    risk_free_rate: float,
) -> dict:
    """
    CAPM regression: (R_i - Rf) = alpha + beta * (R_m - Rf).
    Returns dict: beta, alpha (Jensen), r_squared, t_stat_beta, p_value_beta, ci_low, ci_high.
    """
    excess_asset = asset_returns - risk_free_rate / 252  # daily Rf
    excess_market = market_returns - risk_free_rate / 252

    # Align
    aligned = pd.DataFrame({"excess_asset": excess_asset, "excess_market": excess_market}).dropna()
    if len(aligned) < 10:
        return {
            "success": False,
            "error": "Insufficient data for regression (need at least 10 observations).",
        }

    y = aligned["excess_asset"]
    x = aligned["excess_market"]

    if HAS_STATSMODELS:
        x_const = sm.add_constant(x)
        model = sm.OLS(y, x_const).fit()
        # Intercept = Jensen's Alpha, slope = Beta
        alpha = model.params["const"]
        beta = model.params["excess_market"]
        r_squared = model.rsquared
        t_stat_beta = model.tvalues["excess_market"]
        p_value_beta = model.pvalues["excess_market"]
        ci = model.conf_int(alpha=0.05).loc["excess_market"]
        ci_low, ci_high = ci[0], ci[1]
        return {
            "success": True,
            "beta": float(beta),
            "alpha": float(alpha),
            "r_squared": float(r_squared),
            "t_stat_beta": float(t_stat_beta),
            "p_value_beta": float(p_value_beta),
            "ci_low": float(ci_low),
            "ci_high": float(ci_high),
            "n_obs": int(model.nobs),
            "fitted_excess_market": x_const["excess_market"],
            "fitted_excess_asset": model.fittedvalues,
            "resid": model.resid,
        }
    else:
        # Fallback: numpy OLS (no t-stat, p-value, CI)
        x_mat = np.column_stack([np.ones(len(x)), x.values])
        b, _, _, _ = np.linalg.lstsq(x_mat, y.values, rcond=None)
        alpha, beta = float(b[0]), float(b[1])
        fitted = x_mat @ b
        ss_res = np.sum((y.values - fitted) ** 2)
        ss_tot = np.sum((y.values - y.mean()) ** 2)
        r_squared = 1 - (ss_res / ss_tot) if ss_tot > 0 else 0.0
        return {
            "success": True,
            "beta": beta,
            "alpha": alpha,
            "r_squared": r_squared,
            "t_stat_beta": None,
            "p_value_beta": None,
            "ci_low": None,
            "ci_high": None,
            "n_obs": len(y),
            "fitted_excess_market": pd.Series(x.values.squeeze(), index=y.index),
            "fitted_excess_asset": pd.Series(fitted, index=y.index),
            "resid": None,
        }


def cost_of_equity(risk_free_rate: float, beta: float, expected_market_return: float) -> float:
    """Kₑ = Rf + β × (E(Rm) − Rf). All rates in decimal (e.g. 0.05 for 5%)."""
    return risk_free_rate + beta * (expected_market_return - risk_free_rate)


def rolling_beta(
    asset_returns: pd.Series,
    market_returns: pd.Series,
    risk_free_rate: float,
    window: int,
) -> pd.Series:
    """Rolling CAPM beta over a given window (in days). Returns a series indexed by date."""
    excess_asset = asset_returns - risk_free_rate / 252
    excess_market = market_returns - risk_free_rate / 252
    aligned = pd.DataFrame({"excess_asset": excess_asset, "excess_market": excess_market}).dropna()

    if len(aligned) < window:
        return pd.Series(dtype=float)

    betas = []
    dates = []
    for i in range(window, len(aligned) + 1):
        chunk = aligned.iloc[i - window : i]
        x = chunk["excess_market"].values
        y = chunk["excess_asset"].values
        x_mat = np.column_stack([np.ones(len(x)), x])
        b, _, _, _ = np.linalg.lstsq(x_mat, y, rcond=None)
        betas.append(b[1])
        dates.append(chunk.index[-1])
    return pd.Series(betas, index=dates)


def adjusted_beta(beta: float) -> float:
    """Bloomberg-style: Adjusted Beta = (2/3) × Raw Beta + (1/3) × 1."""
    return (2.0 / 3.0) * beta + (1.0 / 3.0)


def run_capm_analysis(
    asset_ticker: str,
    benchmark_ticker: str,
    start_date: pd.Timestamp,
    end_date: pd.Timestamp,
    risk_free_rate: float,
    rolling_window: int,
) -> dict:
    """
    Full pipeline: download, log returns, OLS, Kₑ, rolling beta, adjusted beta.
    Returns a dict with keys: success, error (if failed), or all results.
    """
    out = download_prices(asset_ticker, benchmark_ticker, start_date, end_date)
    if not out["success"]:
        return out

    asset_prices = out["asset"]
    benchmark_prices = out["benchmark"]

    asset_ret = log_returns(asset_prices)
    market_ret = log_returns(benchmark_prices)

    reg = ols_regression(asset_ret, market_ret, risk_free_rate)
    if not reg.get("success", True):
        return reg

    e_rm = market_ret.mean() * 252  # annualized expected market return (simple approx)
    ke = cost_of_equity(risk_free_rate, reg["beta"], e_rm)
    adj_beta = adjusted_beta(reg["beta"])

    roll = rolling_beta(asset_ret, market_ret, risk_free_rate, rolling_window)

    # Treynor ratio: (annualized excess return) / Beta
    excess_asset = asset_ret - risk_free_rate / 252
    ann_excess = excess_asset.mean() * 252
    treynor = (ann_excess / reg["beta"]) if reg["beta"] != 0 else None

    return {
        "success": True,
        "regression": reg,
        "cost_of_equity": ke,
        "adjusted_beta": adj_beta,
        "expected_market_return_annual": e_rm,
        "asset_returns": asset_ret,
        "market_returns": market_ret,
        "rolling_beta": roll,
        "treynor_ratio": treynor,
    }
