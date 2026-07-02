"""
Backtesting Engine - Business logic (pur, testable sans réseau).

Rejoue une allocation sur l'historique réel : construit la courbe de capital du portefeuille
(buy & hold ou rééquilibré), la série de drawdown, et les métriques de performance/risque
(rendement total, CAGR, volatilité annualisée, Sharpe, max drawdown, VaR/CVaR historiques),
plus la comparaison à un benchmark (alpha, tracking error, information ratio).

Architecture : `compute_backtest` fait tout le calcul à partir de prix déjà fournis
(donc testable avec des données synthétiques) ; `run_backtest` ajoute le téléchargement.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from services.markowitz_service import compute_var_cvar, download_prices_multi

# Facteur d'annualisation pour des données quotidiennes
ANNUALIZE = 252
# Nombre de jours par an (pour le CAGR à partir de dates calendaires)
DAYS_PER_YEAR = 365.25

# Modes de rééquilibrage supportés
REBALANCE_MODES = ("none", "monthly", "quarterly")


def equal_weights(n: int) -> np.ndarray:
    """Poids équipondérés pour n actifs (chacun 1/n)."""
    if n <= 0:
        return np.array([])
    return np.ones(n) / n


def normalize_weights(weights: np.ndarray | list[float]) -> np.ndarray:
    """Normalise des poids pour qu'ils somment à 1 (repli sur équipondéré si somme nulle)."""
    w = np.asarray(weights, dtype=float).ravel()
    total = w.sum()
    if not np.isfinite(total) or abs(total) < 1e-12:
        return equal_weights(len(w))
    return w / total


def _is_new_period(prev: pd.Timestamp, cur: pd.Timestamp, mode: str) -> bool:
    """True si `cur` ouvre une nouvelle période de rééquilibrage par rapport à `prev`."""
    if mode == "monthly":
        return (prev.year, prev.month) != (cur.year, cur.month)
    if mode == "quarterly":
        return (prev.year, (prev.month - 1) // 3) != (cur.year, (cur.month - 1) // 3)
    return False


def portfolio_capital_curve(
    prices: pd.DataFrame,
    weights: np.ndarray,
    initial_capital: float,
    rebalance: str = "none",
) -> pd.Series:
    """
    Construit la courbe de capital du portefeuille à partir des prix.
    - rebalance="none" : buy & hold (les poids dérivent avec la performance).
    - rebalance="monthly"/"quarterly" : on remet les poids cibles au début de chaque période.
    Retourne une Series indexée par les dates de `prices`, démarrant à `initial_capital`.
    """
    if prices is None or prices.shape[0] < 2:
        return pd.Series(dtype=float)
    target_w = normalize_weights(weights)
    # Rendements simples quotidiens (pour la capitalisation du capital)
    returns = prices.pct_change().fillna(0.0)
    index = prices.index

    value = float(initial_capital)
    values = [value]
    w = target_w.copy()
    prev = index[0]
    for i in range(1, len(index)):
        cur = index[i]
        # Rééquilibrage au début d'une nouvelle période : on repart des poids cibles
        if rebalance != "none" and _is_new_period(prev, cur, rebalance):
            w = target_w.copy()
        r = returns.iloc[i].values
        port_ret = float(np.dot(w, r))
        value *= (1.0 + port_ret)
        values.append(value)
        # Dérive des poids jusqu'à la fin de la journée (effet buy & hold intra-période)
        w = w * (1.0 + r)
        s = w.sum()
        if s > 0:
            w = w / s
        prev = cur
    return pd.Series(values, index=index)


def drawdown_series(capital_curve: pd.Series) -> pd.Series:
    """Série de drawdown : (valeur − plus haut atteint) / plus haut, dans [-1, 0]."""
    if capital_curve is None or len(capital_curve) == 0:
        return pd.Series(dtype=float)
    running_max = capital_curve.cummax()
    return capital_curve / running_max - 1.0


def max_drawdown(capital_curve: pd.Series) -> float:
    """Max drawdown (la baisse pic-à-creux la plus sévère), valeur négative (ex. -0.35)."""
    dd = drawdown_series(capital_curve)
    if len(dd) == 0:
        return 0.0
    return float(dd.min())


def cagr(capital_curve: pd.Series) -> float:
    """Taux de croissance annuel composé (CAGR) à partir de la courbe de capital."""
    if capital_curve is None or len(capital_curve) < 2:
        return 0.0
    start_val = float(capital_curve.iloc[0])
    end_val = float(capital_curve.iloc[-1])
    if start_val <= 0 or end_val <= 0:
        return 0.0
    span_days = (capital_curve.index[-1] - capital_curve.index[0]).days
    years = span_days / DAYS_PER_YEAR
    if years <= 0:
        return 0.0
    return float((end_val / start_val) ** (1.0 / years) - 1.0)


def backtest_metrics(capital_curve: pd.Series, risk_free_rate: float) -> dict:
    """
    Métriques de performance/risque à partir de la courbe de capital.
    Retourne : total_return, cagr, annualized_volatility, sharpe_ratio, max_drawdown,
    var_daily, cvar_daily.
    """
    if capital_curve is None or len(capital_curve) < 2:
        return {
            "total_return": 0.0,
            "cagr": 0.0,
            "annualized_volatility": 0.0,
            "sharpe_ratio": 0.0,
            "max_drawdown": 0.0,
            "var_daily": 0.0,
            "cvar_daily": 0.0,
        }
    start_val = float(capital_curve.iloc[0])
    end_val = float(capital_curve.iloc[-1])
    total_return = (end_val / start_val - 1.0) if start_val > 0 else 0.0

    daily_ret = capital_curve.pct_change().dropna().values
    mean_daily = float(np.mean(daily_ret)) if len(daily_ret) else 0.0
    std_daily = float(np.std(daily_ret)) if len(daily_ret) else 0.0
    ann_return = mean_daily * ANNUALIZE
    ann_vol = std_daily * np.sqrt(ANNUALIZE)
    sharpe = (ann_return - risk_free_rate) / ann_vol if ann_vol > 0 else 0.0

    var_daily, cvar_daily = compute_var_cvar(daily_ret, confidence=0.95)

    return {
        "total_return": float(total_return),
        "cagr": cagr(capital_curve),
        "annualized_volatility": float(ann_vol),
        "sharpe_ratio": float(sharpe),
        "max_drawdown": max_drawdown(capital_curve),
        "var_daily": float(var_daily),
        "cvar_daily": float(cvar_daily),
    }


def _benchmark_comparison(
    port_curve: pd.Series,
    bench_curve: pd.Series,
    port_metrics: dict,
) -> dict:
    """Alpha, tracking error et information ratio du portefeuille vs benchmark (annualisés)."""
    common = port_curve.index.intersection(bench_curve.index)
    if len(common) < 2:
        return {}
    port_ret = port_curve.loc[common].pct_change().dropna()
    bench_ret = bench_curve.loc[common].pct_change().dropna()
    common_ret = port_ret.index.intersection(bench_ret.index)
    if len(common_ret) < 2:
        return {}
    p = port_ret.loc[common_ret].values
    b = bench_ret.loc[common_ret].values
    port_ann = float(np.mean(p)) * ANNUALIZE
    bench_ann = float(np.mean(b)) * ANNUALIZE
    diff = p - b
    tracking_error = float(np.std(diff)) * np.sqrt(ANNUALIZE)
    alpha = port_ann - bench_ann
    information_ratio = alpha / tracking_error if tracking_error > 0 else 0.0
    return {
        "benchmark_alpha": alpha,
        "tracking_error": tracking_error,
        "information_ratio": information_ratio,
    }


def compute_backtest(
    prices: pd.DataFrame,
    weights: np.ndarray | None,
    initial_capital: float,
    rebalance: str,
    risk_free_rate: float,
    benchmark_prices: pd.Series | None = None,
) -> dict:
    """
    Calcul complet du backtest à partir de prix déjà téléchargés (pur, testable).
    `prices` : DataFrame (colonnes = actifs). `benchmark_prices` : Series optionnelle.
    Retourne un dict avec success, capital_curve, drawdown, metrics, weights, et
    (si benchmark) benchmark_curve + benchmark_metrics.
    """
    if prices is None or prices.shape[0] < 2 or prices.shape[1] < 1:
        return {"success": False, "error": "Historique de prix insuffisant."}
    if rebalance not in REBALANCE_MODES:
        return {"success": False, "error": f"Mode de rééquilibrage inconnu : {rebalance}."}

    n = prices.shape[1]
    if weights is None:
        w = equal_weights(n)
    else:
        w = normalize_weights(weights)
        if len(w) != n:
            return {"success": False, "error": "Le nombre de poids ne correspond pas au nombre d'actifs."}

    port_curve = portfolio_capital_curve(prices, w, initial_capital, rebalance)
    if len(port_curve) < 2:
        return {"success": False, "error": "Impossible de construire la courbe de capital."}

    dd = drawdown_series(port_curve)
    metrics = backtest_metrics(port_curve, risk_free_rate)

    out = {
        "success": True,
        "error": None,
        "tickers": list(prices.columns),
        "weights": w,
        "rebalance": rebalance,
        "initial_capital": float(initial_capital),
        "capital_curve": port_curve,
        "drawdown": dd,
        "metrics": metrics,
        "benchmark_curve": None,
        "benchmark_metrics": None,
    }

    if benchmark_prices is not None and len(benchmark_prices) >= 2:
        common = prices.index.intersection(benchmark_prices.index)
        if len(common) >= 2:
            bench_aligned = benchmark_prices.loc[common]
            # Courbe de capital du benchmark, même capital de départ
            bench_curve = bench_aligned / float(bench_aligned.iloc[0]) * float(initial_capital)
            out["benchmark_curve"] = bench_curve
            out["benchmark_metrics"] = _benchmark_comparison(port_curve, bench_curve, metrics)

    return out


def _download_close_series(
    ticker: str,
    start_date: pd.Timestamp,
    end_date: pd.Timestamp,
) -> pd.Series | None:
    """Télécharge la série de clôtures ajustées d'un ticker (benchmark). None en cas d'échec."""
    if not ticker or not ticker.strip():
        return None
    try:
        import yfinance as yf

        df = yf.download(
            ticker.strip().upper(),
            start=start_date,
            end=end_date,
            progress=False,
            auto_adjust=True,
            threads=False,
        )
    except Exception:
        return None
    if df is None or df.empty or len(df) < 2:
        return None
    close = df["Close"] if "Close" in df.columns else df.iloc[:, 3]
    if isinstance(close, pd.DataFrame):
        close = close.squeeze()
    s = pd.Series(np.asarray(close).ravel(), index=df.index).dropna()
    return s if len(s) >= 2 else None


def run_backtest(
    tickers: list[str],
    weights: np.ndarray | None,
    start_date: pd.Timestamp,
    end_date: pd.Timestamp,
    initial_capital: float,
    rebalance: str,
    risk_free_rate: float,
    benchmark_ticker: str | None = None,
) -> dict:
    """
    Pipeline complet : télécharge l'historique (réutilise download_prices_multi), puis calcule
    le backtest. Retourne le même dict que compute_backtest (ou success=False + error).
    """
    dl = download_prices_multi(tickers, start_date, end_date)
    if not dl["success"]:
        return {"success": False, "error": dl["error"]}
    prices = dl["prices"]

    benchmark_prices = None
    if benchmark_ticker and benchmark_ticker.strip():
        benchmark_prices = _download_close_series(benchmark_ticker, start_date, end_date)

    return compute_backtest(
        prices=prices,
        weights=weights,
        initial_capital=initial_capital,
        rebalance=rebalance,
        risk_free_rate=risk_free_rate,
        benchmark_prices=benchmark_prices,
    )
