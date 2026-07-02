"""
Risque d'investissement (fonds / mandat) - Business logic (pur, testable).

Couvre : VaR par 3 méthodes (historique, paramétrique variance-covariance, Monte-Carlo) + CVaR,
stress tests (chocs de marché via le bêta du portefeuille), concentration (HHI, positions
effectives, top-N), contribution au risque par ligne, et backtesting de la VaR (test de Kupiec).

Réutilise le téléchargement multi-actifs et la décomposition du risque de markowitz_service.
`run_risk_analysis` télécharge ; toutes les autres fonctions sont pures (testables hors réseau).
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.stats import norm, chi2, skew, kurtosis

from services.markowitz_service import download_prices_multi, risk_decomposition
from services.backtest_service import equal_weights, normalize_weights, _download_close_series

ANNUALIZE = 252

# Scénarios de stress hypothétiques : choc du marché actions (rendement), appliqué via le bêta
DEFAULT_STRESS_SCENARIOS: list[tuple[str, float]] = [
    ("Correction marché −10 %", -0.10),
    ("Choc sévère −20 %", -0.20),
    ("COVID mars 2020 (−34 %)", -0.34),
    ("Krach type 2008 (−40 %)", -0.40),
]


# ----------------------------------------------------------------- VaR / CVaR
def var_historical(returns: np.ndarray, confidence: float = 0.95) -> tuple[float, float]:
    """VaR et CVaR historiques (rendements quotidiens). Retourne (var, cvar), négatifs = perte."""
    r = np.asarray(returns, dtype=float).ravel()
    if len(r) < 2:
        return (0.0, 0.0)
    q = (1 - confidence) * 100
    var = float(np.percentile(r, q))
    tail = r[r <= var]
    cvar = float(np.mean(tail)) if len(tail) > 0 else var
    return (var, cvar)


def var_parametric(returns: np.ndarray, confidence: float = 0.95) -> tuple[float, float]:
    """
    VaR et CVaR paramétriques (variance-covariance, hypothèse normale).
    VaR = μ + z·σ ; CVaR (Expected Shortfall) = μ − σ·φ(z)/α, avec α = 1−confiance.
    Retourne (var, cvar) en rendement (négatifs = perte).
    """
    r = np.asarray(returns, dtype=float).ravel()
    if len(r) < 2:
        return (0.0, 0.0)
    mu = float(np.mean(r))
    sigma = float(np.std(r))
    alpha = 1 - confidence
    z = norm.ppf(alpha)  # négatif (ex. -1.645 à 95 %)
    var = mu + z * sigma
    cvar = mu - sigma * norm.pdf(z) / alpha
    return (float(var), float(cvar))


def var_monte_carlo(
    returns: np.ndarray, confidence: float = 0.95, n_sims: int = 50_000, seed: int | None = 42
) -> tuple[float, float]:
    """
    VaR et CVaR Monte-Carlo : on ajuste une loi normale (μ, σ) aux rendements et on simule.
    Retourne (var, cvar) en rendement (négatifs = perte).
    """
    r = np.asarray(returns, dtype=float).ravel()
    if len(r) < 2:
        return (0.0, 0.0)
    mu = float(np.mean(r))
    sigma = float(np.std(r))
    if seed is not None:
        np.random.seed(seed)
    sims = np.random.normal(mu, sigma, n_sims)
    return var_historical(sims, confidence)


def var_cornish_fisher(returns: np.ndarray, confidence: float = 0.95) -> tuple[float, float]:
    """
    VaR modifiée (Cornish-Fisher) : corrige le quantile normal par l'asymétrie (skewness) et
    l'aplatissement (kurtosis) des rendements → capte mieux les queues épaisses.
    CVaR estimée par la moyenne des rendements sous le seuil corrigé. Retourne (var, cvar).
    """
    r = np.asarray(returns, dtype=float).ravel()
    if len(r) < 4:
        return (0.0, 0.0)
    mu, sigma = float(np.mean(r)), float(np.std(r))
    s = float(skew(r))
    k = float(kurtosis(r))  # excès de kurtosis (Fisher : 0 pour une normale)
    z = norm.ppf(1 - confidence)  # négatif
    z_cf = (
        z
        + (z ** 2 - 1) / 6 * s
        + (z ** 3 - 3 * z) / 24 * k
        - (2 * z ** 3 - 5 * z) / 36 * s ** 2
    )
    var = mu + z_cf * sigma
    tail = r[r <= var]
    cvar = float(np.mean(tail)) if len(tail) > 0 else var
    return (float(var), float(cvar))


def ewma_volatility(returns: np.ndarray, lam: float = 0.94) -> float:
    """Volatilité EWMA (RiskMetrics) : σ²ₜ = λ·σ²ₜ₋₁ + (1−λ)·r²ₜ₋₁ (plus de poids au récent)."""
    r = np.asarray(returns, dtype=float).ravel()
    if len(r) < 2:
        return 0.0
    var = float(np.var(r))  # amorçage sur la variance d'échantillon
    for x in r:
        var = lam * var + (1 - lam) * x ** 2
    return float(np.sqrt(var))


def var_ewma(returns: np.ndarray, confidence: float = 0.95, lam: float = 0.94) -> tuple[float, float]:
    """
    VaR EWMA / RiskMetrics : VaR paramétrique utilisant la volatilité EWMA, moyenne supposée nulle.
    Retourne (var, cvar) en rendement (négatifs = perte).
    """
    r = np.asarray(returns, dtype=float).ravel()
    if len(r) < 2:
        return (0.0, 0.0)
    sigma = ewma_volatility(r, lam)
    alpha = 1 - confidence
    z = norm.ppf(alpha)
    var = z * sigma
    cvar = -sigma * norm.pdf(z) / alpha
    return (float(var), float(cvar))


def var_decomposition(weights: np.ndarray, cov: np.ndarray, confidence: float, aum: float) -> dict:
    """
    Décomposition de la VaR paramétrique (part liée à la volatilité) par position.
    - Marginal VaR : sensibilité de la VaR à une hausse marginale du poids d'une ligne.
    - Component VaR : contribution de chaque ligne (Σ = VaR totale).
    - Incremental VaR : effet de retirer complètement la ligne (poids restants renormalisés).
    Retourne des tableaux (en € et en % pour la Component VaR).
    """
    w = np.asarray(weights, dtype=float).ravel()
    sigma_mat = np.asarray(cov, dtype=float)
    n = len(w)
    sigma_p = float(np.sqrt(w @ sigma_mat @ w))
    z = norm.ppf(1 - confidence)  # négatif
    if sigma_p <= 0:
        zeros = np.zeros(n)
        return {"marginal_var": zeros, "component_var_eur": zeros, "component_var_pct": zeros, "incremental_var_eur": zeros}
    total_var = z * sigma_p  # rendement (négatif), part « volatilité » de la VaR
    marginal = z * (sigma_mat @ w) / sigma_p          # marginal VaR par unité de poids
    component = w * marginal                            # Σ = total_var
    component_pct = component / total_var * 100.0       # somme = 100 %

    incremental = np.zeros(n)
    for i in range(n):
        w2 = np.delete(w, i)
        s = w2.sum()
        if s <= 0:
            incremental[i] = 0.0
            continue
        w2 = w2 / s
        cov2 = np.delete(np.delete(sigma_mat, i, axis=0), i, axis=1)
        sigma2 = float(np.sqrt(w2 @ cov2 @ w2))
        var2 = z * sigma2
        incremental[i] = total_var - var2  # négatif = la ligne ajoute du risque

    return {
        "marginal_var": marginal,
        "component_var_eur": component * aum,
        "component_var_pct": component_pct,
        "incremental_var_eur": incremental * aum,
    }


def scale_to_horizon(var_1d: float, horizon_days: int) -> float:
    """Passe d'une VaR 1 jour à un horizon de h jours par la règle racine du temps (√h)."""
    return var_1d * np.sqrt(max(1, int(horizon_days)))


# ----------------------------------------------------------------- Concentration
def concentration_metrics(weights: np.ndarray) -> dict:
    """
    Indicateurs de concentration à partir des poids (en valeur absolue, normalisés).
    HHI (Herfindahl) = Σ wᵢ² ; positions effectives = 1/HHI ; poids du top-1 et top-3.
    """
    w = np.abs(np.asarray(weights, dtype=float).ravel())
    total = w.sum()
    if total <= 0:
        return {"hhi": 0.0, "effective_n": 0.0, "top1": 0.0, "top3": 0.0}
    w = w / total
    hhi = float(np.sum(w ** 2))
    effective_n = float(1.0 / hhi) if hhi > 0 else 0.0
    sorted_w = np.sort(w)[::-1]
    top1 = float(sorted_w[0])
    top3 = float(np.sum(sorted_w[:3]))
    return {"hhi": hhi, "effective_n": effective_n, "top1": top1, "top3": top3}


def component_risk_pct(weights: np.ndarray, cov: np.ndarray) -> np.ndarray:
    """Contribution de chaque ligne au risque total du portefeuille, en % (somme = 100 %)."""
    contrib = risk_decomposition(np.asarray(weights, dtype=float), np.asarray(cov, dtype=float))
    total = contrib.sum()
    if total <= 0:
        return np.zeros_like(contrib)
    return contrib / total * 100.0


# ----------------------------------------------------------------- Stress tests
def portfolio_beta(port_returns: np.ndarray, bench_returns: np.ndarray) -> float:
    """Bêta du portefeuille vs benchmark (covariance / variance du benchmark)."""
    p = np.asarray(port_returns, dtype=float).ravel()
    b = np.asarray(bench_returns, dtype=float).ravel()
    n = min(len(p), len(b))
    if n < 2:
        return 1.0
    p, b = p[-n:], b[-n:]
    c = np.cov(p, b)  # estimateurs cohérents (même ddof) pour cov et variance
    var_b = float(c[1, 1])
    if var_b <= 0:
        return 1.0
    return float(c[0, 1] / var_b)


def stress_scenarios(
    beta: float, aum: float, scenarios: list[tuple[str, float]] | None = None
) -> list[dict]:
    """
    P&L estimé du portefeuille pour des chocs de marché hypothétiques, via le bêta.
    P&L% = bêta × choc ; P&L€ = P&L% × encours. Retourne une liste de dicts.
    """
    scenarios = scenarios or DEFAULT_STRESS_SCENARIOS
    rows = []
    for name, shock in scenarios:
        pnl_pct = beta * shock
        rows.append({
            "name": name,
            "market_shock": shock,
            "pnl_pct": float(pnl_pct),
            "pnl_eur": float(pnl_pct * aum),
        })
    return rows


def historical_worst(port_returns: np.ndarray, aum: float) -> dict:
    """Pire perte réellement observée sur 1 jour et sur 5 jours (stress historique), en % et €."""
    r = np.asarray(port_returns, dtype=float).ravel()
    if len(r) < 1:
        return {"worst_1d_pct": 0.0, "worst_1d_eur": 0.0, "worst_5d_pct": 0.0, "worst_5d_eur": 0.0}
    worst_1d = float(np.min(r))
    if len(r) >= 5:
        rolling_5d = np.array([np.sum(r[i:i + 5]) for i in range(len(r) - 4)])
        worst_5d = float(np.min(rolling_5d))
    else:
        worst_5d = float(np.sum(r))
    return {
        "worst_1d_pct": worst_1d,
        "worst_1d_eur": worst_1d * aum,
        "worst_5d_pct": worst_5d,
        "worst_5d_eur": worst_5d * aum,
    }


# ----------------------------------------------------------------- Backtesting VaR (Kupiec)
def kupiec_test(returns: np.ndarray, var_level: float, confidence: float = 0.99) -> dict:
    """
    Backtesting de la VaR (test de Kupiec / POF) : compare le nombre de dépassements observés
    (rendement < niveau de VaR) au nombre attendu (1−confiance)·N.
    Retourne n, expected, observed, ratio, lr_stat, p_value, passed (True si non rejeté à 95 %).
    """
    r = np.asarray(returns, dtype=float).ravel()
    n = len(r)
    p = 1 - confidence  # taux d'exception attendu
    if n < 2:
        return {"n": n, "expected": 0.0, "observed": 0, "ratio": 0.0, "lr_stat": 0.0, "p_value": 1.0, "passed": True}
    observed = int(np.sum(r < var_level))
    expected = p * n
    pi = observed / n if n > 0 else 0.0

    # Log-vraisemblance sous H0 (taux p) vs H1 (taux observé pi), avec garde-fous sur les log(0)
    def _safe_ll(rate: float) -> float:
        rate = min(max(rate, 1e-12), 1 - 1e-12)
        return (n - observed) * np.log(1 - rate) + observed * np.log(rate)

    lr_stat = -2.0 * (_safe_ll(p) - _safe_ll(pi))
    lr_stat = float(max(lr_stat, 0.0))
    p_value = float(1 - chi2.cdf(lr_stat, df=1))
    ratio = observed / expected if expected > 0 else 0.0
    return {
        "n": n,
        "expected": float(expected),
        "observed": observed,
        "ratio": float(ratio),
        "lr_stat": lr_stat,
        "p_value": p_value,
        "passed": bool(p_value > 0.05),
    }


# ----------------------------------------------------------------- Orchestrateur
def compute_risk(
    prices: pd.DataFrame,
    weights: np.ndarray | None,
    aum: float,
    confidence: float,
    horizon_days: int,
    benchmark_prices: pd.Series | None = None,
    kupiec_confidence: float = 0.99,
    seed: int | None = 42,
) -> dict:
    """Calcul complet du risque à partir de prix déjà téléchargés (pur, testable)."""
    if prices is None or prices.shape[0] < 2 or prices.shape[1] < 1:
        return {"success": False, "error": "Historique de prix insuffisant."}
    n = prices.shape[1]
    w = equal_weights(n) if weights is None else normalize_weights(weights)
    if len(w) != n:
        return {"success": False, "error": "Le nombre de poids ne correspond pas au nombre d'actifs."}

    returns = prices.pct_change().dropna()
    if len(returns) < 2:
        return {"success": False, "error": "Rendements insuffisants."}
    port_ret = returns.values @ w
    cov = returns.cov().values

    # VaR 3 méthodes (1 jour), puis mise à l'horizon
    var_h, cvar_h = var_historical(port_ret, confidence)
    var_p, cvar_p = var_parametric(port_ret, confidence)
    var_mc, cvar_mc = var_monte_carlo(port_ret, confidence, seed=seed)
    var_cf, cvar_cf = var_cornish_fisher(port_ret, confidence)
    var_ew, cvar_ew = var_ewma(port_ret, confidence)

    def _h(pair):
        return {"var": scale_to_horizon(pair[0], horizon_days), "cvar": scale_to_horizon(pair[1], horizon_days)}

    methods = {
        "historique": _h((var_h, cvar_h)),
        "parametrique": _h((var_p, cvar_p)),
        "monte_carlo": _h((var_mc, cvar_mc)),
        "cornish_fisher": _h((var_cf, cvar_cf)),
        "ewma": _h((var_ew, cvar_ew)),
    }
    ann_vol = float(np.std(port_ret) * np.sqrt(ANNUALIZE))
    return_skew = float(skew(port_ret)) if len(port_ret) >= 4 else 0.0
    return_kurtosis = float(kurtosis(port_ret)) if len(port_ret) >= 4 else 0.0

    # Bêta pour les stress tests (défaut 1.0 sans benchmark)
    beta = 1.0
    if benchmark_prices is not None and len(benchmark_prices) >= 2:
        common = returns.index.intersection(benchmark_prices.pct_change().dropna().index)
        if len(common) >= 2:
            bench_ret = benchmark_prices.pct_change().dropna().loc[common].values
            port_ret_aligned = (returns.loc[common].values @ w)
            beta = portfolio_beta(port_ret_aligned, bench_ret)

    stress = stress_scenarios(beta, aum)
    hist_worst = historical_worst(port_ret, aum)
    concentration = concentration_metrics(w)
    contrib = component_risk_pct(w, cov)
    var_decomp = var_decomposition(w, cov, confidence, aum)
    kupiec = kupiec_test(port_ret, var_historical(port_ret, kupiec_confidence)[0], kupiec_confidence)

    return {
        "success": True,
        "error": None,
        "tickers": list(prices.columns),
        "weights": w,
        "aum": float(aum),
        "confidence": confidence,
        "horizon_days": horizon_days,
        "beta": beta,
        "annualized_volatility": ann_vol,
        "return_skew": return_skew,
        "return_kurtosis": return_kurtosis,
        "var_methods": methods,
        "var_decomposition": var_decomp,
        "stress_scenarios": stress,
        "historical_worst": hist_worst,
        "concentration": concentration,
        "risk_contributions_pct": contrib,
        "kupiec": kupiec,
    }


def run_risk_analysis(
    tickers: list[str],
    weights: np.ndarray | None,
    start_date: pd.Timestamp,
    end_date: pd.Timestamp,
    aum: float,
    confidence: float,
    horizon_days: int,
    benchmark_ticker: str | None = None,
    seed: int | None = 42,
) -> dict:
    """Pipeline complet : télécharge l'historique puis calcule tous les indicateurs de risque."""
    dl = download_prices_multi(tickers, start_date, end_date)
    if not dl["success"]:
        return {"success": False, "error": dl["error"]}
    benchmark_prices = None
    if benchmark_ticker and benchmark_ticker.strip():
        benchmark_prices = _download_close_series(benchmark_ticker, start_date, end_date)
    return compute_risk(
        prices=dl["prices"], weights=weights, aum=aum, confidence=confidence,
        horizon_days=horizon_days, benchmark_prices=benchmark_prices, seed=seed,
    )
