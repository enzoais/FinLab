"""Onglet Portfolio (Markowitz) — refonte claire & pro."""
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from config import DEFAULT_MARKOWITZ_TICKERS, DEFAULT_RISK_FREE_RATE
from services import capm_service, markowitz_service
from utils.formatters import format_pct
from utils.plot_config import apply_theme, COLOR_PRIMARY, COLOR_NEGATIVE, COLOR_BENCHMARK, COLOR_POSITIVE
from utils.ui import kpi_row, section_header, advanced_expander, show_data_error, info_inline, metric_with_info


@st.cache_data(ttl=3600)
def _load_markowitz_results(tickers, start_str, end_str, rf, objective, target_return, long_only, max_w, benchmark_ticker):
    """Wrapper caché : pipeline Markowitz complet."""
    return markowitz_service.run_markowitz_analysis(
        tickers=list(tickers), start_date=pd.Timestamp(start_str), end_date=pd.Timestamp(end_str),
        risk_free_rate=rf, objective=objective, target_return=target_return, long_only=long_only,
        max_weight_per_asset=max_w, benchmark_ticker=benchmark_ticker or None,
        n_frontier_points=50, n_random_portfolios=1,
    )


def render():
    """Affiche l'onglet Portfolio (Markowitz)."""
    section_header("Portefeuille optimal (Markowitz)", "markowitz", level="header")
    st.caption("Trouve la meilleure répartition entre plusieurs actifs : frontière efficiente et ratio de Sharpe.")

    # ----- Paramètres -----
    with st.expander("Paramètres", expanded=True):
        c1, c2 = st.columns(2)
        with c1:
            default_tickers = "\n".join(DEFAULT_MARKOWITZ_TICKERS)
            tickers_text = st.text_area(
                "Actifs (un ticker ou nom par ligne)", value=default_tickers,
                help="Un ticker (AAPL) ou nom (Nvidia) par ligne. Au moins 2 actifs.", height=120,
                key="pf_tickers",
            ).strip() or default_tickers
            tickers, seen = [], set()
            for q in [t.strip() for t in tickers_text.splitlines() if t.strip()]:
                sym = (capm_service.resolve_asset_ticker(q)[0] or q).strip().upper()
                if sym and sym not in seen:
                    tickers.append(sym)
                    seen.add(sym)
            if not tickers:
                tickers = list(DEFAULT_MARKOWITZ_TICKERS)
            risk_free_rate = st.number_input("Taux sans risque", 0.0, 0.30, float(DEFAULT_RISK_FREE_RATE), 0.005, format="%.3f", key="pf_rf")
            period_preset = st.selectbox("Période", ["5 ans", "3 ans", "2 ans", "1 an"], index=0, key="pf_period")
            years = {"5 ans": 5, "3 ans": 3, "2 ans": 2, "1 an": 1}[period_preset]
            end_dt = pd.Timestamp.now()
            start_str = (end_dt - pd.DateOffset(years=years)).strftime("%Y-%m-%d")
            end_str = end_dt.strftime("%Y-%m-%d")
        with c2:
            obj_opts = ["max_sharpe", "min_variance", "target_return"]
            obj_labels = {"max_sharpe": "Maximiser le Sharpe", "min_variance": "Minimiser la variance", "target_return": "Rendement cible"}
            objective = st.selectbox("Objectif d'optimisation", obj_opts, index=0, format_func=lambda o: obj_labels[o], key="pf_objective")
            target_return = None
            if objective == "target_return":
                target_return = st.number_input("Rendement cible annuel (%)", 0.0, 50.0, 10.0, 0.5, format="%.1f", key="pf_target") / 100.0
            benchmark_label = st.selectbox("Benchmark (comparaison)", ["Aucun"] + list(capm_service.BENCHMARK_OPTIONS.keys()), index=0, key="pf_benchmark")
            benchmark_ticker = None if benchmark_label == "Aucun" else capm_service.BENCHMARK_OPTIONS[benchmark_label]
            with st.expander("Contraintes", expanded=False):
                long_only = st.checkbox("Long-only (pas de vente à découvert)", value=True, key="pf_long_only")
                max_weight_pct = st.number_input("Poids max par actif (%)", 5, 100, 100, 5, key="pf_max_weight")
            max_weight_per_asset = max_weight_pct / 100.0
    st.divider()

    with st.spinner("Chargement et optimisation du portefeuille…"):
        result = _load_markowitz_results(
            tuple(tickers), start_str, end_str, risk_free_rate, objective, target_return,
            long_only, max_weight_per_asset, benchmark_ticker,
        )

    if not result["success"]:
        show_data_error(result.get("error"))
        return

    tickers_used = result["tickers"]
    weights = result["weights"]
    var_95 = result.get("var_95_daily")

    st.info(f"**Actifs :** {', '.join(tickers_used)} · {start_str} → {end_str} · {result['n_obs']} observations")

    # ----- KPI en tête -----
    kpi_row([
        ("Ratio de Sharpe", f"{result['sharpe_ratio']:.2f}", "sharpe_ratio"),
        ("Rendement attendu", format_pct(result["expected_return"]), "expected_return"),
        ("Volatilité", format_pct(result["volatility"]), "portfolio_volatility"),
        ("VaR 95 % (1 j)", f"{var_95 * 100:.2f} %" if var_95 is not None else "—", "var"),
    ])

    # Allocation recommandée (sortie centrale de Markowitz)
    st.markdown("")
    alloc_row = st.columns([0.9, 0.1])
    with alloc_row[0]:
        st.markdown("**Allocation optimale**")
    with alloc_row[1]:
        info_inline("optimal_weights", "Poids optimaux")
    df_weights = pd.DataFrame({"Actif": tickers_used, "Poids (%)": np.round(weights * 100, 2)})
    st.dataframe(df_weights, use_container_width=True, hide_index=True)

    # ----- Graphe héros : frontière efficiente -----
    st.markdown("")
    hero_row = st.columns([0.9, 0.1])
    with hero_row[0]:
        st.subheader("Frontière efficiente")
    with hero_row[1]:
        info_inline("efficient_frontier", "Frontière efficiente")

    frontier = result["frontier_curve"]
    fig = go.Figure()
    if frontier is not None and len(frontier) > 0:
        fig.add_trace(go.Scatter(
            x=frontier[:, 0] * 100, y=frontier[:, 1] * 100, mode="lines", name="Frontière efficiente",
            line=dict(color=COLOR_PRIMARY, width=2.5),
            hovertemplate="Volatilité : %{x:.2f}%<br>Rendement : %{y:.2f}%<extra></extra>",
        ))
    if result.get("benchmark_vol_annual") is not None:
        fig.add_trace(go.Scatter(
            x=[result["benchmark_vol_annual"] * 100], y=[result["benchmark_return_annual"] * 100],
            mode="markers", name="Benchmark",
            marker=dict(size=13, color=COLOR_BENCHMARK, symbol="triangle-up", line=dict(width=1.5, color="white")),
            hovertemplate="Benchmark<br>Vol : %{x:.2f}%<br>Rdt : %{y:.2f}%<extra></extra>",
        ))
    fig.add_trace(go.Scatter(
        x=[result["volatility"] * 100], y=[result["expected_return"] * 100], mode="markers",
        name="Portefeuille optimal",
        marker=dict(size=17, color=COLOR_NEGATIVE, symbol="star", line=dict(width=1.5, color="white")),
        hovertemplate="Optimal<br>Vol : %{x:.2f}%<br>Rdt : %{y:.2f}%<extra></extra>",
    ))
    fig.update_layout(title="Frontière efficiente (annualisée)", xaxis_title="Volatilité (%)", yaxis_title="Rendement attendu (%)")
    apply_theme(fig)
    st.plotly_chart(fig, use_container_width=True)

    # ----- Avancé -----
    with advanced_expander():
        # Répartition en barres
        st.markdown("**Répartition des poids**")
        bar_colors = [COLOR_POSITIVE if w >= 0 else COLOR_NEGATIVE for w in weights]
        fig_w = go.Figure(go.Bar(
            x=weights * 100, y=tickers_used, orientation="h", marker=dict(color=bar_colors),
            text=[f"{w * 100:.1f}%" for w in weights], textposition="outside",
            hovertemplate="%{y} : %{x:.1f}%<extra></extra>",
        ))
        fig_w.update_layout(title="Poids du portefeuille optimal", xaxis_title="Poids (%)", yaxis_title="")
        apply_theme(fig_w, height=max(240, 44 * len(tickers_used)), legend=False)
        st.plotly_chart(fig_w, use_container_width=True)

        # Risque de queue
        st.markdown("**Risque de baisse**")
        cvar_95 = result.get("cvar_95_daily")
        rc1, rc2 = st.columns(2)
        with rc1:
            metric_with_info("VaR 95 % (1 j)", f"{var_95 * 100:.2f} %" if var_95 is not None else "—", "var")
        with rc2:
            metric_with_info("CVaR 95 % (1 j)", f"{cvar_95 * 100:.2f} %" if cvar_95 is not None else "—", "cvar")

        # Comparaison benchmark
        if result.get("benchmark_return_annual") is not None:
            st.markdown("**Comparaison au benchmark**")
            b1, b2, b3 = st.columns(3)
            with b1:
                metric_with_info("Alpha", format_pct(result["alpha_vs_benchmark"]), "alpha")
            with b2:
                metric_with_info("Tracking error", format_pct(result["tracking_error_annual"]), "tracking_error")
            with b3:
                metric_with_info("Ratio d'information", f"{result['information_ratio']:.2f}", "information_ratio")
