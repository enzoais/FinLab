"""Onglet Backtest — rejoue une allocation sur l'historique réel (moteur de backtest)."""
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from config import DEFAULT_MARKOWITZ_TICKERS, DEFAULT_RISK_FREE_RATE
from services import backtest_service, capm_service, markowitz_service
from utils.formatters import format_pct
from utils.plot_config import apply_theme, COLOR_PRIMARY, COLOR_BENCHMARK, COLOR_NEGATIVE
from utils.ui import kpi_row, section_header, advanced_expander, show_data_error, info_inline, metric_with_info, asset_multiselect

_REBALANCE_MAP = {"Aucun (buy & hold)": "none", "Mensuel": "monthly", "Trimestriel": "quarterly"}


@st.cache_data(ttl=3600)
def _markowitz_weights_cached(tickers, start_str, end_str, rf):
    """Poids max-Sharpe (Markowitz) pour l'option « utiliser les poids Markowitz »."""
    res = markowitz_service.run_markowitz_analysis(
        tickers=list(tickers), start_date=pd.Timestamp(start_str), end_date=pd.Timestamp(end_str),
        risk_free_rate=rf, objective="max_sharpe", target_return=None,
        n_frontier_points=2, n_random_portfolios=1,
    )
    return res["weights"].tolist() if res.get("success") else None


@st.cache_data(ttl=3600)
def _run_backtest_cached(tickers, weights, start_str, end_str, capital, rebalance, rf, benchmark):
    """Wrapper caché : pipeline de backtest complet."""
    w = np.array(weights) if weights is not None else None
    return backtest_service.run_backtest(
        tickers=list(tickers), weights=w, start_date=pd.Timestamp(start_str), end_date=pd.Timestamp(end_str),
        initial_capital=capital, rebalance=rebalance, risk_free_rate=rf, benchmark_ticker=benchmark or None,
    )


def render():
    """Affiche l'onglet Backtest."""
    section_header("Backtest d'une allocation", "backtest", level="header")
    st.caption("Rejoue un portefeuille sur l'historique réel : courbe de capital, drawdown et comparaison au marché.")

    # ----- Paramètres -----
    with st.expander("Paramètres", expanded=True):
        c1, c2 = st.columns(2)
        with c1:
            tickers = asset_multiselect(
                "Actifs du portefeuille", list(DEFAULT_MARKOWITZ_TICKERS), key="bt_tickers",
                help="Cherchez par nom ou ticker (ex. tapez « app » → Apple). Au moins 2 actifs ; tout ticker peut être ajouté librement.",
            )
            period_preset = st.selectbox("Période", ["5 ans", "3 ans", "2 ans", "1 an"], index=0, key="bt_period")
            years = {"5 ans": 5, "3 ans": 3, "2 ans": 2, "1 an": 1}[period_preset]
            end_dt = pd.Timestamp.now()
            start_str = (end_dt - pd.DateOffset(years=years)).strftime("%Y-%m-%d")
            end_str = end_dt.strftime("%Y-%m-%d")
            initial_capital = st.number_input("Capital initial", 100.0, 1e12, 10_000.0, 1000.0, format="%.0f", key="bt_capital")
        with c2:
            weight_mode = st.radio("Poids", ["Équipondéré", "Personnalisés", "Markowitz (max-Sharpe)"], index=0, key="bt_weight_mode")
            rebalance_label = st.selectbox("Rééquilibrage", list(_REBALANCE_MAP.keys()), index=0, key="bt_rebalance")
            rebalance = _REBALANCE_MAP[rebalance_label]
            benchmark_label = st.selectbox("Benchmark", ["Aucun"] + list(capm_service.BENCHMARK_OPTIONS.keys()), index=1, key="bt_benchmark")
            benchmark_ticker = None if benchmark_label == "Aucun" else capm_service.BENCHMARK_OPTIONS[benchmark_label]
            risk_free_rate = st.number_input("Taux sans risque", 0.0, 0.30, float(DEFAULT_RISK_FREE_RATE), 0.005, format="%.3f", key="bt_rf")

        # Poids personnalisés : une entrée par actif
        weights = None
        if weight_mode == "Équipondéré":
            weights = None
            if tickers:
                st.caption(f"Chaque actif reçoit {100 / len(tickers):.1f} % du capital.")
        elif weight_mode == "Personnalisés":
            if tickers:
                st.markdown("**Poids personnalisés (%)** — normalisés automatiquement pour sommer à 100 %.")
                cols = st.columns(min(len(tickers), 4))
                raw = []
                for i, t in enumerate(tickers):
                    with cols[i % len(cols)]:
                        raw.append(st.number_input(t, 0.0, 100.0, round(100 / len(tickers), 1), 1.0, key=f"bt_w_{t}"))
                weights = list(raw)
        elif len(tickers) >= 2:  # Markowitz
            with st.spinner("Optimisation Markowitz (max-Sharpe)…"):
                mk_weights = _markowitz_weights_cached(tuple(tickers), start_str, end_str, risk_free_rate)
            if mk_weights is None:
                st.info("Optimisation Markowitz impossible ; repli sur équipondéré.")
                weights = None
            else:
                weights = mk_weights
                st.caption("Poids issus du portefeuille Markowitz max-Sharpe : " + ", ".join(f"{t} {w * 100:.0f}%" for t, w in zip(tickers, mk_weights)))
    st.divider()

    if len(tickers) < 2:
        st.info("Sélectionnez au moins 2 actifs pour lancer un backtest.")
        return

    weights_tuple = tuple(weights) if weights is not None else None
    with st.spinner("Backtest en cours…"):
        result = _run_backtest_cached(tuple(tickers), weights_tuple, start_str, end_str, initial_capital, rebalance, risk_free_rate, benchmark_ticker)

    if not result["success"]:
        show_data_error(result.get("error"))
        return

    m = result["metrics"]
    st.info(f"**Actifs :** {', '.join(result['tickers'])} · {start_str} → {end_str} · rééquilibrage : {rebalance_label.lower()}")

    # ----- KPI en tête -----
    kpi_row([
        ("Rendement total", format_pct(m["total_return"]), "total_return"),
        ("CAGR (annualisé)", format_pct(m["cagr"]), "cagr"),
        ("Ratio de Sharpe", f"{m['sharpe_ratio']:.2f}", "sharpe_ratio"),
        ("Max drawdown", format_pct(m["max_drawdown"]), "max_drawdown"),
    ])

    # ----- Graphe héros : courbe de capital vs benchmark -----
    st.markdown("")
    hero_row = st.columns([0.9, 0.1])
    with hero_row[0]:
        st.subheader("Courbe de capital")
    with hero_row[1]:
        info_inline("capital_curve", "Courbe de capital")

    curve = result["capital_curve"]
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=curve.index, y=curve.values, mode="lines", name="Portefeuille",
        line=dict(color=COLOR_PRIMARY, width=2.5),
        hovertemplate="%{x|%Y-%m-%d}<br>Capital : %{y:,.0f}<extra></extra>",
    ))
    bench_curve = result.get("benchmark_curve")
    if bench_curve is not None and len(bench_curve) > 0:
        fig.add_trace(go.Scatter(
            x=bench_curve.index, y=bench_curve.values, mode="lines", name=benchmark_label,
            line=dict(color=COLOR_BENCHMARK, width=2, dash="dash"),
            hovertemplate="%{x|%Y-%m-%d}<br>Benchmark : %{y:,.0f}<extra></extra>",
        ))
    fig.update_layout(title="Évolution du capital (base : capital initial)", xaxis_title="Date", yaxis_title="Capital")
    apply_theme(fig)
    st.plotly_chart(fig, use_container_width=True)

    # ----- Avancé -----
    with advanced_expander():
        a1, a2, a3 = st.columns(3)
        with a1:
            metric_with_info("Volatilité annualisée", format_pct(m["annualized_volatility"]), "annualized_volatility")
        with a2:
            metric_with_info("VaR 95 % (1 j)", f"{m['var_daily'] * 100:.2f} %", "var_historical")
        with a3:
            metric_with_info("CVaR 95 % (1 j)", f"{m['cvar_daily'] * 100:.2f} %", "cvar_historical")

        bm = result.get("benchmark_metrics")
        if bm:
            st.markdown("**Comparaison au benchmark**")
            b1, b2, b3 = st.columns(3)
            with b1:
                metric_with_info("Alpha (annualisé)", format_pct(bm["benchmark_alpha"]), "alpha")
            with b2:
                metric_with_info("Tracking error", format_pct(bm["tracking_error"]), "tracking_error")
            with b3:
                metric_with_info("Ratio d'information", f"{bm['information_ratio']:.2f}", "information_ratio")

        # Courbe de drawdown
        dd_row = st.columns([0.9, 0.1])
        with dd_row[0]:
            st.markdown("**Drawdown (baisse depuis le plus haut)**")
        with dd_row[1]:
            info_inline("max_drawdown", "Max drawdown")
        dd = result["drawdown"]
        fig_dd = go.Figure()
        fig_dd.add_trace(go.Scatter(
            x=dd.index, y=dd.values * 100, mode="lines", name="Drawdown", fill="tozeroy",
            line=dict(color=COLOR_NEGATIVE, width=1.5), fillcolor="rgba(220, 38, 38, 0.15)",
            hovertemplate="%{x|%Y-%m-%d}<br>Drawdown : %{y:.1f}%<extra></extra>",
        ))
        fig_dd.update_layout(title="Drawdown du portefeuille", xaxis_title="Date", yaxis_title="Drawdown (%)")
        apply_theme(fig_dd, height=300, legend=False)
        st.plotly_chart(fig_dd, use_container_width=True)

        # Poids réellement utilisés
        st.markdown("**Poids utilisés**")
        df_w = pd.DataFrame({"Actif": result["tickers"], "Poids (%)": np.round(result["weights"] * 100, 2)})
        st.dataframe(df_w, use_container_width=True, hide_index=True)
