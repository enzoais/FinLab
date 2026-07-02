"""Onglet Beta (CAPM) — refonte claire & pro."""
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from config import DEFAULT_ASSET_TICKER, DEFAULT_RISK_FREE_RATE
from services import capm_service
from utils.formatters import format_pct
from utils.plot_config import apply_theme, COLOR_PRIMARY, COLOR_NEGATIVE, COLOR_ACCENT
from utils.ui import kpi_row, section_header, advanced_expander, show_data_error, info_inline


@st.cache_data(ttl=3600)
def _load_capm_results(asset_ticker: str, benchmark_ticker: str, start_str: str, end_str: str, rf: float, window: int):
    """Wrapper caché : pipeline CAPM complet (téléchargement + régression + bêta glissant)."""
    return capm_service.run_capm_analysis(
        asset_ticker=asset_ticker,
        benchmark_ticker=benchmark_ticker,
        start_date=pd.Timestamp(start_str),
        end_date=pd.Timestamp(end_str),
        risk_free_rate=rf,
        rolling_window=window,
    )


def render():
    """Affiche l'onglet Beta (CAPM)."""
    section_header("CAPM & Bêta", "capm", level="header")
    st.caption("Mesure le risque de marché d'une action (Bêta), son coût des fonds propres et sa surperformance (Alpha).")

    # ----- Paramètres -----
    with st.expander("Paramètres", expanded=True):
        c1, c2, c3 = st.columns(3)
        with c1:
            asset_input = st.text_input(
                "Action (ticker ou nom)",
                value=DEFAULT_ASSET_TICKER,
                placeholder="ex. AAPL, Apple, Microsoft",
                help="Saisissez un ticker (AAPL) ou un nom (Apple). Le nom est résolu vers le ticker principal.",
                key="beta_asset",
            ).strip() or DEFAULT_ASSET_TICKER
            resolved_ticker, resolved_name, resolved_exchange = capm_service.resolve_asset_ticker(asset_input)
            asset_ticker = resolved_ticker if resolved_ticker else asset_input
            benchmark_label = st.selectbox("Benchmark", options=list(capm_service.BENCHMARK_OPTIONS.keys()), index=0, key="beta_benchmark")
            benchmark_ticker = capm_service.BENCHMARK_OPTIONS[benchmark_label]
        with c2:
            risk_free_rate = st.number_input(
                "Taux sans risque", min_value=0.0, max_value=0.30, value=float(DEFAULT_RISK_FREE_RATE),
                step=0.005, format="%.3f", help="Taux annuel, ex. 0.05 pour 5 %.", key="beta_rf",
            )
            period_preset = st.selectbox("Période", options=["5 ans", "3 ans", "2 ans", "1 an"], index=0, key="beta_period")
            years = {"5 ans": 5, "3 ans": 3, "2 ans": 2, "1 an": 1}[period_preset]
            end = pd.Timestamp.now()
            start = end - pd.DateOffset(years=years)
            start_str, end_str = start.strftime("%Y-%m-%d"), end.strftime("%Y-%m-%d")
        with c3:
            rolling_window = st.selectbox(
                "Fenêtre du Bêta glissant", options=capm_service.ROLLING_WINDOWS, index=2,
                format_func=lambda x: f"{x} jours", help="252 jours ≈ 1 an.", key="beta_window",
            )
    st.divider()

    with st.spinner("Chargement des données de marché…"):
        result = _load_capm_results(asset_ticker, benchmark_ticker, start_str, end_str, risk_free_rate, rolling_window)

    if not result["success"]:
        show_data_error(result.get("error"))
        return

    reg = result["regression"]
    ke = result["cost_of_equity"]
    asset_ret = result["asset_returns"]
    market_ret = result["market_returns"]
    rolling_beta_series = result["rolling_beta"]

    # Contexte compact
    name = resolved_name or capm_service.get_asset_display_info(asset_ticker)[0]
    label = f"{name} ({asset_ticker})" if name else asset_ticker
    st.info(f"**{label}** vs **{benchmark_label}** · {start_str} → {end_str}")

    # ----- 3-4 KPI en tête -----
    alpha_ann_pct = reg["alpha"] * 252 * 100
    kpi_row([
        ("Bêta (β)", f"{reg['beta']:.2f}", "beta"),
        ("Coût des fonds propres", format_pct(ke), "cost_of_equity"),
        ("Alpha de Jensen (annualisé)", f"{alpha_ann_pct:+.2f} %", "jensen_alpha"),
        ("R²", f"{reg['r_squared'] * 100:.1f} %", "r_squared"),
    ])

    if reg["r_squared"] < 0.2:
        st.caption("ℹ️ R² faible : le marché explique peu les mouvements de l'action, le Bêta est moins fiable.")

    # ----- Graphe héros : nuage rendements + droite de régression -----
    st.markdown("")
    row = st.columns([0.9, 0.1])
    with row[0]:
        st.subheader("Rendements quotidiens : action vs marché")
    with row[1]:
        info_inline("regression_line", "Droite de régression (CAPM)")

    excess_asset = asset_ret - risk_free_rate / 252
    excess_market = market_ret - risk_free_rate / 252
    common_idx = excess_asset.index.intersection(excess_market.index)
    x_scatter = excess_market.loc[common_idx].values * 100
    y_scatter = excess_asset.loc[common_idx].values * 100
    x_line = reg["fitted_excess_market"].values * 100
    y_line = reg["fitted_excess_asset"].values * 100
    order = np.argsort(x_line)

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=x_scatter, y=y_scatter, mode="markers", name="Rendements en excès",
        marker=dict(size=4, color="rgba(37, 99, 235, 0.45)", line=dict(width=0)),
        hovertemplate="Marché : %{x:.2f}%<br>Action : %{y:.2f}%<extra></extra>",
    ))
    fig.add_trace(go.Scatter(
        x=x_line[order], y=y_line[order], mode="lines", name="Régression CAPM",
        line=dict(color=COLOR_NEGATIVE, width=2.5),
        hovertemplate="Marché : %{x:.2f}%<br>Ajusté : %{y:.2f}%<extra></extra>",
    ))
    fig.update_layout(title="Rendements quotidiens en excès", xaxis_title="Marché (%)", yaxis_title="Action (%)")
    apply_theme(fig)
    st.plotly_chart(fig, use_container_width=True)

    # ----- Avancé -----
    with advanced_expander():
        row2 = st.columns([0.9, 0.1])
        with row2[0]:
            st.markdown(f"**Bêta glissant ({rolling_window} jours)**")
        with row2[1]:
            info_inline("rolling_beta", "Bêta glissant")
        if rolling_beta_series is not None and len(rolling_beta_series) > 0:
            fig2 = go.Figure()
            fig2.add_trace(go.Scatter(
                x=rolling_beta_series.index, y=rolling_beta_series.values, mode="lines",
                name="Bêta glissant", line=dict(color=COLOR_ACCENT, width=2),
                hovertemplate="%{x|%Y-%m-%d}<br>Bêta : %{y:.2f}<extra></extra>",
            ))
            fig2.add_hline(y=1.0, line_dash="dash", line_color="gray", opacity=0.6)
            fig2.update_layout(title=f"Bêta glissant ({rolling_window} jours)", xaxis_title="Date", yaxis_title="Bêta")
            apply_theme(fig2, height=320)
            st.plotly_chart(fig2, use_container_width=True)
        else:
            st.info("Pas assez de données pour le Bêta glissant sur cette fenêtre.")
