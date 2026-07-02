"""Onglet Simulation (Monte Carlo) — refonte claire & pro."""
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from config import (
    DEFAULT_ASSET_TICKER, DEFAULT_MC_DRIFT, DEFAULT_MC_HORIZON_YEARS, DEFAULT_MC_INITIAL_WEALTH,
    DEFAULT_MC_PATHS, DEFAULT_MC_STEPS_PER_YEAR, DEFAULT_MC_VOLATILITY,
)
from services import capm_service, monte_carlo_service
from utils.formatters import format_pct
from utils.plot_config import apply_theme, COLOR_PRIMARY, COLOR_NEGATIVE
from utils.ui import kpi_row, section_header, advanced_expander, info_inline, metric_with_info


@st.cache_data(ttl=3600)
def _run_monte_carlo_cached(S0, mu, sigma, T_years, n_paths, n_steps_per_year, confidence, seed, annual_flow, shortfall_threshold):
    """Wrapper caché : simulation Monte-Carlo complète."""
    return monte_carlo_service.run_monte_carlo_analysis(
        S0=S0, mu=mu, sigma=sigma, T_years=T_years, n_paths=n_paths, n_steps_per_year=n_steps_per_year,
        confidence=confidence, seed=seed, annual_flow=annual_flow, shortfall_threshold=shortfall_threshold,
    )


def render():
    """Affiche l'onglet Simulation (Monte Carlo)."""
    section_header("Simulation Monte-Carlo", "monte_carlo", level="header")
    st.caption("Projette un patrimoine sous incertitude (mouvement brownien géométrique) : éventail des scénarios et risque de baisse.")

    # ----- Paramètres -----
    with st.expander("Paramètres", expanded=True):
        c1, c2 = st.columns(2)
        with c1:
            param_source = st.radio("Paramètres μ et σ", ["Manuel (μ, σ)", "Depuis un ticker"], index=0, key="mc_param_source")
            mu_input, sigma_input, ticker_used = float(DEFAULT_MC_DRIFT), float(DEFAULT_MC_VOLATILITY), None
            if param_source == "Depuis un ticker":
                ticker_input = st.text_input("Ticker", value=DEFAULT_ASSET_TICKER, placeholder="ex. AAPL, SPY", key="mc_ticker").strip() or DEFAULT_ASSET_TICKER
                ticker_used = capm_service.resolve_asset_ticker(ticker_input)[0] or ticker_input
                years = {"5 ans": 5, "3 ans": 3, "2 ans": 2, "1 an": 1}[st.selectbox("Période historique", ["5 ans", "3 ans", "2 ans", "1 an"], index=0, key="mc_period")]
                end_dt = pd.Timestamp.now()
                with st.spinner("Estimation de μ et σ…"):
                    mu_est, sigma_est = monte_carlo_service.estimate_mu_sigma_from_ticker(ticker_used, end_dt - pd.DateOffset(years=years), end_dt)
                if mu_est is not None and sigma_est is not None:
                    mu_input, sigma_input = mu_est, sigma_est
                    st.caption(f"μ = {format_pct(mu_input)} · σ = {format_pct(sigma_input)}")
                else:
                    st.info("Estimation impossible ; valeurs par défaut utilisées.")
            else:
                mu_input = st.number_input("Drift μ (annuel, %)", -20.0, 50.0, float(DEFAULT_MC_DRIFT * 100), 0.5, format="%.2f", key="mc_drift") / 100.0
                sigma_input = st.number_input("Volatilité σ (annuelle, %)", 1.0, 100.0, float(DEFAULT_MC_VOLATILITY * 100), 0.5, format="%.2f", key="mc_vol") / 100.0
            initial_wealth = st.number_input("Capital initial (S₀)", 1.0, 1e12, float(DEFAULT_MC_INITIAL_WEALTH), 10_000.0, format="%.0f", key="mc_S0")
            horizon_years = st.number_input("Horizon (années)", 1.0, 50.0, float(DEFAULT_MC_HORIZON_YEARS), 1.0, format="%.1f", key="mc_T")
        with c2:
            n_paths = st.number_input("Nombre de trajectoires", 500, 50_000, int(DEFAULT_MC_PATHS), 500, key="mc_paths")
            n_steps_per_year = st.selectbox("Pas par an", [12, 52, 252], index=2, format_func=lambda x: {12: "Mensuel (12)", 52: "Hebdo (52)", 252: "Quotidien (252)"}[x], key="mc_steps")
            var_confidence = st.selectbox("Confiance VaR / CVaR", [0.90, 0.95, 0.99], index=1, format_func=lambda x: f"{int(x * 100)} %", key="mc_conf")
            with st.expander("Shortfall & flux", expanded=False):
                shortfall_threshold = st.number_input("Seuil de shortfall", 0.0, value=0.0, step=1000.0, format="%.0f", help="Probabilité que le capital final soit sous ce seuil (0 = risque de ruine).", key="mc_shortfall")
                annual_flow = st.number_input("Flux annuel (retrait + / apport −)", value=0.0, step=1000.0, format="%.0f", key="mc_flow")
            seed_value = st.number_input("Graine aléatoire", min_value=0, value=42, step=1, key="mc_seed") if st.checkbox("Graine fixe (reproductible)", value=False, key="mc_use_seed") else None
    st.divider()

    with st.spinner("Simulation Monte-Carlo en cours…"):
        result = _run_monte_carlo_cached(
            initial_wealth, mu_input, sigma_input, horizon_years, int(n_paths), n_steps_per_year,
            var_confidence, seed_value, annual_flow, shortfall_threshold,
        )

    ts, vc, dd = result["terminal_stats"], result["var_cvar"], result["drawdown_stats"]
    ctx = f"**S₀** {initial_wealth:,.0f} · **μ** {format_pct(mu_input)} · **σ** {format_pct(sigma_input)} · **T** {horizon_years:.0f} ans · {int(n_paths)} trajectoires"
    if ticker_used:
        ctx += f" · μ, σ depuis **{ticker_used}**"
    st.info(ctx)

    # ----- KPI en tête -----
    p_short = result.get("prob_shortfall")
    kpi_row([
        ("Patrimoine médian", f"{ts['p50']:,.0f}", "terminal_wealth"),
        (f"VaR {int(var_confidence * 100)} % (niveau)", f"{vc['var_level']:,.0f}", "var"),
        ("P(shortfall)", f"{p_short * 100:.1f} %" if p_short is not None else "—", "shortfall_probability"),
        ("Max drawdown médian", format_pct(dd["p50"]), "max_drawdown"),
    ])
    st.caption("VaR/CVaR et shortfall sont calculés sur la distribution **simulée** du patrimoine final (prospective).")

    # ----- Graphe héros : fan chart -----
    st.markdown("")
    hero_row = st.columns([0.9, 0.1])
    with hero_row[0]:
        st.subheader("Éventail des scénarios (fan chart)")
    with hero_row[1]:
        info_inline("fan_chart", "Fan chart")

    time_axis, pc = result["time_axis"], result["percentile_curves"]
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=time_axis, y=pc[95], mode="lines", line=dict(width=0), showlegend=False, hoverinfo="skip"))
    fig.add_trace(go.Scatter(x=time_axis, y=pc[5], mode="lines", line=dict(width=0), fill="tonexty", fillcolor="rgba(37, 99, 235, 0.15)", name="P5–P95"))
    fig.add_trace(go.Scatter(x=time_axis, y=pc[75], mode="lines", line=dict(width=0), showlegend=False, hoverinfo="skip"))
    fig.add_trace(go.Scatter(x=time_axis, y=pc[25], mode="lines", line=dict(width=0), fill="tonexty", fillcolor="rgba(37, 99, 235, 0.30)", name="P25–P75"))
    fig.add_trace(go.Scatter(x=time_axis, y=pc[50], mode="lines", line=dict(color=COLOR_NEGATIVE, width=2.5), name="Médiane (P50)"))
    fig.update_layout(title="Patrimoine simulé par percentiles", xaxis_title="Années", yaxis_title="Patrimoine")
    apply_theme(fig)
    st.plotly_chart(fig, use_container_width=True)

    # ----- Avancé -----
    with advanced_expander():
        m1, m2, m3, m4 = st.columns(4)
        with m1:
            metric_with_info("Patrimoine moyen", f"{ts['mean']:,.0f}", "terminal_wealth")
        with m2:
            metric_with_info("P5 (pessimiste)", f"{ts['p5']:,.0f}", "terminal_wealth")
        with m3:
            metric_with_info("P95 (optimiste)", f"{ts['p95']:,.0f}", "terminal_wealth")
        with m4:
            metric_with_info(f"CVaR {int(var_confidence * 100)} % (niveau)", f"{vc['cvar_level']:,.0f}", "cvar")

        st.markdown("**Distribution du patrimoine final**")
        terminal = result["paths"][:, -1]
        fig2 = go.Figure()
        fig2.add_trace(go.Histogram(x=terminal, nbinsx=50, name="Patrimoine final", marker_color="rgba(37, 99, 235, 0.65)"))
        fig2.add_vline(x=vc["var_level"], line_dash="dash", line_color=COLOR_NEGATIVE, annotation_text="VaR")
        fig2.add_vline(x=vc["cvar_level"], line_dash="dot", line_color="darkred", annotation_text="CVaR")
        fig2.update_layout(title="Distribution du patrimoine final", xaxis_title="Patrimoine final", yaxis_title="Nombre de trajectoires")
        apply_theme(fig2, height=340, legend=False)
        st.plotly_chart(fig2, use_container_width=True)
