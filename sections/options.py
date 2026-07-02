"""Onglet Options (Black-Scholes) — refonte claire & pro."""
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from config import (
    DEFAULT_ASSET_TICKER, DEFAULT_BS_SPOT, DEFAULT_BS_STRIKE,
    DEFAULT_BS_TIME_TO_EXPIRY, DEFAULT_BS_VOLATILITY, DEFAULT_RISK_FREE_RATE,
)
from services import black_scholes_service, capm_service
from utils.formatters import format_pct
from utils.plot_config import apply_theme, COLOR_PRIMARY, COLOR_MUTED
from utils.ui import kpi_row, section_header, advanced_expander, show_data_error, info_inline, metric_with_info


def _fmt_greek(value, decimals=4):
    """Formate un Greek : notation scientifique si très petit, sinon décimales fixes."""
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return "—"
    if abs(value) < 1e-6 and value != 0:
        return f"{value:.2e}"
    return f"{value:.{decimals}f}"


@st.cache_data(ttl=3600)
def _fetch_spot_cached(ticker: str):
    return black_scholes_service.fetch_spot_from_ticker(ticker)


@st.cache_data(ttl=3600)
def _historical_vol_cached(ticker: str, window_days: int):
    return black_scholes_service.historical_volatility(ticker, window_days)


def render():
    """Affiche l'onglet Options (Black-Scholes)."""
    section_header("Options (Black-Scholes)", "black_scholes", level="header")
    st.caption("Valorise un call ou un put et calcule les sensibilités (Greeks : Δ, Γ, Θ, Vega, Rho).")

    # ----- Paramètres -----
    with st.expander("Paramètres", expanded=True):
        c1, c2 = st.columns(2)
        with c1:
            option_type = st.radio("Type d'option", ["Call", "Put"], index=0, horizontal=True, key="opt_type")
            option_key = "call" if option_type == "Call" else "put"
            # Défaut « Spot manuel » (100/100, à la monnaie) : Greeks parlants dès l'ouverture
            spot_source = st.radio("Sous-jacent", ["Spot manuel", "Ticker (spot du marché)"], index=0, key="opt_spot_source")
            spot_manual, asset_ticker, resolved_name = None, None, None
            use_hist_vol, hist_vol_window = False, 30
            if spot_source == "Ticker (spot du marché)":
                ticker_input = st.text_input("Ticker", value=DEFAULT_ASSET_TICKER, placeholder="ex. AAPL", key="opt_ticker").strip() or DEFAULT_ASSET_TICKER
                resolved_ticker, resolved_name, _ = capm_service.resolve_asset_ticker(ticker_input)
                asset_ticker = resolved_ticker or ticker_input
                use_hist_vol = st.checkbox("Volatilité historique (pré-remplir σ)", value=False, key="opt_use_hv")
                if use_hist_vol:
                    hist_vol_window = st.selectbox("Fenêtre (jours)", [21, 30, 63, 126], index=1, key="opt_hv_window")
            else:
                spot_manual = st.number_input("Prix spot (S)", 0.01, 1_000_000.0, float(DEFAULT_BS_SPOT), 1.0, format="%.2f", key="opt_spot")
            strike = st.number_input("Strike (K)", 0.01, 1_000_000.0, float(DEFAULT_BS_STRIKE), 1.0, format="%.2f", key="opt_strike")
        with c2:
            time_to_expiry = st.number_input("Échéance (années, T)", 1e-4, 50.0, float(DEFAULT_BS_TIME_TO_EXPIRY), 0.1, format="%.2f", key="opt_T")
            risk_free_rate = st.number_input("Taux sans risque (r)", 0.0, 0.30, float(DEFAULT_RISK_FREE_RATE), 0.005, format="%.3f", key="opt_rf")
            vol_mode = st.radio("Volatilité", ["Saisir σ", "Déduire de l'IV (prix de marché)"], index=0, key="opt_vol_mode")
            volatility_input = market_price_input = None
            if vol_mode == "Saisir σ":
                vol_default = DEFAULT_BS_VOLATILITY * 100
                if spot_source == "Ticker (spot du marché)" and use_hist_vol and asset_ticker:
                    hv = _historical_vol_cached(asset_ticker, hist_vol_window)
                    if hv is not None:
                        vol_default = hv * 100
                volatility_input = st.number_input("Volatilité (σ %)", 0.1, 200.0, float(vol_default), 0.5, format="%.2f", key="opt_sigma") / 100.0
            else:
                market_price_input = st.number_input("Prix de marché de l'option", 0.0, 1_000_000.0, 10.0, 0.5, format="%.2f", key="opt_mkt_price")
    st.divider()

    # Résolution du spot
    if spot_source == "Ticker (spot du marché)":
        spot = _fetch_spot_cached(asset_ticker)
        if spot is None:
            show_data_error(f"Spot introuvable pour « {asset_ticker} ».")
            return
        spot = float(spot)
        if not resolved_name:
            resolved_name = capm_service.get_asset_display_info(asset_ticker)[0]
    else:
        spot = spot_manual

    result = black_scholes_service.run_bs_analysis(
        spot=spot, strike=strike, time_to_expiry=time_to_expiry, risk_free_rate=risk_free_rate,
        volatility=volatility_input, market_price=market_price_input, option_type_for_iv=option_key,
    )
    if not result["success"]:
        show_data_error(result.get("error"))
        return

    sigma = result["sigma"]
    option_price = result["call_price"] if option_key == "call" else result["put_price"]
    delta_val = result["delta_call"] if option_key == "call" else result["delta_put"]
    theta_val = result["theta_call"] if option_key == "call" else result["theta_put"]
    rho_val = result["rho_call"] if option_key == "call" else result["rho_put"]

    # Contexte
    if asset_ticker and resolved_name:
        ctx = f"**{resolved_name} ({asset_ticker})** · spot {spot:.2f} · strike {strike:.2f} · T {time_to_expiry:.2f} an · σ {sigma * 100:.1f}%"
    else:
        ctx = f"**{option_type}** · spot {spot:.2f} · strike {strike:.2f} · T {time_to_expiry:.2f} an · r {risk_free_rate * 100:.1f}% · σ {sigma * 100:.1f}%"
    st.info(ctx)
    if result.get("implied_vol") is not None:
        st.caption(f"Volatilité implicite déduite : **{result['implied_vol'] * 100:.2f}%** (à partir du prix de marché {market_price_input:.2f}).")

    # ----- KPI en tête (prix + 3 Greeks principaux) -----
    kpi_row([
        (f"Prix {option_type}", f"{option_price:.2f}", option_key),
        ("Delta (Δ)", f"{delta_val:.3f}", "delta"),
        ("Gamma (Γ)", _fmt_greek(result["gamma"]), "gamma"),
        ("Vega (par 1 % de vol)", _fmt_greek(result["vega"]), "vega"),
    ])
    st.caption("Theta est exprimé par jour · Vega par variation de 1 % de volatilité · Rho par 1 % de taux.")

    # ----- Graphe héros : valeur de l'option vs spot -----
    st.markdown("")
    hero_row = st.columns([0.9, 0.1])
    with hero_row[0]:
        st.subheader(f"Valeur du {option_type} en fonction du spot")
    with hero_row[1]:
        info_inline("intrinsic_value", "Valeur intrinsèque")

    S_grid = np.linspace(max(1e-6, spot * 0.5), spot * 1.5, 120)
    if option_key == "call":
        price_curve = [black_scholes_service.bsm_call_price(s, strike, time_to_expiry, risk_free_rate, sigma) for s in S_grid]
        intrinsic = np.maximum(S_grid - strike, 0)
    else:
        price_curve = [black_scholes_service.bsm_put_price(s, strike, time_to_expiry, risk_free_rate, sigma) for s in S_grid]
        intrinsic = np.maximum(strike - S_grid, 0)
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=S_grid, y=price_curve, mode="lines", name="Prix Black-Scholes (aujourd'hui)",
        line=dict(color=COLOR_PRIMARY, width=2.5),
        hovertemplate="Spot : %{x:.2f}<br>Valeur : %{y:.2f}<extra></extra>",
    ))
    fig.add_trace(go.Scatter(
        x=S_grid, y=intrinsic, mode="lines", name="Valeur intrinsèque (à l'échéance)",
        line=dict(color=COLOR_MUTED, width=1.5, dash="dash"),
        hovertemplate="Spot : %{x:.2f}<br>Payoff : %{y:.2f}<extra></extra>",
    ))
    fig.add_vline(x=strike, line_dash="dot", line_color="gray", opacity=0.7)
    fig.update_layout(title=f"Valeur du {option_type} vs spot", xaxis_title="Prix spot (S)", yaxis_title="Valeur de l'option")
    apply_theme(fig)
    st.plotly_chart(fig, use_container_width=True)

    # ----- Avancé -----
    with advanced_expander():
        st.markdown("**Autres Greeks**")
        g1, g2, g3 = st.columns(3)
        with g1:
            metric_with_info("Theta (Θ, par jour)", f"{theta_val:.4f}", "theta")
        with g2:
            metric_with_info("Rho (par 1 % de taux)", f"{rho_val:.4f}", "rho")
        with g3:
            iv = result.get("implied_vol")
            metric_with_info("Volatilité implicite", format_pct(iv) if iv is not None else "—", "implied_volatility")

        st.markdown("**Sensibilité (petits chocs)**")
        scenarios = black_scholes_service.sensitivity_scenarios(
            spot, strike, time_to_expiry, risk_free_rate, sigma, option_key,
            spot_shocks_pct=[-5, -1, 1, 5], vol_shocks_pct=[-1, 1], time_shock_days=1,
        )
        df_sens = pd.DataFrame([
            {"Choc": r["shock"], "Nouveau prix": f"{r['new_price']:.2f}", "ΔP/P (%)": f"{r['delta_pct']:+.2f}"}
            for r in scenarios
        ])
        st.dataframe(df_sens, use_container_width=True, hide_index=True)
        st.caption("Hypothèses Black-Scholes : option européenne, pas de dividendes, volatilité constante.")
