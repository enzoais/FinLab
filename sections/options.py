"""Options (Black-Scholes) tab."""
import io
import math
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from config import (
    DEFAULT_ASSET_TICKER,
    DEFAULT_BS_SPOT,
    DEFAULT_BS_STRIKE,
    DEFAULT_BS_TIME_TO_EXPIRY,
    DEFAULT_BS_VOLATILITY,
    DEFAULT_RISK_FREE_RATE,
)
from services import black_scholes_service, capm_service
from utils.plot_config import HOVERLABEL


def _format_pct(value: float, decimals: int = 2) -> str:
    """Format as percentage (e.g. 0.0523 -> 5.23%)."""
    return f"{value * 100:.{decimals}f}%"


def _format_greek(value: float, decimals: int = 4) -> str:
    """Format Greek (Gamma, Vega): use scientific notation when very small so 0.0000 is not misleading."""
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return "—"
    if abs(value) < 1e-6 and value != 0:
        return f"{value:.2e}"
    return f"{value:.{decimals}f}"


@st.cache_data(ttl=3600)
def _fetch_spot_cached(ticker: str):
    """Cached spot fetch from yfinance."""
    return black_scholes_service.fetch_spot_from_ticker(ticker)


@st.cache_data(ttl=3600)
def _historical_vol_cached(ticker: str, window_days: int):
    """Cached historical volatility."""
    return black_scholes_service.historical_volatility(ticker, window_days)


def render():
    """Render Options tab."""
    # ----- 1. Header -----
    st.header("Black-Scholes Option Pricing")
    st.markdown("Price Call/Put options and compute Greeks (Δ, Γ, Θ, Vega, Rho).")

    with st.expander("? **What is this tab for?**", expanded=False):
        st.markdown("""
        **Purpose:** This tab values **options** (rights to buy or sell an asset at a fixed price by a date).  
        A **Call** is the right to buy; a **Put** is the right to sell. The **Black-Scholes** model gives a theoretical price from the spot price, strike, time to expiry, interest rate, and volatility.

        - **Implied volatility** is the volatility that makes the model match the market price; it reflects market uncertainty.  
        - The **Greeks** measure sensitivity: **Delta** (vs spot), **Gamma** (change of Delta), **Theta** (vs time), **Vega** (vs volatility), **Rho** (vs interest rate). They are used to hedge and manage risk.

        **Use it when:** you trade or hedge options, want to see fair value vs market price, or need to understand how option value changes with spot, time, and volatility.
        """)
    st.markdown("")

    # ----- Parameters (in-tab: only Black-Scholes inputs visible here) -----
    with st.expander("Parameters", expanded=True):
        st.caption("Set underlying, strike, expiry, rate and volatility. These inputs apply only to this tab.")
        st.markdown("")
        c1, c2 = st.columns(2)
        with c1:
            spot_source = st.radio(
                "Underlying",
                options=["Manual spot", "Ticker (fetch spot)"],
                index=1,
                help="Manual: enter spot price. Ticker: fetch latest close from market data.",
                key="bs_spot_source",
            )

            spot_manual = None
            ticker_input = DEFAULT_ASSET_TICKER
            resolved_ticker = None
            resolved_name = None
            asset_ticker = None
            use_hist_vol = False
            hist_vol_window = 30

            if spot_source == "Ticker (fetch spot)":
                ticker_input = (
                    st.text_input(
                        "Ticker",
                        value=DEFAULT_ASSET_TICKER,
                        placeholder="e.g. AAPL, MSFT",
                        help="Symbol to fetch spot price and optional historical volatility.",
                        key="bs_ticker",
                    ).strip()
                    or DEFAULT_ASSET_TICKER
                )
                resolved_ticker, resolved_name, _ = capm_service.resolve_asset_ticker(
                    ticker_input
                )
                asset_ticker = resolved_ticker if resolved_ticker else ticker_input
                use_hist_vol = st.checkbox(
                    "Use historical volatility (30d)",
                    value=False,
                    help="Pre-fill volatility with annualized 30-day historical vol from the ticker.",
                    key="bs_use_hist_vol",
                )
                if use_hist_vol:
                    hist_vol_window = st.selectbox(
                        "Historical vol window (days)",
                        options=[21, 30, 63, 126],
                        index=1,
                        key="bs_hist_vol_window",
                    )
            else:
                spot_manual = st.number_input(
                    "Spot price (S)",
                    min_value=0.01,
                    max_value=1_000_000.0,
                    value=float(DEFAULT_BS_SPOT),
                    step=1.0,
                    format="%.2f",
                    help="Current price of the underlying asset.",
                    key="bs_spot_manual",
                )

            strike = st.number_input(
                "Strike (K)",
                min_value=0.01,
                max_value=1_000_000.0,
                value=float(DEFAULT_BS_STRIKE),
                step=1.0,
                format="%.2f",
                help="Strike price of the option.",
                key="bs_strike",
            )

            time_to_expiry = st.number_input(
                "Time to expiry (years, T)",
                min_value=1e-4,
                max_value=50.0,
                value=float(DEFAULT_BS_TIME_TO_EXPIRY),
                step=0.1,
                format="%.2f",
                help="Time to expiration in years (e.g. 0.5 for 6 months).",
                key="bs_T",
            )

            risk_free_rate = st.number_input(
                "Risk-free rate (r)",
                min_value=0.0,
                max_value=0.30,
                value=float(DEFAULT_RISK_FREE_RATE),
                step=0.005,
                format="%.3f",
                help="Annual risk-free rate (decimal, e.g. 0.05 for 5%).",
                key="bs_r",
            )
        with c2:
            vol_mode = st.radio(
                "Volatility",
                options=["Enter volatility (σ)", "Enter market option price (solve for IV)"],
                index=0,
                help="Provide volatility directly or solve for implied volatility from market price.",
                key="bs_vol_mode",
            )

            volatility_input = None
            market_price_input = None
            option_type_iv = "call"

            if vol_mode == "Enter volatility (σ)":
                vol_pct_default = DEFAULT_BS_VOLATILITY * 100
                if spot_source == "Ticker (fetch spot)" and use_hist_vol and ticker_input:
                    hist_vol = _historical_vol_cached(
                        resolved_ticker if resolved_ticker else ticker_input, hist_vol_window
                    )
                    if hist_vol is not None:
                        vol_pct_default = hist_vol * 100
                vol_pct = st.number_input(
                    "Volatility (σ %)",
                    min_value=0.1,
                    max_value=200.0,
                    value=float(vol_pct_default),
                    step=0.5,
                    format="%.2f",
                    help="Annual volatility in % (e.g. 20 for 20%).",
                    key="bs_sigma_pct",
                )
                volatility_input = vol_pct / 100.0
            else:
                option_type_iv = st.selectbox(
                    "Option type (for IV)",
                    options=["call", "put"],
                    index=0,
                    key="bs_option_type_iv",
                )
                market_price_input = st.number_input(
                    "Market option price",
                    min_value=0.0,
                    max_value=1_000_000.0,
                    value=10.0,
                    step=0.5,
                    format="%.2f",
                    help="Observed market price; implied volatility will be solved.",
                    key="bs_market_price",
                )

            with st.expander("Heatmap range", expanded=False):
                st.caption("Axis ranges for Option price and Greeks heatmaps.")
                spot_min_pct = st.number_input(
                    "Spot min (% of current)",
                    value=70.0,
                    step=5.0,
                    key="bs_heat_spot_min",
                )
                spot_max_pct = st.number_input(
                    "Spot max (% of current)",
                    value=130.0,
                    step=5.0,
                    key="bs_heat_spot_max",
                )
                vol_min_pct = st.number_input(
                    "Volatility min (%)",
                    value=5.0,
                    step=2.0,
                    key="bs_heat_vol_min",
                )
                vol_max_pct = st.number_input(
                    "Volatility max (%)",
                    value=50.0,
                    step=2.0,
                    key="bs_heat_vol_max",
                )
        st.markdown("")
    st.divider()

    # Resolve spot
    if spot_source == "Ticker (fetch spot)":
        spot = _fetch_spot_cached(asset_ticker)
        if spot is None:
            st.error(f"Could not fetch spot for '{asset_ticker}'. Check ticker or use Manual spot.")
            return
        spot = float(spot)
    else:
        spot = spot_manual
        asset_ticker = None
        resolved_name = None

    if spot_source == "Ticker (fetch spot)" and not resolved_name:
        resolved_name, _ = capm_service.get_asset_display_info(asset_ticker)

    # Run analysis
    result = black_scholes_service.run_bs_analysis(
        spot=spot,
        strike=strike,
        time_to_expiry=time_to_expiry,
        risk_free_rate=risk_free_rate,
        volatility=volatility_input,
        market_price=market_price_input,
        option_type_for_iv=option_type_iv,
    )

    if not result["success"]:
        st.error(result.get("error", "An error occurred."))
        return

    sigma = result["sigma"]
    call_price = result["call_price"]
    put_price = result["put_price"]
    option_type_display = "Call" if option_type_iv == "call" else "Put"
    option_price = call_price if option_type_iv == "call" else put_price
    delta_val = result["delta_call"] if option_type_iv == "call" else result["delta_put"]
    theta_val = result["theta_call"] if option_type_iv == "call" else result["theta_put"]
    rho_val = result["rho_call"] if option_type_iv == "call" else result["rho_put"]

    # ----- 2. Context block -----
    if asset_ticker and resolved_name:
        st.info(f"**Underlying:** {resolved_name} ({asset_ticker}) · **Spot:** {spot:.2f}")
    else:
        st.info(f"**Spot:** {spot:.2f} · **Strike:** {strike:.2f} · **T:** {time_to_expiry:.2f}y · **r:** {risk_free_rate*100:.2f}% · **σ:** {sigma*100:.2f}%")
    if vol_mode == "Enter market option price (solve for IV)":
        st.caption(f"**Implied volatility:** {result['implied_vol']*100:.2f}% (solved from {option_type_display} market price {market_price_input:.2f}).")
    if spot_source == "Ticker (fetch spot)" and use_hist_vol:
        st.caption(f"Volatility source: historical {hist_vol_window}d annualized (or overridden by IV).")
    st.caption("T in years; σ and r in decimal. Greeks: Theta per day, Vega per 1% vol, Rho per 1% rate.")
    st.markdown("")

    # ----- 3. Key results -----
    st.subheader("Key results")
    with st.expander("? **What do these numbers mean?**", expanded=False):
        st.markdown("""
        This block shows the **theoretical option price** and **Greeks** from the Black-Scholes model.

        - **Option price:** Fair value of the Call or Put given current inputs; compare to market price to spot mispricing.  
        - **Implied volatility (IV):** When you enter a market price, the volatility that makes BSM match it; reflects market uncertainty.  
        - **Delta:** Sensitivity to a $1 move in spot; also approximates probability of finishing in the money (for calls).  
        - **Gamma:** Rate of change of Delta; high near the money.  
        - **Theta:** Time decay per day (typically negative).  
        - **Vega:** Sensitivity to a 1% increase in volatility. Use these to hedge and size risk.
        """)

    n_cols = 6 if result.get("implied_vol") is not None else 5
    cols = st.columns(n_cols)
    idx = 0
    cols[idx].metric(
        f"{option_type_display} price",
        f"{option_price:.2f}",
        help="Purpose: theoretical fair value of the option from Black-Scholes. Compare to market price to see if the option is rich or cheap. Use it for valuation and trading decisions.",
    )
    idx += 1
    if result.get("implied_vol") is not None:
        cols[idx].metric(
            "Implied volatility",
            _format_pct(result["implied_vol"]),
            help="Purpose: the volatility that makes the model price equal to the market price. Higher IV = market expects more uncertainty. Use it to compare across strikes and maturities.",
        )
        idx += 1
    cols[idx].metric(
        "Delta (Δ)",
        f"{delta_val:.4f}",
        help="Purpose: sensitivity of option price to a $1 move in the underlying. Call Delta 0.5 ≈ at the money; Put Delta is negative. Use it to hedge (e.g. Delta-neutral).",
    )
    idx += 1
    cols[idx].metric(
        "Gamma (Γ)",
        _format_greek(result["gamma"]),
        help="Purpose: rate of change of Delta; high when the option is near the money. Use it to see how often you need to rehedge.",
    )
    idx += 1
    cols[idx].metric(
        "Theta (Θ, per day)",
        f"{theta_val:.4f}",
        help="Purpose: time decay per calendar day (usually negative). Use it to see how much value the option loses each day.",
    )
    idx += 1
    cols[idx].metric(
        "Vega (per 1% vol)",
        _format_greek(result["vega"]),
        help="Purpose: sensitivity to a 1% increase in volatility. Use it to hedge volatility risk or to see how much the option gains if vol rises.",
    )

    gamma_val = result.get("gamma")
    vega_val = result.get("vega")
    if (gamma_val is not None and vega_val is not None and
            not (isinstance(gamma_val, float) and np.isnan(gamma_val)) and
            not (isinstance(vega_val, float) and np.isnan(vega_val)) and
            (abs(gamma_val) < 1e-6 or abs(vega_val) < 1e-6)):
        st.caption("**Gamma / Vega ≈ 0:** For deep in-the-money or out-of-the-money options, these sensitivities are negligible (the option behaves like the underlying or like zero). They are largest when the option is *at the money* (spot ≈ strike).")

    if time_to_expiry < 1 / 365:
        st.warning("Time to expiry very short: Greeks may be unstable.")
    if result.get("implied_vol") is not None and (result["implied_vol"] < 0.01 or result["implied_vol"] > 2.0):
        st.warning("Implied volatility is extreme; check market price and inputs.")
    st.markdown("")
    st.divider()

    # ----- 4. Statistics & optional details -----
    st.subheader("Model details & sensitivity")
    st.caption("Expand for Put-Call parity, Rho, model assumptions, and sensitivity scenarios.")
    st.markdown("")

    with st.expander("Put-Call parity", expanded=False):
        st.caption("C − P should equal S − K·e^(-rT). Small numerical difference is normal.")
        lhs = result["put_call_lhs"]
        rhs = result["put_call_rhs"]
        diff = result["put_call_diff"]
        st.write(f"**C − P:** {lhs:.6f}")
        st.write(f"**S − K·e^(-rT):** {rhs:.6f}")
        st.write(f"**|Difference|:** {diff:.6f}")
        if diff < 1e-6:
            st.caption("Parity holds (within numerical precision).")

    with st.expander("Other Greeks (Rho)", expanded=False):
        st.caption("Rho: sensitivity to a 1% increase in the risk-free rate.")
        st.write(f"**Rho Call:** {result['rho_call']:.4f}")
        st.write(f"**Rho Put:** {result['rho_put']:.4f}")

    with st.expander("Model assumptions", expanded=False):
        st.markdown("""
        Black-Scholes assumes: **European** exercise (no early exercise), **no dividends**, **constant volatility**, **lognormal** asset returns, and **constant risk-free rate**.
        Real markets often show a **volatility smile** (different IV across strikes) and **dividends**; use with caution for American options or when these assumptions are violated.
        """)
        st.warning("BSM assumes constant volatility and no dividends; real markets may show volatility smile.")

    with st.expander("Sensitivity scenarios", expanded=False):
        st.caption("How option price changes for small shocks in Spot, Vol, or Time.")
        scenarios = black_scholes_service.sensitivity_scenarios(
            spot, strike, time_to_expiry, risk_free_rate, sigma,
            option_type_iv,
            spot_shocks_pct=[-5, -1, 1, 5],
            vol_shocks_pct=[-1, 1],
            time_shock_days=1,
        )
        df_sens = pd.DataFrame([
            {"Shock": r["shock"], "New price": f"{r['new_price']:.2f}", "ΔP/P (%)": f"{r['delta_pct']:+.2f}"}
            for r in scenarios
        ])
        st.dataframe(df_sens, use_container_width=True, hide_index=True)

    st.markdown("")
    st.divider()

    # ----- 5. Charts -----
    # 5a. Option value vs Spot
    st.subheader("Option value vs Spot")
    with st.expander("? **What is this chart?**", expanded=False):
        st.markdown("""
        **Purpose:** Show how the **option price** (and intrinsic value at expiry) varies with the **underlying spot price**.

        - **Solid line:** Black-Scholes price today.  
        - **Dashed line:** Intrinsic value at expiry (max(S−K,0) for Call, max(K−S,0) for Put).  
        - **Vertical line:** Current strike. Use it to see how much is time value vs intrinsic.
        """)
    S_grid = np.linspace(max(1e-6, spot * 0.5), spot * 1.5, 100)
    price_curve = np.array([
        black_scholes_service.bsm_call_price(s, strike, time_to_expiry, risk_free_rate, sigma)
        if option_type_iv == "call"
        else black_scholes_service.bsm_put_price(s, strike, time_to_expiry, risk_free_rate, sigma)
        for s in S_grid
    ])
    if option_type_iv == "call":
        intrinsic = np.maximum(S_grid - strike, 0)
    else:
        intrinsic = np.maximum(strike - S_grid, 0)
    fig1 = go.Figure()
    fig1.add_trace(
        go.Scatter(
            x=S_grid,
            y=price_curve,
            mode="lines",
            name="BSM price (today)",
            line=dict(color="rgb(31, 119, 180)", width=2),
            hovertemplate="<b>BSM price (today)</b><br>Spot: %{x:.2f}<br>Option value: %{y:.2f}<extra></extra>",
        )
    )
    fig1.add_trace(
        go.Scatter(
            x=S_grid,
            y=intrinsic,
            mode="lines",
            name="Intrinsic (at expiry)",
            line=dict(color="gray", width=1.5, dash="dash"),
            hovertemplate="<b>Intrinsic (at expiry)</b><br>Spot: %{x:.2f}<br>Payoff: %{y:.2f}<extra></extra>",
        )
    )
    fig1.add_vline(x=strike, line_dash="dot", line_color="gray", opacity=0.8)
    fig1.update_layout(
        title=f"{option_type_display} value vs Spot",
        xaxis_title="Spot price (S)",
        yaxis_title="Option value",
        showlegend=True,
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        margin=dict(t=50),
        hovermode="x unified",
        hoverlabel=HOVERLABEL,
    )
    fig1.update_xaxes(showgrid=True, gridwidth=1, gridcolor="rgba(128,128,128,0.2)")
    fig1.update_yaxes(showgrid=True, gridwidth=1, gridcolor="rgba(128,128,128,0.2)")
    st.plotly_chart(fig1, use_container_width=True)
    st.markdown("")

    # 5b. Payoff diagram
    st.subheader("Payoff at expiry")
    with st.expander("? **What is this chart?**", expanded=False):
        st.markdown("""
        **Purpose:** Payoff at **expiry** as a function of spot: Call = max(S−K, 0), Put = max(K−S, 0).  
        Shows the intrinsic value only (no time value). Use it to understand P&L at expiration.
        """)
    S_payoff = np.linspace(max(1, strike * 0.5), strike * 1.5, 150)
    call_payoff = np.maximum(S_payoff - strike, 0)
    put_payoff = np.maximum(strike - S_payoff, 0)
    fig2 = go.Figure()
    fig2.add_trace(
        go.Scatter(
            x=S_payoff,
            y=call_payoff,
            mode="lines",
            name="Call",
            line=dict(color="green", width=2),
            hovertemplate="<b>Call</b><br>Spot at expiry: %{x:.2f}<br>Payoff: %{y:.2f}<extra></extra>",
        )
    )
    fig2.add_trace(
        go.Scatter(
            x=S_payoff,
            y=put_payoff,
            mode="lines",
            name="Put",
            line=dict(color="red", width=2),
            hovertemplate="<b>Put</b><br>Spot at expiry: %{x:.2f}<br>Payoff: %{y:.2f}<extra></extra>",
        )
    )
    fig2.add_vline(x=strike, line_dash="dot", line_color="gray", opacity=0.8)
    fig2.update_layout(
        title="Payoff at expiry",
        xaxis_title="Spot price (S)",
        yaxis_title="Payoff",
        showlegend=True,
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        margin=dict(t=50),
        hovermode="x unified",
        hoverlabel=HOVERLABEL,
    )
    fig2.update_xaxes(showgrid=True, gridwidth=1, gridcolor="rgba(128,128,128,0.2)")
    fig2.update_yaxes(showgrid=True, gridwidth=1, gridcolor="rgba(128,128,128,0.2)")
    st.plotly_chart(fig2, use_container_width=True)
    st.markdown("")

    # 5c. Heatmap: Option price vs Spot vs Vol
    st.subheader("Heatmap: Option price (Spot vs Volatility)")
    with st.expander("? **What is this chart?**", expanded=False):
        st.markdown("""
        **Purpose:** Option price as a **surface** over **Spot** (x) and **Volatility** (y).  
        Warmer colors = higher price. Use it to see how price reacts to joint moves in spot and vol.
        """)
    spot_axis = np.linspace(spot * spot_min_pct / 100, spot * spot_max_pct / 100, 40)
    vol_axis = np.linspace(vol_min_pct / 100, vol_max_pct / 100, 40)
    if option_type_iv == "call":
        heatmap_vals = np.array([
            [black_scholes_service.bsm_call_price(s, strike, time_to_expiry, risk_free_rate, v) for s in spot_axis]
            for v in vol_axis
        ])
    else:
        heatmap_vals = np.array([
            [black_scholes_service.bsm_put_price(s, strike, time_to_expiry, risk_free_rate, v) for s in spot_axis]
            for v in vol_axis
        ])
    fig3 = go.Figure(
        data=go.Heatmap(
            x=spot_axis,
            y=vol_axis * 100,
            z=heatmap_vals,
            colorscale="RdYlGn",
            colorbar=dict(title="Price", tickfont=dict(size=12)),
            hovertemplate="<b>Option price</b><br>Spot: %{x:.2f}<br>Volatility: %{y:.1f}%<br>Price: %{z:.2f}<extra></extra>",
        )
    )
    fig3.update_layout(
        title=f"{option_type_display} price: Spot vs Volatility",
        xaxis_title="Spot price",
        yaxis_title="Volatility (%)",
        margin=dict(t=50),
        height=400,
        hoverlabel=HOVERLABEL,
    )
    fig3.update_xaxes(showgrid=False)
    fig3.update_yaxes(showgrid=False)
    st.plotly_chart(fig3, use_container_width=True)
    st.markdown("")

    # 5d. Greeks vs Spot
    st.subheader("Greeks vs Spot")
    with st.expander("? **What is this chart?**", expanded=False):
        st.markdown("""
        **Purpose:** How **Delta**, **Gamma**, and **Vega** change as the **spot price** moves.  
        Delta and Vega are often highest near the money; Gamma peaks at the strike. Use it to plan hedging.
        """)
    S_greeks = np.linspace(max(1, strike * 0.6), strike * 1.4, 80)
    delta_curve = []
    gamma_curve = []
    vega_curve = []
    for s in S_greeks:
        g = black_scholes_service.bsm_greeks(s, strike, time_to_expiry, risk_free_rate, sigma)
        delta_curve.append(g["delta_call"] if option_type_iv == "call" else g["delta_put"])
        gamma_curve.append(g["gamma"])
        vega_curve.append(g["vega"])
    fig4 = go.Figure()
    fig4.add_trace(
        go.Scatter(
            x=S_greeks,
            y=delta_curve,
            mode="lines",
            name="Delta",
            line=dict(color="blue", width=2),
            hovertemplate="<b>Delta</b><br>Spot: %{x:.2f}<br>Delta: %{y:.4f}<extra></extra>",
        )
    )
    fig4.add_trace(
        go.Scatter(
            x=S_greeks,
            y=gamma_curve,
            mode="lines",
            name="Gamma",
            line=dict(color="green", width=2),
            yaxis="y2",
            hovertemplate="<b>Gamma</b><br>Spot: %{x:.2f}<br>Gamma: %{y:.4f}<extra></extra>",
        )
    )
    fig4.add_trace(
        go.Scatter(
            x=S_greeks,
            y=vega_curve,
            mode="lines",
            name="Vega",
            line=dict(color="orange", width=2),
            hovertemplate="<b>Vega</b><br>Spot: %{x:.2f}<br>Vega: %{y:.4f}<extra></extra>",
        )
    )
    fig4.add_vline(x=strike, line_dash="dot", line_color="gray", opacity=0.8)
    gamma_max = max(gamma_curve) if gamma_curve else 0.01
    fig4.update_layout(
        title="Greeks vs Spot",
        xaxis_title="Spot price (S)",
        yaxis=dict(title="Delta / Vega", side="left", showgrid=True, gridwidth=1, gridcolor="rgba(128,128,128,0.2)"),
        yaxis2=dict(
            title="Gamma",
            side="right",
            overlaying="y",
            showgrid=False,
            range=[0, max(gamma_max * 1.15, 0.001)],
        ),
        showlegend=True,
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        margin=dict(t=50, r=80),
        hovermode="x unified",
        hoverlabel=HOVERLABEL,
    )
    fig4.update_xaxes(showgrid=True, gridwidth=1, gridcolor="rgba(128,128,128,0.2)")
    st.plotly_chart(fig4, use_container_width=True)
    st.markdown("")
    st.divider()

    # ----- 6. Export -----
    with st.expander("Export results (CSV)", expanded=False):
        st.caption("Download all inputs and outputs (prices, Greeks, d1, d2) as CSV.")
        rows = [
            ["spot", f"{spot:.6f}"],
            ["strike", f"{strike:.6f}"],
            ["time_to_expiry", f"{time_to_expiry:.6f}"],
            ["risk_free_rate", f"{risk_free_rate:.6f}"],
            ["volatility", f"{sigma:.6f}"],
            ["call_price", f"{call_price:.6f}"],
            ["put_price", f"{put_price:.6f}"],
            ["implied_vol", f"{result['implied_vol']:.6f}" if result.get("implied_vol") is not None else ""],
            ["delta_call", f"{result['delta_call']:.6f}"],
            ["delta_put", f"{result['delta_put']:.6f}"],
            ["gamma", f"{result['gamma']:.6f}"],
            ["theta_call", f"{result['theta_call']:.6f}"],
            ["theta_put", f"{result['theta_put']:.6f}"],
            ["vega", f"{result['vega']:.6f}"],
            ["rho_call", f"{result['rho_call']:.6f}"],
            ["rho_put", f"{result['rho_put']:.6f}"],
            ["d1", f"{result['d1']:.6f}"],
            ["d2", f"{result['d2']:.6f}"],
        ]
        buf = io.StringIO()
        buf.write("metric,value\n")
        for row in rows:
            buf.write(f"{row[0]},{row[1]}\n")
        csv_bytes = buf.getvalue().encode("utf-8")
        st.download_button(
            label="Download results (CSV)",
            data=csv_bytes,
            file_name="black_scholes_results.csv",
            mime="text/csv",
            key="bs_export_csv",
        )
