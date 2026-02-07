"""Simulation (Monte Carlo) tab."""
import io
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from config import (
    DEFAULT_ASSET_TICKER,
    DEFAULT_MC_DRIFT,
    DEFAULT_MC_HORIZON_YEARS,
    DEFAULT_MC_INITIAL_WEALTH,
    DEFAULT_MC_PATHS,
    DEFAULT_MC_SEED,
    DEFAULT_MC_STEPS_PER_YEAR,
    DEFAULT_MC_VAR_CONFIDENCE,
    DEFAULT_MC_VOLATILITY,
)
from services import capm_service, monte_carlo_service
from utils.plot_config import HOVERLABEL


def _format_pct(value: float, decimals: int = 2) -> str:
    """Format as percentage (e.g. 0.0523 -> 5.23%)."""
    return f"{value * 100:.{decimals}f}%"


@st.cache_data(ttl=3600)
def _run_monte_carlo_cached(
    S0: float,
    mu: float,
    sigma: float,
    T_years: float,
    n_paths: int,
    n_steps_per_year: int,
    confidence: float,
    seed: int | None,
    annual_flow: float,
    shortfall_threshold: float | None,
) -> dict:
    """Cached wrapper: run full Monte Carlo analysis."""
    return monte_carlo_service.run_monte_carlo_analysis(
        S0=S0,
        mu=mu,
        sigma=sigma,
        T_years=T_years,
        n_paths=n_paths,
        n_steps_per_year=n_steps_per_year,
        confidence=confidence,
        seed=seed,
        annual_flow=annual_flow,
        shortfall_threshold=shortfall_threshold,
    )


def render():
    """Render Simulation tab."""
    # ----- 1. Header -----
    st.header("Monte Carlo Simulation (Wealth Management)")
    st.markdown("Project wealth under uncertainty (GBM, VaR, fan chart).")

    with st.expander("? **What is this tab for?**", expanded=False):
        st.markdown("""
        **Purpose:** This tab **simulates** many possible future paths for your wealth (or portfolio value) by drawing random returns over time.  
        Because the future is uncertain, you don't get one number but a *distribution* of outcomes (e.g. "in 20 years, wealth might be between X and Y with 90% probability").

        - **GBM (Geometric Brownian Motion)** is a standard model for how asset prices evolve: drift (expected return) plus random volatility.  
        - **VaR (Value at Risk)** answers: "What is the worst outcome I might see at the horizon with a given confidence?" (e.g. 95% VaR = level below which only 5% of paths fall).  
        - **CVaR (Expected Shortfall)** is the average outcome in those worst cases.  
        - **Shortfall probability** is the chance that terminal wealth falls below a target (e.g. zero or a minimum goal).

        **Use it when:** you plan for retirement, want to see the range of outcomes for savings or a portfolio, or need to quantify downside risk (VaR, shortfall).
        """)
    st.markdown("")

    # ----- Parameters (in-tab: only Monte Carlo inputs visible here) -----
    with st.expander("Parameters", expanded=True):
        st.caption("Set drift/volatility, horizon and simulation options. These inputs apply only to this tab.")
        st.markdown("")
        c1, c2 = st.columns(2)
        with c1:
            param_source = st.radio(
                "Parameters μ and σ",
                options=["Manual (μ, σ)", "From ticker (estimate μ, σ)"],
                index=0,
                help="Manual: enter drift and volatility. From ticker: estimate from historical returns.",
                key="mc_param_source",
            )

            mu_input = float(DEFAULT_MC_DRIFT)
            sigma_input = float(DEFAULT_MC_VOLATILITY)
            ticker_used = None
            if param_source == "From ticker (estimate μ, σ)":
                ticker_input = (
                    st.text_input(
                        "Ticker",
                        value=DEFAULT_ASSET_TICKER,
                        placeholder="e.g. AAPL, SPY",
                        help="Symbol to estimate drift and volatility from historical prices.",
                        key="mc_ticker",
                    ).strip()
                    or DEFAULT_ASSET_TICKER
                )
                resolved_ticker, resolved_name, _ = capm_service.resolve_asset_ticker(ticker_input)
                ticker_used = resolved_ticker if resolved_ticker else ticker_input
                period_preset = st.selectbox(
                    "Historical period",
                    options=["5 years", "3 years", "2 years", "1 year"],
                    index=0,
                    help="Data range for estimating μ and σ.",
                    key="mc_period",
                )
                years = {"5 years": 5, "3 years": 3, "2 years": 2, "1 year": 1}[period_preset]
                end_dt = pd.Timestamp.now()
                start_dt = end_dt - pd.DateOffset(years=years)
                with st.spinner("Estimating μ and σ…"):
                    mu_est, sigma_est = monte_carlo_service.estimate_mu_sigma_from_ticker(
                        ticker_used, start_dt, end_dt
                    )
                if mu_est is not None and sigma_est is not None:
                    mu_input = mu_est
                    sigma_input = sigma_est
                    st.caption(f"μ = {_format_pct(mu_input)}, σ = {_format_pct(sigma_input)}")
                else:
                    st.warning("Could not estimate from ticker; using manual defaults.")
                    mu_input = float(DEFAULT_MC_DRIFT)
                    sigma_input = float(DEFAULT_MC_VOLATILITY)
            else:
                drift_pct = st.number_input(
                    "Drift μ (annual, %)",
                    min_value=-20.0,
                    max_value=50.0,
                    value=float(DEFAULT_MC_DRIFT * 100),
                    step=0.5,
                    format="%.2f",
                    help="Expected annual return (e.g. 7 for 7%).",
                    key="mc_drift_pct",
                )
                vol_pct = st.number_input(
                    "Volatility σ (annual, %)",
                    min_value=1.0,
                    max_value=100.0,
                    value=float(DEFAULT_MC_VOLATILITY * 100),
                    step=0.5,
                    format="%.2f",
                    help="Annual volatility (e.g. 18 for 18%).",
                    key="mc_vol_pct",
                )
                mu_input = drift_pct / 100.0
                sigma_input = vol_pct / 100.0

            initial_wealth = st.number_input(
                "Initial wealth (S₀)",
                min_value=1.0,
                max_value=1e12,
                value=float(DEFAULT_MC_INITIAL_WEALTH),
                step=10_000.0,
                format="%.0f",
                help="Starting capital for the simulation.",
                key="mc_S0",
            )

            horizon_years = st.number_input(
                "Horizon (years)",
                min_value=1.0,
                max_value=50.0,
                value=float(DEFAULT_MC_HORIZON_YEARS),
                step=1.0,
                format="%.1f",
                help="Simulation length in years.",
                key="mc_T",
            )
        with c2:
            n_paths = st.number_input(
                "Number of paths",
                min_value=500,
                max_value=50_000,
                value=int(DEFAULT_MC_PATHS),
                step=500,
                help="More paths give smoother distributions but slower run.",
                key="mc_n_paths",
            )

            n_steps_per_year = st.selectbox(
                "Steps per year",
                options=[12, 52, 252],
                index=2,
                format_func=lambda x: str(x),
                help="252 = daily, 12 = monthly.",
                key="mc_steps",
            )

            var_confidence = st.selectbox(
                "VaR / CVaR confidence",
                options=[0.90, 0.95, 0.99],
                index=1,
                format_func=lambda x: f"{int(x*100)}%",
                help="E.g. 95%: 5% of paths fall below VaR level.",
                key="mc_confidence",
            )

            with st.expander("Shortfall & flows", expanded=False):
                shortfall_threshold = st.number_input(
                    "Shortfall threshold (wealth below = shortfall)",
                    min_value=0.0,
                    value=0.0,
                    step=1000.0,
                    format="%.0f",
                    help="Probability that terminal wealth < this value (e.g. 0 for ruin risk).",
                    key="mc_shortfall",
                )
                annual_flow = st.number_input(
                    "Annual flow (withdrawal + / contribution −)",
                    value=0.0,
                    step=1000.0,
                    format="%.0f",
                    help="Positive = withdrawal each year; negative = contribution.",
                    key="mc_annual_flow",
                )

            use_seed = st.checkbox(
                "Use fixed seed (reproducible)",
                value=False,
                help="Same parameters give identical paths.",
                key="mc_use_seed",
            )
            seed_value = None
            if use_seed:
                seed_value = st.number_input(
                    "Seed",
                    min_value=0,
                    value=42,
                    step=1,
                    key="mc_seed",
                )
        st.markdown("")
    st.divider()

    # ----- 2. Run simulation -----
    with st.spinner("Running Monte Carlo simulation…"):
        result = _run_monte_carlo_cached(
            S0=initial_wealth,
            mu=mu_input,
            sigma=sigma_input,
            T_years=horizon_years,
            n_paths=n_paths,
            n_steps_per_year=n_steps_per_year,
            confidence=var_confidence,
            seed=seed_value,
            annual_flow=annual_flow,
            shortfall_threshold=shortfall_threshold,
        )

    # ----- 3. Context block -----
    context_parts = [
        f"**S₀** = {initial_wealth:,.0f}",
        f"**μ** = {_format_pct(mu_input)}",
        f"**σ** = {_format_pct(sigma_input)}",
        f"**T** = {horizon_years} years",
        f"**Paths** = {n_paths}",
    ]
    if ticker_used:
        context_parts.append(f"**μ, σ from:** {ticker_used}")
    st.info(" | ".join(context_parts))
    st.caption("Parameters used for this simulation. VaR/CVaR are computed on the simulated terminal wealth distribution (forward-looking), not on historical returns.")
    st.markdown("")

    # ----- 4. Key results -----
    st.subheader("Key results")
    with st.expander("? **What do these numbers mean?**", expanded=False):
        st.markdown("""
        This block summarizes the **simulated** distribution of wealth at the end of the horizon.

        - **Mean / Median terminal wealth:** Average and median outcome across all paths.  
        - **VaR (level and %):** Level below which only (1−confidence)% of paths fall; shown in currency and as % of initial wealth.  
        - **CVaR (Expected Shortfall):** Average terminal wealth in those worst (1−confidence)% paths.  
        - **P(shortfall):** Probability that terminal wealth is below the threshold you set (e.g. ruin risk if threshold = 0).  
        - **Max drawdown (median):** Typical worst peak-to-trough decline along a path.
        """)

    ts = result["terminal_stats"]
    vc = result["var_cvar"]
    dd = result["drawdown_stats"]

    c1, c2, c3, c4, c5, c6 = st.columns(6)
    with c1:
        st.metric(
            "Mean terminal wealth",
            f"{ts['mean']:,.0f}",
            help="Purpose: average wealth at horizon across all simulated paths. Use it to set expectations.",
        )
    with c2:
        st.metric(
            "Median terminal wealth",
            f"{ts['p50']:,.0f}",
            help="Purpose: 50th percentile (median) terminal wealth. Less influenced by extreme paths than the mean.",
        )
    with c3:
        st.metric(
            f"VaR {int(var_confidence*100)}% (level)",
            f"{vc['var_level']:,.0f}",
            help="Purpose: level below which only (1-confidence)% of paths fall. Use it to gauge downside.",
        )
    with c4:
        st.metric(
            f"VaR {int(var_confidence*100)}% (%)",
            _format_pct(vc["var_pct"]),
            help="Purpose: VaR as % of initial wealth. Negative = loss in the worst (1-conf)% of paths.",
        )
    with c5:
        st.metric(
            f"CVaR {int(var_confidence*100)}% (level)",
            f"{vc['cvar_level']:,.0f}",
            help="Purpose: average terminal wealth in the worst (1-confidence)% of paths (Expected Shortfall).",
        )
    with c6:
        p_short = result.get("prob_shortfall")
        shortfall_display = f"{p_short*100:.1f}%" if p_short is not None else "—"
        st.metric(
            "P(shortfall)",
            shortfall_display,
            help="Purpose: probability that terminal wealth is below the shortfall threshold. E.g. ruin risk if threshold = 0.",
        )

    st.markdown("")
    c7, c8 = st.columns(2)
    with c7:
        st.metric(
            "CVaR (%)",
            _format_pct(vc["cvar_pct"]),
            help="CVaR as % of initial wealth.",
        )
    with c8:
        st.metric(
            "Median max drawdown",
            _format_pct(dd["p50"]),
            help="Purpose: typical worst peak-to-trough decline along a path. E.g. -30% means half of paths had a drawdown of at least 30%.",
        )

    st.caption("All VaR/CVaR and shortfall are based on the **simulated** terminal wealth distribution, not on historical returns.")
    if sigma_input > 0.4:
        st.warning("High volatility (σ > 40%): results are very sensitive to assumptions; interpret with caution.")
    if horizon_years > 30:
        st.warning("Long horizon: small changes in μ or σ have large effects on terminal wealth; use conservative assumptions.")
    if n_paths < 2000:
        st.warning("Few paths: distribution and percentiles may be noisy; consider increasing the number of paths.")
    st.markdown("")
    st.divider()

    # ----- 5. Statistics & quality -----
    st.subheader("Statistics & quality")
    st.caption("Distribution of terminal wealth and max drawdown; expand for detailed stats.")
    st.markdown("")

    with st.expander("Terminal wealth distribution", expanded=False):
        st.write("Min:", f"{ts['min']:,.0f}", "| Max:", f"{ts['max']:,.0f}")
        st.write("Std:", f"{ts['std']:,.0f}", "| P5:", f"{ts['p5']:,.0f}", "| P95:", f"{ts['p95']:,.0f}")
    with st.expander("Max drawdown distribution", expanded=False):
        st.write("Mean:", _format_pct(dd["mean"]), "| P5:", _format_pct(dd["p5"]), "| P95:", _format_pct(dd["p95"]))
    st.markdown("")
    st.divider()

    # ----- 6. Charts -----
    paths = result["paths"]
    time_axis = result["time_axis"]
    pcurves = result["percentile_curves"]
    n_display = min(200, paths.shape[0])

    st.subheader("Simulated paths (spaghetti)")
    with st.expander("? **What is this chart?**", expanded=False):
        st.markdown("""
        **Purpose:** Show a sample of possible wealth trajectories over time.

        Each **line** is one simulated path (GBM). The spread shows how much outcomes can diverge.  
        **How to read:** Paths that stay high indicate favourable scenarios; paths that drop show downside risk. This is forward-looking simulation, not historical data.
        """)
    fig1 = go.Figure()
    for i in range(n_display):
        fig1.add_trace(
            go.Scatter(
                x=time_axis,
                y=paths[i, :],
                mode="lines",
                line=dict(width=0.8, color="rgba(31, 119, 180, 0.15)"),
                showlegend=False,
            )
        )
    fig1.update_layout(
        title="Sample of simulated wealth paths",
        xaxis_title="Years",
        yaxis_title="Wealth",
        showlegend=False,
        margin=dict(t=50),
        hovermode="x unified",
    )
    fig1.update_xaxes(showgrid=True, gridwidth=1, gridcolor="rgba(128,128,128,0.2)")
    fig1.update_yaxes(showgrid=True, gridwidth=1, gridcolor="rgba(128,128,128,0.2)")
    st.plotly_chart(fig1, use_container_width=True)
    st.markdown("")

    st.subheader("Fan chart (percentiles)")
    with st.expander("? **What is this chart?**", expanded=False):
        st.markdown("""
        **Purpose:** Summarize the range of outcomes at each point in time.

        **Bands:** 5th–95th percentile (outer), 25th–75th (inner), and median (50th) line.  
        **How to read:** At any year, the band shows where most paths lie; the median is the typical path. Useful to communicate uncertainty over the horizon.
        """)
    fig2 = go.Figure()
    fig2.add_trace(
        go.Scatter(
            x=time_axis,
            y=pcurves[95],
            mode="lines",
            line=dict(width=0),
            fill=None,
            showlegend=False,
        )
    )
    fig2.add_trace(
        go.Scatter(
            x=time_axis,
            y=pcurves[5],
            mode="lines",
            line=dict(width=0),
            fill="tonexty",
            fillcolor="rgba(31, 119, 180, 0.2)",
            name="P5–P95",
        )
    )
    fig2.add_trace(
        go.Scatter(
            x=time_axis,
            y=pcurves[75],
            mode="lines",
            line=dict(width=0),
            fill=None,
            showlegend=False,
        )
    )
    fig2.add_trace(
        go.Scatter(
            x=time_axis,
            y=pcurves[25],
            mode="lines",
            line=dict(width=0),
            fill="tonexty",
            fillcolor="rgba(31, 119, 180, 0.35)",
            name="P25–P75",
        )
    )
    fig2.add_trace(
        go.Scatter(
            x=time_axis,
            y=pcurves[50],
            mode="lines",
            line=dict(color="rgb(214, 39, 40)", width=2),
            name="Median (P50)",
        )
    )
    fig2.update_layout(
        title="Wealth percentiles over time",
        xaxis_title="Years",
        yaxis_title="Wealth",
        showlegend=True,
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        margin=dict(t=50),
        hovermode="x unified",
    )
    fig2.update_xaxes(showgrid=True, gridwidth=1, gridcolor="rgba(128,128,128,0.2)")
    fig2.update_yaxes(showgrid=True, gridwidth=1, gridcolor="rgba(128,128,128,0.2)")
    st.plotly_chart(fig2, use_container_width=True)
    st.markdown("")

    st.subheader("Terminal wealth distribution")
    with st.expander("? **What is this chart?**", expanded=False):
        st.markdown("""
        **Purpose:** Show the **distribution** of wealth at the end of the horizon.

        **Histogram:** Each bar is the count of paths ending in that wealth range.  
        **Vertical lines:** VaR and CVaR at the chosen confidence (e.g. 95%): the left tail shows the worst outcomes.
        """)
    terminal = paths[:, -1]
    fig3 = go.Figure()
    fig3.add_trace(go.Histogram(x=terminal, nbinsx=50, name="Terminal wealth", marker_color="rgba(31, 119, 180, 0.7)"))
    fig3.add_vline(x=vc["var_level"], line_dash="dash", line_color="red", annotation_text="VaR")
    fig3.add_vline(x=vc["cvar_level"], line_dash="dot", line_color="darkred", annotation_text="CVaR")
    fig3.update_layout(
        title="Distribution of terminal wealth",
        xaxis_title="Terminal wealth",
        yaxis_title="Count",
        showlegend=False,
        margin=dict(t=50),
        hovermode="x unified",
    )
    fig3.update_xaxes(showgrid=True, gridwidth=1, gridcolor="rgba(128,128,128,0.2)")
    fig3.update_yaxes(showgrid=True, gridwidth=1, gridcolor="rgba(128,128,128,0.2)")
    st.plotly_chart(fig3, use_container_width=True)
    st.markdown("")

    st.subheader("Max drawdown distribution")
    with st.expander("? **What is this chart?**", expanded=False):
        st.markdown("""
        **Purpose:** Show how bad the **worst peak-to-trough decline** can be along each path.

        **Histogram:** Each path has one max drawdown (e.g. -25%); the chart shows the distribution across paths.  
        **How to read:** Helps assess the risk of seeing a large temporary loss during the horizon.
        """)
    max_dd = result["max_drawdowns"]
    fig4 = go.Figure()
    fig4.add_trace(
        go.Histogram(
            x=max_dd * 100,
            nbinsx=40,
            name="Max drawdown %",
            marker_color="rgba(214, 39, 40, 0.6)",
        )
    )
    fig4.update_layout(
        title="Distribution of maximum drawdown per path",
        xaxis_title="Max drawdown (%)",
        yaxis_title="Count",
        showlegend=False,
        margin=dict(t=50),
        hovermode="x unified",
    )
    fig4.update_xaxes(showgrid=True, gridwidth=1, gridcolor="rgba(128,128,128,0.2)")
    fig4.update_yaxes(showgrid=True, gridwidth=1, gridcolor="rgba(128,128,128,0.2)")
    st.plotly_chart(fig4, use_container_width=True)
    st.markdown("")
    st.divider()

    # ----- 7. Export -----
    with st.expander("Export results (CSV)", expanded=False):
        st.caption("Download key metrics and optional percentile curves for your records.")
        rows = [
            ["initial_wealth", str(initial_wealth)],
            ["mu", str(mu_input)],
            ["sigma", str(sigma_input)],
            ["T_years", str(horizon_years)],
            ["n_paths", str(n_paths)],
            ["mean_terminal", f"{ts['mean']:.4f}"],
            ["median_terminal", f"{ts['p50']:.4f}"],
            ["var_level", f"{vc['var_level']:.4f}"],
            ["var_pct", f"{vc['var_pct']:.6f}"],
            ["cvar_level", f"{vc['cvar_level']:.4f}"],
            ["cvar_pct", f"{vc['cvar_pct']:.6f}"],
            ["median_max_drawdown", f"{dd['p50']:.6f}"],
        ]
        if result.get("prob_shortfall") is not None:
            rows.append(["prob_shortfall", f"{result['prob_shortfall']:.6f}"])
        buf = io.StringIO()
        buf.write("metric,value\n")
        for row in rows:
            buf.write(f"{row[0]},{row[1]}\n")
        csv_bytes = buf.getvalue().encode("utf-8")
        st.download_button(
            label="Download results (CSV)",
            data=csv_bytes,
            file_name="monte_carlo_results.csv",
            mime="text/csv",
            key="mc_export_csv",
        )
