"""Beta (CAPM) tab."""
import io
import numpy as np
import streamlit as st
import pandas as pd
import plotly.graph_objects as go

from config import DEFAULT_ASSET_TICKER, DEFAULT_RISK_FREE_RATE
from services import capm_service
from utils.plot_config import HOVERLABEL


def _format_pct(value: float, decimals: int = 2) -> str:
    """Format as percentage (e.g. 0.0523 -> 5.23%)."""
    return f"{value * 100:.{decimals}f}%"


def _format_pvalue(p: float) -> str:
    """Format p-value (3 decimals or scientific if very small)."""
    if p < 0.0001:
        return f"{p:.2e}"
    return f"{p:.3f}"


@st.cache_data(ttl=3600)
def _load_capm_results(asset_ticker: str, benchmark_ticker: str, start_str: str, end_str: str, rf: float, window: int):
    """Cached wrapper: run full CAPM analysis (download + regression + rolling beta)."""
    start = pd.Timestamp(start_str)
    end = pd.Timestamp(end_str)
    return capm_service.run_capm_analysis(
        asset_ticker=asset_ticker,
        benchmark_ticker=benchmark_ticker,
        start_date=start,
        end_date=end,
        risk_free_rate=rf,
        rolling_window=window,
    )


def render():
    """Render Beta tab."""
    # ----- 1. Header -----
    st.header("CAPM & Beta Analysis")
    st.markdown("Estimate Beta (β), cost of equity (Kₑ), and Jensen's Alpha vs your chosen benchmark.")
    with st.expander("? **What is this tab for?**", expanded=False):
        st.markdown("""
        **Purpose:** This tab answers three questions: (1) *How risky is this stock compared to the market?* (Beta),  
        (2) *What return do investors expect from it?* (cost of equity), and (3) *Did it beat or lag the market given its risk?* (Jensen's Alpha).

        - **Beta** tells you how much the stock tends to move when the market moves (e.g. Beta 1.2 → +1.2% when the market is +1%).  
        - **Cost of equity** is the minimum return required to hold the stock; it is used to value companies and decide if a project is worth it.  
        - **Jensen's Alpha** measures outperformance or underperformance vs the market after adjusting for risk.

        **Use it when:** you want to understand a stock's risk vs the market, value a company, or judge past performance (e.g. fund managers).
        """)
    st.markdown("")

    # ----- Parameters (in-tab: only CAPM inputs visible here) -----
    with st.expander("Parameters", expanded=True):
        st.caption("Set asset, benchmark, period and risk-free rate. These inputs apply only to this tab.")
        st.markdown("")
        c1, c2, c3 = st.columns(3)
        with c1:
            asset_input = st.text_input(
                "Asset ticker or company name",
                value=DEFAULT_ASSET_TICKER,
                placeholder="e.g. AAPL, Apple, Microsoft",
                help="Enter a ticker (AAPL, MSFT) or a company name (Apple, Microsoft). Names are resolved to the main listed ticker.",
            ).strip() or DEFAULT_ASSET_TICKER
            resolved_ticker, resolved_name, resolved_exchange = capm_service.resolve_asset_ticker(asset_input)
            asset_ticker = resolved_ticker if resolved_ticker else asset_input

            benchmark_label = st.selectbox(
                "Benchmark",
                options=list(capm_service.BENCHMARK_OPTIONS.keys()),
                index=0,
                help="Market index or ETF for CAPM regression.",
            )
            benchmark_ticker = capm_service.BENCHMARK_OPTIONS[benchmark_label]
        with c2:
            risk_free_rate = st.number_input(
                "Risk-free rate (Rf)",
                min_value=0.0,
                max_value=0.30,
                value=float(DEFAULT_RISK_FREE_RATE),
                step=0.005,
                format="%.3f",
                help="Annual rate, e.g. 0.05 for 5%.",
            )

            period_preset = st.selectbox(
                "Period",
                options=["5 years", "3 years", "2 years", "1 year"],
                index=0,
                help="Historical data range (at least 5 years recommended).",
            )
            years = {"5 years": 5, "3 years": 3, "2 years": 2, "1 year": 1}[period_preset]
            end = pd.Timestamp.now()
            start = end - pd.DateOffset(years=years)
            start_str = start.strftime("%Y-%m-%d")
            end_str = end.strftime("%Y-%m-%d")
        with c3:
            rolling_window = st.selectbox(
                "Rolling Beta window",
                options=capm_service.ROLLING_WINDOWS,
                index=2,
                format_func=lambda x: f"{x} days",
                help="Window size for rolling Beta (252 ≈ 1 year).",
            )
        st.markdown("")
    st.divider()

    # ----- 2. Context: asset + period (compact block) -----
    asset_name = resolved_name
    asset_exchange = resolved_exchange
    if not asset_name or not asset_exchange:
        fetched_name, fetched_exchange = capm_service.get_asset_display_info(asset_ticker)
        asset_name = asset_name or fetched_name
        asset_exchange = asset_exchange or fetched_exchange

    with st.spinner("Loading market data…"):
        result = _load_capm_results(
            asset_ticker, benchmark_ticker, start_str, end_str, risk_free_rate, rolling_window
        )

    if not result["success"]:
        st.error(result.get("error", "An error occurred."))
        return

    reg = result["regression"]
    ke = result["cost_of_equity"]
    adj_beta = result["adjusted_beta"]
    asset_ret = result["asset_returns"]
    market_ret = result["market_returns"]
    rolling_beta_series = result["rolling_beta"]
    e_rm = result.get("expected_market_return_annual")
    treynor = result.get("treynor_ratio")

    # Context block: asset used + period (one compact box)
    if asset_name or asset_exchange:
        parts = [p for p in [asset_name, asset_exchange] if p]
        st.info(f"**Asset:** {' | '.join(parts)} ({asset_ticker})")
    if resolved_name and asset_input.upper() != asset_ticker:
        st.caption(f"Resolved **{asset_input}** → **{asset_ticker}**")
    st.caption(f"**Data:** {start_str} → {end_str} · **E(Rm)** for Kₑ: {e_rm * 100:.1f}% (historical mean)")
    st.markdown("")

    # ----- 3. Key results -----
    st.subheader("Key results")
    with st.expander("? **What do these numbers mean?**", expanded=False):
        st.markdown("""
        This block summarizes the **CAPM regression** in five figures. Hover the **?** next to each metric for a detailed explanation.

        - **Beta (β):** Slope of the regression (asset return vs market return). How much the stock moves when the market moves 1%.  
        - **Adjusted Beta:** Smoothed Beta used in practice (Bloomberg formula).  
        - **Jensen's Alpha (α):** Annualized excess return vs what CAPM would predict; positive = outperformed.  
        - **Cost of equity (Kₑ):** Required return given Beta; used to value the company or discount cash flows.  
        - **R²:** Share of the asset's variance explained by the market; high = moves with the market, low = more idiosyncratic.
        """)

    c1, c2, c3, c4, c5 = st.columns(5)
    with c1:
        st.metric(
            "Beta (β)",
            f"{reg['beta']:.3f}",
            help="Purpose: measure how risky this stock is compared to the market. If the market goes up 1%, a Beta of 1.2 means the stock tends to go up 1.2%; a Beta of 0.8 means it goes up only 0.8%. Use it to understand volatility and to build a diversified portfolio (high Beta = more market risk).",
        )
    with c2:
        st.metric(
            "Adjusted Beta",
            f"{adj_beta:.3f}",
            help="Purpose: a more stable estimate of Beta for real-world use. Raw Beta jumps around with the data; Adjusted Beta pulls it toward 1 (the market). Analysts use it for valuation and cost of equity so one noisy period doesn't distort the result.",
        )
    with c3:
        alpha_pct = reg["alpha"] * 252 * 100  # annualized in %
        st.metric(
            "Jensen's Alpha (α)",
            f"{alpha_pct:.2f}%",
            help="Purpose: see if the stock did better or worse than expected given its risk (Beta). Positive Alpha = it beat the market for the risk taken; negative = it underperformed. Use it to judge past performance and compare fund managers or stocks.",
        )
    with c4:
        st.metric(
            "Cost of equity (Kₑ)",
            _format_pct(ke),
            help="Purpose: the minimum return investors expect from this stock given its risk. Companies use it to decide if a project is worth it (discount future cash flows). Investors use it to check if the stock is cheap or expensive vs that required return.",
        )
    with c5:
        st.metric(
            "R²",
            f"{reg['r_squared']*100:.2f}%",
            help="Purpose: how much of the stock’s moves are explained by the market vs its own news. High R² = moves mostly with the market; low R² = more company-specific moves. Use it to see how much diversification you get from adding this stock.",
        )

    if reg["alpha"] > 0:
        st.caption("Jensen's Alpha > 0: stock has outperformed the CAPM (positive alpha).")
    else:
        st.caption("Jensen's Alpha ≤ 0: stock has underperformed or matched the CAPM.")

    p_val = reg.get("p_value_beta")
    if p_val is not None and p_val > 0.05:
        st.warning("Beta is not statistically significant (p > 0.05). Use with caution.")
    if reg["r_squared"] < 0.2:
        st.warning(f"Low R² ({reg['r_squared']*100:.1f}%): only a small share of variance is explained by the market. Beta may be unstable.")

    st.markdown("")
    st.divider()
    st.subheader("Statistics & quality")
    st.caption("Expand for t-stat, p-value, confidence interval, Treynor ratio, and residual plot.")
    st.markdown("")

    with st.expander("Regression details (t-stat, p-value, 95% CI, Treynor)", expanded=False):
        st.caption("Statistical quality of the Beta estimate. **t-stat** / **p-value**: is Beta reliable? **95% CI**: range for the true Beta. **Treynor**: excess return per unit of market risk.")
        st.markdown("")
        t_stat = reg.get("t_stat_beta")
        st.write(f"**t-stat (Beta):** {t_stat if t_stat is not None else '—'}")
        p_val_beta = reg.get("p_value_beta")
        st.write(f"**p-value (Beta):** {_format_pvalue(p_val_beta) if p_val_beta is not None else '—'}")
        if reg.get("ci_low") is not None and reg.get("ci_high") is not None:
            st.write(f"**95% CI for β:** [{reg['ci_low']:.3f}, {reg['ci_high']:.3f}]")
        st.write(f"**Observations:** {reg['n_obs']}")
        if treynor is not None:
            st.write(f"**Treynor ratio (annualized):** {treynor:.4f}")

    resid = reg.get("resid")
    if resid is not None and len(resid) > 0:
        with st.expander("Residuals (regression quality)", expanded=False):
            st.caption("Residuals = actual minus fitted return. Ideally they scatter randomly around 0; patterns suggest the model may be misspecified.")
            fitted_vals = reg["fitted_excess_asset"]
            fig_res = go.Figure()
            fig_res.add_trace(
                go.Scatter(
                    x=fitted_vals.values * 100,
                    y=resid.values * 100,
                    mode="markers",
                    name="Residuals",
                    marker=dict(size=3, color="rgba(128, 128, 128, 0.5)", line=dict(width=0)),
                    hovertemplate="<b>Residuals</b><br>Fitted return: %{x:.2f}%<br>Residual: %{y:.2f}%<extra></extra>",
                )
            )
            fig_res.add_hline(y=0, line_dash="dash", line_color="gray", opacity=0.7)
            fig_res.update_layout(
                title="Residuals vs fitted (excess return %)",
                xaxis_title="Fitted excess return (%)",
                yaxis_title="Residual (%)",
                margin=dict(t=40, b=40),
                height=280,
                showlegend=False,
                hoverlabel=HOVERLABEL,
            )
            fig_res.update_xaxes(showgrid=True, gridwidth=1, gridcolor="rgba(128,128,128,0.2)")
            fig_res.update_yaxes(showgrid=True, gridwidth=1, gridcolor="rgba(128,128,128,0.2)")
            st.plotly_chart(fig_res, use_container_width=True)

    st.markdown("")
    st.divider()

    # ----- 5. Charts -----
    st.subheader("Daily returns: Asset vs Market")
    with st.expander("? **What is this chart?**", expanded=False):
        st.markdown("""
        **Purpose:** Visualize how the asset's daily returns move with the market's daily returns (both in excess of the risk-free rate).

        - **Dots (scatter):** Each point = one day. X = market excess return (%), Y = asset excess return (%).  
        - **Red line:** CAPM regression line (slope = Beta, intercept = Alpha). Points close to the line = returns well explained by the market; scattered points = more idiosyncratic moves.  
        - **How to read:** Steep line → high Beta; line above the origin on average → positive Alpha (outperformance).
        """)
    fitted_market = reg["fitted_excess_market"]
    fitted_asset = reg["fitted_excess_asset"]
    # Align indices for scatter
    excess_asset = asset_ret - risk_free_rate / 252
    excess_market = market_ret - risk_free_rate / 252
    common_idx = excess_asset.index.intersection(excess_market.index)
    x_scatter = excess_market.loc[common_idx].values * 100  # in %
    y_scatter = excess_asset.loc[common_idx].values * 100
    x_line = fitted_market.values * 100
    y_line = fitted_asset.values * 100
    # Sort for line
    order = np.argsort(x_line)
    x_line = x_line[order]
    y_line = y_line[order]

    fig1 = go.Figure()
    fig1.add_trace(
        go.Scatter(
            x=x_scatter,
            y=y_scatter,
            mode="markers",
            name="Excess returns",
            marker=dict(size=4, color="rgba(31, 119, 180, 0.5)", line=dict(width=0)),
            hovertemplate="<b>Excess returns</b><br>Market: %{x:.2f}%<br>Asset: %{y:.2f}%<extra></extra>",
        )
    )
    fig1.add_trace(
        go.Scatter(
            x=x_line,
            y=y_line,
            mode="lines",
            name="CAPM regression",
            line=dict(color="rgb(214, 39, 40)", width=2),
            hovertemplate="<b>CAPM regression</b><br>Market: %{x:.2f}%<br>Fitted asset: %{y:.2f}%<extra></extra>",
        )
    )
    fig1.update_layout(
        title="Daily excess returns: Asset vs Market",
        xaxis_title="Market excess return (%)",
        yaxis_title="Asset excess return (%)",
        showlegend=True,
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        margin=dict(t=50),
        hovermode="closest",
        hoverlabel=HOVERLABEL,
    )
    fig1.update_xaxes(showgrid=True, gridwidth=1, gridcolor="rgba(128,128,128,0.2)")
    fig1.update_yaxes(showgrid=True, gridwidth=1, gridcolor="rgba(128,128,128,0.2)")
    st.plotly_chart(fig1, use_container_width=True)

    st.markdown("")

    st.subheader(f"Rolling Beta ({rolling_window}-day window)")
    with st.expander("? **What is this chart?**", expanded=False):
        st.markdown("""
        **Purpose:** Show how **Beta changes over time**. Beta is re-estimated on a rolling window (e.g. last 252 days); each point is the Beta for that window.

        - **Green line:** Rolling Beta. If it trends up, the stock has become more sensitive to the market; if it trends down, less sensitive.  
        - **Gray dashed line at 1:** Market level (Beta = 1). Above = more volatile than the market; below = less volatile.  
        - **Use it when:** You want to see if risk exposure has shifted (e.g. after a merger or change in business).
        """)
    if rolling_beta_series is not None and len(rolling_beta_series) > 0:
        fig2 = go.Figure()
        fig2.add_trace(
            go.Scatter(
                x=rolling_beta_series.index,
                y=rolling_beta_series.values,
                mode="lines",
                name="Rolling Beta",
                line=dict(color="rgb(44, 160, 44)", width=2),
                hovertemplate="<b>Rolling Beta</b><br>Date: %{x|%Y-%m-%d}<br>Beta: %{y:.3f}<extra></extra>",
            )
        )
        fig2.add_hline(y=1.0, line_dash="dash", line_color="gray", opacity=0.7)
        fig2.update_layout(
            title=f"Rolling Beta ({rolling_window}-day window)",
            xaxis_title="Date",
            yaxis_title="Beta",
            showlegend=True,
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
            margin=dict(t=50),
            hovermode="x unified",
            hoverlabel=HOVERLABEL,
        )
        fig2.update_xaxes(showgrid=True, gridwidth=1, gridcolor="rgba(128,128,128,0.2)")
        fig2.update_yaxes(showgrid=True, gridwidth=1, gridcolor="rgba(128,128,128,0.2)")
        st.plotly_chart(fig2, use_container_width=True)
    else:
        st.info("Not enough data for rolling Beta with the selected window.")

    st.markdown("")
    st.divider()

    # ----- 6. Export -----
    with st.expander("Export results (CSV)", expanded=False):
        st.caption("Download the key CAPM figures and the data period in a CSV file for your records or further analysis.")
        alpha_ann = reg["alpha"] * 252 * 100
        rows = [
            ["asset_ticker", asset_ticker],
            ["benchmark_ticker", benchmark_ticker],
            ["start_date", start_str],
            ["end_date", end_str],
            ["beta", f"{reg['beta']:.6f}"],
            ["adjusted_beta", f"{adj_beta:.6f}"],
            ["jensen_alpha_pct_annualized", f"{alpha_ann:.4f}"],
            ["cost_of_equity", f"{ke:.6f}"],
            ["r_squared", f"{reg['r_squared']:.6f}"],
            ["treynor_ratio", f"{treynor:.6f}" if treynor is not None else ""],
            ["expected_market_return_annual", f"{e_rm:.6f}" if e_rm is not None else ""],
            ["n_observations", str(reg["n_obs"])],
        ]
        buf = io.StringIO()
        buf.write("metric,value\n")
        for row in rows:
            buf.write(f"{row[0]},{row[1]}\n")
        csv_bytes = buf.getvalue().encode("utf-8")
        st.download_button(
            label="Download results (CSV)",
            data=csv_bytes,
            file_name="capm_results.csv",
            mime="text/csv",
            key="capm_export_csv",
        )
