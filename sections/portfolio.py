"""Portfolio (Markowitz) tab."""
import io
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from config import DEFAULT_MARKOWITZ_TICKERS, DEFAULT_RISK_FREE_RATE
from services import capm_service, markowitz_service
from utils.plot_config import HOVERLABEL


def _format_pct(value: float, decimals: int = 2) -> str:
    """Format as percentage (e.g. 0.0523 -> 5.23%)."""
    return f"{value * 100:.{decimals}f}%"


@st.cache_data(ttl=3600)
def _load_markowitz_results(
    tickers: tuple[str, ...],
    start_str: str,
    end_str: str,
    rf: float,
    objective: str,
    target_return: float | None,
    long_only: bool,
    max_weight_per_asset: float,
    max_gross_exposure: float | None,
    benchmark_ticker: str | None,
):
    """Cached wrapper: run full Markowitz analysis (download, optimize, frontier, random cloud, benchmark, VaR/CVaR)."""
    start = pd.Timestamp(start_str)
    end = pd.Timestamp(end_str)
    return markowitz_service.run_markowitz_analysis(
        tickers=list(tickers),
        start_date=start,
        end_date=end,
        risk_free_rate=rf,
        objective=objective,
        target_return=target_return,
        long_only=long_only,
        max_weight_per_asset=max_weight_per_asset,
        max_gross_exposure=max_gross_exposure,
        benchmark_ticker=benchmark_ticker or None,
        n_frontier_points=50,
        n_random_portfolios=500,
    )


def render():
    """Render Portfolio tab."""
    # ----- 1. Header -----
    st.header("Modern Portfolio Theory (Markowitz)")
    st.markdown("Find the optimal allocation: efficient frontier and max Sharpe.")
    with st.expander("? **What is this tab for?**", expanded=False):
        st.markdown("""
        **Purpose:** This tab helps you build a *portfolio* (a mix of several assets) that offers the best trade-off between return and risk.  
        The key idea is **diversification**: by combining assets that don't move in lockstep, you can often *lower risk without giving up too much return*.

        - The **efficient frontier** is the set of portfolios that maximize return for each level of risk (or minimize risk for each level of return).  
        - The **Sharpe Ratio** measures how much extra return you get per unit of risk; higher is better.  
        - The **optimal portfolio** is the mix that maximizes the Sharpe Ratio (or matches your target risk or return).

        **Use it when:** you want to allocate money across several stocks or ETFs, reduce volatility, or find the best risk-adjusted mix.
        """)
    st.markdown("")

    # ----- Parameters (in-tab: only Markowitz inputs visible here) -----
    with st.expander("Parameters", expanded=True):
        st.caption("Set portfolio tickers, period, objective and constraints. These inputs apply only to this tab.")
        st.markdown("")
        c1, c2 = st.columns([1, 1])
        with c1:
            default_tickers_str = "\n".join(DEFAULT_MARKOWITZ_TICKERS)
            tickers_text = (
                st.text_area(
                    "Tickers or company names (one per line)",
                    value=default_tickers_str,
                    placeholder="AAPL\nNvidia\nLVMH\nGOOGL",
                    help="Enter one ticker (AAPL, NVDA) or company name (Nvidia, LVMH) per line. Names are resolved to the main listed ticker. At least 2 assets required.",
                    height=120,
                    key="markowitz_tickers",
                )
                .strip()
                or default_tickers_str
            )
            raw_lines = [t.strip() for t in tickers_text.splitlines() if t.strip()]
            tickers = []
            seen = set()
            for q in raw_lines:
                resolved_ticker, _, _ = capm_service.resolve_asset_ticker(q)
                sym = (resolved_ticker or q).strip().upper()
                if sym and sym not in seen:
                    tickers.append(sym)
                    seen.add(sym)
            if not tickers:
                tickers = list(DEFAULT_MARKOWITZ_TICKERS)

            risk_free_rate = st.number_input(
                "Risk-free rate (Rf)",
                min_value=0.0,
                max_value=0.30,
                value=float(DEFAULT_RISK_FREE_RATE),
                step=0.005,
                format="%.3f",
                help="Annual rate, e.g. 0.05 for 5%.",
                key="markowitz_rf",
            )

            period_preset = st.selectbox(
                "Period",
                options=["5 years", "3 years", "2 years", "1 year"],
                index=0,
                help="Historical data range (at least 5 years recommended).",
                key="markowitz_period",
            )
            years = {"5 years": 5, "3 years": 3, "2 years": 2, "1 year": 1}[period_preset]
            end_dt = pd.Timestamp.now()
            start_dt = end_dt - pd.DateOffset(years=years)
            start_str = start_dt.strftime("%Y-%m-%d")
            end_str = end_dt.strftime("%Y-%m-%d")
        with c2:
            objective_options = ["max_sharpe", "min_variance", "target_return"]
            objective_labels = ["Maximize Sharpe Ratio", "Minimize variance", "Target return"]
            objective_idx = st.selectbox(
                "Optimization objective",
                options=range(len(objective_options)),
                index=0,
                format_func=lambda i: objective_labels[i],
                help="Max Sharpe: best risk-adjusted return. Min variance: lowest risk. Target return: minimize risk for a given return.",
                key="markowitz_objective",
            )
            objective = objective_options[objective_idx]

            target_return = None
            if objective == "target_return":
                target_return = st.number_input(
                    "Target return (annual, %)",
                    min_value=0.0,
                    max_value=50.0,
                    value=10.0,
                    step=0.5,
                    format="%.1f",
                    help="Target annual return in %. Portfolio will minimize volatility for this return.",
                    key="markowitz_target_return",
                )
                target_return = target_return / 100.0  # convert to decimal

            benchmark_label = st.selectbox(
                "Benchmark (for comparison)",
                options=["None"] + list(capm_service.BENCHMARK_OPTIONS.keys()),
                index=0,
                help="Compare optimal portfolio to this index. Shows alpha, tracking error, Information ratio and benchmark point on the chart.",
                key="markowitz_benchmark",
            )
            benchmark_ticker = None if benchmark_label == "None" else capm_service.BENCHMARK_OPTIONS[benchmark_label]

            with st.expander("Constraints", expanded=False):
                long_only = st.checkbox(
                    "Long-only (no short selling)",
                    value=True,
                    help="Weights between 0% and 100%. If unchecked, short positions are allowed.",
                    key="markowitz_long_only",
                )
                max_weight_pct = st.number_input(
                    "Max weight per asset (%)",
                    min_value=5,
                    max_value=100,
                    value=100,
                    step=5,
                    help="Cap each asset weight (e.g. 40% = no single asset above 40%). 100% = no cap.",
                    key="markowitz_max_weight",
                )
                st.caption("When short selling is allowed:")
                max_gross_pct = st.number_input(
                    "Max gross exposure (%)",
                    min_value=100,
                    max_value=500,
                    value=200,
                    step=25,
                    help="Cap on sum of absolute weights (longs + shorts). E.g. 200% = total exposure at most 200% of capital. Only applies when short selling is allowed.",
                    key="markowitz_max_gross",
                )
            max_weight_per_asset = max_weight_pct / 100.0
            max_gross_exposure = None if long_only else (max_gross_pct / 100.0)
        st.markdown("")
    st.divider()

    # ----- 2. Context: run analysis -----
    with st.spinner("Loading market data and optimizing portfolio…"):
        result = _load_markowitz_results(
            tuple(tickers),
            start_str,
            end_str,
            risk_free_rate,
            objective,
            target_return,
            long_only,
            max_weight_per_asset,
            max_gross_exposure,
            benchmark_ticker,
        )

    if not result["success"]:
        st.error(result.get("error", "An error occurred."))
        return

    tickers_used = result["tickers"]
    n_obs = result["n_obs"]

    # Context block
    st.info(f"**Assets:** {', '.join(tickers_used)}")
    st.caption(f"**Data:** {start_str} → {end_str} · **Observations:** {n_obs}")
    st.markdown("")

    # ----- 3. Key results -----
    st.subheader("Key results")
    with st.expander("? **What do these numbers mean?**", expanded=False):
        st.markdown("""
        This block summarizes the **optimal portfolio** from Markowitz optimization. Hover the **?** next to each metric for a detailed explanation.

        - **Sharpe Ratio:** Excess return per unit of risk; higher means better risk-adjusted performance.  
        - **Portfolio volatility:** Annualized standard deviation of returns; lower = less risk.  
        - **Expected return:** Annualized mean return of the portfolio.  
        - **Diversification ratio:** How much diversification reduces risk vs holding individual volatilities; above 1 means diversification helps.  
        - **Optimal weights:** Recommended allocation (in %) across assets; sum to 100%.  
        - **VaR 95% (1d):** Worst daily return in the worst 5% of days (historical).  
        - **CVaR 95% (1d):** Average daily return in those worst 5% of days (Expected Shortfall).
        """)

    div_ratio = markowitz_service.diversification_ratio(result["weights"], result["cov_matrix"])
    var_95 = result.get("var_95_daily")
    cvar_95 = result.get("cvar_95_daily")

    c1, c2, c3, c4, c5, c6 = st.columns(6)
    with c1:
        st.metric(
            "Sharpe Ratio",
            f"{result['sharpe_ratio']:.3f}",
            help="Purpose: measure risk-adjusted return. Higher = more return per unit of risk. Use it to compare portfolios or choose the best mix; a Sharpe above 1 is often considered good.",
        )
    with c2:
        st.metric(
            "Portfolio volatility",
            _format_pct(result["volatility"]),
            help="Purpose: annualized risk (standard deviation of returns). Lower = less volatility. Use it to gauge how much the portfolio value may swing.",
        )
    with c3:
        st.metric(
            "Expected return",
            _format_pct(result["expected_return"]),
            help="Purpose: annualized mean return of the portfolio. Use it to set return expectations and compare with the risk-free rate.",
        )
    with c4:
        st.metric(
            "Diversification ratio",
            f"{div_ratio:.3f}",
            help="Purpose: ratio of weighted asset volatilities to portfolio volatility. Above 1 means diversification reduced risk. Use it to see how much benefit you get from mixing assets.",
        )
    with c5:
        st.metric(
            "VaR 95% (1d)",
            f"{var_95 * 100:.2f}%" if var_95 is not None else "—",
            help="Purpose: worst daily return expected in 5% of days (historical). E.g. -1.5% means on the worst 5% of days the portfolio lost at least 1.5%. Use it to size downside risk.",
        )
    with c6:
        st.metric(
            "CVaR 95% (1d)",
            f"{cvar_95 * 100:.2f}%" if cvar_95 is not None else "—",
            help="Purpose: average daily return in the worst 5% of days (Expected Shortfall). Typically more negative than VaR. Use it to quantify tail risk.",
        )

    if objective == "max_sharpe":
        st.caption("Portfolio maximizes Sharpe Ratio for the selected period and risk-free rate.")
    elif objective == "min_variance":
        st.caption("Portfolio minimizes variance (lowest risk) for the selected period.")
    else:
        st.caption(f"Portfolio minimizes volatility for target return {target_return * 100:.1f}%.")

    if result.get("benchmark_return_annual") is not None:
        st.markdown("**vs Benchmark**")
        with st.expander("? **What do Alpha, Tracking error, Information ratio mean?**", expanded=False):
            st.markdown("""
            - **Alpha:** Excess return of your portfolio over the benchmark. Positive = you beat the market.  
            - **Tracking error:** Volatility of (portfolio return − benchmark return). Low = your portfolio moves with the market.  
            - **Information ratio:** Alpha / Tracking error. Higher = more excess return per unit of active risk.
            """)
        b1, b2, b3, b4, b5 = st.columns(5)
        with b1:
            st.metric("Benchmark return", _format_pct(result["benchmark_return_annual"]), help="Annualized return of the selected benchmark over the same period.")
        with b2:
            st.metric("Benchmark volatility", _format_pct(result["benchmark_vol_annual"]), help="Annualized volatility of the benchmark.")
        with b3:
            alpha = result["alpha_vs_benchmark"]
            st.metric("Alpha", _format_pct(alpha), help="Excess return of the portfolio over the benchmark. Positive = portfolio beat the benchmark.")
        with b4:
            st.metric("Tracking error", _format_pct(result["tracking_error_annual"]), help="Annualized volatility of (portfolio return − benchmark return). Lower = portfolio tracks the benchmark more closely.")
        with b5:
            st.metric("Information ratio", f"{result['information_ratio']:.3f}", help="Alpha divided by tracking error. Higher = more excess return per unit of active risk.")
        st.markdown("")

    if n_obs < 252:
        st.warning("Less than one year of data: results may be unstable.")
    if not long_only:
        st.warning("Short selling is allowed: weights can be negative and leverage very high; optimal portfolio may be extreme (e.g. 300% long / 200% short). Use with caution.")
    if result.get("condition_number") is not None and result["condition_number"] > 1e6:
        st.warning("High covariance condition number: optimization may be numerically sensitive.")

    st.markdown("")
    st.subheader("Optimal weights")
    st.caption("Click the block below (question mark) to see how to read weights with an example (e.g. 100k capital).")
    with st.expander("❓ How to read these weights? (e.g. with 100k capital)", expanded=False):
        st.markdown("""
        Weights are in **% of your capital**. The sum of all weights = **100%** (net exposure).

        **Example with 100,000 €:**
        - **Positive weight** (e.g. +45%) = **long**: you hold 45,000 € of that asset.
        - **Negative weight** (e.g. −20%) = **short**: you have sold short 20,000 € (borrowed shares sold for 20,000 €).

        **Totals:** Total long (sum of positive weights) + total short (sum of negative weights) can exceed 100% (leverage).  
        Your **capital** = 100% = longs − shorts (net). In theory, short proceeds finance part of the longs; in practice, margin rules limit how much you can reuse.
        """)
    weights_pct = result["weights"] * 100
    df_weights = pd.DataFrame({"Ticker": tickers_used, "Weight (%)": np.round(weights_pct, 2)})
    st.dataframe(df_weights, use_container_width=True, hide_index=True)
    st.markdown("")
    st.divider()

    # ----- 4. Statistics & quality -----
    st.subheader("Statistics & quality")
    st.caption("Correlation matrix (always visible), then expand for covariance condition number and risk decomposition.")
    st.markdown("")

    st.markdown("**Correlation matrix**")
    st.caption("Lower correlations between assets allow better risk reduction through diversification.")
    corr = result["corr_matrix"]
    fig_corr = go.Figure(
        data=go.Heatmap(
            z=corr,
            x=tickers_used,
            y=tickers_used,
            colorscale="RdBu",
            zmid=0,
            zmin=-1,
            zmax=1,
            text=np.round(corr, 2),
            texttemplate="%{text}",
            textfont={"size": 10},
            hovertemplate="<b>Correlation</b><br>%{y} vs %{x}<br>Correlation: %{z:.2f}<extra></extra>",
        )
    )
    fig_corr.update_layout(
        title="",
        xaxis_title="",
        yaxis_title="",
        height=300,
        margin=dict(t=20, b=40),
        hoverlabel=HOVERLABEL,
    )
    st.plotly_chart(fig_corr, use_container_width=True)
    st.markdown("")

    with st.expander("Covariance & numerical quality", expanded=False):
        st.caption("Condition number of the covariance matrix. High values may indicate numerical instability.")
        cond = result.get("condition_number")
        if cond is not None:
            st.write(f"**Condition number:** {cond:.2e}")
        else:
            st.write("Condition number not available.")

    with st.expander("Risk contribution by asset", expanded=False):
        st.caption("Contribution of each asset to portfolio risk (weight × marginal contribution to volatility).")
        contrib = result.get("risk_contributions")
        if contrib is not None and len(contrib) > 0:
            total = contrib.sum()
            contrib_pct = (contrib / total * 100) if total > 0 else contrib * 0
            df_risk = pd.DataFrame(
                {"Ticker": tickers_used, "Contribution (%)": np.round(contrib_pct, 2)}
            )
            st.dataframe(df_risk, use_container_width=True, hide_index=True)
        else:
            st.write("Not available.")

    st.markdown("")
    st.divider()

    # ----- 5. Charts -----
    st.subheader("Efficient frontier")
    with st.expander("? **What is this chart?**", expanded=False):
        st.markdown("""
        **Purpose:** Show the best return you can achieve for each level of risk (volatility).

        - **Curve (efficient frontier):** Portfolios that maximize return for a given risk; you want to be on this curve.  
        - **Optimal point (star):** The portfolio chosen by your objective (e.g. max Sharpe).  
        - **Benchmark (triangle):** Position of the selected index if a benchmark is chosen. Compare to the optimal point.  
        - **Gray cloud:** Random portfolios; the efficient frontier dominates them.  
        - **How to read:** Move right = more risk; move up = more return. The optimal point is where the Sharpe Ratio is highest (steepest line from risk-free rate).
        """)
    frontier = result["frontier_curve"]
    frontier_weights = result.get("frontier_weights")
    random_cloud = result["random_cloud"]
    random_weights = result.get("random_weights")
    fig = go.Figure()
    if random_cloud is not None and len(random_cloud) > 0:
        vol_r = random_cloud[:, 0] * 100
        ret_r = random_cloud[:, 1] * 100
        # Hover for each gray point: Vol, Ret + allocation (e.g. X% Microsoft, X% Nvidia)
        hover_random = []
        if random_weights is not None and len(random_weights) == len(vol_r):
            for i in range(len(vol_r)):
                parts = [
                    f"<b>Random portfolio</b>",
                    f"<b>Volatility:</b> {vol_r[i]:.2f}%",
                    f"<b>Return:</b> {ret_r[i]:.2f}%",
                    "<b>Allocation:</b>",
                ]
                for j, ticker in enumerate(tickers_used):
                    if j < random_weights.shape[1]:
                        parts.append(f"  {ticker}: {random_weights[i, j] * 100:.1f}%")
                hover_random.append("<br>".join(parts))
        else:
            hover_random = [f"Volatility: {vol_r[i]:.2f}%<br>Return: {ret_r[i]:.2f}%" for i in range(len(vol_r))]
        fig.add_trace(
            go.Scatter(
                x=vol_r,
                y=ret_r,
                mode="markers",
                name="Random portfolios",
                marker=dict(size=5, color="rgba(128, 128, 128, 0.5)", line=dict(width=0)),
                text=hover_random,
                hoverinfo="text",
            )
        )
    if frontier is not None and len(frontier) > 0:
        vol_f = frontier[:, 0] * 100
        ret_f = frontier[:, 1] * 100
        # Build hover text for each point: Vol, Ret + allocation (weight per ticker)
        hover_parts = []
        if frontier_weights is not None and len(frontier_weights) == len(vol_f):
            for i in range(len(vol_f)):
                parts = [
                    f"<b>Volatility:</b> {vol_f[i]:.2f}%",
                    f"<b>Return:</b> {ret_f[i]:.2f}%",
                    "<b>Allocation:</b>",
                ]
                for j, ticker in enumerate(tickers_used):
                    if j < frontier_weights.shape[1]:
                        parts.append(f"  {ticker}: {frontier_weights[i, j] * 100:.1f}%")
                hover_parts.append("<br>".join(parts))
        else:
            hover_parts = [f"Volatility: {vol_f[i]:.2f}%<br>Return: {ret_f[i]:.2f}%" for i in range(len(vol_f))]
        fig.add_trace(
            go.Scatter(
                x=vol_f,
                y=ret_f,
                mode="lines+markers",
                name="Efficient frontier",
                line=dict(color="rgb(31, 119, 180)", width=2),
                marker=dict(size=7, color="rgb(31, 119, 180)", line=dict(width=1, color="white"), symbol="circle"),
                text=hover_parts,
                hoverinfo="text",
            )
        )
    if result.get("benchmark_vol_annual") is not None and result.get("benchmark_return_annual") is not None:
        fig.add_trace(
            go.Scatter(
                x=[result["benchmark_vol_annual"] * 100],
                y=[result["benchmark_return_annual"] * 100],
                mode="markers",
                name="Benchmark",
                marker=dict(size=12, color="rgb(255, 127, 14)", symbol="triangle-up", line=dict(width=2, color="white")),
                text=[f"<b>Benchmark</b><br>Volatility: {result['benchmark_vol_annual']*100:.2f}%<br>Return: {result['benchmark_return_annual']*100:.2f}%"],
                hoverinfo="text",
            )
        )
    fig.add_trace(
        go.Scatter(
            x=[result["volatility"] * 100],
            y=[result["expected_return"] * 100],
            mode="markers",
            name="Optimal portfolio",
            marker=dict(size=16, color="rgb(214, 39, 40)", symbol="star", line=dict(width=2, color="white")),
            text=[f"<b>Optimal</b><br>Volatility: {result['volatility']*100:.2f}%<br>Return: {result['expected_return']*100:.2f}%<br>" + "<br>".join(f"{t}: {w*100:.1f}%" for t, w in zip(tickers_used, result["weights"]))],
            hoverinfo="text",
        )
    )
    # Axis range: focus on frontier + optimal + benchmark so outliers (e.g. random portfolios with shorts) don't stretch the scale
    x_vals = [result["volatility"] * 100]
    y_vals = [result["expected_return"] * 100]
    if frontier is not None and len(frontier) > 0:
        x_vals.extend(frontier[:, 0] * 100)
        y_vals.extend(frontier[:, 1] * 100)
    if result.get("benchmark_vol_annual") is not None:
        x_vals.append(result["benchmark_vol_annual"] * 100)
        y_vals.append(result["benchmark_return_annual"] * 100)
    x_min, x_max = 0, max(x_vals) * 1.15 if x_vals else 50
    y_min = min(y_vals) * 0.95 if y_vals else 0
    y_max = max(y_vals) * 1.15 if y_vals else 30
    fig.update_layout(
        title="Efficient frontier (annualized)",
        xaxis_title="Volatility (%)",
        yaxis_title="Expected return (%)",
        showlegend=True,
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        margin=dict(t=50),
        hovermode="closest",
        hoverlabel=HOVERLABEL,
        xaxis=dict(range=[x_min, x_max]),
        yaxis=dict(range=[y_min, y_max]),
    )
    fig.update_xaxes(showgrid=True, gridwidth=1, gridcolor="rgba(128,128,128,0.2)")
    fig.update_yaxes(showgrid=True, gridwidth=1, gridcolor="rgba(128,128,128,0.2)")
    st.plotly_chart(fig, use_container_width=True)

    st.markdown("")

    st.subheader("Optimal weights")
    with st.expander("❓ How to read these weights? (e.g. with 100k capital)", expanded=False):
        st.markdown("""
        Weights are in **% of your capital**. The sum of all weights = **100%** (net exposure).

        **Example with 100,000 €:**
        - **Positive weight** (e.g. +45%) = **long**: you hold 45,000 € of that asset.
        - **Negative weight** (e.g. −20%) = **short**: you have sold short 20,000 € (borrowed shares sold for 20,000 €).

        **Totals:** Total long + total short can exceed 100% (leverage). Your **capital** = 100% = longs − shorts (net). In practice, margin rules limit how much you can reuse from short proceeds.
        """)
    with st.expander("? **What is this chart?**", expanded=False):
        st.markdown("""
        **Purpose:** Visualize the recommended allocation across assets.

        - **Bars:** Weight (in %) of each asset in the optimal portfolio. Green = long, red = short (when short selling is allowed).  
        - **How to read:** Sum of weights = 100%. Use it to rebalance your portfolio.
        """)
    bar_colors = ["rgb(44, 160, 44)" if w >= 0 else "rgb(214, 39, 40)" for w in weights_pct]
    fig_w = go.Figure(
        data=[
            go.Bar(
                x=weights_pct,
                y=tickers_used,
                orientation="h",
                marker=dict(color=bar_colors),
                text=[f"{w:.1f}%" for w in weights_pct],
                textposition="outside",
                hovertemplate="<b>%{y}</b><br>Weight: %{x:.1f}%<extra></extra>",
            )
        ]
    )
    x_min_w = min(weights_pct) * 1.2 if len(weights_pct) else -20
    x_max_w = max(weights_pct) * 1.2 if len(weights_pct) else 100
    if x_min_w > 0:
        x_min_w = 0
    fig_w.update_layout(
        title="Optimal portfolio weights",
        xaxis_title="Weight (%)",
        yaxis_title="",
        margin=dict(t=40, b=40),
        height=max(200, 40 * len(tickers_used)),
        showlegend=False,
        hoverlabel=HOVERLABEL,
    )
    fig_w.update_xaxes(range=[x_min_w, x_max_w], showgrid=True)
    st.plotly_chart(fig_w, use_container_width=True)

    st.markdown("")
    st.divider()

    # ----- 6. Export -----
    with st.expander("Export results (CSV)", expanded=False):
        st.caption("Download optimal weights and key metrics (and optionally frontier points) as CSV.")
        rows = [
            ["ticker", "weight_pct"],
        ]
        for i, t in enumerate(tickers_used):
            rows.append([t, f"{weights_pct[i]:.6f}"])
        rows.append([])
        rows.append(["metric", "value"])
        rows.append(["expected_return_annual", f"{result['expected_return']:.6f}"])
        rows.append(["volatility_annual", f"{result['volatility']:.6f}"])
        rows.append(["sharpe_ratio", f"{result['sharpe_ratio']:.6f}"])
        v95 = result.get("var_95_daily")
        c95 = result.get("cvar_95_daily")
        rows.append(["var_95_daily", f"{v95:.6f}" if v95 is not None else ""])
        rows.append(["cvar_95_daily", f"{c95:.6f}" if c95 is not None else ""])
        rows.append(["risk_free_rate", f"{risk_free_rate:.6f}"])
        rows.append(["objective", objective])
        if result.get("benchmark_return_annual") is not None:
            rows.append(["benchmark_return_annual", f"{result['benchmark_return_annual']:.6f}"])
            rows.append(["benchmark_vol_annual", f"{result['benchmark_vol_annual']:.6f}"])
            rows.append(["alpha_vs_benchmark", f"{result['alpha_vs_benchmark']:.6f}"])
            rows.append(["tracking_error_annual", f"{result['tracking_error_annual']:.6f}"])
            rows.append(["information_ratio", f"{result['information_ratio']:.6f}"])
        rows.append(["start_date", start_str])
        rows.append(["end_date", end_str])
        rows.append(["n_observations", str(n_obs)])
        buf = io.StringIO()
        for row in rows:
            if len(row) == 2:
                buf.write(f"{row[0]},{row[1]}\n")
            else:
                buf.write("\n")
        csv_bytes = buf.getvalue().encode("utf-8")
        st.download_button(
            label="Download results (CSV)",
            data=csv_bytes,
            file_name="markowitz_results.csv",
            mime="text/csv",
            key="markowitz_export_csv",
        )
