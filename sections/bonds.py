"""Bonds tab."""
import io
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from config import (
    DEFAULT_BOND_FACE_VALUE,
    DEFAULT_BOND_COUPON_RATE,
    DEFAULT_BOND_MATURITY_YEARS,
    DEFAULT_BOND_FREQUENCY,
    DEFAULT_BOND_YTM,
    DEFAULT_BOND_YIELD_CURVE_POINTS,
    DEFAULT_RISK_FREE_RATE,
)
from services import fixed_income_service
from utils.plot_config import HOVERLABEL


def _format_pct(value: float, decimals: int = 2) -> str:
    """Format as percentage (e.g. 0.0523 -> 5.23%)."""
    return f"{value * 100:.{decimals}f}%"


@st.cache_data(ttl=3600)
def _load_bond_results(
    face_value: float,
    coupon_rate: float,
    maturity_years: float,
    frequency: int,
    ytm: float | None,
    price: float | None,
    ytm_min: float,
    ytm_max: float,
    n_curve_points: int,
):
    """Cached wrapper: run full bond analysis (price, YTM, duration, convexity, curve)."""
    return fixed_income_service.run_bond_analysis(
        face_value=face_value,
        coupon_rate=coupon_rate,
        maturity_years=maturity_years,
        frequency=frequency,
        ytm=ytm,
        price=price,
        ytm_min=ytm_min,
        ytm_max=ytm_max,
        n_curve_points=n_curve_points,
    )


def render():
    """Render Bonds tab."""
    # ----- 1. Header -----
    st.header("Fixed Income & Bond Pricing")
    st.markdown("Value bonds and measure rate risk (Duration, Convexity).")

    with st.expander("? **What is this tab for?**", expanded=False):
        st.markdown("""
        **Purpose:** This tab is for **bonds** (fixed-income securities): you lend money and receive fixed coupons and principal back.  
        It answers: (1) *What is a bond worth today?* (price / yield to maturity), and (2) *How sensitive is the price to interest-rate changes?* (Duration, Convexity).

        - **Yield to maturity (YTM)** is the annual return you get if you hold the bond until maturity; it moves inversely with price.  
        - **Duration** measures how much the bond price changes when rates move by 1%; it is the main indicator of interest-rate risk.  
        - **Convexity** refines that: it captures the non-linear effect when rates move a lot.

        **Use it when:** you invest in bonds or bond funds, need to compare yields, or want to measure and hedge interest-rate risk.
        """)

    st.markdown("")

    # ----- Parameters (in-tab: only Fixed Income inputs visible here) -----
    with st.expander("Parameters", expanded=True):
        st.caption("Set bond terms, YTM or price, and options. These inputs apply only to this tab.")
        st.markdown("")
        freq_options = [1, 2, 4]
        freq_labels = ["Annual (1)", "Semi-annual (2)", "Quarterly (4)"]
        freq_idx = freq_options.index(DEFAULT_BOND_FREQUENCY) if DEFAULT_BOND_FREQUENCY in freq_options else 1

        c1, c2, c3 = st.columns(3)
        with c1:
            face_value = st.number_input(
                "Face value (par)",
                min_value=1,
                max_value=1_000_000,
                value=int(DEFAULT_BOND_FACE_VALUE),
                step=10,
                help="Principal repaid at maturity.",
                key="fixed_income_face_value",
            )
            face_value = float(face_value)

            coupon_rate_pct = st.number_input(
                "Coupon rate (%)",
                min_value=0.0,
                max_value=20.0,
                value=float(DEFAULT_BOND_COUPON_RATE * 100),
                step=0.25,
                format="%.2f",
                help="Annual coupon rate, e.g. 5 for 5%.",
                key="fixed_income_coupon_rate",
            )
            coupon_rate = coupon_rate_pct / 100.0

            maturity_years = st.number_input(
                "Maturity (years)",
                min_value=0.5,
                max_value=50.0,
                value=float(DEFAULT_BOND_MATURITY_YEARS),
                step=0.5,
                format="%.1f",
                help="Time to maturity in years.",
                key="fixed_income_maturity",
            )

            frequency = st.selectbox(
                "Coupon frequency",
                options=range(len(freq_options)),
                index=freq_idx,
                format_func=lambda i: freq_labels[i],
                help="Number of coupon payments per year.",
                key="fixed_income_frequency",
            )
            frequency = freq_options[frequency]
        with c2:
            input_mode = st.radio(
                "Input",
                options=["Enter YTM (%)", "Enter market price"],
                index=0,
                help="Provide either yield to maturity or current market price; the other is computed.",
                key="fixed_income_input_mode",
            )

            ytm_input = None
            price_input = None
            if input_mode == "Enter YTM (%)":
                ytm_pct = st.number_input(
                    "YTM (%)",
                    min_value=-5.0,
                    max_value=30.0,
                    value=float(DEFAULT_BOND_YTM * 100),
                    step=0.25,
                    format="%.2f",
                    help="Yield to maturity (annual). Price is computed at this yield.",
                    key="fixed_income_ytm",
                )
                ytm_input = ytm_pct / 100.0
            else:
                price_input = st.number_input(
                    "Price",
                    min_value=1.0,
                    max_value=1_000_000.0,
                    value=100.0,
                    step=1.0,
                    format="%.2f",
                    help="Current market price. YTM is solved from this price.",
                    key="fixed_income_price",
                )

            risk_free_rate = st.number_input(
                "Risk-free rate (for comparison)",
                min_value=0.0,
                max_value=0.30,
                value=float(DEFAULT_RISK_FREE_RATE),
                step=0.005,
                format="%.3f",
                help="Annual rate for comparison; used for spread vs risk-free.",
                key="fixed_income_rf",
            )

            with st.expander("Clean / Dirty price (30/360)", expanded=False):
                settlement_date = st.date_input(
                    "Settlement date",
                    value=pd.Timestamp.now().date(),
                    help="Date when the trade settles (you pay dirty price). Bond maturity is fixed from today; changing this date changes accrued interest: further from last coupon = higher accrued.",
                    key="fixed_income_settlement",
                )
                settlement_ts = pd.Timestamp(settlement_date)
        with c3:
            with st.expander("Price–Yield curve range", expanded=False):
                ytm_min_pct = st.number_input(
                    "YTM min (%)",
                    value=0.5,
                    step=0.5,
                    format="%.1f",
                    key="fixed_income_ytm_min",
                )
                ytm_max_pct = st.number_input(
                    "YTM max (%)",
                    value=15.0,
                    step=0.5,
                    format="%.1f",
                    key="fixed_income_ytm_max",
                )
                n_curve_points = st.number_input(
                    "Curve points",
                    min_value=20,
                    max_value=200,
                    value=DEFAULT_BOND_YIELD_CURVE_POINTS,
                    step=10,
                    key="fixed_income_curve_points",
                )
            ytm_min = ytm_min_pct / 100.0
            ytm_max = ytm_max_pct / 100.0

            compare_bonds = st.checkbox(
                "Compare with another bond",
                value=False,
                help="Show second bond inputs and comparison table.",
                key="fixed_income_compare",
            )
            bond2_ytm_input = None
            bond2_price_input = None
            if compare_bonds:
                st.caption("Bond 2 parameters")
                face_value_2 = float(st.number_input("Face value (par)", min_value=1, max_value=1_000_000, value=int(face_value), step=10, key="fi_face_2"))
                coupon_rate_2 = st.number_input("Coupon rate (%)", min_value=0.0, max_value=20.0, value=float(coupon_rate * 100), step=0.25, format="%.2f", key="fi_coupon_2") / 100.0
                maturity_years_2 = st.number_input("Maturity (years)", min_value=0.5, max_value=50.0, value=float(maturity_years), step=0.5, format="%.1f", key="fi_mat_2")
                freq_idx_2 = st.selectbox("Coupon frequency", range(len(freq_options)), index=freq_options.index(frequency) if frequency in freq_options else 1, format_func=lambda i: freq_labels[i], key="fi_freq_2")
                frequency_2 = freq_options[freq_idx_2]
                input_mode_2 = st.radio("Input", ["Enter YTM (%)", "Enter market price"], index=0, key="fi_input_2")
                if input_mode_2 == "Enter YTM (%)":
                    bond2_ytm_pct_default = (ytm_input * 100.0) if ytm_input is not None else (DEFAULT_BOND_YTM * 100.0)
                    bond2_ytm_input = st.number_input("YTM (%)", min_value=-5.0, max_value=30.0, value=float(bond2_ytm_pct_default), step=0.25, format="%.2f", help="Yield in percent, e.g. 5 for 5%.", key="fi_ytm_2") / 100.0
                    bond2_price_input = None
                else:
                    bond2_ytm_input = None
                    bond2_price_input = st.number_input(
                        "Price",
                        min_value=1.0,
                        max_value=1_000_000.0,
                        value=100.0,
                        step=1.0,
                        format="%.2f",
                        help="Market price; YTM is solved from this price.",
                        key="fi_price_2",
                    )
        st.markdown("")
    st.divider()

    # ----- 2. Context: run analysis -----
    with st.spinner("Computing bond metrics…"):
        result = _load_bond_results(
            face_value=face_value,
            coupon_rate=coupon_rate,
            maturity_years=maturity_years,
            frequency=frequency,
            ytm=ytm_input,
            price=price_input,
            ytm_min=ytm_min,
            ytm_max=ytm_max,
            n_curve_points=n_curve_points,
        )

    if not result["success"]:
        st.error(result.get("error", "An error occurred."))
        return

    price = round(result["price"], 2)
    ytm = result["ytm"]
    macaulay_duration = round(result["macaulay_duration"], 2)
    modified_duration = round(result["modified_duration"], 2)
    convexity_val = round(result["convexity"], 2)
    cash_flows = result["cash_flows"]
    yield_grid = result["yield_grid"]
    price_grid = result["price_grid"]

    # Context block
    freq_label = {1: "Annual", 2: "Semi-annual", 4: "Quarterly"}.get(frequency, str(frequency))
    st.info(f"**Bond:** Face {face_value:.0f}, Coupon {coupon_rate*100:.2f}%, {freq_label}, Maturity {maturity_years:.1f}y")
    st.caption(f"**YTM:** {ytm*100:.2f}% · **Price:** {price:.2f} (per 100 par) · **Frequency:** {frequency} payments/year")
    st.markdown("")

    # ----- 3. Key results -----
    st.subheader("Key results")
    with st.expander("? **What do these numbers mean?**", expanded=False):
        st.markdown("""
        This block summarizes the **bond valuation** and **interest-rate sensitivity**. Hover the **?** next to each metric for a detailed explanation.

        - **Price (per 100 par):** Present value of all future cash flows (coupons + principal) discounted at the YTM, per 100 face value. 100 = 100% of par.  
        - **Yield to maturity (YTM):** The annual return you get if you hold the bond until maturity; it moves inversely with price.  
        - **Macaulay Duration:** Weighted average time (in years) to receive cash flows; higher = more rate sensitivity.  
        - **Modified Duration:** Approximate % price change for a 1% rise in yield; main measure of rate risk.  
        - **Convexity:** Refines the price change when yields move a lot; positive convexity means prices rise more when yields fall than they fall when yields rise.
        """)

    c1, c2, c3, c4, c5, c6 = st.columns(6)
    with c1:
        st.metric(
            "Price (per 100 par)",
            f"{price:.2f}",
            help="Purpose: present value of all future cash flows (coupons + principal) discounted at the YTM, per 100 face value. 100 = 100% of par. Use it to see what the bond is worth today; compare to market price to spot mispricing.",
        )
    with c2:
        st.metric(
            "Yield to maturity (YTM)",
            _format_pct(ytm),
            help="Purpose: the annual return you get if you hold the bond until maturity. It moves inversely with price. Use it to compare bonds or to discount cash flows.",
        )
    with c3:
        spread_vs_rf = round(ytm - risk_free_rate, 4)
        st.metric(
            "Spread vs risk-free",
            f"{spread_vs_rf*100:+.2f}%",
            help="Purpose: YTM minus risk-free rate (in %). Positive = bond yields more than the risk-free rate (compensation for risk). Use it to compare bonds or to assess relative value.",
        )
    with c4:
        st.metric(
            "Macaulay Duration",
            f"{macaulay_duration:.2f}",
            help="Purpose: weighted average time (in years) to receive cash flows. Higher duration = more sensitivity to rate changes. Use it to match liabilities or to compare bonds.",
        )
    with c5:
        st.metric(
            "Modified Duration",
            f"{modified_duration:.2f}",
            help="Purpose: approximate % price change for a 1% (100 bp) rise in yield. E.g. Modified Duration 7 → price falls ~7% if yield rises 1%. Use it to hedge or size rate risk.",
        )
    with c6:
        st.metric(
            "Convexity",
            f"{convexity_val:.2f}",
            help="Purpose: captures non-linear price change when yields move a lot. Positive convexity means price rises more when yields fall than it falls when yields rise. Use it for larger rate moves.",
        )

    st.caption(
        "For a 1% rise in yield: price change ≈ -Modified Duration × 1% + 0.5 × Convexity × (1%)²"
    )

    if maturity_years < 1:
        st.warning("Maturity under 1 year: duration and convexity may be less meaningful.")
    if ytm < 0:
        st.warning("Negative YTM: unusual; check inputs or market context.")

    st.markdown("")
    st.divider()

    # ----- 3b. Clean / Dirty price -----
    st.subheader("Clean / Dirty price (30/360)")
    with st.expander("? **What are Clean and Dirty price?**", expanded=False):
        st.markdown("""
        **Clean price** is the quoted market price (no accrued interest). **Dirty price** is what you actually pay: Clean + Accrued interest.  
        **Accrued interest** is the coupon earned since the last payment date (30/360 day count). Use settlement date to compute it.
        """)
    accrued = fixed_income_service.accrued_interest(
        face_value, coupon_rate, maturity_years, frequency, settlement_ts
    )
    accrued = round(accrued, 2)
    clean_price = price
    dirty_price = round(clean_price + accrued, 2)  # exact sum for display consistency
    d1, d2, d3 = st.columns(3)
    with d1:
        st.metric("Clean price", f"{clean_price:.2f}", help="Quoted price per 100 par (DCF price at coupon dates).")
    with d2:
        st.metric("Accrued interest", f"{accrued:.2f}", help="Coupon accrued since last payment (30/360).")
    with d3:
        st.metric("Dirty price", f"{dirty_price:.2f}", help="Clean + Accrued; amount to pay per 100 par at settlement.")
    st.caption(f"Settlement: {settlement_ts.strftime('%Y-%m-%d')} · Day count: 30/360")
    st.markdown("")
    st.divider()

    # ----- 3c. Rate scenarios -----
    st.subheader("Rate scenarios")
    with st.expander("? **What is this section?**", expanded=False):
        st.markdown("""
        **Purpose:** Show how the bond price changes for given yield shocks (+50 bp, +100 bp, -50 bp, -100 bp).  
        **Delta P/P:** Percentage change in price. Use it to stress-test or size hedges.
        """)
    scenarios_bp = [50, 100, -50, -100]
    scenario_rows = fixed_income_service.rate_scenarios(
        face_value, coupon_rate, maturity_years, frequency, ytm, price, scenarios_bp
    )
    df_scenarios = pd.DataFrame([
        {
            "Yield shift": f"{r['shift_bp']:+.0f} bp",
            "New YTM (%)": f"{r['new_ytm']*100:.2f}",
            "New price": f"{round(r['new_price'], 2):.2f}",
            "ΔP/P (%)": f"{round(r['delta_pct'], 2):+.2f}",
        }
        for r in scenario_rows
    ])
    st.dataframe(df_scenarios, use_container_width=True, hide_index=True)
    st.markdown("")
    st.divider()

    # ----- 3d. Compare two bonds -----
    if compare_bonds:
        st.subheader("Compare two bonds")
        with st.expander("? **What is this section?**", expanded=False):
            st.markdown("""
            **Purpose:** Side-by-side comparison of Bond 1 (main inputs) and Bond 2 (comparison inputs). Use it to compare YTM, duration, spread, etc.
            """)
        with st.spinner("Computing Bond 2…"):
            result2 = _load_bond_results(
                face_value_2, coupon_rate_2, maturity_years_2, frequency_2,
                bond2_ytm_input, bond2_price_input, ytm_min, ytm_max, n_curve_points,
            )
        if result2["success"]:
            p2 = round(result2["price"], 2)
            ytm2 = result2["ytm"]
            mac2 = round(result2["macaulay_duration"], 2)
            mod2 = round(result2["modified_duration"], 2)
            conv2 = round(result2["convexity"], 2)
            spread1 = round((ytm - risk_free_rate) * 100, 2)
            spread2 = round((ytm2 - risk_free_rate) * 100, 2)
            comp_metrics = [
                ("Price (per 100 par)", f"{price:.2f}", f"{p2:.2f}"),
                ("YTM (%)", f"{ytm*100:.2f}", f"{ytm2*100:.2f}"),
                ("Spread vs risk-free (%)", f"{spread1:+.2f}", f"{spread2:+.2f}"),
                ("Macaulay Duration", f"{macaulay_duration:.2f}", f"{mac2:.2f}"),
                ("Modified Duration", f"{modified_duration:.2f}", f"{mod2:.2f}"),
                ("Convexity", f"{convexity_val:.2f}", f"{conv2:.2f}"),
            ]
            df_comp = pd.DataFrame(comp_metrics, columns=["Metric", "Bond 1", "Bond 2"])
            st.dataframe(df_comp, use_container_width=True, hide_index=True)
            st.caption("All prices are per 100 par (100 = 100%). YTM and Spread are in %. Same Bond 1 / Bond 2 inputs → similar metrics; different YTM or maturity → different price and duration.")
        else:
            st.warning(f"Bond 2: {result2.get('error', 'Error')}")
        st.markdown("")
        st.divider()

    # ----- 4. Statistics & quality -----
    st.subheader("Statistics & quality")
    st.caption("Expand for cash flow table and duration/convexity formula reference.")
    st.markdown("")

    with st.expander("Cash flows", expanded=False):
        st.caption("Each period: coupon (and principal at maturity), present value at current YTM.")
        if cash_flows:
            df_cf = pd.DataFrame(cash_flows)
            df_cf = df_cf.rename(columns={"period": "Period", "time_years": "Time (y)", "cash_flow": "Cash flow", "pv": "PV"})
            df_cf["Cash flow"] = df_cf["Cash flow"].round(2)
            df_cf["PV"] = df_cf["PV"].round(2)
            df_cf["Time (y)"] = df_cf["Time (y)"].round(2)
            st.dataframe(df_cf, use_container_width=True, hide_index=True)
        else:
            st.write("No cash flows.")

    with st.expander("Duration & Convexity formulas", expanded=False):
        st.caption("Reference: Macaulay D = (1/P) Σ t·PV(CF_t) in years; Modified D = Macaulay D / (1+y/f); Convexity refines ΔP/P for large Δy.")
        st.markdown("""
        - **Macaulay Duration:** \\( D_{\\text{Mac}} = \\frac{1}{P} \\sum_t \\frac{t}{f} \\cdot \\text{PV}(\\text{CF}_t) \\) (years).  
        - **Modified Duration:** \\( D_{\\text{Mod}} = D_{\\text{Mac}} / (1 + y/f) \\).  
        - **Convexity:** \\( \\frac{1}{P(1+y/f)^2 f^2} \\sum_t \\text{CF}_t \\cdot t(t+1) / (1+y/f)^t \\).  
        - **Approx. price change:** \\( \\Delta P/P \\approx -D_{\\text{Mod}} \\cdot \\Delta y + \\frac{1}{2} \\text{Convexity} \\cdot (\\Delta y)^2 \\).
        """)

    st.markdown("")
    st.divider()

    # ----- 5. Charts -----
    st.subheader("Price–Yield curve")
    with st.expander("? **What is this chart?**", expanded=False):
        st.markdown("""
        **Purpose:** Show how **bond price** changes as **yield to maturity** changes.

        - **X-axis:** YTM (%). Higher yield → lower price (inverse relationship).  
        - **Y-axis:** Bond price (same scale as par).  
        - **Curve:** Each point is the price at that yield; the curve is convex (bowed).  
        - **Highlighted point:** Your current YTM and computed price. Use it to see where you sit on the curve.
        """)

    # Use rounded values so chart point matches Key results exactly
    ytm_pct_display = round(ytm * 100, 2)
    price_display = round(price, 2)
    hover_template = (
        "<b>Price–Yield curve</b><br>"
        "YTM: %{x:.2f}%<br>"
        "Price: %{y:.2f} <i>(per 100 par)</i>"
        "<extra></extra>"
    )
    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=yield_grid * 100,
            y=price_grid,
            mode="lines",
            name="Price–Yield curve",
            line=dict(color="rgb(31, 119, 180)", width=2),
            hovertemplate=hover_template,
        )
    )
    fig.add_trace(
        go.Scatter(
            x=[ytm_pct_display],
            y=[price_display],
            mode="markers",
            name="Current (YTM, Price)",
            marker=dict(size=12, color="rgb(214, 39, 40)", symbol="circle", line=dict(width=2, color="white")),
            hovertemplate=(
                "<b>Current point</b><br>"
                "YTM: %{x:.2f}%<br>"
                "Price: %{y:.2f} <i>(per 100 par)</i>"
                "<extra></extra>"
            ),
        )
    )
    fig.update_layout(
        title="Price–Yield curve",
        xaxis_title="Yield to maturity (%)",
        yaxis_title="Price (per 100 par)",
        showlegend=True,
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        margin=dict(t=50),
        hovermode="x unified",
        hoverlabel=HOVERLABEL,
    )
    fig.update_xaxes(showgrid=True, gridwidth=1, gridcolor="rgba(128,128,128,0.2)")
    fig.update_yaxes(showgrid=True, gridwidth=1, gridcolor="rgba(128,128,128,0.2)")
    st.plotly_chart(fig, use_container_width=True)

    st.markdown("")
    st.divider()

    # ----- 6. Export -----
    with st.expander("Export results (CSV)", expanded=False):
        st.caption("Download key bond figures and parameters as CSV.")
        rows = [
            ["face_value", f"{face_value:.6f}"],
            ["coupon_rate", f"{coupon_rate:.6f}"],
            ["maturity_years", f"{maturity_years:.6f}"],
            ["frequency", str(frequency)],
            ["ytm", f"{ytm:.6f}"],
            ["price", f"{price:.6f}"],
            ["spread_vs_risk_free", f"{spread_vs_rf:.6f}"],
            ["clean_price", f"{clean_price:.6f}"],
            ["accrued_interest", f"{accrued:.6f}"],
            ["dirty_price", f"{dirty_price:.6f}"],
            ["macaulay_duration", f"{macaulay_duration:.6f}"],
            ["modified_duration", f"{modified_duration:.6f}"],
            ["convexity", f"{convexity_val:.6f}"],
        ]
        buf = io.StringIO()
        buf.write("metric,value\n")
        for row in rows:
            buf.write(f"{row[0]},{row[1]}\n")
        csv_bytes = buf.getvalue().encode("utf-8")
        st.download_button(
            label="Download results (CSV)",
            data=csv_bytes,
            file_name="fixed_income_results.csv",
            mime="text/csv",
            key="fixed_income_export_csv",
        )
