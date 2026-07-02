"""Onglet Bonds (obligations) — refonte claire & pro."""
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from config import (
    DEFAULT_BOND_FACE_VALUE, DEFAULT_BOND_COUPON_RATE, DEFAULT_BOND_MATURITY_YEARS,
    DEFAULT_BOND_FREQUENCY, DEFAULT_BOND_YTM, DEFAULT_BOND_YIELD_CURVE_POINTS, DEFAULT_RISK_FREE_RATE,
)
from services import fixed_income_service
from utils.formatters import format_pct
from utils.plot_config import apply_theme, COLOR_PRIMARY, COLOR_NEGATIVE
from utils.ui import kpi_row, section_header, advanced_expander, show_data_error, info_inline, metric_with_info


@st.cache_data(ttl=3600)
def _load_bond_results(face_value, coupon_rate, maturity_years, frequency, ytm, price, ytm_min, ytm_max, n_curve_points):
    """Wrapper caché : analyse obligataire complète."""
    return fixed_income_service.run_bond_analysis(
        face_value=face_value, coupon_rate=coupon_rate, maturity_years=maturity_years, frequency=frequency,
        ytm=ytm, price=price, ytm_min=ytm_min, ytm_max=ytm_max, n_curve_points=n_curve_points,
    )


def render():
    """Affiche l'onglet Bonds."""
    section_header("Obligations & risque de taux", "bond_price", level="header")
    st.caption("Valorise une obligation et mesure sa sensibilité aux taux (duration, convexité).")

    # ----- Paramètres -----
    freq_opts = [1, 2, 4]
    freq_labels = {1: "Annuel", 2: "Semestriel", 4: "Trimestriel"}
    with st.expander("Paramètres", expanded=True):
        c1, c2, c3 = st.columns(3)
        with c1:
            face_value = float(st.number_input("Nominal (pair)", 1, 1_000_000, int(DEFAULT_BOND_FACE_VALUE), 10, key="bond_face"))
            coupon_rate = st.number_input("Taux de coupon (%)", 0.0, 20.0, float(DEFAULT_BOND_COUPON_RATE * 100), 0.25, format="%.2f", key="bond_coupon") / 100.0
            maturity_years = st.number_input("Maturité (années)", 0.5, 50.0, float(DEFAULT_BOND_MATURITY_YEARS), 0.5, format="%.1f", key="bond_maturity")
            frequency = st.selectbox("Fréquence des coupons", freq_opts, index=freq_opts.index(DEFAULT_BOND_FREQUENCY), format_func=lambda f: freq_labels[f], key="bond_freq")
        with c2:
            risk_free_rate = st.number_input("Taux sans risque", 0.0, 0.30, float(DEFAULT_RISK_FREE_RATE), 0.005, format="%.3f", key="bond_rf")
            input_mode = st.radio(
                "Saisie du rendement", ["Taux (YTM)", "Prix de marché", "Sans risque + spread crédit"],
                index=0, key="bond_input_mode",
                help="« Sans risque + spread » price l'obligation au taux sans risque augmenté d'un spread de crédit (risque émetteur).",
            )
            ytm_input = price_input = None
            if input_mode == "Taux (YTM)":
                ytm_input = st.number_input("YTM (%)", -5.0, 30.0, float(DEFAULT_BOND_YTM * 100), 0.25, format="%.2f", key="bond_ytm") / 100.0
            elif input_mode == "Prix de marché":
                price_input = st.number_input("Prix", 1.0, 1_000_000.0, 100.0, 1.0, format="%.2f", key="bond_price_in")
            else:  # Sans risque + spread de crédit
                credit_spread_bp = st.number_input("Spread de crédit (bp)", 0.0, 2000.0, 100.0, 5.0, format="%.0f", key="bond_spread")
                ytm_input = risk_free_rate + credit_spread_bp / 10_000.0
                st.caption(f"Rendement total = {risk_free_rate * 100:.2f} % (sans risque) + {credit_spread_bp:.0f} bp = {ytm_input * 100:.2f} %")
        with c3:
            with st.expander("Plage de la courbe prix-taux", expanded=False):
                ytm_min = st.number_input("YTM min (%)", value=0.5, step=0.5, format="%.1f", key="bond_ytm_min") / 100.0
                ytm_max = st.number_input("YTM max (%)", value=15.0, step=0.5, format="%.1f", key="bond_ytm_max") / 100.0
                n_curve_points = st.number_input("Nombre de points", 20, 200, DEFAULT_BOND_YIELD_CURVE_POINTS, 10, key="bond_curve_pts")
    st.divider()

    with st.spinner("Calcul des métriques obligataires…"):
        result = _load_bond_results(face_value, coupon_rate, maturity_years, frequency, ytm_input, price_input, ytm_min, ytm_max, n_curve_points)

    if not result["success"]:
        show_data_error(result.get("error"))
        return

    price = round(result["price"], 2)
    ytm = result["ytm"]
    mac = round(result["macaulay_duration"], 2)
    mod = round(result["modified_duration"], 2)
    conv = round(result["convexity"], 2)

    st.info(f"**Obligation :** nominal {face_value:.0f} · coupon {coupon_rate * 100:.2f}% · {freq_labels[frequency].lower()} · maturité {maturity_years:.1f} ans")

    # ----- KPI en tête -----
    kpi_row([
        ("Prix (pour 100 de pair)", f"{price:.2f}", "bond_price"),
        ("Rendement (YTM)", format_pct(ytm), "ytm"),
        ("Duration modifiée", f"{mod:.2f}", "modified_duration"),
        ("Convexité", f"{conv:.2f}", "convexity"),
    ])
    st.caption("Pour +1 % de taux : variation du prix ≈ − duration modifiée × 1 % + ½ × convexité × (1 %)².")

    # ----- Sensibilités DV01 / CS01 (risque de taux et de crédit) -----
    st.markdown("")
    s1, s2 = st.columns(2)
    with s1:
        dv01_val = result.get("dv01")
        metric_with_info("DV01 (par 1 bp de taux)", f"{dv01_val:.4f}" if dv01_val == dv01_val else "—", "dv01")
    with s2:
        cs01_val = result.get("cs01")
        metric_with_info("CS01 (par 1 bp de spread)", f"{cs01_val:.4f}" if cs01_val == cs01_val else "—", "cs01")
    st.caption(
        "DV01 et CS01 sont en unités de prix par 1 bp. Pour une obligation vanille à taux fixe, ils "
        "coïncident (même décalage du taux d'actualisation) et divergeraient au niveau d'un portefeuille."
    )

    # ----- Graphe héros : courbe prix-taux -----
    st.markdown("")
    hero_row = st.columns([0.9, 0.1])
    with hero_row[0]:
        st.subheader("Courbe prix-taux")
    with hero_row[1]:
        info_inline("price_yield_curve", "Courbe prix-taux")

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=result["yield_grid"] * 100, y=result["price_grid"], mode="lines", name="Prix",
        line=dict(color=COLOR_PRIMARY, width=2.5),
        hovertemplate="YTM : %{x:.2f}%<br>Prix : %{y:.2f}<extra></extra>",
    ))
    fig.add_trace(go.Scatter(
        x=[round(ytm * 100, 2)], y=[price], mode="markers", name="Point actuel",
        marker=dict(size=13, color=COLOR_NEGATIVE, symbol="circle", line=dict(width=1.5, color="white")),
        hovertemplate="YTM : %{x:.2f}%<br>Prix : %{y:.2f}<extra></extra>",
    ))
    fig.update_layout(title="Prix en fonction du rendement", xaxis_title="Rendement (YTM, %)", yaxis_title="Prix (pour 100 de pair)")
    apply_theme(fig)
    st.plotly_chart(fig, use_container_width=True)

    # ----- Avancé -----
    with advanced_expander():
        a1, a2 = st.columns(2)
        with a1:
            metric_with_info("Duration de Macaulay", f"{mac:.2f}", "macaulay_duration")
        with a2:
            spread = round((ytm - risk_free_rate) * 100, 2)
            metric_with_info("Écart vs sans-risque", f"{spread:+.2f} %", "spread_vs_rf")

        st.markdown("**Flux de trésorerie (coupons + remboursement)**")
        cash_flows = result["cash_flows"]
        if cash_flows:
            df_cf = pd.DataFrame(cash_flows).rename(
                columns={"period": "Période", "time_years": "Temps (a)", "cash_flow": "Flux", "pv": "Valeur actualisée"}
            )
            df_cf["Flux"] = df_cf["Flux"].round(2)
            df_cf["Valeur actualisée"] = df_cf["Valeur actualisée"].round(2)
            df_cf["Temps (a)"] = df_cf["Temps (a)"].round(2)
            st.dataframe(df_cf, use_container_width=True, hide_index=True)
        else:
            st.info("Aucun flux à afficher.")
