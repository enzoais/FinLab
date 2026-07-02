"""Onglet Risque (fonds / mandat) — VaR 3 méthodes, stress tests, concentration, backtesting VaR."""
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from config import DEFAULT_MARKOWITZ_TICKERS
from services import capm_service, risk_service
from utils.formatters import format_pct
from utils.plot_config import apply_theme, COLOR_NEGATIVE
from utils.ui import (
    kpi_row, section_header, advanced_expander, show_data_error, info_inline,
    metric_with_info, asset_multiselect,
)

_METHOD_LABELS = {
    "historique": "Historique",
    "parametrique": "Paramétrique",
    "monte_carlo": "Monte-Carlo",
    "cornish_fisher": "Cornish-Fisher (modifiée)",
    "ewma": "EWMA (RiskMetrics)",
}
_METHOD_INFO = {
    "historique": "var", "parametrique": "parametric_var", "monte_carlo": "var",
    "cornish_fisher": "modified_var", "ewma": "ewma_var",
}


@st.cache_data(ttl=3600)
def _load_risk(tickers, weights, start_str, end_str, aum, confidence, horizon, benchmark):
    """Wrapper caché : pipeline de risque complet."""
    w = np.array(weights) if weights is not None else None
    return risk_service.run_risk_analysis(
        tickers=list(tickers), weights=w, start_date=pd.Timestamp(start_str), end_date=pd.Timestamp(end_str),
        aum=aum, confidence=confidence, horizon_days=horizon, benchmark_ticker=benchmark or None,
    )


def _eur(x: float) -> str:
    return f"{x:,.0f} €".replace(",", " ")


def render():
    """Affiche l'onglet Risque."""
    section_header("Risque du portefeuille / fonds", "var", level="header")
    st.caption("Mesure et suivi des risques de marché d'un fonds ou mandat : VaR, stress tests, concentration, contrôle du modèle.")

    # ----- Paramètres -----
    with st.expander("Paramètres", expanded=True):
        c1, c2 = st.columns(2)
        with c1:
            tickers = asset_multiselect(
                "Actifs du fonds", list(DEFAULT_MARKOWITZ_TICKERS), key="risk_tickers",
                help="Cherchez par nom ou ticker. Au moins 2 actifs.",
            )
            weight_mode = st.radio("Poids", ["Équipondéré", "Personnalisés"], index=0, horizontal=True, key="risk_weight_mode")
            aum = st.number_input("Encours du fonds (€)", 1_000.0, 1e12, 1_000_000.0, 100_000.0, format="%.0f", key="risk_aum")
        with c2:
            period_preset = st.selectbox("Période", ["5 ans", "3 ans", "2 ans", "1 an"], index=0, key="risk_period")
            years = {"5 ans": 5, "3 ans": 3, "2 ans": 2, "1 an": 1}[period_preset]
            end_dt = pd.Timestamp.now()
            start_str = (end_dt - pd.DateOffset(years=years)).strftime("%Y-%m-%d")
            end_str = end_dt.strftime("%Y-%m-%d")
            confidence = st.selectbox("Confiance VaR / CVaR", [0.90, 0.95, 0.99], index=1, format_func=lambda x: f"{int(x * 100)} %", key="risk_conf")
            horizon = st.selectbox("Horizon", [1, 10], index=0, format_func=lambda h: f"{h} jour" + ("s" if h > 1 else ""), key="risk_horizon")
            benchmark_label = st.selectbox("Benchmark (bêta pour stress)", ["Aucun"] + list(capm_service.BENCHMARK_OPTIONS.keys()), index=1, key="risk_benchmark")
            benchmark_ticker = None if benchmark_label == "Aucun" else capm_service.BENCHMARK_OPTIONS[benchmark_label]

        weights = None
        if weight_mode == "Personnalisés" and tickers:
            st.markdown("**Poids personnalisés (%)** — normalisés à 100 %.")
            cols = st.columns(min(len(tickers), 4))
            raw = [cols[i % len(cols)].number_input(t, 0.0, 100.0, round(100 / len(tickers), 1), 1.0, key=f"risk_w_{t}") for i, t in enumerate(tickers)]
            weights = list(raw)
    st.divider()

    if len(tickers) < 2:
        st.info("Sélectionnez au moins 2 actifs.")
        return

    weights_tuple = tuple(weights) if weights is not None else None
    with st.spinner("Calcul des indicateurs de risque…"):
        result = _load_risk(tuple(tickers), weights_tuple, start_str, end_str, aum, confidence, horizon, benchmark_ticker)

    if not result["success"]:
        show_data_error(result.get("error"))
        return

    conf_pct = int(confidence * 100)
    methods = result["var_methods"]
    var_hist = methods["historique"]["var"]
    cvar_hist = methods["historique"]["cvar"]
    st.info(f"**Fonds :** {', '.join(result['tickers'])} · encours {_eur(aum)} · {start_str} → {end_str} · bêta {result['beta']:.2f}")

    # ----- KPI en tête -----
    kpi_row([
        (f"VaR {conf_pct} % ({horizon}j)", _eur(var_hist * aum), "var"),
        (f"CVaR {conf_pct} % ({horizon}j)", _eur(cvar_hist * aum), "cvar"),
        ("Volatilité annualisée", format_pct(result["annualized_volatility"]), "annualized_volatility"),
        ("Positions effectives", f"{result['concentration']['effective_n']:.1f}", "concentration"),
    ])

    # ----- VaR par 3 méthodes -----
    st.markdown("")
    vrow = st.columns([0.9, 0.1])
    with vrow[0]:
        st.markdown(f"**VaR / CVaR par méthode** — horizon {horizon} jour(s), confiance {conf_pct} %")
    with vrow[1]:
        info_inline("modified_var", "Méthodes de VaR")
    st.caption(f"Distribution des rendements : skewness {result['return_skew']:+.2f} · kurtosis {result['return_kurtosis']:+.2f} (>0 = queues plus épaisses qu'une loi normale).")
    df_var = pd.DataFrame([
        {
            "Méthode": _METHOD_LABELS[m],
            "VaR (%)": f"{v['var'] * 100:.2f} %",
            "VaR (€)": _eur(v["var"] * aum),
            "CVaR (€)": _eur(v["cvar"] * aum),
        }
        for m, v in methods.items()
    ])
    st.dataframe(df_var, use_container_width=True, hide_index=True)

    # ----- Graphe héros : stress tests -----
    st.markdown("")
    hrow = st.columns([0.9, 0.1])
    with hrow[0]:
        st.subheader("Stress tests — impact sur l'encours")
    with hrow[1]:
        info_inline("stress_test", "Stress tests")
    stress = result["stress_scenarios"]
    names = [s["name"] for s in stress]
    pnl = [s["pnl_eur"] for s in stress]
    fig = go.Figure(go.Bar(
        x=names, y=pnl, marker=dict(color=COLOR_NEGATIVE),
        text=[_eur(p) for p in pnl], textposition="outside",
        hovertemplate="%{x}<br>P&L : %{text}<extra></extra>", customdata=pnl,
    ))
    fig.update_layout(title=f"Perte estimée par scénario (bêta {result['beta']:.2f} × encours)", xaxis_title="", yaxis_title="P&L (€)")
    apply_theme(fig, legend=False)
    st.plotly_chart(fig, use_container_width=True)
    hw = result["historical_worst"]
    st.caption(
        f"Stress **hypothétiques** : choc de marché appliqué via le bêta du portefeuille. "
        f"Stress **historique observé** — pire jour : {_eur(hw['worst_1d_eur'])} ({hw['worst_1d_pct'] * 100:.1f} %) · "
        f"pire semaine (5 j) : {_eur(hw['worst_5d_eur'])} ({hw['worst_5d_pct'] * 100:.1f} %)."
    )

    # ----- Avancé -----
    with advanced_expander():
        # Concentration
        st.markdown("**Concentration**")
        conc = result["concentration"]
        cc1, cc2, cc3 = st.columns(3)
        with cc1:
            metric_with_info("Indice HHI", f"{conc['hhi']:.2f}", "concentration")
        with cc2:
            metric_with_info("Poids du top-3", format_pct(conc["top3"]), "concentration")
        with cc3:
            metric_with_info("Positions effectives", f"{conc['effective_n']:.1f}", "concentration")

        # Décomposition de la VaR (Component / Marginal / Incremental)
        dec_row = st.columns([0.9, 0.1])
        with dec_row[0]:
            st.markdown("**Décomposition de la VaR (Component / Marginal / Incremental)**")
        with dec_row[1]:
            info_inline("component_var", "Décomposition de la VaR")
        vd = result["var_decomposition"]
        tickers_arr = np.array(result["tickers"])
        comp_eur = vd["component_var_eur"]
        order = np.argsort(comp_eur)  # la plus négative (plus risquée) en premier
        fig_c = go.Figure(go.Bar(
            x=tickers_arr[order], y=comp_eur[order], marker=dict(color=COLOR_NEGATIVE),
            text=[_eur(v) for v in comp_eur[order]], textposition="outside",
            hovertemplate="%{x} : %{text}<extra></extra>",
        ))
        fig_c.update_layout(title="Component VaR par position (Σ = VaR du fonds)", xaxis_title="", yaxis_title="Component VaR (€)")
        apply_theme(fig_c, height=320, legend=False)
        st.plotly_chart(fig_c, use_container_width=True)
        df_dec = pd.DataFrame({
            "Actif": tickers_arr,
            "Component VaR (€)": [_eur(v) for v in comp_eur],
            "% du risque": [f"{p:.0f} %" for p in vd["component_var_pct"]],
            "Incremental VaR (€)": [_eur(v) for v in vd["incremental_var_eur"]],
        })
        st.dataframe(df_dec, use_container_width=True, hide_index=True)
        worst = tickers_arr[int(np.argmin(vd["marginal_var"]))]
        st.caption(
            f"**Marginal VaR** la plus élevée : **{worst}** (la ligne qui accroît le plus la VaR à la marge). "
            "**Component VaR** : somme = VaR du fonds. **Incremental VaR** < 0 = la ligne ajoute du risque."
        )

        # Backtesting VaR (Kupiec)
        k = result["kupiec"]
        krow = st.columns([0.9, 0.1])
        with krow[0]:
            st.markdown("**Backtesting de la VaR (test de Kupiec, 99 %)**")
        with krow[1]:
            info_inline("var_backtesting", "Backtesting de la VaR")
        k1, k2, k3 = st.columns(3)
        with k1:
            metric_with_info("Dépassements observés", f"{k['observed']} / {k['n']} j", "var_backtesting")
        with k2:
            metric_with_info("Attendus", f"{k['expected']:.1f}", "var_backtesting")
        with k3:
            metric_with_info("p-value (Kupiec)", f"{k['p_value']:.2f}", "var_backtesting")
        if k["passed"]:
            st.caption(f"✅ Modèle validé : {k['observed']} dépassements pour {k['expected']:.1f} attendus (ratio {k['ratio']:.2f}×), écart non significatif.")
        else:
            st.caption(f"⚠️ Modèle rejeté : {k['observed']} dépassements pour {k['expected']:.1f} attendus (ratio {k['ratio']:.2f}×) — VaR probablement sous-estimée.")
