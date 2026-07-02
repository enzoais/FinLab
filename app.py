"""
FinLab — labo de finance quantitative (application à onglets).
"""
import streamlit as st

from sections import beta, portfolio, bonds, options, simulation, backtest

st.set_page_config(
    page_title="FinLab",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="collapsed",
)

st.title("FinLab")
st.caption("Labo de finance quantitative — Bêta · Portefeuille · Obligations · Options · Simulation · Backtest")

TABS = [
    ("Bêta (CAPM)", beta),
    ("Portefeuille", portfolio),
    ("Obligations", bonds),
    ("Options", options),
    ("Simulation", simulation),
    ("Backtest", backtest),
]

for tab, (label, module) in zip(st.tabs([label for label, _ in TABS]), TABS):
    with tab:
        try:
            module.render()
        except Exception as exc:  # noqa: BLE001 — garde-fou démo live : jamais de traceback à l'écran
            st.warning("⚠️ Une erreur est survenue sur cet onglet. Ajustez les paramètres et réessayez.")
            st.caption(f"Détail : {type(exc).__name__}")
