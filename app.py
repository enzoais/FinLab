"""
FinLab — tab-based finance app.
"""
import importlib
import streamlit as st

from sections import beta, portfolio, bonds, options, simulation

importlib.reload(beta)
importlib.reload(portfolio)
importlib.reload(bonds)
importlib.reload(options)
importlib.reload(simulation)

st.set_page_config(
    page_title="FinLab",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="collapsed",
)

st.title("FinLab")
st.markdown("CAPM · Portfolio · Bonds · Options · Simulation")

tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "Beta (CAPM)",
    "Portfolio (Markowitz)",
    "Bonds",
    "Options (Black-Scholes)",
    "Simulation (Monte Carlo)",
])

with tab1:
    beta.render()

with tab2:
    portfolio.render()

with tab3:
    bonds.render()

with tab4:
    options.render()

with tab5:
    simulation.render()
