<<<<<<< HEAD
# FinLab
=======
# FinLab

**FinLab** is a personal finance dashboard built with Streamlit. It brings together five classic quantitative finance tools in a single web app: CAPM/Beta, portfolio optimization, bond pricing, option pricing, and Monte Carlo simulation.

---

## Purpose

This project is a **decision-support and learning tool**: it helps explore asset risk (Beta, cost of equity), build efficient portfolios (Markowitz), value bonds (duration, convexity), price options (Black-Scholes, Greeks), and simulate long-term wealth paths (Monte Carlo, VaR). Data is loaded from Yahoo Finance where needed; the rest is configurable via the app. The goal is to have one place to run these analyses without switching between spreadsheets or scripts.

---

## What’s inside

| Tab | Content |
|-----|--------|
| **Beta (CAPM)** | Regression vs a benchmark, Beta, cost of equity, Jensen’s Alpha, rolling Beta. |
| **Portfolio (Markowitz)** | Efficient frontier, max Sharpe, min variance or target return, constraints, benchmark comparison. |
| **Bonds** | Price, YTM, Macaulay/Modified duration, convexity, price–yield curve. |
| **Options (Black-Scholes)** | Call/Put price, implied vol, Greeks (Delta, Gamma, Theta, Vega, Rho), sensitivity charts. |
| **Simulation (Monte Carlo)** | GBM paths, fan chart, VaR/CVaR, shortfall probability, optional flows. |

---

## Tech stack

- **Streamlit** for the UI and tabs  
- **yfinance** for market data (cached)  
- **pandas**, **numpy**, **scipy**, **statsmodels** for calculations  
- **Plotly** for interactive charts  

---

## How to run

```bash
# Clone the repo (or use your local folder)
git clone https://github.com/YOUR_USERNAME/FinLab.git
cd FinLab

# Install dependencies
pip install -r requirements.txt

# Launch the app
streamlit run app.py
```

Then open the URL shown in the terminal (usually `http://localhost:8501`).

**Dev mode** (auto-reload when you edit code):

```bash
python run_dev.py
```

---

## Project structure

```
FinLab/
├── app.py              # Entry point, tabs, layout
├── config.py           # Defaults (rates, tickers, etc.)
├── sections/           # One module per tab (beta, portfolio, bonds, options, simulation)
├── services/           # Business logic (CAPM, Markowitz, bonds, options, Monte Carlo)
└── utils/              # Plot config, formatters
```

---

*Personal project — use at your own risk. Not financial advice.*
>>>>>>> 90a1612 (Premier envoi de mon projet Streamlit)
