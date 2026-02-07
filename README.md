# FinLab

**FinLab** is a personal quantitative finance dashboard built with Streamlit. It brings together five classic tools in a single web app: CAPM/Beta, Markowitz portfolio optimization, bond pricing, Black–Scholes option pricing, and Monte Carlo wealth simulation.

---

## Purpose

This project is a **decision-support and learning tool**. It helps you:

- **Explore asset risk** — Beta, cost of equity, Jensen’s Alpha, rolling Beta vs a benchmark  
- **Build efficient portfolios** — Efficient frontier, max Sharpe, min variance or target return, constraints, benchmark comparison  
- **Value bonds** — Price, YTM, Macaulay/Modified duration, convexity, price–yield curve  
- **Price options** — Call/Put (Black–Scholes), implied vol, Greeks (Delta, Gamma, Theta, Vega, Rho), sensitivity charts  
- **Simulate long-term wealth** — GBM paths, fan chart, VaR/CVaR, shortfall probability, optional cash flows  

Data is loaded from **Yahoo Finance** where needed; the rest is configurable via the app. The goal is to have one place to run these analyses without switching between spreadsheets or scripts.

---

## What’s inside

| Tab | Content |
|-----|--------|
| **Beta (CAPM)** | Regression vs benchmark, Beta, cost of equity, Jensen’s Alpha, rolling Beta (configurable window). |
| **Portfolio (Markowitz)** | Efficient frontier, max Sharpe, min variance or target return, constraints, benchmark comparison. |
| **Bonds** | Price, YTM, Macaulay/Modified duration, convexity, price–yield curve. |
| **Options (Black–Scholes)** | Call/Put price, implied vol, Greeks (Delta, Gamma, Theta, Vega, Rho), sensitivity charts. |
| **Simulation (Monte Carlo)** | GBM paths, fan chart, VaR/CVaR, shortfall probability, optional flows. |

---

## Tech stack

| Role | Libraries |
|------|-----------|
| **UI** | Streamlit (tabs, forms, caching) |
| **Data** | yfinance (market data, cached), pandas, numpy |
| **Calculations** | scipy, statsmodels (CAPM regression, optimization) |
| **Charts** | Plotly (interactive), matplotlib |

---

## Prerequisites

- **Python 3.10+** (recommended)
- pip (or your preferred package manager)

---

## How to run

```bash
# Clone the repo (or use your local folder)
git clone https://github.com/YOUR_USERNAME/FinLab.git
cd FinLab

# Create a virtual environment (optional but recommended)
python -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Launch the app
streamlit run app.py
```

Then open the URL shown in the terminal (usually `http://localhost:8501`).

### Dev mode (auto-reload)

To auto-reload when you edit code:

```bash
python run_dev.py
```

---

## Project structure

```
FinLab/
├── app.py                    # Entry point, tabs, layout
├── config.py                 # Defaults (rates, tickers, cache TTL, plot settings)
├── run_dev.py                # Dev launcher with file watching
├── requirements.txt          # Dependencies
├── sections/                 # One module per tab (UI only)
│   ├── beta.py               # CAPM / Beta tab
│   ├── portfolio.py          # Markowitz portfolio tab
│   ├── bonds.py              # Bond pricing tab
│   ├── options.py            # Black–Scholes / Options tab
│   └── simulation.py         # Monte Carlo simulation tab
├── services/                 # Business logic (no Streamlit)
│   ├── capm_service.py       # CAPM regression, rolling Beta
│   ├── markowitz_service.py  # Efficient frontier, optimization
│   ├── fixed_income_service.py # Bond price, duration, convexity
│   ├── black_scholes_service.py # Option price, Greeks, implied vol
│   └── monte_carlo_service.py  # GBM paths, VaR, shortfall
└── utils/                    # Shared helpers
    ├── plot_config.py        # Plotly layout (height, margins, hover)
    └── formatters.py         # Number/percentage formatting
```

---

## Configuration

Key defaults are in `config.py`: risk-free rate, default tickers (e.g. AAPL, ^GSPC), bond parameters, Black–Scholes defaults, Monte Carlo horizon/paths/VaR confidence, and cache TTL for Yahoo Finance data. You can change them there or override via the app inputs.

---

## Disclaimer

*Personal project — use at your own risk. Not financial advice.*
