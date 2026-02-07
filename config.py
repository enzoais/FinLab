"""
App config: defaults and cache.
"""

# Cache
CACHE_TTL_SECONDS = 3600  # 1 hour for yfinance data

# Defaults (can be overridden by user inputs)
DEFAULT_RISK_FREE_RATE = 0.05
DEFAULT_PERIOD_YEARS = 5
DEFAULT_ASSET_TICKER = "AAPL"
DEFAULT_BENCHMARK = "^GSPC"  # S&P 500
DEFAULT_MARKOWITZ_TICKERS = ["AAPL", "MSFT", "GOOGL", "AMZN"]

# Fixed Income (bond pricing)
DEFAULT_BOND_FACE_VALUE = 100
DEFAULT_BOND_COUPON_RATE = 0.05  # 5%
DEFAULT_BOND_MATURITY_YEARS = 10
DEFAULT_BOND_FREQUENCY = 2  # semi-annual
DEFAULT_BOND_YTM = 0.05  # 5% (user input for pricing)
DEFAULT_BOND_YIELD_CURVE_POINTS = 50

# Rolling windows (days)
ROLLING_WINDOWS = [60, 126, 252]  # ~3m, 6m, 1y

# Black-Scholes (option pricing)
DEFAULT_BS_SPOT = 100.0
DEFAULT_BS_STRIKE = 100.0
DEFAULT_BS_TIME_TO_EXPIRY = 1.0  # years
DEFAULT_BS_VOLATILITY = 0.20  # 20% annual

# Monte Carlo (wealth simulation)
DEFAULT_MC_INITIAL_WEALTH = 100_000.0
DEFAULT_MC_HORIZON_YEARS = 20.0
DEFAULT_MC_DRIFT = 0.07  # 7% annual
DEFAULT_MC_VOLATILITY = 0.18  # 18% annual
DEFAULT_MC_PATHS = 5_000
DEFAULT_MC_STEPS_PER_YEAR = 252
DEFAULT_MC_VAR_CONFIDENCE = 0.95
DEFAULT_MC_SEED = None  # set to int for reproducibility

# Plotly layout
PLOT_HEIGHT = 400
PLOT_MARGIN = dict(t=50, b=50, l=50, r=50)
