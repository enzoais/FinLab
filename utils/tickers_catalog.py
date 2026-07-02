"""
Catalogue d'actions et ETF pour la recherche filtrable (menu déroulant natif).

Chaque entrée mappe un libellé lisible « Nom (TICKER) » vers son ticker Yahoo Finance.
L'utilisateur tape quelques lettres → le menu filtre par nom ou ticker. Les tickers hors
catalogue restent saisissables (accept_new_options), donc la liste n'a pas à être exhaustive.

Données de référence pures (aucune dépendance à Streamlit ou au réseau).
"""
from __future__ import annotations

# Libellé « Nom (TICKER) » -> ticker Yahoo Finance. Ordre = ordre d'affichage.
CATALOG: dict[str, str] = {
    # --- Méga-caps US / tech ---
    "Apple (AAPL)": "AAPL",
    "Microsoft (MSFT)": "MSFT",
    "Alphabet / Google (GOOGL)": "GOOGL",
    "Amazon (AMZN)": "AMZN",
    "Nvidia (NVDA)": "NVDA",
    "Meta / Facebook (META)": "META",
    "Tesla (TSLA)": "TSLA",
    "Broadcom (AVGO)": "AVGO",
    "Netflix (NFLX)": "NFLX",
    "Adobe (ADBE)": "ADBE",
    "Salesforce (CRM)": "CRM",
    "Oracle (ORCL)": "ORCL",
    "Advanced Micro Devices (AMD)": "AMD",
    "Intel (INTC)": "INTC",
    "Qualcomm (QCOM)": "QCOM",
    "Texas Instruments (TXN)": "TXN",
    "Micron (MU)": "MU",
    "Cisco (CSCO)": "CSCO",
    "IBM (IBM)": "IBM",
    "Palantir (PLTR)": "PLTR",
    "ServiceNow (NOW)": "NOW",
    "Uber (UBER)": "UBER",
    "Shopify (SHOP)": "SHOP",
    "PayPal (PYPL)": "PYPL",
    "Block / Square (SQ)": "SQ",
    "Snowflake (SNOW)": "SNOW",
    # --- Consommation / distribution ---
    "Walmart (WMT)": "WMT",
    "Costco (COST)": "COST",
    "Home Depot (HD)": "HD",
    "McDonald's (MCD)": "MCD",
    "Starbucks (SBUX)": "SBUX",
    "Nike (NKE)": "NKE",
    "Coca-Cola (KO)": "KO",
    "PepsiCo (PEP)": "PEP",
    "Procter & Gamble (PG)": "PG",
    "Disney (DIS)": "DIS",
    "Booking (BKNG)": "BKNG",
    # --- Finance ---
    "JPMorgan Chase (JPM)": "JPM",
    "Bank of America (BAC)": "BAC",
    "Goldman Sachs (GS)": "GS",
    "Morgan Stanley (MS)": "MS",
    "Visa (V)": "V",
    "Mastercard (MA)": "MA",
    "Berkshire Hathaway (BRK-B)": "BRK-B",
    "BlackRock (BLK)": "BLK",
    "American Express (AXP)": "AXP",
    # --- Santé ---
    "UnitedHealth (UNH)": "UNH",
    "Johnson & Johnson (JNJ)": "JNJ",
    "Eli Lilly (LLY)": "LLY",
    "Pfizer (PFE)": "PFE",
    "Merck (MRK)": "MRK",
    "AbbVie (ABBV)": "ABBV",
    "Thermo Fisher (TMO)": "TMO",
    # --- Industrie / énergie ---
    "Boeing (BA)": "BA",
    "Caterpillar (CAT)": "CAT",
    "General Electric (GE)": "GE",
    "Honeywell (HON)": "HON",
    "Exxon Mobil (XOM)": "XOM",
    "Chevron (CVX)": "CVX",
    # --- Europe (Paris) ---
    "LVMH (MC.PA)": "MC.PA",
    "L'Oréal (OR.PA)": "OR.PA",
    "TotalEnergies (TTE.PA)": "TTE.PA",
    "Airbus (AIR.PA)": "AIR.PA",
    "Sanofi (SAN.PA)": "SAN.PA",
    "Schneider Electric (SU.PA)": "SU.PA",
    "Hermès (RMS.PA)": "RMS.PA",
    "BNP Paribas (BNP.PA)": "BNP.PA",
    "Air Liquide (AI.PA)": "AI.PA",
    "Stellantis (STLAP.PA)": "STLAP.PA",
    # --- Europe (Allemagne / autres) ---
    "SAP (SAP.DE)": "SAP.DE",
    "Siemens (SIE.DE)": "SIE.DE",
    "Allianz (ALV.DE)": "ALV.DE",
    "Volkswagen (VOW3.DE)": "VOW3.DE",
    "ASML (ASML.AS)": "ASML.AS",
    "Nestlé (NESN.SW)": "NESN.SW",
    "Novo Nordisk (NVO)": "NVO",
    "Shell (SHEL.L)": "SHEL.L",
    "AstraZeneca (AZN.L)": "AZN.L",
    # --- ETF larges / obligataires / matières ---
    "S&P 500 ETF (SPY)": "SPY",
    "Nasdaq-100 ETF (QQQ)": "QQQ",
    "Total US Market ETF (VTI)": "VTI",
    "S&P 500 ETF Vanguard (VOO)": "VOO",
    "Russell 2000 ETF (IWM)": "IWM",
    "Monde développé ETF (VEA)": "VEA",
    "Marchés émergents ETF (VWO)": "VWO",
    "Europe ETF (VGK)": "VGK",
    "Obligations US agrégat ETF (AGG)": "AGG",
    "Bons du Trésor US 20+ ans ETF (TLT)": "TLT",
    "Or ETF (GLD)": "GLD",
    "Bitcoin ETF (IBIT)": "IBIT",
}

# Liste des libellés dans l'ordre d'affichage (options du menu)
CATALOG_LABELS: list[str] = list(CATALOG.keys())

# Table inverse : ticker -> libellé
_TICKER_TO_LABEL: dict[str, str] = {ticker: label for label, ticker in CATALOG.items()}


def label_for_ticker(ticker: str) -> str:
    """Libellé lisible d'un ticker (celui du catalogue, sinon le ticker seul en majuscules)."""
    if not ticker:
        return ""
    t = ticker.strip().upper()
    return _TICKER_TO_LABEL.get(t, t)


def ticker_for_label(label: str) -> str:
    """Ticker correspondant à un libellé (celui du catalogue, sinon le libellé brut en majuscules)."""
    if not label:
        return ""
    if label in CATALOG:
        return CATALOG[label]
    return label.strip().upper()
