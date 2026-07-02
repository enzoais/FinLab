# Changelog — FinLab

## 2026-07-02 — Onglet Risque (alignement poste RMM Risques d'Investissement)
- `services/risk_service.py` (pur, testé) : VaR 3 méthodes (historique, paramétrique, Monte-Carlo) + CVaR, mise à l'horizon (√t), stress tests (choc marché × bêta + pire jour/semaine observés), concentration (HHI, positions effectives, top-3), contribution au risque par ligne, backtesting VaR (test de Kupiec).
- `sections/risk.py` : 7ᵉ onglet. KPI (VaR/CVaR €, vol, positions effectives) → table VaR 3 méthodes → héros stress tests (€) → Avancé (concentration, contribution, Kupiec). Encours/confiance/horizon/benchmark paramétrables.
- Glossaire : `stress_test`, `parametric_var`, `concentration`, `risk_contribution`, `var_backtesting`.
- Choix honnête : stress 2008/COVID = chocs hypothétiques via bêta (pas de données 2008 sur 5 ans), assumé à l'écran.
- Tests : `tests/test_risk_service.py` (12) + clés glossaire → **160 verts**. Toujours aucune dépendance pip.

## 2026-07-02 — DV01 / CS01 (fixed income, alignement CV HSBC)
- `fixed_income_service` : ajout `dv01` (repricing +1 bp de taux) et `cs01` (spread) ; exposés par `run_bond_analysis`.
- Onglet Obligations : cartes DV01/CS01 (+ⓘ) + nouveau mode « Sans risque + spread crédit » (prix = actualisation au taux sans risque + spread, champ spread en bp).
- Choix honnête : pour une obligation vanille à taux fixe, CS01 = DV01 (même décalage du taux d'actualisation) — documenté (bon point à l'oral). Glossaire : entrées `dv01`, `cs01`.
- Tests : DV01 ≈ duration modifiée × prix × 1 bp, CS01 = DV01, DV01 croît avec la maturité → **143 verts**.

## 2026-07-02 — Recherche d'actions filtrable
- Remplacement des zones de texte « tape le ticker » par une recherche filtrable native (tape « app » → Apple).
- `utils/tickers_catalog.py` : catalogue Nom→ticker (US, Europe, ETF) ; helpers `asset_selectbox` / `asset_multiselect` (utils/ui.py) avec `accept_new_options` (ticker libre autorisé).
- Recâblage des 5 onglets de saisie (Bêta, Options, Simulation en simple ; Portefeuille, Backtest en multi + chips). Garde-fou « au moins 2 actifs ».
- `tests/test_tickers_catalog.py` → **137 verts**. Toujours aucune dépendance pip.

## 2026-07-02 — Refonte « entretien quant »
- Design clair & pro : `.streamlit/config.toml` (fond blanc, accent bleu, ardoise) + `apply_theme(fig)` commun à tous les graphes.
- Glossaire au clic : `utils/glossary.py` (70 entrées FR) + helpers `utils/ui.py` (`metric_with_info`/popover ⓘ, `kpi_row`, `section_header`, `advanced_expander`, `show_data_error`).
- Refonte des 5 onglets à la grammaire KPI (ⓘ) → graphe héros → « Avancé » ; coupe franche des métriques non défendables (affichage seul, services intacts). UI en français.
- Nouvel onglet **Backtest** : `services/backtest_service.py` (pur, testé) + `sections/backtest.py`. Courbe de capital vs benchmark, drawdown, CAGR/Sharpe/max DD/VaR-CVaR, alpha/TE/IR. Poids équipondéré / perso / Markowitz.
- Tests : +`tests/test_backtest_service.py`, +`tests/test_glossary.py` → **133 verts**. Aucune nouvelle dépendance.
- Erreurs propres partout (jamais de traceback) + garde-fou try/except par onglet dans `app.py`.

## 2026-07-02 — Revue + fiabilisation
- Ajout d'une suite de tests `tests/` (53 tests, tous verts) couvrant les 5 services
  (Black-Scholes, obligations, CAPM, Markowitz, Monte Carlo) sur valeurs analytiques connues.
- Fix : biais de drift Monte Carlo (`estimate_mu_sigma_from_ticker` renvoie désormais un
  drift arithmétique cohérent avec `simulate_gbm_paths`).
- Nettoyage : `_format_pct`/`_format_pvalue` dédupliqués → `utils/formatters.py` (5 sections).
- Nettoyage : retrait des `importlib.reload` de `app.py` (scaffolding dev).
- Ajout `requirements-dev.txt` (pytest hors deps de prod) et `pytest.ini`.
- Initialisation de la mémoire projet (`.claude/memory/`) et `CLAUDE.md`.
