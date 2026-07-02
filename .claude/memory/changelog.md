# Changelog — FinLab

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
