# Changelog — FinLab

## 2026-07-02 — Revue + fiabilisation
- Ajout d'une suite de tests `tests/` (53 tests, tous verts) couvrant les 5 services
  (Black-Scholes, obligations, CAPM, Markowitz, Monte Carlo) sur valeurs analytiques connues.
- Fix : biais de drift Monte Carlo (`estimate_mu_sigma_from_ticker` renvoie désormais un
  drift arithmétique cohérent avec `simulate_gbm_paths`).
- Nettoyage : `_format_pct`/`_format_pvalue` dédupliqués → `utils/formatters.py` (5 sections).
- Nettoyage : retrait des `importlib.reload` de `app.py` (scaffolding dev).
- Ajout `requirements-dev.txt` (pytest hors deps de prod) et `pytest.ini`.
- Initialisation de la mémoire projet (`.claude/memory/`) et `CLAUDE.md`.
