# Contexte — FinLab

## État actuel (2026-07-02)
App fonctionnelle et déployée (Streamlit Cloud, `main` → https://final-dash.streamlit.app).
Session de revue + corrections effectuée : voir changelog.

## Ce qui vient d'être fait
- Suite de tests créée (`tests/`, 53 tests, tous verts) sur les 5 services.
- Bug de drift Monte Carlo corrigé (drift arithmétique cohérent).
- Code mort nettoyé (formatters dédupliqués, `importlib.reload` retiré de `app.py`).

## Prochaines étapes possibles
- Découper les gros fichiers de `sections/` (options.py 658 l., portfolio.py, bonds.py) — dépassent la règle 300 lignes.
- Ajouter des tests d'intégration sur les pipelines `run_*_analysis` (avec yfinance mické).
- Revoir l'annualisation du rendement de marché attendu (CAPM) : `mean*252` sur log-returns = approximation.

## Bugs connus
- Aucun bloquant. Voir lessons.md pour les pièges identifiés.
