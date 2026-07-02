# Lessons — FinLab

## Monte Carlo : drift arithmétique vs drift log (2026-07-02)
`simulate_gbm_paths` attend un drift **arithmétique** `mu` (SDE dS/S = mu·dt + σ·dW),
car il calcule `(mu − 0.5σ²)`. La moyenne des log-rendements estime déjà `(mu − 0.5σ²)`.
→ `estimate_mu_sigma_from_ticker` doit reconvertir : `mu = mean_log·252 + 0.5σ²`.
Sinon les trajectoires sont biaisées à la baisse de 0.5σ²/an.
Test de garde : `test_expected_terminal_matches_arithmetic_drift` (E[S_T] = S0·exp(mu·T)).

## Formatters : ne pas redéfinir en local
`utils/formatters.py` exporte `format_pct` / `format_pvalue`. Les importer, ne jamais
recréer de `_format_pct` local dans les sections (c'était dupliqué dans les 5).

## app.py : pas de scaffolding dev en prod
Ne pas remettre `importlib.reload(...)` dans `app.py` : le rechargement dev est géré
par `run_dev.py` + `runOnSave`.
