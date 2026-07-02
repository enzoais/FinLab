# Lessons — FinLab

## Widgets Streamlit : clé unique obligatoire entre onglets (2026-07-02)
`st.tabs` rend le contenu de **tous** les onglets dans le même run. Deux widgets d'entrée
(`number_input`, `selectbox`, `radio`, `text_input`, `text_area`, `checkbox`) avec des
paramètres identiques dans des onglets différents → `StreamlitDuplicateElementId`.
→ Toujours donner un `key=` unique préfixé par onglet (ex. `beta_rf`, `pf_rf`, `bond_rf`).
Note : `st.popover`, `st.expander`, `st.metric`, `st.plotly_chart` sont des conteneurs/affichage,
pas des widgets à état → pas de collision (plusieurs boutons ⓘ `st.popover("ⓘ")` cohabitent sans clé).

## Options : défaut spot manuel « à la monnaie » pour une démo lisible (2026-07-02)
Défaut sur « Ticker » (spot AAPL ~308) avec strike 100 = très dans la monnaie → Delta=1, Gamma/Vega≈0
(corrects mais illisibles). Défaut = « Spot manuel » (100/100, ATM) → Greeks parlants dès l'ouverture.

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
