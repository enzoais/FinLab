# Contexte — FinLab

## État actuel (2026-07-02)
Refonte « entretien quant » livrée. App claire & pro, **7 onglets** (Bêta, Portefeuille, Obligations,
Options, Simulation, Backtest, **Risque**), déployée sur `main` → Streamlit Cloud.
Tests : **160 verts**. Aucune dépendance pip ajoutée.
Cible entretien : alternance **Risques d'Investissement** (banque privée RMM / Rothschild & Co) — l'onglet
Risque (VaR 3 méthodes, stress tests, concentration, Kupiec) couvre le cœur du poste ; recherche d'actions
filtrable (catalogue Nom→ticker) ; DV01/CS01 pour la partie fixed income du CV (HSBC).

## Ce qui vient d'être fait (session refonte)
- **Design system clair & pro** : `.streamlit/config.toml` (fond blanc, accent bleu #2563EB, texte ardoise),
  `utils/plot_config.apply_theme(fig)` appliqué à TOUS les graphes (palette, grille légère, marges homogènes).
- **Glossaire au clic** : `utils/glossary.py` (70 entrées FR, 3 volets : c'est quoi / à quoi ça sert / comment ça se calcule),
  helpers `utils/ui.py` (`metric_with_info`, `kpi_row`, `section_header`, `advanced_expander`, `show_data_error`).
  Bouton ⓘ (`st.popover`) au contact de chaque chiffre — pas de page dédiée.
- **Grammaire par onglet** : 3-4 KPI en tête → 1 graphe héros → expander « Avancé ». Coupe franche des métriques
  non défendables (t-stats, p-values, IC, condition number, nuage aléatoire, corrélation, parité put-call, heatmap,
  clean/dirty, scénarios verbeux, spaghetti…). Rien retiré des `services/`.
- **Nouvel onglet Backtest** : `services/backtest_service.py` (pur, testé) + `sections/backtest.py`. Courbe de
  capital vs benchmark, drawdown, CAGR/Sharpe/max DD/VaR/CVaR, alpha/TE/IR. Poids équipondéré / perso / Markowitz max-Sharpe.
- **UI en français**, erreurs propres (jamais de traceback : `show_data_error` + garde-fou try/except dans `app.py`).

## Comment lancer
- Tests : `.venv/bin/python -m pytest`
- App locale : `.venv/bin/python -m streamlit run app.py --server.headless true --server.port 8599`
- Déploiement : push sur `main` → Streamlit Cloud redéploie.

## Prochaines étapes possibles
- Bonus : recherche de tickers interactive dans l'onglet Backtest.
- Découper les services les plus longs si besoin (tous < 500 l., OK pour l'instant).
- Tests d'intégration des pipelines `run_*` avec yfinance mocké.

## Bugs connus
- Aucun bloquant. Voir lessons.md (piège des clés de widgets Streamlit entre onglets).
