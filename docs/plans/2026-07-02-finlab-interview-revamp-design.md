# FinLab — Refonte « entretien » (design doc)

Date : 2026-07-02
Contexte : entretien quant le lendemain. FinLab est sur le CV comme
« Quantitative Backtesting Engine (Python) ». Objectif : app **fidèle au CV**,
**démontrable en live** (partage d'écran) et utilisable comme **outil de révision**.

## Objectifs

1. **Fidélité au CV** — combler le seul vrai trou : le mot **« backtesting »** n'existe
   nulle part dans le code aujourd'hui (pricing/optimisation/simulation, mais aucune
   stratégie rejouée sur l'historique). → Ajouter un onglet Backtest réel.
2. **Démo live** — design **clair & pro**, écran épuré, zéro stack trace en public.
3. **Révision** — à côté de **chaque chiffre / notion complexe**, un petit **bouton `ⓘ`
   cliquable** (`st.popover`) ouvre un panneau expliquant simplement : c'est quoi, à quoi
   ça sert, comment ça se calcule (sans équations). **Pas de page séparée** :
   l'explication est au contact direct de la donnée. Couvrir **toutes** les notions.

## Décisions validées

- **Plateforme** : on **garde Streamlit** (bon outil pour démo locale + révision ;
  ne pas migrer la veille). Argument d'archi à l'oral : moteur Python pur et testé
  dans `services/`, UI mince par-dessus.
- **Style** : **clair & pro** (fond clair, accent sobre, beaucoup d'espace).
- **Nettoyage** : **couper franchement** — on retire les métriques trop techniques
  non défendables à l'oral (t-stats, p-values, IC, condition number, nuage aléatoire…).
- **Glossaire** : bouton `ⓘ` cliquable (`st.popover`) à côté de chaque chiffre/notion,
  ouvrant un panneau d'explication au clic. **Pas de page/onglet dédié.**

## Architecture

Principe conservé : `services/` = logique pure (sans Streamlit, testée pytest),
`sections/` = UI, `utils/` = partagé. On ajoute :

- `services/backtest_service.py` — nouveau moteur de backtest (pur, testé).
- `utils/glossary.py` — `GLOSSARY: dict[str, str]` (terme → explication simple, FR),
  **exhaustif** (toutes les notions de l'app).
- `utils/ui.py` — helpers UI : `metric_with_info(label, value, term_key)` (une carte
  `st.metric` + un bouton `ⓘ` `st.popover` qui ouvre le panneau d'explication au clic),
  `kpi_row(...)`, en-têtes homogènes, expander « Avancé ».
- `utils/plot_config.py` — un `apply_theme(fig)` pour un template Plotly commun.
- `.streamlit/config.toml` — section `[theme]` claire & pro.
- `sections/backtest.py` — nouvel onglet Backtest.

## Onglet Backtest (nouveau)

- **Entrées** : tickers + poids (équipondéré par défaut / poids perso / « poids
  Markowitz max-Sharpe »), période, capital initial, rééquilibrage (aucun / mensuel /
  trimestriel), benchmark.
- **Cœur** : téléchargement historique (réutilise `download_prices_multi`), courbe de
  capital du portefeuille (buy & hold ou rebalancé), série de drawdown.
- **Sorties (stats du CV, sur l'historique réel)** : rendement total, **CAGR**, vol
  annualisée, **Sharpe**, **max drawdown**, VaR/CVaR historiques, comparaison vs
  benchmark (alpha, tracking error, information ratio).
- **Visuels** : courbe de capital (portefeuille vs benchmark) + courbe de drawdown.

## Nettoyage par onglet (couper franchement)

| Onglet | On garde (en tête + héros) | On coupe |
|--------|----------------------------|----------|
| **Beta** | Beta, Kₑ, Alpha de Jensen, R² ; droite de régression ; rolling beta | Adjusted Beta, Treynor, t-stat, p-value, IC, n_obs |
| **Portfolio** | poids optimaux, rendement, vol, Sharpe ; frontière efficiente ; VaR/CVaR ; comparaison benchmark | nuage aléatoire, condition number, décomposition du risque, matrice de corrélation |
| **Bonds** | prix, YTM, Macaulay, Modified, convexité ; courbe prix-taux | intérêts courus 30/360, dirty/clean, table des cash-flows (→ Avancé) |
| **Options** | call/put, les 5 Greeks, IV ; 1 graphe de sensibilité | put-call parity, table de scénarios détaillée |
| **Simulation** | fan chart ; stats terminales (p5/p50/p95, moyenne) ; VaR/CVaR ; max drawdown ; proba shortfall | détails de percentiles superflus |
| **Backtest** | courbe de capital vs benchmark ; drawdown ; total return, CAGR, Sharpe, max DD, VaR/CVaR | — |

Rien n'est supprimé des `services/` : on ne retire que de l'**affichage**. Les tests
services restent verts.

## Design system (clair & pro)

- Thème Streamlit clair : accent sobre (bleu/vert), fond blanc, secondaire gris très
  clair, texte ardoise, police sans-serif.
- Template Plotly commun (`apply_theme`) : même palette, grille légère, marges/hauteur
  homogènes (déjà dans `config.py`).
- Grammaire par onglet : **3-4 KPI en tête** → **1 graphe héros** → **expander
  « Avancé »** pour le reste.
- Chaque KPI/notion porte son bouton **`ⓘ`** (`st.popover`) ouvrant l'explication
  pédagogique au clic (glossaire au contact de la donnée, pas de page dédiée).

## Gestion d'erreurs (démo live)

- Yahoo KO / ticker invalide → message clair (`st.warning`/`st.info`), jamais de
  traceback à l'écran. Valeurs par défaut qui marchent du premier coup.

## Tests

- `tests/test_backtest_service.py` : courbe de capital sur returns constants, max
  drawdown connu, signe du Sharpe, équipondéré vs pondéré, rééquilibrage cohérent.
- `tests/test_glossary.py` : intégrité (toutes les entrées non vides, clés attendues).
- La suite existante (53 tests) doit rester verte (services inchangés).
- Vérification visuelle par screenshots (MCP chrome-devtools) après chaque onglet.

## Priorité d'exécution (une nuit)

1. Design system + thème + helpers UI (dont `metric_with_info` / popover `ⓘ`).
2. Nettoyage + glossaire (popovers `ⓘ`) des onglets existants — même passage.
3. Onglet **Backtest** (construit directement dans le style propre, avec ses popovers).
4. Bonus si temps : recherche de tickers interactive.

Honnêteté : perfectionner les 6 onglets + backtest + glossaire en une nuit est tendu ;
on sécurise dans cet ordre, le plus important pour l'entretien passe en premier.
