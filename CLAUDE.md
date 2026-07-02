# FinLab — labo de finance quantitative (Streamlit / Python)

App Streamlit à 5 onglets : CAPM/Beta, Markowitz, obligations, Black-Scholes, Monte Carlo.
Données Yahoo Finance (yfinance). Déployée sur Streamlit Cloud depuis `main`.

## Architecture (à respecter)
- `services/` : logique métier pure, **sans Streamlit** → testable en isolation.
- `sections/` : UI uniquement (un module par onglet), appelle les services.
- `utils/` : formatters et style de graphes partagés.
- `config.py` : valeurs par défaut. `app.py` : point d'entrée (onglets).

## Mémoire projet
@.claude/memory/context.md
@.claude/memory/lessons.md

## Commandes
- Tests : `.venv/bin/python -m pytest`
- Lancer : `.venv/bin/streamlit run app.py` (ou `python run_dev.py` en dev)
- Déploiement : push sur `main` → Streamlit Cloud redéploie automatiquement.

## Règles
- Python : fonctions pures dans `services/`, jamais d'import Streamlit dedans.
- Toute nouvelle formule financière → un test dans `tests/` avec valeur analytique connue.
- Commentaires en français, code/noms en anglais.
