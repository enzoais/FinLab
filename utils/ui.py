"""
Helpers UI partagés : cartes KPI avec bouton ⓘ (glossaire au clic), en-têtes homogènes,
expander « Avancé » et messages d'erreur propres (jamais de stack trace à l'écran).

Ces helpers imposent la grammaire de chaque onglet : 3-4 KPI en tête → 1 graphe héros →
le reste dans un expander replié.
"""
from __future__ import annotations

from typing import Iterable

import streamlit as st

from utils.glossary import get_glossary_text

# Libellé du bouton d'information (glossaire au clic)
INFO_LABEL = "ⓘ"
# Libellé standard de l'expander « Avancé »
ADVANCED_LABEL = "Avancé"


def _glossary_popover(term_key: str, title: str | None = None, *, label: str = INFO_LABEL) -> None:
    """Bouton ⓘ ouvrant un popover avec l'explication pédagogique du terme."""
    with st.popover(label, use_container_width=False):
        if title:
            st.markdown(f"**{title}**")
        st.markdown(get_glossary_text(term_key))


def metric_with_info(
    label: str,
    value: str,
    term_key: str,
    *,
    delta: str | None = None,
    delta_color: str = "normal",
) -> None:
    """
    Une carte `st.metric` + un bouton ⓘ qui ouvre l'explication du terme (glossaire au clic).
    À utiliser dans chaque colonne d'une ligne de KPI.
    """
    st.metric(label, value, delta=delta, delta_color=delta_color)
    _glossary_popover(term_key, title=label)


def kpi_row(items: Iterable[tuple[str, str, str]]) -> None:
    """
    Affiche une ligne de KPI homogène (3-4 max recommandés).
    `items` : itérable de tuples (label, value, term_key).
    """
    items = list(items)
    if not items:
        return
    cols = st.columns(len(items))
    for col, (label, value, term_key) in zip(cols, items):
        with col:
            metric_with_info(label, value, term_key)


def section_header(title: str, term_key: str | None = None, *, level: str = "subheader") -> None:
    """
    En-tête de section homogène, avec un bouton ⓘ optionnel à côté du titre.
    `level` : "subheader" ou "header".
    """
    col_title, col_info = st.columns([0.90, 0.10])
    with col_title:
        if level == "header":
            st.header(title)
        else:
            st.subheader(title)
    if term_key:
        with col_info:
            st.markdown("<div style='height: 0.4rem'></div>", unsafe_allow_html=True)
            _glossary_popover(term_key, title=title)


def info_inline(term_key: str, title: str | None = None) -> None:
    """Bouton ⓘ isolé (ex. à côté d'un titre de graphe ou d'une valeur en ligne)."""
    _glossary_popover(term_key, title=title)


def advanced_expander(label: str = ADVANCED_LABEL, *, expanded: bool = False):
    """Retourne l'expander « Avancé » (replié par défaut) pour y ranger le détail technique."""
    return st.expander(label, expanded=expanded)


def show_data_error(raw_error: str | None = None, *, kind: str = "warning") -> None:
    """
    Affiche un message d'erreur propre (jamais de traceback). `kind` : "warning" ou "info".
    Le détail technique éventuel est relégué en légende discrète.
    """
    message = (
        "⚠️ Impossible de charger les données pour ces paramètres. "
        "Vérifiez le ticker et la période, puis réessayez."
    )
    if kind == "info":
        st.info(message)
    else:
        st.warning(message)
    if raw_error:
        st.caption(f"Détail : {raw_error}")
