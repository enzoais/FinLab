"""Tests d'intégrité du catalogue d'actions (recherche filtrable)."""
import pytest

from config import DEFAULT_ASSET_TICKER, DEFAULT_MARKOWITZ_TICKERS
from utils.tickers_catalog import CATALOG, CATALOG_LABELS, label_for_ticker, ticker_for_label


def test_catalog_non_empty_and_consistent():
    assert len(CATALOG) > 50
    assert CATALOG_LABELS == list(CATALOG.keys())
    for label, ticker in CATALOG.items():
        assert label.strip(), "libellé vide"
        assert ticker.strip() == ticker.strip().upper(), f"ticker non normalisé : {ticker}"
        assert " " not in ticker, f"ticker avec espace : {ticker}"


def test_reverse_mapping_roundtrip():
    for label, ticker in CATALOG.items():
        assert ticker_for_label(label) == ticker
        assert label_for_ticker(ticker) == label


def test_helpers_fallback_on_unknown():
    assert label_for_ticker("rblx") == "RBLX"
    assert ticker_for_label("RBLX") == "RBLX"
    assert label_for_ticker("") == ""
    assert ticker_for_label("") == ""


def test_default_tickers_are_in_catalog():
    # Les défauts servent de valeurs pré-sélectionnées du menu : ils doivent y figurer.
    assert DEFAULT_ASSET_TICKER in CATALOG.values()
    for t in DEFAULT_MARKOWITZ_TICKERS:
        assert t in CATALOG.values(), f"défaut absent du catalogue : {t}"
