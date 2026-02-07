"""Shared number/string formatters for UI display."""


def format_pct(value: float, decimals: int = 2) -> str:
    """Format as percentage (e.g. 0.0523 -> 5.23%)."""
    return f"{value * 100:.{decimals}f}%"


def format_pvalue(p: float) -> str:
    """Format p-value (3 decimals or scientific if very small)."""
    if p is None:
        return "—"
    if p < 0.0001:
        return f"{p:.2e}"
    return f"{p:.3f}"
