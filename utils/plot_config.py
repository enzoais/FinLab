"""Template Plotly commun (palette claire & pro) appliqué à tous les graphes FinLab."""
from __future__ import annotations

import plotly.graph_objects as go

# ----- Palette (cohérente avec le thème Streamlit clair) -----
COLOR_PRIMARY = "#2563EB"    # bleu sobre (série principale, courbe héros)
COLOR_ACCENT = "#059669"     # vert (série secondaire, positif)
COLOR_POSITIVE = "#059669"   # vert (gains, poids longs)
COLOR_NEGATIVE = "#DC2626"   # rouge (pertes, poids courts, VaR)
COLOR_BENCHMARK = "#F59E0B"  # ambre (benchmark)
COLOR_MUTED = "#94A3B8"      # gris ardoise clair (points secondaires, grille)
COLOR_TEXT = "#1E293B"       # ardoise (texte)
GRID_COLOR = "rgba(148, 163, 184, 0.25)"  # grille légère

# Séquence de couleurs pour séries multiples (bandes de percentiles, etc.)
COLOR_SEQUENCE = [COLOR_PRIMARY, COLOR_ACCENT, COLOR_BENCHMARK, "#7C3AED", "#DB2777", "#0891B2"]

# Hauteur homogène des graphes « héros »
PLOT_HEIGHT = 420

# Style d'info-bulle commun (fond clair, bord discret)
HOVERLABEL = dict(
    bgcolor="rgba(255, 255, 255, 0.96)",
    bordercolor="rgba(148, 163, 184, 0.4)",
    font=dict(family="sans-serif", size=13, color=COLOR_TEXT),
    align="left",
    namelength=0,
)


def apply_theme(fig: go.Figure, *, height: int | None = None, legend: bool = True) -> go.Figure:
    """
    Applique le template FinLab commun à une figure Plotly.
    Fond blanc, police ardoise, grille légère, marges/hauteur homogènes, légende en haut.
    À appeler sur CHAQUE figure juste avant `st.plotly_chart`.
    """
    fig.update_layout(
        template="plotly_white",
        height=height or PLOT_HEIGHT,
        margin=dict(t=56, b=48, l=56, r=32),
        font=dict(family="sans-serif", size=13, color=COLOR_TEXT),
        title=dict(font=dict(size=16, color=COLOR_TEXT), x=0.0, xanchor="left"),
        paper_bgcolor="white",
        plot_bgcolor="white",
        hoverlabel=HOVERLABEL,
        colorway=COLOR_SEQUENCE,
    )
    if legend:
        fig.update_layout(
            showlegend=True,
            legend=dict(
                orientation="h",
                yanchor="bottom",
                y=1.02,
                xanchor="right",
                x=1,
                font=dict(size=12),
            ),
        )
    else:
        fig.update_layout(showlegend=False)
    fig.update_xaxes(showgrid=True, gridwidth=1, gridcolor=GRID_COLOR, zeroline=False)
    fig.update_yaxes(showgrid=True, gridwidth=1, gridcolor=GRID_COLOR, zeroline=False)
    return fig
