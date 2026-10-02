"""Shared chart style for notebooks and reports."""
from __future__ import annotations

import matplotlib.pyplot as plt

from src.db import PROJECT_ROOT

FIG_DIR = PROJECT_ROOT / "reports" / "figures"

# Text and surface
INK, INK_2, MUTED, GRID, SURFACE = "#0b0b0b", "#52514e", "#a3a29c", "#e6e5e0", "#fcfcfb"
# Series colours (fixed order)
BLUE, ORANGE, AQUA = "#2a78d6", "#eb6834", "#1baf7a"
BLUE_LIGHT = "#cfe0f6"

# Market stress periods shaded on time-series charts
CRISES = {
    "Dot-com bust": ("2000-03-01", "2003-03-31"),
    "Financial crisis": ("2007-08-01", "2009-06-30"),
    "Euro debt crisis": ("2011-07-01", "2012-07-31"),
    "COVID-19": ("2020-02-15", "2020-06-30"),
    "Energy & rate shock": ("2022-02-15", "2022-10-31"),
}


def set_style() -> None:
    """Light surface, recessive axes and grid, left-aligned bold titles."""
    plt.rcParams.update({
        "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE,
        "axes.edgecolor": GRID, "axes.labelcolor": INK_2, "axes.titlecolor": INK,
        "axes.titlesize": 12, "axes.titleweight": "bold", "axes.titlelocation": "left",
        "axes.spines.top": False, "axes.spines.right": False, "axes.axisbelow": True,
        "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.8,
        "xtick.color": INK_2, "ytick.color": INK_2, "font.size": 10,
        "legend.frameon": False, "figure.dpi": 110,
    })


def shade_crises(ax, label: bool = True, y: float = 0.97) -> None:
    """Shade the stress periods in CRISES on a time-series axis."""
    import pandas as pd

    for name, (start, end) in CRISES.items():
        start, end = pd.Timestamp(start), pd.Timestamp(end)
        ax.axvspan(start, end, color=GRID, alpha=0.7, lw=0, zorder=0)
        if label:
            ax.text(start + (end - start) / 2, y, name.replace(" ", "\n", 1), transform=ax.get_xaxis_transform(),
                    ha="center", va="top", fontsize=8, color=INK_2)


def save(fig, name: str) -> None:
    """Save a figure to reports/figures/<name>.png."""
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    fig.savefig(FIG_DIR / f"{name}.png", dpi=150)
