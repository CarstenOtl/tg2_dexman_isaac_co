"""
Thesis plot style settings — mirrors the MATLAB plot_settings() function.

Usage:
    from plot_style import apply_style, COLORS, save_fig
    apply_style()
    fig, ax = plt.subplots()
    ...
    save_fig(fig, "my_plot")
"""

import matplotlib as mpl
import matplotlib.pyplot as plt
from pathlib import Path

# ---------- Color palette (academic, distinguishable, print-safe) ----------
COLORS = {
    "blue":    "#2171B5",
    "red":     "#CB181D",
    "green":   "#238B45",
    "orange":  "#D94801",
    "purple":  "#6A51A3",
    "gray":    "#636363",
    "cyan":    "#0097A7",
    "brown":   "#8D6E63",
}

# Ordered list for cycling
COLOR_CYCLE = list(COLORS.values())

# ---------- Output directory ----------
FIGURES_DIR = Path(__file__).resolve().parent.parent / "Report" / "figures" / "plots"


def apply_style(font_size: int = 10, fig_width_cm: float = 14.0):
    """Apply consistent thesis plotting style (matching MATLAB plot_settings)."""

    fig_width_in = fig_width_cm / 2.54
    fig_height_in = fig_width_in * 0.618  # golden ratio

    mpl.rcParams.update({
        # Figure
        "figure.figsize":       (fig_width_in, fig_height_in),
        "figure.dpi":           150,
        "figure.facecolor":     "white",
        "savefig.dpi":          300,
        "savefig.bbox":         "tight",
        "savefig.pad_inches":   0.05,

        # Font — Charter is the thesis font; fall back to serif
        "font.family":          "serif",
        "font.size":            font_size,
        "axes.titlesize":       font_size,
        "axes.labelsize":       font_size,
        "xtick.labelsize":      font_size - 1,
        "ytick.labelsize":      font_size - 1,
        "legend.fontsize":      font_size - 1,

        # Axes
        "axes.linewidth":       0.6,
        "axes.grid":            True,
        "axes.prop_cycle":      mpl.cycler(color=COLOR_CYCLE),

        # Grid
        "grid.linewidth":       0.4,
        "grid.alpha":           0.5,

        # Lines
        "lines.linewidth":      1.2,
        "lines.markersize":     4,

        # Ticks
        "xtick.direction":      "in",
        "ytick.direction":      "in",
        "xtick.major.width":    0.6,
        "ytick.major.width":    0.6,
        "xtick.minor.visible":  False,
        "ytick.minor.visible":  False,

        # Legend
        "legend.frameon":       True,
        "legend.framealpha":    0.9,
        "legend.edgecolor":     "0.8",
        "legend.fancybox":      False,
        "legend.borderpad":     0.3,
        "legend.handlelength":  1.5,

        # LaTeX-compatible text
        "text.usetex":          False,
        "mathtext.fontset":     "cm",
    })


def save_fig(fig, name: str, formats=("pdf", "png")):
    """Save figure to Report/figures/plots/ in the specified formats."""
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    for fmt in formats:
        path = FIGURES_DIR / f"{name}.{fmt}"
        fig.savefig(path)
        print(f"  Saved: {path}")
    plt.close(fig)
