"""Fitness distributions of the three variant classes.

The heatmap shows where in the protein the effects fall; this shows how the
whole assay is distributed, which is the plot that tells you whether the
normalisation worked at all.  Synonymous variants should pile up on 1 and
nonsense variants on 0, by construction -- if they do not, the anchors are
wrong.  Missense variants then sit between the two modes, and the shape of
that distribution (one mode or two, how much mass sits at the null peak) is
the headline result of the screen.

All three classes are drawn on one shared grid, so their heights are
comparable across classes and not just within one.
"""

from __future__ import annotations

import textwrap
from dataclasses import dataclass

import matplotlib

matplotlib.use("Agg")

import numpy as np
import pandas as pd
from matplotlib.figure import Figure
from matplotlib.lines import Line2D

from .normalise import missense_mask, nonsense_mask, wild_type_mask
from .palette import CLASS_COLOURS, CLASS_LINESTYLES
from .plot import SURFACE, TEXT_MUTED, TEXT_PRIMARY, TEXT_SECONDARY
from .scale import STOP_FITNESS, WT_FITNESS

__all__ = [
    "ClassSeries",
    "DistributionStyle",
    "class_series",
    "fine_grid",
    "render_distribution",
    "smooth_counts",
]

#: Draw and label the classes in this order: the largest class first, so the
#: smaller ones are drawn over it rather than hidden behind it.
CLASS_ORDER = ("missense", "synonymous", "nonsense")

CLASS_LABELS = {
    "missense": "missense",
    "synonymous": "synonymous (wild-type protein)",
    "nonsense": "nonsense (stop)",
}


@dataclass(frozen=True)
class ClassSeries:
    """One variant class's fitness values, plus the summary shown in the key."""

    name: str
    values: np.ndarray

    @property
    def label(self) -> str:
        return CLASS_LABELS[self.name]

    @property
    def colour(self) -> str:
        return CLASS_COLOURS[self.name]

    @property
    def linestyle(self):
        return CLASS_LINESTYLES[self.name]

    @property
    def n(self) -> int:
        return int(self.values.size)

    @property
    def median(self) -> float:
        return float(np.median(self.values)) if self.n else float("nan")

    def as_dict(self) -> dict:
        if not self.n:
            return {"n": 0}
        return {
            "n": self.n,
            "median": self.median,
            "mean": float(self.values.mean()),
            "sd": float(self.values.std(ddof=1)) if self.n > 1 else float("nan"),
            "q0.05": float(np.quantile(self.values, 0.05)),
            "q0.95": float(np.quantile(self.values, 0.95)),
        }


class DistributionStyle:
    """Geometry and typography of the distribution figure, in inches and points."""

    def __init__(
        self,
        bins: int = 60,
        layout: str = "overlay",
        trim: float = 0.005,
        smoothing: float = 1.0,
        width: float = 7.4,
        panel_height: float = 3.2,
    ) -> None:
        if layout not in {"overlay", "facet"}:
            raise ValueError(f"unknown layout {layout!r}; choose 'overlay' or 'facet'")
        if bins < 2:
            raise ValueError("bins must be at least 2")
        if not 0.0 <= trim < 0.5:
            raise ValueError("trim must be in [0, 0.5)")
        if smoothing < 0:
            raise ValueError("smoothing must be 0 (step histogram) or positive")
        self.bins = bins
        self.layout = layout
        self.trim = trim
        self.smoothing = smoothing
        self.width = width
        self.panel_height = panel_height

        self.margin_left = 0.78
        self.margin_right = 0.28
        self.margin_bottom = 0.92
        # In overlay mode the key goes above the axes rather than into a
        # corner: these distributions put mass across the whole range, so
        # there is no reliably empty corner, and a key sitting on the missense
        # curve hides the very shape it is labelling.  Faceted panels label
        # themselves, so they need no reserved strip.
        self.legend_height = 0.62 if layout == "overlay" else 0.0
        self.margin_top = 0.86 + self.legend_height
        self.panel_gap = 0.28
        self.font_title = 13.0
        self.font_label = 8.5
        self.font_tick = 7.5
        self.font_note = 6.8
        self.fill_alpha = 0.22
        self.line_width = 1.4


def class_series(table: pd.DataFrame, value_column: str = "fitness") -> list[ClassSeries]:
    """Split a normalised table into the three variant classes.

    Classes are mutually exclusive: ``synonymous`` is ``aa_ham == 0`` (the
    protein is wild type), ``nonsense`` is a stop substitution, and
    ``missense`` is everything else with a single amino-acid change.  Variants
    with no fitness value are dropped here rather than plotted as zero.
    """
    masks = {
        "missense": missense_mask(table),
        "synonymous": wild_type_mask(table),
        "nonsense": nonsense_mask(table),
    }
    values = pd.to_numeric(table[value_column], errors="coerce")
    return [
        ClassSeries(name, values[masks[name]].dropna().to_numpy(dtype=float))
        for name in CLASS_ORDER
    ]


def _bandwidth(values: np.ndarray, scale: float) -> float:
    """Silverman's rule of thumb, robustified with the interquartile range.

    The spread estimate is the smaller of the standard deviation and
    IQR/1.34, so a long tail of noisy outliers cannot inflate the bandwidth
    and smear out the real modes.
    """
    n = values.size
    if n < 2:
        return 0.0
    spread = float(values.std(ddof=1))
    iqr = float(np.subtract(*np.percentile(values, [75, 25])))
    if iqr > 0:
        spread = min(spread, iqr / 1.34)
    if spread <= 0:
        return 0.0
    return scale * 1.06 * spread * n ** (-1 / 5)


def fine_grid(low: float, high: float, bandwidths) -> np.ndarray:
    """Fine-bin edges shared by every class on one pair of axes.

    One grid for all classes, stepped for the *narrowest* bandwidth so no
    class is under-resolved and padded by the *widest* so none is truncated
    at the axis limits.  Sharing it means the curves are sampled at the same
    x values, which keeps them directly comparable point for point and keeps
    the vector output tidy.
    """
    usable = [b for b in bandwidths if b > 0]
    if not usable or high <= low:
        return np.array([])
    fine_width = min(min(usable) / 8.0, (high - low) / 512)
    pad = 4 * max(usable)
    return np.arange(low - pad, high + pad + fine_width, fine_width)


def smooth_counts(
    values: np.ndarray,
    low: float,
    high: float,
    bin_width: float,
    bandwidth: float,
    edges: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """A Gaussian kernel density for ``values``, expressed as counts per bin.

    The y units are deliberately the same as the step histogram's: the curve
    integrates to the number of variants, and its height at *x* is how many
    variants fall in a ``bin_width``-wide window there.  So the axis still
    answers "how many variants are around here", just without the chunking.

    Evaluated by binning finely and convolving, which is exact enough well
    below the bandwidth and does not build an n-by-grid matrix.  ``edges``
    is the fine grid to evaluate on, shared across classes by the caller;
    without one, a grid for this bandwidth alone is used.
    """
    if values.size == 0 or bandwidth <= 0 or high <= low:
        return np.array([]), np.array([])

    if edges is None:
        edges = fine_grid(low, high, [bandwidth])
    if edges.size < 3:
        return np.array([]), np.array([])
    fine_width = float(edges[1] - edges[0])

    counts, _ = np.histogram(values, bins=edges)
    centres = 0.5 * (edges[:-1] + edges[1:])

    reach = max(1, int(np.ceil(4 * bandwidth / fine_width)))
    offsets = np.arange(-reach, reach + 1) * fine_width
    kernel = np.exp(-0.5 * (offsets / bandwidth) ** 2)
    kernel /= kernel.sum()

    smoothed = np.convolve(counts.astype(float), kernel, mode="same")
    inside = (centres >= low) & (centres <= high)
    return centres[inside], smoothed[inside] * (bin_width / fine_width)


def _bin_edges(series: list[ClassSeries], style: DistributionStyle):
    """Shared bin edges over a trimmed range, and the values clipped into it.

    A handful of wild outliers would otherwise stretch the axis until the
    three modes collapse into one bar.  Clipping rather than discarding keeps
    every variant counted -- the outermost bins hold the overflow, and the
    caption says how many landed there.
    """
    arrays = [s.values for s in series if s.n]
    pooled = np.concatenate(arrays) if arrays else np.array([], dtype=float)
    if pooled.size == 0:
        raise ValueError("no finite fitness values to plot a distribution from")

    low = float(np.quantile(pooled, style.trim))
    high = float(np.quantile(pooled, 1 - style.trim))
    if high <= low:
        low, high = float(pooled.min()), float(pooled.max())
    if high <= low:  # a single repeated value
        low, high = low - 0.5, high + 0.5

    edges = np.linspace(low, high, style.bins + 1)
    clipped = int(sum(int(np.sum((s.values < low) | (s.values > high))) for s in series))
    return edges, clipped


def render_distribution(
    table: pd.DataFrame,
    dataset: str,
    style: DistributionStyle | None = None,
    value_column: str = "fitness",
    title: str | None = None,
    subtitle: str | None = None,
) -> tuple[Figure, list[ClassSeries]]:
    """Build the distribution figure and return it with the per-class series."""
    style = style or DistributionStyle()
    series = class_series(table, value_column)
    edges, clipped = _bin_edges(series, style)

    drawn = [s for s in series if s.n]
    panels = len(drawn) if style.layout == "facet" else 1
    fig_height = (
        style.margin_top
        + panels * style.panel_height
        + (panels - 1) * style.panel_gap
        + style.margin_bottom
    )
    fig = Figure(figsize=(style.width, fig_height), dpi=300, facecolor=SURFACE)

    axes = []
    for index in range(panels):
        bottom = (
            style.margin_bottom
            + (panels - 1 - index) * (style.panel_height + style.panel_gap)
        )
        ax = fig.add_axes(
            (
                style.margin_left / style.width,
                bottom / fig_height,
                (style.width - style.margin_left - style.margin_right) / style.width,
                style.panel_height / fig_height,
            )
        )
        axes.append(ax)

    # Bandwidths first, so every class can be drawn on one shared fine grid.
    bandwidths = (
        {item.name: _bandwidth(item.values, style.smoothing) for item in drawn}
        if style.smoothing > 0
        else {}
    )
    grid = fine_grid(edges[0], edges[-1], bandwidths.values()) if bandwidths else None

    for index, item in enumerate(drawn):
        ax = axes[index if style.layout == "facet" else 0]
        if style.smoothing > 0:
            _draw_smooth(ax, item, edges, style, bandwidths[item.name], grid)
        else:
            _draw_steps(ax, item, edges, style)
        if style.layout == "facet":
            ax.text(
                0.995, 0.92,
                f"{item.label}  (n = {item.n:,})",
                transform=ax.transAxes, ha="right", va="top",
                fontsize=style.font_label, color=item.colour,
            )

    for index, ax in enumerate(axes):
        _dress_axes(ax, edges, style, is_last=index == len(axes) - 1, panels=panels)

    if style.layout == "overlay":
        _add_legend(axes[0], drawn, style)

    _add_titles(fig, dataset, title, subtitle, style, fig_height)
    _add_note(fig, edges, clipped, bandwidths, style, fig_height)
    return fig, series


def _draw_smooth(
    ax, item: ClassSeries, edges, style: DistributionStyle, bandwidth: float, grid
) -> None:
    """Draw one class as a smooth density curve."""
    bin_width = edges[1] - edges[0]
    x, y = smooth_counts(
        item.values, edges[0], edges[-1], bin_width, bandwidth, edges=grid
    )
    if x.size == 0:
        # Too few points, or no spread, to smooth: fall back rather than
        # silently leaving the class off the figure.
        _draw_steps(ax, item, edges, style)
        return
    ax.fill_between(x, y, color=item.colour, alpha=style.fill_alpha, linewidth=0)
    ax.plot(x, y, color=item.colour, linewidth=style.line_width,
            linestyle=item.linestyle, zorder=3)


def _draw_steps(ax, item: ClassSeries, edges, style: DistributionStyle) -> None:
    """Draw one class as a step histogram, with the tails in the end bins."""
    counts, _ = np.histogram(np.clip(item.values, edges[0], edges[-1]), bins=edges)
    ax.stairs(counts, edges, fill=True, color=item.colour, alpha=style.fill_alpha,
              linewidth=0)
    ax.stairs(counts, edges, color=item.colour, linewidth=style.line_width,
              linestyle=item.linestyle, zorder=3)


def _dress_axes(ax, edges, style: DistributionStyle, is_last: bool, panels: int) -> None:
    ax.set_facecolor(SURFACE)
    ax.set_xlim(edges[0], edges[-1])

    # The two anchors are the reference the whole scale is built on, so they
    # are marked on the axis rather than left to the reader's arithmetic.
    for anchor, name in ((STOP_FITNESS, "nonsense"), (WT_FITNESS, "wild type")):
        if edges[0] < anchor < edges[-1]:
            ax.axvline(anchor, color=TEXT_MUTED, linewidth=0.7, linestyle=(0, (3, 2)),
                       zorder=1)
            if not is_last or panels == 1:
                ax.annotate(
                    name,
                    xy=(anchor, 1.0), xycoords=("data", "axes fraction"),
                    xytext=(2, -2), textcoords="offset points",
                    fontsize=style.font_note, color=TEXT_MUTED, ha="left", va="top",
                )

    ax.tick_params(labelsize=style.font_tick, length=2.4, width=0.5, colors=TEXT_SECONDARY)
    ax.set_ylabel("variants per bin", fontsize=style.font_label, color=TEXT_SECONDARY)
    ax.grid(axis="y", color="#e8e7e4", linewidth=0.5, zorder=0)
    ax.set_axisbelow(True)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color("#d8d7d2")
        ax.spines[side].set_linewidth(0.6)

    if is_last:
        ax.set_xlabel(
            "normalised fitness  (0 = nonsense, 1 = wild type)",
            fontsize=style.font_label, color=TEXT_SECONDARY,
        )
    else:
        ax.set_xticklabels([])


def _add_legend(ax, drawn: list[ClassSeries], style: DistributionStyle) -> None:
    handles = [
        Line2D([], [], color=s.colour, linewidth=style.line_width, linestyle=s.linestyle,
               label=f"{s.label}   n = {s.n:,}   median {s.median:.2f}")
        for s in drawn
    ]
    legend = ax.legend(
        handles=handles,
        loc="lower left", bbox_to_anchor=(0.0, 1.012), borderaxespad=0.0,
        frameon=False, fontsize=style.font_label, handlelength=2.4,
        labelspacing=0.42,
    )
    for text in legend.get_texts():
        text.set_color(TEXT_SECONDARY)


def _add_titles(fig, dataset, title, subtitle, style, fig_height) -> None:
    x = style.margin_left / style.width
    fig.text(
        x, 1 - 0.30 / fig_height,
        title or f"{dataset} - fitness distribution by variant class",
        fontsize=style.font_title, color=TEXT_PRIMARY, va="top", ha="left",
        fontweight="bold",
    )
    if subtitle:
        fig.text(
            x, 1 - 0.54 / fig_height, subtitle,
            fontsize=style.font_label, color=TEXT_SECONDARY, va="top", ha="left",
        )


def _add_note(fig, edges, clipped: int, bandwidths: dict, style, fig_height) -> None:
    width = edges[1] - edges[0]
    if style.smoothing > 0:
        spread = (
            f"{min(bandwidths.values()):.3f} to {max(bandwidths.values()):.3f}"
            if len(set(np.round(list(bandwidths.values()), 3))) > 1
            else f"{next(iter(bandwidths.values()), 0.0):.3f}"
        )
        parts = [
            f"Gaussian kernel density scaled to variants per {width:.3f} fitness "
            f"units; bandwidth {spread} per class (Silverman).",
        ]
        if clipped:
            parts.append(
                f"{clipped:,} variant(s) fall outside {edges[0]:.2f} to "
                f"{edges[-1]:.2f} and are off the axis."
            )
    else:
        parts = [f"{len(edges) - 1} bins of {width:.3f} normalised fitness units."]
        if clipped:
            parts.append(
                f"{clipped:,} variant(s) outside {edges[0]:.2f} to {edges[-1]:.2f} "
                "are counted in the end bins."
            )
    # Wrap on the available width: the note grows with the number of classes
    # that needed their own bandwidth.
    usable = style.width - style.margin_left - style.margin_right
    per_char = 0.56 * style.font_note / 72
    lines = textwrap.wrap("  ".join(parts), width=max(40, int(usable / per_char)))
    for index, line in enumerate(reversed(lines[:2])):
        fig.text(
            style.margin_left / style.width,
            (0.08 + index * 0.13) / fig_height,
            line,
            fontsize=style.font_note, color=TEXT_MUTED, va="bottom", ha="left",
        )
