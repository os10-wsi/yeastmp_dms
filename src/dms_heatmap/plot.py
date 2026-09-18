"""Render a residue x position fitness heatmap.

A 500-residue protein is far too wide for one strip of cells, so the sequence
is wrapped into stacked blocks of fixed width.  Cell size is fixed rather than
the figure size, which keeps every dataset in a run rendering at the same
scale no matter how long the protein is.
"""

from __future__ import annotations

import math
import textwrap
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import ListedColormap
from matplotlib.figure import Figure

from .matrix import FitnessMatrix
from .palette import MISSING, WT_MARK, fitness_cmap
from .scale import ColourScale
from .structure import SecondaryStructure
from .track import StructureTrack, TrackStyle, draw_shape_legend, draw_track

__all__ = ["HeatmapStyle", "render_heatmap", "save_figure"]

SURFACE = "#fcfcfb"
TEXT_PRIMARY = "#0b0b0b"
TEXT_SECONDARY = "#52514e"
TEXT_MUTED = "#75736d"


class HeatmapStyle:
    """Geometry and typography of the figure, in inches and points."""

    def __init__(
        self,
        block_size: int = 100,
        cell_width: float = 0.10,
        cell_height: float = 0.125,
        tick_every: int = 10,
        mark_wild_type: bool = True,
        cell_edge_width: float = 0.2,
        tracks: TrackStyle | None = None,
    ) -> None:
        if block_size < 1:
            raise ValueError("block_size must be at least 1")
        self.block_size = block_size
        self.cell_width = cell_width
        self.cell_height = cell_height
        self.tick_every = tick_every
        self.mark_wild_type = mark_wild_type
        self.cell_edge_width = cell_edge_width
        self.tracks = tracks or TrackStyle()

        # Fixed page furniture.
        self.margin_left = 0.55
        self.margin_right = 0.30
        self.margin_top = 0.95
        self.margin_bottom = 1.20
        self.block_gap = 0.42
        # Extra bottom margin when the shape legend needs a row of its own,
        # below the colourbar and its label.
        self.legend_row = 0.34
        self.font_tick = 5.2
        self.font_residue = 5.2
        self.font_title = 13.0
        self.font_label = 7.5
        self.footer_line = 0.105


def _block_ranges(n_positions: int, block_size: int) -> list[tuple[int, int]]:
    """1-based inclusive (start, end) position ranges, one per stacked block."""
    n_blocks = max(1, math.ceil(n_positions / block_size))
    return [
        (i * block_size + 1, min((i + 1) * block_size, n_positions))
        for i in range(n_blocks)
    ]


def render_heatmap(
    matrix: FitnessMatrix,
    scale: ColourScale,
    style: HeatmapStyle | None = None,
    title: str | None = None,
    subtitle: str | None = None,
    structure: SecondaryStructure | None = None,
    tracks: tuple[StructureTrack, ...] = (),
) -> Figure:
    """Build the heatmap figure for one dataset.

    When a ``structure`` is given, each ``tracks`` entry is drawn as a
    secondary-structure strip below that block of the heatmap, sharing its x
    axis and its colour scale.  The position axis then belongs to the lowest
    strip rather than to the heatmap, so nothing is labelled twice.
    """
    style = style or HeatmapStyle()
    cmap = fitness_cmap()
    norm = scale.norm()
    tracks = tuple(tracks) if structure is not None else ()

    grid = matrix.values
    residues = matrix.residues
    n_rows = len(residues)
    positions = matrix.positions
    n_positions = int(positions.max()) if positions.size else 0
    blocks = _block_ranges(n_positions, style.block_size)

    axes_width = style.block_size * style.cell_width
    axes_height = n_rows * style.cell_height
    strip_height = style.tracks.total_height(len(tracks))
    block_height = axes_height + strip_height

    fig_width = style.margin_left + axes_width + style.margin_right
    usable = fig_width - style.margin_left - style.margin_right

    # The bottom margin has to be sized before the figure exists, because the
    # caption grows with the strips and the shape legend needs a row of its
    # own below the colourbar.  Everything in that margin then shifts up by
    # the same amount rather than being laid on top of something else.
    footer = _wrap(
        _footer_text(matrix, scale, tracks, structure), usable, style.font_tick + 0.4
    )
    # What the caption costs over the one line the margin already allows for,
    # and what the legend row costs over that.  The colourbar clears both; the
    # legend only has to clear the caption.
    over_caption = (len(footer) - 1) * style.footer_line
    over_legend = over_caption + (style.legend_row if tracks else 0.0)
    margin_bottom = style.margin_bottom + over_legend

    fig_height = (
        style.margin_top
        + len(blocks) * block_height
        + (len(blocks) - 1) * style.block_gap
        + margin_bottom
    )

    fig = Figure(figsize=(fig_width, fig_height), dpi=300, facecolor=SURFACE)
    values = np.ma.masked_invalid(grid.to_numpy(dtype=float))
    mesh = None

    def place(bottom_in: float, height_in: float):
        ax = fig.add_axes(
            (
                style.margin_left / fig_width,
                bottom_in / fig_height,
                axes_width / fig_width,
                height_in / fig_height,
            )
        )
        ax.set_facecolor(SURFACE)
        return ax

    for index, (start, end) in enumerate(blocks):
        last_block = index == len(blocks) - 1
        group_bottom = (
            margin_bottom
            + (len(blocks) - 1 - index) * (block_height + style.block_gap)
        )
        ax = place(group_bottom + strip_height, axes_height)

        columns = np.arange(start, end + 1)
        block = values[:, start - 1 : end]
        x_edges = np.arange(start - 0.5, end + 0.5 + 1e-9)
        y_edges = np.arange(n_rows + 1)

        # Unmeasured cells are painted explicitly rather than left to show the
        # axes background, so that the padding after the final block -- which
        # is off the end of the protein, not missing data -- stays blank.
        ax.pcolormesh(
            x_edges,
            y_edges,
            np.zeros_like(block, dtype=float),
            cmap=ListedColormap([MISSING]),
            vmin=0,
            vmax=1,
            edgecolors=SURFACE,
            linewidth=style.cell_edge_width,
            shading="flat",
            zorder=1,
        )
        mesh = ax.pcolormesh(
            x_edges,
            y_edges,
            block,
            cmap=cmap,
            norm=norm,
            edgecolors=SURFACE,
            linewidth=style.cell_edge_width,
            shading="flat",
            zorder=2,
        )

        if style.mark_wild_type:
            _mark_wild_type(ax, matrix, residues, columns)

        ax.set_xlim(start - 0.5, start - 0.5 + style.block_size)
        ax.set_ylim(n_rows, 0)
        ax.set_yticks(np.arange(n_rows) + 0.5)
        ax.set_yticklabels(residues, fontsize=style.font_residue, color=TEXT_SECONDARY)
        for spine in ax.spines.values():
            spine.set_visible(False)
        ax.set_ylabel(
            "substitution", fontsize=style.font_label, color=TEXT_SECONDARY, labelpad=3
        )
        # The position axis belongs to whatever sits at the bottom of the
        # block: labelling it on the heatmap too would put numbers between the
        # cells and the structure they annotate.
        _position_axis(ax, columns, style, labelled=not tracks)

        bottom_axes = ax
        for order, track in enumerate(tracks):
            bottom = group_bottom + (len(tracks) - 1 - order) * (
                style.tracks.height + style.tracks.spacing
            )
            bottom_axes = place(bottom, style.tracks.height)
            draw_track(
                bottom_axes,
                structure,
                track,
                start,
                end,
                style.block_size,
                cmap,
                norm,
                style=style.tracks,
            )
            _position_axis(
                bottom_axes, columns, style, labelled=order == len(tracks) - 1
            )

        if last_block:
            # Left-aligned: the final block is usually short, so a centred
            # label would float out over the empty end of the axes.
            bottom_axes.set_xlabel(
                "residue position",
                fontsize=style.font_label,
                color=TEXT_SECONDARY,
                labelpad=2,
                loc="left",
            )

    _add_colourbar(fig, mesh, scale, style, fig_width, fig_height, over_legend)
    if tracks:
        _add_shape_legend(fig, style, fig_width, fig_height, over_caption)
    _add_titles(fig, matrix, title, subtitle, style, fig_width, fig_height)
    _add_footer(fig, footer, style, fig_width, fig_height)
    return fig


def _position_axis(ax, columns, style: HeatmapStyle, labelled: bool) -> None:
    """Put the residue-position ticks on one axes of a block."""
    ticks = [p for p in columns if p % style.tick_every == 0 or p == 1]
    ax.set_xticks(ticks)
    ax.set_xticklabels(
        [str(t) for t in ticks] if labelled else [],
        fontsize=style.font_tick,
        color=TEXT_SECONDARY,
    )
    ax.tick_params(
        length=1.6 if labelled else 0.0,
        width=0.4,
        pad=1.5,
        colors=TEXT_SECONDARY,
        labelbottom=labelled,
    )


def _mark_wild_type(ax, matrix: FitnessMatrix, residues: list[str], columns) -> None:
    """Dot the wild-type residue at each position, so the sequence is readable."""
    row_of = {aa: i for i, aa in enumerate(residues)}
    xs, ys = [], []
    for pos in columns:
        aa = matrix.wt_residues.get(pos)
        if aa is None or aa is np.nan or str(aa) == "<NA>":
            continue
        row = row_of.get(str(aa))
        if row is not None:
            xs.append(pos)
            ys.append(row + 0.5)
    if xs:
        ax.scatter(xs, ys, s=0.9, color=WT_MARK, marker="o", linewidths=0, zorder=4)


def _add_colourbar(fig, mesh, scale, style, fig_width, fig_height, raised=0.0) -> None:
    if mesh is None:
        return
    width = min(2.6, 0.55 * fig_width)
    bottom = 0.62 + raised
    cax = fig.add_axes(
        (
            style.margin_left / fig_width,
            bottom / fig_height,
            width / fig_width,
            0.12 / fig_height,
        )
    )
    bar = fig.colorbar(mesh, cax=cax, orientation="horizontal", extend="both")
    bar.set_ticks(scale.colourbar_ticks())
    bar.ax.tick_params(labelsize=style.font_tick, length=1.6, width=0.4, pad=1.6)
    bar.outline.set_visible(False)
    bar.set_label(
        "normalised fitness  (0 = nonsense, 1 = wild type)",
        fontsize=style.font_label,
        color=TEXT_SECONDARY,
        labelpad=3,
    )
    for label in bar.ax.get_xticklabels():
        label.set_color(TEXT_SECONDARY)

    # The colourbar cannot show grey, so the "not measured" state gets its own
    # swatch -- otherwise the only unlabelled colour on the figure is the one
    # that means "no data".
    swatch_left = style.margin_left + width + 0.42
    sax = fig.add_axes(
        (
            swatch_left / fig_width,
            bottom / fig_height,
            0.12 / fig_width,
            0.12 / fig_height,
        )
    )
    sax.set_facecolor(MISSING)
    sax.set_xticks([])
    sax.set_yticks([])
    for spine in sax.spines.values():
        spine.set_visible(False)
    fig.text(
        (swatch_left + 0.20) / fig_width,
        (bottom + 0.06) / fig_height,
        "not measured",
        fontsize=style.font_label,
        color=TEXT_SECONDARY,
        va="center",
        ha="left",
    )


def _add_shape_legend(fig, style, fig_width, fig_height, raised=0.0) -> None:
    """Key to the secondary-structure shapes, on its own row under the colourbar.

    Given a fixed width rather than a share of the figure: the glyphs and their
    labels are drawn at a set size, so stretching the axes with the page would
    only spread them apart.
    """
    width = min(2.6, fig_width - style.margin_left - style.margin_right)
    ax = fig.add_axes(
        (
            style.margin_left / fig_width,
            (0.30 + raised) / fig_height,
            width / fig_width,
            0.18 / fig_height,
        )
    )
    ax.set_facecolor(SURFACE)
    draw_shape_legend(ax, style.tracks)


def _add_titles(fig, matrix, title, subtitle, style, fig_width, fig_height) -> None:
    x = style.margin_left / fig_width
    fig.text(
        x,
        1 - 0.34 / fig_height,
        title or matrix.dataset,
        fontsize=style.font_title,
        color=TEXT_PRIMARY,
        va="top",
        ha="left",
        fontweight="bold",
    )
    if not subtitle:
        return
    # Wrap on the available width rather than trusting the caller to keep the
    # subtitle short: it grows with every filter that gets switched on.
    usable = fig_width - style.margin_left - style.margin_right
    lines = _wrap(subtitle, usable, style.font_label)
    for index, line in enumerate(lines[:2]):
        fig.text(
            x,
            1 - (0.56 + index * 0.16) / fig_height,
            line,
            fontsize=style.font_label,
            color=TEXT_SECONDARY,
            va="top",
            ha="left",
        )


def _wrap(text: str, usable: float, fontsize: float) -> list[str]:
    """Wrap ``text`` to the width available on the page, in inches.

    Estimated from the point size rather than measured: the figure does not
    exist yet when the caption has to be sized, because its height depends on
    how many lines the caption takes.
    """
    per_char = 0.56 * fontsize / 72
    return textwrap.wrap(text, width=max(40, int(usable / per_char))) or [""]


def _footer_text(matrix, scale, tracks=(), structure=None) -> str:
    """The caption: how to read the colours, and what the strips are."""
    tail_low = f"{scale.lower_quantile:.0%}"
    tail_high = f"{1 - scale.upper_quantile:.0%}"
    parts = [
        f"Colour saturates below the {tail_low} and above the top {tail_high} of "
        f"measured values ({scale.vmin:.2f} / {scale.vmax:.2f}).",
        "A dot marks the wild-type residue at each position.",
        f"Coverage {matrix.coverage:.0%} of possible substitutions.",
    ]
    if tracks and structure is not None:
        source = structure.source.name if structure.source else "as supplied"
        parts.append(
            f"Secondary structure ({source}) is drawn on the same colour scale as "
            "the cells: "
            + "; ".join(f"{t.label} is {t.description}" for t in tracks if t.description)
            + "."
        )
        # Without this the mean strip reads as a milder result than it is: it
        # is a mean of the cells above it, and the limits are percentiles of
        # those cells, so it cannot reach the ends of the ramp as often.
        parts.append(
            "A position mean spans a narrower range than a single substitution, "
            "so that strip saturates less at the same limits."
        )
    return "  ".join(parts)


def _add_footer(fig, lines: list[str], style, fig_width, fig_height) -> None:
    for index, line in enumerate(reversed(lines)):
        fig.text(
            style.margin_left / fig_width,
            (0.10 + index * style.footer_line) / fig_height,
            line,
            fontsize=style.font_tick + 0.4,
            color=TEXT_MUTED,
            va="bottom",
            ha="left",
        )


def save_figure(fig: Figure, stem: Path, formats=("png", "pdf")) -> list[Path]:
    """Write the figure to ``stem.<ext>`` for each requested format.

    The extension is appended rather than substituted: stems here contain
    dots (``tna1.distribution``), and ``with_suffix`` would replace that last
    component instead of adding to it.
    """
    written: list[Path] = []
    stem = Path(stem)
    stem.parent.mkdir(parents=True, exist_ok=True)
    for extension in formats:
        out = stem.with_name(f"{stem.name}.{extension}")
        fig.savefig(out, facecolor=fig.get_facecolor())
        written.append(out)
    plt.close(fig)
    return written
