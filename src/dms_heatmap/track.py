"""SSDraw-style secondary-structure ribbons, coloured by fitness.

A track is one strip under a heatmap block, sharing its x axis position for
position: helices draw as a coiled ribbon, strands as an arrow pointing towards
the C terminus, loops as a thin bar, and a position the model does not cover as
a hairline.  The shape says where a residue is in the fold; the **colour** says
what mutating it does, taken from the same colormap and the same
:class:`~dms_heatmap.scale.ColourScale` as the heatmap above, so a red helix
turn means exactly what a red heatmap cell means.

The colour is painted as a one-row mesh across the whole strip and then clipped
to the outline of the shapes, which is what lets a single helix carry a
different colour on every residue instead of one colour per element.  The
alternative -- one polygon per residue -- cannot be done for a coil without
visible seams.

Drawing the ribbon here rather than shelling out to SSDraw is deliberate:
SSDraw renders its own figure with its own geometry, and these strips have to
line up cell-for-cell with a heatmap that wraps the protein into blocks of 100
positions.  Only a track that shares the heatmap's axes can do that.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from matplotlib.colors import ListedColormap
from matplotlib.path import Path as MplPath
from matplotlib.patches import PathPatch

from .palette import MISSING, NEUTRAL, SS_OUTLINE
from .structure import HELIX, LOOP, STRAND, UNMODELLED, Run, SecondaryStructure

__all__ = ["TrackStyle", "StructureTrack", "draw_track", "draw_shape_legend"]

#: Half-heights of each shape, as a fraction of the strip's half-height.  The
#: helix amplitude and ribbon half-thickness sum to the same 0.92, so no shape
#: touches the very edge of its strip and adjacent tracks never appear to merge.
_LOOP_HALF = 0.24
_STRAND_HALF = 0.34
_STRAND_HEAD_HALF = 0.82
_HELIX_AMPLITUDE = 0.50
_HELIX_HALF = 0.42
_UNMODELLED_HALF = 0.05

#: Residues per turn of the drawn coil.  This is the real pitch of an alpha
#: helix, so the turns a reader counts are the turns the protein has.
_HELIX_PERIOD = 3.6

#: Positions consumed by an arrowhead, capped at a fraction of the strand so a
#: three-residue strand is still drawn as an arrow and not as a bare triangle.
_HEAD_POSITIONS = 3.0
_HEAD_FRACTION = 0.55

#: Samples per position along a coil.  The ribbon is a polygon, and below about
#: eight the turns visibly flatten into straight segments at print resolution.
_COIL_SAMPLES = 14


@dataclass(frozen=True)
class TrackStyle:
    """Geometry of the secondary-structure strips, in inches."""

    height: float = 0.22
    gap: float = 0.08
    spacing: float = 0.06
    outline_width: float = 0.45
    label_size: float = 5.6

    def total_height(self, n_tracks: int) -> float:
        """Inches added below a heatmap block by ``n_tracks`` strips."""
        if n_tracks <= 0:
            return 0.0
        return (
            self.gap + n_tracks * self.height + (n_tracks - 1) * self.spacing
        )


@dataclass(frozen=True)
class StructureTrack:
    """One colouring of the secondary structure.

    ``values`` is indexed by 1-based position; positions it does not mention,
    or mentions as ``NaN``, draw as unmeasured grey inside the shape.  ``key``
    names it in the manifest and in the output table, ``label`` beside the
    strip itself.
    """

    key: str
    label: str
    values: pd.Series
    description: str = ""
    notes: tuple[str, ...] = field(default_factory=tuple)

    def as_dict(self) -> dict:
        finite = self.values.to_numpy(dtype=float)
        finite = finite[np.isfinite(finite)]
        return {
            "label": self.label,
            "description": self.description,
            "n_positions": int(finite.size),
            "median": round(float(np.median(finite)), 4) if finite.size else None,
            "min": round(float(finite.min()), 4) if finite.size else None,
            "max": round(float(finite.max()), 4) if finite.size else None,
            "notes": list(self.notes),
        }


# -- shape geometry --------------------------------------------------------


def _rectangle(x0: float, x1: float, half: float) -> MplPath:
    return MplPath(
        [(x0, -half), (x1, -half), (x1, half), (x0, half), (x0, -half)],
        [MplPath.MOVETO, *[MplPath.LINETO] * 3, MplPath.CLOSEPOLY],
    )


def _arrow(x0: float, x1: float, head: bool = True) -> MplPath:
    """A strand: a flat body that widens into a head pointing at ``x1``.

    ``head=False`` draws the body alone, for a strand that runs off the end of
    a block and continues on the next one -- an arrowhead there would say the
    strand ends at the fold, which is the one thing it does not do.
    """
    if not head:
        return _rectangle(x0, x1, _STRAND_HALF)
    length = min(_HEAD_POSITIONS, (x1 - x0) * _HEAD_FRACTION)
    neck = x1 - length
    points = [
        (x0, -_STRAND_HALF),
        (neck, -_STRAND_HALF),
        (neck, -_STRAND_HEAD_HALF),
        (x1, 0.0),
        (neck, _STRAND_HEAD_HALF),
        (neck, _STRAND_HALF),
        (x0, _STRAND_HALF),
        (x0, -_STRAND_HALF),
    ]
    codes = [MplPath.MOVETO, *[MplPath.LINETO] * 6, MplPath.CLOSEPOLY]
    return MplPath(points, codes)


def _coil(x0: float, x1: float, phase_from: float | None = None) -> MplPath:
    """A helix: a ribbon of constant thickness following a sine midline.

    The midline crosses the centre of the strip at ``phase_from`` -- where the
    helix began, which is before ``x0`` for a helix continued from the block
    above -- so a helix wrapped across blocks reads as one helix rather than as
    two that both happen to start mid-turn.
    """
    origin = x0 if phase_from is None else phase_from
    n = max(int(round((x1 - x0) * _COIL_SAMPLES)), 8)
    xs = np.linspace(x0, x1, n + 1)
    mid = _HELIX_AMPLITUDE * np.sin(2 * np.pi * (xs - origin) / _HELIX_PERIOD)

    upper = list(zip(xs, mid + _HELIX_HALF))
    lower = list(zip(xs[::-1], (mid - _HELIX_HALF)[::-1]))
    points = [*upper, *lower, upper[0]]
    codes = [MplPath.MOVETO, *[MplPath.LINETO] * (len(points) - 2), MplPath.CLOSEPOLY]
    return MplPath(points, codes)


def _shape(run: Run) -> MplPath:
    """The outline of one run, spanning the full width of its positions."""
    x0, x1 = run.start - 0.5, run.end + 0.5
    if run.code == HELIX:
        return _coil(x0, x1, phase_from=run.origin - 0.5)
    if run.code == STRAND:
        return _arrow(x0, x1, head=not run.open_end)
    if run.code == UNMODELLED:
        return _rectangle(x0, x1, _UNMODELLED_HALF)
    return _rectangle(x0, x1, _LOOP_HALF)


def _compound(paths: list[MplPath]) -> MplPath | None:
    """Several closed outlines as one path, for use as a clip region."""
    if not paths:
        return None
    return MplPath(
        np.concatenate([p.vertices for p in paths]),
        np.concatenate([p.codes for p in paths]),
    )


# -- drawing ---------------------------------------------------------------


def draw_track(
    ax,
    structure: SecondaryStructure,
    track: StructureTrack,
    start: int,
    end: int,
    block_size: int,
    cmap,
    norm,
    style: TrackStyle | None = None,
) -> None:
    """Draw one coloured secondary-structure strip for positions ``start..end``.

    ``block_size`` sets the x limit rather than ``end``, so a short final block
    keeps the same scale as every other block on the figure.
    """
    style = style or TrackStyle()
    runs = structure.runs(start, end)
    shapes = [(run, _shape(run)) for run in runs]
    structured = [path for run, path in shapes if run.code != UNMODELLED]

    # A position outside the model gets a hairline rather than a loop: the
    # model not covering a residue is not an observation that it is disordered.
    for run, path in shapes:
        if run.code == UNMODELLED:
            ax.add_patch(
                PathPatch(path, facecolor=MISSING, edgecolor="none", zorder=1.5)
            )

    clip = _compound(structured)
    if clip is not None:
        values = track.values.reindex(range(start, end + 1)).to_numpy(dtype=float)
        x_edges = np.arange(start - 0.5, end + 0.5 + 1e-9)
        y_edges = np.array([-1.0, 1.0])

        # Unmeasured positions are painted, not left transparent, so a residue
        # with no data reads as grey inside the shape instead of as background.
        base = ax.pcolormesh(
            x_edges,
            y_edges,
            np.zeros((1, values.size)),
            cmap=ListedColormap([MISSING]),
            vmin=0,
            vmax=1,
            shading="flat",
            zorder=2,
        )
        mesh = ax.pcolormesh(
            x_edges,
            y_edges,
            np.ma.masked_invalid(values)[None, :],
            cmap=cmap,
            norm=norm,
            shading="flat",
            zorder=3,
        )
        for artist in (base, mesh):
            artist.set_clip_path(clip, ax.transData)

    for run, path in shapes:
        if run.code != UNMODELLED:
            ax.add_patch(
                PathPatch(
                    path,
                    facecolor="none",
                    edgecolor=SS_OUTLINE,
                    linewidth=style.outline_width,
                    joinstyle="round",
                    zorder=4,
                )
            )

    ax.set_xlim(start - 0.5, start - 0.5 + block_size)
    ax.set_ylim(-1.05, 1.05)
    ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_visible(False)
    ax.set_ylabel(
        track.label,
        fontsize=style.label_size,
        color=SS_OUTLINE,
        rotation=0,
        ha="right",
        va="center",
        labelpad=4,
    )


def draw_shape_legend(ax, style: TrackStyle | None = None) -> None:
    """Key to the four shapes, drawn with the same geometry as the tracks.

    Filled with the ramp's neutral rather than a fitness colour: this legend
    explains the shapes, and the colourbar beside it explains the colours.
    """
    style = style or TrackStyle()
    entries = (
        (Run(HELIX, 1, 11), "helix"),
        (Run(STRAND, 23, 33), "strand"),
        (Run(LOOP, 47, 55), "loop"),
        (Run(UNMODELLED, 69, 77), "not in model"),
    )
    for run, label in entries:
        path = _shape(run)
        ax.add_patch(
            PathPatch(
                path,
                facecolor=MISSING if run.code == UNMODELLED else NEUTRAL,
                edgecolor="none" if run.code == UNMODELLED else SS_OUTLINE,
                linewidth=style.outline_width,
                joinstyle="round",
            )
        )
        ax.text(
            run.end + 2.0,
            0.0,
            label,
            fontsize=style.label_size,
            color=SS_OUTLINE,
            va="center",
            ha="left",
        )

    ax.set_xlim(0, 100)
    ax.set_ylim(-1.05, 1.05)
    ax.set_xticks([])
    ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_visible(False)
