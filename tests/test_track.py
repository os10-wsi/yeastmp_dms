"""The drawn secondary-structure strips: shape geometry and colouring."""

from __future__ import annotations

import matplotlib

matplotlib.use("Agg")

import numpy as np
import pandas as pd
import pytest
from matplotlib.collections import QuadMesh
from matplotlib.figure import Figure
from matplotlib.patches import PathPatch

from dms_heatmap.palette import fitness_cmap
from dms_heatmap.scale import ColourScale
from dms_heatmap.structure import HELIX, LOOP, STRAND, UNMODELLED, Run
from dms_heatmap.track import StructureTrack, TrackStyle, _shape, draw_track

from .test_structure import structure_from


def extent(run: Run) -> tuple[float, float, float]:
    """(left, right, greatest distance from the spine) of a run's outline."""
    vertices = _shape(run).vertices
    return (
        float(vertices[:, 0].min()),
        float(vertices[:, 0].max()),
        float(np.abs(vertices[:, 1]).max()),
    )


@pytest.mark.parametrize("code", [HELIX, STRAND, LOOP, UNMODELLED])
def test_every_shape_spans_the_full_width_of_its_positions(code):
    left, right, _ = extent(Run(code, 4, 9))
    assert (left, right) == (3.5, 9.5)


def test_the_shapes_are_ordered_by_thickness():
    # Thickness is the shapes' only shared cue, so their order carries meaning:
    # a stretch with no model is the flattest thing on a strip, a loop is
    # clearly thinner than a strand, and the coil of a helix is the boldest.
    unmodelled = extent(Run(UNMODELLED, 1, 12))[2]
    loop = extent(Run(LOOP, 1, 12))[2]
    body = extent(Run(STRAND, 1, 12, open_end=True))[2]
    head = extent(Run(STRAND, 1, 12))[2]
    helix = extent(Run(HELIX, 1, 12))[2]

    assert unmodelled < loop < body < head < helix
    assert helix < 1.0  # nothing touches the edge of its strip


def test_a_strand_that_ends_in_the_window_gets_an_arrowhead():
    vertices = _shape(Run(STRAND, 1, 10)).vertices
    # The head is the only part that comes to a point on the spine.
    tip = vertices[np.isclose(vertices[:, 1], 0.0)]
    assert len(tip) == 1
    assert tip[0][0] == pytest.approx(10.5)


def test_a_strand_continued_on_the_next_block_gets_no_arrowhead():
    # An arrowhead at the fold would say the strand ends there, which is the
    # one thing it does not do.
    run = Run(STRAND, 1, 10, origin=1, open_end=True)
    vertices = _shape(run).vertices
    assert not np.any(np.isclose(vertices[:, 1], 0.0))
    # It is still drawn at the thickness of a strand body, not of a loop, so
    # the element reads as the same element on both blocks.
    assert extent(run)[2] > extent(Run(LOOP, 1, 10))[2]


def test_a_helix_starts_its_coil_at_the_centre_of_the_strip():
    # The midline crosses the spine where the helix begins, so a helix meets
    # the loop before it without a step: the ribbon's two edges sit at plus and
    # minus its half-thickness there.
    vertices = _shape(Run(HELIX, 1, 20)).vertices
    at_left = vertices[np.isclose(vertices[:, 0], 0.5)]
    assert np.abs(at_left[:, 1]) == pytest.approx(0.42, abs=1e-6)


def test_a_helix_continued_from_the_block_above_keeps_its_phase():
    # Half a turn in from the start of the helix the ribbon is at its highest,
    # so a helix wrapped onto the next block must not restart from the centre.
    whole = _shape(Run(HELIX, 1, 20)).vertices
    continued = _shape(Run(HELIX, 11, 20, origin=1)).vertices

    for x in (11.0, 14.0, 17.0):
        expected = whole[np.isclose(whole[:, 0], x, atol=0.04)][:, 1].max()
        actual = continued[np.isclose(continued[:, 0], x, atol=0.04)][:, 1].max()
        assert actual == pytest.approx(expected, abs=0.02)


# --- drawing --------------------------------------------------------------


def draw(codes: str, values: dict[int, float], start=1, end=None, **kwargs):
    """Draw one strip and hand back its axes."""
    end = end or len(codes)
    figure = Figure(figsize=(4, 0.3))
    ax = figure.add_subplot()
    scale = ColourScale.from_values(np.array([0.0, 0.5, 1.0, 1.5]))
    draw_track(
        ax,
        structure_from(codes),
        StructureTrack(key="t", label="t", values=pd.Series(values, dtype=float)),
        start,
        end,
        end - start + 1,
        fitness_cmap(),
        scale.norm(),
        style=TrackStyle(**kwargs),
    )
    return ax


def test_a_strip_carries_one_mesh_of_values_clipped_to_the_shapes():
    ax = draw("HHHEEELLL", {p: 0.5 for p in range(1, 10)})
    meshes = [a for a in ax.get_children() if isinstance(a, QuadMesh)]

    # One mesh of values over one of "not measured" grey, both clipped to the
    # outline: a bare mesh would paint the whole rectangle of the strip.
    assert len(meshes) == 2
    assert all(mesh.get_clip_path() is not None for mesh in meshes)


def test_unmodelled_positions_are_outside_the_clip():
    ax = draw("HHH", {p: 0.5 for p in range(1, 7)}, end=6)
    outlined = [
        p for p in ax.get_children()
        if isinstance(p, PathPatch) and p.get_edgecolor()[3] > 0
    ]
    # Three positions of helix are outlined; the unmodelled tail is not, so no
    # fitness colour can reach it.
    assert len(outlined) == 1
    assert outlined[0].get_path().vertices[:, 0].max() == pytest.approx(3.5)


def test_the_strip_keeps_the_block_width_when_the_block_is_short():
    # A final block of 12 positions in a figure of 60 must still draw its
    # positions the same width as every other block.
    figure = Figure(figsize=(4, 0.3))
    ax = figure.add_subplot()
    scale = ColourScale.from_values(np.array([0.0, 1.0]))
    draw_track(
        ax,
        structure_from("HHHHHHHHHHHH"),
        StructureTrack(key="t", label="t", values=pd.Series(dtype=float)),
        1,
        12,
        60,
        fitness_cmap(),
        scale.norm(),
    )
    assert ax.get_xlim() == (0.5, 60.5)


def test_a_track_summarises_itself_for_the_manifest():
    track = StructureTrack(
        key="position_mean",
        label="mean",
        values=pd.Series({1: 0.2, 2: float("nan"), 3: 0.8}),
        description="the mean",
    )
    summary = track.as_dict()
    assert summary["n_positions"] == 2
    assert summary["median"] == pytest.approx(0.5)
    assert (summary["min"], summary["max"]) == (0.2, 0.8)


def test_an_empty_track_summarises_without_dividing_by_zero():
    summary = StructureTrack(key="k", label="l", values=pd.Series(dtype=float)).as_dict()
    assert summary["n_positions"] == 0
    assert summary["median"] is None
