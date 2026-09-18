"""Per-class fitness distributions."""

from __future__ import annotations

import numpy as np
import pytest

from dms_heatmap.dataset import load
from dms_heatmap.distribution import (
    CLASS_ORDER,
    DistributionStyle,
    class_series,
    render_distribution,
)
from dms_heatmap.normalise import normalise
from dms_heatmap.palette import CLASS_COLOURS, CLASS_LINESTYLES
from dms_heatmap.plot import save_figure

from .conftest import REP_COLUMNS


@pytest.fixture
def normalised(synthetic_file):
    dataset = load(synthetic_file)
    table, _, _ = normalise(dataset.table, REP_COLUMNS)
    return table


def test_three_classes_in_a_fixed_order(normalised):
    series = class_series(normalised)
    assert [s.name for s in series] == list(CLASS_ORDER)
    assert [s.name for s in series] == ["missense", "synonymous", "nonsense"]


def test_classes_are_mutually_exclusive_and_complete(normalised):
    series = class_series(normalised)
    total = sum(s.n for s in series)
    measured = normalised["fitness"].notna().sum()
    assert total == measured


def test_classes_pick_up_the_planted_counts(normalised):
    by_name = {s.name: s for s in class_series(normalised)}
    assert by_name["synonymous"].n == 5   # the fixture's synonymous rows
    assert by_name["nonsense"].n == 7     # the fixture's stop rows
    assert by_name["missense"].n == 9     # 7 planted + 2 extra positions


def test_anchor_classes_sit_on_one_and_zero(normalised):
    """The whole point of the plot: this is the normalisation's own check."""
    by_name = {s.name: s for s in class_series(normalised)}
    assert by_name["synonymous"].median == pytest.approx(1.0)
    assert by_name["nonsense"].median == pytest.approx(0.0)


def test_variants_without_fitness_are_dropped_not_zeroed(normalised):
    table = normalised.copy()
    table.loc[table.index[-1], "fitness"] = np.nan
    before = sum(s.n for s in class_series(normalised))
    after = sum(s.n for s in class_series(table))
    assert after == before - 1


def test_colours_and_linestyles_are_assigned_per_class(normalised):
    for item in class_series(normalised):
        assert item.colour == CLASS_COLOURS[item.name]
        assert item.linestyle == CLASS_LINESTYLES[item.name]


def test_every_class_gets_its_own_colour():
    assert len(set(CLASS_COLOURS.values())) == len(CLASS_COLOURS)
    assert len(set(map(str, CLASS_LINESTYLES.values()))) == len(CLASS_LINESTYLES)


def test_series_summary_for_the_manifest(normalised):
    summary = {s.name: s.as_dict() for s in class_series(normalised)}
    assert summary["synonymous"]["median"] == pytest.approx(1.0)
    assert {"n", "median", "mean", "sd", "q0.05", "q0.95"} <= set(summary["missense"])


def test_render_overlay_uses_one_axes(normalised):
    figure, series = render_distribution(normalised, "gene1")
    assert len(figure.axes) == 1
    assert len(series) == 3


def test_render_facet_uses_one_axes_per_class(normalised):
    figure, _ = render_distribution(
        normalised, "gene1", style=DistributionStyle(layout="facet")
    )
    assert len(figure.axes) == 3


def test_all_classes_share_one_set_of_bin_edges(normalised):
    """Counts are only comparable between classes if the bins line up."""
    figure, _ = render_distribution(normalised, "gene1", style=DistributionStyle(bins=20))
    ax = figure.axes[0]
    # Two artists per class (fill + outline), all over the same x range.
    spans = {(round(p.get_extents().x0, 6), round(p.get_extents().x1, 6))
             for p in ax.patches} if ax.patches else set()
    assert len(spans) <= 1


def test_trimmed_values_are_counted_in_the_end_bins(normalised):
    """Clipping keeps the totals honest: no variant silently disappears."""
    table = normalised.copy()
    table.loc[table.index[0], "fitness"] = 500.0
    style = DistributionStyle(bins=10, trim=0.05)
    figure, series = render_distribution(table, "gene1", style=style)

    ax = figure.axes[0]
    assert ax.get_xlim()[1] < 500.0            # the outlier did not stretch the axis
    assert sum(s.n for s in series) == int(table["fitness"].notna().sum())


def test_axis_marks_both_anchors(normalised):
    figure, _ = render_distribution(normalised, "gene1")
    positions = {round(line.get_xdata()[0], 6) for line in figure.axes[0].lines}
    assert {0.0, 1.0} <= positions


@pytest.mark.parametrize(
    "kwargs,match",
    [
        ({"layout": "sideways"}, "unknown layout"),
        ({"bins": 1}, "bins must be at least 2"),
        ({"trim": 0.7}, "trim must be in"),
    ],
)
def test_invalid_style_is_rejected(kwargs, match):
    with pytest.raises(ValueError, match=match):
        DistributionStyle(**kwargs)


def test_a_table_with_no_fitness_is_an_error(normalised):
    empty = normalised.copy()
    empty["fitness"] = np.nan
    with pytest.raises(ValueError, match="no finite fitness values"):
        render_distribution(empty, "gene1")


def test_save_figure_appends_rather_than_replaces_the_extension(tmp_path, normalised):
    """A stem like 'gene1.distribution' must not lose its second component."""
    figure, _ = render_distribution(normalised, "gene1")
    written = save_figure(figure, tmp_path / "gene1.distribution", formats=("png",))
    assert [p.name for p in written] == ["gene1.distribution.png"]
    assert written[0].exists()
