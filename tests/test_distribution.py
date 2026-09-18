"""Per-class fitness distributions."""

from __future__ import annotations

import numpy as np
import pytest

from dms_heatmap.dataset import load
from dms_heatmap.distribution import (
    _bandwidth,
    CLASS_ORDER,
    DistributionStyle,
    class_series,
    render_distribution,
    smooth_counts,
)
from dms_heatmap.normalise import normalise
from dms_heatmap.palette import CLASS_COLOURS, CLASS_LINESTYLES
from dms_heatmap.plot import save_figure

from .conftest import REP_COLUMNS


def _xdata(line) -> np.ndarray:
    """Line x values as an array: axvline stores a list, plot an ndarray."""
    return np.asarray(line.get_xdata(), dtype=float)


def _curves(ax):
    """The density/step curves, excluding the two-point anchor rules."""
    return [line for line in ax.lines if _xdata(line).size > 2]


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


def test_all_classes_share_one_x_grid(normalised):
    """Heights are only comparable between classes if the grids line up."""
    figure, _ = render_distribution(normalised, "gene1")
    curves = _curves(figure.axes[0])
    assert len(curves) == 3
    first = _xdata(curves[0])
    for curve in curves[1:]:
        assert np.allclose(_xdata(curve), first)


def test_an_outlier_does_not_stretch_the_axis(normalised):
    table = normalised.copy()
    table.loc[table.index[0], "fitness"] = 500.0
    figure, series = render_distribution(
        table, "gene1", style=DistributionStyle(trim=0.05)
    )
    assert figure.axes[0].get_xlim()[1] < 500.0
    assert sum(s.n for s in series) == int(table["fitness"].notna().sum())


def test_step_mode_counts_trimmed_values_in_the_end_bins(normalised):
    """With smoothing off the tails are folded in, so no variant disappears."""
    from dms_heatmap.distribution import _bin_edges

    series = class_series(normalised)
    style = DistributionStyle(bins=10, trim=0.05, smoothing=0)
    edges, clipped = _bin_edges(series, style)

    total = 0
    for item in series:
        counts, _ = np.histogram(np.clip(item.values, edges[0], edges[-1]), bins=edges)
        total += int(counts.sum())
    assert total == sum(s.n for s in series)
    assert clipped > 0


def test_axis_marks_both_anchors(normalised):
    figure, _ = render_distribution(normalised, "gene1")
    # The anchor rules are two-point vertical lines; the density curves are not.
    rules = {
        round(float(_xdata(line)[0]), 6)
        for line in figure.axes[0].lines
        if _xdata(line).size == 2 and _xdata(line)[0] == _xdata(line)[1]
    }
    assert {0.0, 1.0} <= rules


# --- smoothing ------------------------------------------------------------


def test_smooth_curve_integrates_to_the_variant_count():
    """The y axis still means counts: the area under the curve is n."""
    rng = np.random.default_rng(0)
    values = rng.normal(1.0, 0.3, size=4000)
    bin_width = 0.05
    x, y = smooth_counts(values, -2.0, 4.0, bin_width, bandwidth=0.1)

    area = np.trapezoid(y, x) / bin_width
    assert area == pytest.approx(values.size, rel=0.02)


def test_smooth_curve_peaks_at_the_mode():
    rng = np.random.default_rng(1)
    values = rng.normal(0.7, 0.2, size=3000)
    x, y = smooth_counts(values, -1.0, 2.5, 0.05, bandwidth=0.08)
    assert x[int(np.argmax(y))] == pytest.approx(0.7, abs=0.06)


def test_smooth_curve_resolves_two_modes():
    """A bimodal screen must not be smoothed into one hump."""
    rng = np.random.default_rng(2)
    values = np.concatenate(
        [rng.normal(0.0, 0.12, 2000), rng.normal(1.0, 0.12, 2000)]
    )
    x, y = smooth_counts(values, -1.0, 2.0, 0.05, bandwidth=0.08)
    interior = (y[1:-1] > y[:-2]) & (y[1:-1] > y[2:])
    assert interior.sum() == 2


def test_bandwidth_scales_with_the_multiplier():
    rng = np.random.default_rng(3)
    values = rng.normal(0.0, 0.5, size=1000)
    assert _bandwidth(values, 2.0) == pytest.approx(2 * _bandwidth(values, 1.0))
    assert _bandwidth(values, 1.0) > 0


def test_bandwidth_is_robust_to_a_heavy_tail():
    """A few wild outliers must not smear out the real structure."""
    rng = np.random.default_rng(4)
    clean = rng.normal(0.0, 0.2, size=1000)
    with_tail = np.concatenate([clean, rng.normal(60.0, 5.0, size=20)])
    assert _bandwidth(with_tail, 1.0) < 2 * _bandwidth(clean, 1.0)


def test_degenerate_inputs_produce_no_curve():
    assert smooth_counts(np.array([]), 0, 1, 0.1, 0.1)[0].size == 0
    assert smooth_counts(np.ones(10), 0, 1, 0.1, 0.0)[0].size == 0
    assert _bandwidth(np.ones(10), 1.0) == 0.0     # no spread
    assert _bandwidth(np.array([1.0]), 1.0) == 0.0  # one point


def test_smoothing_zero_draws_steps_instead(normalised):
    smooth, _ = render_distribution(normalised, "gene1")
    stepped, _ = render_distribution(
        normalised, "gene1", style=DistributionStyle(smoothing=0)
    )
    assert len(_curves(smooth.axes[0])) == 3
    assert _curves(stepped.axes[0]) == []


def test_negative_smoothing_is_rejected():
    with pytest.raises(ValueError, match="smoothing must be"):
        DistributionStyle(smoothing=-1)


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
