"""Colour limits, tail saturation and the diverging midpoint."""

from __future__ import annotations

import numpy as np
import pytest
from matplotlib.colors import Normalize, TwoSlopeNorm

from dms_heatmap.palette import BLUE_RAMP, RED_RAMP, fitness_cmap
from dms_heatmap.scale import ColourScale


@pytest.fixture
def values():
    return np.linspace(0.0, 1.0, 101)


def test_limits_are_the_requested_quantiles(values):
    scale = ColourScale.from_values(values, 0.10, 0.90, center=None)
    assert scale.vmin == pytest.approx(0.10)
    assert scale.vmax == pytest.approx(0.90)


def test_tails_are_counted(values):
    scale = ColourScale.from_values(values, 0.10, 0.90, center=None)
    # 101 evenly spaced points: 10 below 0.10 and 10 above 0.90.
    assert scale.clipped_low == 10
    assert scale.clipped_high == 10


def test_both_tails_saturate_to_the_same_colour(values):
    """The whole point: everything past the limit gets one colour, not a ramp."""
    scale = ColourScale.from_values(values, 0.10, 0.90, center=None)
    cmap = fitness_cmap()
    norm = scale.norm()

    low = [cmap(norm(v)) for v in (-5.0, 0.0, 0.05, scale.vmin)]
    high = [cmap(norm(v)) for v in (scale.vmax, 0.95, 1.0, 20.0)]
    assert len(set(low)) == 1
    assert len(set(high)) == 1
    assert low[0] != high[0]


def test_low_is_red_and_high_is_blue(values):
    cmap = fitness_cmap()
    scale = ColourScale.from_values(values, 0.10, 0.90, center=None)
    norm = scale.norm()
    red = np.array(cmap(norm(scale.vmin))[:3])
    blue = np.array(cmap(norm(scale.vmax))[:3])
    assert red[0] > red[2]
    assert blue[2] > blue[0]


def test_nan_is_not_a_colour_on_the_ramp():
    """Unmeasured cells must not be painted with any colour the ramp can produce."""
    cmap = fitness_cmap()
    bad = np.asarray(cmap.get_bad())
    assert np.allclose(cmap(np.ma.masked_invalid(np.array([np.nan])))[0], bad)

    ramp = cmap(np.linspace(0, 1, 256))
    assert np.min(np.abs(ramp[:, :3] - bad[:3]).sum(axis=1)) > 0.05


def test_centred_scale_uses_a_two_slope_norm(values):
    scale = ColourScale.from_values(values, 0.10, 0.90, center=0.5)
    norm = scale.norm()
    assert isinstance(norm, TwoSlopeNorm)
    assert norm(0.5) == pytest.approx(0.5)


def test_uncentred_scale_is_linear(values):
    norm = ColourScale.from_values(values, 0.10, 0.90, center=None).norm()
    assert isinstance(norm, Normalize)
    assert not isinstance(norm, TwoSlopeNorm)


def test_midpoint_above_the_data_widens_the_empty_arm():
    """Every variant below wild type must still leave grey meaning 'wild type'."""
    values = np.linspace(-1.0, 0.4, 100)
    scale = ColourScale.from_values(values, 0.10, 0.90, center=1.0)
    assert scale.vmax > 1.0
    assert scale.notes and "below the midpoint" in scale.notes[0]
    assert scale.norm()(1.0) == pytest.approx(0.5)


def test_midpoint_below_the_data_widens_the_other_arm():
    values = np.linspace(2.0, 5.0, 100)
    scale = ColourScale.from_values(values, 0.10, 0.90, center=1.0)
    assert scale.vmin < 1.0
    assert scale.notes and "above the midpoint" in scale.notes[0]


def test_colourbar_ticks_include_the_anchors_when_in_range():
    scale = ColourScale.from_values(np.linspace(-1, 3, 100), 0.10, 0.90, center=1.0)
    ticks = scale.colourbar_ticks()
    assert 0.0 in ticks and 1.0 in ticks
    assert ticks == sorted(ticks)


def test_a_limit_that_would_collide_with_an_anchor_is_dropped():
    """vmin just below 0 must not print a second label on top of '0'."""
    scale = ColourScale.from_values(
        np.linspace(-0.13, 2.07, 500), 0.0, 1.0, center=1.0
    )
    ticks = scale.colourbar_ticks()
    assert 0.0 in ticks and 1.0 in ticks
    assert scale.vmin not in ticks       # too close to the nonsense anchor
    assert scale.vmax in ticks           # far from either anchor
    assert all(
        abs(a - b) > 0.05 for a, b in zip(sorted(ticks), sorted(ticks)[1:])
    )


def test_colourbar_ticks_omit_anchors_outside_the_range():
    scale = ColourScale.from_values(np.linspace(2.0, 5.0, 100), 0.10, 0.90, center=None)
    assert 0.0 not in scale.colourbar_ticks()


def test_empty_input_is_rejected():
    with pytest.raises(ValueError, match="no finite values"):
        ColourScale.from_values(np.array([np.nan, np.inf]))


def test_collapsed_limits_are_rejected():
    with pytest.raises(ValueError, match="collapsed"):
        ColourScale.from_values(np.zeros(50), 0.10, 0.90, center=None)


def test_bad_quantiles_are_rejected():
    with pytest.raises(ValueError, match="quantiles must satisfy"):
        ColourScale.from_values(np.linspace(0, 1, 10), 0.9, 0.1)


def test_the_two_arms_are_lightness_matched():
    """Matched arms keep either pole from visually dominating the figure."""
    def luminance(hex_colour: str) -> float:
        rgb = [int(hex_colour[i : i + 2], 16) / 255 for i in (1, 3, 5)]
        linear = [c / 12.92 if c <= 0.04045 else ((c + 0.055) / 1.055) ** 2.4 for c in rgb]
        return 0.2126 * linear[0] + 0.7152 * linear[1] + 0.0722 * linear[2]

    assert len(RED_RAMP) == len(BLUE_RAMP)
    for red, blue in zip(RED_RAMP, BLUE_RAMP):
        assert luminance(red) == pytest.approx(luminance(blue), abs=0.06)
