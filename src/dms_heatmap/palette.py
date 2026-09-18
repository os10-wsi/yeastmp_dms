"""Colour for DMS fitness heatmaps.

The heatmap encodes a *diverging* quantity: fitness relative to two biological
anchors (0 = nonsense/null, 1 = wild-type protein).  A diverging encoding needs
two opposing hues and a neutral midpoint that reads as "nothing here" -- never a
rainbow, and never a hue at the midpoint.

The two arms are lightness-matched.  ``BLUE_RAMP`` is a documented sequential
blue ramp; ``RED_RAMP`` was derived from it by converting each blue step to
OKLCh, holding lightness and chroma, and rotating the hue to that of the
reference red (``#e34948``, OKLCh hue 24.9 deg).  Every derived step is inside
the sRGB gamut, so the red arm needed no chroma compression.  The practical
consequence is that a variant 0.3 units below the midpoint and one 0.3 units
above are equally dark, so neither arm visually dominates the figure.

Red vs blue is also the colour-vision-deficiency-safe diverging pair: under
deuteranopia and protanopia the two poles stay separable by hue (unlike
red/green), and they additionally differ in lightness away from the midpoint.
"""

from __future__ import annotations

from matplotlib.colors import LinearSegmentedColormap

__all__ = [
    "BLUE_RAMP",
    "RED_RAMP",
    "NEUTRAL",
    "MISSING",
    "WT_MARK",
    "fitness_cmap",
]

#: Sequential blue, light -> dark (steps 100..700).
BLUE_RAMP: tuple[str, ...] = (
    "#cde2fb", "#b7d3f6", "#9ec5f4", "#86b6ef", "#6da7ec", "#5598e7", "#3987e5",
    "#2a78d6", "#256abf", "#1c5cab", "#184f95", "#104281", "#0d366b",
)

#: Sequential red, light -> dark; lightness-matched to ``BLUE_RAMP`` step for step.
RED_RAMP: tuple[str, ...] = (
    "#fad6d2", "#f4c2be", "#f1aea8", "#ea9a93", "#e4857e", "#dd716a", "#d75853",
    "#c74845", "#b13f3c", "#9e3432", "#892b2a", "#762221", "#621b1a",
)

#: Neutral midpoint of the diverging ramp.
NEUTRAL = "#f0efec"

#: Cells with no measurement (variant not assayed, or filtered out).  This has
#: to be a grey no step of the ramp comes near -- the ramp's own midpoint is
#: near-white, so a pale "missing" would be mistaken for a wild-type-like
#: measurement.  This one is flatly desaturated and clearly darker than the
#: midpoint, so an absent variant never passes for a measured one.
MISSING = "#c6c5bf"

#: Marker ink for the wild-type residue at each position.
WT_MARK = "#0b0b0b"


def fitness_cmap(name: str = "dms_fitness") -> LinearSegmentedColormap:
    """Return the red -> neutral -> blue diverging colormap.

    Low values (loss of function) are red, high values (gain of function) are
    blue, and the midpoint is neutral grey.
    """
    stops = [*reversed(RED_RAMP), NEUTRAL, *BLUE_RAMP]
    return LinearSegmentedColormap.from_list(name, stops, N=256).with_extremes(
        bad=MISSING
    )
