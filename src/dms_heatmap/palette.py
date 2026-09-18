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
    "CLASS_COLOURS",
    "CLASS_LINESTYLES",
    "RED_RAMP",
    "NEUTRAL",
    "MISSING",
    "SS_OUTLINE",
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

#: Outline of the secondary-structure shapes.  Ink, not a hue: the shapes are
#: already carrying a fitness colour inside them, and a coloured outline would
#: be read as a second encoding.  Dark enough to define a hairline loop against
#: the surface, light enough not to out-weigh the fill on a thin coil.
SS_OUTLINE = "#3f3e3b"


#: Colours for the three variant classes in the distribution plot.
#:
#: Blue is ``BLUE_RAMP`` step 450 and red is ``RED_RAMP`` step 550, so the two
#: figures share their ink.  Green is new, and its step was chosen by running
#: the trio through a colour-vision-deficiency check rather than by eye: a
#: saturated green against a mid red is the classic red/green failure, and the
#: obvious pairing (``#008300`` with ``#e34948``) lands at protanope Delta E 7.2
#: -- inside the band where a palette is only legal with secondary encoding.
#: This trio's worst all-pairs separation is Delta E 15.0 (deutan), with
#: normal-vision 29.1 and every colour above 3:1 contrast on the surface.
CLASS_COLOURS: dict[str, str] = {
    "missense": "#2a78d6",
    "synonymous": "#00a83d",
    "nonsense": "#9e3432",
}

#: Line style per class.  Redundant with colour on screen, but it is what keeps
#: the three curves apart in greyscale print and under any residual colour
#: confusion.
CLASS_LINESTYLES: dict[str, object] = {
    "missense": "solid",
    "synonymous": (0, (5, 1.6)),
    "nonsense": (0, (1.4, 1.4)),
}


def fitness_cmap(name: str = "dms_fitness") -> LinearSegmentedColormap:
    """Return the red -> neutral -> blue diverging colormap.

    Low values (loss of function) are red, high values (gain of function) are
    blue, and the midpoint is neutral grey.
    """
    stops = [*reversed(RED_RAMP), NEUTRAL, *BLUE_RAMP]
    return LinearSegmentedColormap.from_list(name, stops, N=256).with_extremes(
        bad=MISSING
    )
