"""Colour limits for the fitness heatmap.

The scale has two jobs.  It must saturate the tails -- everything in the top
decile gets one blue and everything in the bottom decile one red, so the middle
of the distribution keeps the whole ramp instead of being flattened by a
handful of outliers.  And its neutral midpoint must sit on a value that means
something, otherwise a diverging colormap is just decoration.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from matplotlib.colors import Normalize, TwoSlopeNorm

__all__ = ["ColourScale", "WT_FITNESS", "STOP_FITNESS"]

#: Fitness of the wild-type protein after normalisation, by construction.
WT_FITNESS = 1.0

#: Fitness of a null (nonsense) variant after normalisation, by construction.
STOP_FITNESS = 0.0


@dataclass(frozen=True)
class ColourScale:
    """Resolved colour limits, plus how they were arrived at."""

    vmin: float
    vmax: float
    center: float | None
    lower_quantile: float
    upper_quantile: float
    n_values: int
    clipped_low: int = 0
    clipped_high: int = 0
    notes: tuple[str, ...] = ()

    @classmethod
    def from_values(
        cls,
        values: np.ndarray,
        lower_quantile: float = 0.10,
        upper_quantile: float = 0.90,
        center: float | None = WT_FITNESS,
    ) -> "ColourScale":
        """Derive limits from the measured cells that will be plotted."""
        finite = np.asarray(values, dtype=float)
        finite = finite[np.isfinite(finite)]
        if finite.size == 0:
            raise ValueError("no finite values to build a colour scale from")
        if not 0.0 <= lower_quantile < upper_quantile <= 1.0:
            raise ValueError(
                f"quantiles must satisfy 0 <= lower < upper <= 1, "
                f"got {lower_quantile} and {upper_quantile}"
            )

        vmin = float(np.quantile(finite, lower_quantile))
        vmax = float(np.quantile(finite, upper_quantile))
        notes: list[str] = []

        if vmax - vmin <= 0:
            raise ValueError(
                f"colour limits collapsed: the {lower_quantile:.0%} and "
                f"{upper_quantile:.0%} quantiles are both {vmin:.4g}"
            )

        if center is not None:
            # A diverging ramp needs the midpoint strictly inside the range.
            # If the data sit entirely on one side of the anchor, widen the
            # empty arm slightly rather than dropping the anchor -- the same
            # colour must mean the same fitness in every dataset in the run.
            pad = 0.05 * (vmax - vmin)
            if center <= vmin:
                vmin = center - pad
                notes.append(
                    f"all plotted values lie above the midpoint {center:g}; "
                    f"lower limit widened to {vmin:.4g}"
                )
            elif center >= vmax:
                vmax = center + pad
                notes.append(
                    f"all plotted values lie below the midpoint {center:g}; "
                    f"upper limit widened to {vmax:.4g}"
                )

        return cls(
            vmin=vmin,
            vmax=vmax,
            center=center,
            lower_quantile=lower_quantile,
            upper_quantile=upper_quantile,
            n_values=int(finite.size),
            clipped_low=int(np.sum(finite < vmin)),
            clipped_high=int(np.sum(finite > vmax)),
            notes=tuple(notes),
        )

    def norm(self) -> Normalize:
        """The matplotlib norm for these limits.

        Values outside ``[vmin, vmax]`` are not masked; the colormap clamps
        them to its end colours, which is exactly the requested saturation.
        """
        if self.center is None:
            return Normalize(vmin=self.vmin, vmax=self.vmax)
        return TwoSlopeNorm(vmin=self.vmin, vcenter=self.center, vmax=self.vmax)

    def colourbar_ticks(self) -> list[float]:
        """The biological anchors inside the range, plus the limits.

        Anchors are placed first and the limits only where they will not
        collide with one: when a limit sits almost on top of an anchor it is
        the limit that goes, because "0 = nonsense" is worth more to a reader
        than a repeat of a number already in the caption.
        """
        span = self.vmax - self.vmin
        ticks = [
            anchor
            for anchor in (STOP_FITNESS, WT_FITNESS)
            if self.vmin < anchor < self.vmax
        ]
        for limit in (self.vmin, self.vmax):
            if all(abs(limit - tick) > 0.10 * span for tick in ticks):
                ticks.append(limit)
        return sorted(ticks)

    def as_dict(self) -> dict:
        return {
            "vmin": self.vmin,
            "vmax": self.vmax,
            "center": self.center,
            "lower_quantile": self.lower_quantile,
            "upper_quantile": self.upper_quantile,
            "n_values": self.n_values,
            "n_clipped_low": self.clipped_low,
            "n_clipped_high": self.clipped_high,
            "notes": list(self.notes),
        }
