"""Normalise replicate fitness onto the wild-type = 1 / nonsense = 0 scale.

Each replicate is anchored independently and only then averaged.  Replicates
differ in dynamic range (their raw wild-type and nonsense modes sit at
different values), so averaging first and rescaling afterwards would let the
widest replicate dominate the mean.

For replicate *r*::

    fitness_r = (raw_r - stop_r) / (wt_r - stop_r)

where ``wt_r`` is the median raw fitness of the wild-type-protein variants
(``aa_ham == 0``: synonymous, so the protein is wild type) and ``stop_r`` is
the median raw fitness of the nonsense variants (``mut_aa == "*"``).  That maps
the wild-type mode to 1 and the nonsense mode to 0 by construction, leaves the
scale linear, and does not clamp anything: variants fitter than wild type
exceed 1 and variants worse than a stop fall below 0.

Medians, not means, are the default anchor statistic -- both anchor
distributions have long tails (nonsense variants near the C-terminus are often
tolerated), and a median is unmoved by them.

The reported per-variant value is the mean of the available normalised
replicates.  Note that these files ship their own ``mean fitness`` column; the
pipeline ignores it.  See README.md ("A note on the ``mean fitness`` column").
"""

from __future__ import annotations

from dataclasses import dataclass, asdict

import numpy as np
import pandas as pd

__all__ = ["Anchors", "NormalisationError", "normalise", "replicate_anchors"]

#: Below this, the wild-type/nonsense separation is too small to rescale against.
MIN_DYNAMIC_RANGE = 1e-6

#: Anchors computed from fewer observations than this are reported as suspect.
MIN_ANCHOR_N = 5


class NormalisationError(ValueError):
    """The anchors needed to place the wild-type and nonsense modes are unusable."""


@dataclass(frozen=True)
class Anchors:
    """Per-replicate anchor values on the raw fitness scale."""

    replicate: str
    wt: float
    stop: float
    n_wt: int
    n_stop: int

    @property
    def dynamic_range(self) -> float:
        return self.wt - self.stop

    def rescale(self, values: pd.Series) -> pd.Series:
        return (values - self.stop) / self.dynamic_range

    def as_dict(self) -> dict:
        d = asdict(self)
        d["dynamic_range"] = self.dynamic_range
        return d


def wild_type_mask(table: pd.DataFrame) -> pd.Series:
    """Rows whose encoded protein is wild type (synonymous variants)."""
    return table["aa_ham"] == 0


def nonsense_mask(table: pd.DataFrame) -> pd.Series:
    """Rows introducing a premature stop codon."""
    return (table["mut_aa"] == "*").fillna(False)


def missense_mask(table: pd.DataFrame) -> pd.Series:
    """Single-amino-acid substitutions that are not stops."""
    return (table["aa_ham"] == 1) & ~nonsense_mask(table)


def replicate_anchors(
    table: pd.DataFrame, column: str, statistic: str = "median"
) -> Anchors:
    """Compute the wild-type and nonsense anchors for one replicate column."""
    if statistic not in {"median", "mean"}:
        raise ValueError(f"unknown anchor statistic: {statistic!r}")
    agg = np.nanmedian if statistic == "median" else np.nanmean

    values = pd.to_numeric(table[column], errors="coerce")
    wt_values = values[wild_type_mask(table)].dropna()
    stop_values = values[nonsense_mask(table)].dropna()

    if wt_values.empty:
        raise NormalisationError(
            f"{column}: no wild-type-protein (aa_ham == 0) measurements to anchor on"
        )
    if stop_values.empty:
        raise NormalisationError(
            f"{column}: no nonsense (mut_aa == '*') measurements to anchor on"
        )

    anchors = Anchors(
        replicate=column,
        wt=float(agg(wt_values)),
        stop=float(agg(stop_values)),
        n_wt=int(wt_values.size),
        n_stop=int(stop_values.size),
    )
    if anchors.dynamic_range <= MIN_DYNAMIC_RANGE:
        raise NormalisationError(
            f"{column}: wild-type anchor ({anchors.wt:.4g}) is not above the "
            f"nonsense anchor ({anchors.stop:.4g}); cannot rescale"
        )
    return anchors


def normalise(
    table: pd.DataFrame,
    rep_columns,
    statistic: str = "median",
    min_replicates: int = 1,
) -> tuple[pd.DataFrame, list[Anchors], list[str]]:
    """Add normalised per-replicate and summary fitness columns.

    Returns the augmented table, the anchors used, and any warnings raised.

    The summary columns are ``fitness`` (mean of available normalised
    replicates), ``fitness_sd`` (sample standard deviation, ``NaN`` with fewer
    than two replicates) and ``n_replicates``.  Variants measured in fewer than
    ``min_replicates`` replicates get ``NaN`` fitness, so a thinly supported
    variant never reaches the heatmap as if it were solid.
    """
    out = table.copy()
    anchors: list[Anchors] = []
    warnings: list[str] = []
    norm_columns: list[str] = []

    for column in rep_columns:
        anchor = replicate_anchors(out, column, statistic=statistic)
        anchors.append(anchor)
        if anchor.n_wt < MIN_ANCHOR_N or anchor.n_stop < MIN_ANCHOR_N:
            warnings.append(
                f"{column}: anchors rest on few observations "
                f"(n_wt={anchor.n_wt}, n_stop={anchor.n_stop})"
            )
        target = f"norm_{column}"
        out[target] = anchor.rescale(pd.to_numeric(out[column], errors="coerce"))
        norm_columns.append(target)

    values = out[norm_columns]
    out["n_replicates"] = values.notna().sum(axis=1).astype(int)
    out["fitness"] = values.mean(axis=1, skipna=True)
    out["fitness_sd"] = values.std(axis=1, skipna=True, ddof=1)

    if min_replicates > 1:
        thin = out["n_replicates"] < min_replicates
        if thin.any():
            warnings.append(
                f"{int(thin.sum())} variant(s) measured in fewer than "
                f"{min_replicates} replicates were set to NaN"
            )
            out.loc[thin, ["fitness", "fitness_sd"]] = np.nan

    return out, anchors, warnings
