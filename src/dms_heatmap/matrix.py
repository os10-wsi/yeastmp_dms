"""Reshape normalised variant rows into a residue x position matrix."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from .normalise import missense_mask, nonsense_mask

__all__ = ["FitnessMatrix", "AA_ORDERS", "build_matrix"]

STOP = "*"

#: Amino acids grouped by side-chain chemistry, so blocks of similar
#: substitutions sit together and buried/exposed patterns read as bands.
_CHEMISTRY = "AVLIMFWYSTNQCGPHKRDE"

#: Row orders the heatmap can use.  Stops are always the last row.
AA_ORDERS: dict[str, str] = {
    "chemistry": _CHEMISTRY,
    "alphabetical": "".join(sorted(_CHEMISTRY)),
}


@dataclass(frozen=True)
class FitnessMatrix:
    """A residue x position grid of normalised fitness.

    ``values`` is indexed by amino acid (rows, ``*`` last) and by 1-based
    position (columns).  ``wt_residues`` gives the wild-type residue at each
    position where it is known, and ``coverage`` the fraction of substitutions
    actually measured.
    """

    values: pd.DataFrame
    wt_residues: pd.Series
    dataset: str

    @property
    def positions(self) -> np.ndarray:
        return self.values.columns.to_numpy()

    @property
    def residues(self) -> list[str]:
        return list(self.values.index)

    @property
    def coverage(self) -> float:
        measurable = self.values.notna().to_numpy().sum()
        total = self.values.size - int(self.wt_residues.notna().sum())
        return float(measurable / total) if total else 0.0

    def finite_values(self) -> np.ndarray:
        """All measured cells as a flat array, for percentile limits."""
        flat = self.values.to_numpy(dtype=float).ravel()
        return flat[np.isfinite(flat)]


def _wt_residue_series(
    table: pd.DataFrame, positions: np.ndarray, wt_sequence: str | None
) -> pd.Series:
    """Wild-type residue per position, from the table and/or the WT sequence."""
    series = pd.Series(pd.NA, index=positions, dtype="string")

    annotated = table.loc[table["pos"].notna() & table["wt_aa"].notna(), ["pos", "wt_aa"]]
    if not annotated.empty:
        per_pos = annotated.groupby(annotated["pos"].astype(int))["wt_aa"].agg(
            lambda s: s.mode().iloc[0]
        )
        series.update(per_pos.reindex(series.index).dropna())

    if wt_sequence:
        from_seq = pd.Series(
            {p: wt_sequence[p - 1] for p in positions if 1 <= p <= len(wt_sequence)},
            dtype="string",
        )
        series = series.fillna(from_seq)

    return series


def build_matrix(
    table: pd.DataFrame,
    dataset: str,
    value_column: str = "fitness",
    aa_order: str = "chemistry",
    wt_sequence: str | None = None,
    include_stops: bool = True,
) -> FitnessMatrix:
    """Pivot normalised variants into a :class:`FitnessMatrix`.

    Positions span 1..N with gaps preserved as empty columns, where N is the
    wild-type protein length when known and otherwise the highest assayed
    position -- so a region that was never assayed is visibly absent rather
    than silently closed up.
    """
    if aa_order not in AA_ORDERS:
        raise ValueError(
            f"unknown aa_order {aa_order!r}; choose from {sorted(AA_ORDERS)}"
        )

    rows = list(AA_ORDERS[aa_order])
    if include_stops:
        rows.append(STOP)

    keep = missense_mask(table) | (nonsense_mask(table) if include_stops else False)
    variants = table.loc[keep, ["pos", "mut_aa", value_column]].copy()
    variants["pos"] = variants["pos"].astype(int)

    highest = int(variants["pos"].max()) if not variants.empty else 0
    length = max(highest, len(wt_sequence) if wt_sequence else 0)
    positions = np.arange(1, length + 1, dtype=int)

    grid = (
        variants.pivot_table(
            index="mut_aa", columns="pos", values=value_column, aggfunc="mean"
        )
        .reindex(index=rows, columns=positions)
        .astype(float)
    )
    grid.index.name = "mut_aa"
    grid.columns.name = "pos"

    return FitnessMatrix(
        values=grid,
        wt_residues=_wt_residue_series(table, positions, wt_sequence),
        dataset=dataset,
    )
