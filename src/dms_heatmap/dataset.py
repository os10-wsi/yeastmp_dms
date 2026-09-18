"""Discovery, loading and schema validation of DMS fitness tables.

Every dataset in the collection has the same shape: one row per assayed
variant, with a wild-type amino acid, a 1-based position, a mutant amino acid
(``*`` for a stop), an amino-acid Hamming distance, and one fitness column per
replicate.  This module turns any such file into a canonical frame so the rest
of the pipeline never has to think about column spelling.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path

import pandas as pd

__all__ = [
    "Dataset",
    "DatasetError",
    "canonical_columns",
    "dataset_name",
    "discover",
    "filter_by_input_count",
    "input_count_columns",
    "load",
]

#: Suffixes stripped from a file stem to get the dataset (gene) name.
_NAME_SUFFIXES = ("_fitness_estimation", "_fitness", "_dms")

#: Replicate value columns, in order of preference.
_REP_PATTERNS = (
    ("raw", re.compile(r"^raw_fitness_rep(\d+)$")),
    ("rescaled", re.compile(r"^rescaled_fitness_rep(\d+)$")),
)

_REQUIRED = ("wt_aa", "pos", "mut_aa", "aa_ham")

#: Columns the loader will not keep: per-row sequences make the frame enormous
#: (the full protein and ORF are repeated on every line) and nothing downstream
#: needs them except the wild-type protein sequence, which is captured
#: separately as :attr:`Dataset.wt_sequence`.
_BULK_COLUMNS = ("aa_seq", "nt_seq")


class DatasetError(ValueError):
    """A file does not look like a DMS fitness table this pipeline can read."""


def canonical_columns(columns) -> dict[str, str]:
    """Map raw column labels to snake_case canonical names.

    ``"wt aa"`` -> ``"wt_aa"``, ``"mean fitness"`` -> ``"mean_fitness"``.
    """
    out: dict[str, str] = {}
    for col in columns:
        key = re.sub(r"[^0-9a-z]+", "_", str(col).strip().lower()).strip("_")
        out[col] = key
    return out


def dataset_name(path: Path) -> str:
    """Derive a short dataset name (usually the gene) from a file path."""
    stem = Path(path).stem
    for suffix in _NAME_SUFFIXES:
        if stem.lower().endswith(suffix):
            stem = stem[: -len(suffix)]
            break
    return stem or Path(path).stem


def discover(root: Path, patterns: tuple[str, ...] = ("*.tsv", "*.csv")) -> list[Path]:
    """Recursively find candidate dataset files under ``root``.

    ``root`` may also be a single file, which makes ``--input`` accept either.
    """
    root = Path(root)
    if root.is_file():
        return [root]
    if not root.is_dir():
        raise DatasetError(f"input path does not exist: {root}")
    found: set[Path] = set()
    for pattern in patterns:
        found.update(p for p in root.rglob(pattern) if p.is_file())
    # Skip anything this pipeline wrote itself, so re-running over a folder that
    # contains a results directory does not pick up its own output.
    found = {
        p
        for p in found
        if not p.name.endswith((".normalised.tsv", ".matrix.tsv"))
        and p.name != "summary.tsv"
    }
    return sorted(found)


@dataclass(frozen=True)
class Dataset:
    """A loaded, canonicalised DMS fitness table."""

    name: str
    path: Path
    table: pd.DataFrame
    rep_columns: tuple[str, ...]
    value_kind: str
    wt_sequence: str | None = None
    notes: tuple[str, ...] = field(default_factory=tuple)

    @property
    def n_variants(self) -> int:
        return len(self.table)


def _separator(path: Path) -> str:
    return "," if path.suffix.lower() == ".csv" else "\t"


def _find_rep_columns(columns) -> tuple[str, tuple[str, ...]]:
    """Return (value_kind, replicate columns sorted by replicate number)."""
    for kind, pattern in _REP_PATTERNS:
        hits = [(int(m.group(1)), c) for c in columns if (m := pattern.match(c))]
        if hits:
            return kind, tuple(c for _, c in sorted(hits))
    raise DatasetError(
        "no replicate fitness columns found "
        "(expected raw_fitness_rep<N> or rescaled_fitness_rep<N>)"
    )


def load(path: Path) -> Dataset:
    """Read one dataset file into canonical form.

    Raises :class:`DatasetError` if required columns are missing.
    """
    path = Path(path)
    raw = pd.read_csv(path, sep=_separator(path), dtype=str, low_memory=False)
    raw = raw.rename(columns=canonical_columns(raw.columns))

    missing = [c for c in _REQUIRED if c not in raw.columns]
    if missing:
        raise DatasetError(f"{path.name}: missing required column(s): {', '.join(missing)}")

    value_kind, rep_columns = _find_rep_columns(raw.columns)

    wt_sequence = None
    if "aa_seq" in raw.columns:
        wt_sequence = _wild_type_sequence(raw)

    keep = [c for c in raw.columns if c not in _BULK_COLUMNS]
    table = raw[keep].copy()

    numeric = ["pos", "aa_ham", *rep_columns]
    numeric += [c for c in ("nt_ham",) if c in table.columns]
    for col in numeric:
        table[col] = pd.to_numeric(table[col], errors="coerce")

    for col in ("wt_aa", "mut_aa"):
        table[col] = table[col].astype("string").str.strip().replace({"": pd.NA})

    notes: list[str] = []
    unusable = table["aa_ham"].isna()
    if unusable.any():
        notes.append(f"dropped {int(unusable.sum())} row(s) with no aa_ham value")
        table = table.loc[~unusable]

    # Variant rows (aa_ham == 1) must carry a position and a mutant residue;
    # wild-type-protein rows (aa_ham == 0) legitimately have neither.
    variant = table["aa_ham"] == 1
    incomplete = variant & (table["pos"].isna() | table["mut_aa"].isna())
    if incomplete.any():
        notes.append(f"dropped {int(incomplete.sum())} variant row(s) with no pos/mut_aa")
        table = table.loc[~incomplete]

    table = table.reset_index(drop=True)
    if table.empty:
        raise DatasetError(f"{path.name}: no usable rows after validation")

    return Dataset(
        name=dataset_name(path),
        path=path,
        table=table,
        rep_columns=rep_columns,
        value_kind=value_kind,
        wt_sequence=wt_sequence,
        notes=tuple(notes),
    )


def input_count_columns(table: pd.DataFrame) -> list[str]:
    """Pre-selection read-count columns (``input1``, ``input2``, ...)."""
    pattern = re.compile(r"^input(\d+)$")
    hits = [(int(m.group(1)), c) for c in table.columns if (m := pattern.match(c))]
    return [c for _, c in sorted(hits)]


def filter_by_input_count(
    table: pd.DataFrame, minimum: int
) -> tuple[pd.DataFrame, int]:
    """Drop variants whose *lowest* input read count is below ``minimum``.

    Fitness from a handful of reads is dominated by sampling noise, and a
    heatmap gives a three-read estimate exactly as much ink as a
    three-thousand-read one.  Filtering before normalisation also keeps the
    wild-type and nonsense anchors off the noisiest rows.

    Returns the filtered table and the number of rows dropped.  A ``minimum``
    of 0, or a table with no ``input*`` columns, is a no-op.
    """
    columns = input_count_columns(table)
    if minimum <= 0 or not columns:
        return table, 0
    counts = table[columns].apply(pd.to_numeric, errors="coerce")
    keep = (counts.fillna(0) >= minimum).all(axis=1)
    return table.loc[keep].reset_index(drop=True), int((~keep).sum())


def _wild_type_sequence(raw: pd.DataFrame) -> str | None:
    """Pull the wild-type protein sequence out of a raw table, if present.

    Prefers an explicit ``wt`` flag; otherwise falls back to the most common
    sequence among rows with ``aa_ham == 0``, which are the synonymous
    (wild-type protein) variants.
    """
    seqs = raw["aa_seq"].astype("string")
    if "wt" in raw.columns:
        flag = raw["wt"].astype("string").str.strip().str.lower()
        hit = seqs[flag.isin({"true", "1", "yes"})].dropna()
        if not hit.empty:
            return str(hit.iloc[0])
    ham = pd.to_numeric(raw["aa_ham"], errors="coerce")
    syn = seqs[ham == 0].dropna()
    if syn.empty:
        return None
    return str(syn.mode().iloc[0])
