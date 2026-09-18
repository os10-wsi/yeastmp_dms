"""Orchestration: turn a folder of DMS fitness tables into heatmaps.

One pass normalises every dataset and writes its tables; a second pass renders
the figures, which is what lets ``shared_scale`` put every dataset in a run on
one set of colour limits.
"""

from __future__ import annotations

import itertools
import json
import logging
import platform
import sys
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path

import matplotlib
import numpy as np
import pandas as pd

from . import __version__
from .dataset import (
    Dataset,
    DatasetError,
    discover,
    filter_by_input_count,
    input_count_columns,
    load,
)
from .matrix import FitnessMatrix, build_matrix
from .normalise import (
    Anchors,
    NormalisationError,
    missense_mask,
    nonsense_mask,
    normalise,
    wild_type_mask,
)
from .plot import HeatmapStyle, render_heatmap, save_heatmap
from .scale import WT_FITNESS, ColourScale

__all__ = ["PipelineConfig", "DatasetResult", "run"]

log = logging.getLogger("dms_heatmap")

#: Columns carried into the per-variant output table, when present.
_OUTPUT_COLUMNS = (
    "wt_aa",
    "pos",
    "mut_aa",
    "aa_ham",
    "nt_ham",
    "n_replicates",
    "fitness",
    "fitness_sd",
)


@dataclass
class PipelineConfig:
    """Everything that changes the numbers or the picture.

    Serialised into the run manifest, so a figure can always be traced back to
    the settings that produced it.
    """

    input: Path = Path("data/raw")
    output: Path = Path("results")
    patterns: tuple[str, ...] = ("*.tsv", "*.csv")

    # Quality filtering, applied before the anchors are computed
    min_input_count: int = 0

    # Normalisation
    anchor_statistic: str = "median"
    min_replicates: int = 1

    # Colour scale
    lower_quantile: float = 0.10
    upper_quantile: float = 0.90
    center: float | None = WT_FITNESS
    shared_scale: bool = False

    # Figure
    block_size: int = 100
    aa_order: str = "chemistry"
    include_stops: bool = True
    tick_every: int = 10
    mark_wild_type: bool = True
    formats: tuple[str, ...] = ("png", "pdf")

    # Outputs
    write_tables: bool = True

    def style(self) -> HeatmapStyle:
        return HeatmapStyle(
            block_size=self.block_size,
            tick_every=self.tick_every,
            mark_wild_type=self.mark_wild_type,
        )

    def as_dict(self) -> dict:
        out = asdict(self)
        out["input"] = str(self.input)
        out["output"] = str(self.output)
        out["patterns"] = list(self.patterns)
        out["formats"] = list(self.formats)
        return out


@dataclass
class DatasetResult:
    """Per-dataset outcome, both for the manifest and for the caller."""

    name: str
    source: Path
    n_variants: int
    n_missense: int
    n_nonsense: int
    n_wild_type: int
    coverage: float
    anchors: list[Anchors]
    replicate_correlations: dict[str, float]
    fitness_quantiles: dict[str, float]
    read_depth: dict = field(default_factory=dict)
    warnings: list[str] = field(default_factory=list)
    outputs: list[Path] = field(default_factory=list)
    scale: ColourScale | None = None

    def as_dict(self) -> dict:
        return {
            "dataset": self.name,
            "source": str(self.source),
            "n_variants": self.n_variants,
            "n_missense": self.n_missense,
            "n_nonsense": self.n_nonsense,
            "n_wild_type_protein": self.n_wild_type,
            "coverage": self.coverage,
            "anchors": [a.as_dict() for a in self.anchors],
            "replicate_correlations": self.replicate_correlations,
            "fitness_quantiles": self.fitness_quantiles,
            "read_depth": self.read_depth,
            "colour_scale": self.scale.as_dict() if self.scale else None,
            "warnings": self.warnings,
            "outputs": [str(p) for p in self.outputs],
        }


def _replicate_correlations(table: pd.DataFrame, rep_columns) -> dict[str, float]:
    """Pearson r between every pair of normalised replicates, over variants."""
    out: dict[str, float] = {}
    subset = table.loc[missense_mask(table) | nonsense_mask(table)]
    for a, b in itertools.combinations(rep_columns, 2):
        pair = subset[[f"norm_{a}", f"norm_{b}"]].dropna()
        key = f"{a}~{b}"
        out[key] = float(pair.iloc[:, 0].corr(pair.iloc[:, 1])) if len(pair) > 2 else float("nan")
    return out


def _fitness_quantiles(values: np.ndarray) -> dict[str, float]:
    finite = values[np.isfinite(values)]
    if finite.size == 0:
        return {}
    qs = (0.01, 0.05, 0.10, 0.25, 0.50, 0.75, 0.90, 0.95, 0.99)
    out = {f"q{q:.2f}": float(np.quantile(finite, q)) for q in qs}
    out["min"] = float(finite.min())
    out["max"] = float(finite.max())
    out["mean"] = float(finite.mean())
    return out


def _read_depth_report(table: pd.DataFrame) -> dict:
    """How many variants survive a range of input read-count thresholds.

    Reported whether or not filtering is switched on, so the cost of a
    threshold is visible before anyone applies one.
    """
    columns = input_count_columns(table)
    if not columns:
        return {}
    lowest = table[columns].apply(pd.to_numeric, errors="coerce").fillna(0).min(axis=1)
    total = len(lowest)
    return {
        "median_lowest_input_count": float(lowest.median()),
        "fraction_retained": {
            str(t): round(float((lowest >= t).mean()), 4) for t in (5, 10, 20, 50, 100)
        },
        "n_total": total,
    }


def _prepare(dataset: Dataset, config: PipelineConfig):
    """Normalise one dataset and pivot it, returning everything downstream needs."""
    depth = _read_depth_report(dataset.table)
    source, dropped = filter_by_input_count(dataset.table, config.min_input_count)

    table, anchors, warnings = normalise(
        source,
        dataset.rep_columns,
        statistic=config.anchor_statistic,
        min_replicates=config.min_replicates,
    )
    warnings = [*dataset.notes, *warnings]
    if dropped:
        warnings.append(
            f"dropped {dropped} variant(s) with an input read count below "
            f"{config.min_input_count}"
        )
    elif depth and config.min_input_count <= 0:
        thin = 1 - depth["fraction_retained"]["20"]
        if thin > 0.15:
            warnings.append(
                f"{thin:.0%} of variants have fewer than 20 input reads in at least "
                "one replicate and their fitness is mostly sampling noise; consider "
                "--min-input-count 20"
            )
    if dataset.value_kind != "raw":
        warnings.append(
            f"no raw_fitness_rep* columns; normalised the "
            f"{dataset.value_kind}_fitness_rep* columns instead"
        )

    matrix = build_matrix(
        table,
        dataset=dataset.name,
        value_column="fitness",
        aa_order=config.aa_order,
        wt_sequence=dataset.wt_sequence,
        include_stops=config.include_stops,
    )

    result = DatasetResult(
        name=dataset.name,
        source=dataset.path,
        n_variants=len(table),
        n_missense=int(missense_mask(table).sum()),
        n_nonsense=int(nonsense_mask(table).sum()),
        n_wild_type=int(wild_type_mask(table).sum()),
        coverage=matrix.coverage,
        anchors=anchors,
        replicate_correlations=_replicate_correlations(table, dataset.rep_columns),
        fitness_quantiles=_fitness_quantiles(matrix.finite_values()),
        read_depth=depth,
        warnings=warnings,
    )
    return table, matrix, result


def _write_tables(
    table: pd.DataFrame, matrix: FitnessMatrix, dataset: Dataset, out_dir: Path
) -> list[Path]:
    out_dir.mkdir(parents=True, exist_ok=True)
    columns = [c for c in _OUTPUT_COLUMNS if c in table.columns]
    columns += [c for c in table.columns if c.startswith("norm_")]

    tidy = out_dir / f"{dataset.name}.normalised.tsv"
    table[columns].to_csv(tidy, sep="\t", index=False, float_format="%.6g")

    wide = out_dir / f"{dataset.name}.matrix.tsv"
    matrix.values.to_csv(wide, sep="\t", float_format="%.6g")
    return [tidy, wide]


def _subtitle(config: PipelineConfig, result: DatasetResult, scale: ColourScale) -> str:
    reps = len(result.anchors)
    centre = (
        "neutral grey at wild type"
        if scale.center == WT_FITNESS
        else (
            f"neutral grey at {scale.center:g}"
            if scale.center is not None
            else "linear ramp between limits"
        )
    )
    parts = [
        f"{result.n_missense:,} substitutions and {result.n_nonsense:,} nonsense "
        f"variants over {reps} replicates",
        "red = loss of function, blue = gain",
        centre,
    ]
    if config.min_input_count > 0:
        parts.insert(1, f"filtered to >={config.min_input_count} input reads per replicate")
    if config.min_replicates > 1:
        parts.insert(1, f"at least {config.min_replicates} replicates per variant")
    return "  |  ".join(parts)


def run(config: PipelineConfig) -> list[DatasetResult]:
    """Run the whole pipeline and return one result per dataset."""
    paths = discover(config.input, config.patterns)
    if not paths:
        raise DatasetError(
            f"no dataset files matching {', '.join(config.patterns)} under {config.input}"
        )
    log.info("found %d dataset file(s) under %s", len(paths), config.input)

    config.output.mkdir(parents=True, exist_ok=True)
    tables_dir = config.output / "tables"
    figures_dir = config.output / "figures"

    prepared: list[tuple[Dataset, FitnessMatrix, DatasetResult]] = []
    failures: list[tuple[Path, str]] = []

    for path in paths:
        try:
            dataset = load(path)
            table, matrix, result = _prepare(dataset, config)
        except (DatasetError, NormalisationError, ValueError) as exc:
            log.error("skipping %s: %s", path.name, exc)
            failures.append((path, str(exc)))
            continue

        log.info(
            "%s: %d variants, %d replicates, coverage %.0f%%",
            dataset.name,
            result.n_variants,
            len(dataset.rep_columns),
            100 * result.coverage,
        )
        for warning in result.warnings:
            log.warning("%s: %s", dataset.name, warning)

        if config.write_tables:
            result.outputs.extend(_write_tables(table, matrix, dataset, tables_dir))
        prepared.append((dataset, matrix, result))

    if not prepared:
        raise DatasetError("no dataset could be processed; see errors above")

    shared: ColourScale | None = None
    if config.shared_scale:
        pooled = np.concatenate([m.finite_values() for _, m, _ in prepared])
        shared = ColourScale.from_values(
            pooled,
            lower_quantile=config.lower_quantile,
            upper_quantile=config.upper_quantile,
            center=config.center,
        )
        log.info(
            "shared colour scale across %d dataset(s): %.3f to %.3f",
            len(prepared),
            shared.vmin,
            shared.vmax,
        )

    style = config.style()
    for dataset, matrix, result in prepared:
        scale = shared or ColourScale.from_values(
            matrix.finite_values(),
            lower_quantile=config.lower_quantile,
            upper_quantile=config.upper_quantile,
            center=config.center,
        )
        result.scale = scale
        result.warnings.extend(scale.notes)

        figure = render_heatmap(
            matrix,
            scale,
            style=style,
            title=f"{dataset.name} - deep mutational scanning fitness",
            subtitle=_subtitle(config, result, scale),
        )
        written = save_heatmap(figure, figures_dir / dataset.name, formats=config.formats)
        result.outputs.extend(written)
        log.info("%s: wrote %s", dataset.name, ", ".join(p.name for p in written))

    results = [r for _, _, r in prepared]
    _write_summary(results, failures, config)
    return results


def _write_summary(
    results: list[DatasetResult], failures: list[tuple[Path, str]], config: PipelineConfig
) -> None:
    summary = pd.DataFrame(
        [
            {
                "dataset": r.name,
                "source": str(r.source),
                "n_variants": r.n_variants,
                "n_missense": r.n_missense,
                "n_nonsense": r.n_nonsense,
                "n_wild_type_protein": r.n_wild_type,
                "coverage": round(r.coverage, 4),
                "median_fitness": round(r.fitness_quantiles.get("q0.50", float("nan")), 4),
                "vmin": round(r.scale.vmin, 4) if r.scale else None,
                "vmax": round(r.scale.vmax, 4) if r.scale else None,
                "min_replicate_r": (
                    round(min(r.replicate_correlations.values()), 4)
                    if r.replicate_correlations
                    else None
                ),
                "n_warnings": len(r.warnings),
            }
            for r in results
        ]
    )
    summary.to_csv(config.output / "summary.tsv", sep="\t", index=False)

    manifest = {
        "generated_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "dms_heatmap_version": __version__,
        "python": sys.version.split()[0],
        "platform": platform.platform(),
        "packages": {
            "numpy": np.__version__,
            "pandas": pd.__version__,
            "matplotlib": matplotlib.__version__,
        },
        "config": config.as_dict(),
        "datasets": [r.as_dict() for r in results],
        "failures": [{"source": str(p), "error": e} for p, e in failures],
    }
    with (config.output / "run_manifest.json").open("w") as handle:
        json.dump(manifest, handle, indent=2)
