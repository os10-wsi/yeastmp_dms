"""Command line entry point.

Settings come from three places, later ones winning: the dataclass defaults in
:mod:`dms_heatmap.pipeline`, a TOML config file, and command line flags.  The
resolved settings are written into ``run_manifest.json`` alongside the figures.
"""

from __future__ import annotations

import argparse
import logging
import sys
import tomllib
from pathlib import Path

from .dataset import DatasetError
from .matrix import AA_ORDERS
from .normalise import NormalisationError
from .pipeline import PipelineConfig, run

__all__ = ["build_parser", "config_from_args", "main"]

_PATH_FIELDS = {"input", "output"}
_TUPLE_FIELDS = {"patterns", "formats"}


def _describe(value) -> str:
    if value is None:
        return "none"
    if isinstance(value, bool):
        return "on" if value else "off"
    if isinstance(value, (tuple, list)):
        return " ".join(str(v) for v in value)
    return str(value)


def build_parser() -> argparse.ArgumentParser:
    defaults = PipelineConfig()

    def helptext(field: str, text: str) -> str:
        """Help text carrying the real default.

        Every flag defaults to ``None`` so that leaving it off does not shadow
        a value set in the config file, which means argparse cannot report the
        defaults itself -- they live on :class:`PipelineConfig`.
        """
        value = getattr(defaults, field)
        if field == "center" and value == 1.0:
            value = "wt"
        return f"{text} (default: {_describe(value)})"

    parser = argparse.ArgumentParser(
        prog="dms-heatmap",
        description=(
            "Normalise deep mutational scanning fitness data to wild type = 1 / "
            "nonsense = 0 and render residue x position heatmaps."
        ),
    )
    parser.add_argument(
        "-i", "--input", type=Path,
        help=helptext("input", "dataset file, or folder searched recursively"),
    )
    parser.add_argument(
        "-o", "--output", type=Path, help=helptext("output", "output folder")
    )
    parser.add_argument(
        "-c", "--config", type=Path, help="TOML config file with any of these settings"
    )
    parser.add_argument(
        "--patterns", nargs="+", metavar="GLOB",
        help=helptext("patterns", "filename globs to treat as datasets"),
    )

    group = parser.add_argument_group("quality filtering")
    group.add_argument(
        "--min-input-count", type=int, metavar="N",
        help=helptext(
            "min_input_count",
            "drop variants with fewer than N input reads in any replicate, "
            "before the anchors are computed; 0 disables filtering",
        ),
    )

    group = parser.add_argument_group("normalisation")
    group.add_argument(
        "--anchor-statistic", choices=("median", "mean"),
        help=helptext("anchor_statistic", "how the wild-type and nonsense anchors are summarised"),
    )
    group.add_argument(
        "--min-replicates", type=int, metavar="N",
        help=helptext("min_replicates", "blank variants measured in fewer than N replicates"),
    )

    group = parser.add_argument_group("colour scale")
    group.add_argument(
        "--lower-quantile", type=float, metavar="Q",
        help=helptext("lower_quantile", "values at or below this quantile all take the deepest red"),
    )
    group.add_argument(
        "--upper-quantile", type=float, metavar="Q",
        help=helptext("upper_quantile", "values at or above this quantile all take the deepest blue"),
    )
    group.add_argument(
        "--center", metavar="VALUE",
        help=helptext(
            "center",
            "fitness placed at the neutral midpoint: 'wt' (1.0), a number, "
            "or 'none' for a plain red-to-blue ramp between the limits",
        ),
    )
    group.add_argument(
        "--shared-scale", action="store_true", default=None,
        help=helptext("shared_scale", "one set of colour limits across every dataset in the run"),
    )
    group.add_argument(
        "--per-dataset-scale", dest="shared_scale", action="store_false",
        help="colour limits from each dataset's own distribution (default)",
    )

    group = parser.add_argument_group("figure")
    group.add_argument(
        "--block-size", type=int, metavar="N",
        help=helptext("block_size", "positions per stacked block"),
    )
    group.add_argument(
        "--aa-order", choices=sorted(AA_ORDERS), help=helptext("aa_order", "row order")
    )
    group.add_argument(
        "--tick-every", type=int, metavar="N",
        help=helptext("tick_every", "position axis tick interval")
    )
    group.add_argument(
        "--no-stops", dest="include_stops", action="store_false", default=None,
        help="omit the nonsense row from the heatmap",
    )
    group.add_argument(
        "--no-wt-marks", dest="mark_wild_type", action="store_false", default=None,
        help="do not dot the wild-type residue in each column",
    )
    group.add_argument(
        "--formats", nargs="+", metavar="EXT",
        help=helptext("formats", "figure formats to write")
    )
    group.add_argument(
        "--no-tables", dest="write_tables", action="store_false", default=None,
        help="render figures only, skip the normalised output tables",
    )

    parser.add_argument(
        "-v", "--verbose", action="store_true", help="debug level logging"
    )
    parser.add_argument(
        "-q", "--quiet", action="store_true", help="warnings and errors only"
    )
    return parser


def _parse_center(value) -> float | None:
    if value is None:
        return None
    text = str(value).strip().lower()
    if text in {"none", "off", "linear"}:
        return None
    if text in {"wt", "wild-type", "wildtype"}:
        return 1.0
    if text in {"stop", "null"}:
        return 0.0
    try:
        return float(text)
    except ValueError as exc:
        raise SystemExit(f"--center: expected 'wt', 'none' or a number, got {value!r}") from exc


def _load_config_file(path: Path) -> dict:
    with Path(path).open("rb") as handle:
        data = tomllib.load(handle)
    # Accept either a flat table or everything under [dms_heatmap].
    return data.get("dms_heatmap", data)


def config_from_args(args: argparse.Namespace) -> PipelineConfig:
    """Merge defaults, config file and command line into one config."""
    settings: dict = {}
    valid = set(PipelineConfig().__dict__)

    if getattr(args, "config", None):
        for key, value in _load_config_file(args.config).items():
            key = key.replace("-", "_")
            if key not in valid:
                raise SystemExit(f"{args.config}: unknown setting {key!r}")
            settings[key] = value

    for key, value in vars(args).items():
        if key.startswith("_") or key in {"config", "verbose", "quiet"}:
            continue
        if value is None or key not in valid:
            continue
        settings[key] = value

    if "center" in settings:
        settings["center"] = _parse_center(settings["center"])
    for key in _PATH_FIELDS & settings.keys():
        settings[key] = Path(settings[key])
    for key in _TUPLE_FIELDS & settings.keys():
        settings[key] = tuple(settings[key])

    return PipelineConfig(**settings)


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    level = logging.DEBUG if args.verbose else logging.WARNING if args.quiet else logging.INFO
    logging.basicConfig(level=level, format="%(levelname)-8s %(message)s", stream=sys.stderr)

    config = config_from_args(args)
    try:
        results = run(config)
    except (DatasetError, NormalisationError, ValueError) as exc:
        logging.getLogger("dms_heatmap").error("%s", exc)
        return 1

    logging.getLogger("dms_heatmap").info(
        "done: %d dataset(s) -> %s", len(results), config.output
    )
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
