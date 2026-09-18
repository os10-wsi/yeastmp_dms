"""End-to-end runs, CLI argument merging and the outputs on disk."""

from __future__ import annotations

import json
import shutil

import pandas as pd
import pytest

from dms_heatmap.cli import build_parser, config_from_args, main
from dms_heatmap.dataset import DatasetError
from dms_heatmap.pipeline import PipelineConfig, run


@pytest.fixture
def folder(tmp_path, synthetic_file):
    """Two datasets, one of them in a subfolder, to exercise the recursion."""
    root = tmp_path / "datasets"
    (root / "nested").mkdir(parents=True)
    shutil.copy(synthetic_file, root / "gene1_fitness_estimation.tsv")
    shutil.copy(synthetic_file, root / "nested" / "gene2_fitness_estimation.tsv")
    return root


def _config(folder, tmp_path, **overrides):
    return PipelineConfig(
        input=folder, output=tmp_path / "out", formats=("png",), **overrides
    )


def test_run_processes_every_dataset_in_the_tree(folder, tmp_path):
    config = _config(folder, tmp_path)
    results = run(config)

    assert sorted(r.name for r in results) == ["gene1", "gene2"]
    for name in ("gene1", "gene2"):
        assert (config.output / "figures" / f"{name}.png").exists()
        assert (config.output / "tables" / f"{name}.normalised.tsv").exists()
        assert (config.output / "tables" / f"{name}.matrix.tsv").exists()
    assert (config.output / "summary.tsv").exists()
    assert (config.output / "run_manifest.json").exists()


def test_normalised_table_has_the_expected_columns(folder, tmp_path):
    config = _config(folder, tmp_path)
    run(config)

    table = pd.read_csv(config.output / "tables" / "gene1.normalised.tsv", sep="\t")
    assert {"pos", "mut_aa", "fitness", "fitness_sd", "n_replicates"} <= set(table.columns)
    assert {"norm_raw_fitness_rep1", "norm_raw_fitness_rep2"} <= set(table.columns)
    # Wild type lands on 1 and nonsense on 0, as written to disk.
    assert table.loc[table["aa_ham"] == 0, "fitness"].median() == pytest.approx(1.0)
    assert table.loc[table["mut_aa"] == "*", "fitness"].median() == pytest.approx(0.0)


def test_matrix_table_is_the_grid(folder, tmp_path):
    config = _config(folder, tmp_path)
    run(config)

    grid = pd.read_csv(config.output / "tables" / "gene1.matrix.tsv", sep="\t", index_col=0)
    assert grid.shape[0] == 21
    assert grid.loc["A", "2"] == pytest.approx(0.5)


def test_manifest_records_provenance(folder, tmp_path):
    config = _config(folder, tmp_path, min_input_count=20)
    run(config)

    manifest = json.loads((config.output / "run_manifest.json").read_text())
    assert manifest["config"]["min_input_count"] == 20
    assert manifest["config"]["center"] == 1.0
    assert set(manifest["packages"]) == {"numpy", "pandas", "matplotlib"}

    dataset = next(d for d in manifest["datasets"] if d["dataset"] == "gene1")
    assert len(dataset["anchors"]) == 2
    assert dataset["anchors"][0]["n_wt"] > 0
    assert dataset["colour_scale"]["lower_quantile"] == 0.10
    assert "q0.50" in dataset["fitness_quantiles"]
    assert dataset["read_depth"]["fraction_retained"]["20"] < 1.0
    assert any("below 20" in w for w in dataset["warnings"])


def test_shared_scale_gives_every_dataset_the_same_limits(folder, tmp_path):
    results = run(_config(folder, tmp_path, shared_scale=True))
    limits = {(r.scale.vmin, r.scale.vmax) for r in results}
    assert len(limits) == 1


def test_a_broken_file_is_skipped_not_fatal(folder, tmp_path):
    (folder / "junk_fitness_estimation.tsv").write_text("nothing\tuseful\n1\t2\n")
    config = _config(folder, tmp_path)

    results = run(config)
    assert sorted(r.name for r in results) == ["gene1", "gene2"]

    manifest = json.loads((config.output / "run_manifest.json").read_text())
    assert len(manifest["failures"]) == 1
    assert "junk" in manifest["failures"][0]["source"]


def test_an_empty_folder_is_an_error(tmp_path):
    empty = tmp_path / "empty"
    empty.mkdir()
    with pytest.raises(DatasetError, match="no dataset files"):
        run(PipelineConfig(input=empty, output=tmp_path / "out"))


def test_no_tables_writes_figures_only(folder, tmp_path):
    config = _config(folder, tmp_path, write_tables=False)
    run(config)
    assert (config.output / "figures" / "gene1.png").exists()
    assert not (config.output / "tables").exists()


def test_summary_has_one_row_per_dataset(folder, tmp_path):
    config = _config(folder, tmp_path)
    run(config)
    summary = pd.read_csv(config.output / "summary.tsv", sep="\t")
    assert len(summary) == 2
    assert {"dataset", "coverage", "vmin", "vmax", "min_replicate_r"} <= set(summary.columns)


# --- CLI ------------------------------------------------------------------


def test_cli_defaults_match_the_dataclass():
    config = config_from_args(build_parser().parse_args([]))
    assert config == PipelineConfig()


@pytest.mark.parametrize(
    "text,expected", [("wt", 1.0), ("none", None), ("0.5", 0.5), ("stop", 0.0)]
)
def test_cli_parses_center(text, expected):
    config = config_from_args(build_parser().parse_args(["--center", text]))
    assert config.center == expected


def test_cli_rejects_a_nonsense_center():
    with pytest.raises(SystemExit):
        config_from_args(build_parser().parse_args(["--center", "blueish"]))


def test_cli_flags_override_the_config_file(tmp_path):
    config_file = tmp_path / "settings.toml"
    config_file.write_text(
        '[dms_heatmap]\nblock_size = 42\ncenter = "none"\nmin_input_count = 7\n'
    )
    args = build_parser().parse_args(
        ["--config", str(config_file), "--block-size", "13"]
    )
    config = config_from_args(args)

    assert config.block_size == 13      # command line wins
    assert config.min_input_count == 7  # file value kept
    assert config.center is None


def test_cli_rejects_unknown_settings_in_a_config_file(tmp_path):
    config_file = tmp_path / "settings.toml"
    config_file.write_text("[dms_heatmap]\ncolour = 'purple'\n")
    args = build_parser().parse_args(["--config", str(config_file)])
    with pytest.raises(SystemExit, match="unknown setting"):
        config_from_args(args)


def test_main_runs_and_returns_zero(folder, tmp_path):
    out = tmp_path / "cli-out"
    code = main(
        ["--input", str(folder), "--output", str(out), "--formats", "png", "--quiet"]
    )
    assert code == 0
    assert (out / "figures" / "gene1.png").exists()


def test_main_reports_failure_without_traceback(tmp_path):
    empty = tmp_path / "empty"
    empty.mkdir()
    assert main(["--input", str(empty), "--output", str(tmp_path / "o"), "--quiet"]) == 1
