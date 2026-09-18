"""End-to-end runs, CLI argument merging and the outputs on disk."""

from __future__ import annotations

import json
import shutil

import pandas as pd
import pytest
from matplotlib.image import imread

from dms_heatmap.cli import build_parser, config_from_args, main
from dms_heatmap.dataset import DatasetError
from dms_heatmap.pipeline import PipelineConfig, run

from .conftest import WT_SEQUENCE
from .test_structure import dssp_text


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
        assert (config.output / "figures" / f"{name}.distribution.png").exists()
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


def test_distribution_can_be_switched_off(folder, tmp_path):
    config = _config(folder, tmp_path, distribution=False)
    run(config)
    assert (config.output / "figures" / "gene1.png").exists()
    assert not (config.output / "figures" / "gene1.distribution.png").exists()


def test_manifest_records_the_class_distribution(folder, tmp_path):
    config = _config(folder, tmp_path)
    run(config)

    manifest = json.loads((config.output / "run_manifest.json").read_text())
    classes = next(d for d in manifest["datasets"] if d["dataset"] == "gene1")[
        "class_distribution"
    ]
    assert set(classes) == {"missense", "synonymous", "nonsense"}
    assert classes["synonymous"]["median"] == pytest.approx(1.0)
    assert classes["nonsense"]["median"] == pytest.approx(0.0)


def test_summary_carries_the_normalisation_sanity_check(folder, tmp_path):
    config = _config(folder, tmp_path)
    run(config)
    summary = pd.read_csv(config.output / "summary.tsv", sep="\t")
    assert summary["synonymous_median"].eq(1.0).all()
    assert summary["nonsense_median"].eq(0.0).all()


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


# --- secondary structure strips -------------------------------------------


@pytest.fixture
def structures(tmp_path):
    """A structure for gene1 only, so the unmatched case is exercised too."""
    folder = tmp_path / "structures"
    folder.mkdir()
    (folder / "gene1.dssp").write_text(dssp_text("LLHHHHEEEL"))
    return folder


def test_structure_strips_are_drawn_and_recorded(folder, tmp_path, structures):
    config = _config(folder, tmp_path, structures=structures)
    results = run(config)

    gene1 = next(r for r in results if r.name == "gene1")
    assert gene1.structure is not None
    assert [t.key for t in gene1.tracks] == ["substitution_P", "position_mean"]

    manifest = json.loads((config.output / "run_manifest.json").read_text())
    recorded = next(d for d in manifest["datasets"] if d["dataset"] == "gene1")[
        "secondary_structure"
    ]
    assert recorded["n_residues"] == 10
    assert recorded["composition"] == {"H": 0.4, "E": 0.3, "L": 0.3}
    assert recorded["offset"] == 0
    assert recorded["sequence_identity"] == pytest.approx(1.0)
    assert set(recorded["tracks"]) == {"substitution_P", "position_mean"}

    # gene2 has no structure file, and that is not a failure.
    gene2 = next(r for r in results if r.name == "gene2")
    assert gene2.structure is None
    assert (config.output / "figures" / "gene2.png").exists()


def test_the_strips_make_the_figure_taller_not_wider(folder, tmp_path, structures):
    plain = run(_config(folder, tmp_path / "plain"))
    with_strips = run(_config(folder, tmp_path / "strips", structures=structures))
    assert plain and with_strips

    bare = imread(tmp_path / "plain" / "out" / "figures" / "gene1.png")
    annotated = imread(tmp_path / "strips" / "out" / "figures" / "gene1.png")
    assert annotated.shape[0] > bare.shape[0]   # taller
    assert annotated.shape[1] == bare.shape[1]  # same width


def test_the_numbers_behind_the_strips_are_written_out(folder, tmp_path, structures):
    config = _config(folder, tmp_path, structures=structures)
    run(config)

    table = pd.read_csv(config.output / "tables" / "gene1.structure.tsv", sep="\t")
    assert list(table.columns) == [
        "pos", "wt_aa", "secondary_structure", "substitution_P", "position_mean",
    ]
    assert len(table) == 10
    assert "".join(table["secondary_structure"]) == "LLHHHHEEEL"
    # The one assayed proline substitution, at position 7.
    assert table.loc[table["pos"] == 7, "substitution_P"].iloc[0] == pytest.approx(0.25)


def test_the_strip_residue_can_be_changed(folder, tmp_path, structures):
    config = _config(folder, tmp_path, structures=structures, structure_residue="W")
    results = run(config)
    gene1 = next(r for r in results if r.name == "gene1")
    assert [t.key for t in gene1.tracks] == ["substitution_W", "position_mean"]


def test_strips_can_be_switched_off(folder, tmp_path, structures):
    config = _config(
        folder, tmp_path, structures=structures, structure_tracks=False
    )
    results = run(config)
    assert all(r.structure is None for r in results)
    assert not (config.output / "tables" / "gene1.structure.tsv").exists()


def test_an_unreadable_structure_does_not_lose_the_figure(folder, tmp_path):
    broken = tmp_path / "structures"
    broken.mkdir()
    (broken / "gene1.dssp").write_text("this is not a dssp file\n")
    config = _config(folder, tmp_path, structures=broken)

    results = run(config)
    gene1 = next(r for r in results if r.name == "gene1")
    assert gene1.structure is None
    assert (config.output / "figures" / "gene1.png").exists()
    assert any("no secondary-structure strips" in w for w in gene1.warnings)


def test_a_structure_of_the_wrong_protein_is_refused_not_drawn(folder, tmp_path):
    wrong = tmp_path / "structures"
    wrong.mkdir()
    (wrong / "gene1.dssp").write_text(dssp_text("HHHHHHHHHH", sequence="CCCCCCCCCC"))
    config = _config(folder, tmp_path, structures=wrong)

    gene1 = next(r for r in run(config) if r.name == "gene1")
    assert gene1.structure is None
    assert any("cannot place" in w for w in gene1.warnings)


def test_a_shifted_structure_is_placed_by_its_sequence(folder, tmp_path):
    # A model of residues 4-10 numbering them 1-7: drawn at face value the
    # strips would sit three positions to the left of the biology.
    shifted = tmp_path / "structures"
    shifted.mkdir()
    (shifted / "gene1.dssp").write_text(
        dssp_text("HHHEEEL", sequence=WT_SEQUENCE[3:])
    )
    config = _config(folder, tmp_path, structures=shifted)

    gene1 = next(r for r in run(config) if r.name == "gene1")
    assert gene1.structure.offset == 3
    assert gene1.structure.code_map()[4] == "H"


def test_placing_a_structure_is_not_counted_as_a_warning(folder, tmp_path, structures):
    config = _config(folder, tmp_path, structures=structures)
    run(config)
    summary = pd.read_csv(config.output / "summary.tsv", sep="\t")
    assert summary.loc[summary["dataset"] == "gene1", "n_warnings"].iloc[0] == 0


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


def test_asking_for_stops_in_the_mean_without_a_stop_row_is_flagged(
    folder, tmp_path, structures
):
    # --no-stops leaves the nonsense row out of the grid, so there is nothing
    # to average in; the caption must not claim otherwise.
    config = _config(
        folder,
        tmp_path,
        structures=structures,
        include_stops=False,
        structure_mean_include_stops=True,
    )
    gene1 = next(r for r in run(config) if r.name == "gene1")
    mean = next(t for t in gene1.tracks if t.key == "position_mean")

    assert "nonsense excluded" in mean.description
    assert any("has no effect with include_stops off" in w for w in gene1.warnings)


def test_summary_surfaces_the_numbering_offset(folder, tmp_path, structures):
    config = _config(folder, tmp_path, structures=structures)
    run(config)

    summary = pd.read_csv(config.output / "summary.tsv", sep="\t")
    gene1 = summary.loc[summary["dataset"] == "gene1"].iloc[0]
    assert gene1["structure_offset"] == 0
    assert gene1["structure_identity"] == pytest.approx(1.0)
    # gene2 has no structure, so the columns are empty rather than zero.
    assert pd.isna(summary.loc[summary["dataset"] == "gene2", "structure_offset"]).all()
