"""Loading, column canonicalisation, validation and discovery."""

from __future__ import annotations

import pandas as pd
import pytest

from dms_heatmap.dataset import (
    DatasetError,
    canonical_columns,
    dataset_name,
    discover,
    filter_by_input_count,
    load,
)

from .conftest import WT_SEQUENCE


def test_canonical_columns_normalises_spelling():
    mapping = canonical_columns(["wt aa", "mut aa", "mean fitness", "  nt_ham "])
    assert mapping["wt aa"] == "wt_aa"
    assert mapping["mut aa"] == "mut_aa"
    assert mapping["mean fitness"] == "mean_fitness"
    assert mapping["  nt_ham "] == "nt_ham"


@pytest.mark.parametrize(
    "filename,expected",
    [
        ("tna1_fitness_estimation.tsv", "tna1"),
        ("PMA1_fitness.tsv", "PMA1"),
        ("sub/dir/ste2_dms.csv", "ste2"),
        ("oddly_named.tsv", "oddly_named"),
    ],
)
def test_dataset_name(tmp_path, filename, expected):
    assert dataset_name(tmp_path / filename) == expected


def test_load_reads_replicates_and_sequence(synthetic_file):
    dataset = load(synthetic_file)
    assert dataset.name == "gene1"
    assert dataset.rep_columns == ("raw_fitness_rep1", "raw_fitness_rep2")
    assert dataset.value_kind == "raw"
    assert dataset.wt_sequence == WT_SEQUENCE
    # Per-row sequence columns are dropped; the rest survive canonicalised.
    assert "aa_seq" not in dataset.table.columns
    assert {"wt_aa", "pos", "mut_aa", "aa_ham"} <= set(dataset.table.columns)
    assert dataset.table["pos"].dtype.kind == "f"


def test_load_prefers_raw_but_falls_back_to_rescaled(tmp_path, synthetic_frame):
    frame = synthetic_frame.rename(
        columns={
            "raw_fitness_rep1": "rescaled_fitness_rep1",
            "raw_fitness_rep2": "rescaled_fitness_rep2",
        }
    )
    path = tmp_path / "gene2_fitness_estimation.tsv"
    frame.to_csv(path, sep="\t", index=False)

    dataset = load(path)
    assert dataset.value_kind == "rescaled"
    assert dataset.rep_columns == ("rescaled_fitness_rep1", "rescaled_fitness_rep2")


def test_load_rejects_a_table_with_no_fitness_columns(tmp_path, synthetic_frame):
    frame = synthetic_frame.drop(columns=["raw_fitness_rep1", "raw_fitness_rep2"])
    path = tmp_path / "broken_fitness_estimation.tsv"
    frame.to_csv(path, sep="\t", index=False)

    with pytest.raises(DatasetError, match="no replicate fitness columns"):
        load(path)


def test_load_rejects_a_table_missing_required_columns(tmp_path, synthetic_frame):
    path = tmp_path / "broken2_fitness_estimation.tsv"
    synthetic_frame.drop(columns=["aa_ham"]).to_csv(path, sep="\t", index=False)

    with pytest.raises(DatasetError, match="missing required column"):
        load(path)


def test_load_drops_variant_rows_with_no_position(tmp_path, synthetic_frame):
    frame = pd.concat(
        [
            synthetic_frame,
            pd.DataFrame([{**synthetic_frame.iloc[-1].to_dict(), "pos": None}]),
        ],
        ignore_index=True,
    )
    path = tmp_path / "gene3_fitness_estimation.tsv"
    frame.to_csv(path, sep="\t", index=False)

    dataset = load(path)
    assert len(dataset.table) == len(synthetic_frame)
    assert any("no pos/mut_aa" in note for note in dataset.notes)


def test_discover_is_recursive_and_sorted(tmp_path):
    (tmp_path / "sub").mkdir()
    made = [tmp_path / "b.tsv", tmp_path / "sub" / "a.csv", tmp_path / "sub" / "c.tsv"]
    for path in made:
        path.write_text("x\n")
    (tmp_path / "notes.md").write_text("ignore me\n")

    assert discover(tmp_path) == sorted(made)


def test_discover_skips_pipeline_output(tmp_path):
    (tmp_path / "real.tsv").write_text("x\n")
    (tmp_path / "gene.normalised.tsv").write_text("x\n")
    (tmp_path / "gene.matrix.tsv").write_text("x\n")
    (tmp_path / "summary.tsv").write_text("x\n")

    assert discover(tmp_path) == [tmp_path / "real.tsv"]


def test_discover_accepts_a_single_file(synthetic_file):
    assert discover(synthetic_file) == [synthetic_file]


def test_discover_rejects_a_missing_path(tmp_path):
    with pytest.raises(DatasetError, match="does not exist"):
        discover(tmp_path / "nope")


def test_filter_by_input_count_uses_the_lowest_replicate(synthetic_file):
    table = load(synthetic_file).table
    kept, dropped = filter_by_input_count(table, 20)

    # Exactly the planted (3, 400) row falls below the threshold.
    assert dropped == 1
    assert len(kept) == len(table) - 1
    assert not (kept["input1"].astype(int) < 20).any()


def test_filter_by_input_count_is_a_no_op_at_zero(synthetic_file):
    table = load(synthetic_file).table
    kept, dropped = filter_by_input_count(table, 0)
    assert dropped == 0
    assert len(kept) == len(table)
