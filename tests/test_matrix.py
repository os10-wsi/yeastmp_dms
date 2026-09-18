"""Reshaping variants into the residue x position grid."""

from __future__ import annotations

import numpy as np
import pytest

from dms_heatmap.dataset import load
from dms_heatmap.matrix import AA_ORDERS, build_matrix
from dms_heatmap.normalise import normalise

from .conftest import REP_COLUMNS, WT_SEQUENCE


@pytest.fixture
def normalised(synthetic_file):
    dataset = load(synthetic_file)
    table, _, _ = normalise(dataset.table, REP_COLUMNS)
    return table, dataset


def test_rows_are_the_twenty_amino_acids_plus_stop(normalised):
    table, dataset = normalised
    matrix = build_matrix(table, "gene1", wt_sequence=dataset.wt_sequence)
    assert len(matrix.residues) == 21
    assert matrix.residues[-1] == "*"
    assert set(matrix.residues[:-1]) == set(AA_ORDERS["chemistry"])


def test_stops_can_be_excluded(normalised):
    table, dataset = normalised
    matrix = build_matrix(
        table, "gene1", wt_sequence=dataset.wt_sequence, include_stops=False
    )
    assert "*" not in matrix.residues
    assert len(matrix.residues) == 20


def test_values_land_in_the_right_cells(normalised):
    table, dataset = normalised
    matrix = build_matrix(table, "gene1", wt_sequence=dataset.wt_sequence)
    assert matrix.values.loc["A", 2] == pytest.approx(0.5)
    assert matrix.values.loc["A", 4] == pytest.approx(1.5)
    assert matrix.values.loc["W", 6] == pytest.approx(0.75)
    assert matrix.values.loc["*", 1] == pytest.approx(0.0, abs=0.2)


def test_unmeasured_cells_are_nan_not_zero(normalised):
    table, dataset = normalised
    matrix = build_matrix(table, "gene1", wt_sequence=dataset.wt_sequence)
    assert np.isnan(matrix.values.loc["Y", 2])


def test_columns_span_the_whole_protein_including_gaps(normalised):
    """Position 10 has no variants but must still be a column."""
    table, dataset = normalised
    matrix = build_matrix(table, "gene1", wt_sequence=dataset.wt_sequence)
    assert list(matrix.positions) == list(range(1, len(WT_SEQUENCE) + 1))
    assert matrix.values[10].isna().all()


def test_wild_type_residues_come_from_the_sequence(normalised):
    table, dataset = normalised
    matrix = build_matrix(table, "gene1", wt_sequence=dataset.wt_sequence)
    for pos, residue in enumerate(WT_SEQUENCE, start=1):
        assert matrix.wt_residues[pos] == residue


def test_wild_type_residues_work_without_a_sequence(normalised):
    """Positions that carry variants are still annotated from the wt_aa column."""
    table, _ = normalised
    matrix = build_matrix(table, "gene1", wt_sequence=None)
    assert matrix.wt_residues[1] == WT_SEQUENCE[0]
    assert matrix.wt_residues[6] == WT_SEQUENCE[5]


def test_coverage_excludes_the_wild_type_diagonal(normalised):
    table, dataset = normalised
    matrix = build_matrix(table, "gene1", wt_sequence=dataset.wt_sequence)
    measured = int(matrix.values.notna().to_numpy().sum())
    possible = matrix.values.size - len(WT_SEQUENCE)
    assert matrix.coverage == pytest.approx(measured / possible)
    assert 0 < matrix.coverage < 1


def test_alphabetical_order(normalised):
    table, dataset = normalised
    matrix = build_matrix(
        table, "gene1", aa_order="alphabetical", wt_sequence=dataset.wt_sequence
    )
    assert matrix.residues[:3] == ["A", "C", "D"]


def test_unknown_order_is_rejected(normalised):
    table, _ = normalised
    with pytest.raises(ValueError, match="unknown aa_order"):
        build_matrix(table, "gene1", aa_order="rainbow")


def test_finite_values_drops_nan(normalised):
    table, dataset = normalised
    matrix = build_matrix(table, "gene1", wt_sequence=dataset.wt_sequence)
    finite = matrix.finite_values()
    assert np.isfinite(finite).all()
    assert finite.size == int(matrix.values.notna().to_numpy().sum())
