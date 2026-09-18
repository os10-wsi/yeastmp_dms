"""The normalisation maths, against a fixture with hand-placed anchors."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from dms_heatmap.dataset import load
from dms_heatmap.normalise import (
    NormalisationError,
    nonsense_mask,
    normalise,
    replicate_anchors,
)

from .conftest import REP_COLUMNS, STOP_ANCHOR, WT_ANCHOR


@pytest.fixture
def table(synthetic_file):
    return load(synthetic_file).table


def test_anchors_land_on_the_planted_values(table):
    for column in REP_COLUMNS:
        anchors = replicate_anchors(table, column)
        assert anchors.wt == pytest.approx(WT_ANCHOR[column])
        assert anchors.stop == pytest.approx(STOP_ANCHOR[column])
        assert anchors.n_wt == 5
        assert anchors.n_stop == 7


def test_median_anchor_ignores_the_tolerated_stop_tail(table):
    """The fixture plants one stop at +3.5; a mean anchor moves, a median does not."""
    column = REP_COLUMNS[0]
    assert replicate_anchors(table, column, "median").stop == pytest.approx(
        STOP_ANCHOR[column]
    )
    assert replicate_anchors(table, column, "mean").stop > STOP_ANCHOR[column] + 0.3


def test_wild_type_maps_to_one_and_nonsense_to_zero(table):
    out, anchors, _ = normalise(table, REP_COLUMNS)
    for anchor in anchors:
        assert anchor.rescale(pd.Series([anchor.wt])).iloc[0] == pytest.approx(1.0)
        assert anchor.rescale(pd.Series([anchor.stop])).iloc[0] == pytest.approx(0.0)

    wild_type = out.loc[out["aa_ham"] == 0, "fitness"]
    nonsense = out.loc[out["mut_aa"] == "*", "fitness"]
    assert np.median(wild_type) == pytest.approx(1.0)
    assert np.median(nonsense) == pytest.approx(0.0)


def test_planted_missense_values_come_back_exactly(table):
    out, _, _ = normalise(table, REP_COLUMNS)
    # Select on the (position, substitution) pair: stop rows share these positions.
    planted = [
        (1, "A", 0.0),
        (2, "A", 0.5),
        (3, "A", 1.0),
        (4, "A", 1.5),
        (5, "D", -0.25),
        (6, "W", 0.75),
        (7, "P", 0.25),
    ]
    for pos, mut, expected in planted:
        hit = out.loc[(out["pos"] == pos) & (out["mut_aa"] == mut), "fitness"]
        assert len(hit) == 1
        assert hit.iloc[0] == pytest.approx(expected)


def test_nothing_is_clamped(table):
    """Values outside [0, 1] survive; only the colour scale saturates."""
    out, _, _ = normalise(table, REP_COLUMNS)
    assert out["fitness"].max() > 1.0
    assert out["fitness"].min() < 0.0


def test_replicates_are_anchored_independently(table):
    """The two replicates have different dynamic ranges (2.0 and 2.0 offset by 1).

    A variant placed at the same *fraction* in both must normalise to that
    fraction, which only holds if each replicate is rescaled on its own
    anchors before averaging.
    """
    out, _, _ = normalise(table, REP_COLUMNS)
    row = out.loc[(out["pos"] == 2) & (out["mut_aa"] == "A")].iloc[0]
    assert row["norm_raw_fitness_rep1"] == pytest.approx(0.5)
    assert row["norm_raw_fitness_rep2"] == pytest.approx(0.5)
    assert row["fitness"] == pytest.approx(0.5)
    assert row["fitness_sd"] == pytest.approx(0.0, abs=1e-12)


def test_replicate_counts_and_sd(table):
    out, _, _ = normalise(table, REP_COLUMNS)
    single = out.loc[out["pos"] == 8].iloc[0]
    assert single["n_replicates"] == 1
    assert np.isnan(single["fitness_sd"])
    assert not np.isnan(single["fitness"])


def test_min_replicates_blanks_thinly_supported_variants(table):
    out, _, warnings = normalise(table, REP_COLUMNS, min_replicates=2)
    single = out.loc[out["pos"] == 8].iloc[0]
    assert np.isnan(single["fitness"])
    assert single["n_replicates"] == 1
    assert any("fewer than 2 replicates" in w for w in warnings)


def test_missing_anchor_group_is_an_error(table):
    with pytest.raises(NormalisationError, match="nonsense"):
        replicate_anchors(table.loc[~nonsense_mask(table)], REP_COLUMNS[0])
    with pytest.raises(NormalisationError, match="wild-type"):
        replicate_anchors(table.loc[table["aa_ham"] != 0], REP_COLUMNS[0])


def test_inverted_anchors_are_an_error(table):
    """Nonsense fitter than wild type means the scale is meaningless, not flipped."""
    flipped = table.copy()
    column = REP_COLUMNS[0]
    flipped[column] = np.where(flipped["aa_ham"] == 0, -9.0, flipped[column])
    with pytest.raises(NormalisationError, match="not above"):
        replicate_anchors(flipped, column)


def test_few_anchor_observations_warn_rather_than_fail(table):
    trimmed = pd.concat(
        [
            table.loc[table["aa_ham"] == 0].head(2),
            table.loc[table["mut_aa"] == "*"],
            table.loc[(table["aa_ham"] == 1) & (table["mut_aa"] != "*")],
        ]
    )
    _, _, warnings = normalise(trimmed, REP_COLUMNS)
    assert any("few observations" in w for w in warnings)
