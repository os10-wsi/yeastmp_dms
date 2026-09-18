"""A small synthetic dataset with anchors we know exactly.

Building the fixture from known ingredients is what makes the normalisation
tests meaningful: the wild-type and nonsense modes are placed by hand, so the
expected output can be written down rather than read back off the code.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

#: Protein used by the fixture; position i holds WT_SEQUENCE[i - 1].
WT_SEQUENCE = "MKVLAGTWSY"

#: Raw-scale anchors planted in the fixture, per replicate.
WT_ANCHOR = {"raw_fitness_rep1": -2.0, "raw_fitness_rep2": -1.0}
STOP_ANCHOR = {"raw_fitness_rep1": -4.0, "raw_fitness_rep2": -3.0}

REP_COLUMNS = ("raw_fitness_rep1", "raw_fitness_rep2")


def _row(wt_aa, pos, mut_aa, aa_ham, values, inputs=(500, 500)):
    row = {
        "wt aa": wt_aa,
        "pos": pos,
        "mut aa": mut_aa,
        "aa_ham": aa_ham,
        "aa_seq": WT_SEQUENCE,
        "nt_ham": 3,
        "input1": inputs[0],
        "input2": inputs[1],
        "wt": "",
        "stop": "",
    }
    row.update(dict(zip(REP_COLUMNS, values)))
    return row


@pytest.fixture
def synthetic_frame() -> pd.DataFrame:
    """Raw-style table: original column spellings, everything as text-ish.

    The wild-type and nonsense groups are symmetric about their anchors, so
    the median of each lands exactly on the planted value.
    """
    rows = []

    # Wild-type-protein (synonymous) rows, median exactly on WT_ANCHOR.
    # Odd count and symmetric, so the median is the planted value exactly.
    for offset in (-0.5, -0.2, 0.0, 0.2, 0.5):
        rows.append(
            _row(
                None,
                None,
                None,
                0,
                [WT_ANCHOR[c] + offset for c in REP_COLUMNS],
            )
        )
    rows[2]["wt"] = "True"

    # Nonsense rows, median exactly on STOP_ANCHOR, plus one tolerated
    # C-terminal-style stop far above it, so a mean anchor and a median anchor
    # visibly disagree.
    for index, offset in enumerate((-0.3, -0.1, -0.05, 0.0, 0.05, 0.1, 3.5)):
        rows.append(
            _row(
                WT_SEQUENCE[index],
                index + 1,
                "*",
                1,
                [STOP_ANCHOR[c] + offset for c in REP_COLUMNS],
            )
        )

    # Missense variants at known normalised values.
    #   raw = stop + fraction * (wt - stop)  ->  normalises to `fraction`
    plan = [
        (1, "A", 0.0),    # as dead as a stop
        (2, "A", 0.5),    # half way
        (3, "A", 1.0),    # wild-type like
        (4, "A", 1.5),    # fitter than wild type
        (5, "D", -0.25),  # worse than a stop (position 5 is already A)
        (6, "W", 0.75),
        (7, "P", 0.25),
    ]
    for pos, mut, fraction in plan:
        values = [
            STOP_ANCHOR[c] + fraction * (WT_ANCHOR[c] - STOP_ANCHOR[c])
            for c in REP_COLUMNS
        ]
        rows.append(_row(WT_SEQUENCE[pos - 1], pos, mut, 1, values))

    # A variant seen in one replicate only, and one below any read threshold.
    rows.append(_row(WT_SEQUENCE[7], 8, "A", 1, [-3.0, np.nan]))
    rows.append(_row(WT_SEQUENCE[8], 9, "A", 1, [-2.0, -1.0], inputs=(3, 400)))

    return pd.DataFrame(rows)


@pytest.fixture
def synthetic_file(tmp_path, synthetic_frame) -> "pathlib.Path":
    path = tmp_path / "gene1_fitness_estimation.tsv"
    synthetic_frame.to_csv(path, sep="\t", index=False)
    return path
