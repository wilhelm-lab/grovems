"""Tier 0 handling: distance-0 pairs must never be reported as Tier 1.

Two defects are covered here. ``_get_tier`` had no Tier 0 branch, so a distance of 0 fell
through to the first upper bound (``< 4``) and came back as Tier 1. And ``_add_psa_columns``
gated the classifier on ``sequence_match`` (modification- and charge-aware) while handing it
the *unmodified* sequences, so a pair differing only in its modifications reached the
classifier as two identical strings and was labelled from them.
"""

import pandas as pd

from grovems.psa.psa import PSA
from grovems.psa.psa_merge import _add_psa_columns, add_sequence_match_columns
from grovems.psa.psa_utils import UtilsMixin


def test_get_tier_returns_tier_zero_for_distance_zero():
    """A distance of 0 is Tier 0, and the other boundaries are left where they were."""
    assert UtilsMixin._get_tier(0) == "Tier 0"
    assert UtilsMixin._get_tier(1) == "Tier 1"
    assert UtilsMixin._get_tier(3) == "Tier 1"
    assert UtilsMixin._get_tier(4) == "Tier 2"


def test_classify_identical_sequences_is_tier_zero():
    """The classifier itself, not just the merge shortcut, reports Tier 0 for a match."""
    psa = PSA("PEPTIDEK", "PEPTIDEK")
    psa.classify()

    assert psa.result.levenshtein_distance == 0
    assert psa.result.label.startswith("PSA - Tier 0 - ")


def _merged_frame(rows):
    frame = pd.DataFrame(rows)
    add_sequence_match_columns(frame)
    return frame


def test_dispatch_splits_shared_rows_three_ways():
    """Identical peptide, same residues with different mods, and different residues."""
    frame = _merged_frame(
        [
            {  # same peptide on both sides
                "_merge": "shared",
                "SEQUENCE_database": "PEPTIDEK",
                "SEQUENCE_denovo": "PEPTIDEK",
                "MODIFIED_SEQUENCE_database": "PEPTIDEK",
                "MODIFIED_SEQUENCE_denovo": "PEPTIDEK",
                "PRECURSOR_CHARGE_database": 2,
                "PRECURSOR_CHARGE_denovo": 2,
            },
            {  # same residues, oxidation placed on the other methionine
                "_merge": "shared",
                "SEQUENCE_database": "MMLVLPR",
                "SEQUENCE_denovo": "MMLVLPR",
                "MODIFIED_SEQUENCE_database": "M[UNIMOD:35]MLVLPR",
                "MODIFIED_SEQUENCE_denovo": "MM[UNIMOD:35]LVLPR",
                "PRECURSOR_CHARGE_database": 2,
                "PRECURSOR_CHARGE_denovo": 2,
            },
            {  # same residues, one oxidation fewer -- 16 Da apart, not isobaric
                "_merge": "shared",
                "SEQUENCE_database": "EGQMESVEAAMSSK",
                "SEQUENCE_denovo": "EGQMESVEAAMSSK",
                "MODIFIED_SEQUENCE_database": "EGQM[UNIMOD:35]ESVEAAM[UNIMOD:35]SSK",
                "MODIFIED_SEQUENCE_denovo": "EGQMESVEAAM[UNIMOD:35]SSK",
                "PRECURSOR_CHARGE_database": 2,
                "PRECURSOR_CHARGE_denovo": 2,
            },
            {  # genuinely different residues -- the classifier's job
                "_merge": "shared",
                "SEQUENCE_database": "PEPTIDEK",
                "SEQUENCE_denovo": "PEPTIEDK",
                "MODIFIED_SEQUENCE_database": "PEPTIDEK",
                "MODIFIED_SEQUENCE_denovo": "PEPTIEDK",
                "PRECURSOR_CHARGE_database": 2,
                "PRECURSOR_CHARGE_denovo": 2,
            },
        ]
    )

    _add_psa_columns(frame)

    assert frame.loc[0, "PSA"] == "PSA - Tier 0 - IDENTICAL"
    assert frame.loc[1, "PSA"] == "PSA - Tier 0 - IDENTICAL_UNMODIFIED"
    assert frame.loc[2, "PSA"] == "PSA - Tier 0 - IDENTICAL_UNMODIFIED"
    assert frame.loc[3, "PSA"].startswith("PSA - Tier 1 - ")

    assert list(frame["PSA_LEVENSHTEIN"]) == [0, 0, 0, 2]

    # the modification-only rows must not claim isobaricity PSA cannot judge; row 2 is
    # 16 Da apart and used to come back ISOBARIC because both masses were computed from
    # the same unmodified string
    assert "ISOBARIC" not in frame.loc[1, "PSA"]
    assert "ISOBARIC" not in frame.loc[2, "PSA"]


def test_dispatch_covers_every_shared_row_exactly_once():
    """The three masks must partition the shared rows -- no row unlabelled or overwritten."""
    frame = _merged_frame(
        [
            {
                "_merge": merge,
                "SEQUENCE_database": database,
                "SEQUENCE_denovo": denovo,
                "MODIFIED_SEQUENCE_database": database,
                "MODIFIED_SEQUENCE_denovo": denovo,
                "PRECURSOR_CHARGE_database": charge_database,
                "PRECURSOR_CHARGE_denovo": charge_denovo,
            }
            for merge, database, denovo, charge_database, charge_denovo in [
                ("shared", "PEPTIDEK", "PEPTIDEK", 2, 2),
                ("shared", "PEPTIDEK", "PEPTIDEK", 2, 3),  # charge-only mismatch
                ("shared", "PEPTIDEK", "PEPTIEDK", 2, 2),
                ("database_only", "PEPTIDEK", None, 2, 2),
                ("denovo_only", None, "PEPTIDEK", 2, 2),
            ]
        ]
    )

    _add_psa_columns(frame)

    shared = frame["_merge"].eq("shared")
    assert frame.loc[shared, "PSA"].notna().all()
    assert frame.loc[~shared, "PSA"].isna().all()

    # a charge-only mismatch is the same peptide read at a different charge: Tier 0, and
    # it must not be silently promoted to "IDENTICAL"
    assert frame.loc[1, "PSA"] == "PSA - Tier 0 - IDENTICAL_UNMODIFIED"
