from grovems.psa.psa import PSA


def _classify(sequence1: str, sequence2: str) -> PSA:
    psa = PSA(sequence1, sequence2)
    psa.classify()
    return psa


def test_two_independent_shuffles_both_detected():
    psa = _classify("PEPTIDAEPGRWHKFEK", "PEPIATDEPGRHWFKEK")
    assert psa.result.details["label_event_name"] == "SHUFFLE+SHUFFLE"
    assert psa.result.details["change_summary"] == "SHUFFLE(TIDA->IATD)@4+SHUFFLE(WHKF->HWFK)@12"


def test_two_independent_swaps_both_detected():
    psa = _classify("AKDEITFGHKMPQ", "ADKEITFGHMKPQ")
    assert psa.result.details["label_event_name"] == "SWAP+SWAP"


def test_single_shuffle_split_across_a_gap():
    psa = _classify("PEPTIDEK", "PEPIDTEK")
    assert psa.result.details["label_event_name"] == "SHUFFLE"


def test_shuffle_plus_unrelated_insertion():
    psa = _classify("PEPTIDEK", "PEPIDTEIK")
    assert psa.result.details["label_event_name"] == "SHUFFLE+INSERTION"


def test_near_isobaric_substitution():
    psa = _classify("PEPTIKEK", "PEPTIGAEK")
    assert psa.result.details["label_event_name"] == "INSERTION+SUBSTITUTION"


def test_exact_isobaric_substitution_unequal_residue_count():
    # GG <-> N: classic exactly-isobaric compound substitution, needs a 2-column window
    # (one real mismatch column + one gap column) since the two sides differ in length.
    psa = _classify("PEPTGGEK", "PEPTNEK")
    assert psa.result.details["label_event_name"] == "ISOBARIC-SUBSTITUTION"
    assert psa.result.details["change_summary"] == "ISOBARIC-SUBSTITUTION(GG->N)@5"


def test_plain_substitution():
    psa = _classify("PEPTIDEK", "PEPTQDEK")
    assert psa.result.details["label_event_name"] == "SUBSTITUTION"


def test_n_term_mismatch():
    psa = _classify("PEPTIDEK", "AEPTIDEK")
    assert psa.result.details["label_event_name"] == "N-TERM-MISMATCH"


def test_c_term_mismatch():
    psa = _classify("PEPTIDEK", "PEPTIDER")
    assert psa.result.details["label_event_name"] == "C-TERM-MISMATCH"


def test_both_termini_mismatch_combine():
    psa = _classify("PEPTIDEKAR", "AEPTIDEKAK")
    assert psa.result.details["label_event_name"] == "N-TERM-MISMATCH+C-TERM-MISMATCH"


def test_mostly_mismatched_termini_are_not_terminal():
    # Only 5/8 residues match (62.5%, below the 70% floor), so the edge runs are too large a
    # fraction of the sequence to call "terminal" and fall back to SUBSTITUTION.
    psa = _classify("PEPTIDEK", "AAPTIDEA")
    assert psa.result.details["label_event_name"] == "SUBSTITUTION+SUBSTITUTION"


def test_full_mismatch_is_not_terminal():
    psa = _classify("HTVGELLMADR", "TFAHTESHLSK")
    assert psa.result.details["label_event_name"] == "SUBSTITUTION"


def test_middle_substitution_is_not_terminal():
    psa = _classify("PEPTIDEK", "PEPTVDEK")
    assert psa.result.details["label_event_name"] == "SUBSTITUTION"


def test_plain_deletion():
    psa = _classify("PEPTIDEK", "PEPTIEK")
    assert psa.result.details["label_event_name"] == "DELETION"


def test_identical_sequences():
    psa = _classify("PEPTIDEK", "PEPTIDEK")
    assert psa.result.details["label_event_name"] == "IDENTICAL"
