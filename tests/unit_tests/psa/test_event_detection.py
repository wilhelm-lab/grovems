from grovems.psa.psa import PSA


def _classify(sequence1: str, sequence2: str) -> PSA:
    psa = PSA(sequence1, sequence2)
    psa.classify()
    return psa


def test_two_independent_shuffles_both_detected():
    psa = _classify("PEPTIDAEPGRWHKFEK", "PEPIATDEPGRHWFKEK")
    assert psa.result.selected_event == "SHUFFLE + SHUFFLE"
    assert psa.result.details["change_summary"] == "SHUFFLE(TIDA->IATD)@4 + SHUFFLE(WHKF->HWFK)@12"


def test_two_independent_swaps_both_detected():
    psa = _classify("AKDEITFGHKMPQ", "ADKEITFGHMKPQ")
    assert psa.result.selected_event == "SWAP + SWAP"


def test_single_shuffle_split_across_a_gap():
    psa = _classify("PEPTIDEK", "PEPIDTEK")
    assert psa.result.selected_event == "BLOCK_SHUFFLE"


def test_shuffle_plus_unrelated_insertion():
    psa = _classify("PEPTIDEK", "PEPIDTEIK")
    assert psa.result.selected_event == "SHUFFLE + INSERTION"


def test_near_isobaric_substitution():
    psa = _classify("PEPTIKEK", "PEPTIGAEK")
    assert psa.result.selected_event == "ISOBARIC-SUBSTITUTION"


def test_exact_isobaric_substitution_unequal_residue_count():
    # GG <-> N: classic exactly-isobaric compound substitution, needs a 2-column window
    # (one real mismatch column + one gap column) since the two sides differ in length.
    psa = _classify("PEPTGGEK", "PEPTNEK")
    assert psa.result.selected_event == "ISOBARIC-SUBSTITUTION"
    assert psa.result.details["change_summary"] == "SUB(GG->N)@5"


def test_plain_substitution():
    psa = _classify("PEPTIDEK", "PEPTQDEK")
    assert psa.result.selected_event == "SUBSTITUTION"


def test_n_term_mismatch():
    psa = _classify("PEPTIDEK", "AEPTIDEK")
    assert psa.result.selected_event == "N-TERM-MISMATCH"


def test_c_term_mismatch():
    psa = _classify("PEPTIDEK", "PEPTIDER")
    assert psa.result.selected_event == "C-TERM-MISMATCH"


def test_both_termini_mismatch_combine():
    psa = _classify("PEPTIDEK", "AAPTIDEA")
    assert psa.result.selected_event == "N-TERM-MISMATCH + C-TERM-MISMATCH"


def test_middle_substitution_is_not_terminal():
    psa = _classify("PEPTIDEK", "PEPTVDEK")
    assert psa.result.selected_event == "SUBSTITUTION"


def test_plain_deletion():
    psa = _classify("PEPTIDEK", "PEPTIEK")
    assert psa.result.selected_event == "DELETION"


def test_identical_sequences():
    psa = _classify("PEPTIDEK", "PEPTIDEK")
    assert psa.result.selected_event == "IDENTICAL"
