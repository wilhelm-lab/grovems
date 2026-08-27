"""Exercises the PSA usage contract that psa_merge._run_psa_pair relies on."""

from grovems.psa.psa import PSA


def test_empty_constructor_then_set_sequences():
    """psa_merge's worker initializer constructs PSA() with no sequences."""
    psa = PSA()
    psa.set_sequences("PEPTIDEK", "PEPTIEDK")
    psa.classify()

    assert isinstance(psa.result.label, str) and psa.result.label
    assert isinstance(psa.result.levenshtein_distance, int)


def test_instance_is_reused_across_pairs():
    """Workers keep a single PSA instance and call set_sequences/classify per row."""
    psa = PSA()
    pairs = [
        ("PEPTIDEK", "PEPTIEDK"),
        ("AKDEITFGHKMPQ", "ADKEITFGHMKPQ"),
        ("PEPTIDEK", "PEPTIDEK"),
    ]

    results = []
    for sequence1, sequence2 in pairs:
        psa.set_sequences(sequence1, sequence2)
        psa.classify()
        results.append((psa.result.label, psa.result.levenshtein_distance))

    labels, distances = zip(*results)
    # each call must overwrite the previous result rather than leak state
    assert len(set(labels)) > 1
    assert distances == (2, 4, 0)


def test_classify_raises_on_none_sequence():
    """psa_merge wraps set_sequences/classify in try/except and reports PSA_ERROR on failure."""
    psa = PSA()
    try:
        psa.set_sequences(None, "PEPTIDEK")
        psa.classify()
    except Exception as exc:
        assert isinstance(exc, Exception)
    else:
        raise AssertionError("expected set_sequences/classify to raise for a None sequence")
