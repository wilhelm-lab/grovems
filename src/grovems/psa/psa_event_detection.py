from __future__ import annotations

from collections import Counter
from typing import Any, Dict, Optional, Tuple

MAX_EVENT_WINDOW_LENGTH = 5
LOCAL_ISOBARIC_MASS_TOLERANCE_DA = 0.04

# Searched shortest-window-first: each has a fixed, exact shape (2 residues for a swap, 3-4
# for a shuffle, or a strict mass match for isobaric), so the first/smallest match found is
# already the correct one -- there's nothing to gain from preferring a larger window.
_SHORTEST_FIRST_TIERS = ("ADJACENT_SWAP", "BLOCK_SHUFFLE", "ISOBARIC-SUBSTITUTION")
# Searched longest-window-first: these have no fixed shape (anything non-identical qualifies),
# so the search must prefer the largest natural block to avoid fragmenting e.g. one real
# 3-residue substitution into three separate 1-residue ones.
_LONGEST_FIRST_TIERS = ("SUBSTITUTION", "DELETION", "INSERTION")
# ADJACENT_SWAP/BLOCK_SHUFFLE carry their details as a single dict everywhere (event_change_summary,
# _find_priority_window's callers); every other event type carries a [dict] list.
_DICT_SHAPED_EVENTS = {"ADJACENT_SWAP", "BLOCK_SHUFFLE"}


class EventDetectionMixin:
    """Pairwise alignment and per-window event detection."""

    @staticmethod
    def _alignment_columns_for(sequence1: str, sequence2: str, alignment: Any) -> list[Dict[str, Any]]:
        """Expand an alignment into one dict per column: alignment_idx, seq1/2_idx (None=gap), seq1/2_aa ('' =gap)."""
        seq1_indices, seq2_indices = alignment.indices
        columns: list[Dict[str, Any]] = []

        for column_idx, (seq1_idx, seq2_idx) in enumerate(zip(seq1_indices, seq2_indices)):
            columns.append(
                {
                    "alignment_idx": column_idx,
                    "seq1_idx": None if seq1_idx < 0 else int(seq1_idx),
                    "seq2_idx": None if seq2_idx < 0 else int(seq2_idx),
                    "seq1_aa": "" if seq1_idx < 0 else sequence1[seq1_idx],
                    "seq2_aa": "" if seq2_idx < 0 else sequence2[seq2_idx],
                }
            )

        return columns

    @staticmethod
    def _window_sequences(columns: list[Dict[str, Any]]) -> Tuple[str, str, Tuple[int, ...]]:
        """Reconstruct the gap-free (sequence1_fragment, sequence2_fragment, seq1_positions) covered by columns."""
        sequence1 = "".join(column["seq1_aa"] for column in columns if column["seq1_idx"] is not None)
        sequence2 = "".join(column["seq2_aa"] for column in columns if column["seq2_idx"] is not None)
        seq1_positions = tuple(column["seq1_idx"] for column in columns if column["seq1_idx"] is not None)
        return sequence1, sequence2, seq1_positions

    @classmethod
    def _block_details(cls, columns: list[Dict[str, Any]]) -> Dict[str, Any]:
        """Standard event-detail dict (pos, k, sequences, indices) for a block of alignment columns."""
        sequence1, sequence2, seq1_positions = cls._window_sequences(columns)
        seq2_positions = tuple(column["seq2_idx"] for column in columns if column["seq2_idx"] is not None)
        return {
            "pos": min(seq1_positions or seq2_positions),
            "k": max(len(seq1_positions), len(seq2_positions), 1),
            "sequence 1": sequence1,
            "sequence 2": sequence2,
            "indices": list(seq1_positions),
            "seq2_indices": list(seq2_positions),
            "alignment_indices": [column["alignment_idx"] for column in columns],
        }

    def local_alignment_events_for(self, sequence1: str, sequence2: str) -> list[Tuple[str, Any]]:
        """Align two sequences and repeatedly pull the next highest-priority event out of it.

        _find_priority_window already searches the whole column list, so this just keeps
        asking it for the next event on whatever's left (blacklisting each one's columns)
        until nothing more is found. SUBSTITUTION/DELETION/INSERTION are themselves part of
        that same priority search (as the lowest-priority, catch-all tiers), so every changed
        column is guaranteed to eventually be claimed by something -- there's no separate
        leftover-handling step.
        """
        alignment = self.aligner.align(sequence1, sequence2)[0]
        columns = self._alignment_columns_for(sequence1, sequence2, alignment)

        terminal_events = self._terminal_mismatch_events(columns)
        if terminal_events is not None:
            return terminal_events

        consumed: set[int] = set()
        events: list[Tuple[str, Any]] = []
        while True:
            remaining = [column for column in columns if column["alignment_idx"] not in consumed]
            found = self._find_priority_window(remaining)
            if found is None:
                break
            name, details = found
            events.append((name, details if name in _DICT_SHAPED_EVENTS else [details]))
            consumed |= set(details["alignment_indices"])

        return events

    @classmethod
    def _detect_direct_adjacent_swap(cls, columns: list[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
        """Detect a 2-residue block that is exactly its own reverse (seq1[i:i+2] reversed == seq2[i:i+2])."""
        sequence1, sequence2, seq1_positions = cls._window_sequences(columns)
        if len(sequence1) != 2 or len(sequence2) != 2:
            return None

        mismatches = [idx for idx, column in enumerate(columns) if column["seq1_aa"] != column["seq2_aa"]]
        if len(mismatches) != 2:
            return None

        left, right = mismatches
        if right != left + 1:
            return None

        if (sequence1[left] != sequence2[right]) or (sequence1[right] != sequence2[left]):
            return None

        return {
            "pos": seq1_positions[0],
            "k": 2,
            "indices": list(seq1_positions),
            "sequence 1": sequence1,
            "sequence 2": sequence2,
            "alignment_indices": [column["alignment_idx"] for column in columns],
        }

    @classmethod
    def _detect_shuffle(cls, columns: list[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
        """Detect a 3-4 residue block that's a same-composition anagram of itself, differing only in order."""
        sequence1, sequence2, seq1_positions = cls._window_sequences(columns)
        if len(sequence1) < 3 or len(sequence1) > 4 or len(sequence1) != len(sequence2):
            return None

        mismatch_count = sum(column["seq1_aa"] != column["seq2_aa"] for column in columns)
        if mismatch_count < 2 or sequence1 == sequence2 or Counter(sequence1) != Counter(sequence2):
            return None

        return {
            "pos": seq1_positions[0],
            "k": len(sequence1),
            "indices": list(seq1_positions),
            "sequence 1": sequence1,
            "sequence 2": sequence2,
            "alignment_indices": [column["alignment_idx"] for column in columns],
        }

    @classmethod
    def _terminal_mismatch_events(cls, columns: list[Dict[str, Any]]) -> Optional[list[Tuple[str, Any]]]:
        """N-TERM-MISMATCH/C-TERM-MISMATCH if every differing column falls within the first or
        last 2 alignment columns, or None if any difference lies outside those two ends.

        De novo calls are notoriously unreliable at the very ends of a peptide (weak terminal
        fragment-ion coverage), so a difference confined there is worth flagging as its own
        category up front, regardless of whether it'd otherwise read as a swap/shuffle/
        substitution/indel -- takes priority over all of those rather than competing with them.
        """
        changed = [column for column in columns if column["seq1_aa"] != column["seq2_aa"]]
        n_term = range(0, min(2, len(columns)))
        c_term = range(max(0, len(columns) - 2), len(columns))
        if any(column["alignment_idx"] not in n_term and column["alignment_idx"] not in c_term for column in changed):
            return None

        events: list[Tuple[str, Any]] = []
        for name, term_range in (("N-TERM-MISMATCH", n_term), ("C-TERM-MISMATCH", c_term)):
            term_columns = [column for column in changed if column["alignment_idx"] in term_range]
            if term_columns:
                events.append((name, [cls._block_details(term_columns)]))
        return events

    @classmethod
    def _detect_isobaric_block(cls, columns: list[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
        """Detect a block whose two sides are mass-equal (within LOCAL_ISOBARIC_MASS_TOLERANCE_DA), not a pure indel."""
        sequence1, sequence2, seq1_positions = cls._window_sequences(columns)
        if not sequence1 or not sequence2 or sequence1 == sequence2:
            return None

        mass_1 = cls.sequence_mass(sequence1)
        mass_2 = cls.sequence_mass(sequence2)
        if not cls.same_mass(mass_1, mass_2, min_tolerance_da=LOCAL_ISOBARIC_MASS_TOLERANCE_DA):
            return None

        seq2_positions = tuple(column["seq2_idx"] for column in columns if column["seq2_idx"] is not None)
        return {
            "pos": min(seq1_positions or seq2_positions),
            "k": max(len(seq1_positions), len(seq2_positions), 1),
            "sequence 1": sequence1,
            "sequence 2": sequence2,
            "indices": list(seq1_positions),
            "seq2_indices": list(seq2_positions),
            "alignment_indices": [column["alignment_idx"] for column in columns],
            "mass 1": mass_1,
            "mass 2": mass_2,
            "isobaric": True,
            "adjacent_event": len(columns) > 1,
        }

    @classmethod
    def _detect_substitution_block(cls, columns: list[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
        """Detect a block with real content on both sides and no embedded match column.

        Catch-all for whatever isn't a swap/shuffle/isobaric match -- gap columns are fine
        within the block (that's how a compound substitution like D->QQ or K->GA is
        represented), but a true match column would mean merging two unrelated changes
        across an untouched residue, so that's rejected.
        """
        if any(column["seq1_aa"] == column["seq2_aa"] for column in columns):
            return None
        details = cls._block_details(columns)
        if details["sequence 1"] and details["sequence 2"]:
            return details
        return None

    @classmethod
    def _detect_deletion_block(cls, columns: list[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
        """Detect a block that's pure deletion: real seq1 content throughout, no seq2 content at all."""
        details = cls._block_details(columns)
        if details["sequence 1"] and not details["sequence 2"]:
            return details
        return None

    @classmethod
    def _detect_insertion_block(cls, columns: list[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
        """Detect a block that's pure insertion: real seq2 content throughout, no seq1 content at all."""
        details = cls._block_details(columns)
        if details["sequence 2"] and not details["sequence 1"]:
            return details
        return None

    @classmethod
    def _find_priority_window(cls, columns: list[Dict[str, Any]]) -> Optional[Tuple[str, Dict[str, Any]]]:
        """The single highest-priority event found anywhere in columns.

        Tries every sliding window of length 1..MAX_EVENT_WINDOW_LENGTH for each event type in
        turn -- ADJACENT_SWAP, then BLOCK_SHUFFLE, then ISOBARIC-SUBSTITUTION, then SUBSTITUTION,
        then DELETION, then INSERTION (matching _label_name_priority) -- searching the *whole*
        column list for one type before moving to the next, so e.g. any swap anywhere always
        wins over any shuffle anywhere. The first three have a fixed shape and are searched
        shortest-window-first; the last three are open-ended catch-alls and are searched
        longest-window-first so a natural multi-residue block isn't fragmented into several
        1-residue events.
        """
        detectors = {
            "ADJACENT_SWAP": cls._detect_direct_adjacent_swap,
            "BLOCK_SHUFFLE": cls._detect_shuffle,
            "ISOBARIC-SUBSTITUTION": cls._detect_isobaric_block,
            "SUBSTITUTION": cls._detect_substitution_block,
            "DELETION": cls._detect_deletion_block,
            "INSERTION": cls._detect_insertion_block,
        }
        max_length = min(len(columns), MAX_EVENT_WINDOW_LENGTH)

        for name, window_lengths in (
            *((name, range(1, max_length + 1)) for name in _SHORTEST_FIRST_TIERS),
            *((name, range(max_length, 0, -1)) for name in _LONGEST_FIRST_TIERS),
        ):
            detect = detectors[name]
            for window_length in window_lengths:
                for start in range(len(columns) - window_length + 1):
                    window = columns[start : start + window_length]
                    if all(column["seq1_aa"] == column["seq2_aa"] for column in window):
                        continue
                    match = detect(window)
                    if match is not None:
                        return (name, match)

        return None
