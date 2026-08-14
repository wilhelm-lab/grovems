from __future__ import annotations

from collections import Counter
from typing import Any, Dict, Optional, Tuple

MAX_SHUFFLE_WINDOW_LENGTH = 5
LOCAL_ISOBARIC_MASS_TOLERANCE_DA = 0.04


class EventDetectionMixin:
    """Pairwise alignment and per-block event-candidate detection."""

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

    def local_event_from_columns(self, columns: list[Dict[str, Any]]) -> Tuple[str, Any]:
        """Classify one changed run of columns (always non-empty -- see local_alignment_events_for) into a named event.

        Never itself matches SWAP/SHUFFLE -- local_alignment_events_for's repeated
        _find_priority_window search already tried every window (including this run's own
        span) before falling back to per-run classification, so those checks here would
        always be redundant (confirmed empirically, 0/30000 random trials).
        """
        details = self._block_details(columns)
        block_sequence1, block_sequence2 = details["sequence 1"], details["sequence 2"]

        if block_sequence1 and not block_sequence2:
            return ("DELETION", [details])
        if block_sequence2 and not block_sequence1:
            return ("INSERTION", [details])

        isobaric_block = self._detect_isobaric_block(columns)
        if isobaric_block is not None:
            return ("ISOBARIC-SUBSTITUTION", [isobaric_block])

        return ("SUBSTITUTION", [details])

    def local_alignment_events_for(self, sequence1: str, sequence2: str) -> list[Tuple[str, Any]]:
        """Align two sequences and classify every changed run into an event.

        Keeps searching for priority (swap/shuffle/isobaric) windows on whatever's left
        after each one found -- a single search would only ever catch the first of two
        independent ones in one pair, leaving the second to fragment into unrelated
        DELETION+SUBSTITUTION+INSERTION pieces.
        """
        alignment = self.aligner.align(sequence1, sequence2)[0]
        columns = self._alignment_columns_for(sequence1, sequence2, alignment)

        terminal_events = self._terminal_mismatch_events(columns)
        if terminal_events is not None:
            return terminal_events

        consumed: set[int] = set()
        priority_events: list[Tuple[str, Any]] = []
        while True:
            remaining = [column for column in columns if column["alignment_idx"] not in consumed]
            priority_event = self._find_priority_window(remaining)
            if priority_event is None:
                break
            priority_events.append(priority_event)
            consumed |= set(priority_event[1]["alignment_indices"])

        changed_runs: list[list[Dict[str, Any]]] = []
        current_run: list[Dict[str, Any]] = []
        for column in columns:
            if column["seq1_aa"] == column["seq2_aa"] or column["alignment_idx"] in consumed:
                if current_run:
                    changed_runs.append(current_run)
                    current_run = []
                continue
            current_run.append(column)
        if current_run:
            changed_runs.append(current_run)

        # ISOBARIC-SUBSTITUTION carries its details as a [dict] list everywhere else (see
        # local_event_from_columns/event_change_summary); ADJACENT_SWAP/BLOCK_SHUFFLE stay dicts.
        events: list[Tuple[str, Any]] = [
            (name, [details] if name == "ISOBARIC-SUBSTITUTION" else details) for name, details in priority_events
        ]
        events.extend(self.local_event_from_columns(changed_run) for changed_run in changed_runs)
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
    def _find_priority_window(cls, columns: list[Dict[str, Any]]) -> Optional[Tuple[str, Dict[str, Any]]]:
        """First SWAP, then SHUFFLE, then ISOBARIC-SUBSTITUTION found scanning every sliding
        window of length 2..MAX_SHUFFLE_WINDOW_LENGTH -- each type is searched for across the
        *whole* column list before falling back to the next, so e.g. a swap anywhere always
        wins over a shuffle anywhere, matching the SWAP > SHUFFLE > ISOBARIC-SUBSTITUTION
        priority used elsewhere (event_change_summary's ordering, _label_name_priority).
        """
        for name, detect in (
            ("ADJACENT_SWAP", cls._detect_direct_adjacent_swap),
            ("BLOCK_SHUFFLE", cls._detect_shuffle),
            ("ISOBARIC-SUBSTITUTION", cls._detect_isobaric_block),
        ):
            for window_length in range(2, min(len(columns), MAX_SHUFFLE_WINDOW_LENGTH) + 1):
                for start in range(len(columns) - window_length + 1):
                    window = columns[start : start + window_length]
                    if all(column["seq1_aa"] == column["seq2_aa"] for column in window):
                        continue
                    match = detect(window)
                    if match is not None:
                        return (name, match)

        return None
