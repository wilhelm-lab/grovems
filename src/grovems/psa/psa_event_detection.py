from __future__ import annotations

import logging
from collections import Counter
from typing import Any, Dict, Optional, Tuple

from Bio import Align

logger = logging.getLogger(__name__)

MAX_SHUFFLE_WINDOW_LENGTH = 5


class EventDetectionMixin:
    """Pairwise alignment and per-block event-candidate detection."""

    def build_aligner(self) -> Align.PairwiseAligner:
        """Global pairwise aligner PSA uses for all sequence comparisons (match=1, mismatch=0, gap=-1)."""
        aligner = Align.PairwiseAligner()
        aligner.mode = "global"
        aligner.match_score = 1.0
        aligner.mismatch_score = 0.0
        aligner.open_gap_score = -1.0
        aligner.extend_gap_score = -1.0
        self._debug("Using PairwiseAligner in global mode")
        return aligner

    def sequence_alignment(self) -> None:
        """Align ``self.sequence1``/``self.sequence2`` and store the result on ``self.result``."""
        self._debug("Running pairwise alignment for seq1=%s seq2=%s", self.sequence1, self.sequence2)
        aln = self.aligner.align(self.sequence1, self.sequence2)[0]
        self.result.update(alignment=aln)
        counts = aln.counts()
        self._debug("Alignment stored: identities=%s gaps=%s", counts.identities, counts.gaps)

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

    def alignment_columns(self) -> list[Dict[str, Any]]:
        """Per-column view of ``self.result.alignment`` for ``self.sequence1``/``self.sequence2``."""
        return self._alignment_columns_for(self.sequence1, self.sequence2, self.result.alignment)

    @staticmethod
    def _window_sequences(columns: list[Dict[str, Any]]) -> Tuple[str, str, Tuple[int, ...]]:
        """Reconstruct the gap-free (sequence1_fragment, sequence2_fragment, seq1_positions) covered by columns."""
        sequence1 = "".join(column["seq1_aa"] for column in columns if column["seq1_idx"] is not None)
        sequence2 = "".join(column["seq2_aa"] for column in columns if column["seq2_idx"] is not None)
        seq1_positions = tuple(column["seq1_idx"] for column in columns if column["seq1_idx"] is not None)
        return sequence1, sequence2, seq1_positions

    def local_event_from_columns(self, columns: list[Dict[str, Any]]) -> Optional[Tuple[str, Any]]:
        """Classify one changed run of columns into a single named event, or None if not a change."""
        if not columns:
            return None

        # Summarize the run into a generic event-detail dict (pos, k, sequences, indices);
        # every branch below returns this, some adding their own extra fields first.
        block_sequence1, block_sequence2, seq1_positions = self._window_sequences(columns)
        seq2_positions = tuple(column["seq2_idx"] for column in columns if column["seq2_idx"] is not None)
        anchor_positions = seq1_positions or seq2_positions or (0,)
        details = {
            "pos": min(anchor_positions),
            "k": max(len(seq1_positions), len(seq2_positions), 1),
            "sequence 1": block_sequence1,
            "sequence 2": block_sequence2,
            "indices": list(seq1_positions),
            "seq2_indices": list(seq2_positions),
            "alignment_indices": [column["alignment_idx"] for column in columns],
        }
        if block_sequence1 == block_sequence2:
            return None

        if block_sequence1 and not block_sequence2:
            return ("DELETION", [details])
        if block_sequence2 and not block_sequence1:
            return ("INSERTION", [details])
        if not block_sequence1 or not block_sequence2:
            return None

        adjacent_swap = self._detect_direct_adjacent_swap(columns)
        if adjacent_swap is not None:
            return ("ADJACENT_SWAP", adjacent_swap)

        block_shuffle = self._detect_shuffle(columns)
        if block_shuffle is not None:
            return ("BLOCK_SHUFFLE", block_shuffle)

        mass_1 = self.sequence_mass(block_sequence1)
        mass_2 = self.sequence_mass(block_sequence2)
        isobaric = self.same_mass(mass_1, mass_2)
        details.update(
            {
                "mass 1": mass_1,
                "mass 2": mass_2,
                "isobaric": isobaric,
                "adjacent_event": len(columns) > 1,
            }
        )
        if isobaric:
            return ("ISOBARIC-SUBSTITUTION", [details])

        has_gap = any(column["seq1_idx"] is None or column["seq2_idx"] is None for column in columns)
        if has_gap:
            return ("LOCAL-REARRANGEMENT", [details])

        return ("SUBSTITUTION", [details])

    def local_alignment_events_for(self, sequence1: str, sequence2: str) -> list[Tuple[str, Any]]:
        """Align two sequences and classify every changed run into an event."""
        alignment = self.aligner.align(sequence1, sequence2)[0]
        columns = self._alignment_columns_for(sequence1, sequence2, alignment)

        # Split into maximal runs of consecutive mismatched/gapped columns.
        changed_runs: list[list[Dict[str, Any]]] = []
        current_run: list[Dict[str, Any]] = []
        for column in columns:
            if column["seq1_aa"] == column["seq2_aa"]:
                if current_run:
                    changed_runs.append(current_run)
                    current_run = []
                continue
            current_run.append(column)
        if current_run:
            changed_runs.append(current_run)

        events: list[Tuple[str, Any]] = []
        for changed_run in changed_runs:
            local_event = self.local_event_from_columns(changed_run)
            if local_event is not None:
                events.append(local_event)
        return events

    def adjacent_substitution_event_for(
        self, sequence1: str, sequence2: str
    ) -> Optional[Tuple[str, list[Dict[str, Any]]]]:
        """The single SUBSTITUTION/ISOBARIC-SUBSTITUTION/LOCAL-REARRANGEMENT event, if there's exactly one."""
        local_events = self.local_alignment_events_for(sequence1, sequence2)
        if len(local_events) != 1:
            return None
        event_name, event_details = local_events[0]
        if event_name not in {"SUBSTITUTION", "ISOBARIC-SUBSTITUTION", "LOCAL-REARRANGEMENT"}:
            return None
        if not isinstance(event_details, list):
            return None
        return (event_name, event_details)

    @classmethod
    def _detect_direct_adjacent_swap(cls, columns: list[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
        """Detect a 2-residue block that is exactly its own reverse (seq1[i:i+2] reversed == seq2[i:i+2])."""
        sequence1, sequence2, seq1_positions = cls._window_sequences(columns)
        if len(sequence1) != 2 or len(sequence2) != 2 or len(seq1_positions) != 2:
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
        if len(sequence1) < 3 or len(sequence1) > 4 or len(sequence1) != len(sequence2) or not seq1_positions:
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

    def reordering_event_for(self, sequence1: str, sequence2: str) -> Optional[Tuple[str, Dict[str, Any]]]:
        """Align two sequences and look for a swap/shuffle reordering event anywhere in them.

        Searches every sliding window of length 2..MAX_SHUFFLE_WINDOW_LENGTH for the first
        swap/shuffle found.
        """
        alignment = self.aligner.align(sequence1, sequence2)[0]
        columns = self._alignment_columns_for(sequence1, sequence2, alignment)
        if not columns or all(column["seq1_aa"] == column["seq2_aa"] for column in columns):
            return None

        for window_length in range(2, min(len(columns), MAX_SHUFFLE_WINDOW_LENGTH) + 1):
            for start in range(len(columns) - window_length + 1):
                window = columns[start : start + window_length]
                if all(column["seq1_aa"] == column["seq2_aa"] for column in window):
                    continue

                adjacent_swap = self._detect_direct_adjacent_swap(window)
                if adjacent_swap is not None:
                    return ("ADJACENT_SWAP", adjacent_swap)

                block_shuffle = self._detect_shuffle(window)
                if block_shuffle is not None:
                    return ("BLOCK_SHUFFLE", block_shuffle)

        return None

    def reordering_event(self) -> Optional[Tuple[str, Dict[str, Any]]]:
        """``reordering_event_for`` applied to ``self.sequence1``/``self.sequence2``."""
        return self.reordering_event_for(self.sequence1, self.sequence2)
