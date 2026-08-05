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
        """Build the global pairwise aligner PSA uses for all sequence comparisons.

        Returns:
            A configured ``Bio.Align.PairwiseAligner`` (global mode, match=1,
            mismatch=0, gap-open=gap-extend=-1).
        """
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
    def _alignment_columns_for(
        sequence1: str,
        sequence2: str,
        alignment: Any,
    ) -> list[Dict[str, Any]]:
        """Expand an alignment into one dict per aligned column.

        Args:
            sequence1: First sequence (as passed to the aligner).
            sequence2: Second sequence (as passed to the aligner).
            alignment: A ``Bio.Align`` alignment of ``sequence1``/``sequence2``.

        Returns:
            List of per-column dicts: ``alignment_idx``, ``seq1_idx``/``seq2_idx``
            (``None`` for a gap), and ``seq1_aa``/``seq2_aa`` (``""`` for a gap).
        """
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
        """Reconstruct the (gap-free) subsequences and seq1 positions covered by ``columns``.

        Args:
            columns: A contiguous slice of alignment columns.

        Returns:
            ``(sequence1_fragment, sequence2_fragment, seq1_positions)``.
        """
        sequence1 = "".join(column["seq1_aa"] for column in columns if column["seq1_idx"] is not None)
        sequence2 = "".join(column["seq2_aa"] for column in columns if column["seq2_idx"] is not None)
        seq1_positions = tuple(column["seq1_idx"] for column in columns if column["seq1_idx"] is not None)
        return sequence1, sequence2, seq1_positions

    @staticmethod
    def _changed_alignment_runs(columns: list[Dict[str, Any]]) -> list[list[Dict[str, Any]]]:
        """Split alignment columns into maximal runs of consecutive mismatched/gapped columns.

        Args:
            columns: Full per-column alignment view (as from ``_alignment_columns_for``).

        Returns:
            List of column-runs, each a contiguous block where ``seq1_aa != seq2_aa``.
        """
        runs: list[list[Dict[str, Any]]] = []
        current_run: list[Dict[str, Any]] = []

        for column in columns:
            if column["seq1_aa"] == column["seq2_aa"]:
                if current_run:
                    runs.append(current_run)
                    current_run = []
                continue
            current_run.append(column)

        if current_run:
            runs.append(current_run)

        return runs

    @classmethod
    def _local_block_details(cls, columns: list[Dict[str, Any]]) -> Dict[str, Any]:
        """Summarize one changed run of alignment columns into a generic event-detail dict.

        Args:
            columns: One run of changed columns, as from ``_changed_alignment_runs``.

        Returns:
            Dict with ``pos``, ``k`` (width), ``sequence 1``/``sequence 2`` fragments,
            and the seq1/seq2/alignment index lists covered by the run.
        """
        block_sequence1, block_sequence2, seq1_positions = cls._window_sequences(columns)
        seq2_positions = tuple(column["seq2_idx"] for column in columns if column["seq2_idx"] is not None)
        anchor_positions = seq1_positions or seq2_positions or (0,)

        return {
            "pos": min(anchor_positions),
            "k": max(len(seq1_positions), len(seq2_positions), 1),
            "sequence 1": block_sequence1,
            "sequence 2": block_sequence2,
            "indices": list(seq1_positions),
            "seq2_indices": list(seq2_positions),
            "alignment_indices": [column["alignment_idx"] for column in columns],
        }

    def local_event_from_columns(
        self,
        columns: list[Dict[str, Any]],
    ) -> Optional[Tuple[str, Any]]:
        """Classify one changed run of alignment columns into a single named event.

        Args:
            columns: One run of changed columns, as from ``_changed_alignment_runs``.

        Returns:
            ``(event_name, event_details)`` -- one of ``DELETION``/``INSERTION``/
            ``ADJACENT_SWAP``/``BLOCK_SHUFFLE``/``ISOBARIC-SUBSTITUTION``/
            ``LOCAL-REARRANGEMENT``/``SUBSTITUTION`` -- or ``None`` if the run turns
            out not to represent a change (e.g. empty).
        """
        if not columns:
            return None

        details = self._local_block_details(columns)
        block_sequence1 = details["sequence 1"]
        block_sequence2 = details["sequence 2"]
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

    def local_alignment_events_for(
        self,
        sequence1: str,
        sequence2: str,
    ) -> list[Tuple[str, Any]]:
        """Align two sequences and classify every changed run into an event.

        Args:
            sequence1: First sequence.
            sequence2: Second sequence.

        Returns:
            List of ``(event_name, event_details)`` pairs, one per changed run.
        """
        alignment = self.aligner.align(sequence1, sequence2)[0]
        columns = self._alignment_columns_for(sequence1, sequence2, alignment)
        changed_runs = self._changed_alignment_runs(columns)
        events: list[Tuple[str, Any]] = []
        for changed_run in changed_runs:
            local_event = self.local_event_from_columns(changed_run)
            if local_event is not None:
                events.append(local_event)
        return events

    def adjacent_substitution_event_for(
        self,
        sequence1: str,
        sequence2: str,
    ) -> Optional[Tuple[str, list[Dict[str, Any]]]]:
        """Return the single substitution-like event between two sequences, if there's exactly one.

        Args:
            sequence1: First sequence.
            sequence2: Second sequence.

        Returns:
            ``(event_name, event_details)`` if the two sequences differ by exactly one
            ``SUBSTITUTION``/``ISOBARIC-SUBSTITUTION``/``LOCAL-REARRANGEMENT`` event,
            else ``None``.
        """
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
    def _detect_direct_adjacent_swap(
        cls,
        columns: list[Dict[str, Any]],
    ) -> Optional[Dict[str, Any]]:
        """Detect a 2-residue block that is exactly its own reverse (an adjacent swap).

        Args:
            columns: A run of changed alignment columns.

        Returns:
            Event-detail dict if ``columns`` is exactly a 2-residue direct swap
            (``seq1[i:i+2]`` reversed equals ``seq2[i:i+2]``), else ``None``.
        """
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
    def _detect_shuffle(
        cls,
        columns: list[Dict[str, Any]],
    ) -> Optional[Dict[str, Any]]:
        """Detect a 3-4 residue block that is a shuffled anagram of itself between sequences.

        Args:
            columns: A run of changed alignment columns.

        Returns:
            Event-detail dict if ``columns`` covers a same-length, same-composition
            (anagram), 3-4 residue block that differs in order, else ``None``.
        """
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

    @classmethod
    def _reordering_event_from_columns(
        cls,
        columns: list[Dict[str, Any]],
    ) -> Optional[Tuple[str, Dict[str, Any]]]:
        """Search every sliding window of ``columns`` for a swap or shuffle event.

        Args:
            columns: Full per-column alignment view.

        Returns:
            The first ``(event_name, event_details)`` swap/shuffle found in any
            window of length 2..``MAX_SHUFFLE_WINDOW_LENGTH``, else ``None``.
        """
        if not columns:
            return None

        if all(column["seq1_aa"] == column["seq2_aa"] for column in columns):
            return None

        for window_length in range(2, min(len(columns), MAX_SHUFFLE_WINDOW_LENGTH) + 1):
            for start in range(len(columns) - window_length + 1):
                window = columns[start : start + window_length]
                if all(column["seq1_aa"] == column["seq2_aa"] for column in window):
                    continue

                adjacent_swap = cls._detect_direct_adjacent_swap(window)
                if adjacent_swap is not None:
                    return ("ADJACENT_SWAP", adjacent_swap)

                block_shuffle = cls._detect_shuffle(window)
                if block_shuffle is not None:
                    return ("BLOCK_SHUFFLE", block_shuffle)

        return None

    def reordering_event_for(self, sequence1: str, sequence2: str) -> Optional[Tuple[str, Dict[str, Any]]]:
        """Align two sequences and look for a swap/shuffle reordering event anywhere in them.

        Args:
            sequence1: First sequence.
            sequence2: Second sequence.

        Returns:
            ``(event_name, event_details)`` for the first swap/shuffle found, else ``None``.
        """
        alignment = self.aligner.align(sequence1, sequence2)[0]
        columns = self._alignment_columns_for(sequence1, sequence2, alignment)
        return self._reordering_event_from_columns(columns)

    def reordering_event(self) -> Optional[Tuple[str, Dict[str, Any]]]:
        """``reordering_event_for`` applied to ``self.sequence1``/``self.sequence2``."""
        return self.reordering_event_for(self.sequence1, self.sequence2)

    def _candidate_explains_full_sequence(
        self,
        sequence1: str,
        sequence2: str,
        event_name: str,
        event_details: Any,
    ) -> bool:
        """Check whether one candidate event accounts for every difference in the full alignment.

        Args:
            sequence1: First sequence.
            sequence2: Second sequence.
            event_name: Candidate event name (only ``IDENTICAL``/``ADJACENT_SWAP``/
                ``BLOCK_SHUFFLE`` are meaningfully checked; anything else returns
                ``False``).
            event_details: The candidate's event-detail dict.

        Returns:
            ``True`` if every mismatched alignment column outside the candidate's own
            covered columns is unexplained (i.e. the candidate alone fully explains the
            difference between ``sequence1`` and ``sequence2``).
        """
        if event_name == "IDENTICAL":
            return sequence1 == sequence2

        if event_name not in {"ADJACENT_SWAP", "BLOCK_SHUFFLE"} or not isinstance(event_details, dict):
            return False

        alignment_indices = event_details.get("alignment_indices")
        if not isinstance(alignment_indices, list) or not alignment_indices:
            return False

        covered_alignment_indices = {int(idx) for idx in alignment_indices}
        alignment = self.aligner.align(sequence1, sequence2)[0]
        columns = self._alignment_columns_for(sequence1, sequence2, alignment)

        for column in columns:
            if int(column["alignment_idx"]) in covered_alignment_indices:
                continue
            if column["seq1_aa"] != column["seq2_aa"]:
                return False

        covered_columns = [column for column in columns if int(column["alignment_idx"]) in covered_alignment_indices]
        covered_sequence1 = "".join(column["seq1_aa"] for column in covered_columns if column["seq1_idx"] is not None)
        covered_sequence2 = "".join(column["seq2_aa"] for column in covered_columns if column["seq2_idx"] is not None)

        return covered_sequence1 == str(event_details.get("sequence 1", "")) and covered_sequence2 == str(
            event_details.get("sequence 2", "")
        )

    @staticmethod
    def _apply_reordering_event(
        sequence: str,
        event_name: str,
        event_details: Any,
    ) -> Optional[str]:
        """Apply a swap/shuffle event's replacement fragment onto ``sequence``.

        Args:
            sequence: The sequence to apply the event to.
            event_name: Must be ``ADJACENT_SWAP`` or ``BLOCK_SHUFFLE`` for anything to
                happen.
            event_details: The event's detail dict (needs ``indices`` and
                ``sequence 2``).

        Returns:
            ``sequence`` with the covered span replaced by ``event_details["sequence 2"]``,
            or ``None`` if ``event_name``/``event_details`` don't describe an applicable event.
        """
        if event_name not in {"ADJACENT_SWAP", "BLOCK_SHUFFLE"} or not isinstance(event_details, dict):
            return None

        indices = event_details.get("indices")
        replacement = event_details.get("sequence 2")
        if not isinstance(indices, list) or not indices or not isinstance(replacement, str):
            return None

        start = min(int(idx) for idx in indices)
        stop = max(int(idx) for idx in indices) + 1
        return sequence[:start] + replacement + sequence[stop:]
