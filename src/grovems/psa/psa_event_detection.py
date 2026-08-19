from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple


class EventDetectionMixin:
    """Pairwise alignment and per-window event detection."""

    @staticmethod
    def _alignment_runs(mk: str) -> List[Tuple[str, int, int]]:
        """Group alignment columns into contiguous same-symbol (symbol, start, end) ranges."""
        runs = []
        start = 0
        for i in range(1, len(mk) + 1):
            if i == len(mk) or mk[i] != mk[start]:
                runs.append((mk[start], start, i))
                start = i
        return runs

    @staticmethod
    def _block_event(g1: str, g2: str, start: int, end: int, name: str) -> Dict[str, Any]:
        sequence_1 = g1[start:end].replace("-", "")
        sequence_2 = g2[start:end].replace("-", "")
        return {
            "event": name,
            "pos": start,
            "sequence 1": sequence_1,
            "sequence 2": sequence_2,
            "k": end - start,
            "description": f"{name}({sequence_1}->{sequence_2})@{start + 1}",
        }

    @classmethod
    def _gap_run_events(cls, alignment_meta) -> List[Dict[str, Any]]:
        """Split each "-" run by which side is gapped: INSERTION (seq1 gapped) or DELETION (seq2 gapped)."""
        g1, g2 = alignment_meta["g1"], alignment_meta["g2"]
        events = []
        for symbol, start, end in alignment_meta["_runs"]:
            if symbol != "-":
                continue
            sub_start = start
            for i in range(start + 1, end + 1):
                if i == end or (g1[i] == "-") != (g1[sub_start] == "-"):
                    name = "INSERTION" if g1[sub_start] == "-" else "DELETION"
                    events.append(cls._block_event(g1, g2, sub_start, i, name))
                    sub_start = i
        return events

    @classmethod
    def _detect_shuffle(cls, alignment_meta) -> Optional[List[Dict[str, Any]]]:
        """SWAP (2-residue block) or SHUFFLE (3+) events, one per contiguous "x" run."""
        g1, g2 = alignment_meta["g1"], alignment_meta["g2"]
        events = [
            cls._block_event(g1, g2, start, end, "SWAP" if end - start == 2 else "SHUFFLE")
            for symbol, start, end in alignment_meta["_runs"]
            if symbol == "x"
        ]
        return events or None

    @classmethod
    def _terminal_mismatch_events(cls, alignment_meta) -> Optional[List[Dict[str, Any]]]:
        """N-/C-TERM-MISMATCH: a mismatch run touching the very start or end of the alignment."""
        g1, g2, mk = alignment_meta["g1"], alignment_meta["g2"], alignment_meta["mk"]
        events = []
        for symbol, start, end in alignment_meta["_runs"]:
            if symbol != ".":
                continue
            if start == 0:
                events.append(cls._block_event(g1, g2, start, end, "N-TERM-MISMATCH"))
            elif end == len(mk):
                events.append(cls._block_event(g1, g2, start, end, "C-TERM-MISMATCH"))
        return events or None

    @classmethod
    def _detect_substitution(cls, alignment_meta) -> Optional[List[Dict[str, Any]]]:
        """Non-terminal SUBSTITUTION ("." runs) and ISOBARIC-SUBSTITUTION ("=" runs)."""
        g1, g2, mk = alignment_meta["g1"], alignment_meta["g2"], alignment_meta["mk"]
        events = []
        for symbol, start, end in alignment_meta["_runs"]:
            if symbol == "." and start > 0 and end < len(mk):
                events.append(cls._block_event(g1, g2, start, end, "SUBSTITUTION"))
            elif symbol == "=":
                events.append(cls._block_event(g1, g2, start, end, "ISOBARIC-SUBSTITUTION"))
        return events or None

    @staticmethod
    def _detect_deletion(alignment_meta) -> Optional[List[Dict[str, Any]]]:
        """DELETION events: "-" runs where sequence1 has residues sequence2 lacks."""
        events = [event for event in alignment_meta["_gap_events"] if event["event"] == "DELETION"]
        return events or None

    @staticmethod
    def _detect_insertion(alignment_meta) -> Optional[List[Dict[str, Any]]]:
        """INSERTION events: "-" runs where sequence2 has residues sequence1 lacks."""
        events = [event for event in alignment_meta["_gap_events"] if event["event"] == "INSERTION"]
        return events or None

    @classmethod
    def candidate_events(cls, alignment_result: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Every candidate event in an alignment, ordered by position.

        Returns ``[{"event": "IDENTICAL", "pos": 0, "sequence 1": "", "sequence 2": "", "k": 0}]``
        when the alignment has no mismatches, gaps, shuffles, or isobaric substitutions.
        """
        mk = alignment_result["mk"]

        if not mk or set(mk) <= {"|"}:
            return [{"event": "IDENTICAL", "pos": 0, "sequence 1": "", "sequence 2": "", "k": 0, "description": ""}]

        alignment_meta = dict(alignment_result)
        alignment_meta["_runs"] = cls._alignment_runs(mk)
        alignment_meta["_gap_events"] = cls._gap_run_events(alignment_meta)

        events: List[Dict[str, Any]] = []
        events.extend(cls._terminal_mismatch_events(alignment_meta) or [])
        events.extend(cls._detect_substitution(alignment_meta) or [])
        events.extend(cls._detect_shuffle(alignment_meta) or [])
        events.extend(cls._detect_deletion(alignment_meta) or [])
        events.extend(cls._detect_insertion(alignment_meta) or [])

        events.sort(key=lambda event: event["pos"])
        return events
