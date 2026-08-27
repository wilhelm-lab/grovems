from __future__ import annotations

from collections import Counter
from typing import Optional

from rapidfuzz.distance import Levenshtein

from .psa_aligner import AlignerMixin
from .psa_utils import UtilsMixin
from .psa_event_detection import EventDetectionMixin
from .psa_result import PSAResult

__all__ = ["PSA", "PSAResult"]


class PSA(UtilsMixin, EventDetectionMixin, AlignerMixin):
    """Classifies the difference between two peptide sequences."""

    def __init__(
        self,
        sequence1: Optional[str] = None,
        sequence2: Optional[str] = None,
        *,
        debug: bool = False,
    ) -> None:
        """Create a PSA classifier, optionally setting the sequence pair immediately (or via set_sequences later)."""
        self.result = PSAResult()
        self.debug = debug
        self.sequence1: Optional[str] = None
        self.mass_1: Optional[float] = None
        self.sequence2: Optional[str] = None
        self.mass_2: Optional[float] = None

        if sequence1 is not None or sequence2 is not None:
            if sequence1 is None or sequence2 is None:
                raise ValueError("Provide both sequence1 and sequence2, or neither.")
            self.set_sequences(sequence1, sequence2)

    def set_sequences(self, sequence1: str, sequence2: str, *, reset_result: bool = True) -> None:
        """Set the sequence pair to compare, computing their masses and resetting the result."""
        self.sequence1 = sequence1
        self.sequence2 = sequence2
        self.mass_1 = float(self.calculate_mass(sequence1))
        self.mass_2 = float(self.calculate_mass(sequence2))

        if reset_result:
            self.result.reset()

        self.result.update(
            peptide_sequence=(sequence1, sequence2),
            monoisotopic_mass=(self.mass_1, self.mass_2),
        )

    def tier_assignment(self) -> None:
        """assign a tier"""

        levenshtein_distance = Levenshtein.distance(self.sequence1, self.sequence2)
        final_tier = self._get_tier(levenshtein_distance)

        self.result.update(
            tier=final_tier[-1],
            levenshtein_distance=levenshtein_distance,
            final_tier=final_tier,
        )

    def classify(self) -> None:
        """Run :meth:`tier_assignment` and build the final PSA label/change summary."""

        reference_mass = max(abs(self.mass_1), abs(self.mass_2), 1.0)
        tolerance = max(reference_mass * 5.0 * 1e-6, 0.0)  # 5.0 ppm
        self.isobaric = abs(self.mass_1 - self.mass_2) <= tolerance
        self.anagram = len(self.sequence1) == len(self.sequence2) and Counter(self.sequence1) == Counter(self.sequence2)

        self.tier_assignment()

        # Align sequences
        aln_res = self.align(self.sequence1, self.sequence2)
        # Take alignment and define candidate events
        event_labels = self.candidate_events(aln_res)
        # Select or Label the event
        event_names = [event["event"] for event in event_labels]
        if len(event_names) == 2:
            label_event_name = "+".join(event_names)
        elif len(event_names) > 2:
            label_event_name = "MULTIEVENT"
        else:
            label_event_name = event_names[0]
        isobaric_label = "ISOBARIC" if self.isobaric else "NONISOBARIC"

        label = f"PSA - Tier {self.result.tier} - {isobaric_label} - {label_event_name}"

        change_summary = "+".join(event["description"] for event in event_labels if event.get("description"))

        self.result.update(
            isobaric=self.isobaric,
            anagram=self.anagram,
            alignment=aln_res,
            label=f"{label}",
            label_event_name=label_event_name,
            isobaric_label=isobaric_label,
            change_summary=change_summary,
        )

    def __str__(self) -> str:
        return str(self.result)

    def __repr__(self) -> str:
        return repr(self.result)
