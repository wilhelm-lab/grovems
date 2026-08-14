from __future__ import annotations

import logging
from collections import Counter
from typing import Any, Optional

from Bio import Align

from .psa_event_detection import EventDetectionMixin
from .psa_event_labeling import EventLabelingMixin
from .psa_levenshtein import LevenshteinMixin
from .psa_mass_utils import MassUtilsMixin
from .psa_result import PSAResult

__all__ = ["PSA", "PSAResult"]

logger = logging.getLogger(__name__)
logger.addHandler(logging.NullHandler())

DEBUG_LOG_FORMAT = "%(asctime)s - %(levelname)s - %(message)s"


class PSA(MassUtilsMixin, LevenshteinMixin, EventDetectionMixin, EventLabelingMixin):
    """Classifies the difference between two peptide sequences."""

    _debug_handler: Optional[logging.Handler] = None

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
        self.aligner = Align.PairwiseAligner()
        self.aligner.mode = "global"
        self.aligner.match_score = 1.0
        self.aligner.mismatch_score = 0.0
        self.aligner.open_gap_score = -1.0
        self.aligner.extend_gap_score = -1.0
        self.sequence1: Optional[str] = None
        self.mass_1: Optional[float] = None
        self.sequence2: Optional[str] = None
        self.mass_2: Optional[float] = None

        if sequence1 is not None or sequence2 is not None:
            if sequence1 is None or sequence2 is None:
                raise ValueError("Provide both sequence1 and sequence2, or neither.")
            self.set_sequences(sequence1, sequence2)

    def _debug(self, msg: str, *args: Any) -> None:
        if not self.debug:
            return
        if PSA._debug_handler is None:
            handler = logging.StreamHandler()
            handler.setLevel(logging.DEBUG)
            handler.setFormatter(logging.Formatter(DEBUG_LOG_FORMAT))
            PSA._debug_handler = handler
        record = logger.makeRecord(logger.name, logging.DEBUG, __file__, 0, msg, args, None)
        PSA._debug_handler.emit(record)

    def set_sequences(self, sequence1: str, sequence2: str, *, reset_result: bool = True) -> None:
        """Set the sequence pair to compare, computing their masses and resetting the result."""
        self.sequence1 = sequence1
        self.sequence2 = sequence2
        self.mass_1 = float(self.calculate_mass(sequence1))
        self.mass_2 = float(self.calculate_mass(sequence2))
        self._debug(
            "PSA sequences set: seq1=%s seq2=%s len1=%d len2=%d mass1=%.4f mass2=%.4f",
            sequence1,
            sequence2,
            len(sequence1),
            len(sequence2),
            self.mass_1,
            self.mass_2,
        )

        if reset_result:
            self.result.reset()

        self.result.update(
            peptide_sequence=(sequence1, sequence2),
            monoisotopic_mass=(self.mass_1, self.mass_2),
        )

    def select_event(self, selected_event: str, event_details: Any) -> None:
        """Record the classified event onto self.result (event_details stored under selected_event, if not None)."""
        self._debug(
            "PSA event selected: event=%s tier=%d lev_distance=%d isobaric=%s anagram=%s details=%s",
            selected_event,
            self.result.tier,
            self.result.levenshtein_distance,
            self.isobaric,
            self.anagram,
            event_details,
        )

        self.result.update(
            selected_event=selected_event,
            **({selected_event: event_details} if event_details is not None else {}),
        )


    def tier_assignment(self) -> None:
        """Run the full PSA rulebook: align + score similarity, assign a tier, detect the event."""
        self.isobaric = self.same_mass(self.mass_1, self.mass_2)
        self.anagram = len(self.sequence1) == len(self.sequence2) and Counter(self.sequence1) == Counter(self.sequence2)
        self.result.update(isobaric=self.isobaric, anagram=self.anagram)

        aln = self.aligner.align(self.sequence1, self.sequence2)[0]
        self.result.update(alignment=aln)
        self.aln_cnt = aln.counts()

        levenshtein_distance = self.levenshtein_distance(self.sequence1, self.sequence2)
        final_tier = self.levenshtein_tier(levenshtein_distance)
        observed_changes = self.collect_observed_changes()
        self.result.update(
            tier=final_tier,
            levenshtein_distance=levenshtein_distance,
            final_tier=final_tier,
            alignment_counts={
                "identities": self.aln_cnt.identities,
                "mismatches": self.aln_cnt.mismatches,
                "gaps": self.aln_cnt.gaps,
            },
            observed_changes=observed_changes,
        )
        self._debug(
            "Starting PSA tier assignment: seq1=%s seq2=%s isobaric=%s anagram=%s " "lev_distance=%d final_tier=%d",
            self.sequence1,
            self.sequence2,
            self.isobaric,
            self.anagram,
            levenshtein_distance,
            final_tier,
        )

        raw_candidates = self.collect_event_candidates()
        if raw_candidates:
            self.result.update(
                raw_candidate_events=[self.summarize_candidate(candidate) for candidate in raw_candidates]
            )

            if len(raw_candidates) == 1:
                selected_event, event_details = raw_candidates[0]
                self.select_event(selected_event, event_details)
            else:
                self.assign_multi_event_variant(raw_candidates)
            return

        self.select_event("UNCLASSIFIED_VARIANT", None)
        self._debug("No explicit event matched; keeping Levenshtein-based tier with unclassified event")

    def classify(self) -> None:
        """Run :meth:`tier_assignment` and build the final PSA label/change summary."""
        self.tier_assignment()
        change_summary = self.event_change_summary(
            self.result.selected_event,
            self.result.details.get(self.result.selected_event),
        )
        label_event_name = self.label_event_name()
        isobaric_label = "ISOBARIC" if self.result.isobaric else "NONISOBARIC"
        label = f"PSA - Tier {self.result.tier} - {isobaric_label} - {label_event_name}"
        self.result.update(
            label=f"{label}",
            label_event_name=label_event_name,
            isobaric_label=isobaric_label,
            change_summary=change_summary,
        )
        logger.debug(
            "PSA classified: seq1=%s seq2=%s label=%s lev_distance=%d",
            self.sequence1,
            self.sequence2,
            self.result.label,
            self.result.levenshtein_distance,
        )

    def __str__(self) -> str:
        return str(self.result)

    def __repr__(self) -> str:
        return repr(self.result)
