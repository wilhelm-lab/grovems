from __future__ import annotations

from dataclasses import dataclass, field
from pprint import pformat
from typing import Any, Dict, Optional, Tuple


@dataclass
class PSAResult:
    """Mutable record of one PSA classification, updated in place as PSA runs.

    Attributes:
        peptide_sequence: The two compared sequences, ``(sequence1, sequence2)``.
        monoisotopic_mass: Monoisotopic mass of each sequence, ``(mass1, mass2)``.
        tier: Levenshtein-distance-derived PSA tier (0 = identical).
        selected_event: Name of the classified event (e.g. ``"SUBSTITUTION"``).
        alignment: The ``Bio.Align`` alignment object used for scoring.
        identity_count: Number of identical aligned positions.
        normalized_identity: ``identity_count`` normalized by the longer sequence.
        jaccard_similarity: Legacy Jaccard-style similarity score.
        levenshtein_distance: Edit distance between the two sequences.
        similarity: Final similarity score (currently equal to ``normalized_identity``).
        similarity_level: Coarse bucket label for ``similarity``.
        isobaric: Whether the two sequences have (near-)equal mass.
        anagram: Whether the two sequences are exact anagrams of each other.
        label: Final human-readable PSA label, e.g. ``"PSA - Tier 1 - ISOBARIC - SWAP"``.
        details: Free-form extra fields attached via :meth:`update` (event details,
            observed changes, alignment counts, ...).
    """

    peptide_sequence: Tuple[str, str] = ("", "")
    monoisotopic_mass: Tuple[float, float] = (0.0, 0.0)
    tier: int = 0
    selected_event: str = ""
    alignment: Any = None
    identity_count: int = 0
    normalized_identity: float = 0.0
    jaccard_similarity: float = 0.0
    levenshtein_distance: int = 0
    similarity: float = 0.0
    similarity_level: str = ""
    isobaric: bool = False
    anagram: bool = False
    label: str = ""
    details: Dict[str, Any] = field(default_factory=dict)

    def reset(self) -> PSAResult:
        """Clear all fields back to their "no result yet" defaults, in place."""
        self.peptide_sequence = ("", "")
        self.monoisotopic_mass = (0.0, 0.0)
        self.tier = 0
        self.selected_event = "UNIDENTIFIED"
        self.alignment = None
        self.identity_count = 0
        self.normalized_identity = 0.0
        self.jaccard_similarity = 0.0
        self.levenshtein_distance = 0
        self.similarity = 0.0
        self.similarity_level = "NAN"
        self.isobaric = False
        self.anagram = False
        self.label = "NAN"
        self.details.clear()
        return self

    def update(
        self,
        *,
        peptide_sequence: Optional[Tuple[str, str]] = None,
        monoisotopic_mass: Optional[Tuple[float, float]] = None,
        tier: Optional[int] = None,
        selected_event: Optional[str] = None,
        alignment: Any = None,
        identity_count: Optional[int] = None,
        normalized_identity: Optional[float] = None,
        jaccard_similarity: Optional[float] = None,
        levenshtein_distance: Optional[int] = None,
        similarity: Optional[float] = None,
        similarity_level: Optional[str] = None,
        isobaric: Optional[bool] = None,
        anagram: Optional[bool] = None,
        label: Optional[str] = None,
        **details: Any,
    ) -> PSAResult:
        """Set any given fields (leaving the rest untouched) and merge extra ``details``.

        Args:
            peptide_sequence: New ``(sequence1, sequence2)`` pair, if provided.
            monoisotopic_mass: New ``(mass1, mass2)`` pair, if provided.
            tier: New PSA tier, if provided.
            selected_event: New selected event name, if provided.
            alignment: New alignment object, if provided.
            identity_count: New identity count, if provided.
            normalized_identity: New normalized identity, if provided.
            jaccard_similarity: New Jaccard similarity, if provided.
            levenshtein_distance: New Levenshtein distance, if provided.
            similarity: New similarity score, if provided.
            similarity_level: New similarity level label, if provided.
            isobaric: New isobaric flag, if provided.
            anagram: New anagram flag, if provided.
            label: New final label, if provided.
            **details: Arbitrary extra key/value pairs merged into ``self.details``.

        Returns:
            ``self``, for chaining.
        """
        if peptide_sequence is not None:
            self.peptide_sequence = peptide_sequence
        if monoisotopic_mass is not None:
            self.monoisotopic_mass = monoisotopic_mass
        if tier is not None:
            self.tier = tier
        if selected_event is not None:
            self.selected_event = selected_event
        if alignment is not None:
            self.alignment = alignment
        if identity_count is not None:
            self.identity_count = identity_count
        if normalized_identity is not None:
            self.normalized_identity = normalized_identity
        if jaccard_similarity is not None:
            self.jaccard_similarity = jaccard_similarity
        if levenshtein_distance is not None:
            self.levenshtein_distance = levenshtein_distance
        if similarity is not None:
            self.similarity = similarity
        if similarity_level is not None:
            self.similarity_level = similarity_level
        if isobaric is not None:
            self.isobaric = isobaric
        if anagram is not None:
            self.anagram = anagram
        if label is not None:
            self.label = label
        if details:
            self.details.update(details)
        return self

    def __str__(self) -> str:
        s1, s2 = self.peptide_sequence
        m1, m2 = self.monoisotopic_mass
        alignment_str = str(self.alignment) if self.alignment is not None else "None"

        details_str = ""
        if self.details:
            formatted_details = []
            for key, value in self.details.items():
                pretty_value = pformat(value, width=100, sort_dicts=False)
                indented_value = pretty_value.replace("\n", "\n    ")
                formatted_details.append(f"{key}: {indented_value}")
            details_str = "\nDetails:\n  " + "\n  ".join(formatted_details)

        return (
            "PSA Grading Result: \n"
            "-------------------\n"
            f"tier= {self.label}\n"
            f"peptide_sequence= {s1!r}, {s2!r} \n"
            f"monoisotopic_mass= {m1:.4f}, {m2:.4f} \n\n"
            f"isobaric= {self.isobaric} | anagram= {self.anagram}\n"
            f"alignment:\n"
            "-----------\n"
            f"{alignment_str}\n"
            f"identity_count= {self.identity_count}\n"
            "sequence_similarity_definition= identity_count / max(len(sequence1), len(sequence2))\n"
            f"sequence_similarity= {self.similarity:.2f} - {self.similarity_level}\n"
            f"legacy_jaccard_similarity= {self.jaccard_similarity:.2f}\n"
            f"levenshtein_distance= {self.levenshtein_distance}\n"
            f"\t{details_str}"
        )

    def __repr__(self) -> str:
        return self.__str__()
