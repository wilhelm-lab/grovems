from __future__ import annotations

from dataclasses import dataclass, field
from pprint import pformat
from typing import Any, Dict, Optional, Tuple


@dataclass
class PSAResult:
    """Mutable record of one PSA classification, updated in place as PSA runs."""

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
        """Set any given fields (leaving the rest untouched); extra kwargs merge into self.details."""
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
