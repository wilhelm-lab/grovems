from __future__ import annotations

from dataclasses import dataclass, field
from pprint import pformat
from typing import Any, Dict, Optional, Tuple


@dataclass
class PSAResult:
    """Data holder for PSA classification, updated in place as PSA runs for each peptide pair."""

    peptide_sequence: Tuple[str, str] = ("", "")
    monoisotopic_mass: Tuple[float, float] = (0.0, 0.0)
    tier: int = 0
    selected_event: str = ""
    alignment: Any = None
    identity_count: int = 0
    levenshtein_distance: int = 0
    isobaric: bool = False
    anagram: bool = False
    label: str = ""
    details: Dict[str, Any] = field(default_factory=dict)

    def reset(self) -> PSAResult:
        """Clear all fields."""
        self.peptide_sequence = ("", "")
        self.monoisotopic_mass = (0.0, 0.0)
        self.tier = 0
        self.selected_event = "UNIDENTIFIED"
        self.alignment = None
        self.identity_count = 0
        self.levenshtein_distance = 0
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
        levenshtein_distance: Optional[int] = None,
        isobaric: Optional[bool] = None,
        anagram: Optional[bool] = None,
        label: Optional[str] = None,
        **details: Any,
    ) -> PSAResult:
        """Set any given fields."""
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
        if levenshtein_distance is not None:
            self.levenshtein_distance = levenshtein_distance
        if isobaric is not None:
            self.isobaric = isobaric
        if anagram is not None:
            self.anagram = anagram
        if label is not None:
            self.label = label
        if details:
            self.details.update(details)
        return self

    def _alignment_str(self) -> str:
        """Render the alignment as target/match/query lines, e.g. 'target  PEPTIDE'."""
        if self.alignment is None:
            return "None"
        if isinstance(self.alignment, dict) and {"g1", "mk", "g2"} <= self.alignment.keys():
            return f"target  {self.alignment['g1']}\n        {self.alignment['mk']}\nquery   {self.alignment['g2']}"
        return str(self.alignment)

    def __str__(self) -> str:
        s1, s2 = self.peptide_sequence
        m1, m2 = self.monoisotopic_mass
        alignment_str = self._alignment_str()

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
            f"isobaric= {self.isobaric} | anagram= {self.anagram}\n\n"
            f"alignment:\n"
            "-----------\n"
            f"{alignment_str}\n\n"
            f"identity_count= {self.identity_count}\n"
            f"levenshtein_distance= {self.levenshtein_distance}\n"
            f"\t{details_str}"
        )

    def __repr__(self) -> str:
        return self.__str__()
