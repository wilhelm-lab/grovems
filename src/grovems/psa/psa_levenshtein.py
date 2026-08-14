from __future__ import annotations

from typing import Tuple

LEVENSHTEIN_TIER_THRESHOLDS: Tuple[Tuple[int, int], ...] = (
    (4, 1),
    (7, 2),
    (10, 3),
    (13, 4),
)


class LevenshteinMixin:
    """PSA tier assignment from Levenshtein distance."""

    @staticmethod
    def levenshtein_tier(distance: int) -> int:
        """Map a distance to a coarse PSA tier via LEVENSHTEIN_TIER_THRESHOLDS (0=identical, 5=beyond every tier)."""
        if distance == 0:
            return 0
        for threshold, tier in LEVENSHTEIN_TIER_THRESHOLDS:
            if distance < threshold:
                return tier
        return 5
