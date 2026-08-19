from __future__ import annotations

import logging

from spectrum_fundamentals.constants import AA_MOD
from .psa_constant import *


class UtilsMixin:
    """Monoisotopic mass calculation and mass-equality checks for peptide sequences."""

    @staticmethod
    def calculate_mass(sequence: str) -> float:
        """Monoisotopic mass of ``sequence`` in Da, rounded to 4 decimals."""
        invalid_residues = sorted({aa for aa in sequence if aa not in AA_MOD})
        if invalid_residues:
            raise ValueError(f"Unsupported residue(s) in sequence: {', '.join(invalid_residues)}")
        return round(float(sum(AA_MOD[aa] for aa in sequence)), 4)

    @staticmethod
    def _get_tier(distance):

        LEVENSHTEIN_TIER_UPPERBOND = {
            "Tier 1": 4,
            "Tier 2": 7,
            "Tier 3": 10,
            "Tier 4": 13,
        }

        for label, upper_bond in LEVENSHTEIN_TIER_UPPERBOND.items():
            if distance < upper_bond:
                return label
            else:
                return "Tier 5"
