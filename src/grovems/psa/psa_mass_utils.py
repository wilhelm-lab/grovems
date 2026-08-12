from __future__ import annotations

import logging

from spectrum_fundamentals.constants import AA_MOD

logger = logging.getLogger(__name__)

ISOBARIC_MASS_TOLERANCE_PPM = 5.0


class MassUtilsMixin:
    """Monoisotopic mass calculation and mass-equality checks for peptide sequences."""

    @staticmethod
    def calculate_mass(sequence: str) -> float:
        """Monoisotopic mass of ``sequence`` in Da, rounded to 4 decimals."""
        invalid_residues = sorted({aa for aa in sequence if aa not in AA_MOD})
        if invalid_residues:
            raise ValueError(f"Unsupported residue(s) in sequence: {', '.join(invalid_residues)}")
        return round(float(sum(AA_MOD[aa] for aa in sequence)), 4)

    @staticmethod
    def sequence_mass(sequence: str) -> float:
        """Monoisotopic mass of ``sequence`` in Da, unrounded."""
        invalid_residues = sorted({aa for aa in sequence if aa not in AA_MOD})
        if invalid_residues:
            raise ValueError(f"Unsupported residue(s) in sequence: {', '.join(invalid_residues)}")
        return float(sum(AA_MOD[aa] for aa in sequence))

    @staticmethod
    def same_mass(mass_1: float, mass_2: float, *, ppm: float = ISOBARIC_MASS_TOLERANCE_PPM) -> bool:
        """Whether two masses are equal within ``ppm`` relative tolerance."""
        reference_mass = max(abs(mass_1), abs(mass_2), 1.0)
        return abs(mass_1 - mass_2) <= reference_mass * ppm * 1e-6
