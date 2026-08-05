from __future__ import annotations

import logging

from spectrum_fundamentals.constants import AA_MOD

logger = logging.getLogger(__name__)

ISOBARIC_MASS_TOLERANCE_PPM = 5.0


class MassUtilsMixin:
    """Monoisotopic mass calculation and mass-equality checks for peptide sequences."""

    @staticmethod
    def calculate_mass(sequence: str) -> float:
        """Compute the monoisotopic mass of ``sequence``, rounded to 4 decimals.

        Args:
            sequence: Amino-acid sequence (may include UNIMOD-style modification tags
                recognized by ``AA_MOD``).

        Returns:
            Monoisotopic mass in Da, rounded to 4 decimal places.

        Raises:
            ValueError: If ``sequence`` contains a residue not present in ``AA_MOD``.
        """
        invalid_residues = sorted({aa for aa in sequence if aa not in AA_MOD})
        if invalid_residues:
            invalid_list = ", ".join(invalid_residues)
            raise ValueError(f"Unsupported residue(s) in sequence: {invalid_list}")
        mass = float(sum(AA_MOD[aa] for aa in sequence))
        return round(mass, 4)

    @staticmethod
    def sequence_mass(sequence: str) -> float:
        """Compute the monoisotopic mass of ``sequence``, unrounded.

        Args:
            sequence: Amino-acid sequence (may include UNIMOD-style modification tags
                recognized by ``AA_MOD``).

        Returns:
            Monoisotopic mass in Da.

        Raises:
            ValueError: If ``sequence`` contains a residue not present in ``AA_MOD``.
        """
        invalid_residues = sorted({aa for aa in sequence if aa not in AA_MOD})
        if invalid_residues:
            invalid_list = ", ".join(invalid_residues)
            raise ValueError(f"Unsupported residue(s) in sequence: {invalid_list}")
        return float(sum(AA_MOD[aa] for aa in sequence))

    @staticmethod
    def same_mass(mass_1: float, mass_2: float, *, ppm: float = ISOBARIC_MASS_TOLERANCE_PPM) -> bool:
        """Check whether two masses are equal within ``ppm`` relative tolerance.

        Args:
            mass_1: First mass, in Da.
            mass_2: Second mass, in Da.
            ppm: Relative tolerance in parts-per-million of the larger mass.

        Returns:
            ``True`` if ``mass_1`` and ``mass_2`` are within ``ppm`` of each other.
        """
        reference_mass = max(abs(mass_1), abs(mass_2), 1.0)
        return abs(mass_1 - mass_2) <= reference_mass * ppm * 1e-6
