from __future__ import annotations

import logging
from typing import Any, Dict, Tuple

logger = logging.getLogger(__name__)

LEVENSHTEIN_TIER_THRESHOLDS: Tuple[Tuple[int, int], ...] = (
    (4, 1),
    (7, 2),
    (10, 3),
    (13, 4),
)


class LevenshteinMixin:
    """Edit distance, PSA tier assignment, and edit-operation tracing between two sequences."""

    @staticmethod
    def levenshtein_distance(sequence1: str, sequence2: str) -> int:
        """Compute the classic Levenshtein (edit) distance between two sequences.

        Args:
            sequence1: First sequence.
            sequence2: Second sequence.

        Returns:
            Minimum number of single-residue insertions/deletions/substitutions
            needed to turn ``sequence1`` into ``sequence2``.
        """
        if sequence1 == sequence2:
            return 0
        if not sequence1:
            return len(sequence2)
        if not sequence2:
            return len(sequence1)

        previous_row = list(range(len(sequence2) + 1))
        for i, aa1 in enumerate(sequence1, start=1):
            current_row = [i]
            for j, aa2 in enumerate(sequence2, start=1):
                substitution_cost = 0 if aa1 == aa2 else 1
                current_row.append(
                    min(
                        previous_row[j] + 1,
                        current_row[j - 1] + 1,
                        previous_row[j - 1] + substitution_cost,
                    )
                )
            previous_row = current_row

        return previous_row[-1]

    @staticmethod
    def levenshtein_tier(distance: int) -> int:
        """Map a Levenshtein distance to a coarse PSA tier via ``LEVENSHTEIN_TIER_THRESHOLDS``.

        Args:
            distance: Levenshtein distance between two sequences.

        Returns:
            ``0`` for identical sequences, otherwise the tier whose threshold the
            distance falls under, or ``5`` if it exceeds every threshold.
        """
        if distance == 0:
            return 0
        for threshold, tier in LEVENSHTEIN_TIER_THRESHOLDS:
            if distance < threshold:
                return tier
        return 5

    @staticmethod
    def _trace_edit_operations(sequence1: str, sequence2: str) -> list[Dict[str, Any]]:
        """Recover the ordered substitution/insertion/deletion ops of an optimal edit path.

        Args:
            sequence1: First sequence.
            sequence2: Second sequence.

        Returns:
            Ordered list of edit-operation dicts (``type``, ``pos``, ``seq1_pos``,
            ``seq2_pos``, ``from``, ``to``) tracing one minimum-cost edit path from
            ``sequence1`` to ``sequence2``.
        """
        m = len(sequence1)
        n = len(sequence2)
        dp = [[0] * (n + 1) for _ in range(m + 1)]

        for i in range(1, m + 1):
            dp[i][0] = i
        for j in range(1, n + 1):
            dp[0][j] = j

        for i in range(1, m + 1):
            for j in range(1, n + 1):
                substitution_cost = 0 if sequence1[i - 1] == sequence2[j - 1] else 1
                dp[i][j] = min(
                    dp[i - 1][j] + 1,
                    dp[i][j - 1] + 1,
                    dp[i - 1][j - 1] + substitution_cost,
                )

        operations: list[Dict[str, Any]] = []
        i = m
        j = n
        while i > 0 or j > 0:
            if i > 0 and j > 0:
                substitution_cost = 0 if sequence1[i - 1] == sequence2[j - 1] else 1
                if dp[i][j] == dp[i - 1][j - 1] + substitution_cost:
                    if substitution_cost:
                        operations.append(
                            {
                                "type": "substitution",
                                "pos": i - 1,
                                "seq1_pos": i,
                                "seq2_pos": j,
                                "from": sequence1[i - 1],
                                "to": sequence2[j - 1],
                            }
                        )
                    i -= 1
                    j -= 1
                    continue

            if i > 0 and dp[i][j] == dp[i - 1][j] + 1:
                operations.append(
                    {
                        "type": "deletion",
                        "pos": i - 1,
                        "seq1_pos": i,
                        "seq2_pos": j,
                        "from": sequence1[i - 1],
                        "to": "",
                    }
                )
                i -= 1
                continue

            operations.append(
                {
                    "type": "insertion",
                    "pos": i,
                    "seq1_pos": i + 1,
                    "seq2_pos": j,
                    "from": "",
                    "to": sequence2[j - 1],
                }
            )
            j -= 1

        operations.reverse()
        return operations

    @staticmethod
    def _group_indel_operations(
        operations: list[Dict[str, Any]],
        op_type: str,
    ) -> list[Dict[str, Any]]:
        """Collapse consecutive single-residue insertions/deletions into multi-residue blocks.

        Args:
            operations: Edit operations as returned by ``_trace_edit_operations``.
            op_type: Either ``"insertion"`` or ``"deletion"``.

        Returns:
            List of grouped blocks (``type``, ``pos``, ``k``, ``sequence 1``, ``sequence 2``).
        """
        grouped: list[Dict[str, Any]] = []

        for op in operations:
            if op["type"] != op_type:
                continue

            if not grouped:
                grouped.append(
                    {
                        "type": op_type,
                        "pos": op["pos"],
                        "k": 1,
                        "sequence 1": op["from"],
                        "sequence 2": op["to"],
                    }
                )
                continue

            current = grouped[-1]
            same_anchor = op_type == "insertion" and op["pos"] == current["pos"]
            consecutive = op_type == "deletion" and op["pos"] == current["pos"] + current["k"]

            if same_anchor or consecutive:
                current["k"] += 1
                current["sequence 1"] += op["from"]
                current["sequence 2"] += op["to"]
            else:
                grouped.append(
                    {
                        "type": op_type,
                        "pos": op["pos"],
                        "k": 1,
                        "sequence 1": op["from"],
                        "sequence 2": op["to"],
                    }
                )

        return grouped

    @classmethod
    def collect_observed_changes_for_sequences(cls, sequence1: str, sequence2: str) -> Dict[str, Any]:
        """Trace and summarize every edit operation between two sequences.

        Args:
            sequence1: First sequence.
            sequence2: Second sequence.

        Returns:
            Dict with the raw ``edit_operations``, grouped ``substitutions``/
            ``insertions``/``deletions``, and their counts (plus a total
            ``change_count``).
        """
        operations = cls._trace_edit_operations(sequence1, sequence2)
        substitutions = [
            {
                "pos": op["pos"],
                "seq1_pos": op["seq1_pos"],
                "seq2_pos": op["seq2_pos"],
                "from": op["from"],
                "to": op["to"],
                "change": cls._format_substitution_change(op),
            }
            for op in operations
            if op["type"] == "substitution"
        ]
        insertions = cls._group_indel_operations(operations, "insertion")
        deletions = cls._group_indel_operations(operations, "deletion")

        return {
            "edit_operations": operations,
            "substitutions": substitutions,
            "insertions": insertions,
            "deletions": deletions,
            "substitution_count": len(substitutions),
            "insertion_count": len(insertions),
            "deletion_count": len(deletions),
            "change_count": len(operations),
        }
