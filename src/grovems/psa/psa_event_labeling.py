from __future__ import annotations

import logging
from typing import Any, Dict, Optional, Tuple

logger = logging.getLogger(__name__)

MAX_MULTI_AA_INDEL_SIZE = 3


class EventLabelingMixin:
    """Event-candidate collection, combination, and human-readable labeling/summaries."""

    def collect_observed_changes(self) -> Dict[str, Any]:
        """``collect_observed_changes_for_sequences`` for ``self.sequence1``/``self.sequence2``."""
        return self.collect_observed_changes_for_sequences(self.sequence1, self.sequence2)

    def observed_changes(self) -> Dict[str, Any]:
        """Cached (or freshly computed) observed-changes dict for the current sequence pair."""
        observed = self.result.details.get("observed_changes")
        if isinstance(observed, dict):
            return observed
        return self.collect_observed_changes()

    def substitution_event(
        self,
        observed_changes: Optional[Dict[str, Any]] = None,
    ) -> Optional[Tuple[str, list[Dict[str, Any]]]]:
        """Build a ``SUBSTITUTION`` event from every observed substitution, if any.

        Args:
            observed_changes: Precomputed observed-changes dict; computed fresh if not given.

        Returns:
            ``("SUBSTITUTION", details)`` with one detail entry per substitution, or
            ``None`` if there were no substitutions.
        """
        observed = observed_changes if observed_changes is not None else self.observed_changes()
        substitutions = observed["substitutions"]
        if not substitutions:
            return None

        details = [{"pos": sub["pos"], "from": sub["from"], "to": sub["to"]} for sub in substitutions]
        return ("SUBSTITUTION", details)

    @staticmethod
    def _indel_event_details(event: Dict[str, Any]) -> list[Dict[str, Any]]:
        """Reduce a grouped indel block to its reportable fields."""
        return [
            {
                "pos": event["pos"],
                "k": event["k"],
                "sequence 1": event["sequence 1"],
                "sequence 2": event["sequence 2"],
            }
        ]

    @staticmethod
    def _indel_event_name(event: Dict[str, Any]) -> str:
        """Name a grouped indel block as ``DELETION``, ``INSERTION``, or ``INDEL``."""
        if event["sequence 1"] and not event["sequence 2"]:
            return "DELETION"
        if event["sequence 2"] and not event["sequence 1"]:
            return "INSERTION"
        return "INDEL"

    def indel_events(
        self,
        observed_changes: Optional[Dict[str, Any]] = None,
    ) -> list[Tuple[str, list[Dict[str, Any]]]]:
        """Build one event per grouped indel block up to ``MAX_MULTI_AA_INDEL_SIZE`` residues.

        Args:
            observed_changes: Precomputed observed-changes dict; computed fresh if not given.

        Returns:
            List of ``(event_name, event_details)`` pairs, ordered by position
            (insertions before deletions at the same position).
        """
        observed = observed_changes if observed_changes is not None else self.observed_changes()
        indel_blocks = sorted(
            observed["insertions"] + observed["deletions"],
            key=lambda event: (int(event["pos"]), 0 if event["sequence 1"] else 1),
        )

        events: list[Tuple[str, list[Dict[str, Any]]]] = []
        for event in indel_blocks:
            k = int(event["k"])
            if k < 1 or k > MAX_MULTI_AA_INDEL_SIZE:
                continue

            events.append((self._indel_event_name(event), self._indel_event_details(event)))

        return events

    def collect_event_candidates(self) -> list[Tuple[str, Any]]:
        """Collect every event candidate explaining the current sequence pair's difference.

        Tries, in order: exact identity, alignment-based local events (swaps,
        shuffles, isobaric/local-rearrangement/plain substitutions, indels detected
        per changed run), then falls back to a coarser substitution + indel-block view
        if the alignment-based pass found nothing.

        Returns:
            List of ``(event_name, event_details)`` candidates (possibly more than one,
            which ``assign_multi_event_variant`` then combines into one label).
        """
        if self.sequence1 == self.sequence2:
            return [("IDENTICAL", [])]

        candidates = self.local_alignment_events_for(self.sequence1, self.sequence2)
        if candidates:
            return candidates

        observed_changes = self.observed_changes()
        substitution_event = self.substitution_event(observed_changes=observed_changes)
        if substitution_event is not None:
            candidates.append(substitution_event)

        candidates.extend(self.indel_events(observed_changes=observed_changes))

        return candidates

    def event_positions(self, event_details: Any) -> Tuple[int, ...]:
        """Extract every sequence position an event (or list of events) covers.

        Args:
            event_details: A single event-detail dict, or a list of them.

        Returns:
            Sorted tuple of covered 0-based positions.
        """
        positions: set[int] = set()

        if isinstance(event_details, dict):
            if "indices" in event_details:
                positions.update(int(idx) for idx in event_details["indices"])
            elif "pos" in event_details:
                start = int(event_details["pos"])
                width = int(event_details.get("k", 1))
                positions.update(range(start, start + max(width, 1)))
        elif isinstance(event_details, list):
            for item in event_details:
                if not isinstance(item, dict):
                    continue
                if "indices" in item:
                    positions.update(int(idx) for idx in item["indices"])
                elif "pos" in item:
                    start = int(item["pos"])
                    width = int(item.get("k", 1))
                    positions.update(range(start, start + max(width, 1)))

        return tuple(sorted(positions))

    def summarize_candidate(self, candidate: Tuple[str, Any]) -> Dict[str, Any]:
        """Reduce one ``(event_name, event_details)`` candidate to a compact summary dict."""
        event_name, event_details = candidate
        return {
            "event": event_name,
            "positions": list(self.event_positions(event_details)),
            "details": event_details,
        }

    def combined_event_name(self, component_events: list[Dict[str, Any]]) -> str:
        """Name the combination of exactly two component events, e.g. ``"SWAP + INSERTION"``.

        Args:
            component_events: Summarized candidates, as from ``summarize_candidate``.

        Returns:
            The two components' labels joined with ``" + "`` (priority-ordered), or
            ``"MULTI_EVENT"`` if there aren't exactly two, or either isn't a dict.
        """
        if len(component_events) != 2:
            return "MULTI_EVENT"

        combined_names = []
        for component in component_events:
            if not isinstance(component, dict):
                return "MULTI_EVENT"
            combined_names.append(self._component_event_label(str(component.get("event", ""))))

        combined_names.sort(key=self._label_name_priority)
        return " + ".join(combined_names)

    def assign_multi_event_variant(self, candidates: list[Tuple[str, Any]]) -> None:
        """Summarize multiple event candidates and record them as the selected combined event.

        Args:
            candidates: Event candidates, as from ``collect_event_candidates``.
        """
        component_events = [self.summarize_candidate(candidate) for candidate in candidates]
        self._debug("Assigning PSA multi-event variant: %s", component_events)
        combined_event_name = self.combined_event_name(component_events)
        self.select_event(combined_event_name, component_events)
        self.result.update(
            component_events=component_events,
            component_event_count=len(component_events),
        )

    @staticmethod
    def _format_substitution_change(change: Dict[str, Any]) -> str:
        """Render one substitution as e.g. ``"E7R"`` or ``"SUB(AB->BA)@5"`` for multi-residue blocks."""
        if "sequence 1" in change or "sequence 2" in change:
            position = int(change["pos"]) + 1
            sequence_1 = change.get("sequence 1", "")
            sequence_2 = change.get("sequence 2", "")
            return f"SUB({sequence_1}->{sequence_2})@{position}"
        return f"{change['from']}{int(change['pos']) + 1}{change['to']}"

    @staticmethod
    def _format_indel_change(change: Dict[str, Any]) -> str:
        """Render one indel block as e.g. ``"DEL(AB)@3"``, ``"INS(K)@8"``, or ``"INDEL(...)"."""
        position = int(change["pos"]) + 1
        sequence_1 = change.get("sequence 1", "")
        sequence_2 = change.get("sequence 2", "")
        if sequence_1 and not sequence_2:
            return f"DEL({sequence_1})@{position}"
        if sequence_2 and not sequence_1:
            return f"INS({sequence_2})@{position}"
        return f"INDEL({sequence_1}->{sequence_2})@{position}"

    @staticmethod
    def _format_shuffle_change(event_name: str, change: Dict[str, Any]) -> str:
        """Render an adjacent-swap or block-shuffle event as e.g. ``"SWAP(AB->BA)@5"``."""
        position = int(change["pos"]) + 1
        sequence_1 = change.get("sequence 1", "")
        sequence_2 = change.get("sequence 2", "")
        action = "SWAP" if event_name == "ADJACENT_SWAP" else "SHUFFLE"
        return f"{action}({sequence_1}->{sequence_2})@{position}"

    def event_change_summary(self, event_name: str, event_details: Any) -> str:
        """Render any selected event (including combined multi-events) as a change-summary string.

        Args:
            event_name: The selected event's name (may be a ``" + "``-joined combination).
            event_details: The selected event's details.

        Returns:
            A compact human-readable summary, e.g. ``"E7R"``, ``"INS(K)@8"``,
            ``"SWAP(ID->DI)@5"``, or a ``" + "``-joined combination thereof.
        """
        if event_name == "IDENTICAL":
            return "NO_CHANGE"

        if event_name in {"SUBSTITUTION", "ISOBARIC-SUBSTITUTION", "LOCAL-REARRANGEMENT"} and isinstance(
            event_details, list
        ):
            changes = [self._format_substitution_change(change) for change in event_details if isinstance(change, dict)]
            return ",".join(changes) if changes else event_name

        if event_name in {"INSERTION", "DELETION", "INDEL"} and isinstance(event_details, list):
            changes = [self._format_indel_change(change) for change in event_details if isinstance(change, dict)]
            return ",".join(changes) if changes else "INDEL"

        if event_name in {"ADJACENT_SWAP", "BLOCK_SHUFFLE"} and isinstance(event_details, dict):
            return self._format_shuffle_change(event_name, event_details)

        if (event_name == "MULTI_EVENT" or " + " in event_name) and isinstance(event_details, list):
            component_changes = []
            for component in event_details:
                if not isinstance(component, dict):
                    continue
                component_event = str(component.get("event", ""))
                component_details = component.get("details")
                component_changes.append(self.event_change_summary(component_event, component_details))
            summarized = [change for change in component_changes if change]
            return " + ".join(summarized) if summarized else "MULTI_EVENT"

        if event_name == "UNCLASSIFIED_VARIANT":
            return self.unclassified_change_summary()

        return event_name

    @staticmethod
    def _component_event_label(event_name: str) -> str:
        """Map an internal event name to its short label used in combined names."""
        if event_name == "SUBSTITUTION":
            return "SUBSTITUTION"
        if event_name == "ISOBARIC-SUBSTITUTION":
            return "ISOBARIC-SUBSTITUTION"
        if event_name == "LOCAL-REARRANGEMENT":
            return "LOCAL-REARRANGEMENT"
        if event_name in {"INSERTION", "DELETION"}:
            return event_name
        if event_name == "INDEL":
            return "INDEL"
        if event_name == "ADJACENT_SWAP":
            return "SWAP"
        if event_name == "BLOCK_SHUFFLE":
            return "SHUFFLE"
        if event_name == "IDENTICAL":
            return "IDENTICAL"
        return event_name

    @staticmethod
    def _label_name_priority(label_name: str) -> int:
        """Sort key controlling the order of components in a combined event name."""
        order = {
            "IDENTICAL": 0,
            "SWAP": 1,
            "SHUFFLE": 2,
            "ISOBARIC-SUBSTITUTION": 3,
            "LOCAL-REARRANGEMENT": 4,
            "SUBSTITUTION": 5,
            "DELETION": 6,
            "INSERTION": 7,
            "INDEL": 8,
        }
        return order.get(label_name, 99)

    def label_event_name(self) -> str:
        """The selected event's name as it should appear in the final PSA label."""
        if " + " in self.result.selected_event or self.result.selected_event == "MULTI_EVENT":
            return self.result.selected_event

        return self._component_event_label(self.result.selected_event)

    def unclassified_change_summary(self) -> str:
        """Best-effort change summary for sequence pairs no specific event rule matched.

        Returns:
            ``"NO_CHANGE"`` if identical, a joined substitution list if only
            substitutions were observed, a single indel description if only one indel
            block was observed, or ``"SEQUENCE_VARIANT"`` otherwise.
        """
        if self.sequence1 == self.sequence2:
            return "NO_CHANGE"

        observed = self.observed_changes()
        substitutions = observed["substitutions"]
        indels = observed["insertions"] + observed["deletions"]

        if substitutions and not indels:
            return ",".join(self._format_substitution_change(change) for change in substitutions)

        if len(indels) == 1 and not substitutions:
            return self._format_indel_change(indels[0])

        return "SEQUENCE_VARIANT"
