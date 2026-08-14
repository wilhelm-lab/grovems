from __future__ import annotations

from typing import Any, Dict, Tuple


class EventLabelingMixin:
    """Event-candidate collection, combination, and human-readable labeling/summaries."""

    def collect_event_candidates(self) -> list[Tuple[str, Any]]:
        """Every event candidate for the sequence pair: identity, or every changed run from the aligner.

        local_alignment_events_for always returns at least one event for a differing pair
        (proven: a run is guaranteed to exist for any changed column not claimed by a
        priority window), so there's no fallback branch to fall through to here.
        """
        if self.sequence1 == self.sequence2:
            return [("IDENTICAL", [])]

        return self.local_alignment_events_for(self.sequence1, self.sequence2)

    def event_positions(self, event_details: Any) -> Tuple[int, ...]:
        """Sorted 0-based positions an event (or list of events) covers."""
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

    def assign_multi_event_variant(self, candidates: list[Tuple[str, Any]]) -> None:
        """Summarize multiple event candidates and record them as the selected combined event.

        Exactly two components get a compact combined name, e.g. "SWAP + INSERTION",
        ordered by _label_name_priority; three or more collapse to "MULTI_EVENT".
        """
        component_events = [self.summarize_candidate(candidate) for candidate in candidates]
        self._debug("Assigning PSA multi-event variant: %s", component_events)

        combined_event_name = "MULTI_EVENT"
        if len(component_events) == 2 and all(isinstance(component, dict) for component in component_events):
            combined_names = [
                self._component_event_label(str(component.get("event", ""))) for component in component_events
            ]
            combined_names.sort(key=self._label_name_priority)
            combined_event_name = " + ".join(combined_names)

        self.select_event(combined_event_name, component_events)
        self.result.update(
            component_events=component_events,
            component_event_count=len(component_events),
        )

    @staticmethod
    def _format_substitution_change(change: Dict[str, Any]) -> str:
        """Render one substitution as e.g. ``"SUB(AB->BA)@5"`` for multi-residue blocks."""
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

    def event_change_summary(self, event_name: str, event_details: Any) -> str:
        """Render any selected event (incl. combined multi-events) as a summary string, e.g."INS(K)@8"."""
        if event_name == "IDENTICAL":
            return "NO_CHANGE"

        if event_name in {
            "SUBSTITUTION",
            "ISOBARIC-SUBSTITUTION",
            "N-TERM-MISMATCH",
            "C-TERM-MISMATCH",
        } and isinstance(event_details, list):
            changes = [self._format_substitution_change(change) for change in event_details if isinstance(change, dict)]
            return ",".join(changes) if changes else event_name

        if event_name in {"INSERTION", "DELETION", "INDEL"} and isinstance(event_details, list):
            changes = [self._format_indel_change(change) for change in event_details if isinstance(change, dict)]
            return ",".join(changes) if changes else "INDEL"

        if event_name in {"ADJACENT_SWAP", "BLOCK_SHUFFLE"} and isinstance(event_details, dict):
            position = int(event_details["pos"]) + 1
            sequence_1 = event_details.get("sequence 1", "")
            sequence_2 = event_details.get("sequence 2", "")
            action = "SWAP" if event_name == "ADJACENT_SWAP" else "SHUFFLE"
            return f"{action}({sequence_1}->{sequence_2})@{position}"

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

        return event_name

    @staticmethod
    def _component_event_label(event_name: str) -> str:
        """Map an internal event name to its short label used in combined names."""
        if event_name == "SUBSTITUTION":
            return "SUBSTITUTION"
        if event_name == "ISOBARIC-SUBSTITUTION":
            return "ISOBARIC-SUBSTITUTION"
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
            "N-TERM-MISMATCH": -2,
            "C-TERM-MISMATCH": -1,
            "IDENTICAL": 0,
            "SWAP": 1,
            "SHUFFLE": 2,
            "ISOBARIC-SUBSTITUTION": 3,
            "SUBSTITUTION": 4,
            "DELETION": 5,
            "INSERTION": 6,
            "INDEL": 7,
        }
        return order.get(label_name, 99)

    def label_event_name(self) -> str:
        """The selected event's name as it should appear in the final PSA label."""
        if " + " in self.result.selected_event or self.result.selected_event == "MULTI_EVENT":
            return self.result.selected_event

        return self._component_event_label(self.result.selected_event)
