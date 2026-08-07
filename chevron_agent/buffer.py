"""Small, explicit provisional storage for delayed consolidation outcomes."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class ProvisionalEntry:
    event_id: int
    created_step: int
    payload: Any


class BoundedProvisionalBuffer:
    """A deterministic FIFO buffer whose entries cannot mutate retained N."""

    def __init__(self, capacity: int) -> None:
        if capacity <= 0:
            raise ValueError("capacity must be positive")
        self.capacity = capacity
        self._entries: list[ProvisionalEntry] = []

    @property
    def entries(self) -> tuple[ProvisionalEntry, ...]:
        return tuple(self._entries)

    def __len__(self) -> int:
        return len(self._entries)

    def add(self, entry: ProvisionalEntry) -> ProvisionalEntry | None:
        if any(existing.event_id == entry.event_id for existing in self._entries):
            raise ValueError(f"duplicate event_id {entry.event_id}")
        evicted = None
        if len(self._entries) == self.capacity:
            evicted = self._entries.pop(0)
        self._entries.append(entry)
        return evicted

    def resolve(self, event_id: int) -> ProvisionalEntry | None:
        for index, entry in enumerate(self._entries):
            if entry.event_id == event_id:
                return self._entries.pop(index)
        return None
