"""Bounded per-slot votes over detection observations, without vehicle tracking."""

from collections import deque

from .occupancy import OccupancyResult


class OccupancySmoother:
    def __init__(self, window: int = 5, required: int = 3):
        if type(window) is not int or type(required) is not int or not 1 <= required <= window:
            raise ValueError("Smoothing requires integer values with 1 <= required <= window.")
        self.window = window
        self.required = required
        self.history = {}

    def update(self, raw: OccupancyResult) -> OccupancyResult:
        # Drop removed IDs if the helper is reused with a different configuration.
        self.history = {slot_id: self.history.get(slot_id, deque(maxlen=self.window))
                        for slot_id in raw.occupied}
        for slot_id, occupied in raw.occupied.items():
            self.history[slot_id].append(bool(occupied))
        # Startup has only the observations actually received; no invented votes.
        # A slot becomes occupied once 'required' positives have been observed.
        return OccupancyResult({slot_id: sum(history) >= self.required
                                for slot_id, history in self.history.items()})
