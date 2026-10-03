"""Replaceable center-point occupancy engine, independent of YOLO inference."""

from dataclasses import dataclass
from collections.abc import Iterable
import math

import cv2

from .detector import Detection
from .parking_slots import ParkingSlot
from .geometry import polygon_rectangle_intersection_area


@dataclass(frozen=True)
class OccupancyResult:
    occupied: dict[str, bool]

    @property
    def total(self) -> int:
        return len(self.occupied)

    @property
    def occupied_count(self) -> int:
        return sum(self.occupied.values())

    @property
    def available_count(self) -> int:
        return self.total - self.occupied_count

    @property
    def occupancy_percent(self) -> float:
        return 100 * self.occupied_count / self.total if self.total else 0.0


def validate_occupancy_options(method: str, overlap_threshold: float) -> None:
    if method not in ("center", "overlap"):
        raise ValueError("Occupancy method must be 'center' or 'overlap'.")
    if not math.isfinite(overlap_threshold) or not 0 < overlap_threshold <= 1:
        raise ValueError("Overlap threshold must be a finite number greater than 0 and at most 1.")


def calculate_occupancy(slots: list[ParkingSlot], detections: Iterable[Detection], *,
                        method: str = "center", overlap_threshold: float = 0.20) -> OccupancyResult:
    validate_occupancy_options(method, overlap_threshold)
    if method == "overlap":
        boxes = [detection.bbox for detection in detections]
        states = {}
        for slot in slots:
            area = abs(cv2.contourArea(slot.polygon))
            if area <= 0:
                raise ValueError(f"Slot '{slot.slot_id}' polygon must have non-zero area.")
            states[slot.slot_id] = any(
                polygon_rectangle_intersection_area(slot.polygon, box) / area >= overlap_threshold
                for box in boxes
            )
        return OccupancyResult(states)
    centers = [detection.center for detection in detections]
    # A center on an edge is inside. Each polygon contributes at most one count.
    return OccupancyResult({
        slot.slot_id: any(cv2.pointPolygonTest(slot.polygon, center, False) >= 0
                          for center in centers)
        for slot in slots
    })
