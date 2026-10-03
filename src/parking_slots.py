"""Load simple parking polygons expressed in original-frame pixel coordinates."""

from dataclasses import dataclass
import json
from pathlib import Path

import cv2
import numpy as np


@dataclass(frozen=True)
class ParkingSlot:
    slot_id: str
    label: str
    polygon: np.ndarray


@dataclass(frozen=True)
class ParkingConfiguration:
    slots: list[ParkingSlot]
    frame_width: int | None = None
    frame_height: int | None = None

    def validate_frame(self, width: int, height: int) -> None:
        if self.frame_width is not None and (width, height) != (self.frame_width, self.frame_height):
            raise ValueError(
                f"Parking ROIs were calibrated at {self.frame_width}x{self.frame_height}, "
                f"but video is {width}x{height}. Re-run ROI calibration for this video."
            )
        validate_frame_bounds(self.slots, width, height)


def _segments_intersect(a, b, c, d) -> bool:
    def cross(p, q, r):
        return float((q[0] - p[0]) * (r[1] - p[1]) -
                     (q[1] - p[1]) * (r[0] - p[0]))

    def on_segment(p, q, r):
        return (min(p[0], q[0]) <= r[0] <= max(p[0], q[0]) and
                min(p[1], q[1]) <= r[1] <= max(p[1], q[1]))

    values = cross(a, b, c), cross(a, b, d), cross(c, d, a), cross(c, d, b)
    u, v, w, x = values
    if ((u > 0 and v < 0) or (u < 0 and v > 0)) and \
            ((w > 0 and x < 0) or (w < 0 and x > 0)):
        return True
    return any(abs(value) < 1e-6 and on_segment(p, q, r)
               for value, p, q, r in ((u, a, b, c), (v, a, b, d),
                                       (w, c, d, a), (x, c, d, b)))


def _unique_object(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"Duplicate slot ID '{key}' in parking configuration.")
        result[key] = value
    return result


def load_parking_slots(path: str | Path) -> list[ParkingSlot]:
    """Backward-compatible list API for both legacy and metadata configurations."""
    return load_parking_config(path).slots


def load_parking_config(path: str | Path) -> ParkingConfiguration:
    path = Path(path)
    try:
        data = json.loads(path.read_text(encoding="utf-8-sig"), object_pairs_hook=_unique_object)
    except OSError as exc:
        raise ValueError(f"Cannot read parking slots '{path}': {exc}") from exc
    except json.JSONDecodeError as exc:
        raise ValueError(f"Malformed parking JSON '{path}' at line {exc.lineno}: {exc.msg}") from exc
    return parse_parking_config(data)


def parse_parking_config(data) -> ParkingConfiguration:
    """Shared validation for loaded JSON and interactive calibration edits."""
    if not isinstance(data, dict) or not data:
        raise ValueError("Parking configuration must be a non-empty object mapping slot IDs to polygons.")

    frame_width = frame_height = None
    # A legacy slot may itself be called 'slots'; only treat metadata keys as a
    # schema marker if they are not polygon lists.
    metadata_format = any(key in data and not isinstance(data[key], list)
                          for key in ("frame_width", "frame_height")) or (
                              isinstance(data.get("slots"), list) and
                              any(isinstance(item, dict) for item in data["slots"]))
    if metadata_format:
        frame_width, frame_height = data.get("frame_width"), data.get("frame_height")
        if any(type(value) is not int or value <= 0 for value in (frame_width, frame_height)):
            raise ValueError("frame_width and frame_height must both be positive integers.")
        entries = data.get("slots")
        if not isinstance(entries, list) or not entries:
            raise ValueError("Metadata configuration needs a non-empty 'slots' list.")
        slot_data = {}
        for entry in entries:
            if not isinstance(entry, dict) or "id" not in entry or "points" not in entry:
                raise ValueError("Each slot needs an 'id' and 'points'.")
            slot_id = entry["id"]
            if not isinstance(slot_id, str) or not slot_id.strip():
                raise ValueError("Parking slot IDs must be non-empty strings.")
            if slot_id in slot_data:
                raise ValueError(f"Duplicate slot ID '{slot_id}' in parking configuration.")
            slot_data[slot_id] = entry["points"]
        data = slot_data

    slots = []
    for index, (slot_id, points) in enumerate(data.items(), start=1):
        prefix = f"Slot '{slot_id}'"
        if not slot_id.strip():
            raise ValueError("Parking slot IDs must not be blank.")
        if (not isinstance(points, list) or len(points) < 3 or
                any(not isinstance(p, list) or len(p) != 2 or
                    any(isinstance(v, bool) or not isinstance(v, (int, float)) for v in p)
                    for p in points)):
            raise ValueError(f"{prefix} needs at least three [x, y] numeric vertices.")
        try:
            with np.errstate(over="ignore"):
                polygon = np.asarray(points, dtype=np.float32)
        except (ValueError, OverflowError) as exc:
            raise ValueError(f"{prefix} coordinates exceed the supported numeric range.") from exc
        if not np.isfinite(polygon).all() or (polygon < 0).any():
            raise ValueError(f"{prefix} coordinates must be finite, non-negative pixel values.")
        # Accept an optional closing vertex; OpenCV closes the contour itself.
        if np.array_equal(polygon[0], polygon[-1]):
            polygon = polygon[:-1]
        if len(polygon) < 3 or len(np.unique(polygon, axis=0)) != len(polygon):
            raise ValueError(f"{prefix} needs at least three distinct vertices without repeats.")
        for i in range(len(polygon)):
            for j in range(i + 1, len(polygon)):
                if j == i + 1 or (i == 0 and j == len(polygon) - 1):
                    continue
                if _segments_intersect(polygon[i], polygon[(i + 1) % len(polygon)],
                                       polygon[j], polygon[(j + 1) % len(polygon)]):
                    raise ValueError(f"{prefix} polygon crosses itself; order vertices around its boundary.")
        if abs(cv2.contourArea(polygon)) <= 1e-6:
            raise ValueError(f"{prefix} polygon must have non-zero area.")
        slots.append(ParkingSlot(slot_id, f"P{index}", polygon))
    configuration = ParkingConfiguration(slots, frame_width, frame_height)
    if frame_width is not None:
        configuration.validate_frame(frame_width, frame_height)
    return configuration


def validate_frame_bounds(slots: list[ParkingSlot], width: int, height: int) -> None:
    for slot in slots:
        if (slot.polygon[:, 0] >= width).any() or (slot.polygon[:, 1] >= height).any():
            raise ValueError(
                f"Slot '{slot.slot_id}' extends outside the {width}x{height} video. "
                "Use polygon coordinates from this video's original resolution."
            )
