"""Readable vehicle boxes, parking states, and counts for the demo."""

import cv2
import numpy as np

from .detector import Detection
from .occupancy import OccupancyResult
from .parking_slots import ParkingSlot
from .geometry import polygon_rectangle_intersection_area


VACANT_COLOR = (65, 210, 65)
OCCUPIED_COLOR = (65, 65, 240)
VEHICLE_COLOR = (255, 200, 70)


def _dashboard_bounds(width, height, panel_width, panel_height, slots):
    """Choose the corner covering the least configured parking area."""
    margin = min(6, max(0, (min(width, height) - 1) // 2))
    panel_width = min(panel_width, width - 2 * margin)
    panel_height = min(panel_height, height - 2 * margin)
    candidates = [(margin, margin), (width - panel_width - margin, margin),
                  (margin, height - panel_height - margin),
                  (width - panel_width - margin, height - panel_height - margin)]
    rectangles = [(x, y, x + panel_width, y + panel_height) for x, y in candidates]
    return min(rectangles, key=lambda rectangle: sum(
        polygon_rectangle_intersection_area(slot.polygon, rectangle) for slot in slots))


def _label(frame, text, anchor, color, scale):
    height, width = frame.shape[:2]
    thickness = max(1, round(scale * 2))
    (text_width, text_height), baseline = cv2.getTextSize(
        text, cv2.FONT_HERSHEY_SIMPLEX, scale, thickness)
    x = max(0, min(int(anchor[0]), width - text_width - 8))
    y = max(text_height + 6, min(int(anchor[1]), height - baseline - 6))
    cv2.rectangle(frame, (x, y - text_height - 5),
                  (x + text_width + 8, y + baseline + 4), (20, 20, 20), -1)
    cv2.putText(frame, text, (x + 4, y), cv2.FONT_HERSHEY_SIMPLEX,
                scale, color, thickness, cv2.LINE_AA)


LARGE_LOT_SLOTS = 50


def default_show_ids(slot_count: int) -> bool:
    """Slot IDs clutter large lots; the polygon colour already carries the state."""
    return slot_count <= LARGE_LOT_SLOTS


def annotate_frame(frame: np.ndarray, detections: list[Detection],
                   slots: list[ParkingSlot], occupancy: OccupancyResult, *,
                   show_rois: bool = True, show_ids: bool | None = None,
                   show_boxes: bool = True,
                   engine_label: str = "YOLO11n + ROI") -> np.ndarray:
    annotated = frame.copy()
    height, width = frame.shape[:2]
    if show_ids is None:
        show_ids = default_show_ids(len(slots))
    scale = max(0.35, min(1.0, width / 1100))
    thickness = max(1, round(scale * 3))
    crowded = len(slots) > LARGE_LOT_SLOTS
    border = 1 if crowded else thickness
    if show_rois:
        tint = annotated.copy()
        for slot in slots:
            color = OCCUPIED_COLOR if occupancy.occupied[slot.slot_id] else VACANT_COLOR
            cv2.fillPoly(tint, [np.rint(slot.polygon).astype(np.int32)], color)
        cv2.addWeighted(tint, 0.12 if crowded else 0.18, annotated, 0.88 if crowded else 0.82, 0, annotated)

    for detection in (detections if show_boxes else []):
        x1, y1, x2, y2 = (round(v) for v in detection.bbox)
        cv2.rectangle(annotated, (x1, y1), (x2, y2), VEHICLE_COLOR, thickness)
        _label(annotated, f"{detection.class_name} {detection.confidence:.2f}",
               (x1, y1 - 8), VEHICLE_COLOR, scale)
    for slot in (slots if show_rois else []):
        color = OCCUPIED_COLOR if occupancy.occupied[slot.slot_id] else VACANT_COLOR
        cv2.polylines(annotated, [np.rint(slot.polygon).astype(np.int32)], True, color, border, cv2.LINE_AA)
    if show_ids:
        label_scale = 0.3 if crowded else scale
        for slot in slots:
            # Anchor to a real vertex so labels remain close to concave polygons too.
            anchor = slot.polygon[np.argmin(slot.polygon[:, 1])]
            _label(annotated, slot.label, (anchor[0], anchor[1] + 14 * label_scale),
                   (240, 240, 240), label_scale)

    # Two title lines stay readable on small videos without a full-width banner.
    lines = [("CloudForge", (255, 200, 70)),
             (engine_label, (240, 240, 240)),
             (f"TOTAL: {occupancy.total}", (240, 240, 240)),
             (f"OCCUPIED: {occupancy.occupied_count}", OCCUPIED_COLOR),
             (f"AVAILABLE: {occupancy.available_count}", VACANT_COLOR),
             (f"OCCUPANCY: {occupancy.occupancy_percent:.1f}%", (240, 240, 240))]
    dashboard_scale = max(0.3, scale)
    line_height = max(16, round(29 * dashboard_scale))
    dashboard_width = max(cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX,
                                         dashboard_scale, max(1, round(dashboard_scale * 2)))[0][0]
                          for text, _ in lines) + 20
    x1, y1, x2, y2 = _dashboard_bounds(width, height, dashboard_width,
                                      12 + line_height * len(lines), slots)
    # Slight transparency leaves context visible even when every corner has ROIs.
    panel = annotated[y1:y2, x1:x2]
    cv2.addWeighted(np.full_like(panel, 20), 0.9, panel, 0.1, 0, panel)
    for index, (text, color) in enumerate(lines, start=1):
        cv2.putText(annotated, text, (x1 + 6, y1 + line_height * index),
                    cv2.FONT_HERSHEY_SIMPLEX, dashboard_scale, color,
                    max(1, round(dashboard_scale * 2)), cv2.LINE_AA)
    return annotated
