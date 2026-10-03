"""OpenCV polygon calibration with a scaled preview and original-pixel storage."""

from dataclasses import dataclass
import json
from pathlib import Path
import tempfile

import cv2
import numpy as np

from .parking_slots import ParkingSlot, parse_parking_config


WINDOW_NAME = "CloudForge - Parking ROI Calibration"
CONTROLS_HEIGHT = 112


@dataclass(frozen=True)
class DisplayTransform:
    frame_width: int
    frame_height: int
    display_width: int
    display_height: int

    @classmethod
    def fit(cls, width: int, height: int, max_width: int = 1280, max_height: int = 720):
        if min(width, height, max_width, max_height) <= 0:
            raise ValueError("Frame and preview dimensions must be positive.")
        scale = min(1.0, max_width / width, max_height / height)
        return cls(width, height, max(1, round(width * scale)), max(1, round(height * scale)))

    def to_original(self, x: int, y: int) -> list[int] | None:
        if not (0 <= x < self.display_width and 0 <= y < self.display_height):
            return None  # Footer and out-of-image clicks never become vertices.
        return [min(self.frame_width - 1, round(x * self.frame_width / self.display_width)),
                min(self.frame_height - 1, round(y * self.frame_height / self.display_height))]

    def to_display(self, points) -> np.ndarray:
        return np.rint(np.asarray(points, np.float64) *
                       [self.display_width / self.frame_width,
                        self.display_height / self.frame_height]).astype(np.int32)


def read_reference_frame(input_path: str | Path) -> np.ndarray:
    input_path = Path(input_path)
    if not input_path.is_file():
        raise ValueError(f"Input video does not exist: {input_path}")
    capture = cv2.VideoCapture(str(input_path))
    try:
        if not capture.isOpened():
            raise RuntimeError(f"OpenCV cannot open input video: {input_path}")
        # Bounded retries can advance past a damaged first frame without hanging
        # forever on an empty/undecodable source.
        for _ in range(120):
            success, frame = capture.read()
            if success and frame is not None and frame.size:
                return frame
        raise RuntimeError("Input has no valid frame within the first 120 read attempts.")
    finally:
        capture.release()


def draw_reference(frame: np.ndarray, slots: list[ParkingSlot]) -> np.ndarray:
    reference = frame.copy()
    scale = max(0.4, min(1.2, frame.shape[1] / 1000))
    crowded = len(slots) > 50  # large lots: thin borders, no per-slot labels
    for slot in slots:
        polygon = np.rint(slot.polygon).astype(np.int32)
        cv2.polylines(reference, [polygon], True, (50, 230, 80), 1 if crowded else max(2, round(scale * 3)), cv2.LINE_AA)
        if crowded:
            continue
        x, y = polygon[np.argmin(polygon[:, 1])]
        y = min(frame.shape[0] - 5, max(round(25 * scale), int(y) + round(25 * scale)))
        (label_width, label_height), baseline = cv2.getTextSize(
            slot.label, cv2.FONT_HERSHEY_SIMPLEX, scale, 2)
        x = max(0, min(int(x), frame.shape[1] - label_width - 10))
        cv2.rectangle(reference, (x, y - label_height - 5),
                      (x + label_width + 8, y + baseline + 3), (20, 20, 20), -1)
        cv2.putText(reference, slot.label, (x + 4, y), cv2.FONT_HERSHEY_SIMPLEX,
                    scale, (50, 230, 80), 2, cv2.LINE_AA)
    return reference


class CalibrationState:
    def __init__(self, transform: DisplayTransform):
        self.transform = transform
        self.slots = []
        self.current_points = []
        self.message = "Click the corners of P1 in boundary order."
        self.dirty = False

    def payload(self):
        return {"frame_width": self.transform.frame_width,
                "frame_height": self.transform.frame_height, "slots": self.slots}

    def add_point(self, x: int, y: int) -> None:
        point = self.transform.to_original(x, y)
        if point is not None:
            self.current_points.append(point)
            self.dirty = True
            self.message = f"P{len(self.slots) + 1}: {len(self.current_points)} points. ENTER to complete."

    def undo_point(self) -> None:
        if self.current_points:
            self.current_points.pop()
            self.dirty = True
        self.message = f"Current polygon: {len(self.current_points)} points."

    def delete_slot(self) -> None:
        if self.slots:
            removed = self.slots.pop()
            self.dirty = True
            self.message = f"Deleted {removed['id']}. Next slot: P{len(self.slots) + 1}."
        else:
            self.message = "No completed slots to delete."

    def complete_slot(self) -> bool:
        slot = {"id": f"P{len(self.slots) + 1}", "points": [p[:] for p in self.current_points]}
        candidate = {**self.payload(), "slots": self.slots + [slot]}
        try:
            parse_parking_config(candidate)
        except ValueError as exc:
            self.message = str(exc)
            return False
        self.slots.append(slot)
        self.current_points.clear()
        self.dirty = True
        self.message = f"Completed {slot['id']}. Start P{len(self.slots) + 1}, or S to save."
        return True

    def save(self, output_path: str | Path, reference_path: str | Path, frame: np.ndarray) -> None:
        if self.current_points:
            raise ValueError("Finish the current polygon with ENTER, or undo its points before saving.")
        configuration = parse_parking_config(self.payload())
        configuration.validate_frame(frame.shape[1], frame.shape[0])
        output_path, reference_path = Path(output_path).resolve(), Path(reference_path).resolve()
        if output_path.suffix.lower() != ".json" or reference_path.suffix.lower() not in (".jpg", ".jpeg"):
            raise ValueError("Configuration output must be .json and reference output must be .jpg/.jpeg.")
        temporary_paths = []
        try:
            for path in (output_path, reference_path):
                path.parent.mkdir(parents=True, exist_ok=True)
                with tempfile.NamedTemporaryFile(prefix=".roi-", suffix=path.suffix,
                                                 dir=path.parent, delete=False) as temporary:
                    temporary_paths.append(Path(temporary.name))
            json_temp, image_temp = temporary_paths
            json_temp.write_text(json.dumps(self.payload(), indent=2, allow_nan=False) + "\n", encoding="utf-8")
            if not cv2.imwrite(str(image_temp), draw_reference(frame, configuration.slots)):
                raise RuntimeError("OpenCV could not export the parking reference JPEG.")
            # Prepare both files before replacing an existing configuration.
            image_temp.replace(reference_path)
            json_temp.replace(output_path)
        finally:
            for path in temporary_paths:
                path.unlink(missing_ok=True)
        self.dirty = False
        self.message = f"Saved {len(self.slots)} slots. Q / ESC to exit, or continue editing."

    def render(self, frame: np.ndarray) -> np.ndarray:
        transform = self.transform
        preview = cv2.resize(frame, (transform.display_width, transform.display_height))
        # Reuse reference drawing after transforming completed polygons.
        display_slots = [ParkingSlot(slot["id"], slot["id"],
                                     transform.to_display(slot["points"]).astype(np.float32))
                         for slot in self.slots]
        preview = draw_reference(preview, display_slots)
        if self.current_points:
            points = transform.to_display(self.current_points)
            cv2.polylines(preview, [points], False, (0, 220, 255), 2, cv2.LINE_AA)
            for index, point in enumerate(points, start=1):
                cv2.circle(preview, tuple(point), 4, (0, 220, 255), -1, cv2.LINE_AA)
                cv2.putText(preview, str(index), tuple(point + [5, -5]),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 220, 255), 1, cv2.LINE_AA)
        # Controls live below the image so no parking content is covered.
        canvas_width = max(transform.display_width, 700)
        canvas = np.full((transform.display_height + CONTROLS_HEIGHT, canvas_width, 3), 20, np.uint8)
        canvas[:transform.display_height, :transform.display_width] = preview
        lines = ["CloudForge - Parking ROI Calibration",
                 "Click: add | ENTER: complete | U: undo point | D/BACKSPACE: delete slot",
                 "S: save JSON + reference | Q / ESC: exit | Coordinates use original pixels",
                 self.message]
        for index, text in enumerate(lines):
            cv2.putText(canvas, text, (8, transform.display_height + 23 + index * 26),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, (240, 240, 240), 1, cv2.LINE_AA)
        return canvas


def run_calibration(input_path: str | Path, output_path: str | Path, reference_path: str | Path,
                    *, max_width: int = 1280, max_height: int = 720) -> CalibrationState:
    if Path(input_path).resolve() in (Path(output_path).resolve(), Path(reference_path).resolve()):
        raise ValueError("Calibration output must not overwrite the input video.")
    if max_width < 700 or max_height <= CONTROLS_HEIGHT:
        raise ValueError("Preview max width must be >= 700 and max height must be > 112.")
    frame = read_reference_frame(input_path)
    transform = DisplayTransform.fit(frame.shape[1], frame.shape[0], max_width,
                                     max_height - CONTROLS_HEIGHT)
    state = CalibrationState(transform)

    def on_mouse(event, x, y, flags, parameter):
        if event == cv2.EVENT_LBUTTONDOWN:
            state.add_point(x, y)

    created = False
    try:
        # AUTOSIZE prevents arbitrary OS window resizing from changing the image
        # scale. The preview is explicitly resized once; transform uses its exact
        # dimensions, including independent rounding of width and height.
        cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_AUTOSIZE)
        created = True
        cv2.setMouseCallback(WINDOW_NAME, on_mouse)
        while True:
            cv2.imshow(WINDOW_NAME, state.render(frame))
            key = cv2.waitKey(30) & 0xFF
            if key in (27, ord("q"), ord("Q")):
                break
            try:
                visible = cv2.getWindowProperty(WINDOW_NAME, cv2.WND_PROP_VISIBLE)
            except cv2.error:
                break  # Some GUI backends throw after the user clicks close.
            if visible < 1:
                break
            if key in (10, 13):
                state.complete_slot()
            elif key in (ord("u"), ord("U")):
                state.undo_point()
            elif key in (8, 127, ord("d"), ord("D")):
                state.delete_slot()
            elif key in (ord("s"), ord("S")):
                try:
                    state.save(output_path, reference_path, frame)
                    print(f"Saved {len(state.slots)} slots at {frame.shape[1]}x{frame.shape[0]} to {output_path}", flush=True)
                    print(f"Reference frame: {reference_path}", flush=True)
                except (ValueError, OSError, RuntimeError, cv2.error) as exc:
                    state.message = str(exc)
                    print(f"Save failed: {exc}", flush=True)
    except cv2.error as exc:
        raise RuntimeError("OpenCV calibration window failed. Run on a desktop with opencv-python "
                           "(not opencv-python-headless). " + str(exc)) from exc
    finally:
        if created:
            try:
                cv2.destroyWindow(WINDOW_NAME)
            except cv2.error:
                pass  # The user may already have closed the window.
    if state.dirty:
        print("Exited; unsaved edits were discarded.")
    return state
