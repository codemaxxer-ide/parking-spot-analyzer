"""Stream a local video through detection, occupancy, annotation, and MP4 writing."""

from dataclasses import dataclass
import math
from pathlib import Path
import tempfile
import warnings

import cv2

from .detector import VehicleDetector
from .occupancy import calculate_occupancy, validate_occupancy_options
from .parking_slots import load_parking_config
from .temporal import OccupancySmoother
from .visualization import annotate_frame

ENGINE_LABELS = {"yolo": "YOLO11n + ROI", "classifier": "Slot Classifier - MobileNetV2"}


@dataclass(frozen=True)
class VideoSummary:
    output_path: Path
    frames_written: int
    fps: float
    width: int
    height: int
    total_spaces: int
    occupied_last_frame: int
    available_last_frame: int
    inference_frames: int = 0
    slot_probabilities: dict | None = None


def process_video(input_path: str | Path, output_path: str | Path,
                  slots_path: str | Path, *, model_path: str = "yolo11n.pt",
                  confidence: float = 0.35, device: str | None = None,
                  image_size: int = 640, overwrite: bool = False,
                  detector=None, occupancy_method: str = "center",
                  overlap_threshold: float = 0.20, smoothing_window: int = 1,
                  smoothing_required: int = 1, process_every: int = 1,
                  progress_callback=None, engine: str = "yolo", classifier=None,
                  classification_threshold: float = 0.5, show_rois: bool = True,
                  show_ids: bool | None = None, show_boxes: bool = True) -> VideoSummary:
    if engine not in ENGINE_LABELS:
        raise ValueError("Engine must be 'yolo' or 'classifier'.")
    if not math.isfinite(classification_threshold) or not 0 < classification_threshold < 1:
        raise ValueError("Classification threshold must be between 0 and 1.")
    input_path, output_path = Path(input_path).resolve(), Path(output_path).resolve()
    if input_path == output_path or (output_path.exists() and input_path.exists() and
                                     input_path.samefile(output_path)):
        raise ValueError("Input and output must be different files.")
    if not input_path.is_file():
        raise ValueError(f"Input video does not exist: {input_path}")
    if output_path.suffix.lower() != ".mp4":
        raise ValueError("Output must have an .mp4 extension.")
    if output_path.exists() and not overwrite:
        raise ValueError(f"Output already exists: {output_path}. Use --overwrite to replace it.")
    if not math.isfinite(confidence) or not 0 <= confidence <= 1:
        raise ValueError("Confidence must be between 0 and 1.")
    if image_size <= 0:
        raise ValueError("Inference image size must be positive.")
    validate_occupancy_options(occupancy_method, overlap_threshold)
    smoother = OccupancySmoother(smoothing_window, smoothing_required)
    if type(process_every) is not int or process_every < 1:
        raise ValueError("process_every must be a positive integer.")
    configuration = load_parking_config(slots_path)
    slots = configuration.slots
    capture = cv2.VideoCapture(str(input_path))
    writer = None
    temporary_path = None
    try:
        if not capture.isOpened():
            raise RuntimeError(f"OpenCV cannot open input video: {input_path}")
        success, frame = capture.read()
        if not success or frame is None or frame.size == 0:
            raise RuntimeError("Input video has no readable frames.")
        height, width = frame.shape[:2]
        reported_width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
        reported_height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
        if (reported_width, reported_height) != (width, height):
            warnings.warn("Video dimension metadata differs from decoded frames; using decoded resolution.")
        if width % 2 or height % 2:
            raise ValueError("MP4 encoding needs even frame dimensions to preserve resolution. "
                             f"Resize the {width}x{height} source to even dimensions first.")
        configuration.validate_frame(width, height)
        fps = capture.get(cv2.CAP_PROP_FPS)
        if not math.isfinite(fps) or fps <= 0:
            fps = 30.0
            warnings.warn("Input FPS is missing/invalid; output uses 30 FPS.")
        frame_count = capture.get(cv2.CAP_PROP_FRAME_COUNT)
        expected_frames = int(frame_count) if math.isfinite(frame_count) and frame_count > 0 else 0
        if engine == "classifier":
            if classifier is None:
                from .slot_classifier import SlotClassifier
                classifier = SlotClassifier(device=device or "cpu")
        elif detector is None:
            detector = VehicleDetector(model_path, confidence, device, image_size)
        probabilities = None
        output_path.parent.mkdir(parents=True, exist_ok=True)
        # Publish only after the writer closes and the output can be decoded.
        with tempfile.NamedTemporaryFile(prefix=".parking-", suffix=".mp4",
                                         dir=output_path.parent, delete=False) as temporary:
            temporary_path = Path(temporary.name)
        writer = cv2.VideoWriter(str(temporary_path), cv2.VideoWriter_fourcc(*"mp4v"),
                                 fps, (width, height))
        if not writer.isOpened():
            raise RuntimeError("OpenCV cannot initialize the MP4 writer (mp4v codec). Check output permissions/codecs.")
        frames_written = 0
        inference_frames = 0
        while True:
            if frame.shape[:2] != (height, width):
                raise RuntimeError("Video frame resolution changed during processing.")
            if frames_written % process_every == 0:
                if engine == "classifier":
                    detections = []
                    raw, probabilities = classifier.classify(frame, slots, classification_threshold)
                else:
                    detections = detector.detect(frame)
                    raw = calculate_occupancy(slots, detections, method=occupancy_method,
                                              overlap_threshold=overlap_threshold)
                occupancy = smoother.update(raw)
                inference_frames += 1
            # Keep every output frame. Skipped detections hold the latest boxes
            # and stable state; they do not contribute extra smoothing votes.
            writer.write(annotate_frame(frame, detections, slots, occupancy, show_rois=show_rois,
                                        show_ids=show_ids, show_boxes=show_boxes,
                                        engine_label=ENGINE_LABELS[engine]))
            frames_written += 1
            if progress_callback is not None:
                progress_callback({"frames_written": frames_written,
                                   "expected_frames": expected_frames,
                                   "occupied": dict(occupancy.occupied)})
            if frames_written % 100 == 0:
                print(f"Processed {frames_written} frames", flush=True)
            success, frame = capture.read()
            if not success or frame is None or frame.size == 0:
                if expected_frames and frames_written < expected_frames:
                    raise RuntimeError(f"Video read failed after {frames_written} of {expected_frames} "
                                       "reported frames; incomplete output was not published.")
                break
        writer.release()
        writer = None
        check = cv2.VideoCapture(str(temporary_path))
        try:
            success, first_output = check.read()
            if not success or first_output is None or first_output.shape[:2] != (height, width):
                raise RuntimeError("Written MP4 could not be decoded at the expected resolution.")
        finally:
            check.release()
        temporary_path.replace(output_path)
        return VideoSummary(output_path, frames_written, fps, width, height,
                            occupancy.total, occupancy.occupied_count, occupancy.available_count,
                            inference_frames, probabilities)
    finally:
        capture.release()
        if writer is not None:
            writer.release()
        if temporary_path is not None:
            temporary_path.unlink(missing_ok=True)
