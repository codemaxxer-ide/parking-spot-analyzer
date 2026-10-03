"""Vehicle detection only; parking occupancy belongs to occupancy.py."""

from dataclasses import dataclass
import math

import numpy as np


VEHICLE_CLASSES = {"car", "motorcycle", "bus", "truck"}


@dataclass(frozen=True)
class Detection:
    bbox: tuple[float, float, float, float]
    class_name: str
    confidence: float

    @property
    def center(self) -> tuple[float, float]:
        x1, y1, x2, y2 = self.bbox
        return ((x1 + x2) / 2, (y1 + y2) / 2)


class VehicleDetector:
    def __init__(self, model_path: str = "yolo11n.pt", confidence: float = 0.35,
                 device: str | None = None, image_size: int = 640):
        if not math.isfinite(confidence) or not 0 <= confidence <= 1:
            raise ValueError("Confidence must be a finite number between 0 and 1.")
        if image_size <= 0:
            raise ValueError("Inference image size must be positive.")
        # Keep geometry tests independent of model loading and weight downloads.
        from ultralytics import YOLO

        self.model = YOLO(model_path, task="detect")
        if self.model.task != "detect":
            raise ValueError("Use a pretrained YOLO detection model, not classification/segmentation.")
        self.class_ids = [class_id for class_id, name in self.model.names.items()
                          if name in VEHICLE_CLASSES]
        if not self.class_ids:
            raise ValueError("Model has no car, motorcycle, bus, or truck classes.")
        self.confidence = confidence
        self.device = device
        self.image_size = image_size

    def detect(self, frame: np.ndarray) -> list[Detection]:
        result = self.model.predict(
            source=frame, conf=self.confidence, classes=self.class_ids,
            device=self.device, imgsz=self.image_size, verbose=False,
        )[0]
        if result.boxes is None:
            return []
        # Transfer all box data once, rather than synchronizing the GPU per box.
        coordinates = result.boxes.xyxy.cpu().numpy()
        classes = result.boxes.cls.cpu().numpy()
        confidences = result.boxes.conf.cpu().numpy()
        detections = []
        for bbox, class_id, confidence in zip(coordinates, classes, confidences):
            name = result.names[int(class_id)]
            if name in VEHICLE_CLASSES and float(confidence) >= self.confidence:
                detections.append(Detection(tuple(float(v) for v in bbox), name, float(confidence)))
        return detections
