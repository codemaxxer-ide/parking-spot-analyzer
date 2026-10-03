"""Classifier-backed occupancy: crop every parking slot and classify all crops in one batch."""

from pathlib import Path

import cv2
import numpy as np

from .occupancy import OccupancyResult
from .parking_slots import ParkingSlot

EXPECTED_LABELS = {0: "VACANT", 1: "OCCUPIED"}
DEFAULT_CHECKPOINT = Path(__file__).resolve().parents[1] / "models" / "mobilenetv2_parking.pt"


def slot_crop(frame: np.ndarray, slot: ParkingSlot, mask_outside: bool = False) -> np.ndarray:
    """Axis-aligned bounding crop, matching how the training crops were produced.

    mask_outside blacks out pixels outside the polygon; it is off by default
    because the model never saw masked crops during training.
    """
    height, width = frame.shape[:2]
    x1, y1 = np.floor(slot.polygon.min(axis=0)).astype(int)
    x2, y2 = np.ceil(slot.polygon.max(axis=0)).astype(int)
    x1, y1 = max(0, x1), max(0, y1)
    x2, y2 = min(width, max(x2, x1 + 1)), min(height, max(y2, y1 + 1))
    crop = frame[y1:y2, x1:x2]
    if mask_outside:
        mask = np.zeros(crop.shape[:2], np.uint8)
        cv2.fillPoly(mask, [np.rint(slot.polygon - [x1, y1]).astype(np.int32)], 255)
        crop = cv2.bitwise_and(crop, crop, mask=mask)
    return crop


class SlotClassifier:
    """Loads the trained MobileNetV2 once; classify() runs one no-grad batch per frame."""

    def __init__(self, checkpoint_path=DEFAULT_CHECKPOINT, device: str = "cpu",
                 batch_size: int = 128, mask_outside: bool = False):
        import torch
        from ml.transfer_model import ParkingMobileNet
        from ml.transforms import build_transforms

        checkpoint_path = Path(checkpoint_path)
        if not checkpoint_path.is_file():
            raise FileNotFoundError(f"Slot classifier checkpoint missing: {checkpoint_path}")
        checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=True)
        label_map = {int(k): v for k, v in checkpoint.get("label_map", {}).items()}
        if label_map != EXPECTED_LABELS or checkpoint.get("model_type") != "mobilenet":
            raise ValueError(f"Unexpected checkpoint labels {label_map}; expected {EXPECTED_LABELS}.")
        self.torch, self.device = torch, device
        self.batch_size, self.mask_outside = batch_size, mask_outside
        self.image_size = int(checkpoint["image_size"])
        self.transform = build_transforms(training=False, image_size=self.image_size)
        self.model = ParkingMobileNet(pretrained=False)
        self.model.load_state_dict(checkpoint["state_dict"])
        self.model.to(device).eval()

    def preprocess(self, frame: np.ndarray, slots: list[ParkingSlot]):
        """BGR frame -> one normalised (N, 3, S, S) tensor of RGB slot crops."""
        from PIL import Image

        tensors = [self.transform(Image.fromarray(cv2.cvtColor(
            slot_crop(frame, slot, self.mask_outside), cv2.COLOR_BGR2RGB))) for slot in slots]
        return self.torch.stack(tensors)

    def occupied_probabilities(self, frame: np.ndarray, slots: list[ParkingSlot]) -> np.ndarray:
        batch = self.preprocess(frame, slots)
        outputs = []
        with self.torch.inference_mode():
            for start in range(0, len(batch), self.batch_size):
                logits = self.model(batch[start:start + self.batch_size].to(self.device))
                outputs.append(self.torch.softmax(logits, dim=1)[:, 1].cpu())
        return self.torch.cat(outputs).numpy()

    def classify(self, frame: np.ndarray, slots: list[ParkingSlot], threshold: float = 0.5):
        """Return (raw OccupancyResult, {slot_id: P(occupied)}) for one frame."""
        probabilities = self.occupied_probabilities(frame, slots)
        return (OccupancyResult({slot.slot_id: bool(p >= threshold) for slot, p in zip(slots, probabilities)}),
                {slot.slot_id: float(p) for slot, p in zip(slots, probabilities)})
