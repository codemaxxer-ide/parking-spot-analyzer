"""Held-out and external inference; test results never choose model checkpoints."""

from dataclasses import asdict
import csv
import hashlib
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from .baseline_cnn import BaselineCNN
from .datasets import ParkingCropDataset, read_manifest, validate_no_leakage
from .metrics import classification_metrics
from .transfer_model import ParkingMobileNet


def evaluate_checkpoint(checkpoint_path: Path, manifest: Path, split: str,
                        predictions_path: Path, *, device_name: str = "auto", threads: int = 4):
    if split not in ("test", "external"):
        raise ValueError("Report evaluation uses held-out test or external samples only.")
    torch.set_num_threads(threads)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu") if device_name == "auto" else torch.device(device_name)
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=True)
    if checkpoint["training_sources"] != ["dataset_a"]:
        raise ValueError("This cross-dataset report requires training on Dataset A only.")
    if split == "test" and hashlib.sha256(manifest.read_bytes()).hexdigest() != checkpoint["manifest_sha256"]:
        raise ValueError("Dataset A manifest changed after training; refusing mismatched test evaluation.")
    records = read_manifest(manifest)
    if split == "test":
        validate_no_leakage(records)
    model_type = checkpoint["model_type"]
    model = BaselineCNN() if model_type == "baseline" else ParkingMobileNet(pretrained=False)
    model.load_state_dict(checkpoint["state_dict"])
    model.to(device).eval()
    dataset = ParkingCropDataset(records, split, checkpoint["image_size"])
    if split == "external" and any(record.dataset_source != "dataset_b" for record in dataset.records):
        raise ValueError("External evaluation must contain Dataset B only.")
    loader = DataLoader(dataset, batch_size=64, shuffle=False, num_workers=0)
    predictions, confidences, positive_probabilities, truth = [], [], [], []
    with torch.inference_mode():
        for inputs, labels in loader:
            probabilities = model(inputs.to(device)).softmax(1).cpu()
            confidence, predicted = probabilities.max(1)
            truth.extend(labels.tolist())
            predictions.extend(predicted.tolist())
            confidences.extend(confidence.tolist())
            positive_probabilities.extend(probabilities[:, 1].tolist())
    metrics = classification_metrics(truth, predictions)
    metrics.update(model=model_type, split=split, checkpoint_best_epoch=checkpoint["best_epoch"],
                   checkpoint_sha256=hashlib.sha256(checkpoint_path.read_bytes()).hexdigest(),
                   manifest_sha256=hashlib.sha256(manifest.read_bytes()).hexdigest())
    predictions_path.parent.mkdir(parents=True, exist_ok=True)
    rows = [{**asdict(record), "prediction": prediction, "confidence": confidence,
             "probability_occupied": positive_probability}
            for record, prediction, confidence, positive_probability in
            zip(dataset.records, predictions, confidences, positive_probabilities)]
    with predictions_path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    return metrics
