"""Recompute saved metrics and verify experiment inputs and presentation artifacts."""

from collections import Counter
import csv
import hashlib
import json
import math
from pathlib import Path
import platform
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import cv2
import matplotlib
import numpy as np
from PIL import Image, __version__ as pillow_version
import torch
import torchvision

from ml.analysis import condition_metrics, read_analysis_tags, read_predictions
from ml.datasets import read_manifest, validate_no_leakage
from ml.metrics import classification_metrics


def file_hash(path):
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def main():
    a_path, b_path = (ROOT / "data/manifests" / name for name in ("dataset_a.csv", "dataset_b.csv"))
    a, b = read_manifest(a_path), read_manifest(b_path)
    validate_no_leakage(a)
    if {row.sha256 for row in a} & {row.sha256 for row in b}:
        raise ValueError("A/B exact content overlap")
    if any(row.split != "external" for row in b):
        raise ValueError("Dataset B must remain exclusively external")
    if Counter(row.label for row in b) != Counter({0: 2000, 1: 2000}):
        raise ValueError("Unexpected measured external subset")
    tags = read_analysis_tags(ROOT / "data/analysis_tags.csv")
    predictions, checks = [], []
    for name, filename in (("baseline", "baseline_cnn.pt"), ("mobilenet", "mobilenetv2_parking.pt")):
        result = json.loads((ROOT / "metrics" / f"{name}_metrics.json").read_text())
        checkpoint_hash = file_hash(ROOT / "models" / filename)
        if result["training"]["training_sources"] != ["dataset_a"]:
            raise ValueError("Unexpected training source")
        if result["training"]["manifest_sha256"] != file_hash(a_path):
            raise ValueError("Training manifest mismatch")
        for split, key, manifest_path, records in (("test", "in_domain", a_path, a), ("external", "external", b_path, b)):
            rows = read_predictions(ROOT / "metrics" / f"predictions_{name}_{split}.csv")
            expected = {row.file_path: row.label for row in records if row.split == split}
            if len(rows) != len(expected) or {row["file_path"]: row["label"] for row in rows} != expected:
                raise ValueError("Prediction paths/labels mismatch")
            measured = classification_metrics([row["label"] for row in rows], [row["prediction"] for row in rows])
            for field, value in measured.items():
                if result[key][field] != value:
                    raise ValueError(f"Recomputed {name}/{split}/{field} mismatch")
            if result[key]["checkpoint_sha256"] != checkpoint_hash or result[key]["manifest_sha256"] != file_hash(manifest_path):
                raise ValueError("Evaluation provenance mismatch")
            for row in rows:
                row["model"] = name
                if row["file_path"] in tags:
                    row.update({field: tags[row["file_path"]].get(field, "") for field in ("lighting_tag", "occlusion_tag")})
                predictions.append(row)
            checks.append(f"{name}/{split}: all metrics recomputed from {len(rows)} predictions")
    external_paths = {row.file_path for row in b}
    if set(tags) - external_paths:
        raise ValueError("Review tags outside external subset")
    for filename, fields in (("lighting_analysis.csv", ("weather", "lighting_tag")), ("occlusion_analysis.csv", ("occlusion_tag",))):
        expected = [row for field in fields for row in condition_metrics(predictions, field)]
        with (ROOT / "metrics" / filename).open(newline="") as stream:
            actual = list(csv.DictReader(stream))
        if len(actual) != len(expected):
            raise ValueError("Condition row count mismatch")
        for saved, measured in zip(actual, expected):
            for key, value in measured.items():
                match = math.isclose(float(saved[key]), value, abs_tol=1e-12) if isinstance(value, float) else saved[key] == str(value)
                if not match:
                    raise ValueError(f"Condition metric mismatch: {filename}/{key}")
        checks.append(f"{filename}: condition counts recomputed; unknown tags omitted")
    expected_artifacts = [
        "results_summary.md", "baseline_confusion_matrix.png", "mobilenet_confusion_matrix.png",
        "baseline_external_confusion_matrix.png", "mobilenet_external_confusion_matrix.png",
        "model_comparison.png", "cross_dataset_comparison.png", "training_curve_baseline.png",
        "training_curve_mobilenet.png", "lighting_analysis.png", "occlusion_analysis.png", "weather_analysis.png",
        "correct_predictions.jpg", "failure_cases.jpg", "shadow_failures.jpg", "occlusion_failures.jpg",
    ]
    artifacts = []
    for filename in expected_artifacts:
        path = ROOT / "artifacts/evaluation" / filename
        if not path.is_file() or not path.stat().st_size:
            raise ValueError(f"Missing measured artifact: {filename}")
        if path.suffix in (".png", ".jpg"):
            with Image.open(path) as image:
                image.verify()
        artifacts.append({"file": filename, "bytes": path.stat().st_size, "sha256": file_hash(path)})
    report = {
        "status": "passed", "checks": checks,
        "dataset_a_split_counts": dict(Counter(row.split for row in a)),
        "dataset_b_count": len(b), "reviewed_unique_crops": len(tags),
        "runtime": {"python": platform.python_version(), "platform": platform.platform(), "torch": torch.__version__,
                    "torchvision": torchvision.__version__, "numpy": np.__version__, "pillow": pillow_version,
                    "opencv": cv2.__version__, "matplotlib": matplotlib.__version__, "cuda_available": torch.cuda.is_available()},
        "artifacts": artifacts,
    }
    (ROOT / "metrics/audit.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print("PASS: leakage/provenance checks, all saved classification and condition metrics, and 16 required artifacts")


if __name__ == "__main__":
    main()
