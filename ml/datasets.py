"""Strict crop adapters and auditable splits; source data is never relabeled."""

from collections import Counter, defaultdict
import csv
from dataclasses import asdict, dataclass, fields
import hashlib
from pathlib import Path
import random

from PIL import Image
from torch.utils.data import Dataset

ROOT = Path(__file__).resolve().parents[1]
LABEL_NAMES = {0: "VACANT", 1: "OCCUPIED"}


def normalize_label(value) -> int:
    if isinstance(value, bool):
        raise ValueError("Boolean labels are ambiguous; supply an explicit occupancy label.")
    mapping = {"0": 0, "empty": 0, "vacant": 0, "free": 0,
               "1": 1, "not_empty": 1, "occupied": 1, "busy": 1}
    key = str(value).strip().lower()
    if key not in mapping:
        raise ValueError(f"Unknown parking occupancy label: {value!r}")
    return mapping[key]


@dataclass
class Record:
    file_path: str
    label: int
    dataset_source: str
    camera: str | None = None
    sequence_day: str | None = None
    weather: str | None = None
    lighting_tag: str | None = None
    occlusion_tag: str | None = None
    frame_token: str | None = None
    slot_token: str | None = None
    split: str = "unassigned"
    sha256: str | None = None


def relative_path(path: Path) -> str:
    path = path.resolve()
    return path.relative_to(ROOT).as_posix() if path.is_relative_to(ROOT) else str(path)


def inspect_dataset_a(root: Path) -> list[Record]:
    paths = sorted(root.rglob("*.jpg"))
    records = []
    for path in paths:
        if path.parent.name not in ("empty", "not_empty"):
            continue
        tokens = path.stem.split("_")
        if len(tokens) != 2 or not all(token.isdigit() for token in tokens):
            raise ValueError(f"Cannot group this Dataset A filename safely: {path.name}")
        with Image.open(path) as image:
            image.verify()
        records.append(Record(relative_path(path), normalize_label(path.parent.name), "dataset_a",
                              frame_token=tokens[0], slot_token=tokens[1],
                              sha256=hashlib.sha256(path.read_bytes()).hexdigest()))
    if not records or {record.label for record in records} != {0, 1}:
        raise ValueError("Dataset A needs both empty/ and not_empty/ labeled JPEG crops.")
    return records


def split_dataset_a(records: list[Record], seed: int = 42, embargo: int = 20) -> dict:
    """Hold out both opaque slot tokens and chronological frame-token ranges.

    These tokens are inferred grouping keys, not asserted camera/day annotations.
    Cross-products outside the corresponding split and boundary bands are excluded.
    """
    if embargo < 0:
        raise ValueError("Embargo must be non-negative.")
    rng = random.Random(seed)
    slot_labels = defaultdict(set)
    for record in records:
        slot_labels[record.slot_token].add(record.label)
    if any(len(labels) != 1 for labels in slot_labels.values()):
        raise ValueError("This Dataset A split expects the inspected constant-label slot-token groups.")
    slot_assignment = {}
    for label in (0, 1):
        slots = sorted(slot for slot, labels in slot_labels.items() if labels == {label})
        if len(slots) < 6:
            raise ValueError("Need at least six distinct slot-token groups per class for a held-out split.")
        rng.shuffle(slots)
        first, second = int(len(slots) * 0.70), int(len(slots) * 0.85)
        for index, slot in enumerate(slots):
            slot_assignment[slot] = "train" if index < first else "val" if index < second else "test"
    minimum = min(int(record.frame_token) for record in records)
    maximum = max(int(record.frame_token) for record in records)
    boundary1 = minimum + int((maximum - minimum) * 0.60)
    boundary2 = minimum + int((maximum - minimum) * 0.80)
    hashes = defaultdict(list)
    for record in records:
        frame = int(record.frame_token)
        frame_split = ("train" if frame <= boundary1 - embargo else
                       "val" if boundary1 + embargo <= frame <= boundary2 - embargo else
                       "test" if frame >= boundary2 + embargo else "excluded")
        record.split = frame_split if slot_assignment[record.slot_token] == frame_split else "excluded"
        hashes[record.sha256].append(record)
    # Any exact duplicate spanning partitions is excluded from all partitions.
    for duplicates in hashes.values():
        if len({record.split for record in duplicates if record.split != "excluded"}) > 1:
            for record in duplicates:
                record.split = "excluded"
    validate_no_leakage(records)
    for split in ("train", "val", "test"):
        if {record.label for record in records if record.split == split} != {0, 1}:
            raise ValueError(f"Split '{split}' lost a class; inspect grouping or reduce embargo.")
    return {"seed": seed, "frame_token_boundaries": [boundary1, boundary2],
            "embargo_token_units_each_side": embargo,
            "strategy": "disjoint stratified slot tokens AND separated chronological frame-token ranges",
            "token_semantics": "inferred from filenames; original video/camera/day provenance is absent",
            "counts": {split: dict(Counter(record.label for record in records if record.split == split))
                       for split in ("train", "val", "test", "excluded")}}


def validate_no_leakage(records: list[Record]) -> None:
    partitions = {split: [record for record in records if record.split == split]
                  for split in ("train", "val", "test")}
    for field in ("file_path", "sha256", "slot_token", "frame_token"):
        sets = {split: {getattr(record, field) for record in group if getattr(record, field) is not None}
                for split, group in partitions.items()}
        for left, right in (("train", "val"), ("train", "test"), ("val", "test")):
            if sets[left] & sets[right]:
                raise ValueError(f"Leakage in {field} between {left} and {right}.")


def balanced_subset(records: list[Record], total: int, seed: int = 42) -> list[Record]:
    if total < 2 or total % 2:
        raise ValueError("Balanced subset size must be a positive even number >= 2.")
    rng = random.Random(seed)
    chosen = []
    for label in (0, 1):
        strata = defaultdict(list)
        for record in records:
            if record.label == label:
                strata[(record.weather, record.camera, record.sequence_day)].append(record)
        if sum(map(len, strata.values())) < total // 2:
            raise ValueError(f"Not enough class {label} samples for the requested subset.")
        keys = sorted(strata, key=str)
        rng.shuffle(keys)
        for values in strata.values():
            rng.shuffle(values)
        selected = []
        while len(selected) < total // 2:
            for key in keys:
                if strata[key]:
                    selected.append(strata[key].pop())
                if len(selected) == total // 2:
                    break
        chosen.extend(selected)
    return sorted(chosen, key=lambda record: record.file_path)


def write_manifest(records: list[Record], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=[field.name for field in fields(Record)])
        writer.writeheader()
        writer.writerows(asdict(record) for record in records)


def read_manifest(path: str | Path) -> list[Record]:
    with Path(path).open(newline="", encoding="utf-8-sig") as stream:
        records = []
        for row in csv.DictReader(stream):
            row = {key: value or None for key, value in row.items()}
            row["label"] = normalize_label(row["label"])
            records.append(Record(**row))
    if len({record.file_path for record in records}) != len(records):
        raise ValueError("Manifest contains duplicate file paths.")
    return records


class ParkingCropDataset(Dataset):
    def __init__(self, records: list[Record], split: str, image_size: int = 128):
        from .transforms import build_transforms
        if split not in ("train", "val", "test", "external"):
            raise ValueError("Unknown dataset split.")
        self.records = [record for record in records if record.split == split]
        self.training = split == "train"
        self.transform = build_transforms(training=self.training, image_size=image_size)
        if not self.records:
            raise ValueError(f"No samples in '{split}' split.")

    def __len__(self):
        return len(self.records)

    def __getitem__(self, index):
        record = self.records[index]
        with Image.open(ROOT / record.file_path) as image:
            tensor = self.transform(image.convert("RGB"))
        return tensor, record.label
