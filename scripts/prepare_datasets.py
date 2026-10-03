"""Inspect actual sources, create grouped Dataset A splits and balanced CNR-EXT test data."""

import argparse
from collections import Counter
import csv
import hashlib
import json
from pathlib import Path
import sys
import zipfile

from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from ml.datasets import Record, balanced_subset, inspect_dataset_a, relative_path, split_dataset_a, write_manifest


def prepare_b(root: Path, count: int, seed: int):
    archive_path = root / "CNR-EXT-Patches-150x150.zip"
    csv_path = root / "CNRPark+EXT.csv"
    with zipfile.ZipFile(archive_path) as archive:
        names = archive.namelist()
        images = [name for name in names if name.lower().endswith(".jpg")]
        by_basename = {}
        for name in images:
            basename = Path(name).name
            if basename in by_basename:
                raise ValueError("CNR archive has ambiguous basenames; use complete path mapping.")
            by_basename[basename] = name
        candidates = []
        with csv_path.open(newline="", encoding="utf-8-sig") as stream:
            reader = csv.DictReader(stream)
            csv_columns = reader.fieldnames
            for row in reader:
                basename = Path(row["image_url"]).name
                if basename not in by_basename:
                    continue  # Preliminary CNRPark rows are not in the CNR-EXT archive.
                if row["occupancy"] not in ("0", "1"):
                    raise ValueError("Unexpected official CNR occupancy label.")
                day = f"{int(row['year']):04d}-{int(row['month']):02d}-{int(row['day']):02d}"
                weather = {"S": "SUNNY", "O": "OVERCAST", "R": "RAINY"}.get(row["weather"])
                if weather is None:
                    raise ValueError(f"Unknown official weather code: {row['weather']}")
                candidates.append(Record(by_basename[basename], int(row["occupancy"]), "dataset_b",
                                         camera=row["camera"], sequence_day=day, weather=weather,
                                         frame_token=row["datetime"], slot_token=row["slot_id"], split="external"))
        if len({record.file_path for record in candidates}) != len(candidates):
            raise ValueError("Official CSV contains duplicate crop records.")
        selected = balanced_subset(candidates, count, seed)
        for record in selected:
            member_name = record.file_path
            destination = (root / "subset" / member_name).resolve()
            if not destination.is_relative_to((root / "subset").resolve()):
                raise ValueError("Unsafe archive path.")
            destination.parent.mkdir(parents=True, exist_ok=True)
            if not destination.exists():
                destination.write_bytes(archive.read(member_name))
            with Image.open(destination) as image:
                image.verify()
            record.file_path = relative_path(destination)
            record.sha256 = hashlib.sha256(destination.read_bytes()).hexdigest()
        report = {"archive_jpeg_count": len(images), "csv_matched_count": len(candidates),
                  "csv_columns": csv_columns, "archive_non_image_examples": [n for n in names if not n.endswith('/') and not n.lower().endswith('.jpg')][:20],
                  "selected": len(selected), "class_counts": dict(Counter(record.label for record in selected)),
                  "camera_counts": dict(Counter(record.camera for record in selected)),
                  "weather_counts": dict(Counter(record.weather for record in selected)),
                  "day_count": len({record.sequence_day for record in selected}),
                  "selection": "balanced labels; seeded round-robin strata by weather, camera, day",
                  "lighting_occlusion_annotations": "not present in the official CSV; unknown until visual review"}
    return selected, report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-a", type=Path, default=ROOT / "data/raw/dataset_a")
    parser.add_argument("--dataset-b", type=Path, default=ROOT / "data/raw/dataset_b")
    parser.add_argument("--external-count", type=int, default=4000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--embargo", type=int, default=20)
    parser.add_argument("--skip-b", action="store_true", help="Prepare A while B is still downloading")
    args = parser.parse_args()
    output = ROOT / "data/manifests"
    records = inspect_dataset_a(args.dataset_a)
    split_report = split_dataset_a(records, args.seed, args.embargo)
    write_manifest(records, output / "dataset_a.csv")
    files = [path for path in args.dataset_a.rglob("*") if path.is_file()]
    report = {"dataset_a": {"source": "https://www.kaggle.com/datasets/iasadpanwhar/parking-lot-detection-counter",
                             "actual_root": str(args.dataset_a.resolve()), "raw_crops": len(records),
                             "class_counts": dict(Counter(record.label for record in records)),
                             "source_file_extensions": dict(Counter(path.suffix for path in files)),
                             "non_crop_files": [relative_path(path) for path in files if "clf-data" not in path.parts],
                             "frame_token_groups": len({record.frame_token for record in records}),
                             "slot_token_groups": len({record.slot_token for record in records}),
                             "split": split_report, "bounding_box_ground_truth": "none found; masks mark slots, not vehicle boxes"},
              "dataset_b": {"status": "not prepared"}, "label_convention": {"0": "VACANT", "1": "OCCUPIED"}}
    if not args.skip_b:
        external, report["dataset_b"] = prepare_b(args.dataset_b, args.external_count, args.seed)
        a_hashes = {record.sha256 for record in records if record.split != "excluded"}
        if a_hashes & {record.sha256 for record in external}:
            raise ValueError("Exact duplicate content between A and external B; remove before evaluating.")
        write_manifest(external, output / "dataset_b.csv")
    (output / "dataset_inventory.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2), flush=True)


if __name__ == "__main__":
    main()
