"""Create a reproducible small, balanced visual-review subset before examining errors."""

import argparse
import csv
from dataclasses import asdict
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from ml.analysis import contact_sheet
from ml.datasets import balanced_subset, read_manifest


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, default=ROOT / "data/manifests/dataset_b.csv")
    parser.add_argument("--count", type=int, default=96)
    parser.add_argument("--seed", type=int, default=17)
    parser.add_argument("--output", type=Path, default=ROOT / "data/analysis_tags.csv")
    args = parser.parse_args()
    if args.output.exists():
        raise ValueError("Annotation file already exists; preserve reviewed tags or choose a new output.")
    rows = [{**asdict(record), "review_id": index + 1, "annotation_source": "", "notes": ""}
            for index, record in enumerate(balanced_subset(read_manifest(args.manifest), args.count, args.seed))]
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0])); writer.writeheader(); writer.writerows(rows)
    for index in range(0, len(rows), 16):
        contact_sheet(rows[index:index + 16], ROOT / f"artifacts/evaluation/review_{index // 16 + 1:02d}.jpg",
                      f"CNR-EXT visual review {index + 1}-{min(index + 16, len(rows))}", limit=16)
    print(f"Created {len(rows)} review records in {args.output}. Tags are blank until visually reviewed.")


if __name__ == "__main__":
    main()
