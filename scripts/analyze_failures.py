"""Generate condition tables, contact sheets, and a summary from actual predictions."""

import argparse
from collections import Counter
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from ml.analysis import condition_metrics, contact_sheet, read_analysis_tags, read_predictions, write_table
from ml.plots import condition_plot


def metric_line(metrics):
    return " | ".join(f"{key.capitalize()}: {metrics[key] * 100:.3f}%" for key in ("accuracy", "precision", "recall", "f1"))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tags", type=Path, default=ROOT / "data/analysis_tags.csv")
    args = parser.parse_args()
    metrics_dir, artifacts_dir = ROOT / "metrics", ROOT / "artifacts/evaluation"
    tags = read_analysis_tags(args.tags)
    results, rows = {}, []
    for model in ("baseline", "mobilenet"):
        metrics_path = metrics_dir / f"{model}_metrics.json"
        if not metrics_path.exists():
            continue
        results[model] = json.loads(metrics_path.read_text())
        for split in ("test", "external"):
            predictions_path = metrics_dir / f"predictions_{model}_{split}.csv"
            if not predictions_path.exists():
                continue
            for row in read_predictions(predictions_path):
                row["model"] = model
                if row["file_path"] in tags:
                    for field in ("lighting_tag", "occlusion_tag", "annotation_source", "notes"):
                        row[field] = tags[row["file_path"]].get(field, "")
                rows.append(row)
    if not rows:
        raise ValueError("No measured prediction CSVs found. Evaluate trained models first.")
    external_paths = {row["file_path"] for row in rows if row["dataset_source"] == "dataset_b"}
    if set(tags) - external_paths:
        raise ValueError("Condition tags contain images outside the evaluated external set.")
    weather = condition_metrics(rows, "weather")
    lighting = condition_metrics(rows, "lighting_tag")
    occlusion = condition_metrics(rows, "occlusion_tag")
    write_table(weather + lighting, metrics_dir / "lighting_analysis.csv")
    write_table(occlusion, metrics_dir / "occlusion_analysis.csv")
    # Weather is an official metadata condition; it is not relabeled as lighting.
    condition_plot(weather, artifacts_dir / "weather_analysis.png", "External accuracy by official weather metadata")
    condition_plot(lighting, artifacts_dir / "lighting_analysis.png", "External accuracy on the visually reviewed lighting subset")
    condition_plot(occlusion, artifacts_dir / "occlusion_analysis.png", "External accuracy on the visually reviewed occlusion subset")
    correct = []
    for model in results:
        for source in ("dataset_a", "dataset_b"):
            for label in (0, 1):
                correct += [row for row in rows if row["model"] == model and row["dataset_source"] == source
                            and row["label"] == row["prediction"] == label][:2]
    failures = [row for row in rows if row["label"] != row["prediction"]]
    failures.sort(key=lambda row: (row.get("occlusion_tag") in ("PARTIAL", "HEAVY") or
                                   row.get("lighting_tag") in ("SHADOW", "DARK"), row["confidence"]), reverse=True)
    contact_sheet(correct, artifacts_dir / "correct_predictions.jpg", "Correct VACANT and OCCUPIED predictions")
    # Ensure both false-VACANT and false-OCCUPIED categories are represented when present.
    failure_examples = []
    for model in results:
        for label in (0, 1):
            failure_examples += [row for row in failures if row["model"] == model and row["label"] == label][:3]
    contact_sheet(failure_examples, artifacts_dir / "failure_cases.jpg", "False VACANT and false OCCUPIED predictions")
    contact_sheet([row for row in failures if row.get("lighting_tag") == "SHADOW"],
                  artifacts_dir / "shadow_failures.jpg", "Measured errors in visually reviewed SHADOW crops")
    contact_sheet([row for row in failures if row.get("occlusion_tag") in ("PARTIAL", "HEAVY")],
                  artifacts_dir / "occlusion_failures.jpg", "Measured errors in visually reviewed occluded crops")
    write_table([{key: row.get(key, "") for key in ("file_path", "model", "dataset_source", "label", "prediction",
                "confidence", "weather", "lighting_tag", "occlusion_tag", "annotation_source")}
                 for row in failures], metrics_dir / "failure_examples.csv")
    inventory = json.loads((ROOT / "data/manifests/dataset_inventory.json").read_text())
    a, b = inventory["dataset_a"], inventory["dataset_b"]
    counts = a["split"]["counts"]
    a_vacant = sum(counts[split].get("0", 0) for split in ("train", "val", "test"))
    a_occupied = sum(counts[split].get("1", 0) for split in ("train", "val", "test"))
    lines = ["# CloudForge measured ML results", "", "## DATASETS", "",
             f"Dataset A: Parking Lot Detection Counter. Raw: {a['raw_crops']} crops; "
             f"used: {a_vacant + a_occupied} (VACANT {a_vacant}, OCCUPIED {a_occupied}).",
             f"Train: {sum(counts['train'].values())}; validation: {sum(counts['val'].values())}; "
             f"test: {sum(counts['test'].values())}; excluded: {sum(counts['excluded'].values())}.",
             f"Dataset B: CNR-EXT subset of CNRPark+EXT. External test: {b['selected']} "
             f"(VACANT {b['class_counts']['0']}, OCCUPIED {b['class_counts']['1']}); "
             f"{len(b['camera_counts'])} cameras; {b['day_count']} days.", "",
             "OCCUPIED (1) is the positive class. Confusion matrices have true-label rows, predicted-label columns, class order [VACANT, OCCUPIED].",
             "", "## BASELINE CNN", ""]
    for name, title in (("baseline", "BASELINE CNN"), ("mobilenet", "MOBILENETV2 TRANSFER LEARNING")):
        if name not in results:
            lines += [f"{title}: NOT MEASURED."]
            continue
        if name == "mobilenet":
            lines += ["", f"## {title}", ""]
        result = results[name]
        lines += [f"Dataset A held-out: {metric_line(result['in_domain'])}",
                  f"Epochs completed: {result['training']['epochs_completed']}; "
                  f"best validation-loss checkpoint: epoch {result['training']['best_epoch']}.",
                  f"Confusion matrix: {result['in_domain']['confusion_matrix']}."]
        if "external" in result:
            lines += [f"Dataset B external: {metric_line(result['external'])}",
                      f"External confusion matrix: {result['external']['confusion_matrix']}."]
    lines += ["", "## CROSS-DATASET TEST", ""]
    for name, result in results.items():
        if "external" in result:
            lines += [f"{name}: Dataset A F1 {result['in_domain']['f1'] * 100:.2f}%; "
                      f"Dataset B F1 {result['external']['f1'] * 100:.2f}%. No Dataset B retraining or threshold selection."]
    lines += ["", "## LIGHTING FINDINGS", "",
              f"Visual-review subset: {len(tags)} unique external images, selected using seed 17 before inspecting model errors.",
              "Tags are subjective assistant visual review of source crops, not official annotations. "
              "Weather categories are provided by the dataset; SUNNY is not equated with BRIGHT or SHADOW."]
    for row in weather + lighting:
        lines += [f"{row['model']} / {row['dimension']} / {row['condition']}: "
                  f"{row['correct']}/{row['sample_count']} correct ({row['accuracy'] * 100:.2f}%)." +
                  (" SMALL SUBSET (<30)." if row['small_sample_warning'] else "")]
    lines += ["", "## OCCLUSION FINDINGS", "",
              "Crop-level foreground intrusion from branches/posts/other objects was visually tagged. "
              "Cutting a car at the crop edge alone is not treated as physical occlusion. "
              "An almost-black crop has unknown occlusion and is excluded from occlusion categories."]
    for row in occlusion:
        lines += [f"{row['model']} / {row['condition']}: {row['correct']}/{row['sample_count']} correct "
                  f"({row['accuracy'] * 100:.2f}%)." + (" SMALL SUBSET (<30)." if row['small_sample_warning'] else "")]
    lines += ["", "## YOLO", "", "Model: YOLO11n pretrained.", "Role: vehicle detection in the unchanged video pipeline.",
              "mAP: NOT MEASURED ON A CUSTOM BOUNDING-BOX DATASET.",
              "Slot masks, occupancy labels, and crop coordinates are not vehicle bounding-box ground truth.",
              "", "## LIMITATIONS", "",
              "Dataset A is a short, repetitive scene with constant labels within every slot-token group. "
              "Disjoint slot/frame groups and temporal embargo reduce leakage, but the test is not an unseen parking lot. "
              "Its high accuracy must be interpreted alongside external results.",
              "Dataset B crops have different viewpoints and aspect ratios; any domain-shift cause is a hypothesis, "
              "not established by these experiments. Nearby external frames and slot views are correlated.",
              "128x128 square resizing was used for CPU speed for both models; official pretrained ImageNet evaluation uses 224x224. "
              "Source-label issues and almost-black images are retained, not silently corrected.",
              "Condition tags are subjective, small, and may overlap. These classifier metrics do not measure YOLO/video occupancy accuracy."]
    (artifacts_dir / "results_summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"Generated measured summary: {artifacts_dir / 'results_summary.md'}")
    print("Reviewed lighting counts:", dict(Counter(tag['lighting_tag'] for tag in tags.values() if tag['lighting_tag'])))
    print("Reviewed occlusion counts:", dict(Counter(tag['occlusion_tag'] for tag in tags.values() if tag['occlusion_tag'])))


if __name__ == "__main__":
    main()
