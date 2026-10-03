"""Evaluate best classifiers on separate in-domain and external test sets."""

import argparse
import csv
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from ml.evaluate import evaluate_checkpoint
from ml.plots import comparison_plots, confusion_plot, training_plot


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--only", choices=("baseline", "mobilenet", "both"), default="both")
    parser.add_argument("--dataset-a", type=Path, default=ROOT / "data/manifests/dataset_a.csv")
    parser.add_argument("--dataset-b", type=Path, default=ROOT / "data/manifests/dataset_b.csv")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--threads", type=int, default=4)
    args = parser.parse_args()
    metrics_dir, plots_dir = ROOT / "metrics", ROOT / "artifacts/evaluation"
    metrics_dir.mkdir(exist_ok=True); plots_dir.mkdir(parents=True, exist_ok=True)
    results = {}
    names = ("baseline", "mobilenet") if args.only == "both" else (args.only,)
    for name in names:
        checkpoint = ROOT / "models" / ("baseline_cnn.pt" if name == "baseline" else "mobilenetv2_parking.pt")
        in_domain = evaluate_checkpoint(checkpoint, args.dataset_a, "test", metrics_dir / f"predictions_{name}_test.csv",
                                        device_name=args.device, threads=args.threads)
        result = {"in_domain": in_domain,
                  "training": json.loads(checkpoint.with_suffix(".training.json").read_text())}
        if args.dataset_b.exists():
            result["external"] = evaluate_checkpoint(checkpoint, args.dataset_b, "external",
                                                       metrics_dir / f"predictions_{name}_external.csv",
                                                       device_name=args.device, threads=args.threads)
        else:
            result["external_status"] = "BLOCKED: Dataset B manifest is missing"
        (metrics_dir / f"{name}_metrics.json").write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
        results[name] = result
        confusion_plot(in_domain, plots_dir / f"{name}_confusion_matrix.png", f"{name}: Dataset A (n={in_domain['sample_count']})")
        if "external" in result:
            confusion_plot(result["external"], plots_dir / f"{name}_external_confusion_matrix.png",
                           f"{name}: Dataset B (n={result['external']['sample_count']})")
        training_plot(checkpoint.with_suffix(".history.csv"), plots_dir / f"training_curve_{name}.png", f"{name} training")
        print(name, json.dumps(result), flush=True)
    # Include already completed other model results without rerunning them.
    for name in ("baseline", "mobilenet"):
        path = metrics_dir / f"{name}_metrics.json"
        if name not in results and path.exists():
            results[name] = json.loads(path.read_text())
    rows = [{"model": name, "dataset": split, **{key: metrics[key] for key in
             ("sample_count", "vacant_count", "occupied_count", "accuracy", "precision", "recall", "f1")}}
            for name, result in results.items() for split, metrics in result.items() if split in ("in_domain", "external")]
    with (metrics_dir / "model_comparison.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0])); writer.writeheader(); writer.writerows(rows)
    comparison_plots(results, plots_dir)


if __name__ == "__main__":
    main()
