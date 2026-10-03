"""Train the simple CNN on the prepared Dataset A split."""

from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from ml.train import train_model


def main(model_type="baseline"):
    import argparse
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, default=ROOT / "data/manifests/dataset_a.csv")
    parser.add_argument("--output", type=Path, default=ROOT / "models" /
                        ("baseline_cnn.pt" if model_type == "baseline" else "mobilenetv2_parking.pt"))
    parser.add_argument("--epochs", type=int, default=8 if model_type == "baseline" else 5)
    parser.add_argument("--image-size", type=int, default=128)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--learning-rate", type=float, default=0.001)
    parser.add_argument("--patience", type=int, default=3)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--fine-tune-epochs", type=int, default=0)
    args = parser.parse_args()
    train_model(model_type, args.manifest, args.output, epochs=args.epochs,
                image_size=args.image_size, batch_size=args.batch_size,
                learning_rate=args.learning_rate, patience=args.patience, seed=args.seed,
                device_name=args.device, threads=args.threads, fine_tune_epochs=args.fine_tune_epochs)


if __name__ == "__main__":
    main()
