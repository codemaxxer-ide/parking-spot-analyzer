"""Short, seeded training on Dataset A only, with validation-loss checkpointing."""

import csv
import hashlib
import json
import os
from pathlib import Path
import random
import time

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader

from .baseline_cnn import BaselineCNN
from .datasets import ParkingCropDataset, read_manifest, validate_no_leakage
from .transfer_model import ParkingMobileNet


def seed_everything(seed: int):
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.use_deterministic_algorithms(True)


def train_model(model_type: str, manifest: Path, output: Path, *, epochs: int = 8,
                image_size: int = 128, batch_size: int = 32, learning_rate: float = 0.001,
                patience: int = 3, seed: int = 42, device_name: str = "auto",
                threads: int = 4, fine_tune_epochs: int = 0):
    if model_type not in ("baseline", "mobilenet") or min(epochs, batch_size, patience, threads) < 1:
        raise ValueError("Invalid model/training options.")
    if learning_rate <= 0 or fine_tune_epochs < 0 or (model_type == "baseline" and fine_tune_epochs):
        raise ValueError("Invalid learning rate or fine-tuning schedule.")
    seed_everything(seed)
    torch.set_num_threads(threads)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu") if device_name == "auto" else torch.device(device_name)
    records = read_manifest(manifest)
    if any(record.dataset_source != "dataset_a" for record in records if record.split in ("train", "val")):
        raise ValueError("Primary experiments must train/validate on Dataset A only.")
    validate_no_leakage(records)
    train_data = ParkingCropDataset(records, "train", image_size)
    val_data = ParkingCropDataset(records, "val", image_size)
    if any({record.label for record in data.records} != {0, 1} for data in (train_data, val_data)):
        raise ValueError("Train and validation must each include both occupancy classes.")
    generator = torch.Generator().manual_seed(seed)
    train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True, generator=generator, num_workers=0)
    val_loader = DataLoader(val_data, batch_size=batch_size, shuffle=False, num_workers=0)
    model = (BaselineCNN() if model_type == "baseline" else ParkingMobileNet(pretrained=True)).to(device)
    loss_function = nn.CrossEntropyLoss()
    output.parent.mkdir(parents=True, exist_ok=True)
    history, best_loss, best_epoch = [], float("inf"), 0
    manifest_hash = hashlib.sha256(manifest.read_bytes()).hexdigest()
    started = time.monotonic()
    stages = [("baseline" if model_type == "baseline" else "head", epochs, learning_rate)]
    if fine_tune_epochs:
        stages.append(("fine_tune_last_3_blocks", fine_tune_epochs, learning_rate / 10))
    config = {"model_type": model_type, "image_size": image_size, "batch_size": batch_size,
              "requested_epochs": epochs, "requested_fine_tune_epochs": fine_tune_epochs,
              "learning_rate": learning_rate, "patience": patience, "seed": seed,
              "device": str(device), "threads": threads, "manifest_sha256": manifest_hash,
              "train_count": len(train_data), "validation_count": len(val_data),
              "pretrained_weights": "IMAGENET1K_V2" if model_type == "mobilenet" else None,
              "label_map": {0: "VACANT", 1: "OCCUPIED"}, "training_sources": ["dataset_a"],
              "augmentation": "train only: scale 0.90-1, rotation +/-7 deg, mild color jitter, 15% blur"}
    print(json.dumps(config), flush=True)
    for stage, stage_epochs, lr in stages:
        if stage.startswith("fine_tune"):
            model.load_state_dict(torch.load(output, map_location=device, weights_only=True)["state_dict"])
            model.set_fine_tuning(3)
        optimizer = torch.optim.AdamW((parameter for parameter in model.parameters() if parameter.requires_grad), lr=lr)
        stale = 0
        for _ in range(stage_epochs):
            epoch_started = time.monotonic()
            model.train()
            training_loss = 0.0
            for inputs, labels in train_loader:
                inputs, labels = inputs.to(device), labels.to(device)
                optimizer.zero_grad(set_to_none=True)
                loss = loss_function(model(inputs), labels)
                if not torch.isfinite(loss):
                    raise RuntimeError("Non-finite training loss; no valid result can be reported.")
                loss.backward()
                optimizer.step()
                training_loss += loss.item() * len(labels)
            model.eval()
            validation_loss, correct = 0.0, 0
            with torch.inference_mode():
                for inputs, labels in val_loader:
                    inputs, labels = inputs.to(device), labels.to(device)
                    logits = model(inputs)
                    validation_loss += loss_function(logits, labels).item() * len(labels)
                    correct += (logits.argmax(1) == labels).sum().item()
            row = {"epoch": len(history) + 1, "stage": stage, "learning_rate": lr,
                   "train_loss": training_loss / len(train_data), "validation_loss": validation_loss / len(val_data),
                   "validation_accuracy": correct / len(val_data), "seconds": time.monotonic() - epoch_started}
            history.append(row)
            print(f"Epoch {row['epoch']} [{stage}]: train_loss={row['train_loss']:.5f}, "
                  f"val_loss={row['validation_loss']:.5f}, val_accuracy={row['validation_accuracy']:.4f}, "
                  f"seconds={row['seconds']:.1f}", flush=True)
            if row["validation_loss"] < best_loss:
                best_loss, best_epoch, stale = row["validation_loss"], row["epoch"], 0
                temporary = output.with_suffix(".tmp")
                torch.save({**config, "state_dict": model.state_dict(), "best_epoch": best_epoch,
                            "best_validation_loss": best_loss}, temporary)
                temporary.replace(output)
            else:
                stale += 1
            log_path = output.with_suffix(".history.csv")
            with log_path.open("w", newline="", encoding="utf-8") as stream:
                writer = csv.DictWriter(stream, fieldnames=list(row))
                writer.writeheader()
                writer.writerows(history)
            if stale >= patience:
                print(f"Early stopping [{stage}] after {stale} epochs without validation-loss improvement.", flush=True)
                break
    config.update(epochs_completed=len(history), best_epoch=best_epoch, best_validation_loss=best_loss,
                  total_seconds=time.monotonic() - started)
    output.with_suffix(".training.json").write_text(json.dumps(config, indent=2) + "\n", encoding="utf-8")
    print(f"Saved best checkpoint: {output}; best epoch {best_epoch}.", flush=True)
    return config
