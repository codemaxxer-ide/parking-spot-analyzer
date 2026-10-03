from collections import Counter
from pathlib import Path
import tempfile
import unittest

import numpy as np
from PIL import Image
import torch
from torchvision import transforms

from ml.baseline_cnn import BaselineCNN
from ml.analysis import condition_metrics, read_analysis_tags
from ml.datasets import (ParkingCropDataset, Record, balanced_subset, inspect_dataset_a,
                         normalize_label, read_manifest, split_dataset_a, validate_no_leakage, write_manifest)
from ml.metrics import classification_metrics
from ml.transfer_model import ParkingMobileNet


class MLTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(4)

    def test_label_normalization(self):
        for value in (0, "empty", "VACANT", "free", " 0 "):
            self.assertEqual(normalize_label(value), 0)
        for value in (1, "not_empty", "OCCUPIED", "busy", "1"):
            self.assertEqual(normalize_label(value), 1)
        for value in (True, "car", 2):
            with self.assertRaises(ValueError):
                normalize_label(value)

    def test_dataset_a_adapter_and_manifest_round_trip(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            for label in ("empty", "not_empty"):
                (root / label).mkdir()
                Image.new("RGB", (69, 29), "white").save(root / label / "00000000_00000003.jpg")
            records = inspect_dataset_a(root)
            self.assertEqual({record.label for record in records}, {0, 1})
            self.assertTrue(all(record.camera is None and record.weather is None for record in records))
            manifest = root / "manifest.csv"
            write_manifest(records, manifest)
            self.assertEqual(read_manifest(manifest), records)

    def test_unknown_filename_grouping_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory); (root / "empty").mkdir()
            Image.new("RGB", (10, 10)).save(root / "empty/random.jpg")
            with self.assertRaisesRegex(ValueError, "group this Dataset A filename"):
                inspect_dataset_a(root)

    def synthetic_groups(self):
        return [Record(f"{slot}_{frame}.jpg", slot % 2, "dataset_a", slot_token=str(slot),
                       frame_token=str(frame), sha256=f"sha-{slot}-{frame}")
                for slot in range(40) for frame in range(0, 1001, 20)]

    def test_split_disjoint_slots_frames_hashes_and_seed(self):
        records = self.synthetic_groups()
        report = split_dataset_a(records, seed=42, embargo=20)
        validate_no_leakage(records)
        again = self.synthetic_groups(); split_dataset_a(again, seed=42, embargo=20)
        self.assertEqual([record.split for record in records], [record.split for record in again])
        self.assertTrue(report["counts"]["excluded"])
        for split in ("train", "val", "test"):
            self.assertEqual({record.label for record in records if record.split == split}, {0, 1})

    def test_duplicate_hash_leakage_is_rejected(self):
        records = self.synthetic_groups(); split_dataset_a(records)
        train = next(record for record in records if record.split == "train")
        test = next(record for record in records if record.split == "test")
        test.sha256 = train.sha256
        with self.assertRaisesRegex(ValueError, "Leakage in sha256"):
            validate_no_leakage(records)

    def test_frame_boundary_bands_are_excluded(self):
        records = self.synthetic_groups(); split_dataset_a(records, embargo=20)
        self.assertTrue(all(record.split == "excluded" for record in records
                            if int(record.frame_token) in (600, 800)))

    def test_balanced_external_subset_and_seed(self):
        records = [Record(str(index), index % 2, "dataset_b", camera=str(index % 3), weather="RAINY")
                   for index in range(100)]
        result = balanced_subset(records, 20)
        self.assertEqual(Counter(record.label for record in result), {0: 10, 1: 10})
        self.assertEqual(result, balanced_subset(records, 20))
        self.assertEqual(len({record.file_path for record in result}), 20)
        with self.assertRaises(ValueError):
            balanced_subset(records, 101)

    def test_loader_rgb_and_deterministic_test_preprocessing(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "gray.jpg"
            Image.fromarray(np.arange(80 * 40, dtype=np.uint8).reshape(40, 80)).save(path)
            records = [Record(str(path), 0, "dataset_a", split="test")]
            dataset = ParkingCropDataset(records, "test", 64)
            first, label = dataset[0]; second, _ = dataset[0]
            self.assertEqual(first.shape, (3, 64, 64))
            self.assertEqual(label, 0)
            self.assertTrue(torch.equal(first, second))
            self.assertFalse(dataset.training)

    def test_augmentation_only_enabled_for_training(self):
        records = [Record("unused.jpg", 0, "dataset_a", split=split) for split in ("train", "val", "test")]
        for split in ("train", "val", "test"):
            dataset = ParkingCropDataset(records, split)
            random_crop = any(isinstance(operation, transforms.RandomResizedCrop) for operation in dataset.transform.transforms)
            self.assertEqual(random_crop, split == "train")

    def test_baseline_output_shape(self):
        self.assertEqual(BaselineCNN()(torch.zeros(2, 3, 128, 128)).shape, (2, 2))

    def test_mobilenet_output_shape_without_downloading_weights(self):
        model = ParkingMobileNet(pretrained=False).eval()
        with torch.inference_mode():
            self.assertEqual(model(torch.zeros(2, 3, 128, 128)).shape, (2, 2))

    def test_frozen_features_and_batchnorm_then_partial_unfreeze(self):
        model = ParkingMobileNet(pretrained=False).train()
        self.assertFalse(any(parameter.requires_grad for parameter in model.model.features.parameters()))
        self.assertFalse(model.model.features[0].training)
        self.assertTrue(model.model.classifier.training)
        model.set_fine_tuning(3); model.train()
        self.assertFalse(model.model.features[0].training)
        self.assertTrue(model.model.features[-1].training)
        self.assertTrue(any(parameter.requires_grad for parameter in model.model.features[-3:].parameters()))

    def test_binary_metrics_with_occupied_positive(self):
        result = classification_metrics([0, 0, 0, 1, 1, 1], [0, 0, 1, 0, 1, 1])
        self.assertEqual(result["confusion_matrix"], [[2, 1], [1, 2]])
        for metric in ("accuracy", "precision", "recall", "f1"):
            self.assertAlmostEqual(result[metric], 2 / 3)

    def test_metrics_zero_denominator_and_invalid_inputs(self):
        result = classification_metrics([0, 0], [0, 0])
        self.assertEqual((result["accuracy"], result["precision"], result["recall"], result["f1"]), (1, 0, 0, 0))
        for truth, prediction in (([], []), ([0], [0, 1]), ([2], [1])):
            with self.assertRaises(ValueError):
                classification_metrics(truth, prediction)

    def test_condition_metrics_keep_models_separate_and_skip_unknown_tags(self):
        rows = [
            dict(model="baseline", dataset_source="dataset_b", occlusion_tag="HEAVY", label=1, prediction=0),
            dict(model="baseline", dataset_source="dataset_b", occlusion_tag="HEAVY", label=1, prediction=1),
            dict(model="mobilenet", dataset_source="dataset_b", occlusion_tag="HEAVY", label=1, prediction=1),
            dict(model="baseline", dataset_source="dataset_b", occlusion_tag="", label=0, prediction=0),
        ]
        results = condition_metrics(rows, "occlusion_tag")
        self.assertEqual(len(results), 2)
        self.assertEqual((results[0]["sample_count"], results[0]["correct"], results[0]["incorrect"]), (2, 1, 1))
        self.assertEqual(results[0]["accuracy"], 0.5)
        self.assertEqual(results[1]["sample_count"], 1)
        self.assertTrue(all(result["small_sample_warning"] for result in results))

    def test_review_tags_require_provenance_and_known_categories(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "tags.csv"
            path.write_text("file_path,lighting_tag,occlusion_tag,annotation_source\nimage.jpg,SHADOW,NONE,\n")
            with self.assertRaisesRegex(ValueError, "provenance"):
                read_analysis_tags(path)
            path.write_text("file_path,lighting_tag,occlusion_tag,annotation_source\nimage.jpg,UNKNOWN,NONE,visual_review\n")
            with self.assertRaisesRegex(ValueError, "Invalid lighting_tag"):
                read_analysis_tags(path)
