import json
from pathlib import Path
import tempfile
import time
import unittest
from unittest.mock import Mock, patch

import cv2
import numpy as np
from fastapi.testclient import TestClient
import torch

from src.detector import Detection
from src.occupancy import OccupancyResult
from src.parking_slots import ParkingSlot
from src.slot_classifier import DEFAULT_CHECKPOINT, SlotClassifier, slot_crop
from src.video_processor import process_video
from src.visualization import annotate_frame, default_show_ids
from web.app import create_app

CROPS = Path(__file__).resolve().parents[1] / "data/raw/dataset_a/parking/clf-data"


def box(x1, y1, x2, y2):
    return np.array([[x1, y1], [x2, y1], [x2, y2], [x1, y2]], np.float32)


class StubClassifier:
    """Scripted raw predictions; used to prove temporal smoothing is applied."""

    def __init__(self, pattern):
        self.pattern, self.calls = pattern, 0

    def classify(self, frame, slots, threshold):
        value = self.pattern[self.calls % len(self.pattern)]
        self.calls += 1
        return (OccupancyResult({slot.slot_id: value for slot in slots}),
                {slot.slot_id: 0.9 if value else 0.1 for slot in slots})


def write_video(path, frames, size=(320, 240)):
    writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), 10, size)
    for _ in range(frames):
        writer.write(np.zeros((size[1], size[0], 3), np.uint8))
    writer.release()


class SlotCropTests(unittest.TestCase):
    def test_crop_is_clamped_bounding_box(self):
        frame = np.arange(100 * 200 * 3, dtype=np.uint8).reshape(100, 200, 3)
        slot = ParkingSlot("a", "P1", box(10.4, 20.2, 60.6, 50.1))
        np.testing.assert_array_equal(slot_crop(frame, slot), frame[20:51, 10:61])
        edge = ParkingSlot("b", "P2", box(180, 80, 260, 140))
        self.assertEqual(slot_crop(frame, edge).shape[:2], (20, 20))

    def test_masking_blacks_out_pixels_outside_polygon(self):
        frame = np.full((100, 100, 3), 255, np.uint8)
        triangle = ParkingSlot("t", "P1", np.array([[10, 10], [60, 10], [10, 60]], np.float32))
        crop = slot_crop(frame, triangle, mask_outside=True)
        self.assertEqual(int(crop[2, 2, 0]), 255)
        self.assertEqual(int(crop[-2, -2, 0]), 0)


@unittest.skipUnless(DEFAULT_CHECKPOINT.is_file() and CROPS.is_dir(), "trained checkpoint/dataset not present")
class ClassifierTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.classifier = SlotClassifier()

    def test_label_mapping_comes_from_checkpoint(self):
        checkpoint = torch.load(DEFAULT_CHECKPOINT, map_location="cpu", weights_only=True)
        self.assertEqual(checkpoint["label_map"], {0: "VACANT", 1: "OCCUPIED"})
        checkpoint["label_map"] = {0: "OCCUPIED", 1: "VACANT"}
        with tempfile.TemporaryDirectory() as folder:
            swapped = Path(folder) / "swapped.pt"
            torch.save(checkpoint, swapped)
            with self.assertRaisesRegex(ValueError, "Unexpected checkpoint labels"):
                SlotClassifier(swapped)

    def test_missing_checkpoint_is_reported(self):
        with self.assertRaises(FileNotFoundError):
            SlotClassifier(Path("missing.pt"))

    def test_batch_preprocessing_shape_and_normalisation(self):
        frame = np.full((100, 200, 3), 128, np.uint8)
        slots = [ParkingSlot(str(i), f"P{i}", box(10 * i, 10, 10 * i + 30, 50)) for i in range(1, 6)]
        batch = self.classifier.preprocess(frame, slots)
        self.assertEqual(tuple(batch.shape), (5, 3, 128, 128))
        self.assertAlmostEqual(float(batch[0, 0].mean()), (128 / 255 - 0.485) / 0.229, places=2)

    def test_real_crops_are_classified_in_one_batch(self):
        empty = sorted((CROPS / "empty").glob("*.jpg"))[:3]
        full = sorted((CROPS / "not_empty").glob("*.jpg"))[:3]
        frame = np.zeros((200, 600, 3), np.uint8)
        slots, expected = [], []
        for index, (path, state) in enumerate([(p, False) for p in empty] + [(p, True) for p in full]):
            x = index * 90
            frame[10:90, x:x + 80] = cv2.resize(cv2.imread(str(path)), (80, 80))
            slots.append(ParkingSlot(f"s{index}", f"P{index + 1}", box(x, 10, x + 80, 90)))
            expected.append(state)
        raw, probabilities = self.classifier.classify(frame, slots, 0.5)
        self.assertEqual(list(raw.occupied.values()), expected)
        self.assertTrue(all(0 <= p <= 1 for p in probabilities.values()))
        self.assertEqual((raw.total, raw.occupied_count), (6, 3))

    def test_threshold_controls_decision(self):
        frame = np.zeros((100, 100, 3), np.uint8)
        slot = [ParkingSlot("a", "P1", box(0, 0, 50, 50))]
        probability = float(self.classifier.occupied_probabilities(frame, slot)[0])
        self.assertTrue(self.classifier.classify(frame, slot, 1e-9)[0].occupied["a"])
        self.assertEqual(self.classifier.classify(frame, slot, 0.5)[0].occupied["a"], probability >= 0.5)


class ProcessingTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        self.video = self.root / "in.mp4"
        write_video(self.video, 6)
        self.slots = self.root / "slots.json"
        self.slots.write_text(json.dumps({"frame_width": 320, "frame_height": 240, "slots": [
            {"id": "A", "points": [[10, 100], [100, 100], [100, 200], [10, 200]]}]}))

    def tearDown(self):
        self.temp.cleanup()

    def run_engine(self, pattern, **options):
        stub = StubClassifier(pattern)
        summary = process_video(self.video, self.root / "out.mp4", self.slots, engine="classifier",
                                classifier=stub, overwrite=True, **options)
        return summary, stub

    def test_smoothing_is_applied_to_classifier_predictions(self):
        smoothed, _ = self.run_engine([False, True, True, False, True, False], smoothing_window=5, smoothing_required=3)
        raw, _ = self.run_engine([False, True, True, False, True, False])
        self.assertEqual(smoothed.occupied_last_frame, 1)  # three positive votes inside the window
        self.assertEqual(raw.occupied_last_frame, 0)       # unsmoothed final prediction is vacant
        brief, _ = self.run_engine([True] + [False] * 5, smoothing_window=5, smoothing_required=3)
        self.assertEqual(brief.occupied_last_frame, 0)

    def test_process_every_runs_classifier_on_selected_frames_only(self):
        summary, stub = self.run_engine([True], process_every=3)
        self.assertEqual((stub.calls, summary.inference_frames, summary.frames_written), (2, 2, 6))
        self.assertEqual(summary.slot_probabilities, {"A": 0.9})

    def test_engine_selection_and_threshold_validation(self):
        with self.assertRaises(ValueError):
            process_video(self.video, self.root / "o.mp4", self.slots, engine="magic")
        with self.assertRaises(ValueError):
            process_video(self.video, self.root / "o.mp4", self.slots, engine="classifier",
                          classifier=StubClassifier([True]), classification_threshold=1.5)

    def test_yolo_engine_remains_default(self):
        detector = Mock()
        detector.detect.return_value = []
        summary = process_video(self.video, self.root / "y.mp4", self.slots, detector=detector)
        self.assertEqual((detector.detect.call_count, summary.slot_probabilities), (6, None))


class OverlayTests(unittest.TestCase):
    def setUp(self):
        self.slots = [ParkingSlot(f"s{i}", f"P{i}", box(10 + 30 * i, 120, 35 + 30 * i, 220)) for i in range(10)]
        self.occupancy = OccupancyResult({s.slot_id: i % 2 == 0 for i, s in enumerate(self.slots)})
        self.frame = np.zeros((300, 400, 3), np.uint8)

    def test_large_lots_hide_ids_by_default(self):
        self.assertTrue(default_show_ids(50))
        self.assertFalse(default_show_ids(51))

    def test_ids_and_roi_toggles_change_the_frame(self):
        with_ids = annotate_frame(self.frame, [], self.slots, self.occupancy, show_ids=True)
        without_ids = annotate_frame(self.frame, [], self.slots, self.occupancy, show_ids=False)
        no_rois = annotate_frame(self.frame, [], self.slots, self.occupancy, show_rois=False, show_ids=False)
        self.assertFalse(np.array_equal(with_ids, without_ids))
        self.assertFalse(np.array_equal(no_rois, without_ids))

    def test_vehicle_boxes_toggle_and_engine_label(self):
        detections = [Detection((300, 20, 350, 70), "car", .9)]
        shown = annotate_frame(self.frame, detections, self.slots, self.occupancy, show_boxes=True)
        hidden = annotate_frame(self.frame, detections, self.slots, self.occupancy, show_boxes=False)
        self.assertFalse(np.array_equal(shown, hidden))
        classifier = annotate_frame(self.frame, [], self.slots, self.occupancy, engine_label="Slot Classifier - MobileNetV2")
        yolo = annotate_frame(self.frame, [], self.slots, self.occupancy, engine_label="YOLO11n + ROI")
        self.assertFalse(np.array_equal(classifier, yolo))


class WebEngineTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        self.video = self.root / "in.mp4"
        write_video(self.video, 3)
        self.roi = json.dumps({"frame_width": 320, "frame_height": 240, "slots": [
            {"id": "P1", "points": [[10, 100], [100, 100], [100, 200], [10, 200]]}]}).encode()
        self.app = create_app(self.root / "jobs")
        self.client = TestClient(self.app).__enter__()

    def tearDown(self):
        self.client.__exit__(None, None, None)
        self.temp.cleanup()

    def post(self, **data):
        files = {"video": ("in.mp4", self.video.read_bytes(), "video/mp4"),
                 "slots": ("slots.json", self.roi, "application/json")}
        return self.client.post("/api/process-video", files=files, data=data)

    def finish(self, job_id):
        deadline = time.monotonic() + 15
        while time.monotonic() < deadline:
            job = self.client.get(f"/api/status/{job_id}").json()
            if job["state"] in ("complete", "failed"):
                return job
            time.sleep(.02)
        self.fail("job did not finish")

    def test_classifier_engine_end_to_end_with_stub(self):
        with patch("src.slot_classifier.SlotClassifier", return_value=StubClassifier([True])), \
                patch("web.app.find_ffmpeg", return_value=None):
            response = self.post(engine="classifier", classification_threshold="0.6", smoothing="false")
            self.assertEqual(response.status_code, 202)
            settings = response.json()["settings"]
            self.assertEqual((settings["engine"], settings["classification_threshold"], settings["show_boxes"]),
                             ("classifier", 0.6, False))
            job = self.finish(response.json()["job_id"])
        self.assertEqual(job["state"], "complete")
        self.assertEqual(job["result"]["engine"], "classifier")
        self.assertEqual(job["result"]["occupied"], 1)
        self.assertEqual(job["result"]["slots"][0]["occupied_probability"], 0.9)

    def test_missing_classifier_is_friendly(self):
        with patch("src.slot_classifier.SlotClassifier", side_effect=FileNotFoundError("C:/secret/path.pt")), \
                self.assertLogs(level="ERROR"):
            job = self.finish(self.post(engine="classifier").json()["job_id"])
        self.assertEqual(job["state"], "failed")
        self.assertIn("Slot classifier unavailable", job["error"])
        self.assertNotIn("secret", json.dumps(job))

    def test_invalid_engine_and_threshold_rejected(self):
        self.assertEqual(self.post(engine="magic").status_code, 422)
        self.assertEqual(self.post(engine="classifier", classification_threshold="1.0").status_code, 422)

    def test_yolo_is_default_engine_and_ids_follow_slot_count(self):
        with patch("src.detector.VehicleDetector") as detector:
            detector.return_value.detect.return_value = []
            response = self.post()
            settings = response.json()["settings"]
            self.finish(response.json()["job_id"])
        self.assertEqual(settings["engine"], "yolo")
        self.assertTrue(settings["show_ids"])
        self.assertIn("classifier_available", self.client.get("/api/health").json())


if __name__ == "__main__":
    unittest.main()
