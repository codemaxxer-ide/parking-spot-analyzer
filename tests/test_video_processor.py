import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import Mock, patch

import cv2
import numpy as np

from src.detector import Detection
from src.video_processor import process_video


class VideoProcessorTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.directory = Path(self.temporary.name)
        self.input = self.directory / "input.mp4"
        self.output = self.directory / "output.mp4"
        self.slots = self.directory / "slots.json"
        self.slots.write_text(json.dumps({
            "left": [[10, 100], [150, 100], [150, 220], [10, 220]],
            "right": [[170, 100], [310, 100], [310, 220], [170, 220]],
        }), encoding="utf-8")
        writer = cv2.VideoWriter(str(self.input), cv2.VideoWriter_fourcc(*"mp4v"), 12, (320, 240))
        self.assertTrue(writer.isOpened(), "MP4 encoder must be available for integration tests")
        try:
            for _ in range(3):
                writer.write(np.full((240, 320, 3), 80, np.uint8))
        finally:
            writer.release()
        self.detector = Mock()
        self.detector.detect.side_effect = [
            [Detection((20, 110, 80, 170), "car", 0.9)],
            [Detection((20, 110, 80, 170), "car", 0.9),
             Detection((180, 110, 240, 170), "truck", 0.8)], [],
        ]

    def test_mp4_round_trip_preserves_frames_fps_resolution(self):
        summary = process_video(self.input, self.output, self.slots, detector=self.detector)
        self.assertEqual(summary.frames_written, 3)
        self.assertEqual((summary.total_spaces, summary.occupied_last_frame, summary.available_last_frame), (2, 0, 2))
        self.assertEqual(self.detector.detect.call_count, 3)
        capture = cv2.VideoCapture(str(self.output))
        try:
            self.assertTrue(capture.isOpened())
            self.assertAlmostEqual(capture.get(cv2.CAP_PROP_FPS), 12)
            frames = []
            while True:
                ok, frame = capture.read()
                if not ok:
                    break
                self.assertEqual(frame.shape, (240, 320, 3))
                frames.append(frame)
            self.assertEqual(len(frames), 3)
            self.assertGreater(int(frames[0][200, 100, 2]), int(frames[0][200, 100, 1]))
            self.assertGreater(int(frames[2][200, 100, 1]), int(frames[2][200, 100, 2]))
        finally:
            capture.release()
        self.assertFalse(list(self.directory.glob(".parking-*.mp4")))

    def test_same_input_output_rejected_without_damage(self):
        original = self.input.read_bytes()
        with self.assertRaisesRegex(ValueError, "different files"):
            process_video(self.input, self.input, self.slots, detector=self.detector)
        self.assertEqual(self.input.read_bytes(), original)

    def test_existing_output_requires_overwrite(self):
        self.output.write_bytes(b"existing output")
        with self.assertRaisesRegex(ValueError, "already exists"):
            process_video(self.input, self.output, self.slots, detector=self.detector)
        self.assertEqual(self.output.read_bytes(), b"existing output")

    def test_inference_failure_releases_resources_and_preserves_existing_output(self):
        self.output.write_bytes(b"existing output")
        capture, writer = Mock(), Mock()
        capture.isOpened.return_value = True
        capture.read.return_value = (True, np.zeros((240, 320, 3), np.uint8))
        capture.get.side_effect = lambda key: {
            cv2.CAP_PROP_FRAME_WIDTH: 320, cv2.CAP_PROP_FRAME_HEIGHT: 240,
            cv2.CAP_PROP_FPS: 12, cv2.CAP_PROP_FRAME_COUNT: 3,
        }[key]
        self.detector.detect.side_effect = RuntimeError("inference failed")
        with patch("src.video_processor.cv2.VideoCapture", return_value=capture), \
                patch("src.video_processor.cv2.VideoWriter", return_value=writer):
            with self.assertRaisesRegex(RuntimeError, "inference failed"):
                process_video(self.input, self.output, self.slots, detector=self.detector, overwrite=True)
        capture.release.assert_called_once()
        writer.release.assert_called_once()
        self.assertEqual(self.output.read_bytes(), b"existing output")
        self.assertFalse(list(self.directory.glob(".parking-*.mp4")))

    def test_early_read_failure_does_not_publish_truncated_video(self):
        capture = Mock()
        capture.isOpened.return_value = True
        capture.read.side_effect = [(True, np.zeros((240, 320, 3), np.uint8)), (False, None)]
        capture.get.side_effect = lambda key: {
            cv2.CAP_PROP_FRAME_WIDTH: 320, cv2.CAP_PROP_FRAME_HEIGHT: 240,
            cv2.CAP_PROP_FPS: 12, cv2.CAP_PROP_FRAME_COUNT: 3,
        }[key]
        with patch("src.video_processor.cv2.VideoCapture", return_value=capture):
            with self.assertRaisesRegex(RuntimeError, "Video read failed after 1 of 3"):
                process_video(self.input, self.output, self.slots, detector=self.detector)
        capture.release.assert_called_once()
        self.assertFalse(self.output.exists())
        self.assertFalse(list(self.directory.glob(".parking-*.mp4")))

    def test_unreadable_video_releases_capture(self):
        capture = Mock()
        capture.isOpened.return_value = True
        capture.read.return_value = (False, None)
        with patch("src.video_processor.cv2.VideoCapture", return_value=capture):
            with self.assertRaisesRegex(RuntimeError, "no readable frames"):
                process_video(self.input, self.output, self.slots, detector=self.detector)
        capture.release.assert_called_once()
        self.assertFalse(self.output.exists())

    def test_missing_video_has_useful_error(self):
        with self.assertRaisesRegex(ValueError, "Input video does not exist"):
            process_video(self.directory / "missing.mp4", self.output, self.slots, detector=self.detector)

    def test_frame_skipping_preserves_all_output_frames(self):
        summary = process_video(self.input, self.output, self.slots, detector=self.detector, process_every=2)
        self.assertEqual((summary.frames_written, summary.inference_frames), (3, 2))
        self.assertEqual(self.detector.detect.call_count, 2)
        capture = cv2.VideoCapture(str(self.output))
        try:
            self.assertEqual(int(capture.get(cv2.CAP_PROP_FRAME_COUNT)), 3)
            self.assertAlmostEqual(capture.get(cv2.CAP_PROP_FPS), 12)
        finally:
            capture.release()

    def test_skipped_frames_do_not_add_smoothing_votes(self):
        self.detector.detect.side_effect = None
        self.detector.detect.return_value = [Detection((20, 110, 80, 170), "car", 0.9)]
        summary = process_video(self.input, self.output, self.slots, detector=self.detector,
                                process_every=2, smoothing_window=5, smoothing_required=3)
        self.assertEqual(summary.inference_frames, 2)
        self.assertEqual(summary.occupied_last_frame, 0)

    def test_overlap_strategy_flows_through_video_pipeline(self):
        self.detector.detect.side_effect = None
        self.detector.detect.return_value = [Detection((-110, 100, 70, 220), "car", 0.9)]
        summary = process_video(self.input, self.output, self.slots, detector=self.detector,
                                occupancy_method="overlap", overlap_threshold=0.2)
        self.assertEqual(summary.occupied_last_frame, 1)

    def test_calibrated_metadata_mismatch_rejected_before_inference(self):
        self.slots.write_text(json.dumps({"frame_width": 640, "frame_height": 480,
                                         "slots": [{"id": "P1", "points": [[0, 0], [30, 0], [0, 30]]}]}))
        with self.assertRaisesRegex(ValueError, "Re-run ROI calibration"):
            process_video(self.input, self.output, self.slots, detector=self.detector)
        self.detector.detect.assert_not_called()
        self.assertFalse(self.output.exists())
