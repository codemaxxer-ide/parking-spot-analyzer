import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import Mock, patch

import cv2
import numpy as np

from src.calibration import CalibrationState, DisplayTransform, read_reference_frame, run_calibration
from src.parking_slots import load_parking_config


class CalibrationTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.directory = Path(self.temporary.name)
        self.output = self.directory / "slots.json"
        self.reference = self.directory / "reference.jpg"
        self.frame = np.full((1080, 1920, 3), 80, np.uint8)
        self.state = CalibrationState(DisplayTransform.fit(1920, 1080, 960, 540))

    def complete_polygon(self):
        for point in ((50, 50), (100, 40), (130, 80), (100, 120), (40, 100)):
            self.state.add_point(*point)
        self.assertTrue(self.state.complete_slot())

    def test_scaled_clicks_store_original_coordinates(self):
        self.state.add_point(50, 100)
        self.assertEqual(self.state.current_points, [[100, 200]])

    def test_independently_rounded_preview_dimensions(self):
        transform = DisplayTransform.fit(1919, 1079, 800, 600)
        self.assertEqual(transform.to_original(400, 100),
                         [round(400 * 1919 / transform.display_width), round(100 * 1079 / transform.display_height)])

    def test_footer_and_outside_clicks_ignored(self):
        for point in ((10, 540), (960, 10), (-1, 2)):
            self.state.add_point(*point)
        self.assertEqual(self.state.current_points, [])

    def test_five_point_polygon_and_slot_ids(self):
        self.complete_polygon()
        self.complete_polygon()
        self.assertEqual([slot["id"] for slot in self.state.slots], ["P1", "P2"])
        self.assertEqual(len(self.state.slots[0]["points"]), 5)

    def test_undo_and_delete_completed_slot(self):
        self.state.add_point(1, 1)
        self.state.undo_point()
        self.assertEqual(self.state.current_points, [])
        self.complete_polygon()
        self.state.delete_slot()
        self.assertEqual(self.state.slots, [])
        self.complete_polygon()
        self.assertEqual(self.state.slots[0]["id"], "P1")

    def test_invalid_completion_keeps_points_for_correction(self):
        self.state.add_point(1, 1)
        self.assertFalse(self.state.complete_slot())
        self.assertEqual(self.state.current_points, [[2, 2]])
        self.assertIn("at least three", self.state.message)

    def test_save_metadata_and_full_resolution_annotated_jpeg(self):
        self.complete_polygon()
        self.state.save(self.output, self.reference, self.frame)
        config = load_parking_config(self.output)
        self.assertEqual((config.frame_width, config.frame_height), (1920, 1080))
        self.assertEqual(config.slots[0].polygon[0].tolist(), [100, 100])
        reference = cv2.imread(str(self.reference))
        self.assertEqual(reference.shape, self.frame.shape)
        self.assertFalse(np.array_equal(reference, self.frame))
        self.assertFalse(self.state.dirty)
        self.assertFalse(list(self.directory.glob(".roi-*")))

    def test_save_rejects_empty_or_unfinished_work(self):
        self.output.write_text("original", encoding="utf-8")
        with self.assertRaises(ValueError):
            self.state.save(self.output, self.reference, self.frame)
        self.complete_polygon()
        self.state.add_point(1, 1)
        with self.assertRaisesRegex(ValueError, "Finish the current polygon"):
            self.state.save(self.output, self.reference, self.frame)
        self.assertEqual(self.output.read_text(), "original")

    def test_export_failure_preserves_existing_config(self):
        self.complete_polygon()
        self.output.write_text("original", encoding="utf-8")
        with patch("src.calibration.cv2.imwrite", return_value=False):
            with self.assertRaisesRegex(RuntimeError, "reference JPEG"):
                self.state.save(self.output, self.reference, self.frame)
        self.assertEqual(self.output.read_text(), "original")
        self.assertFalse(list(self.directory.glob(".roi-*")))

    def test_render_does_not_change_reference_frame(self):
        self.complete_polygon()
        self.state.add_point(30, 30)
        rendered = self.state.render(self.frame)
        self.assertEqual(rendered.shape, (652, 960, 3))
        self.assertTrue((self.frame == 80).all())

    def test_first_valid_frame_and_capture_cleanup(self):
        source = self.directory / "video.mp4"
        source.touch()
        capture = Mock()
        capture.isOpened.return_value = True
        capture.read.side_effect = [(False, None), (True, self.frame)]
        with patch("src.calibration.cv2.VideoCapture", return_value=capture):
            self.assertIs(read_reference_frame(source), self.frame)
        capture.release.assert_called_once()

    def test_unreadable_video_retries_are_bounded_and_capture_closes(self):
        source = self.directory / "video.mp4"
        source.touch()
        capture = Mock()
        capture.isOpened.return_value = True
        capture.read.return_value = (False, None)
        with patch("src.calibration.cv2.VideoCapture", return_value=capture):
            with self.assertRaisesRegex(RuntimeError, "first 120 read attempts"):
                read_reference_frame(source)
        self.assertEqual(capture.read.call_count, 120)
        capture.release.assert_called_once()

    def test_missing_video_has_useful_error(self):
        with self.assertRaisesRegex(ValueError, "Input video does not exist"):
            read_reference_frame(self.directory / "missing.mp4")

    def test_window_close_exits_cleanly_without_saving(self):
        with patch("src.calibration.read_reference_frame", return_value=self.frame), \
                patch("src.calibration.cv2.namedWindow"), patch("src.calibration.cv2.imshow"), \
                patch("src.calibration.cv2.setMouseCallback"), \
                patch("src.calibration.cv2.waitKey", return_value=-1), \
                patch("src.calibration.cv2.getWindowProperty", side_effect=cv2.error("window closed")), \
                patch("src.calibration.cv2.destroyWindow") as destroy:
            run_calibration("input.mp4", self.output, self.reference)
        destroy.assert_called_once()
        self.assertFalse(self.output.exists())

    def test_gui_event_workflow_save_and_exit_closes_window(self):
        callbacks = {}
        keys = iter((13, ord("s"), ord("q")))
        def wait_key(delay):
            if not callbacks.get("clicked"):
                for x, y in ((50, 50), (100, 50), (100, 100), (50, 100)):
                    callbacks["mouse"](cv2.EVENT_LBUTTONDOWN, x, y, 0, None)
                callbacks["clicked"] = True
            return next(keys)
        with patch("src.calibration.read_reference_frame", return_value=self.frame), \
                patch("src.calibration.cv2.namedWindow"), patch("src.calibration.cv2.imshow"), \
                patch("src.calibration.cv2.setMouseCallback", side_effect=lambda name, callback: callbacks.update(mouse=callback)), \
                patch("src.calibration.cv2.getWindowProperty", return_value=1), \
                patch("src.calibration.cv2.waitKey", side_effect=wait_key), \
                patch("src.calibration.cv2.destroyWindow") as destroy:
            state = run_calibration("input.mp4", self.output, self.reference)
        self.assertEqual(len(state.slots), 1)
        self.assertEqual(json.loads(self.output.read_text())["frame_width"], 1920)
        destroy.assert_called_once()
