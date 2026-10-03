import json
from pathlib import Path
import tempfile
import unittest

from src.parking_slots import load_parking_config, load_parking_slots, parse_parking_config


def metadata(points=None):
    return {"frame_width": 320, "frame_height": 240,
            "slots": [{"id": "P1", "points": points or [[10, 10], [100, 10], [100, 100], [10, 100]]}]}


class MetadataTests(unittest.TestCase):
    def test_metadata_round_trip_and_legacy_list_api(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "slots.json"
            path.write_text(json.dumps(metadata()), encoding="utf-8")
            config = load_parking_config(path)
            self.assertEqual((config.frame_width, config.frame_height), (320, 240))
            self.assertEqual(load_parking_slots(path)[0].slot_id, "P1")
            config.validate_frame(320, 240)

    def test_resolution_mismatch_rejected_even_when_polygons_fit(self):
        with self.assertRaisesRegex(ValueError, "calibrated at 320x240, but video is 640x480"):
            parse_parking_config(metadata()).validate_frame(640, 480)

    def test_legacy_metadata_names_remain_valid_slot_ids(self):
        config = parse_parking_config({"slots": [[0, 0], [10, 0], [0, 10]],
                                       "frame_width": [[20, 0], [30, 0], [20, 10]]})
        self.assertIsNone(config.frame_width)
        self.assertEqual(len(config.slots), 2)

    def test_invalid_dimensions_rejected(self):
        for dimensions in ((0, 240), (-1, 240), (320.0, 240), (True, 240), (320, None)):
            data = metadata()
            data["frame_width"], data["frame_height"] = dimensions
            with self.subTest(dimensions=dimensions), self.assertRaisesRegex(ValueError, "positive integers"):
                parse_parking_config(data)

    def test_duplicate_slot_ids_rejected(self):
        data = metadata()
        data["slots"].append(data["slots"][0].copy())
        with self.assertRaisesRegex(ValueError, "Duplicate slot ID"):
            parse_parking_config(data)

    def test_missing_fields_and_invalid_ids_rejected(self):
        for entries in ([], {}, [{"id": "P1"}], [{"points": []}],
                        [{"id": 1, "points": []}], [{"id": " ", "points": []}]):
            with self.subTest(entries=entries), self.assertRaises(ValueError):
                parse_parking_config({**metadata(), "slots": entries})

    def test_polygon_outside_declared_frame_rejected(self):
        with self.assertRaisesRegex(ValueError, "outside the 320x240 video"):
            parse_parking_config(metadata([[0, 0], [320, 0], [0, 239]]))

    def test_zero_area_polygon_rejected(self):
        with self.assertRaisesRegex(ValueError, "non-zero area"):
            parse_parking_config(metadata([[1, 1], [10, 10], [20, 20]]))

    def test_invalid_polygon_rejected(self):
        for points in ([[0, 0], [10, 10], [0, 10], [10, 0]],
                       [[0, 0], ["x", 10], [0, 10]], [[0, 0], [10, 10]]):
            with self.subTest(points=points), self.assertRaises(ValueError):
                parse_parking_config(metadata(points))
