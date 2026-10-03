import json
from pathlib import Path
import tempfile
import unittest

from src.parking_slots import load_parking_slots, validate_frame_bounds


class ParkingSlotTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.path = Path(self.temporary.name) / "slots.json"

    def load(self, data):
        self.path.write_text(json.dumps(data), encoding="utf-8")
        return load_parking_slots(self.path)

    def test_non_rectangular_polygon_and_labels(self):
        slots = self.load({"angled": [[1, 2], [20, 2], [15, 20], [2, 18]]})
        self.assertEqual(slots[0].slot_id, "angled")
        self.assertEqual(slots[0].label, "P1")
        validate_frame_bounds(slots, 30, 30)

    def test_missing_config_has_useful_error(self):
        with self.assertRaisesRegex(ValueError, "Cannot read parking slots"):
            load_parking_slots(self.path)

    def test_malformed_json_has_useful_error(self):
        self.path.write_text("{oops", encoding="utf-8")
        with self.assertRaisesRegex(ValueError, "Malformed parking JSON"):
            load_parking_slots(self.path)

    def test_duplicate_ids_rejected(self):
        self.path.write_text('{"s": [], "s": []}', encoding="utf-8")
        with self.assertRaisesRegex(ValueError, "Duplicate slot ID"):
            load_parking_slots(self.path)

    def test_invalid_polygons_rejected(self):
        invalid = [[], {}, {"": [[0, 0], [2, 0], [0, 2]]},
                   {"s": [[0, 0], [1, 1]]},
                   {"s": [[0, 0], [1, 1], [2, 2]]},
                   {"s": [[0, 0], [2, 0], [-1, 2]]},
                   {"s": [[0, 0], [2, 0], [0, float("nan")]]},
                   {"s": [[0, 0], [2, 0], [0, float("inf")]]},
                   {"s": [[0, 0], [True, 0], [0, 2]]},
                   {"s": [[0, 0], ["2", 0], [0, 2]]},
                   {"s": [[0, 0], [2, 0], [0, 2], [2, 0]]},
                   {"s": [[0, 0], [3, 3], [0, 3], [3, 0]]}]
        for data in invalid:
            with self.subTest(data=data), self.assertRaises(ValueError):
                self.load(data)

    def test_closed_polygon_accepted(self):
        self.assertEqual(len(self.load({"s": [[0, 0], [2, 0], [0, 2], [0, 0]]})[0].polygon), 3)

    def test_out_of_frame_coordinates_rejected(self):
        slots = self.load({"s": [[0, 0], [30, 0], [0, 30]]})
        with self.assertRaisesRegex(ValueError, "outside the 30x30 video"):
            validate_frame_bounds(slots, 30, 30)
