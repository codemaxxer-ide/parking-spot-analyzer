import unittest

import numpy as np

from src.detector import Detection
from src.geometry import polygon_rectangle_intersection_area
from src.occupancy import calculate_occupancy
from src.parking_slots import ParkingSlot


class OverlapTests(unittest.TestCase):
    def setUp(self):
        self.slot = ParkingSlot("s", "P1", np.array([[0, 0], [10, 0], [10, 10], [0, 10]], np.float32))

    def occupied(self, box, threshold=0.2):
        return calculate_occupancy([self.slot], [Detection(box, "car", 0.9)],
                                   method="overlap", overlap_threshold=threshold).occupied["s"]

    def test_no_overlap(self):
        self.assertFalse(self.occupied((20, 20, 30, 30)))

    def test_partial_overlap_below_threshold(self):
        self.assertFalse(self.occupied((-4, 0, 1, 10)))

    def test_overlap_above_threshold_while_center_is_outside(self):
        box = (-5, 0, 3, 10)
        self.assertTrue(self.occupied(box))
        self.assertFalse(calculate_occupancy([self.slot], [Detection(box, "car", 0.9)]).occupied["s"])

    def test_threshold_is_inclusive(self):
        self.assertTrue(self.occupied((0, 0, 2, 10)))

    def test_boundary_touch_has_zero_area(self):
        self.assertFalse(self.occupied((-10, 0, 0, 10)))

    def test_multiple_detections_count_slot_once(self):
        result = calculate_occupancy([self.slot], [Detection((0, 0, 10, 10), "car", 0.9)] * 2,
                                     method="overlap")
        self.assertEqual(result.occupied_count, 1)

    def test_small_overlaps_are_not_summed_between_vehicles(self):
        result = calculate_occupancy([self.slot], [Detection((0, 0, 1, 10), "car", 0.9),
                                                  Detection((9, 0, 10, 10), "car", 0.9)], method="overlap")
        self.assertFalse(result.occupied["s"])

    def test_triangle_uses_true_polygon_area(self):
        triangle = np.array([[0, 0], [10, 0], [0, 10]], np.float32)
        self.assertAlmostEqual(polygon_rectangle_intersection_area(triangle, (0, 0, 5, 5)), 25)
        slot = ParkingSlot("t", "P1", triangle)
        self.assertTrue(calculate_occupancy([slot], [Detection((0, 0, 5, 5), "car", 0.9)],
                                            method="overlap", overlap_threshold=0.5).occupied["t"])

    def test_concave_notch_is_not_part_of_polygon(self):
        polygon = np.array([[0, 0], [10, 0], [10, 10], [7, 10], [7, 3], [3, 3], [3, 10], [0, 10]], np.float32)
        self.assertEqual(polygon_rectangle_intersection_area(polygon, (4, 4, 6, 8)), 0)

    def test_disconnected_intersections_and_reverse_winding(self):
        polygon = np.array([[0, 0], [10, 0], [10, 10], [7, 10], [7, 3], [3, 3], [3, 10], [0, 10]], np.float32)
        for points in (polygon, polygon[::-1]):
            self.assertAlmostEqual(polygon_rectangle_intersection_area(points, (-1, 5, 11, 9)), 24)

    def test_invalid_boxes_have_no_area(self):
        for box in ((0, 0, 0, 5), (5, 5, 0, 0), (0, 0, float("nan"), 5)):
            with self.subTest(box=box):
                self.assertFalse(self.occupied(box))

    def test_invalid_strategy_or_threshold_rejected(self):
        for method, threshold in (("iou", 0.2), ("overlap", 0), ("overlap", 1.1),
                                  ("center", float("nan"))):
            with self.subTest(method=method, threshold=threshold), self.assertRaises(ValueError):
                calculate_occupancy([self.slot], [], method=method, overlap_threshold=threshold)

    def test_no_vehicles_means_all_available(self):
        self.assertEqual(calculate_occupancy([self.slot], [], method="overlap").available_count, 1)
