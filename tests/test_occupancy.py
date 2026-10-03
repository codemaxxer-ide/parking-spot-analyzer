import unittest

import numpy as np

from src.detector import Detection
from src.occupancy import calculate_occupancy
from src.parking_slots import ParkingSlot


def vehicle(x, y):
    return Detection((x - 2, y - 2, x + 2, y + 2), "car", 0.9)


class OccupancyTests(unittest.TestCase):
    def setUp(self):
        self.slots = [
            ParkingSlot("left", "P1", np.array([[0, 0], [10, 0], [10, 10], [0, 10]], np.float32)),
            ParkingSlot("right", "P2", np.array([[20, 0], [30, 0], [30, 10], [20, 10]], np.float32)),
        ]

    def test_point_clearly_inside_polygon(self):
        result = calculate_occupancy(self.slots, [vehicle(5, 5)])
        self.assertEqual(result.occupied, {"left": True, "right": False})
        self.assertEqual((result.total, result.occupied_count, result.available_count), (2, 1, 1))

    def test_point_clearly_outside_polygon(self):
        self.assertEqual(calculate_occupancy(self.slots, [vehicle(15, 15)]).occupied_count, 0)

    def test_multiple_vehicles_in_one_slot_count_once(self):
        result = calculate_occupancy(self.slots, [vehicle(3, 3), vehicle(7, 7)])
        self.assertEqual((result.occupied_count, result.available_count), (1, 1))

    def test_no_vehicles_means_all_vacant(self):
        result = calculate_occupancy(self.slots, [])
        self.assertEqual(result.occupied, {"left": False, "right": False})
        self.assertEqual(result.available_count, 2)

    def test_boundary_is_occupied(self):
        self.assertTrue(calculate_occupancy(self.slots, [vehicle(0, 5)]).occupied["left"])

    def test_bbox_overlap_without_center_does_not_occupy(self):
        result = calculate_occupancy(self.slots, [Detection((8, 2, 18, 8), "truck", 0.8)])
        self.assertEqual(result.occupied_count, 0)

    def test_concave_polygon_uses_polygon_not_bounding_rectangle(self):
        slot = ParkingSlot("L", "P1", np.array(
            [[0, 0], [10, 0], [10, 4], [4, 4], [4, 10], [0, 10]], np.float32))
        self.assertFalse(calculate_occupancy([slot], [vehicle(8, 8)]).occupied["L"])
        self.assertTrue(calculate_occupancy([slot], [vehicle(2, 8)]).occupied["L"])

    def test_all_slots_occupied(self):
        result = calculate_occupancy(self.slots, [vehicle(5, 5), vehicle(25, 5)])
        self.assertEqual((result.occupied_count, result.available_count), (2, 0))
