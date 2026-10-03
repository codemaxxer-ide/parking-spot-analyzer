import unittest

import numpy as np

from src.geometry import polygon_rectangle_intersection_area
from src.occupancy import OccupancyResult
from src.parking_slots import ParkingSlot
from src.visualization import _dashboard_bounds, annotate_frame


class VisualizationTests(unittest.TestCase):
    def test_dashboard_chooses_corner_without_parking_content(self):
        slot = ParkingSlot("s", "P1", np.array([[0, 0], [200, 0], [200, 150], [0, 150]], np.float32))
        rectangle = _dashboard_bounds(640, 360, 200, 120, [slot])
        self.assertEqual(polygon_rectangle_intersection_area(slot.polygon, rectangle), 0)

    def test_small_frame_annotation_does_not_crash(self):
        frame = np.zeros((8, 8, 3), np.uint8)
        annotated = annotate_frame(frame, [], [], OccupancyResult({}))
        self.assertEqual(annotated.shape, frame.shape)
