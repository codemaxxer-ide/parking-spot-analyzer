import unittest

from src.occupancy import OccupancyResult
from src.temporal import OccupancySmoother


class TemporalTests(unittest.TestCase):
    def setUp(self):
        self.smoother = OccupancySmoother(5, 3)

    def vote(self, occupied):
        return self.smoother.update(OccupancyResult({"P1": occupied})).occupied["P1"]

    def test_stable_occupied(self):
        states = [self.vote(True) for _ in range(10)]
        self.assertEqual(states[:3], [False, False, True])
        self.assertTrue(all(states[2:]))
        self.assertEqual(len(self.smoother.history["P1"]), 5)

    def test_one_frame_false_negative_does_not_immediately_flip(self):
        for _ in range(5):
            self.vote(True)
        self.assertTrue(self.vote(False))
        self.assertTrue(self.vote(True))

    def test_transition_eventually_occurs(self):
        for _ in range(5):
            self.vote(True)
        self.assertEqual([self.vote(False) for _ in range(3)], [True, True, False])

    def test_stable_available(self):
        self.assertFalse(any(self.vote(False) for _ in range(10)))

    def test_disabled_smoothing_tracks_every_observation(self):
        smoother = OccupancySmoother(1, 1)
        for raw in (True, False, True):
            self.assertEqual(smoother.update(OccupancyResult({"P1": raw})).occupied["P1"], raw)

    def test_slots_have_independent_histories_and_counts(self):
        for _ in range(3):
            result = self.smoother.update(OccupancyResult({"P1": True, "P2": False}))
        self.assertEqual(result.occupied, {"P1": True, "P2": False})
        self.assertEqual((result.total, result.occupied_count, result.available_count), (2, 1, 1))
        self.assertEqual(result.occupancy_percent, 50)

    def test_invalid_settings_rejected(self):
        for window, required in ((0, 0), (5, 6), (5, 0), (1.5, 1), (True, 1)):
            with self.subTest(window=window, required=required), self.assertRaises(ValueError):
                OccupancySmoother(window, required)
