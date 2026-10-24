"""Run with python3 -m unittest discover -s ros/car_control_ros/test."""
import math
from pathlib import Path
import sys
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))
from car_control_ros.core import (Parameters, Controller, Command, Bicycle, circle_track,
                                  world_to_cones, valid_command, fresh)
from car_control_ros.legacy import load_module


class CoreTests(unittest.TestCase):
    def setUp(self):
        self.controller = Controller(Parameters())

    @staticmethod
    def lane(offset=0):
        return [(offset - 0.75, z, 'blue') for z in (0.5, 1, 2, 3)] + [
            (offset + 0.75, z, 'yellow') for z in (0.5, 1, 2, 3)]

    def test_straight_and_steering_sign(self):
        self.assertAlmostEqual(self.controller.step(self.lane(), 0.02).steering, 0)
        self.controller.reset()
        self.assertGreater(self.controller.step(self.lane(0.3), 0.02).steering, 0)
        self.controller.reset()
        self.assertLess(self.controller.step(self.lane(-0.3), 0.02).steering, 0)

    def test_one_boundary(self):
        self.assertAlmostEqual(self.controller.step(self.lane()[:4], 0.02).steering, 0)

    def test_no_cones_and_invalid_cones_brake(self):
        self.controller.step(self.lane(), 0.02)
        self.assertEqual(self.controller.step([], 0.02), Command())
        self.assertEqual(self.controller.step([(math.nan, 2, 'blue'), (0, -2, 'yellow')], 0.02), Command())

    def test_orange_latched(self):
        self.assertEqual(self.controller.step([(0, 0.05, 'orange')], 0.02), Command())
        self.assertEqual(self.controller.step(self.lane(), 0.02), Command())
        self.controller.reset()
        self.assertGreater(self.controller.step(self.lane(), 0.02).throttle, 0)

    def test_fsds_orange_is_not_finish(self):
        c = Controller(Parameters(stop_on_orange=False))
        self.assertGreater(c.step(self.lane() + [(0, 0.2, 'orange')], 0.02).throttle, 0)

    def test_clock_and_command_validation(self):
        for stamp in (None, 0, 11, math.nan):
            self.assertFalse(fresh(stamp, 10, 0.4))
        self.assertTrue(fresh(9.8, 10, 0.4))
        for cmd in (Command(math.nan, 0, 0), Command(1, 0, 1), Command(0, 2, 0)):
            self.assertFalse(valid_command(cmd))
        self.assertEqual(self.controller.step(self.lane(), 0), Command())

    def test_coordinates(self):
        self.assertEqual(world_to_cones([(5, 1, 'blue')], 0, 0, 0, 10), [(-1, 5, 'blue')])
        cones = world_to_cones([(-1, 5, 'blue')], 0, 0, math.pi / 2, 10)
        self.assertAlmostEqual(cones[0][0], -1)
        self.assertAlmostEqual(cones[0][1], 5)

    def test_existing_config_loads_without_hardware(self):
        root = Path(__file__).resolve().parents[3] / 'Jetson Xavier'
        config = load_module(root / 'Code' / 'Config_load.py', 'test_config').Config()
        self.assertEqual(config.track_width, Parameters().track_width)

    def test_closed_loop_two_laps(self):
        car = Bicycle()
        track = circle_track()
        worst = 0
        for _ in range(12000):
            cones = world_to_cones(track, car.x, car.y, car.yaw, 4.0)
            command = self.controller.step(cones, 0.02)
            self.assertTrue(valid_command(command))
            car.step(command, 0.02)
            worst = max(worst, abs(math.hypot(car.x, car.y - 8) - 8))
        self.assertGreater(car.yaw, 4 * math.pi)
        self.assertLess(worst, 0.5)


if __name__ == '__main__':
    unittest.main()
