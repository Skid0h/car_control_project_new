import math
import unittest
from car_jetson.actuation import Actuator, Limits
from car_jetson.session import Session


class ActuationTests(unittest.TestCase):
    def make(self):
        a = Actuator(Limits())
        a.packet(0)
        return a

    def test_all_changed_commands_rate_limited(self):
        a = self.make()
        sent = []
        for i in range(1, 1001):
            now = i / 1000
            a.set_target(0.3 + 0.2 * (i % 2), math.sin(now * 8), now)
            if a.packet(now):
                sent.append(now)
        self.assertLessEqual(len(sent), 20)
        self.assertTrue(all(y - x >= 0.05 - 1e-8 for x, y in zip(sent, sent[1:])))

    def test_identical_heartbeat_not_command_spam(self):
        a = self.make()
        times = []
        for i in range(1, 101):
            t = i / 100
            a.set_target(0, 0, t)
            if a.packet(t):
                times.append(t)
        self.assertEqual(len(times), 5)
        self.assertAlmostEqual(times[0], 0.05)
        self.assertAlmostEqual(times[1], 0.25)

    def test_hysteresis_ignores_jitter_but_accumulates_small_drift(self):
        a = self.make()
        for steering in (0.01, -0.02, 0.03, -0.01):
            a.set_target(0, steering, 0.1)
            self.assertEqual(a.target_angle, 100)
        a.set_target(0, 0.05, 0.2)
        self.assertEqual(a.target_angle, 97.5)

    def test_slew_and_calibrated_stops(self):
        a = self.make()
        prev = a.angle
        for i in range(1, 101):
            t = i / 100
            a.set_target(1, 1, t)
            a.packet(t)
            self.assertLessEqual(abs(a.angle - prev), 0.60001)
            self.assertGreaterEqual(a.angle, 70)
            prev = a.angle
        self.assertEqual(a.angle, 70)
        a.set_target(1, -1, 1)
        a.packet(20)
        self.assertEqual(a.last_packet[0], 1500)
        self.assertLessEqual(a.angle - 70, 3.001)  # no giant slew after a scheduling stall
        self.assertEqual(a.target_angle, 100)

    def test_emergency_neutral_then_slews_to_center(self):
        a = self.make()
        a.set_target(1, 1, 0.05)
        a.packet(0.05)
        angle = a.angle
        a.stop()
        self.assertEqual(a.packet(0.051), ('<1500,%d,S>' % round(a.angle)).encode())
        self.assertLessEqual(abs(a.angle - angle), 0.061)
        self.assertEqual(a.target_angle, a.p.center)
        for i in range(1, 61):
            a.packet(0.051 + i * 0.01)
        self.assertAlmostEqual(a.angle, a.p.center)

    def test_watchdog_measures_input_not_output_heartbeat(self):
        a = self.make()
        a.set_target(1, 0, 0.05)
        a.packet(0.05)
        a.packet(0.25)
        self.assertEqual(a.packet(0.36), b'<1500,100,S>')

    def test_nan_and_out_of_range_stop(self):
        for bad in (math.nan, math.inf, 2, -2):
            a = self.make()
            a.set_target(1, 0, 0.05)
            a.packet(0.05)
            self.assertFalse(a.set_target(bad, 0, 0.06))
            self.assertEqual(a.packet(0.06), b'<1500,100,S>')

    def test_stop_marker_emitted_even_at_neutral_motor(self):
        a = self.make()
        a.set_target(0, 1, 0.05)  # manual steering while stationary is supported
        self.assertEqual(a.packet(0.05), b'<1500,97>')
        a.stop()
        self.assertEqual(a.packet(0.051), b'<1500,97,S>')

    def test_invalid_limits(self):
        for kwargs in (dict(rate_hz=0), dict(heartbeat_s=0.5), dict(slew_deg_s=-1),
                       dict(steering_min=100), dict(forward=2200), dict(deadband_deg=math.nan)):
            with self.assertRaises(ValueError):
                Limits(**kwargs)


class SessionTests(unittest.TestCase):
    lane = [(-0.75, 1, 'blue'), (0.75, 1, 'yellow')]
    def setUp(self):
        self.a = Actuator(Limits())
        self.s = Session(self.a)
        self.s.capture_speed_limits()
        self.s.tick(0)

    def test_pc_protocol_auto_and_manual_override(self):
        self.s.receive('A', 0.01)
        self.s.receive('0,0', 0.02)
        self.assertTrue(self.s.auto)
        self.s.cones(self.lane, 0.1, 0.03)
        self.assertGreater(self.a.target_motor, 1500)
        self.s.receive('-1,0.5', 0.11)
        self.assertFalse(self.s.auto)
        self.assertEqual(self.a.target_motor, 1420)
        self.s.receive('R', 0.12)
        self.assertTrue(self.s.record)
        self.s.receive('C', 0.13)
        self.assertFalse(self.s.record)

    def test_no_cones_stop_instead_of_drive_blind(self):
        self.s.receive('A', 0.01)
        self.s.cones(self.lane, 0.1, 0.03)
        self.s.tick(0.1)
        self.s.cones([], 0.11, 0)
        self.assertEqual(self.s.tick(0.11), b'<1500,100,S>')

    def test_invalid_udp_cannot_keep_moving(self):
        self.s.receive('1,0', 0.05)
        self.s.tick(0.05)
        for text in ('nan,0', '1,inf', '2,0', 'garbage', 'speed:2000,1000'):
            self.assertFalse(self.s.receive(text, 0.06))
            self.assertEqual(self.a.target_motor, 1500)

    def test_stale_future_cones_do_not_refresh(self):
        self.s.receive('A', 0.01)
        for age in (0.5, -1, math.nan):
            self.s.cones(self.lane, 0.1, age)
            self.assertEqual(self.a.target_motor, 1500)

    def test_camera_freeze_requires_rearm(self):
        self.s.receive('A', 0.01)
        self.s.cones(self.lane, 0.1, 0)
        self.s.tick(0.1)
        self.s.receive('0,0', 0.45)
        self.s.tick(0.51)
        self.assertFalse(self.s.auto)
        self.s.cones(self.lane, 0.52, 0)
        self.assertEqual(self.a.target_motor, 1500)

    def test_pc_loss_disables_auto_even_with_camera(self):
        self.s.receive('A', 0.01)
        self.s.cones(self.lane, 0.1, 0)
        self.s.cones(self.lane, 0.6, 0)
        self.assertFalse(self.s.auto)

    def test_finish_latched_until_explicit_rearm(self):
        self.s.receive('A', 0.01)
        self.s.cones([(0, 0.2, 'orange')], 0.1, 0)
        self.s.cones(self.lane, 0.2, 0)
        self.assertEqual(self.a.target_motor, 1500)
        self.s.receive('S', 0.21)
        self.s.receive('A', 0.22)
        self.s.cones(self.lane, 0.3, 0)
        self.assertGreater(self.a.target_motor, 1500)

    def test_stop_is_not_cancelled_by_pc_neutral_heartbeat(self):
        self.s.receive('1,1', 0.05)
        self.s.tick(0.05)
        self.s.receive('S', 0.06)
        self.assertEqual(self.s.tick(0.06)[:6], b'<1500,')
        self.assertTrue(self.a.stopped)
        for i in range(1, 20):
            t = 0.06 + i * 0.05
            self.s.receive('0,0', t)
            self.s.tick(t)
            self.assertTrue(self.a.stopped)
        self.assertAlmostEqual(self.a.angle, self.a.p.center)
        self.s.receive('0,1', 1.1)  # explicit steering request rearms manual
        self.assertFalse(self.a.stopped)
        self.assertEqual(self.a.target_motor, 1500)

    def test_remote_speed_ceiling_and_stop(self):
        self.s.receive('speed:1560,1430', 0.05)
        self.s.receive('1,0', 0.06)
        self.assertEqual(self.a.target_motor, 1560)
        self.s.receive('S', 0.07)
        self.assertEqual(self.a.target_motor, 1500)
        self.s.receive('Q', 0.08)
        self.assertTrue(self.s.quit)
