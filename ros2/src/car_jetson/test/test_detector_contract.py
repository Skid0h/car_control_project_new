"""Output parsing tests do not pretend to run CUDA."""
import importlib.util
from pathlib import Path
import sys
import types
import unittest
from unittest.mock import patch
import numpy as np


class DetectorContractTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        fake = {name: types.ModuleType(name) for name in ('cv2', 'tensorrt', 'pycuda', 'pycuda.driver')}
        with patch.dict(sys.modules, fake):
            file = Path(__file__).resolve().parents[1] / 'car_jetson/detector.py'
            spec = importlib.util.spec_from_file_location('detector_contract_under_test', file)
            module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(module)
            cls.rows = staticmethod(module.output_rows)

    def test_yolo_channel_first_and_channel_last(self):
        raw = np.arange(1 * 7 * 8400).reshape(1, 7, 8400)
        np.testing.assert_array_equal(self.rows(raw, 3), raw[0].T)
        np.testing.assert_array_equal(self.rows(raw.transpose(0, 2, 1), 3), raw[0].T)

    def test_wrong_classes_nms_and_batch_rejected(self):
        for shape in ((1, 84, 8400), (1, 8, 8400), (2, 7, 8400), (1, 100, 6), (7, 8400)):
            with self.assertRaises(ValueError):
                self.rows(np.zeros(shape), 3)
