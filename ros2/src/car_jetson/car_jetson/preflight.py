"""Read-only environment checks; never opens Serial or sends drive commands."""
import argparse
import importlib
import platform
import sys
from pathlib import Path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', required=True, help='Trusted cone TensorRT engine built on this Xavier')
    parser.add_argument('--project-config')
    parser.add_argument('--camera', action='store_true', help='Also open ZED and grab one frame')
    args = parser.parse_args()
    failures = []
    def check(name, fn):
        try:
            detail = fn()
            print('OK   %s: %s' % (name, detail))
        except Exception as exc:
            failures.append(name)
            print('FAIL %s: %s' % (name, exc))
    def require(condition, detail):
        if not condition:
            raise RuntimeError(detail)
        return detail
    check('architecture', lambda: require(platform.machine() == 'aarch64', platform.machine()))
    check('Python', lambda: require(sys.version_info[:2] == (3, 8), sys.version))
    check('Ubuntu', lambda: require('VERSION_ID="20.04"' in Path('/etc/os-release').read_text(), 'Ubuntu 20.04 required'))
    check('L4T', lambda: require('R35' in Path('/etc/nv_tegra_release').read_text() and
                                'REVISION: 5.0' in Path('/etc/nv_tegra_release').read_text(),
                                Path('/etc/nv_tegra_release').read_text().strip()))
    for name in ('rclpy', 'serial', 'numpy', 'cv2', 'tensorrt', 'pycuda.driver', 'pyzed.sl'):
        check(name, lambda name=name: getattr(importlib.import_module(name), '__version__', 'imported'))
    def inference():
        import numpy as np
        import tensorrt as trt
        from ament_index_python.packages import get_package_share_directory
        from .config_load import Config
        from .detector import ConeDetector
        if not trt.__version__.startswith('8.5.2'):
            raise RuntimeError('Expected TensorRT 8.5.2, got ' + trt.__version__)
        filename = args.project_config or str(Path(get_package_share_directory('car_jetson')) / 'config/project.jsonc')
        cfg = Config(filename)
        cfg.yolo_model_path = str(Path(args.model).expanduser().resolve())
        detector = ConeDetector(cfg)
        try:
            detector.detect(np.zeros((720, 1280, 3), dtype=np.uint8))
            return 'deserialization + synthetic inference passed (does not validate model accuracy)'
        finally:
            detector.close()
    check('engine/GPU', inference)
    if args.camera:
        def camera():
            import pyzed.sl as sl
            cam = sl.Camera()
            try:
                init = sl.InitParameters()
                init.camera_resolution = sl.RESOLUTION.HD720
                init.camera_fps = 15
                init.depth_mode = sl.DEPTH_MODE.NONE
                result = cam.open(init)
                if result != sl.ERROR_CODE.SUCCESS:
                    raise RuntimeError(str(result))
                if cam.grab(sl.RuntimeParameters()) != sl.ERROR_CODE.SUCCESS:
                    raise RuntimeError('grab failed')
                return str(cam.get_camera_information().camera_model)
            finally:
                cam.close()
        check('ZED camera', camera)
    if failures:
        raise SystemExit(1)
