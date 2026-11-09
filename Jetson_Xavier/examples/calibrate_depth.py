"""Веб-стрим для проверки расстояния до конусов по площади рамки.

Запуск на Jetson: python3 examples/calibrate_depth.py
На телефоне в той же сети: http://<IP Jetson>:5000/
"""

import logging
import math
from pathlib import Path
import sys
import time

import cv2
import pyzed.sl as sl

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from Code.Config_load import Config
from Code.Cone_detector import ConeDetector
from Code.Web import set_frame, start


def detect_cones(detector, frame):
    """Повторяет геометрию подготовки кадра и пересчёта рамок в server.py."""
    full_h, full_w = frame.shape[:2]
    if full_w > 640 or full_h > 480:
        scale = min(480 / full_w, 270 / full_h)
        size = (max(320, int(full_w * scale)), max(180, int(full_h * scale)))
        frame = cv2.resize(frame, size, interpolation=cv2.INTER_AREA)

    height, width = frame.shape[:2]
    crop = int(height * 0.20)
    cropped = frame[crop:height - crop]
    scale_x, scale_y = full_w / width, full_h / height
    offset_y = int(crop * scale_y)

    cones = []
    for det in detector.detect(cropped):
        x1, y1, x2, y2 = det['bbox']
        clipped = x1 <= 0 or y1 <= 0 or x2 >= width or y2 >= cropped.shape[0]
        x1, x2 = int(x1 * scale_x), int(x2 * scale_x)
        y1, y2 = int(y1 * scale_y) + offset_y, int(y2 * scale_y) + offset_y
        area = max(x2 - x1, 1) * max(y2 - y1, 1)
        cones.append((det['name'], (x1, y1, x2, y2), area, clipped))
    return cones


def main():
    logging.basicConfig(level=logging.INFO)
    config = Config(str(PROJECT_ROOT / 'config.jsonc'))
    detector = ConeDetector(config)

    zed = sl.Camera()
    params = sl.InitParameters()
    params.camera_resolution = getattr(sl.RESOLUTION, config.zed_resolution, sl.RESOLUTION.HD720)
    params.camera_fps = config.zed_fps
    params.depth_mode = sl.DEPTH_MODE.NONE
    if zed.open(params) != sl.ERROR_CODE.SUCCESS:
        sys.exit('Не удалось открыть ZED-камеру')

    runtime = sl.RuntimeParameters()
    image = sl.Mat()
    try:
        start()
        while True:
            if (zed.grab(runtime) != sl.ERROR_CODE.SUCCESS or
                    zed.retrieve_image(image, sl.VIEW.LEFT) != sl.ERROR_CODE.SUCCESS):
                time.sleep(0.05)
                continue

            frame = image.get_data()
            if frame.shape[2] == 4:
                frame = cv2.cvtColor(frame, cv2.COLOR_BGRA2BGR)
            else:
                frame = frame.copy()
            cones = detect_cones(detector, frame)
            for name, (x1, y1, x2, y2), area, clipped in cones:
                color = tuple(config.cone_colors.get(name, [255, 255, 255]))
                cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                z = config.area_depth_constant / math.sqrt(area)
                label = f'{name}: area={area} Z_cam={z:.2f} m'
                if clipped:
                    label += ' (clipped)'
                cv2.putText(frame, label, (x1, max(20, y1 - 8)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

            set_frame(frame)

    except KeyboardInterrupt:
        pass
    finally:
        zed.close()


if __name__ == '__main__':
    main()
