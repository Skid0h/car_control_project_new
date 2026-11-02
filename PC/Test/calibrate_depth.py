"""
Калибровка area_depth_constant для оценки глубины по площади bbox.

Формула: z = K / sqrt(area)
Где:   K = area_depth_constant
       area = ширина_bbox * высота_bbox (пиксели)

Калибровка: поставь конус на известное расстояние D,
            K = D * sqrt(area)

Запуск на Jetson с ZED-камерой.
"""

import os
import sys
import math
import logging
import cv2
import pyzed.sl as sl

sys.path.insert(0, os.path.dirname(__file__))
from Code.Config_load import Config
from Code.Cone_detector import ConeDetector

logging.basicConfig(level=logging.INFO, format='%(asctime)s [%(levelname)s] %(message)s')
logger = logging.getLogger(__name__)

# Текущее значение из config.jsonc
CURRENT_K = 150.0
PRINT_COOLDOWN = 30  # выводить в консоль раз в N кадров
_frame_count = 0     # счётчик кадров

# Цвета для отрисовки
COLORS = {
    'blue':   (255, 0, 0),
    'yellow': (0, 255, 255),
    'orange': (0, 165, 255),
}


def main():
    config = Config("config.jsonc")
    detector = ConeDetector(config)

    # Инициализация ZED
    zed = sl.Camera()
    init_params = sl.InitParameters()
    init_params.camera_resolution = getattr(sl.RESOLUTION, config.zed_resolution, sl.RESOLUTION.HD720)
    init_params.camera_fps = config.zed_fps
    init_params.coordinate_units = getattr(sl.UNIT, config.coordinate_units, sl.UNIT.METER)

    if zed.open(init_params) != sl.ERROR_CODE.SUCCESS:
        logger.error("Не удалось открыть ZED-камеру")
        sys.exit(1)

    # Получаем фокусное расстояние
    cam_info = zed.get_camera_information()
    fx = cam_info.camera_configuration.calibration_parameters.left_cam.fx
    cx_cam = cam_info.camera_configuration.calibration_parameters.left_cam.cx
    logger.info(f"Камера: fx={fx:.1f}, cx={cx_cam:.1f}")
    logger.info(f"Текущий K (area_depth_constant) = {CURRENT_K}")

    runtime_params = sl.RuntimeParameters()
    image_zed = sl.Mat()

    print("\n" + "=" * 60)
    print("КАЛИБРОВКА ГЛУБИНЫ ПО ПЛОЩАДИ КОНУСА")
    print("=" * 60)
    print(f"\nФормула:      z = K / sqrt(bbox_area)")
    print(f"Текущий K:    {CURRENT_K}")
    print(f"\nИнструкция:")
    print(f"  1. Поставь конус на известное расстояние (например 1м)")
    print(f"  2. Смотри в консоль — увидишь bbox_area и расчётный K")
    print(f"  3. Введи реальное расстояние конуса → скрипт посчитает K")
    print(f"  4. Повтори для 3-5 конусов, возьми среднее")
    print(f"  5. Обнови area_depth_constant в config.jsonc")
    print(f"\nНажми 'q' для выхода")
    print("=" * 60 + "\n")

    cv2.namedWindow('Calibration', cv2.WINDOW_NORMAL)
    cv2.resizeWindow('Calibration', 1280, 720)

    # Для сбора измерений
    measurements = []

    try:
        while True:
            if zed.grab(runtime_params) == sl.ERROR_CODE.SUCCESS:
                zed.retrieve_image(image_zed, sl.VIEW.LEFT)

                img_data = image_zed.get_data()
                if img_data.shape[2] == 4:
                    image_np = cv2.cvtColor(img_data, cv2.COLOR_BGRA2BGR)
                else:
                    image_np = img_data

                detections = detector.detect(image_np)

                # Инфо для консоли
                cone_info_lines = []

                for det in detections:
                    x1, y1, x2, y2 = det['bbox']
                    name = det.get('name', '?')
                    conf = det.get('conf', 0)
                    width = max(x2 - x1, 1)
                    height = max(y2 - y1, 1)
                    area = width * height
                    sqrt_area = math.sqrt(area)

                    # Текущая оценка глубины
                    z_current = CURRENT_K / sqrt_area

                    # Какой нужен K для расстояний 1м, 2м, 3м
                    k_1m = 1.0 * sqrt_area
                    k_2m = 2.0 * sqrt_area
                    k_3m = 3.0 * sqrt_area

                    # Отрисовка на кадре
                    color = COLORS.get(name, (255, 255, 255))
                    cv2.rectangle(image_np, (x1, y1), (x2, y2), color, 2)
                    cv2.putText(image_np, f"{name} {conf:.2f}", (x1, y1-10),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
                    cv2.putText(image_np, f"Area:{area:.0f} Z:{z_current:.2f}m",
                               (x1, y2+20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

                    cone_info_lines.append(
                        f"  {name:6s} | area={area:6.0f} | sqrt(area)={sqrt_area:5.1f} | "
                        f"Текущий Z={z_current:.2f}м | "
                        f"K для 1м={k_1m:.0f} | K для 2м={k_2m:.0f} | K для 3м={k_3m:.0f}"
                    )

                # Статистика
                cv2.putText(image_np, f"Текущий K = {CURRENT_K}", (10, 30),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                cv2.putText(image_np, f"Конусов: {len(detections)}", (10, 60),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                cv2.putText(image_np, "Нажми 'q' для выхода", (10, 90),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (100, 100, 100), 1)

                cv2.imshow('Calibration', image_np)

                # Вывод в консоль (только 1 раз в PRINT_COOLDOWN кадров чтобы не спамить)
                global _frame_count
                _frame_count += 1
                if cone_info_lines and _frame_count % PRINT_COOLDOWN == 1:
                    print("\n--- КОНУСЫ НА КАДРЕ ---")
                    for line in cone_info_lines:
                        print(line)
                    print(f"💡 Нажми 'c' для калибровки или 'q' для выхода")

                key = cv2.waitKey(1) & 0xFF
                if key == ord('q'):
                    break
                elif key == ord('c'):  # калибровка по первому конусу
                    if detections:
                        det = detections[0]
                        area_w = max(det['bbox'][2] - det['bbox'][0], 1)
                        area_h = max(det['bbox'][3] - det['bbox'][1], 1)
                        area = area_w * area_h
                        sqrt_a = math.sqrt(area)
                        try:
                            real_dist = float(input(f"  Введи реальное расстояние (м) до конуса '{det.get('name','?')}': "))
                            if real_dist > 0:
                                new_k = real_dist * sqrt_a
                                measurements.append(new_k)
                                avg_k = sum(measurements) / len(measurements)
                                print(f"\n  ✅ K = {real_dist:.1f}м * sqrt({area:.0f}) = {new_k:.1f}")
                                print(f"  📊 Измерений: {len(measurements)}, средний K = {avg_k:.1f}")
                                print(f"  💡 Рекомендуемое значение area_depth_constant = {avg_k:.1f}")
                            else:
                                print("  ❌ Расстояние должно быть > 0")
                        except ValueError:
                            print("  ❌ Введи число, например 1.5")

    except KeyboardInterrupt:
        pass
    finally:
        zed.close()
        cv2.destroyAllWindows()

        if measurements:
            avg_k = sum(measurements) / len(measurements)
            print("\n" + "=" * 60)
            print("ИТОГ КАЛИБРОВКИ")
            print("=" * 60)
            print(f"  Измерений: {len(measurements)}")
            print(f"  Средний K: {avg_k:.1f}")
            print(f"  Текущий K: {CURRENT_K}")
            print(f"  Изменение: {'+' if avg_k > CURRENT_K else ''}{avg_k - CURRENT_K:.1f}")
            if abs(avg_k - CURRENT_K) / CURRENT_K > 0.15:
                print(f"\n  ⚠️  Отклонение > 15%! Нужно обновить config.jsonc:")
                print(f"      \"area_depth_constant\": {avg_k:.1f}")
            else:
                print(f"\n  ✅ K={CURRENT_K} в пределах нормы (±15%)")
            print("=" * 60)
        else:
            print("\nИзмерений не было. Запусти скрипт снова и нажми 'c' для калибровки.")


if __name__ == "__main__":
    main()
