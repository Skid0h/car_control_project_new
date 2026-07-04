"""
Тест FPS: ConeDetector + ZED + Web-трансляция.
Замеряет скорость YOLO-детекции без лишней логики управления.
"""

import time
import logging
from datetime import datetime
import socket
import json
import cv2
import pyzed.sl as sl
import numpy as np
import os
import sys
import threading

sys.path.insert(0, os.path.dirname(__file__))

from Code.Config_load import Config
from Code.Cone_detector import ConeDetector
from Code.Web import start, set_frame

logging.basicConfig(level=logging.INFO, format='%(asctime)s [%(levelname)s] %(message)s')
logger = logging.getLogger(__name__)


class RecController:
    """Управление записью через UDP (команды R/C с пульта)"""
    def __init__(self, config):
        self.config = config
        self.is_recording = False
        self.video_writer = None
        self.video_path = None
        self.lock = threading.Lock()

    def start(self, frame):
        with self.lock:
            if self.video_writer is not None:
                return
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            self.video_path = os.path.join(self.config.output_folder, f"test_{timestamp}.avi")
            if not os.path.exists(self.config.output_folder):
                os.makedirs(self.config.output_folder)
            height, width = frame.shape[:2]
            fourcc = cv2.VideoWriter_fourcc(*'MJPG')
            self.video_writer = cv2.VideoWriter(self.video_path, fourcc, self.config.zed_fps, (width, height))
            self.is_recording = True
            logger.info(f"Запись начата: {self.video_path}")

    def stop(self):
        with self.lock:
            if self.video_writer is not None:
                self.video_writer.release()
                self.video_writer = None
                logger.info(f"Запись остановлена: {self.video_path}")
            self.is_recording = False

    def write(self, frame):
        with self.lock:
            if self.video_writer is not None:
                self.video_writer.write(frame)

    def close(self):
        self.stop()


def udp_listener(config, rec_ctrl, stop_event):
    """Поток для приёма UDP-команд"""
    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    sock.settimeout(config.socket_timeout)
    try:
        sock.bind((config.udp_ip, config.udp_port))
    except Exception as e:
        logger.error(f"UDP bind error: {e}")
        return

    logger.info(f"UDP слушатель на порту {config.udp_port}")

    while not stop_event.is_set():
        try:
            data, addr = sock.recvfrom(1024)
            cmd = data.decode('utf-8').strip()

            if cmd == "R":
                rec_ctrl.is_recording = True
            elif cmd == "C":
                rec_ctrl.is_recording = False

            # Отправляем телеметрию обратно на пульт
            telemetry = {
                "mode": "TEST",
                "rec": rec_ctrl.is_recording,
                "cam_connected": True,
                "fwd": 0,
                "bck": 0,
                "msg": ""
            }
            sock.sendto(json.dumps(telemetry).encode('utf-8'), addr)

        except socket.timeout:
            pass
        except Exception as e:
            logger.error(f"UDP error: {e}")

    sock.close()


def main():
    config = Config("config.jsonc")

    # Запуск веб-сервера
    start()
    logger.info("Веб-сервер: http://<ip_jetson>:5000")

    # Загрузка детектора
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

    logger.info("ZED камера открыта")

    runtime_params = sl.RuntimeParameters()
    image_zed = sl.Mat()

    # Управление записью
    rec_ctrl = RecController(config)

    # UDP-слушатель в отдельном потоке
    stop_event = threading.Event()
    udp_thread = threading.Thread(
        target=udp_listener, args=(config, rec_ctrl, stop_event), daemon=True
    )
    udp_thread.start()

    # Счётчики FPS
    fps_counter = 0
    fps_last_time = time.time()
    current_fps = 0

    # Статистика времени детекции (скользящее среднее за 30 кадров)
    detect_times = []
    SMOOTH_WINDOW = 30

    logger.info("=" * 50)
    logger.info("ТЕСТ FPS ЗАПУЩЕН")
    logger.info("Открой в браузере http://<ip_jetson>:5000")
    logger.info("Управление: R — старт записи, C — стоп (с пульта)")
    logger.info("Нажми Ctrl+C для остановки")
    logger.info("=" * 50)

    try:
        while True:
            if zed.grab(runtime_params) == sl.ERROR_CODE.SUCCESS:
                zed.retrieve_image(image_zed, sl.VIEW.LEFT)

                img_data = image_zed.get_data()
                if img_data.shape[2] == 4:
                    image_np = cv2.cvtColor(img_data, cv2.COLOR_BGRA2BGR)
                else:
                    image_np = img_data

                # Управление записью по флагу из UDP
                if rec_ctrl.is_recording:
                    if rec_ctrl.video_writer is None:
                        rec_ctrl.start(image_np)
                    rec_ctrl.write(image_np)
                else:
                    if rec_ctrl.video_writer is not None:
                        rec_ctrl.stop()

                # Замер времени детекции
                t_start = time.perf_counter()
                detections = detector.detect(image_np)
                t_end = time.perf_counter()

                detect_ms = (t_end - t_start) * 1000
                detect_times.append(detect_ms)
                if len(detect_times) > SMOOTH_WINDOW:
                    detect_times.pop(0)
                avg_detect_ms = sum(detect_times) / len(detect_times)

                # Сбор статистики по цветам
                blues = sum(1 for d in detections if d.get('name') == 'blue')
                yellows = sum(1 for d in detections if d.get('name') == 'yellow')
                oranges = sum(1 for d in detections if d.get('name') == 'orange')

                # Отрисовка информации на кадре
                cv2.putText(image_np, f"FPS: {current_fps}", (10, 30),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                cv2.putText(image_np, f"Detect: {detect_ms:.1f}ms (avg: {avg_detect_ms:.1f}ms)",
                           (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
                cv2.putText(image_np, f"Blue:{blues} Yellow:{yellows} Orange:{oranges}",
                           (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

                # Индикатор записи
                if rec_ctrl.is_recording:
                    cv2.putText(image_np, "REC", (10, 120),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

                # Отрисовка рамок детекций
                for det in detections:
                    x1, y1, x2, y2 = det['bbox']
                    name = det.get('name', '?')
                    conf = det.get('conf', 0)

                    if name == 'blue':
                        color = (255, 0, 0)
                    elif name == 'yellow':
                        color = (0, 255, 255)
                    elif name == 'orange':
                        color = (0, 165, 255)
                    else:
                        color = (255, 255, 255)

                    cv2.rectangle(image_np, (x1, y1), (x2, y2), color, 2)
                    cv2.putText(image_np, f"{name} {conf:.2f}", (x1, y1-10),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

                # Подсчёт FPS (обновление раз в секунду)
                fps_counter += 1
                elapsed = time.time() - fps_last_time
                if elapsed >= 1.0:
                    current_fps = int(fps_counter / elapsed)
                    rec_status = "REC" if rec_ctrl.is_recording else ""
                    logger.info(
                        f"FPS: {current_fps} {rec_status} | "
                        f"Detect: {avg_detect_ms:.1f}ms ({1000/max(avg_detect_ms, 0.1):.1f} FPS детекции) | "
                        f"Конусов: B{blues} Y{yellows} O{oranges}"
                    )
                    fps_counter = 0
                    fps_last_time = time.time()

                # Отправка кадра на веб
                set_frame(image_np)

    except KeyboardInterrupt:
        logger.info("Тест остановлен")
    finally:
        stop_event.set()
        udp_thread.join(timeout=1.0)
        rec_ctrl.close()
        zed.close()
        if detect_times:
            final_avg = sum(detect_times) / len(detect_times)
            logger.info(f"Итог: средняя детекция {final_avg:.1f}ms ({1000/max(final_avg, 0.1):.1f} FPS)")
        logger.info("ZED закрыта")


if __name__ == "__main__":
    main()
