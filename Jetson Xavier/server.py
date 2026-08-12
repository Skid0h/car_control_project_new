"""
Запускается на Jetson.
Исправленная версия server.py

Фиксы записи:
- кадры записи выравниваются по реальным таймстампам (нет слоу-мо/ускорений)
- дырки заполняются дубликатами, лишние кадры выбрасываются
- убран setpts-хак ffmpeg, конвертация = простой транскод
- кадр всегда копируется из буфера SDK (нет рваных кадров)
- дедупликация детекций (нет двойных рамок)
- конусы и траектория рисуются каждый кадр
"""

import socket
import time
import logging
import sys
import json
import math
import subprocess
from datetime import datetime

import cv2
import pyzed.sl as sl
import threading
import numpy as np
import os
import queue

try:
    CUDA_AVAILABLE = cv2.cuda.getCudaEnabledDeviceCount() > 0
except Exception:
    CUDA_AVAILABLE = False

from Code.Config_load import Config
from Code.Car_control import CarController
from Code.Cone_detector import ConeDetector
from Code.Web import start, set_frame

logging.basicConfig(level=logging.INFO, format='%(asctime)s [%(levelname)s] %(message)s')
logger = logging.getLogger(__name__)

config = Config("config.jsonc")

# Диагностика через переменные окружения:
# DISABLE_WEB=1    -> не запускать Web и не отправлять кадры
# FORCE_HD720_60=1 -> форсировать HD720 60 FPS
# USE_CUDA=1       -> включить CUDA-конвейер
DISABLE_WEB = os.environ.get("DISABLE_WEB", "0") == "1"
FORCE_HD720_60 = os.environ.get("FORCE_HD720_60", "0") == "1"
USE_CUDA = os.environ.get("USE_CUDA", "0") == "1" and CUDA_AVAILABLE


class VisionLoop:
    def __init__(self, config, detector, car, robot_state):
        self.config = config
        self.detector = detector
        self.car = car
        self.robot_state = robot_state

        # --- АТРИБУТЫ ДО ЗАПУСКА ПОТОКОВ ---
        self.running = True
        self.is_recording = False
        self.frame_counter = 0

        target_fps = float(getattr(self.config, "target_fps", self.config.zed_fps))
        self.process_every = max(
            1,
            int(round(self.config.zed_fps / max(1.0, target_fps)))
        )

        publish_fps = float(getattr(self.config, "publish_fps", 0.0) or 0.0)
        if publish_fps <= 0:
            publish_fps = float(getattr(self.config, "web_fps", 0.0) or 0.0)

        if publish_fps > 0:
            self.publish_every = max(
                1,
                int(round(self.config.zed_fps / max(1.0, publish_fps)))
            )
        else:
            self.publish_every = max(1, self.process_every)

        logger.info(
            f"FPS SETTINGS: zed_fps={self.config.zed_fps}, "
            f"target_fps={target_fps}, "
            f"publish_fps={publish_fps if publish_fps > 0 else 'same_as_target'}, "
            f"process_every={self.process_every}, "
            f"publish_every={self.publish_every}"
        )

        self.last_detections = []
        self.last_detect_frame_shape = None
        self.last_detection_time = 0.0

        self.smooth_tx = 0.0
        self.smooth_tz = self.config.lookahead_distance

        self.pid_integral = 0.0
        self.pid_last_error = 0.0
        self.last_pid_time = time.time()

        self.fx = 0
        self.cx_cam = 0

        self.rec_dropped_frames = 0

        if not os.path.exists(self.config.output_folder):
            os.makedirs(self.config.output_folder)

        # --- ОЧЕРЕДИ И ПОТОКИ ---
        self.detect_queue = queue.Queue(maxsize=1)
        self.result_queue = queue.Queue(maxsize=1)

        self.detect_thread = threading.Thread(target=self._detect_loop, daemon=True)
        self.detect_thread.start()

        self.zed = sl.Camera()

        rec_queue_size = int(getattr(self.config, "rec_queue_size", 120))
        self.rec_queue = queue.Queue(maxsize=rec_queue_size)

        self.rec_thread = threading.Thread(target=self._rec_loop, daemon=True)
        self.rec_thread.start()

        self.vision_thread = threading.Thread(target=self._vision_loop, daemon=True)
        self.vision_thread.start()

    def _detect_loop(self):
        """Отдельный поток для YOLO — не блокирует grab()"""
        while self.running:
            try:
                frame = self.detect_queue.get(timeout=0.5)
            except queue.Empty:
                continue

            frame_shape = frame.shape[:2]

            try:
                detections = self.detector.detect(frame)
            except Exception as e:
                logger.error(f"Ошибка детектора: {e}")
                detections = []

            if self.result_queue.full():
                try:
                    self.result_queue.get_nowait()
                except queue.Empty:
                    pass

            try:
                self.result_queue.put((detections, frame_shape))
            except queue.Full:
                pass

    def _get_boundary_data(self, cones, z_targets):
        """Интерполяция X-координат с фильтрацией выбросов и сортировкой по Z"""
        valid_cones = [c for c in cones if abs(c[0]) < 2.5]
        if not valid_cones:
            return None, 999.0, -1.0

        valid_cones.sort(key=lambda c: c[1])

        z_vals = []
        x_vals = []
        last_z = None

        for c in valid_cones:
            z = c[1]
            x = c[0]
            if last_z is None or z > last_z + 1e-3:
                z_vals.append(z)
                x_vals.append(x)
                last_z = z

        if not z_vals:
            return None, 999.0, -1.0

        min_z, max_z = z_vals[0], z_vals[-1]

        if len(z_vals) == 1:
            bound_x = np.full_like(z_targets, x_vals[0], dtype=float)
        else:
            bound_x = np.interp(
                z_targets,
                z_vals,
                x_vals,
                left=x_vals[0],
                right=x_vals[-1]
            )

        return bound_x, min_z, max_z

    @staticmethod
    def _dedup_detections(dets, min_dist=25):
        """Убирает дубликаты одного конуса (одинаковый класс, центр рядом)."""
        kept = []
        for det in sorted(dets, key=lambda d: -float(d.get('conf', 0.0))):
            cx, cy = det.get('center', (0, 0))
            dup = False
            for k in kept:
                if k.get('name') == det.get('name'):
                    kx, ky = k.get('center', (10**6, 10**6))
                    if (cx - kx) ** 2 + (cy - ky) ** 2 < min_dist * min_dist:
                        dup = True
                        break
            if not dup:
                kept.append(det)
        return kept

    def _rec_loop(self):
        """
        Поток записи с выравниванием по реальным таймстампам.

        ГЛАВНОЕ ПРАВИЛО: дубликаты для заполнения дыр пишем ТОЛЬКО тогда,
        когда очередь ПУСТАЯ (есть свободная пропускная способность).
        Если в очереди ждут реальные кадры — дубли не пишем, прыгаем по
        времени. Иначе дубликаты воруют пропускную способность и возникает
        спираль: очередь полна -> дыры -> дубли -> очередь ещё полнее.
        """
        video_writer = None
        temp_video_path = None
        final_video_path = None

        fps = float(self.config.zed_fps)
        frame_period = 1.0 / fps
        max_fill = int(fps * 0.5)  # максимум 0.5 с заморозки при пустой очереди

        first_ts = None
        last_ts = None
        written = 0        # сколько кадров должно быть по времени
        real_written = 0   # сколько реальных кадров записано
        filled = 0         # дубликатов заполнения
        jumped = 0         # прыжков по времени (реальные кадры ждали в очереди)
        skipped = 0        # поздних кадров выброшено
        last_frame = None

        while True:
            try:
                item = self.rec_queue.get(timeout=0.5)
            except queue.Empty:
                if not self.running:
                    break
                continue

            if item is None:
                if video_writer is not None:
                    video_writer.release()
                    video_writer = None

                    duration = (last_ts - first_ts) if (first_ts is not None and last_ts is not None) else 0.0

                    logger.info(
                        f"Запись завершена: реальных кадров={real_written}, "
                        f"длительность={duration:.2f}s, "
                        f"дубликатов заполнения={filled}, "
                        f"прыжков={jumped}, "
                        f"поздних выброшено={skipped}"
                    )

                    threading.Thread(
                        target=self._convert_video,
                        args=(temp_video_path, final_video_path)
                    ).start()

                if not self.running:
                    break

                continue

            if not self.running and video_writer is None:
                break

            frame, frame_ts, file_ts = item

            if video_writer is None:
                temp_video_path = os.path.join(
                    self.config.output_folder,
                    f"temp_{file_ts}.{self.config.temp_extension}"
                )
                final_video_path = os.path.join(
                    self.config.output_folder,
                    f"{self.config.output_prefix}_{file_ts}.{self.config.output_extension}"
                )

                h, w = frame.shape[:2]
                fourcc = cv2.VideoWriter_fourcc(*self.config.temp_codec)

                video_writer = cv2.VideoWriter(
                    temp_video_path,
                    fourcc,
                    fps,
                    (w, h)
                )

                if not video_writer.isOpened():
                    logger.error("Не удалось открыть VideoWriter для записи.")
                    video_writer = None
                    continue

                first_ts = frame_ts
                last_ts = frame_ts
                written = 0
                real_written = 0
                filled = 0
                jumped = 0
                skipped = 0
                last_frame = None

            # Первый кадр записи
            if last_frame is None:
                video_writer.write(frame)
                last_frame = frame
                written = 1
                real_written = 1
                continue

            # В какой временной слот должен попасть этот кадр
            target_index = int(round((frame_ts - first_ts) * fps))

            # Кадр опоздал / лишний — выбрасываем
            if target_index < written:
                skipped += 1
                continue

            gap = target_index - written

            if gap > 0:
                if self.rec_queue.empty():
                    # Писатель свободен (камера/визен затупили) —
                    # заполняем дыру дубликатами, чтобы сохранить тайминг.
                    fill = min(gap, max_fill)

                    for _ in range(fill):
                        video_writer.write(last_frame)

                    filled += fill
                    written += fill

                    if gap > fill:
                        written += gap - fill
                        jumped += gap - fill
                else:
                    # В очереди ждут РЕАЛЬНЫЕ кадры. Дубли не пишем —
                    # вся пропускная способность на полезные кадры.
                    written += gap
                    jumped += gap

            video_writer.write(frame)
            last_frame = frame
            written += 1
            real_written += 1
            last_ts = frame_ts

        if video_writer is not None:
            video_writer.release()

    def _convert_video(self, input_path, output_path):
        """Простой транскод без изменения таймингов — тайминги уже ровные."""
        try:
            cmd = [
                'ffmpeg',
                '-i', input_path,
                '-c:v', self.config.output_codec,
                '-preset', self.config.output_preset,
                '-crf', str(self.config.output_crf),
                '-pix_fmt', self.config.output_pix_fmt,
                '-movflags', '+faststart',
                '-y',
                output_path
            ]

            subprocess.run(cmd, check=True, capture_output=True, text=True)

            logger.info(f"Видео сконвертировано: {output_path}")

            os.remove(input_path)

            self.robot_state['msg'] = "ВИДЕО СОХРАНЕНО!"
            self.robot_state['msg_time'] = time.time()

        except subprocess.CalledProcessError as e:
            logger.error(f"Ошибка конвертации: {e}")
            if e.stderr:
                logger.error(f"ffmpeg stderr: {e.stderr[-2000:]}")
        except Exception as e:
            logger.error(f"Ошибка конвертации: {e}")

    def _vision_loop(self):
        init_params = sl.InitParameters()

        if FORCE_HD720_60:
            init_params.camera_resolution = sl.RESOLUTION.HD720
            init_params.camera_fps = 60
        else:
            init_params.camera_resolution = getattr(
                sl.RESOLUTION,
                self.config.zed_resolution,
                sl.RESOLUTION.HD720
            )
            init_params.camera_fps = int(self.config.zed_fps)

        init_params.coordinate_units = getattr(
            sl.UNIT,
            self.config.coordinate_units,
            sl.UNIT.METER
        )

        # глубина не используется — отключаем полностью
        init_params.depth_mode = sl.DEPTH_MODE.NONE

        if self.zed.open(init_params) != sl.ERROR_CODE.SUCCESS:
            logger.error("Не удалось открыть ZED-камеру.")
            self.robot_state['cam_connected'] = False
            self.running = False
            return

        self.robot_state['cam_connected'] = True

        cam_info = self.zed.get_camera_information()
        self.fx = cam_info.camera_configuration.calibration_parameters.left_cam.fx
        self.cx_cam = cam_info.camera_configuration.calibration_parameters.left_cam.cx

        runtime_params = sl.RuntimeParameters()

        try:
            runtime_params.enable_depth = False
        except Exception:
            pass

        image_zed = sl.Mat()

        fps_counter = 0
        current_fps = 0.0
        fps_last_time = time.time()

        rec_file_timestamp = None
        was_recording = False

        grab_error_count = 0
        max_grab_errors = 5

        logger.info("Автопилот запущен: Быстрая реакция + Без памяти.")
        logger.info(f"CUDA pipeline enabled: {USE_CUDA}")
        logger.info(f"Web disabled: {DISABLE_WEB}")

        while self.running:
            try:
                grab_result = self.zed.grab(runtime_params)

                if grab_result != sl.ERROR_CODE.SUCCESS:
                    grab_error_count += 1

                    if grab_error_count >= max_grab_errors:
                        logger.error("Ошибка захвата кадра. Переподключение камеры...")
                        self.robot_state['cam_connected'] = False
                        self._reconnect_camera(init_params)
                        grab_error_count = 0
                        continue
                    else:
                        logger.warning(
                            f"ZED grab failed: {grab_result}, count={grab_error_count}"
                        )

                    time.sleep(0.1)
                    continue

                grab_error_count = 0

            except Exception as e:
                logger.error(f"Исключение при grab(): {e}")
                self.robot_state['cam_connected'] = False
                time.sleep(0.5)
                continue

            self.frame_counter += 1

            should_process = (self.frame_counter % self.process_every) == 0
            need_publish = (self.frame_counter % self.publish_every) == 0

            if self.zed.retrieve_image(image_zed, sl.VIEW.LEFT) != sl.ERROR_CODE.SUCCESS:
                continue

            img_data = image_zed.get_data()
            if img_data is None:
                logger.warning("image_zed.get_data() вернул None, кадр пропущен")
                continue

            detect_frame = None
            detect_frame_shape = None

            # ==========================================================
            # === КОНВЕЙЕР ПОЛУЧЕНИЯ BGR-КАДРА ===
            # ==========================================================
            if USE_CUDA and len(img_data.shape) == 3 and img_data.shape[2] == 4:
                gpu_src = cv2.cuda_GpuMat()
                gpu_src.upload(img_data)

                gpu_bgr = cv2.cuda.cvtColor(gpu_src, cv2.COLOR_BGRA2BGR)

                target_width, target_height = 640, 360
                scale = min(
                    1.0,
                    target_width / img_data.shape[1],
                    target_height / img_data.shape[0]
                )

                if scale < 1.0:
                    new_w = max(256, int(img_data.shape[1] * scale))
                    new_h = max(144, int(img_data.shape[0] * scale))
                    detect_frame_shape = (new_h, new_w)

                    if should_process:
                        gpu_resized = cv2.cuda.resize(gpu_bgr, (new_w, new_h))
                        detect_frame = gpu_resized.download()

                    image_np = gpu_bgr.download()
                else:
                    image_np = gpu_bgr.download()
                    detect_frame_shape = image_np.shape[:2]

                    if should_process:
                        detect_frame = image_np

            else:
                if len(img_data.shape) == 3 and img_data.shape[2] == 4:
                    image_np = cv2.cvtColor(img_data, cv2.COLOR_BGRA2BGR)
                else:
                    # ВАЖНО: всегда своя копия, а не view в буфер SDK.
                    # Иначе SDK перезапишет буфер следующим кадром
                    # прямо во время отрисовки/записи -> рваные кадры.
                    image_np = img_data.copy()

                detect_frame_shape = image_np.shape[:2]

                target_width, target_height = 640, 360

                if image_np.shape[1] > target_width or image_np.shape[0] > target_height:
                    scale = min(
                        1.0,
                        target_width / image_np.shape[1],
                        target_height / image_np.shape[0]
                    )

                    new_w = max(256, int(image_np.shape[1] * scale))
                    new_h = max(144, int(image_np.shape[0] * scale))

                    detect_frame_shape = (new_h, new_w)

                    if should_process:
                        detect_frame = cv2.resize(
                            image_np,
                            (new_w, new_h),
                            interpolation=cv2.INTER_AREA
                        )
                else:
                    if should_process:
                        detect_frame = image_np

            # ==========================================================
            # === 1. АСИНХРОННАЯ ДЕТЕКЦИЯ ===
            # ==========================================================
            while True:
                try:
                    fresh_detections, fresh_shape = self.result_queue.get_nowait()
                    self.last_detections = fresh_detections
                    self.last_detect_frame_shape = fresh_shape
                    self.last_detection_time = time.monotonic()
                except queue.Empty:
                    break
                except Exception:
                    break

            detections = self.last_detections

            if should_process and detect_frame is not None:
                if not self.detect_queue.full():
                    try:
                        self.detect_queue.put_nowait(detect_frame)
                    except queue.Full:
                        pass

            # ==========================================================
            # === 2. МАСШТАБИРОВАНИЕ КООРДИНАТ + ДЕДУПЛИКАЦИЯ ===
            # ==========================================================
            active_detections = detections

            detection_source_shape = (
                self.last_detect_frame_shape
                if self.last_detect_frame_shape is not None
                else detect_frame_shape
            )

            if (
                len(detections) > 0 and
                detection_source_shape is not None and
                detection_source_shape != image_np.shape[:2]
            ):
                scale_x = image_np.shape[1] / float(detection_source_shape[1])
                scale_y = image_np.shape[0] / float(detection_source_shape[0])

                active_detections = []

                for det in detections:
                    scaled_det = det.copy()

                    x1, y1, x2, y2 = det['bbox']

                    scaled_det['bbox'] = (
                        int(x1 * scale_x),
                        int(y1 * scale_y),
                        int(x2 * scale_x),
                        int(y2 * scale_y),
                    )

                    center = det.get('center')
                    if center is not None:
                        scaled_det['center'] = (
                            int(center[0] * scale_x),
                            int(center[1] * scale_y)
                        )

                    active_detections.append(scaled_det)

            active_detections = self._dedup_detections(active_detections)

            # ==========================================================
            # === 3. ОБРАБОТКА И ОТРИСОВКА ===
            # ==========================================================
            current_time = time.time()

            current_cones = []

            for det in active_detections:
                x1, y1, x2, y2 = det['bbox']

                width = max(x2 - x1, 1)
                height = max(y2 - y1, 1)
                area = width * height

                z = self.config.area_depth_constant / math.sqrt(area)

                if self.config.min_depth < z <= self.config.max_depth:
                    u, v = det['center']
                    x_cam = (u - self.cx_cam) * z / self.fx

                    cone_pos = (x_cam, z - self.config.camera_offset_z)

                    current_cones.append({
                        'name': det.get('name', ''),
                        'pos_3d': cone_pos
                    })

                    if self.config.draw_target_z:
                        cv2.putText(
                            image_np,
                            f"Z:{z:.1f}m",
                            (x1, y1 - 25),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            self.config.z_text_scale,
                            self.config.z_text_color,
                            self.config.z_text_thickness
                        )

            blues = sorted(
                [c['pos_3d'] for c in current_cones if c['name'] in self.config.blue_cones],
                key=lambda p: p[1]
            )[:6]

            yellows = sorted(
                [c['pos_3d'] for c in current_cones if c['name'] in self.config.yellow_cones],
                key=lambda p: p[1]
            )[:6]

            orange_cones = [c for c in current_cones if c['name'] in self.config.orange_cones]

            # Интерполяция траектории
            centerline = []
            half_track = self.config.track_width / 2.0
            z_grid = np.arange(0.3, self.config.max_depth, 0.2)

            left_bound_x, l_min_z, l_max_z = self._get_boundary_data(blues, z_grid)
            right_bound_x, r_min_z, r_max_z = self._get_boundary_data(yellows, z_grid)

            for i, z in enumerate(z_grid):
                lx = left_bound_x[i] if left_bound_x is not None else None
                rx = right_bound_x[i] if right_bound_x is not None else None

                valid_l = lx is not None and (l_min_z - 0.4 <= z <= l_max_z + 0.4)
                valid_r = rx is not None and (r_min_z - 0.4 <= z <= r_max_z + 0.4)

                if valid_l and valid_r:
                    cx = (lx + rx) / 2.0
                elif valid_l:
                    cx = lx + half_track
                elif valid_r:
                    cx = rx - half_track
                else:
                    if lx is not None and rx is not None:
                        cx = (lx + rx) / 2.0
                    elif lx is not None:
                        cx = lx + half_track
                    elif rx is not None:
                        cx = rx - half_track
                    else:
                        cx = 0.0

                centerline.append((cx, z))

            waypoints_3d = [
                {'x': cx, 'z': cz, 'type': 'centerline'}
                for cx, cz in centerline
            ]

            # Выбор цели (Lookahead)
            lookahead_dist = self.config.lookahead_distance
            target_wp = None

            for cx, cz in centerline:
                if cz >= lookahead_dist:
                    target_wp = (cx, cz)
                    break

            if target_wp is None and len(centerline) > 0:
                target_wp = centerline[-1]

            # EMA сглаживание целевой точки
            if target_wp is not None:
                tx, tz = target_wp
                alpha = getattr(self.config, 'ema_alpha', 0.85)

                self.smooth_tx = self.smooth_tx + alpha * (tx - self.smooth_tx)
                self.smooth_tz = self.smooth_tz + alpha * (tz - self.smooth_tz)
            else:
                decay = getattr(self.config, 'error_decay_rate', 0.5)
                self.smooth_tx *= decay

            target_x = self.smooth_tx
            target_z = self.smooth_tz

            # Проверка стоп-конуса
            target_detected = False

            if orange_cones:
                stop_threshold = getattr(self.config, 'stop_cone_z_threshold', 0.5)
                if any(oc['pos_3d'][1] <= stop_threshold for oc in orange_cones):
                    target_detected = True

            # Отрисовка траектории — КАЖДЫЙ кадр
            if self.config.draw_trajectory:
                pts_2d = [[image_np.shape[1] // 2, image_np.shape[0]]]

                for wp in waypoints_3d:
                    u = int((wp['x'] * self.fx / wp['z']) + self.cx_cam)
                    v = int(image_np.shape[0] * self.config.cone_base_v)
                    pts_2d.append([u, v])

                if len(pts_2d) > 1:
                    pts_arr = np.array(pts_2d, np.int32).reshape((-1, 1, 2))
                    cv2.polylines(
                        image_np,
                        [pts_arr],
                        isClosed=False,
                        color=self.config.trajectory_color,
                        thickness=self.config.trajectory_thickness
                    )

            # Отрисовка цели
            if target_x is not None and target_z > 0:
                if self.config.draw_target:
                    target_u = int((target_x * self.fx / target_z) + self.cx_cam)
                    target_v = int(image_np.shape[0] * self.config.cone_base_v)

                    cv2.drawMarker(
                        image_np,
                        (target_u, target_v),
                        (0, 0, 255),
                        cv2.MARKER_CROSS,
                        self.config.target_cross_size,
                        self.config.target_cross_thickness
                    )

            # Отрисовка конусов — КАЖДЫЙ кадр
            if self.config.draw_detections:
                for det in active_detections:
                    x1, y1, x2, y2 = det['bbox']
                    cone_name = det.get('name', '')

                    if cone_name in self.config.blue_cones:
                        color = (255, 0, 0)
                    elif cone_name in self.config.yellow_cones:
                        color = (0, 255, 255)
                    elif cone_name in self.config.orange_cones:
                        color = (0, 165, 255)
                    else:
                        color = (255, 255, 255)

                    cv2.rectangle(image_np, (x1, y1), (x2, y2), color, 2)

                    cv2.putText(
                        image_np,
                        cone_name,
                        (x1, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.5,
                        color,
                        2
                    )

            # Управление и ПИД-регулятор
            if self.robot_state.get('auto_mode', False):
                if target_detected:
                    self.robot_state['auto_mode'] = False
                    self.robot_state['msg'] = "ФИНИШ! ОРАНЖЕВЫЙ КОНУС."
                    self.robot_state['msg_time'] = time.time()

                    def _brake(car=self.car):
                        try:
                            car.stop()
                        except Exception as e:
                            logger.error(f"Ошибка торможения: {e}")

                    threading.Thread(target=_brake, daemon=True).start()

                    self.pid_integral = 0.0
                    self.pid_last_error = 0.0

                elif target_x is not None and target_z > 0:
                    error = math.atan2(target_x, target_z)

                    dt = current_time - self.last_pid_time
                    if dt <= 0.0:
                        dt = 0.03

                    self.pid_integral += error * dt

                    max_i = self.config.max_integral
                    self.pid_integral = max(-max_i, min(max_i, self.pid_integral))

                    derivative = (error - self.pid_last_error) / dt

                    steering = (
                        self.config.kp_gain * error +
                        self.config.ki_gain * self.pid_integral +
                        self.config.kd_gain * derivative
                    )

                    max_s = self.config.max_steering_output
                    steering = max(-max_s, min(max_s, steering))

                    self.pid_last_error = error

                    try:
                        self.car.update(1.0, steering)
                    except Exception as e:
                        logger.error(f"Ошибка car.update(): {e}")

                else:
                    self.pid_integral = 0.0
                    self.pid_last_error = 0.0

                    try:
                        self.car.update(1.0, 0.0)
                    except Exception as e:
                        logger.error(f"Ошибка car.update(): {e}")

                self.last_pid_time = current_time

            # FPS
            fps_counter += 1
            elapsed_fps_time = time.time() - fps_last_time

            if elapsed_fps_time >= self.config.fps_update_interval:
                current_fps = fps_counter / max(elapsed_fps_time, 1e-6)
                fps_counter = 0
                fps_last_time = time.time()

            if self.config.draw_fps:
                mode_txt = 'AUTO' if self.robot_state.get('auto_mode') else 'MANUAL'

                cv2.putText(
                    image_np,
                    f"FPS: {current_fps:.1f} Mode: {mode_txt}",
                    (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    self.config.fps_text_scale,
                    self.config.fps_text_color,
                    self.config.fps_text_thickness
                )

            if target_x is not None and self.config.draw_target_z:
                cv2.putText(
                    image_np,
                    f"Target Z: {target_z:.2f}m",
                    (10, 85),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    self.config.target_z_text_scale,
                    self.config.target_z_text_color,
                    self.config.target_z_text_thickness
                )

            # ==========================================================
            # === ЗАПИСЬ ===
            # ==========================================================
            if self.is_recording:
                if self.config.draw_rec:
                    cv2.putText(
                        image_np,
                        "REC",
                        (10, 55),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        self.config.rec_text_scale,
                        self.config.rec_text_color,
                        self.config.rec_text_thickness
                    )

                if not was_recording:
                    rec_file_timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                    was_recording = True
                    self.rec_dropped_frames = 0

                frame_ts = time.monotonic()

                if self.rec_queue.full():
                    self.rec_dropped_frames += 1
                    if self.rec_dropped_frames % 30 == 1:
                        logger.warning(
                            f"rec_queue переполнена, пропущено кадров записи: "
                            f"{self.rec_dropped_frames}"
                        )
                else:
                    try:
                        self.rec_queue.put_nowait(
                            (image_np.copy(), frame_ts, rec_file_timestamp)
                        )
                    except queue.Full:
                        self.rec_dropped_frames += 1
                        if self.rec_dropped_frames % 30 == 1:
                            logger.warning(
                                f"rec_queue Full при put_nowait, пропущено кадров: "
                                f"{self.rec_dropped_frames}"
                            )

            else:
                if was_recording:
                    try:
                        self.rec_queue.put_nowait(None)
                    except queue.Full:
                        try:
                            self.rec_queue.get_nowait()
                            self.rec_queue.put_nowait(None)
                        except Exception:
                            logger.warning(
                                "rec_queue полна при отправке сигнала остановки записи."
                            )

                    was_recording = False

            # Отправка на Web
            if need_publish and not DISABLE_WEB:
                try:
                    set_frame(image_np)
                except Exception as e:
                    logger.error(f"Ошибка Web set_frame: {e}")

        self.zed.close()
        self.robot_state['cam_connected'] = False

    def _reconnect_camera(self, init_params):
        """Попытка переподключения к камере"""
        try:
            self.zed.close()
            time.sleep(0.5)
        except Exception:
            pass

        try:
            self.zed = sl.Camera()

            if self.zed.open(init_params) == sl.ERROR_CODE.SUCCESS:
                self.robot_state['cam_connected'] = True

                cam_info = self.zed.get_camera_information()
                self.fx = cam_info.camera_configuration.calibration_parameters.left_cam.fx
                self.cx_cam = cam_info.camera_configuration.calibration_parameters.left_cam.cx
            else:
                self.robot_state['cam_connected'] = False

        except Exception as e:
            logger.error(f"Ошибка переподключения камеры: {e}")
            self.robot_state['cam_connected'] = False

    def close(self):
        self.running = False
        self.robot_state['cam_connected'] = False

        if getattr(self, 'vision_thread', None) is not None and self.vision_thread.is_alive():
            self.vision_thread.join(timeout=self.config.vision_thread_join_timeout)

        try:
            if getattr(self, 'zed', None) is not None:
                self.zed.close()
        except Exception:
            pass

        if getattr(self, 'detect_thread', None) is not None and self.detect_thread.is_alive():
            self.detect_thread.join(timeout=1.0)

        if getattr(self, 'rec_thread', None) is not None and self.rec_thread.is_alive():
            try:
                self.rec_queue.put_nowait(None)
            except queue.Full:
                try:
                    self.rec_queue.get_nowait()
                    self.rec_queue.put_nowait(None)
                except Exception:
                    pass

            self.rec_thread.join(timeout=3.0)

        self.zed = None

    def restart(self):
        self.close()
        time.sleep(0.5)

        self.running = True
        self.is_recording = False
        self.frame_counter = 0

        self.fx = 0
        self.cx_cam = 0

        self.last_detections = []
        self.last_detect_frame_shape = None
        self.last_detection_time = 0.0

        self.rec_dropped_frames = 0

        self.smooth_tx = 0.0
        self.smooth_tz = self.config.lookahead_distance

        self.pid_integral = 0.0
        self.pid_last_error = 0.0
        self.last_pid_time = time.time()

        self.detect_queue = queue.Queue(maxsize=1)
        self.result_queue = queue.Queue(maxsize=1)

        self.detect_thread = threading.Thread(target=self._detect_loop, daemon=True)
        self.detect_thread.start()

        self.zed = sl.Camera()

        rec_queue_size = int(getattr(self.config, "rec_queue_size", 120))
        self.rec_queue = queue.Queue(maxsize=rec_queue_size)

        self.rec_thread = threading.Thread(target=self._rec_loop, daemon=True)
        self.rec_thread.start()

        self.vision_thread = threading.Thread(target=self._vision_loop, daemon=True)
        self.vision_thread.start()


def main():
    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    sock.settimeout(config.socket_timeout)

    try:
        sock.bind((config.udp_ip, config.udp_port))
    except OSError as e:
        logger.error(
            f"Ошибка bind UDP: {e}. "
            f"Проверь, не запущен ли старый server.py. "
            f"Адрес: {config.udp_ip}:{config.udp_port}"
        )
        sys.exit(1)

    if not DISABLE_WEB:
        start()

    detector = ConeDetector(config)
    car = CarController(config)

    robot_state = {
        'auto_mode': False,
        'cam_connected': False,
        'arduino_connected': False,
        'msg': '',
        'msg_time': 0
    }

    robot_state['arduino_connected'] = car.arduino is not None

    loop = VisionLoop(config, detector, car, robot_state)

    running = True
    last_addr = None

    try:
        while running:
            try:
                data, addr = sock.recvfrom(1024)
                last_addr = addr
                command = data.decode('utf-8').strip()

                if command == "Q":
                    running = False
                    break

                elif command == "A":
                    if not robot_state['auto_mode']:
                        robot_state['auto_mode'] = True
                        robot_state['msg'] = ''
                        loop.pid_integral = 0.0
                        loop.pid_last_error = 0.0

                elif command == "S":
                    if robot_state['auto_mode']:
                        robot_state['auto_mode'] = False
                        car.stop()

                elif command == "R":
                    loop.is_recording = True

                elif command == "C":
                    loop.is_recording = False

                elif command == "F":
                    robot_state['auto_mode'] = False
                    robot_state['msg'] = 'ПЕРЕЗАГРУЗКА...'
                    robot_state['msg_time'] = time.time()

                    logger.info("Инициирована перезагрузка системы...")

                    car.close()
                    time.sleep(0.5)

                    loop.close()
                    time.sleep(0.5)

                    car.restart()
                    time.sleep(0.5)

                    robot_state['cam_connected'] = False
                    loop.restart()
                    time.sleep(1.0)

                    robot_state['cam_connected'] = loop.robot_state.get('cam_connected', False)
                    robot_state['arduino_connected'] = car.arduino is not None

                    robot_state['msg'] = 'СИСТЕМА ПЕРЕЗАГРУЖЕНА!'
                    robot_state['msg_time'] = time.time()

                    logger.info("Система успешно перезагружена!")

                elif command.startswith("speed:"):
                    try:
                        fwd, bck = map(int, command[6:].split(','))
                        car.set_speeds(fwd, bck)
                    except Exception:
                        pass

                else:
                    try:
                        speed, steering = map(float, command.split(','))

                        if robot_state['auto_mode'] and (speed != 0.0 or steering != 0.0):
                            robot_state['auto_mode'] = False
                            robot_state['msg_time'] = time.time()

                        if not robot_state['auto_mode']:
                            car.update(speed, steering)

                    except Exception:
                        pass

                if time.time() - robot_state['msg_time'] > config.message_clear_timeout:
                    robot_state['msg'] = ''

                robot_state['arduino_connected'] = car.arduino is not None

                telemetry = {
                    "mode": "AUTO" if robot_state['auto_mode'] else "MANUAL",
                    "rec": loop.is_recording,
                    "cam_connected": robot_state['cam_connected'],
                    "arduino_connected": robot_state['arduino_connected'],
                    "fwd": car.config.forward_speed,
                    "bck": car.config.back_speed,
                    "msg": robot_state['msg']
                }

                if last_addr:
                    sock.sendto(json.dumps(telemetry).encode('utf-8'), last_addr)

            except socket.timeout:
                if not robot_state['auto_mode']:
                    car.check_stop()

    except KeyboardInterrupt:
        pass

    finally:
        loop.close()
        car.close()
        sock.close()


if __name__ == "__main__":
    main()
