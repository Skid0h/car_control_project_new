"""
Сервер управления машинкой на Jetson Xavier.

В ручном режиме принимает команды по UDP и передаёт их Arduino по UART.
В авторежиме YOLO распознаёт конусы на кадрах ZED в отдельном потоке.
По площади рамок сервер оценивает расстояние до конусов, строит траекторию
по сине-жёлтым парам или одиночным конусам и управляет рулём через ПИД.
Ближний оранжевый конус завершает заезд.

Настройки загружаются из config.jsonc. Видео доступно через веб-интерфейс
и может записываться на диск.
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
except:
    CUDA_AVAILABLE = False

from Code.Config_load import Config
from Code.Car_control import CarController
from Code.Speed_control import SpeedController
from Code.Cone_detector import ConeDetector
from Code.Web import start, set_frame

logging.basicConfig(level=logging.INFO, format='%(asctime)s [%(levelname)s] %(message)s')
logger = logging.getLogger(__name__)
pid_logger = logging.getLogger(f"{__name__}.pid")

config = Config("config.jsonc")
pid_logger.setLevel(logging.DEBUG if config.pid_debug else logging.INFO)

start()  # Запуск Web

def draw_outlined_text(image, text, origin, scale, color, thickness):
    font = cv2.FONT_HERSHEY_SIMPLEX
    cv2.putText(image, text, origin, font, scale, (0, 0, 0), thickness + 2, cv2.LINE_AA)
    cv2.putText(image, text, origin, font, scale, color, thickness, cv2.LINE_AA)


class VisionLoop:
    TARGET_TIMEOUT = 0.5

    def __init__(self, config, detector, car, robot_state):
        self.config = config
        self.detector = detector
        self.car = car
        self.robot_state = robot_state
        self.control_lock = threading.Lock()
        self.last_target_at = None
        self.last_steering = 0.0

        self.zed = sl.Camera()
        self.running = True
        self.is_recording = False
        self.frame_counter = 0
        # Камера и обработка работают чаще отрисовки и публикации.
        self.process_every = 1
        self.last_detection_at = 0.0
        self.last_detections = []
        self.last_waypoints_3d = []
        self.last_target_x = None
        self.last_target_z = None
        self.last_target_detected = False
       
        if not os.path.exists(self.config.output_folder):
            os.makedirs(self.config.output_folder)
       
        self.fx = 0
        self.cx_cam = 0

        self.rec_queue = queue.Queue(maxsize=30)
        self.rec_thread = threading.Thread(target=self._rec_loop, daemon=True)
        self.rec_thread.start()

        # Детекция YOLO в отдельном потоке: инференс не блокирует захват/публикацию кадров
        self.detect_queue = queue.Queue(maxsize=1)
        self.result_queue = queue.Queue(maxsize=1)
        self.detect_thread = threading.Thread(target=self._detect_loop, daemon=True)
        self.detect_thread.start()

        self.speed_controller = SpeedController(config.auto_speed)
        self._reset_pid()

        self.vision_thread = threading.Thread(target=self._vision_loop, daemon=True)
        self.vision_thread.start()

    def _reset_pid(self):
        self.speed_controller.reset()
        self.pid_integral = 0.0
        self.pid_prev_error = 0.0
        self.pid_prev_time = None
        self.pid_smoothed_error = 0.0

    def enable_auto(self):
        with self.control_lock:
            if not self.robot_state['cam_connected']:
                self.robot_state['msg'] = 'АВТОРЕЖИМ НЕДОСТУПЕН: НЕТ КАДРОВ С КАМЕРЫ'
                self.robot_state['msg_time'] = time.time()
            elif not self.robot_state['auto_mode']:
                self.car.stop()
                self._reset_pid()
                self.last_target_at = time.monotonic()
                self.last_steering = 0.0
                self.robot_state['auto_mode'] = True
                self.robot_state['msg'] = ''

    def _camera_lost(self):
        with self.control_lock:
            self.robot_state['cam_connected'] = False
            if self.robot_state['auto_mode']:
                self.robot_state['auto_mode'] = False
                self.car.stop()
                self.robot_state['msg'] = 'ПОТЕРЯ КАМЕРЫ: АВТОРЕЖИМ ОТКЛЮЧЁН'
                self.robot_state['msg_time'] = time.time()

    def check_target_timeout(self):
        with self.control_lock:
            if (self.robot_state['auto_mode'] and
                    time.monotonic() - self.last_target_at >= self.TARGET_TIMEOUT):
                if self.pid_prev_time is not None:
                    self._reset_pid()
                self.car.update(0.0, self.last_steering)

    def _detect_loop(self):
        """Детекция YOLO в отдельном потоке.

        Основной цикл кладёт кадр в detect_queue и НЕ ждёт результата:
        инференс больше не ограничивает FPS захвата и веб-стрима.
        В result_queue всегда лежит только самый свежий результат.
        """
        while self.running:
            try:
                task = self.detect_queue.get(timeout=0.5)
            except queue.Empty:
                continue
            if task is None:
                break

            captured_at, detect_frame, full_size, pre_crop_h, crop_top = task
            try:
                detections = self.detector.detect(detect_frame)
            except Exception as e:
                logger.warning(f"Ошибка детекции: {e}")
                detections = []

            # Пересчёт координат обратно к полному кадру image_np
            # detect_frame был уменьшен (resize) и обрезан (crop_top)
            if detect_frame.shape[1] != full_size[0] or detect_frame.shape[0] != full_size[1]:
                scale_x = full_size[0] / detect_frame.shape[1]
                scale_y = full_size[1] / pre_crop_h
                crop_top_full = int(crop_top * scale_y)
                for det in detections:
                    x1, y1, x2, y2 = det['bbox']
                    det['bbox'] = (
                        int(x1 * scale_x),
                        int(y1 * scale_y) + crop_top_full,
                        int(x2 * scale_x),
                        int(y2 * scale_y) + crop_top_full,
                    )
                    center = det.get('center')
                    if center is not None:
                        det['center'] = (int(center[0] * scale_x), int(center[1] * scale_y) + crop_top_full)

            # В очереди результатов — только последний результат, старые выбрасываем
            try:
                self.result_queue.get_nowait()
            except queue.Empty:
                pass
            try:
                self.result_queue.put_nowait((captured_at, detections))
            except queue.Full:
                pass

    def _rec_loop(self):
        video_writer = None
        temp_video_path = None
        final_video_path = None
        rec_frame_count = 0
        rec_start_time = None

        while self.running:
            try:
                item = self.rec_queue.get(timeout=0.5)
            except queue.Empty:
                continue

            if item is None:
                if video_writer is not None:
                    video_writer.release()
                    video_writer = None
                    rec_duration = time.time() - rec_start_time
                    real_fps = rec_frame_count / rec_duration if rec_duration > 0 else float(self.config.fps)
                    threading.Thread(target=self._convert_video, args=(temp_video_path, final_video_path, real_fps)).start()
                continue

            frame, timestamp = item
            if video_writer is None:
                temp_video_path = os.path.join(self.config.output_folder, f"temp_{timestamp}.{self.config.temp_extension}")
                final_video_path = os.path.join(self.config.output_folder, f"{self.config.output_prefix}_{timestamp}.{self.config.output_extension}")
                h, w = frame.shape[:2]
                fourcc = cv2.VideoWriter_fourcc(*self.config.temp_codec)
                video_writer = cv2.VideoWriter(temp_video_path, fourcc, float(self.config.fps), (w, h))
                rec_frame_count = 0
                rec_start_time = time.time()
            video_writer.write(frame)
            rec_frame_count += 1

            # Проверка записи: если кадр не удалось записать (например, диск заполнен),
            # отправляем предупреждение в телеметрию и обрываем запись
            if not video_writer.isOpened():
                self.robot_state['msg'] = "НЕДОСТАТОЧНО МЕСТА: запись прервана"
                self.robot_state['msg_time'] = time.time()
                logger.warning("Запись прервана: не удалось записать кадр (заполнен диск)")
                break

        if video_writer is not None:
            video_writer.release()

    def _convert_video(self, input_path, output_path, fps):
        try:
            cmd = ['ffmpeg', '-i', input_path, '-r', str(fps), '-c:v', self.config.output_codec, 
                   '-preset', self.config.output_preset, '-crf', str(self.config.output_crf), 
                   '-pix_fmt', self.config.output_pix_fmt, '-y', output_path]
            subprocess.run(cmd, check=True, capture_output=True)
            logger.info(f"Видео сконвертировано: {output_path}")
            os.remove(input_path)
            self.robot_state['msg'] = "ВИДЕО СОХРАНЕНО!"
            self.robot_state['msg_time'] = time.time()
        except Exception as e:
            logger.error(f"Ошибка конвертации: {e}")

    def _pid_steering(self, error, steer_limit, captured_at):
        """ПИД-регулятор руля с EMA-сглаживанием ошибки и анти-windup.

        В отличие от простого P (steering = error * k), здесь:
          * I-часть убирает систематическое смещение (машину тянет к краю из-за кривого руля);
          * D-часть демпфирует резкие движения руля;
          * EMA-фильтр сглаживает шум от детекции конусов;
        Все коэффициенты берутся из блока 'pid' в config.jsonc.
        """
        cfg = self.config
        # EMA-сглаживание ошибки
        self.pid_smoothed_error = cfg.ema_alpha * error + (1.0 - cfg.ema_alpha) * self.pid_smoothed_error
        err = self.pid_smoothed_error

        # Производная: d(err)/dt
        if self.pid_prev_time is not None:
            # Интервал между кадрами, а не между получением результатов YOLO.
            dt = max(captured_at - self.pid_prev_time, cfg.min_dt)
            derivative = (err - self.pid_prev_error) / dt
        else:
            dt = 0.0
            derivative = 0.0
        self.pid_prev_time = captured_at
        self.pid_prev_error = err

        # Интеграл с анти-windup (клиппинг по max_integral)
        self.pid_integral += cfg.ki_gain * err * dt
        self.pid_integral = max(-cfg.max_integral, min(cfg.max_integral, self.pid_integral))

        # Выход ПИД
        proportional = cfg.kp_gain * err
        differential = cfg.kd_gain * derivative
        raw_output = cfg.max_steering_output * (proportional + self.pid_integral + differential)

        # Ограничение максимального хода руля и приведение к [-1, 1]
        steer_limit = max(0.0, min(1.0, steer_limit))
        output = max(-steer_limit, min(steer_limit, raw_output))
        pid_logger.debug(
            "PID dt=%.3fs error=%.3f filtered=%.3f P=%.3f I=%.3f D=%.3f raw=%.3f steer=%.3f",
            dt, error, err, proportional, self.pid_integral, differential, raw_output, output,
        )
        return output

    def _select_lookahead_target(self, waypoints_3d, lookahead):
        """Интерполяция цели по продольной дистанции между точками пути."""
        if not waypoints_3d:
            return None
        
        # Адаптируем дистанцию упреждения под скорость (если pure_pursuit_alpha > 0)
        if hasattr(self.config, 'pure_pursuit_alpha') and self.config.pure_pursuit_alpha > 0:
            # Предполагаем, что скорость пропорциональна forward_speed (условно)
            speed_factor = self.config.forward_speed / 1570.0  # Нормализация к базовой скорости
            adjusted_lookahead = lookahead * (1.0 + self.config.pure_pursuit_alpha * (speed_factor - 1.0))
        else:
            adjusted_lookahead = lookahead
        
        if adjusted_lookahead <= waypoints_3d[0]['z']:
            return waypoints_3d[0]
        for near, far in zip(waypoints_3d, waypoints_3d[1:]):
            if near['z'] < adjusted_lookahead <= far['z']:
                weight = (adjusted_lookahead - near['z']) / (far['z'] - near['z'])
                near_z_cam = near['z'] + self.config.camera_offset_z
                far_z_cam = far['z'] + self.config.camera_offset_z
                target_z_cam = adjusted_lookahead + self.config.camera_offset_z
                # Интерполируем высоту в пространстве и проецируем обратно в кадр.
                image_v = ((1 - weight) * near['image_v'] * near_z_cam +
                           weight * far['image_v'] * far_z_cam) / target_z_cam
                return {
                    'x': near['x'] + weight * (far['x'] - near['x']),
                    'z': adjusted_lookahead,
                    'image_v': image_v,
                    'steer_k': min(near.get('steer_k', 2.0), far.get('steer_k', 2.0)),
                }
        return waypoints_3d[-1]

    def _draw_overlay(self, image_np, detections, waypoints_3d, target_x, target_z):
        for det in detections:
            if self.config.draw_target_z and 'pos_3d' in det:
                x1, y1, _, _ = det['bbox']
                draw_outlined_text(image_np, f"Z_cam:{det['z_cam']:.2f}m", (x1, y1 - 25),
                                   self.config.z_text_scale, self.config.z_text_color, self.config.z_text_thickness)
        if self.config.draw_trajectory:
            pts_2d = [[image_np.shape[1]//2, image_np.shape[0]]]
            for wp in sorted(waypoints_3d, key=lambda point: point['z']):
                # Защита от деления на ноль: точки с z <= 0 не проецируем
                if wp['z'] <= 0:
                    continue
                x_cam = wp['x'] - self.config.camera_offset_x
                z_cam = wp['z'] + self.config.camera_offset_z
                u = int((x_cam * self.fx / z_cam) + self.cx_cam)
                v = int(wp['image_v'])
                pts_2d.append([u, v])
            if len(pts_2d) > 1:
                pts_arr = np.array(pts_2d, np.int32).reshape((-1, 1, 2))
                cv2.polylines(image_np, [pts_arr], isClosed=False, 
                             color=self.config.trajectory_color, 
                             thickness=self.config.trajectory_thickness)

        # Отрисовка цели только если z корректен для проекции
        if self.config.draw_target and target_z is not None and target_z > 0:
            x_cam = target_x - self.config.camera_offset_x
            z_cam = target_z + self.config.camera_offset_z
            target_u = int((x_cam * self.fx / z_cam) + self.cx_cam)
            target_wp = self._select_lookahead_target(waypoints_3d, self.config.lookahead_distance)
            target_v = int(target_wp['image_v'])
            cv2.drawMarker(image_np, (target_u, target_v), (0, 0, 255), 
                          cv2.MARKER_CROSS, 
                          self.config.target_cross_size, 
                          self.config.target_cross_thickness)

        # ОТРИСОВКА КОНУСОВ (draw_detections) 
        if self.config.draw_detections:
            for det in detections:
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
                draw_outlined_text(image_np, cone_name, (x1, y1 - 10), 0.5, color, 2)

        # ЛИНИИ МЕЖДУ ПАРАМИ 
        if self.config.pair_line_color:
            for wp in waypoints_3d:
                if wp.get('type') == 'pair':
                    cv2.line(image_np, wp['b_cone']['center'], wp['y_cone']['center'], 
                            self.config.pair_line_color, 
                            self.config.pair_line_thickness)


    def _vision_loop(self):
        try:
            self._capture_loop()
        except Exception:
            logger.exception('Поток камеры завершился с ошибкой')
        finally:
            self._camera_lost()

    def _capture_loop(self):
        init_params = sl.InitParameters()
        init_params.camera_resolution = getattr(sl.RESOLUTION, self.config.zed_resolution, sl.RESOLUTION.HD720)
        init_params.camera_fps = self.config.zed_fps
        init_params.coordinate_units = getattr(sl.UNIT, self.config.coordinate_units, sl.UNIT.METER)
        # NONE отключает вычисление неиспользуемой карты глубины.
        init_params.depth_mode = getattr(sl.DEPTH_MODE, self.config.depth_mode, sl.DEPTH_MODE.PERFORMANCE)
       
        if self.zed.open(init_params) != sl.ERROR_CODE.SUCCESS:
            logger.error("Не удалось открыть ZED-камеру.")
            self.running = False
            return

        cam_info = self.zed.get_camera_information()
        self.fx = cam_info.camera_configuration.calibration_parameters.left_cam.fx
        self.cx_cam = cam_info.camera_configuration.calibration_parameters.left_cam.cx

        runtime_params = sl.RuntimeParameters()
        runtime_params.enable_depth = self.config.depth_mode != 'NONE'
        image_zed = sl.Mat()

        fps_counter = 0
        stream_counter = 0
        current_fps = 0
        current_stream_fps = 0.0
        publish_phase = 0
        fps_last_time = time.monotonic()
        rec_timestamp = None
        was_recording = False

        grab_error_count = 0
        max_grab_errors = 5

        logger.info(f"Обычный автопилот: Оценка глубины по площади + Блокировка виртуальных точек.")
        logger.info(f"FPS: камера/цикл={init_params.camera_fps}, отрисовка/стрим={self.config.fps}, детекция в отдельном потоке")

        while self.running:
            # Частоту захвата задаёт камера, отдельная задержка здесь не нужна.
            try:
                grab_result = self.zed.grab(runtime_params)
                
                if grab_result != sl.ERROR_CODE.SUCCESS:
                    self._camera_lost()
                    grab_error_count += 1
                    
                    if grab_error_count >= max_grab_errors:
                        logger.error("Ошибка захвата кадра. Переподключение камеры...")
                        self._reconnect_camera(init_params)
                        grab_error_count = 0
                        continue
                    
                    time.sleep(0.1)
                    continue
                
                grab_error_count = 0
            except Exception as e:
                logger.error(f"Исключение при grab(): {e}")
                self._camera_lost()
                time.sleep(0.5)
                continue

            if grab_result == sl.ERROR_CODE.SUCCESS:
                captured_at = time.monotonic()
                self.zed.retrieve_image(image_zed, sl.VIEW.LEFT)
                
                img_data = image_zed.get_data()
                gpu_bgr = None
                if CUDA_AVAILABLE and img_data.shape[2] == 4:
                    gpu_src = cv2.cuda_GpuMat()
                    gpu_src.upload(img_data)
                    gpu_bgr = cv2.cuda.cvtColor(gpu_src, cv2.COLOR_BGRA2BGR)
                    image_np = gpu_bgr.download()
                elif img_data.shape[2] == 4:
                    image_np = cv2.cvtColor(img_data, cv2.COLOR_BGRA2BGR)
                else:
                    image_np = img_data

                with self.control_lock:
                    if not self.running:
                        break
                    self.robot_state['cam_connected'] = True

                detect_frame = image_np
                if image_np.shape[1] > 640 or image_np.shape[0] > 480:
                    target_width = 480
                    target_height = 270
                    scale = min(1.0, target_width / image_np.shape[1], target_height / image_np.shape[0])
                    if scale < 1.0:
                        new_w = max(320, int(image_np.shape[1] * scale))
                        new_h = max(180, int(image_np.shape[0] * scale))
                        if CUDA_AVAILABLE:
                            if gpu_bgr is None:
                                gpu_bgr = cv2.cuda_GpuMat()
                                gpu_bgr.upload(image_np)
                            detect_frame = cv2.cuda.resize(gpu_bgr, (new_w, new_h)).download()
                        else:
                            detect_frame = cv2.resize(image_np, (new_w, new_h), interpolation=cv2.INTER_AREA)

                # Сохраняем размеры до кропа для корректного пересчёта координат
                pre_crop_h = detect_frame.shape[0]
                crop_top = int(pre_crop_h * 0.20)
                crop_bottom = int(pre_crop_h * 0.20)
                if crop_top > 0 or crop_bottom > 0:
                    detect_frame = detect_frame[crop_top:pre_crop_h - crop_bottom, :]

                self.frame_counter += 1
                should_process = (self.frame_counter % self.process_every) == 0

                # Кадр уходит в отдельный поток детекции: инференс YOLO
                # не блокирует захват кадров и публикацию в веб (это и был затык на 8 FPS)
                if should_process:
                    task = (captured_at,
                            detect_frame.copy(),
                            (image_np.shape[1], image_np.shape[0]),
                            pre_crop_h,
                            crop_top)
                    try:
                        self.detect_queue.get_nowait()  # сбрасываем кадр, который детектор не успел взять
                    except queue.Empty:
                        pass
                    try:
                        self.detect_queue.put_nowait(task)
                    except queue.Full:
                        pass

                detections = self.last_detections
                waypoints_3d = self.last_waypoints_3d
                target_x = self.last_target_x
                target_z = self.last_target_z
                target_detected = self.last_target_detected

                # Забираем свежий результат детекции, если детектор успел его посчитать
                fresh_detections = None
                while True:
                    try:
                        detected_at, fresh_detections = self.result_queue.get_nowait()
                    except queue.Empty:
                        break

                if fresh_detections is not None:
                    self.last_detection_at = detected_at
                    detections = fresh_detections
                    blue_cones = []
                    yellow_cones = []
                    orange_cones = []
                    
                    for det in detections:
                        x1, y1, x2, y2 = det['bbox']
                        width = max(x2 - x1, 1)
                        height = max(y2 - y1, 1)
                        area = width * height
                        
                        cone_name = det.get('name', '')
                        z = self.config.cone_distance_from_area(area, cone_name)
                        
                        if self.config.min_depth < z <= self.config.max_depth:
                            det['z_cam'] = z
                            u, v = det['center']
                            x_cam = (u - self.cx_cam) * z / self.fx
                            # Переводим координаты левого объектива в систему машинки.
                            det['pos_3d'] = (x_cam + self.config.camera_offset_x, z - self.config.camera_offset_z)
                            
                            if cone_name in self.config.blue_cones:
                                blue_cones.append(det)
                            elif cone_name in self.config.yellow_cones:
                                yellow_cones.append(det)
                            elif cone_name in self.config.orange_cones:
                                orange_cones.append(det)

                    waypoints_3d = []
                    
                    # Сортируем только конусы с вычисленной 3D-позицией (защита от KeyError)
                    blue_cones = sorted((c for c in blue_cones if 'pos_3d' in c), key=lambda c: c['pos_3d'][1])
                    yellow_cones = sorted((c for c in yellow_cones if 'pos_3d' in c), key=lambda c: c['pos_3d'][1])
                    orange_cones = [c for c in orange_cones if 'pos_3d' in c]
                    
                    used_yellows = set()
                    pairs_found_count = 0
                    
                    for b_cone in blue_cones:
                        best_y = None
                        best_diff = float('inf')
                        b_x, b_z = b_cone['pos_3d']
                        
                        for i, y_cone in enumerate(yellow_cones):
                            if i in used_yellows: continue
                            y_x, y_z = y_cone['pos_3d']
                            
                            z_diff = abs(b_z - y_z)
                            x_dist = abs(b_x - y_x) 
                            
                            if z_diff < self.config.pair_z_tolerance and x_dist < (self.config.track_width * self.config.pair_x_tolerance_multiplier):
                                if z_diff < best_diff:
                                    best_diff = z_diff
                                    best_y = (i, y_cone)
                        
                        if best_y:
                            y_idx, y_cone = best_y
                            used_yellows.add(y_idx)
                            y_x, y_z = y_cone['pos_3d']
                            
                            mid_x = (b_x + y_x) / 2.0
                            mid_z = (b_z + y_z) / 2.0
                            # Проекция середины пары по основаниям bbox с учётом глубины.
                            image_v = (b_cone['bbox'][3] * b_cone['z_cam'] +
                                       y_cone['bbox'][3] * y_cone['z_cam']) / (
                                           b_cone['z_cam'] + y_cone['z_cam'])
                            waypoints_3d.append({'x': mid_x, 'z': mid_z, 'image_v': image_v,
                                                 'type': 'pair', 'b_cone': b_cone, 'y_cone': y_cone})
                            pairs_found_count += 1

                    if pairs_found_count == 0:
                        for b_cone in blue_cones:
                            b_x, b_z = b_cone['pos_3d']
                            waypoints_3d.append({'x': b_x + self.config.virtual_point_offset, 'z': b_z,
                                                 'image_v': b_cone['bbox'][3], 'type': 'virtual_blue',
                                                 'steer_k': self.config.virtual_steer_k})
                            
                        for i, y_cone in enumerate(yellow_cones):
                            if i not in used_yellows:
                                y_x, y_z = y_cone['pos_3d']
                                waypoints_3d.append({'x': y_x - self.config.virtual_point_offset, 'z': y_z,
                                                     'image_v': y_cone['bbox'][3], 'type': 'virtual_yellow',
                                                     'steer_k': self.config.virtual_steer_k})

                    waypoints_3d.sort(key=lambda wp: wp['z'])

                    target_detected = False
                    if orange_cones:
                        closest_orange = min(orange_cones, key=lambda c: c['pos_3d'][1])
                        distance_from_front = closest_orange['z_cam'] - self.config.camera_to_front_m
                        if distance_from_front < 2.0:
                            # На финишном участке ориентируемся только на стоп-конус.
                            o_x, o_z = closest_orange['pos_3d']
                            waypoints_3d = [{'x': o_x, 'z': o_z,
                                             'image_v': closest_orange['bbox'][3], 'type': 'stop'}]
                        if distance_from_front < self.config.stop_cone_z_threshold:
                            target_detected = True

                    target_x, target_z = None, None
                    target_wp = None
                    if len(waypoints_3d) > 0:
                        # Pure Pursuit: выбираем точку на дистанции упреждения
                        target_wp = self._select_lookahead_target(waypoints_3d, self.config.lookahead_distance)
                        if target_wp:
                            target_x = target_wp['x']
                            target_z = target_wp['z']
                        
                    # УПРАВЛЕНИЕ 
                    with self.control_lock:
                        if (self.robot_state['auto_mode'] and
                                detected_at >= self.last_target_at and
                                time.monotonic() - detected_at < self.TARGET_TIMEOUT):
                            if target_detected:
                                self.robot_state['auto_mode'] = False
                                self.car.stop()
                                self.robot_state['msg'] = "ФИНИШ! ОРАНЖЕВЫЙ КОНУС."
                                self.robot_state['msg_time'] = time.time()
                            elif target_x is not None and target_z > 0:
                                if detected_at - self.last_target_at >= self.TARGET_TIMEOUT:
                                    self._reset_pid()
                                self.last_target_at = detected_at
                                error = math.atan2(target_x, target_z)
                                steer_k = target_wp.get('steer_k', 2.0)
                                steering = self._pid_steering(error, steer_k, detected_at)
                                self.last_steering = steering
                                try:
                                    speed, sampled_at = self.car.get_speed_sample()
                                    pwm = self.speed_controller.update(time.monotonic(), speed, sampled_at)
                                    self.car.update_pwm(pwm, steering)
                                except TimeoutError as exc:
                                    self.car.stop()
                                    self._reset_pid()
                                    self.robot_state['auto_mode'] = False
                                    self.robot_state['msg'] = str(exc)
                                    self.robot_state['msg_time'] = time.time()
                            else:
                                self.car.update(0.0, self.last_steering)
                                self._reset_pid()

                    self.last_detections = detections
                    self.last_waypoints_3d = waypoints_3d
                    self.last_target_x = target_x
                    self.last_target_z = target_z
                    self.last_target_detected = target_detected

                if time.monotonic() - self.last_detection_at >= self.TARGET_TIMEOUT:
                    detections, waypoints_3d = [], []
                    target_x, target_z = None, None

                # FPS
                fps_counter += 1
                fps_now = time.monotonic()
                fps_elapsed = fps_now - fps_last_time
                if fps_elapsed > 0 and fps_elapsed >= self.config.fps_update_interval:
                    current_fps = round(fps_counter / fps_elapsed, 1)
                    current_stream_fps = stream_counter / fps_elapsed
                    fps_counter = 0
                    stream_counter = 0
                    fps_last_time = fps_now

                # При 30/15 публикуем каждый второй кадр. Дробные отношения
                # распределяются равномерно без задержки цикла управления.
                publish_phase += self.config.fps
                if publish_phase < self.config.zed_fps:
                    continue
                publish_phase -= self.config.zed_fps
                self._draw_overlay(image_np, detections, waypoints_3d, target_x, target_z)
                
                rec_x = 10
                if self.config.draw_fps:
                    processing_text = f"PROC {current_fps:.1f} / "
                    draw_outlined_text(image_np, processing_text, (10, 30),
                                       self.config.fps_text_scale, (0, 255, 0), self.config.fps_text_thickness)
                    (text_width, _), _ = cv2.getTextSize(processing_text, cv2.FONT_HERSHEY_SIMPLEX,
                                                       self.config.fps_text_scale, self.config.fps_text_thickness)
                    draw_outlined_text(image_np, f"STREAM {current_stream_fps:.1f}", (10 + text_width, 30),
                                       self.config.fps_text_scale, (0, 0, 255), self.config.fps_text_thickness)
                    mode_text = 'AUTO' if self.robot_state['auto_mode'] else 'MANUAL'
                    draw_outlined_text(image_np, mode_text, (10, 55),
                                       self.config.fps_text_scale, self.config.fps_text_color, self.config.fps_text_thickness)
                    (mode_width, _), _ = cv2.getTextSize(mode_text, cv2.FONT_HERSHEY_SIMPLEX,
                                                       self.config.fps_text_scale, self.config.fps_text_thickness)
                    rec_x += mode_width + 12
                if self.config.draw_target_z:
                    display_target_z = target_z if target_z is not None else 0.0
                    draw_outlined_text(image_np, f"Target Z: {display_target_z:.2f}m", (10, 115),
                                       self.config.target_z_text_scale, self.config.target_z_text_color,
                                       self.config.target_z_text_thickness)
                
                if self.config.draw_speed and self.car is not None:
                    speed_text = f'Speed: {self.car.get_speed():.2f} km/h'
                    draw_outlined_text(image_np, speed_text, (10, 85),
                                       self.config.speed_text_scale, self.config.speed_text_color, self.config.speed_text_thickness)
                
                # ЗАПИСЬ
                if self.robot_state['auto_mode']:
                    pwm = self.speed_controller.pwm
                    if pwm is None:
                        pwm = self.config.neutral_speed
                    draw_outlined_text(image_np, f"AUTO PWM: {pwm}", (10, 145),
                                       self.config.speed_text_scale, self.config.speed_text_color,
                                       self.config.speed_text_thickness)

                if self.is_recording:
                    if self.config.draw_rec:
                        draw_outlined_text(image_np, "REC", (rec_x, 55),
                                           self.config.rec_text_scale, self.config.rec_text_color, self.config.rec_text_thickness)
                    if not was_recording:
                        rec_timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                        was_recording = True
                    if self.is_recording and not self.rec_queue.full():
                        self.rec_queue.put((image_np.copy(), rec_timestamp))
                else:
                    if was_recording:
                        self.rec_queue.put(None)
                        was_recording = False

                # ОТПРАВКА НА WEB
                if set_frame(image_np):
                    stream_counter += 1

        self.zed.close()
    
    def _reconnect_camera(self, init_params):
        """Попытка переподключения к камере"""
        try:
            self.zed.close()
            time.sleep(0.5)
        except Exception as e:
            pass
        
        try:
            self.zed = sl.Camera()
            if self.zed.open(init_params) == sl.ERROR_CODE.SUCCESS:
                cam_info = self.zed.get_camera_information()
                self.fx = cam_info.camera_configuration.calibration_parameters.left_cam.fx
                self.cx_cam = cam_info.camera_configuration.calibration_parameters.left_cam.cx
        except Exception as e:
            logger.error(f"Ошибка переподключения камеры: {e}")

    def close(self):
        self.running = False
        self._camera_lost()
        try:
            if getattr(self, 'zed', None) is not None:
                self.zed.close()
        except Exception:
            pass
        if getattr(self, 'vision_thread', None) is not None and self.vision_thread.is_alive():
            self.vision_thread.join(timeout=self.config.vision_thread_join_timeout)
        if getattr(self, 'detect_thread', None) is not None and self.detect_thread.is_alive():
            try:
                self.detect_queue.put_nowait(None)
            except queue.Full:
                pass
            self.detect_thread.join(timeout=2.0)
        if getattr(self, 'rec_thread', None) is not None and self.rec_thread.is_alive():
            try:
                self.rec_queue.put_nowait(None)
            except queue.Full:
                pass  # Поток завершится по running=False после текущей записи.
            self.rec_thread.join(timeout=3.0)
        self.zed = None

    def restart(self):
        """Перезапуск системы с проверкой завершения старых потоков"""
        self.close()
        
        # Ждём завершения потоков с проверкой is_alive()
        max_wait = 5.0  # Максимальное время ожидания в секундах
        start_time = time.time()
        
        while (getattr(self, 'vision_thread', None) is not None and self.vision_thread.is_alive()) or \
              (getattr(self, 'detect_thread', None) is not None and self.detect_thread.is_alive()) or \
              (getattr(self, 'rec_thread', None) is not None and self.rec_thread.is_alive()):
            if time.time() - start_time > max_wait:
                logger.error("Перезапуск отменён: старые потоки ещё не завершились")
                return False
            time.sleep(0.1)
        
        # Дополнительная пауза для стабилизации
        time.sleep(0.5)
        
        # Пересоздаём объекты
        self.running = True
        self.last_detection_at = 0.0
        self.is_recording = False
        self.fx = 0
        self.cx_cam = 0
        self.zed = sl.Camera()
        self.rec_queue = queue.Queue(maxsize=30)
        
        # Создаём новые потоки только если старые завершились
        self.rec_thread = threading.Thread(target=self._rec_loop, daemon=True)
        self.rec_thread.start()
        self.detect_queue = queue.Queue(maxsize=1)
        self.result_queue = queue.Queue(maxsize=1)
        self.detect_thread = threading.Thread(target=self._detect_loop, daemon=True)
        self.detect_thread.start()
        self.vision_thread = threading.Thread(target=self._vision_loop, daemon=True)
        self.vision_thread.start()
        return True


def main():
    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    sock.settimeout(min(config.socket_timeout, 0.02))
    try:
        sock.bind((config.udp_ip, config.udp_port))
    except:
        sys.exit(1)
    
    detector = ConeDetector(config)
    car = CarController(config)
    robot_state = {'auto_mode': False, 'cam_connected': False, 'arduino_connected': False, 'msg': '', 'msg_time': 0}
    robot_state['arduino_connected'] = car.arduino is not None
    loop = VisionLoop(config, detector, car, robot_state)
    running = True
    last_addr = None
    
    try:
        while running:
            loop.check_target_timeout()
            try:
                data, addr = sock.recvfrom(1024)
                last_addr = addr
                command = data.decode('utf-8').strip()
                if command == "Q":
                    running = False
                    break
                elif command == "A":
                    loop.enable_auto()
                elif command == "S":
                    with loop.control_lock:
                        if robot_state['auto_mode']:
                            robot_state['auto_mode'] = False
                            car.stop()
                elif command == "R":
                    loop.is_recording = True
                elif command == "C":
                    loop.is_recording = False
                elif command == "F":
                    # ПЕРЕЗАГРУЗКА: отключение и повторное включение системы
                    with loop.control_lock:
                        robot_state['auto_mode'] = False
                    robot_state['msg'] = 'ПЕРЕЗАГРУЗКА...'
                    robot_state['msg_time'] = time.time()
                    logger.info("Инициирована перезагрузка системы...")
                    
                    # Отключение Arduino
                    car.close()
                    time.sleep(0.5)
                    
                    # Отключение камеры
                    loop.close()
                    time.sleep(0.5)
                    
                    # Повторное включение Arduino
                    car.restart()
                    time.sleep(0.5)
                    
                    # Повторное включение камеры
                    vision_restarted = loop.restart()
                    time.sleep(1.0)
                    
                    # Обновляем статусы подключения
                    robot_state['arduino_connected'] = car.arduino is not None
                    robot_state['msg'] = ('СИСТЕМА ПЕРЕЗАГРУЖЕНА!'
                                          if vision_restarted and robot_state['cam_connected'] and robot_state['arduino_connected']
                                          else 'ОШИБКА ПЕРЕЗАГРУЗКИ: ПРОВЕРЬТЕ ПОДКЛЮЧЕНИЯ И ПОТОКИ')
                    robot_state['msg_time'] = time.time()
                    logger.info(robot_state['msg'])
                elif command.startswith("speed:"):
                    try:
                        parts = command[len("speed:"):].split(',')
                        if len(parts) != 2:
                            raise ValueError(f"ожидалось 2 значения, получено {len(parts)}")
                        fwd, bck = (int(p.strip()) for p in parts)
                        # Контроль диапазона ШИМ (защита от опечаток клиента)
                        if not (config.esc_min <= fwd <= config.esc_max and config.esc_min <= bck <= config.esc_max):
                            raise ValueError(f"ШИМ вне диапазона [{config.esc_min}, {config.esc_max}]")
                        car.set_speeds(fwd, bck)
                    except (ValueError, TypeError) as e:
                        logger.warning(f"Некорректная команда speed: {command!r} ({e})")
                else:
                    try:
                        speed, steering = map(float, command.split(','))
                        with loop.control_lock:
                            if robot_state['auto_mode'] and (speed != 0.0 or steering != 0.0):
                                robot_state['auto_mode'] = False
                                robot_state['msg_time'] = time.time()
                            if not robot_state['auto_mode']:
                                car.update(speed, steering)
                    except:
                        pass

                if time.time() - robot_state['msg_time'] > config.message_clear_timeout:
                    robot_state['msg'] = ''

                # Обновляем статус подключения Arduino
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
                with loop.control_lock:
                    if not robot_state['auto_mode']:
                        car.check_stop()
    except KeyboardInterrupt:
        pass
    finally:
        with loop.control_lock:
            robot_state['auto_mode'] = False
            car.stop()
        try:
            loop.close()
        finally:
            try:
                car.close()
            finally:
                sock.close()


if __name__ == "__main__":
    main()
