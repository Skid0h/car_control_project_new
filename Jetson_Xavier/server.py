"""
Сервер управления машинкой на Jetson Xavier.

В ручном режиме принимает команды по UDP и передаёт их Arduino по UART.
В авторежиме YOLO распознаёт конусы на кадрах ZED в отдельном потоке.
По площади рамок сервер оценивает расстояние до конусов, строит траекторию
по сине-жёлтым парам. Стрим и регулятор руля используют одну траекторию.
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
from Code.Track_path import TrackPath
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
        self.target_available = False
        self.last_steering = 0.0

        self.zed = sl.Camera()
        self.running = True
        self.is_recording = False
        self.frame_counter = 0
        # Камера и обработка работают чаще отрисовки и публикации.
        self.process_every = 1
        self.last_detection_at = 0.0
        self.last_detections = []
        self.track_path = TrackPath(config, self.TARGET_TIMEOUT)
        self.last_target_z = None
       
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
        self.pid_derivative = 0.0
        self.last_steering_command_at = time.monotonic()

    def enable_auto(self):
        with self.control_lock:
            if self.robot_state['autopilot_mode'] == 'RACE':
                self.robot_state['msg'] = 'В разработке...'
                self.robot_state['msg_time'] = time.time()
            elif not self.robot_state['cam_connected']:
                self.robot_state['msg'] = 'АВТОРЕЖИМ НЕДОСТУПЕН: НЕТ КАДРОВ С КАМЕРЫ'
                self.robot_state['msg_time'] = time.time()
            elif not self.robot_state['auto_mode']:
                self.car.stop()
                self._reset_pid()
                self.last_target_at = time.monotonic()
                self.target_available = False
                self.last_steering = 0.0
                self.robot_state['auto_mode'] = True
                self.robot_state['msg'] = ''

    def apply_autopilot_setting(self, command):
        if command.startswith('mode:'):
            profile = command[5:]
            if profile not in ('PWM', 'MANUAL', 'RACE'):
                raise ValueError('Неизвестный профиль автопилота')
            with self.control_lock:
                if profile != self.robot_state['autopilot_mode']:
                    self.robot_state['auto_mode'] = False
                    self.car.stop()
                    self._reset_pid()
                    self.robot_state['autopilot_mode'] = profile
                self.robot_state['msg'] = ''
            return
        target = float(command)
        if not math.isfinite(target) or not 0 <= target <= 10:
            raise ValueError('Целевая скорость должна быть от 0 до 10 км/ч')
        with self.control_lock:
            self.config.auto_speed['target_kmh'] = target
            self.robot_state['target_speed_kmh'] = target
            if target == 0 and self.robot_state['autopilot_mode'] == 'MANUAL':
                self.speed_controller.reset()
                if self.robot_state['auto_mode']:
                    self.car.stop()
            self.robot_state['msg'] = ''

    def _autopilot_pwm(self):
        profile = self.robot_state['autopilot_mode']
        if profile == 'PWM':
            self.speed_controller.pwm = self.config.forward_speed
            return self.config.forward_speed
        if profile == 'MANUAL' and self.robot_state['target_speed_kmh'] > 0:
            speed, sampled_at = self.car.get_speed_sample()
            return self.speed_controller.update(time.monotonic(), speed, sampled_at)
        return self.config.neutral_speed

    def _camera_lost(self):
        with self.control_lock:
            self.robot_state['cam_connected'] = False
            if self.robot_state['auto_mode']:
                self.robot_state['auto_mode'] = False
                self.car.stop()
                self.robot_state['msg'] = 'ПОТЕРЯ КАМЕРЫ: АВТОРЕЖИМ ОТКЛЮЧЁН'
                self.robot_state['msg_time'] = time.time()

    def _stop_for_missing_target(self):
        # Вызывается под control_lock, чтобы регулятор не вернул тягу после остановки.
        if self.target_available:
            self.target_available = False
            self._reset_pid()
        self.car.update(0.0, self.last_steering)

    def update_autopilot_speed(self):
        with self.control_lock:
            if not self.robot_state['auto_mode']:
                return
            if (not self.target_available or
                    time.monotonic() - self.last_target_at >= self.TARGET_TIMEOUT):
                self._stop_for_missing_target()
                return
            try:
                pwm = self._autopilot_pwm()
                self.car.update_pwm(pwm, self.last_steering)
            except TimeoutError as exc:
                self.robot_state['auto_mode'] = False
                self.car.stop()
                self._reset_pid()
                self.robot_state['msg'] = str(exc)
                self.robot_state['msg_time'] = time.time()

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
            detect_h, detect_w = detect_frame.shape[:2]
            for det in detections:
                x1, y1, x2, y2 = det['bbox']
                det['clipped'] = min(x1, y1, detect_w - x2, detect_h - y2) <= 2
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
          * D-часть реагирует на скорость изменения ошибки, сглаженную по времени;
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
        alpha = -math.expm1(-dt / cfg.derivative_smoothing_s)
        self.pid_derivative += alpha * (derivative - self.pid_derivative)
        differential = cfg.kd_gain * self.pid_derivative
        raw_output = cfg.max_steering_output * (proportional + self.pid_integral + differential)

        # Ограничение максимального хода руля и приведение к [-1, 1]
        steer_limit = max(0.0, min(1.0, steer_limit))
        desired = max(-steer_limit, min(steer_limit, raw_output))
        output = self._limit_steering_change(desired)
        pid_logger.debug(
            "PID dt=%.3fs error=%.3f filtered=%.3f P=%.3f I=%.3f D=%.3f desired=%.3f steer=%.3f",
            dt, error, err, proportional, self.pid_integral, differential, desired, output,
        )
        return output

    def _limit_steering_change(self, desired):
        now = time.monotonic()
        command_dt = min(now - self.last_steering_command_at, 0.1)
        self.last_steering_command_at = now
        max_step = self.config.max_steering_rate_deg_s * command_dt / max(1, self.config.steering_range)
        change = max(-max_step, min(max_step, desired - self.last_steering))
        return self.last_steering + change

    def _draw_track_path(self, image):
        points, floor = self.track_path.points, self.track_path.floor
        if points is None or floor is None:
            return
        slope, horizon = floor
        z_cam = points[:, 1] + self.config.camera_offset_z
        valid = z_cam > 0.05
        z_cam = z_cam[valid]
        if len(z_cam) < 2:
            return
        x_cam = points[valid, 0] - self.config.camera_offset_x
        u = self.fx * x_cam / z_cam + self.cx_cam
        v = slope / z_cam + horizon
        pixels = np.rint(np.column_stack((u, v))).astype(np.int32)
        cv2.polylines(image, [pixels], False, self.config.trajectory_color,
                      self.config.trajectory_thickness, cv2.LINE_AA)

    def _draw_overlay(self, image_np, detections):
        for det in detections:
            if self.config.draw_target_z and 'pos_3d' in det:
                x1, y1, _, _ = det['bbox']
                draw_outlined_text(image_np, f"Z_cam:{det['z_cam']:.2f}m", (x1, y1 - 25),
                                   self.config.z_text_scale, self.config.z_text_color, self.config.z_text_thickness)
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
                target_z = self.last_target_z

                # Забираем свежий результат детекции, если детектор успел его посчитать
                fresh_detections = None
                while True:
                    try:
                        detected_at, fresh_detections = self.result_queue.get_nowait()
                    except queue.Empty:
                        break

                if (fresh_detections is not None and
                        time.monotonic() - detected_at < self.TARGET_TIMEOUT):
                    profile = self.robot_state['autopilot_mode']
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
                            
                            if cone_name in self.config.blue_cones and not det['clipped']:
                                blue_cones.append(det)
                            elif cone_name in self.config.yellow_cones and not det['clipped']:
                                yellow_cones.append(det)
                            elif cone_name in self.config.orange_cones:
                                orange_cones.append(det)

                    finish = None
                    target_detected = False
                    if orange_cones:
                        closest_orange = min(orange_cones, key=lambda c: c['pos_3d'][1])
                        distance_from_front = closest_orange['z_cam'] - self.config.camera_to_front_m
                        if distance_from_front < 2.0 and not closest_orange['clipped']:
                            finish = closest_orange
                        if distance_from_front < self.config.stop_cone_z_threshold:
                            target_detected = True

                    # Plan in the coordinate frame of the latest detection.
                    auto = self.robot_state['auto_mode']
                    self.track_path.update(blue_cones, yellow_cones, finish, detected_at,
                                           self.car.get_speed() if auto else 0.,
                                           self.last_steering if auto else 0.)
                    target = self.track_path.target(self.config.lookahead_distance)
                    target_z = target[1] if target is not None else None

                    with self.control_lock:
                        if (self.robot_state['auto_mode'] and
                                self.robot_state['autopilot_mode'] == profile and
                                detected_at >= self.last_target_at and
                                time.monotonic() - detected_at < self.TARGET_TIMEOUT):
                            if target_detected:
                                self.robot_state['auto_mode'] = False
                                self.car.stop()
                                self.robot_state['msg'] = "ФИНИШ! ОРАНЖЕВЫЙ КОНУС."
                                self.robot_state['msg_time'] = time.time()
                            elif (target is not None and
                                  time.monotonic() - self.track_path.observed_at < self.TARGET_TIMEOUT):
                                if detected_at - self.last_target_at >= self.TARGET_TIMEOUT:
                                    self._reset_pid()
                                # Prediction must not extend the timeout of the last valid observation.
                                self.last_target_at = max(self.last_target_at, self.track_path.observed_at)
                                self.target_available = True
                                target_x, target_z = target
                                error = math.atan2(target_x, target_z)
                                steering = self._pid_steering(error, 1.0, detected_at)
                                self.last_steering = steering
                            else:
                                self._stop_for_missing_target()

                    self.last_detections = detections
                    self.last_target_z = target_z

                if time.monotonic() - self.last_detection_at >= self.TARGET_TIMEOUT:
                    detections = []
                    target_z = None
                    self.track_path.reset()

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
                self._draw_overlay(image_np, detections)
                if self.config.draw_trajectory:
                    self._draw_track_path(image_np)

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
                    draw_outlined_text(image_np, mode_text, (rec_x, 55),
                                       self.config.fps_text_scale, self.config.fps_text_color,
                                       self.config.fps_text_thickness)
                    (mode_width, _), _ = cv2.getTextSize(mode_text, cv2.FONT_HERSHEY_SIMPLEX,
                                                        self.config.fps_text_scale,
                                                        self.config.fps_text_thickness)
                    rec_x += mode_width + 18
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
        self.track_path.reset()
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
    robot_state = {'auto_mode': False, 'autopilot_mode': 'PWM',
                   'target_speed_kmh': config.auto_speed['target_kmh'],
                   'cam_connected': False, 'arduino_connected': False, 'msg': '', 'msg_time': 0}
    robot_state['arduino_connected'] = car.arduino is not None
    loop = VisionLoop(config, detector, car, robot_state)
    running = True
    last_addr = None
    
    try:
        while running:
            loop.update_autopilot_speed()
            try:
                data, addr = sock.recvfrom(1024)
                last_addr = addr
                try:
                    command = data.decode('utf-8').strip()
                except UnicodeDecodeError:
                    continue
                if command == "Q":
                    running = False
                    break
                elif command == "A":
                    loop.enable_auto()
                elif command == "S":
                    with loop.control_lock:
                        robot_state['auto_mode'] = False
                        car.stop()
                        loop._reset_pid()
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
                        with loop.control_lock:
                            car.set_speeds(fwd, bck)
                    except (ValueError, TypeError) as e:
                        logger.warning(f"Некорректная команда speed: {command!r} ({e})")
                elif command == 'telemetry':
                    pass
                elif command.startswith('mode:') or ',' not in command:
                    try:
                        loop.apply_autopilot_setting(command)
                    except ValueError as exc:
                        robot_state['msg'] = str(exc)
                        robot_state['msg_time'] = time.time()
                else:
                    try:
                        speed, steering = map(float, command.split(','))
                        if not all(math.isfinite(v) and -1 <= v <= 1 for v in (speed, steering)):
                            raise ValueError('Некорректная команда движения')
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
                    "autopilot_mode": robot_state['autopilot_mode'],
                    "target_speed_kmh": robot_state['target_speed_kmh'],
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
