"""
Запускается на Jetson.
ОПТИМИЗИРОВАНО ДЛЯ МАКСИМАЛЬНОГО FPS:
1. Режим глубины ZED: PERFORMANCE (не грузит GPU нейросетью).
2. Кэширование массивов (z_grid) и параметров конфига.
3. Асинхронная запись видео и умный ребут.
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
import traceback
import queue
import gc

from Code.Config_load import Config
from Code.Car_control import CarController
from Code.Cone_detector import ConeDetector
from Code.Web import start, set_frame

logging.basicConfig(level=logging.INFO, format='%(asctime)s [%(levelname)s] %(message)s')
logger = logging.getLogger(__name__)

config = Config("config.jsonc")
start()

class VisionLoop:
    def __init__(self, config, detector, car, robot_state):
        self.config = config
        self.detector = detector
        self.car = car
        self.robot_state = robot_state

        self.zed = None
        self.running = True
        self.is_recording = False
        self.vision_thread = None
        self.reconnecting = False

        self.frame_queue = queue.Queue(maxsize=5)
        self.writer_thread = None
        self.writer_running = False

        if not os.path.exists(self.config.output_folder):
            os.makedirs(self.config.output_folder)

        self.fx = 0
        self.cx_cam = 0

        # Кэшируем параметры для горячего цикла (избегаем getattr внутри цикла)
        self.min_depth = config.min_depth
        self.max_depth = config.max_depth
        self.track_width_half = getattr(config, 'track_width', 1.4) / 2.0
        self.lookahead_dist = getattr(config, 'lookahead_distance', 0.6)
        self.stop_z_thresh = config.stop_cone_z_threshold
        self.area_const = config.area_depth_constant
        
        # ОПТИМИЗАЦИЯ: Считаем сетку Z один раз, а не каждый кадр!
        self.z_grid = np.arange(self.min_depth, self.max_depth, 0.2)

        time.sleep(0.5)
        self.vision_thread = threading.Thread(target=self._vision_loop, daemon=True)
        self.vision_thread.start()

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

    def _writer_loop(self, video_writer):
        while self.writer_running or not self.frame_queue.empty():
            try:
                frame = self.frame_queue.get(timeout=0.5)
                if frame is None: break
                video_writer.write(frame)
            except queue.Empty:
                continue
        video_writer.release()

    def reconnect_all(self):
        if self.reconnecting:
            logger.warning("⚠️ Переподключение уже выполняется")
            return False
        self.reconnecting = True

        def _do_reconnect():
            try:
                logger.info("🔁 ===== НАЧАЛО ПЕРЕПОДКЛЮЧЕНИЯ =====")
                self.robot_state['msg'] = "🔄 REBOOT..."
                self.robot_state['msg_time'] = time.time()
                self.robot_state['cam_connected'] = False
                self.robot_state['auto_mode'] = False
                self.is_recording = False

                if self.car is not None:
                    try: self.car.stop()
                    except: pass

                self.running = False
                if self.vision_thread and self.vision_thread.is_alive():
                    self.vision_thread.join(timeout=self.config.vision_thread_join_timeout)
                self.zed = None

                gc.collect()
                time.sleep(1.5)

                try: self.detector = ConeDetector(self.config)
                except Exception as e: logger.error(f"❌ Ошибка детектора: {e}")

                new_car = None
                try:
                    new_car = CarController(self.config)
                    if new_car.arduino is None: logger.error("❌ Arduino не найдена")
                except Exception as e: logger.error(f"❌ Ошибка Arduino: {e}")

                if self.car is not None:
                    try: self.car.close()
                    except: pass
                self.car = new_car

                self.running = True
                self.vision_thread = threading.Thread(target=self._vision_loop, daemon=True)
                self.vision_thread.start()

                for _ in range(30):
                    time.sleep(0.5)
                    if self.robot_state.get('cam_connected', False): break

                self.robot_state['msg'] = "✅ REBOOT OK" if self.robot_state.get('cam_connected') else "⚠️ NO CAM"
                self.robot_state['msg_time'] = time.time()
            except Exception as e:
                logger.error(f"❌ Ошибка ребута: {e}")
            finally:
                self.reconnecting = False

        threading.Thread(target=_do_reconnect, daemon=True).start()
        return True

    def _get_boundary_data(self, cones, z_targets):
        valid_cones = [c for c in cones if abs(c['pos_3d'][0]) < 2.5]
        if not valid_cones:
            return None, 999.0, -1.0

        z_vals = [c['pos_3d'][1] for c in valid_cones]
        x_vals = [c['pos_3d'][0] for c in valid_cones]
        min_z, max_z = min(z_vals), max(z_vals)

        if len(valid_cones) == 1:
            bound_x = np.full_like(z_targets, x_vals[0])
        else:
            bound_x = np.interp(z_targets, z_vals, x_vals, left=x_vals[0], right=x_vals[-1])

        return bound_x, min_z, max_z

    def _vision_loop(self):
        try:
            self.zed = sl.Camera()
            init_params = sl.InitParameters()
            init_params.camera_resolution = sl.RESOLUTION.HD720
            init_params.camera_fps = 15
            init_params.coordinate_units = sl.UNIT.METER
            
            # ==========================================================
            # ОПТИМИЗАЦИЯ 2: Режим глубины из конфига (PERFORMANCE)
            # NEURAL убивает FPS на Jetson, забирая GPU у YOLO
            # ==========================================================
            depth_mode_str = getattr(self.config, 'depth_mode', 'PERFORMANCE').upper()
            depth_mode_map = {
                'PERFORMANCE': sl.DEPTH_MODE.PERFORMANCE,
                'QUALITY': sl.DEPTH_MODE.QUALITY,
                'ULTRA': sl.DEPTH_MODE.ULTRA,
                'NEURAL': sl.DEPTH_MODE.NEURAL
            }
            init_params.depth_mode = depth_mode_map.get(depth_mode_str, sl.DEPTH_MODE.PERFORMANCE)
            logger.info(f"Режим глубины ZED: {depth_mode_str} (Оптимизировано для GPU)")

            error_code = self.zed.open(init_params)
            if error_code != sl.ERROR_CODE.SUCCESS:
                logger.error(f"Не удалось открыть ZED. Код: {error_code}")
                self.robot_state['cam_connected'] = False
                self.running = False
                return

            self.robot_state['cam_connected'] = True
            cam_info = self.zed.get_camera_information()
            self.fx = cam_info.camera_configuration.calibration_parameters.left_cam.fx
            self.cx_cam = cam_info.camera_configuration.calibration_parameters.left_cam.cx

            runtime_params = sl.RuntimeParameters()
            image_zed = sl.Mat()

            fps_counter = 0
            current_fps = 0
            fps_last_time = time.time()
            video_writer = None
            temp_video_path = None
            final_video_path = None

            # Локальные переменные для ускорения цикла (избегаем обращений к self.config)
            draw_traj = self.config.draw_trajectory
            draw_target = self.config.draw_target
            draw_det = self.config.draw_detections
            draw_fps = self.config.draw_fps
            draw_z = self.config.draw_target_z
            draw_rec = self.config.draw_rec
            cone_base_v = self.config.cone_base_v
            
            blue_names = self.config.blue_cones
            yellow_names = self.config.yellow_cones
            orange_names = self.config.orange_cones

            while self.running and self.robot_state.get('cam_connected', False):
                try:
                    if self.zed.grab(runtime_params) == sl.ERROR_CODE.SUCCESS:
                        self.zed.retrieve_image(image_zed, sl.VIEW.LEFT)
                        img_data = image_zed.get_data()
                        image_np = cv2.cvtColor(img_data, cv2.COLOR_BGRA2BGR) if img_data.shape[2] == 4 else img_data

                        detections = self.detector.detect(image_np)

                        blue_cones, yellow_cones, orange_cones = [], [], []
                        h_img, w_img = image_np.shape[:2]

                        for det in detections:
                            x1, y1, x2, y2 = det['bbox']
                            area = max(x2 - x1, 1) * max(y2 - y1, 1)
                            z = self.area_const / math.sqrt(area)

                            if self.min_depth < z <= self.max_depth:
                                u, v = det['center']
                                x_cam = (u - self.cx_cam) * z / self.fx
                                det['pos_3d'] = (x_cam, z)

                                if draw_z:
                                    cv2.putText(image_np, f"Z:{z:.1f}", (x1, y1-25), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255,255,255), 1)

                                name = det.get('name', '')
                                if name in blue_names: blue_cones.append(det)
                                elif name in yellow_names: yellow_cones.append(det)
                                elif name in orange_names: orange_cones.append(det)

                        blue_cones.sort(key=lambda c: c['pos_3d'][1])
                        yellow_cones.sort(key=lambda c: c['pos_3d'][1])

                        # Используем кэшированную сетку z_grid
                        left_bound_x, l_min_z, l_max_z = self._get_boundary_data(blue_cones, self.z_grid)
                        right_bound_x, r_min_z, r_max_z = self._get_boundary_data(yellow_cones, self.z_grid)

                        centerline = []
                        for i, z_val in enumerate(self.z_grid):
                            lx = left_bound_x[i] if left_bound_x is not None else None
                            rx = right_bound_x[i] if right_bound_x is not None else None

                            valid_l = lx is not None and (l_min_z - 0.4 <= z_val <= l_max_z + 0.4)
                            valid_r = rx is not None and (r_min_z - 0.4 <= z_val <= r_max_z + 0.4)

                            if valid_l and valid_r: cx = (lx + rx) / 2.0
                            elif valid_l: cx = lx + self.track_width_half
                            elif valid_r: cx = rx - self.track_width_half
                            else: cx = 0.0
                            centerline.append((cx, z_val))

                        target_wp = None
                        for cx, cz in centerline:
                            if cz >= self.lookahead_dist:
                                target_wp = (cx, cz)
                                break
                        if target_wp is None and centerline: target_wp = centerline[-1]

                        target_x = target_wp[0] if target_wp else None
                        target_z = target_wp[1] if target_wp else None

                        target_detected = False
                        if orange_cones:
                            if min(orange_cones, key=lambda c: c['pos_3d'][1])['pos_3d'][1] < self.stop_z_thresh:
                                target_detected = True

                        # Отрисовка (оптимизирована локальными флагами)
                        if draw_traj and centerline:
                            pts_2d = [[w_img//2, h_img]]
                            for cx, cz in centerline:
                                if cz > 0:
                                    u = int((cx * self.fx / cz) + self.cx_cam)
                                    v = int(h_img * cone_base_v)
                                    pts_2d.append([max(0, min(w_img, u)), v])
                            if len(pts_2d) > 1:
                                cv2.polylines(image_np, [np.array(pts_2d, np.int32)], False, (0, 255, 0), 2)

                        if draw_target and target_x and target_z > 0:
                            tu = int((target_x * self.fx / target_z) + self.cx_cam)
                            tv = int(h_img * cone_base_v)
                            cv2.drawMarker(image_np, (tu, tv), (0, 0, 255), cv2.MARKER_CROSS, 25, 3)

                        if draw_det:
                            for det in detections:
                                x1, y1, x2, y2 = det['bbox']
                                name = det.get('name', '')
                                color = (255,0,0) if name in blue_names else (0,255,255) if name in yellow_names else (0,165,255) if name in orange_names else (255,255,255)
                                cv2.rectangle(image_np, (x1, y1), (x2, y2), color, 2)

                        if self.robot_state.get('auto_mode', False):
                            if target_detected:
                                self.robot_state['auto_mode'] = False
                                self.robot_state['msg'] = "ФИНИШ!"
                                self.robot_state['msg_time'] = time.time()
                                if self.car: self.car.stop()
                            elif target_x and target_z > 0:
                                steering = max(-1.0, min(1.0, math.atan2(target_x, target_z) * 2.0))
                                if self.car: self.car.update(1.0, steering)

                        fps_counter += 1
                        if time.time() - fps_last_time >= 0.5:
                            current_fps = fps_counter
                            fps_counter = 0
                            fps_last_time = time.time()

                        if draw_fps:
                            cv2.putText(image_np, f"FPS: {current_fps}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

                        # Асинхронная запись
                        if self.is_recording:
                            if draw_rec: cv2.putText(image_np, "REC", (10, 55), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                            if video_writer is None:
                                ts = datetime.now().strftime("%Y%m%d_%H%M%S")
                                temp_video_path = os.path.join(self.config.output_folder, f"temp_{ts}.{self.config.temp_extension}")
                                final_video_path = os.path.join(self.config.output_folder, f"rec_{ts}.{self.config.output_extension}")
                                video_writer = cv2.VideoWriter(temp_video_path, cv2.VideoWriter_fourcc(*self.config.temp_codec), self.config.zed_fps, (w_img, h_img))
                                self.writer_running = True
                                self.writer_thread = threading.Thread(target=self._writer_loop, args=(video_writer,), daemon=True)
                                self.writer_thread.start()
                            try: self.frame_queue.put(image_np.copy(), timeout=0.05)
                            except queue.Full: pass
                        else:
                            if video_writer is not None:
                                self.writer_running = False
                                self.frame_queue.put(None)
                                video_writer = None
                                threading.Thread(target=self._convert_video, args=(temp_video_path, final_video_path, self.config.zed_fps), daemon=True).start()

                        set_frame(image_np)
                except Exception as e:
                    logger.error(f"Ошибка итерации: {e}")
                    continue

        except Exception as e:
            logger.error(f"❌ Критическая ошибка vision_loop: {e}")
            self.robot_state['cam_connected'] = False
        finally:
            if self.writer_running:
                self.writer_running = False
                self.frame_queue.put(None)
            if self.zed is not None:
                try: self.zed.close()
                except: pass
            self.robot_state['cam_connected'] = False

    def start_recording(self): self.is_recording = True
    def stop_recording(self): self.is_recording = False
    def close(self):
        self.running = False
        if self.vision_thread: self.vision_thread.join(timeout=self.config.vision_thread_join_timeout)

def main():
    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    sock.settimeout(config.socket_timeout)
    try: sock.bind((config.udp_ip, config.udp_port))
    except: sys.exit(1)

    detector = ConeDetector(config)
    car = CarController(config)
    robot_state = {'auto_mode': False, 'cam_connected': False, 'msg': '', 'msg_time': 0}
    loop = VisionLoop(config, detector, car, robot_state)

    running = True
    try:
        while running:
            try:
                data, addr = sock.recvfrom(1024)
                command = data.decode('utf-8').strip()

                if command == "Q":
                    running = False; break
                elif command == "A":
                    robot_state['auto_mode'] = True; robot_state['msg'] = ''
                elif command == "S":
                    robot_state['auto_mode'] = False
                    if loop.car: loop.car.stop()
                elif command == "R": loop.start_recording()
                elif command == "C": loop.stop_recording()
                elif command == "F":
                    if not loop.reconnect_all(): logger.error("❌ Ребут уже идет")
                elif command.startswith("speed:"):
                    try:
                        fwd, bck = map(int, command[6:].split(','))
                        if loop.car: loop.car.set_speeds(fwd, bck)
                    except: pass
                else:
                    if not robot_state['auto_mode']:
                        try:
                            speed, steering = map(float, command.split(','))
                            if loop.car: loop.car.update(speed, steering)
                        except: pass

                if time.time() - robot_state['msg_time'] > config.message_clear_timeout:
                    robot_state['msg'] = ''

                telemetry = {
                    "mode": "AUTO" if robot_state['auto_mode'] else "MANUAL",
                    "rec": loop.is_recording,
                    "cam_connected": robot_state['cam_connected'],
                    "msg": robot_state['msg']
                }
                sock.sendto(json.dumps(telemetry).encode('utf-8'), addr)

            except socket.timeout:
                if not robot_state['auto_mode'] and loop.car:
                    loop.car.check_stop()

    except KeyboardInterrupt: pass
    finally:
        loop.close()
        if loop.car: loop.car.close()
        sock.close()

if __name__ == "__main__":
    main()
