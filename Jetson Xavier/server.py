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

from Code.Config_load import Config
from Code.Car_control import CarController
from Code.Cone_detector import ConeDetector
from Code.Web import start, set_frame

logging.basicConfig(level=logging.INFO, format='%(asctime)s [%(levelname)s] %(message)s')
logger = logging.getLogger(__name__)

config = Config("config.jsonc")

start()

# ==========================================
# ПИД-РЕГУЛЯТОР (С ДИНАМИЧЕСКИМ DT)
# ==========================================
class SimplePID:
    def __init__(self, kp=1.5, ki=0.1, kd=0.3):
        self.kp = kp
        self.ki = ki
        self.kd = kd
        self.integral = 0.0
        self.last_error = 0.0

    def compute(self, error, dt):
        # Защита от деления на ноль или слишком больших скачков времени
        if dt <= 0: 
            dt = 0.03
        if dt > 0.5: 
            dt = 0.5
            
        self.integral += error * dt
        self.integral = max(-1.0, min(1.0, self.integral))
        
        derivative = (error - self.last_error) / dt
        output = self.kp * error + self.ki * self.integral + self.kd * derivative
        self.last_error = error
        return max(-1.0, min(1.0, output))

    def reset(self):
        self.integral = 0.0
        self.last_error = 0.0

class VisionLoop:
    def __init__(self, config, detector, car, robot_state):
        self.config = config
        self.detector = detector
        self.car = car
        self.robot_state = robot_state

        self.zed = sl.Camera()
        self.running = True
        self.is_recording = False
       
        if not os.path.exists(self.config.output_folder):
            os.makedirs(self.config.output_folder)
       
        self.fx = 0
        self.cx_cam = 0
        
        kp = getattr(self.config, 'pid_kp', 1.5)
        ki = getattr(self.config, 'pid_ki', 0.1)
        kd = getattr(self.config, 'pid_kd', 0.3)
        self.pid = SimplePID(kp=kp, ki=ki, kd=kd)
        
        self.memory_cones = []
        self.last_error_angle = 0.0
        
        # Переменные для динамического расчета dt
        self.last_frame_time = time.time()
        
        # Переменные для EMA сглаживания целевой точки
        self.smooth_tx = 0.0
        self.smooth_tz = getattr(self.config, 'lookahead_min', 0.3)
        
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
        except Exception as e:
            logger.error(f"Ошибка конвертации: {e}")

    def _get_robust_depth(self, depth_data, u, v, window_size=5):
        h, w = depth_data.shape
        half = window_size // 2
        v_min, v_max = max(0, v - half), min(h, v + half + 1)
        u_min, u_max = max(0, u - half), min(w, u + half + 1)
        roi = depth_data[v_min:v_max, u_min:u_max]
        valid_depths = roi[np.isfinite(roi) & (roi > 0)]
        if len(valid_depths) > 0:
            return float(np.median(valid_depths))
        return -1.0

    def _get_boundary_data(self, cones, z_targets):
        """Интерполяция X-координат с фильтрацией выбросов (Пункт 6)"""
        # Фильтруем дикие выбросы: конусы, которые распознались дальше 2.5м вбок, явно ошибочны
        valid_cones = [c for c in cones if abs(c[0]) < 2.5]
        
        if not valid_cones:
            return None, 999.0, -1.0
            
        z_vals = [c[1] for c in valid_cones]
        x_vals = [c[0] for c in valid_cones]
        min_z, max_z = min(z_vals), max(z_vals)
        
        if len(valid_cones) == 1:
            bound_x = np.full_like(z_targets, x_vals[0])
        else:
            bound_x = np.interp(z_targets, z_vals, x_vals, left=x_vals[0], right=x_vals[-1])
            
        return bound_x, min_z, max_z

    def _vision_loop(self):
        init_params = sl.InitParameters()
        init_params.camera_resolution = getattr(sl.RESOLUTION, self.config.zed_resolution, sl.RESOLUTION.HD720)
        init_params.camera_fps = getattr(self.config, 'zed_fps', 15)
        init_params.coordinate_units = getattr(sl.UNIT, self.config.coordinate_units, sl.UNIT.METER)
       
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
        image_zed = sl.Mat()
        depth_zed = sl.Mat()
        
        fps_counter = 0
        fps_last_time = time.time()
        current_fps = 0
        video_writer = None
        temp_video_path = None
        final_video_path = None

        self.last_frame_time = time.time()

        while self.running:
            if self.zed.grab(runtime_params) == sl.ERROR_CODE.SUCCESS:
                current_time = time.time()
                
                # Считаем честный dt для ПИД-регулятора
                dt = current_time - self.last_frame_time
                self.last_frame_time = current_time
                
                self.zed.retrieve_image(image_zed, sl.VIEW.LEFT)
                self.zed.retrieve_measure(depth_zed, sl.MEASURE.DEPTH)
                
                img_data = image_zed.get_data()
                depth_data = depth_zed.get_data()
                
                image_np = cv2.cvtColor(img_data, cv2.COLOR_BGRA2BGR) if img_data.shape[2] == 4 else img_data
                
                detections = self.detector.detect(image_np)
                
                # Память конусов
                mem_timeout = getattr(self.config, 'memory_timeout', 0.8)
                self.memory_cones = [c for c in self.memory_cones if current_time - c['updated_at'] < mem_timeout]
                
                for det in detections:
                    u, v = det['center']
                    z = self._get_robust_depth(depth_data, u, v)
                    
                    if getattr(self.config, 'min_depth', 0.1) < z <= getattr(self.config, 'max_depth', 3.0):
                        x_cam = (u - self.cx_cam) * z / self.fx
                        cone_pos = (x_cam, z)
                        new_class = det.get('name', '')
                        
                        found = False
                        for mc in self.memory_cones:
                            if mc['name'] == new_class:
                                mx, mz = mc['pos_3d']
                                dist = math.sqrt((x_cam - mx)**2 + (z - mz)**2)
                                if dist < 0.4:
                                    mc['pos_3d'] = cone_pos
                                    mc['updated_at'] = current_time
                                    found = True
                                    break
                        
                        if not found:
                            self.memory_cones.append({'name': new_class, 'pos_3d': cone_pos, 'updated_at': current_time})

                blues = sorted([c['pos_3d'] for c in self.memory_cones if c['name'] in getattr(self.config, 'blue_cones', ['blue'])], key=lambda p: p[1])[:6]
                yellows = sorted([c['pos_3d'] for c in self.memory_cones if c['name'] in getattr(self.config, 'yellow_cones', ['yellow'])], key=lambda p: p[1])[:6]
                orange_cones = [c for c in self.memory_cones if c['name'] in getattr(self.config, 'orange_cones', ['orange'])]

                # Интерполяция с защитой от срезов углов
                centerline = []
                half_track = getattr(self.config, 'track_width', 1.4) / 2.0
                z_grid = np.arange(0.3, getattr(self.config, 'max_depth', 3.0), 0.2)
                
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

                # ==========================================
                # ДИНАМИЧЕСКИЙ LOOKAHEAD (ОТ СКОРОСТИ)
                # ==========================================
                current_pwm = self.robot_state.get('current_pwm', getattr(self.config, 'forward_speed', 1570))
                neutral_pwm = getattr(self.config, 'neutral_speed', 1500)
                max_pwm = getattr(self.config, 'max_speed_pwm', 1600)
                
                # Теперь factor реально зависит от того, что сейчас выставлено в роботе
                speed_factor = max(0.0, min(1.0, (current_pwm - neutral_pwm) / (max_pwm - neutral_pwm + 1e-5)))
                
                lookahead_min = getattr(self.config, 'lookahead_min', 0.3)
                lookahead_max = getattr(self.config, 'lookahead_max', 1.2)
                lookahead_dist = lookahead_min + speed_factor * (lookahead_max - lookahead_min)
                
                target_wp = None
                for cx, cz in centerline:
                    if cz >= lookahead_dist:
                        target_wp = (cx, cz)
                        break
                        
                if target_wp is None and len(centerline) > 0:
                    target_wp = centerline[-1]

                # EMA Сглаживание цели
                if target_wp is not None:
                    tx, tz = target_wp
                    alpha = getattr(self.config, 'ema_alpha', 0.3)
                    self.smooth_tx = self.smooth_tx + alpha * (tx - self.smooth_tx)
                    self.smooth_tz = self.smooth_tz + alpha * (tz - self.smooth_tz)
                    error_angle = math.atan2(self.smooth_tx, self.smooth_tz)
                else:
                    decay = getattr(self.config, 'error_decay_rate', 0.85)
                    self.smooth_tx *= decay
                    error_angle = math.atan2(self.smooth_tx, self.smooth_tz)
                    
                self.last_error_angle = error_angle

                # Управление и ПИД (с передачей честного dt)
                stop_threshold = getattr(self.config, 'stop_cone_z_threshold', 0.4)
                stop_detected = any(oc['pos_3d'][1] <= stop_threshold for oc in orange_cones)
                steering = 0.0

                if self.robot_state.get('auto_mode', False):
                    if stop_detected:
                        self.robot_state['auto_mode'] = False
                        self.robot_state['msg'] = "ФИНИШ! СТОП-КОНУС."
                        self.car.stop()
                    else:
                        steering = self.pid.compute(error_angle, dt=dt)  # Передаем посчитанный dt!
                        self.car.update(1.0, steering)
                else:
                    self.pid.reset()

                # Рисование (Визуализацию не меняем по запросу)
                start_u, start_v = image_np.shape[1] // 2, image_np.shape[0]
                if getattr(self.config, 'draw_trajectory', True) and len(centerline) > 0:
                    pts_2d = [[start_u, start_v]]
                    for cx, cz in centerline:
                        u = int((cx * self.fx / cz) + self.cx_cam)
                        v = int(image_np.shape[0] * getattr(self.config, 'cone_base_v', 0.65))
                        u = max(-5000, min(image_np.shape[1] + 5000, u))
                        pts_2d.append([u, v])
                    if len(pts_2d) > 1:
                        pts_arr = np.array(pts_2d, np.int32).reshape((-1, 1, 2))
                        cv2.polylines(image_np, [pts_arr], False, getattr(self.config, 'trajectory_color', [0,255,0]), 2)

                if getattr(self.config, 'draw_target', True) and self.smooth_tz > 0:
                    tu = int((self.smooth_tx * self.fx / self.smooth_tz) + self.cx_cam)
                    tv = int(image_np.shape[0] * getattr(self.config, 'cone_base_v', 0.65))
                    cv2.drawMarker(image_np, (tu, tv), (0, 0, 255), cv2.MARKER_CROSS, 25, 3)
                    cv2.line(image_np, (start_u, start_v), (tu, tv), (0, 100, 255), 2)

                fps_counter += 1
                if current_time - fps_last_time >= getattr(self.config, 'fps_update_interval', 0.5):
                    current_fps = fps_counter / getattr(self.config, 'fps_update_interval', 0.5)
                    fps_counter = 0
                    fps_last_time = current_time
                
                if getattr(self.config, 'draw_fps', True):
                    status_txt = f"FPS:{current_fps:.1f} | Lookahead:{lookahead_dist:.2f}m | Steer:{steering:.2f}"
                    cv2.putText(image_np, status_txt, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
                
                if self.is_recording:
                    if getattr(self.config, 'draw_rec', True):
                        cv2.putText(image_np, "REC", (10, 55), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                    if video_writer is None:
                        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                        temp_video_path = os.path.join(self.config.output_folder, f"temp_{timestamp}.avi")
                        final_video_path = os.path.join(self.config.output_folder, f"rec_{timestamp}.mp4")
                        height, width = image_np.shape[:2]
                        fourcc = cv2.VideoWriter_fourcc(*'MJPG')
                        video_writer = cv2.VideoWriter(temp_video_path, fourcc, 15, (width, height))
                    video_writer.write(image_np)
                else:
                    if video_writer is not None:
                        video_writer.release()
                        video_writer = None
                        threading.Thread(target=self._convert_video, args=(temp_video_path, final_video_path, 15)).start()

                set_frame(image_np)

        if video_writer is not None:
            video_writer.release()
        self.zed.close()
        self.robot_state['cam_connected'] = False

    def start_recording(self):
        self.is_recording = True

    def stop_recording(self):
        self.is_recording = False

    def close(self):
        self.running = False
        self.vision_thread.join(timeout=3.0)

def main():
    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    sock.settimeout(config.socket_timeout)
    try:
        sock.bind((config.udp_ip, config.udp_port))
    except:
        sys.exit(1)
    
    detector = ConeDetector(config)
    car = CarController(config)
    
    # Инициализируем стартовый PWM в состоянии робота
    start_speed = getattr(config, 'forward_speed', 1570)
    robot_state = {'auto_mode': False, 'cam_connected': False, 'msg': '', 'msg_time': 0, 'current_pwm': start_speed}
    
    loop = VisionLoop(config, detector, car, robot_state)
    running = True
    
    try:
        while running:
            try:
                data, addr = sock.recvfrom(1024)
                command = data.decode('utf-8').strip()
                if command == "Q":
                    running = False
                    break
                elif command == "A":
                    if not robot_state['auto_mode']:
                        robot_state['auto_mode'] = True
                        robot_state['msg'] = ''
                elif command == "S":
                    if robot_state['auto_mode']:
                        robot_state['auto_mode'] = False
                        car.stop()
                elif command == "R":
                    loop.start_recording()
                elif command == "C":
                    loop.stop_recording()
                elif command.startswith("speed:"):
                    try:
                        fwd, bck = map(int, command[6:].split(','))
                        car.set_speeds(fwd, bck)
                        robot_state['current_pwm'] = fwd  # Запоминаем измененную скорость!
                    except:
                        pass
                else:
                    if not robot_state['auto_mode']:
                        try:
                            speed, steering = map(float, command.split(','))
                            car.update(speed, steering)
                        except:
                            pass

                if time.time() - robot_state['msg_time'] > config.message_clear_timeout:
                    robot_state['msg'] = ''
                telemetry = {
                    "mode": "AUTO" if robot_state['auto_mode'] else "MANUAL",
                    "rec": loop.is_recording,
                    "cam_connected": robot_state['cam_connected'],
                    "fwd": robot_state['current_pwm'],
                    "bck": getattr(config, 'back_speed', 1430),
                    "msg": robot_state['msg']
                }
                sock.sendto(json.dumps(telemetry).encode('utf-8'), addr)
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
