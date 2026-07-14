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

# Импорты собственных модулей проекта
from Code.Config_load import Config          # Загрузка настроек из jsonc
from Code.Car_control import CarController   # Управление Arduino
from Code.Cone_detector import ConeDetector  # Нейросеть YOLO (TensorRT)
from Code.Web import start, set_frame        # Веб-сервер для стриминга видео

# Настройка логирования: формат вывода сообщений в консоль
logging.basicConfig(level=logging.INFO, format='%(asctime)s [%(levelname)s] %(message)s')
logger = logging.getLogger(__name__)

# Загружаем настройки из конфигурационного файла
config = Config("config.jsonc")

# Запускаем веб-сервер для трансляции видео в браузер (поток запускается внутри Web.py)
start()

# ==========================================
# ПИД-РЕГУЛЯТОР (С ДИНАМИЧЕСКИМ DT)
# ==========================================
# Класс для вычисления угла поворота руля на основе ошибки (отклонения машинки от трассы)
class SimplePID:
    def __init__(self, kp=1.5, ki=0.1, kd=0.3):
        self.kp = kp          # Пропорциональный коэффициент (реакция на текущую ошибку)
        self.ki = ki          # Интегральный коэффициент (реакция на накопленную ошибку)
        self.kd = kd          # Дифференциальный коэффициент (реакция на скорость изменения ошибки, убирает раскачку)
        self.integral = 0.0   # Накопленная ошибка за всё время
        self.last_error = 0.0 # Ошибка на предыдущем шаге

    def compute(self, error, dt):
        """
        Расчет управляющего воздействия на руль.
        error: текущее отклонение (угол)
        dt: время, прошедшее с предыдущего расчета (в секундах)
        """
        # Защита от деления на ноль или слишком больших скачков времени при лагах
        if dt <= 0: 
            dt = 0.03
        if dt > 0.5: 
            dt = 0.5
            
        # Интегральная часть: накапливаем ошибку со временем
        self.integral += error * dt
        # Ограничиваем интеграл, чтобы избежать эффекта "windup" (чтобы руль не "залип" в крайнем положении)
        self.integral = max(-1.0, min(1.0, self.integral))
        
        # Дифференциальная часть: скорость изменения ошибки (тормозит руль, если мы быстро возвращаемся к цели)
        derivative = (error - self.last_error) / dt
        
        # Общая формула ПИД
        output = self.kp * error + self.ki * self.integral + self.kd * derivative
        
        # Запоминаем текущую ошибку для следующего кадра
        self.last_error = error
        
        # Возвращаем значение руля строго в пределах от -1.0 до 1.0
        return max(-1.0, min(1.0, output))

    def reset(self):
        """Сброс внутренних переменных при остановке или переходе в ручной режим"""
        self.integral = 0.0
        self.last_error = 0.0

# ==========================================
# ОСНОВНОЙ ЦИКЛ КОМПЬЮТЕРНОГО ЗРЕНИЯ
# ==========================================
class VisionLoop:
    def __init__(self, config, detector, car, robot_state):
        self.config = config
        self.detector = detector        # Объект нейросети (TensorRT)
        self.car = car                  # Контроллер машинки (отправка команд на Arduino)
        self.robot_state = robot_state  # Общий словарь состояния робота (разделяемый с главным потоком)

        self.zed = sl.Camera()          # Инициализация объекта камеры ZED
        self.running = True             # Флаг работы потока (пока True — цикл крутится)
        self.is_recording = False       # Флаг записи видео
       
        # Создаем папку для сохранения видео, если ее еще нет
        if not os.path.exists(self.config.output_folder):
            os.makedirs(self.config.output_folder)
       
        # Параметры калибровки камеры (фокусное расстояние и оптический центр)
        self.fx = 0
        self.cx_cam = 0
        
        # Инициализация ПИД-регулятора параметрами из конфига
        kp = getattr(self.config, 'pid_kp', 1.5)
        ki = getattr(self.config, 'pid_ki', 0.1)
        kd = getattr(self.config, 'pid_kd', 0.3)
        self.pid = SimplePID(kp=kp, ki=ki, kd=kd)
        
        # Память конусов: хранит конусы между кадрами, чтобы они не пропадали при одиночных пропусках детекции
        self.memory_cones = []
        self.last_error_angle = 0.0
        
        # Переменные для расчета времени (dt) между кадрами
        self.last_frame_time = time.time()
        
        # Переменные для сглаживания целевой точки (Exponential Moving Average)
        self.smooth_tx = 0.0
        self.smooth_tz = getattr(self.config, 'lookahead_min', 0.3)
        
        # Запускаем цикл обработки видео в отдельном потоке (background thread)
        self.vision_thread = threading.Thread(target=self._vision_loop, daemon=True)
        self.vision_thread.start()

    def _convert_video(self, input_path, output_path, fps):
        """Конвертация видео через FFmpeg для сильного сжатия (уменьшения размера файла)"""
        try:
            cmd = ['ffmpeg', '-i', input_path, '-r', str(fps), '-c:v', self.config.output_codec, 
                   '-preset', self.config.output_preset, '-crf', str(self.config.output_crf), 
                   '-pix_fmt', self.config.output_pix_fmt, '-y', output_path]
            subprocess.run(cmd, check=True, capture_output=True)
            logger.info(f"Видео сконвертировано: {output_path}")
            os.remove(input_path) # Удаляем тяжелый сырой (временный) файл
        except Exception as e:
            logger.error(f"Ошибка конвертации: {e}")

    def _get_robust_depth(self, depth_data, u, v, window_size=5):
        """
        Получение надежной глубины (дистанции) из 3D-карты глубины камеры ZED.
        Берет окно пикселей вокруг центра конуса (u,v) и возвращает медиану (отсеивая шумы и нули).
        """
        h, w = depth_data.shape
        half = window_size // 2
        
        # Защита от выхода за границы картинки (чтобы скрипт не упал с ошибкой индекса)
        v_min, v_max = max(0, v - half), min(h, v + half + 1)
        u_min, u_max = max(0, u - half), min(w, u + half + 1)
        
        # Вырезаем область и берем только корректные значения (глубина > 0 и не NaN)
        roi = depth_data[v_min:v_max, u_min:u_max]
        valid_depths = roi[np.isfinite(roi) & (roi > 0)]
        
        if len(valid_depths) > 0:
            return float(np.median(valid_depths))
        return -1.0 # Возвращаем -1, если дистанцию определить не удалось

    def _get_boundary_data(self, cones, z_targets):
        """
        Интерполяция (построение) границы трассы по найденным конусам.
        Возвращает массив X-координат границы для заданных Z-дальностей.
        """
        # Отбрасываем ошибочные срабатывания: конусы дальше 2.5 метров вбок считаем мусором
        valid_cones = [c for c in cones if abs(c[0]) < 2.5]
        
        if not valid_cones:
            return None, 999.0, -1.0
            
        z_vals = [c[1] for c in valid_cones] # Глубины (Z) конусов
        x_vals = [c[0] for c in valid_cones] # Смещения (X) конусов влево/вправо
        min_z, max_z = min(z_vals), max(z_vals)
        
        if len(valid_cones) == 1:
            # Если конус всего один, считаем границу прямой линией, параллельной оси Z
            bound_x = np.full_like(z_targets, x_vals[0])
        else:
            # Если конусов несколько, строим интерполированную кривую по точкам
            bound_x = np.interp(z_targets, z_vals, x_vals, left=x_vals[0], right=x_vals[-1])
            
        return bound_x, min_z, max_z

    def _vision_loop(self):
        """Основной цикл захвата кадров и их обработки (работает нон-стоп в отдельном потоке)"""
        # 1. Настройка и открытие камеры ZED
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
        
        # Получение параметров камеры (необходимо для перевода 2D пикселей в 3D метры и обратно)
        cam_info = self.zed.get_camera_information()
        self.fx = cam_info.camera_configuration.calibration_parameters.left_cam.fx
        self.cx_cam = cam_info.camera_configuration.calibration_parameters.left_cam.cx

        runtime_params = sl.RuntimeParameters()
        image_zed = sl.Mat() # Область памяти для 2D картинки
        depth_zed = sl.Mat() # Область памяти для 3D карты глубины
        
        # Переменные для расчета и вывода FPS (кадров в секунду)
        fps_counter = 0
        fps_last_time = time.time()
        current_fps = 0
        
        # Переменные для записи видео
        video_writer = None
        temp_video_path = None
        final_video_path = None

        self.last_frame_time = time.time()

        # Бесконечный цикл обработки
        while self.running:
            # Если успешно захватили новый кадр из ZED
            if self.zed.grab(runtime_params) == sl.ERROR_CODE.SUCCESS:
                current_time = time.time()
                
                # Расчет реального времени (dt) для ПИД-регулятора
                dt = current_time - self.last_frame_time
                self.last_frame_time = current_time
                
                # Извлекаем картинку и карту глубины
                self.zed.retrieve_image(image_zed, sl.VIEW.LEFT)
                self.zed.retrieve_measure(depth_zed, sl.MEASURE.DEPTH)
                
                img_data = image_zed.get_data()
                depth_data = depth_zed.get_data()
                
                # Конвертируем формат цвета BGRA -> BGR (убираем альфа-канал, как того требует OpenCV)
                image_np = cv2.cvtColor(img_data, cv2.COLOR_BGRA2BGR) if img_data.shape[2] == 4 else img_data
                
                # === ИНФЕРЕНС (ОБНАРУЖЕНИЕ КОНУСОВ) ===
                # Отправляем кадр в нейросеть YOLO
                detections = self.detector.detect(image_np)
                
                # === ПАМЯТЬ ОБЪЕКТОВ (ТРЕКИНГ) ===
                # Удаляем из памяти устаревшие конусы (которых давно не было видно)
                mem_timeout = getattr(self.config, 'memory_timeout', 0.8)
                self.memory_cones = [c for c in self.memory_cones if current_time - c['updated_at'] < mem_timeout]
                
                # Добавляем или обновляем найденные конусы в памяти
                for det in detections:
                    u, v = det['center'] # 2D координаты центра конуса (в пикселях экрана)
                    z = self._get_robust_depth(depth_data, u, v) # Глубина конуса (в метрах от камеры)
                    
                    # Фильтруем слишком близкие и слишком далекие выбросы
                    if getattr(self.config, 'min_depth', 0.1) < z <= getattr(self.config, 'max_depth', 3.0):
                        # Переводим 2D в 3D (вычисляем смещение X относительно центра камеры в метрах)
                        x_cam = (u - self.cx_cam) * z / self.fx
                        cone_pos = (x_cam, z)
                        new_class = det.get('name', '')
                        
                        found = False
                        # Проверяем, есть ли уже этот конус в памяти (сравниваем по имени и 3D-дистанции)
                        for mc in self.memory_cones:
                            if mc['name'] == new_class:
                                mx, mz = mc['pos_3d']
                                # Если расстояние между старой и новой позицией меньше 40 см, считаем, что это один и тот же конус
                                dist = math.sqrt((x_cam - mx)**2 + (z - mz)**2)
                                if dist < 0.4:
                                    mc['pos_3d'] = cone_pos
                                    mc['updated_at'] = current_time
                                    found = True
                                    break
                        
                        # Если конус новый (или старый сместился слишком сильно), добавляем его в память
                        if not found:
                            self.memory_cones.append({'name': new_class, 'pos_3d': cone_pos, 'updated_at': current_time})

                # Сортируем конусы по дальности (Z) и берем 6 ближайших для каждой стороны трассы
                blues = sorted([c['pos_3d'] for c in self.memory_cones if c['name'] in getattr(self.config, 'blue_cones', ['blue'])], key=lambda p: p[1])[:6]
                yellows = sorted([c['pos_3d'] for c in self.memory_cones if c['name'] in getattr(self.config, 'yellow_cones', ['yellow'])], key=lambda p: p[1])[:6]
                orange_cones = [c for c in self.memory_cones if c['name'] in getattr(self.config, 'orange_cones', ['orange'])]

                # === ПОСТРОЕНИЕ ЦЕНТРА ТРАССЫ ===
                centerline = []
                half_track = getattr(self.config, 'track_width', 1.4) / 2.0 # Половина ширины трассы (0.7м)
                
                # Создаем виртуальную сетку дальностей от 0.3м до max_depth (3.0м) с шагом 20см
                z_grid = np.arange(0.3, getattr(self.config, 'max_depth', 3.0), 0.2)
                
                # Получаем X-координаты границ трассы слева и справа с помощью интерполяции
                left_bound_x, l_min_z, l_max_z = self._get_boundary_data(blues, z_grid)
                right_bound_x, r_min_z, r_max_z = self._get_boundary_data(yellows, z_grid)
                
                # Вычисляем центральную линию для каждого метража (z)
                for i, z in enumerate(z_grid):
                    lx = left_bound_x[i] if left_bound_x is not None else None
                    rx = right_bound_x[i] if right_bound_x is not None else None
                    
                    # Проверяем, что граница в этой точке (на глубине Z) опирается на реальные данные, а не уходит в бесконечность
                    valid_l = lx is not None and (l_min_z - 0.4 <= z <= l_max_z + 0.4)
                    valid_r = rx is not None and (r_min_z - 0.4 <= z <= r_max_z + 0.4)
                    
                    if valid_l and valid_r:
                        cx = (lx + rx) / 2.0         # Видим обе границы: центр ровно посередине
                    elif valid_l:
                        cx = lx + half_track         # Видим только левую границу: отступаем 0.7м вправо
                    elif valid_r:
                        cx = rx - half_track         # Видим только правую границу: отступаем 0.7м влево
                    else:
                        # Если вышли за пределы достоверных данных, берем последнюю известную логику
                        if lx is not None and rx is not None:
                            cx = (lx + rx) / 2.0
                        elif lx is not None:
                            cx = lx + half_track
                        elif rx is not None:
                            cx = rx - half_track
                        else:
                            cx = 0.0 # Если конусов нет вообще, центр трассы прямо по курсу
                            
                    centerline.append((cx, z))

                # ==========================================
                # ДИНАМИЧЕСКИЙ LOOKAHEAD (ВЗГЛЯД ВПЕРЕД)
                # ==========================================
                # Смысл: чем быстрее едем, тем дальше вперед нужно смотреть, чтобы успеть повернуть.
                current_pwm = self.robot_state.get('current_pwm', getattr(self.config, 'forward_speed', 1570))
                neutral_pwm = getattr(self.config, 'neutral_speed', 1500)
                max_pwm = getattr(self.config, 'max_speed_pwm', 1600)
                
                # Вычисляем процент текущей скорости от 0.0 до 1.0
                speed_factor = max(0.0, min(1.0, (current_pwm - neutral_pwm) / (max_pwm - neutral_pwm + 1e-5)))
                
                lookahead_min = getattr(self.config, 'lookahead_min', 0.3)
                lookahead_max = getattr(self.config, 'lookahead_max', 1.2)
                
                # Линейно масштабируем дальность взгляда от скорости
                lookahead_dist = lookahead_min + speed_factor * (lookahead_max - lookahead_min)
                
                # Ищем целевую точку (target_wp) на вычисленном расстоянии
                target_wp = None
                for cx, cz in centerline:
                    if cz >= lookahead_dist:
                        target_wp = (cx, cz)
                        break
                        
                if target_wp is None and len(centerline) > 0:
                    target_wp = centerline[-1] # Если трасса короче, чем lookahead, берем самую дальнюю точку

                # === СГЛАЖИВАНИЕ ЦЕЛЕВОЙ ТОЧКИ (EMA) ===
                if target_wp is not None:
                    tx, tz = target_wp
                    alpha = getattr(self.config, 'ema_alpha', 0.3) # Коэффициент сглаживания
                    # Плавно двигаем старую цель к новой, чтобы избежать резких рывков руля
                    self.smooth_tx = self.smooth_tx + alpha * (tx - self.smooth_tx)
                    self.smooth_tz = self.smooth_tz + alpha * (tz - self.smooth_tz)
                    # Вычисляем нужный угол поворота в радианах
                    error_angle = math.atan2(self.smooth_tx, self.smooth_tz)
                else:
                    # Если точек нет (потеряли трассу), плавно возвращаем руль в центр
                    decay = getattr(self.config, 'error_decay_rate', 0.85)
                    self.smooth_tx *= decay
                    error_angle = math.atan2(self.smooth_tx, self.smooth_tz)
                    
                self.last_error_angle = error_angle

                # === УПРАВЛЕНИЕ АВТОМОБИЛЕМ ===
                stop_threshold = getattr(self.config, 'stop_cone_z_threshold', 0.4)
                # Проверяем, есть ли оранжевый стоп-конус ближе дистанции остановки (0.4м)
                stop_detected = any(oc['pos_3d'][1] <= stop_threshold for oc in orange_cones)
                steering = 0.0

                if self.robot_state.get('auto_mode', False):
                    if stop_detected:
                        self.robot_state['auto_mode'] = False
                        self.robot_state['msg'] = "ФИНИШ! СТОП-КОНУС."
                        self.car.stop() # Тормозим!
                    else:
                        # Вычисляем угол руля через ПИД-регулятор (используя честный dt)
                        steering = self.pid.compute(error_angle, dt=dt)
                        # Отправляем газ (1.0 = скорость из конфига) и руль на Arduino
                        self.car.update(1.0, steering)
                else:
                    # В ручном режиме сбрасываем накопленную ошибку ПИД, чтобы при включении автопилота машину не дернуло
                    self.pid.reset()

                # === ОТРИСОВКА ИНТЕРФЕЙСА (ВИЗУАЛИЗАЦИЯ) ===
                start_u, start_v = image_np.shape[1] // 2, image_np.shape[0] # Координаты центра низа экрана
                
                # Рисуем линию траектории (по точкам centerline)
                if getattr(self.config, 'draw_trajectory', True) and len(centerline) > 0:
                    pts_2d = [[start_u, start_v]]
                    for cx, cz in centerline:
                        # Перевод координат из 3D-метров обратно в пиксели
                        u = int((cx * self.fx / cz) + self.cx_cam)
                        v = int(image_np.shape[0] * getattr(self.config, 'cone_base_v', 0.65)) # Примерная высота пола
                        u = max(-5000, min(image_np.shape[1] + 5000, u)) # Защита от вылета за границы экрана
                        pts_2d.append([u, v])
                    if len(pts_2d) > 1:
                        pts_arr = np.array(pts_2d, np.int32).reshape((-1, 1, 2))
                        cv2.polylines(image_np, [pts_arr], False, getattr(self.config, 'trajectory_color', [0,255,0]), 2)

                # Рисуем крестик целевой точки (куда стремится машина в данный момент)
                if getattr(self.config, 'draw_target', True) and self.smooth_tz > 0:
                    tu = int((self.smooth_tx * self.fx / self.smooth_tz) + self.cx_cam)
                    tv = int(image_np.shape[0] * getattr(self.config, 'cone_base_v', 0.65))
                    cv2.drawMarker(image_np, (tu, tv), (0, 0, 255), cv2.MARKER_CROSS, 25, 3)
                    cv2.line(image_np, (start_u, start_v), (tu, tv), (0, 100, 255), 2)

                # Вычисление FPS с интервалом обновления (чтобы цифры не мельтешили каждый кадр)
                fps_counter += 1
                if current_time - fps_last_time >= getattr(self.config, 'fps_update_interval', 0.5):
                    current_fps = fps_counter / getattr(self.config, 'fps_update_interval', 0.5)
                    fps_counter = 0
                    fps_last_time = current_time
                
                # Рисуем текст с телеметрией в левом верхнем углу
                if getattr(self.config, 'draw_fps', True):
                    status_txt = f"FPS:{current_fps:.1f} | Lookahead:{lookahead_dist:.2f}m | Steer:{steering:.2f}"
                    cv2.putText(image_np, status_txt, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
                
                # === ЗАПИСЬ ВИДЕО ===
                if self.is_recording:
                    if getattr(self.config, 'draw_rec', True):
                        cv2.putText(image_np, "REC", (10, 55), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                    
                    # Если запись только началась, создаем объект VideoWriter
                    if video_writer is None:
                        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                        temp_video_path = os.path.join(self.config.output_folder, f"temp_{timestamp}.avi")
                        final_video_path = os.path.join(self.config.output_folder, f"rec_{timestamp}.mp4")
                        height, width = image_np.shape[:2]
                        # Используем быстрый кодек MJPG для временного файла, чтобы не нагружать процессор (jetson/raspberry) в полете
                        fourcc = cv2.VideoWriter_fourcc(*'MJPG')
                        video_writer = cv2.VideoWriter(temp_video_path, fourcc, 15, (width, height))
                    
                    video_writer.write(image_np)
                else:
                    # Если запись выключили, закрываем файл и запускаем фоновую конвертацию в H.264
                    if video_writer is not None:
                        video_writer.release()
                        video_writer = None
                        threading.Thread(target=self._convert_video, args=(temp_video_path, final_video_path, 15)).start()

                # Отправляем готовый кадр на веб-сервер (в файл Web.py) для трансляции в браузер
                set_frame(image_np)

        # При выходе из цикла (выключение программы) закрываем файлы и камеру
        if video_writer is not None:
            video_writer.release()
        self.zed.close()
        self.robot_state['cam_connected'] = False

    def start_recording(self):
        """Включить запись видео (вызывается из главного потока по UDP)"""
        self.is_recording = True

    def stop_recording(self):
        """Выключить запись видео (вызывается из главного потока по UDP)"""
        self.is_recording = False

    def close(self):
        """Остановить поток зрения корректно"""
        self.running = False
        self.vision_thread.join(timeout=3.0) # Ждем завершения потока до 3 секунд

# ==========================================
# ТОЧКА ВХОДА (СЕРВЕРНАЯ ЧАСТЬ)
# ==========================================
def main():
    # Настройка UDP сокета для приема команд (например, с джойстика, смартфона или другого ПК)
    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    sock.settimeout(config.socket_timeout)
    try:
        sock.bind((config.udp_ip, config.udp_port))
    except:
        sys.exit(1)
    
    # Инициализация нейросети (TensorRT) и контроллера Arduino
    detector = ConeDetector(config)
    car = CarController(config)
    
    # Общее состояние робота (словарь, который читает/пишет и этот цикл, и цикл компьютерного зрения)
    start_speed = getattr(config, 'forward_speed', 1570)
    robot_state = {'auto_mode': False, 'cam_connected': False, 'msg': '', 'msg_time': 0, 'current_pwm': start_speed}
    
    # Запуск потока камеры
    loop = VisionLoop(config, detector, car, robot_state)
    running = True
    
    try:
        while running:
            try:
                # Ожидаем команды по UDP
                data, addr = sock.recvfrom(1024)
                command = data.decode('utf-8').strip()
                
                # Обработка входящих команд
                if command == "Q":         # Выход из программы
                    running = False
                    break
                elif command == "A":       # Включить автопилот
                    if not robot_state['auto_mode']:
                        robot_state['auto_mode'] = True
                        robot_state['msg'] = ''
                elif command == "S":       # Остановка / Выключение автопилота
                    if robot_state['auto_mode']:
                        robot_state['auto_mode'] = False
                        car.stop()
                elif command == "R":       # Старт записи видео
                    loop.start_recording()
                elif command == "C":       # Стоп записи видео
                    loop.stop_recording()
                elif command.startswith("speed:"): # Изменение скорости на лету (например "speed:1580,1420")
                    try:
                        fwd, bck = map(int, command[6:].split(','))
                        car.set_speeds(fwd, bck)
                        robot_state['current_pwm'] = fwd  # Обновляем для расчета динамического взгляда
                    except:
                        pass
                else:
                    # Если мы НЕ в авторежиме, обрабатываем прямые команды управления (ручной пульт)
                    # Ожидаемый формат: "speed,steering" (от -1.0 до 1.0)
                    if not robot_state['auto_mode']:
                        try:
                            speed, steering = map(float, command.split(','))
                            car.update(speed, steering)
                        except:
                            pass

                # Очистка старых сообщений интерфейса (например "ФИНИШ! СТОП-КОНУС")
                if time.time() - robot_state['msg_time'] > config.message_clear_timeout:
                    robot_state['msg'] = ''
                    
                # Формируем и отправляем телеметрию обратно клиенту
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
                # Если долго нет команд по UDP, проверяем, не пора ли остановить машинку (Watchdog)
                # Это защита на случай, если мы управляем руками и оборвалась связь
                if not robot_state['auto_mode']:
                    car.check_stop()
    except KeyboardInterrupt:
        pass
    finally:
        # Корректное закрытие всех процессов при завершении программы (Ctrl+C или команда Q)
        loop.close()
        car.close()
        sock.close()

if __name__ == "__main__":
    main()
