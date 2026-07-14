"""
Загрузка конфигурации из JSONC-файла с поддержкой однострочных комментариев (//).
Все параметры вынесены в атрибуты класса для быстрого доступа и автодополнения в IDE.
"""

import os               # Для проверки существования файла и путей
import sys              # Для экстренного завершения программы (sys.exit)
import json             # Для парсинга структуры словарей из текста
import re               # ИСПРАВЛЕНИЕ: Регулярные выражения для удаления комментариев


class Config:
    def __init__(config_self, config_path="config.jsonc"):
        # Проверяем, существует ли файл конфигурации по указанному пути
        if not os.path.exists(config_path):
            print(f"Файл конфигурации '{config_path}' не найден.")
            print(f"Текущая рабочая директория: {os.getcwd()}")
            sys.exit(1) # Завершаем программу с кодом ошибки, если файла нет
        
        # Читаем весь текст из файла конфигурации
        with open(config_path, 'r', encoding='utf-8') as f:
            content = f.read()
        
        # --- ВАЖНОЕ ИСПРАВЛЕНИЕ ---
        # Библиотека json в Python не понимает комментарии вида "//".
        # Эта строчка находит все совпадения от "//" до конца строки и заменяет их на пустоту.
        content = re.sub(r'//.*', '', content)
        
        # Пытаемся превратить очищенный текст в Python-словарь
        try:
            cfg = json.loads(content)
        except json.JSONDecodeError as e:
            # Если где-то забыли запятую или кавычку — покажем точную строку с ошибкой
            print(f"Ошибка парсинга JSON в {config_path}:")
            print(f"Строка {e.lineno}, колонка {e.colno}: {e.msg}")
            sys.exit(1)
        
        # ==========================================
        # БЛОК 1: Сетевые настройки (Network)
        # ==========================================
        # Метод .get(ключ, значение_по_умолчанию) гарантирует, что если ключа нет, программа не упадет
        net = cfg.get('network', {})
        config_self.udp_ip = net.get('udp_ip', '0.0.0.0') # 0.0.0.0 означает "слушать все сетевые интерфейсы"
        config_self.udp_port = net.get('udp_port', 5005)
        
        # ==========================================
        # БЛОК 2: Настройки машинки (Car)
        # ==========================================
        car = cfg.get('car', {})
        config_self.baud_rate = car.get('baud_rate', 9600)             # Скорость общения с Arduino
        config_self.neutral_speed = car.get('neutral_speed', 1500)     # ШИМ остановки мотора
        config_self.forward_speed = car.get('forward_speed', 1570)     # Базовая скорость вперед
        config_self.max_speed_pwm = car.get('max_speed_pwm', 1600)     # Для расчета динамического lookahead
        config_self.back_speed = car.get('back_speed', 1430)           # Скорость заднего хода (реверса)
        config_self.center_steering = car.get('center_steering', 135)  # Центральное положение руля (в градусах)
        config_self.command_interval = car.get('command_interval', 0.1)# Защита от спама одинаковыми командами
        config_self.watchdog_timeout = car.get('watchdog_timeout', 0.4)# Таймаут автоостановки (если пропала связь)
        config_self.steering_range = car.get('steering_range', 90)     # Допустимое отклонение руля (вправо/влево)
        
        # ==========================================
        # БЛОК 3: Автопилот и ПИД-регулятор (Autopilot)
        # ==========================================
        ap = cfg.get('autopilot', {})
        config_self.pid_kp = ap.get('pid_kp', 1.5)             # Резкость поворота к цели
        config_self.pid_ki = ap.get('pid_ki', 0.1)             # Доворот при затяжной ошибке
        config_self.pid_kd = ap.get('pid_kd', 0.3)             # Сопротивление раскачке
        config_self.max_integral = ap.get('max_integral', 1.5)
        config_self.error_decay_rate = ap.get('error_decay_rate', 0.85) # Скорость возврата руля прямо при потере конусов
        
        config_self.lookahead_min = ap.get('lookahead_min', 0.3) # Куда смотреть на низкой скорости
        config_self.lookahead_max = ap.get('lookahead_max', 1.2) # Куда смотреть на высокой скорости
        config_self.max_depth = ap.get('max_depth', 3.0)         # Дальше 3м конусы игнорируем
        config_self.min_depth = ap.get('min_depth', 0.1)         # Ближе 10см конусы игнорируем
        config_self.track_width = ap.get('track_width', 1.4)     # Ширина трассы в метрах
        config_self.memory_timeout = ap.get('memory_timeout', 0.8) # Сколько секунд помнить конус после его исчезновения
        config_self.stop_cone_z_threshold = ap.get('stop_cone_z_threshold', 0.4) # Дистанция торможения перед оранжевым конусом
        
        # Старые ключи для обратной совместимости (чтобы не сломать логику, если они где-то используются)
        config_self.lookahead_distance = ap.get('lookahead_distance', 0.5)
        config_self.pair_z_tolerance = ap.get('pair_z_tolerance', 1.0)
        config_self.pair_x_tolerance_multiplier = ap.get('pair_x_tolerance_multiplier', 1.5)
        config_self.virtual_point_offset = ap.get('virtual_point_offset', 0.7)
        config_self.area_depth_constant = ap.get('area_depth_constant', 150.0)
        
        # ==========================================
        # БЛОК 4: Компьютерное зрение (Vision)
        # ==========================================
        vis = cfg.get('vision', {})
        config_self.yolo_model_path = vis.get('yolo_model_path', '')
        config_self.yolo_img_size = vis.get('yolo_img_size', 640)         # Под какой размер обучена YOLO
        config_self.roi_crop_top = vis.get('roi_crop_top', 0.3)           # Отрезаем 30% неба (ускоряет работу сети)
        config_self.roi_crop_bottom = vis.get('roi_crop_bottom', 1.0) 
        config_self.confidence_threshold = vis.get('confidence_threshold', 0.5) # Порог уверенности нейросети
        config_self.iou_threshold = vis.get('iou_threshold', 0.5)               # Порог склейки накладывающихся рамок
        config_self.target_fps = vis.get('target_fps', 10.0)
        config_self.output_folder = vis.get('output_folder', 'zed_recordings')
        config_self.camera_offset_x = vis.get('camera_offset_x', -0.06)   # Смещение камеры относительно центра машины (в метрах)
        config_self.zed_resolution = vis.get('zed_resolution', 'HD720')
        config_self.zed_fps = vis.get('zed_fps', 15)
        config_self.depth_mode = vis.get('depth_mode', 'PERFORMANCE')     # Режим расчета глубины (быстрый)
        config_self.coordinate_units = vis.get('coordinate_units', 'METER')
        config_self.cone_base_v = vis.get('cone_base_v', 0.65)            # Y-координата (в пикселях), где ожидается пол
        config_self.point_of_view_offset_y = vis.get('point_of_view_offset_y', 0.8) # Смещение центра конуса к его основанию
        config_self.target_cross_size = vis.get('target_cross_size', 25)
        config_self.target_cross_thickness = vis.get('target_cross_thickness', 3)
        
        # ==========================================
        # БЛОК 5: Логика обнаружения (Detection)
        # ==========================================
        det = cfg.get('detection', {})
        config_self.cone_colors = det.get('cone_colors', {})
        config_self.class_names = det.get('class_names', {})              # Важно для связки ID(0,1,2) с именами(yellow, orange, blue)
        config_self.blue_cones = det.get('blue_cones', ['blue'])          # Конусы правой стороны
        config_self.yellow_cones = det.get('yellow_cones', ['yellow'])    # Конусы левой стороны
        config_self.orange_cones = det.get('orange_cones', ['orange'])    # Стоп-конусы
        config_self.circle_marker_radius = det.get('circle_marker_radius', 4)
        config_self.circle_marker_color = det.get('circle_marker_color', [255, 255, 255])
        config_self.text_scale = det.get('text_scale', 0.5)
        config_self.text_thickness = det.get('text_thickness', 2)
        config_self.z_text_scale = det.get('z_text_scale', 0.4)
        config_self.z_text_thickness = det.get('z_text_thickness', 1)
        config_self.z_text_color = det.get('z_text_color', [255, 255, 255])
        
        # ==========================================
        # БЛОК 6: Отрисовка интерфейса (Display)
        # ==========================================
        disp = cfg.get('display', {})
        config_self.draw_detections = disp.get('draw_detections', True)       # Рамки YOLO
        config_self.draw_trajectory = disp.get('draw_trajectory', True)       # Линия траектории
        config_self.draw_target = disp.get('draw_target', True)               # Крестик цели
        config_self.draw_fps = disp.get('draw_fps', True)                     # Текст FPS
        config_self.draw_target_z = disp.get('draw_target_z', True)           # Дистанция до цели
        config_self.draw_rec = disp.get('draw_rec', True)                     # Значок записи
        config_self.fps_text_scale = disp.get('fps_text_scale', 0.7)
        config_self.fps_text_thickness = disp.get('fps_text_thickness', 2)
        config_self.fps_text_color = disp.get('fps_text_color', [0, 255, 0])
        config_self.target_z_text_scale = disp.get('target_z_text_scale', 0.6)
        config_self.target_z_text_thickness = disp.get('target_z_text_thickness', 2)
        config_self.target_z_text_color = disp.get('target_z_text_color', [0, 255, 255])
        config_self.rec_text_scale = disp.get('rec_text_scale', 0.7)
        config_self.rec_text_thickness = disp.get('rec_text_thickness', 2)
        config_self.rec_text_color = disp.get('rec_text_color', [0, 0, 255])
        config_self.trajectory_thickness = disp.get('trajectory_thickness', 2)
        config_self.trajectory_color = disp.get('trajectory_color', [0, 255, 0])
        config_self.waypoint_radius = disp.get('waypoint_radius', 6)
        config_self.waypoint_color_pair = disp.get('waypoint_color_pair', [0, 255, 0])
        config_self.waypoint_color_virtual = disp.get('waypoint_color_virtual', [255, 200, 0])
        config_self.waypoint_color_stop = disp.get('waypoint_color_stop', [0, 0, 255])
        config_self.pair_line_thickness = disp.get('pair_line_thickness', 1)
        config_self.pair_line_color = disp.get('pair_line_color', [0, 255, 255])
        
        # ==========================================
        # БЛОК 7: Настройки видеозаписи (Video)
        # ==========================================
        vid = cfg.get('video', {})
        config_self.temp_codec = vid.get('temp_codec', 'MJPG')       # Быстрый кодек, не грузит CPU при езде
        config_self.output_codec = vid.get('output_codec', 'libx264')# Тяжелый кодек, сжимает видео после остановки записи
        config_self.output_preset = vid.get('output_preset', 'fast')
        config_self.output_crf = vid.get('output_crf', '23')
        config_self.output_pix_fmt = vid.get('output_pix_fmt', 'yuv420p')
        config_self.output_extension = vid.get('output_extension', 'mp4')
        config_self.temp_extension = vid.get('temp_extension', 'avi')
        config_self.output_prefix = vid.get('output_prefix', 'zed_recording')
        config_self.fps_update_interval = vid.get('fps_update_interval', 0.5)
        
        # ==========================================
        # БЛОК 8: Системные тайминги (Timing)
        # ==========================================
        tim = cfg.get('timing', {})
        config_self.vision_thread_join_timeout = tim.get('vision_thread_join_timeout', 3.0)
        config_self.message_clear_timeout = tim.get('message_clear_timeout', 3.0)
        config_self.socket_timeout = tim.get('socket_timeout', 0.2)
        config_self.arduino_init_delay = tim.get('arduino_init_delay', 2.0)     # Ардуино перезагружается при подключении порта, ждем 2 сек
        config_self.arduino_post_stop_delay = tim.get('arduino_post_stop_delay', 0.5)
        config_self.arduino_close_delay = tim.get('arduino_close_delay', 0.1)
    
    def reload(self, config_path="config.jsonc"):
        """Перезагрузка конфига без перезапуска программы (удобно для настройки ПИД на лету)"""
        self.__init__(config_path)
