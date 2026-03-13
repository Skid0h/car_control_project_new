"""Запускается на Jetson, принимает команды управления с JETSON_IP + UDP_PORT
   Поддерживает:
   - Числовые команды: "скорость,поворот" (например "-1,-1")
   - Команды установки скорости: "speed:100,80" (установить скорость вперед=100, назад=80)
   - Буквенные команды: R (начать запись), C (остановить запись), Q (выход)
"""

import socket
import serial
import serial.tools.list_ports
import time
import logging
import threading
import sys
import os
from datetime import datetime
import cv2
import pyzed.sl as sl
import subprocess

logging.basicConfig(level=logging.INFO, format='%(asctime)s [%(levelname)s] %(message)s')
logger = logging.getLogger(__name__)

UDP_IP = "192.168.137.50"
UDP_PORT = 5005

Baud = 9600

base_speed = 90
forward_speed = 98
back_speed = 82

base_rotation = 90
right_rotation = 0
left_rotation = 180

def find_arduino_port():
    """Автоматически находит порт с Arduino"""
    ports = serial.tools.list_ports.comports()
    
    for port in ports:
        if ('Arduino' in port.description or
            'CH340' in port.description or
            'USB Serial' in port.description):
            logger.info(f"Найден Arduino на порту {port.device} ({port.description})")
            return port.device
    
    # Дополнительная проверка по VID/PID для Arduino
    for port in ports:
        if port.vid and port.pid:
            if (port.vid == 0x2341) or (port.vid == 0x1A86):  # Arduino или CH340
                logger.info(f"Найден Arduino по VID/PID на порту {port.device}")
                return port.device
    
    logger.error("Arduino не найден")
    return None

class VideoRecorder:
    """Класс для управления записью ZED-камеры"""
    
    def __init__(self):
        self.zed = None
        self.video_writer = None
        self.is_recording = False
        self.recording_thread = None
        self.stop_recording_flag = False
        self.output_folder = "zed_recordings"
        
        # Создаем папку для записей
        if not os.path.exists(self.output_folder):
            os.makedirs(self.output_folder)
            logger.info(f"Создана папка для записей: {self.output_folder}")
    
    def convert_to_compatible_format(self, input_path, output_path):
        """Конвертирует видео в формат MP4"""
        try:
            cmd = [
                'ffmpeg', '-i', input_path,
                '-c:v', 'libx264',
                '-preset', 'fast',
                '-crf', '23',
                '-pix_fmt', 'yuv420p',
                '-y', output_path
            ]
            subprocess.run(cmd, check=True, capture_output=True)
            logger.info(f"Видео сконвертировано: {output_path}")
            return True
        except Exception as e:
            logger.error(f"Ошибка конвертации: {e}")
            return False
    
    def start_recording(self):
        """Запуск записи видео"""
        if self.is_recording:
            logger.warning("Запись уже идет")
            return False
        
        logger.info("Запуск записи видео...")
        self.stop_recording_flag = False
        self.recording_thread = threading.Thread(target=self._recording_loop)
        self.recording_thread.start()
        return True
    
    def _recording_loop(self):
        """Основной цикл записи"""
        self.zed = sl.Camera()
        init_params = sl.InitParameters()
        init_params.camera_resolution = sl.RESOLUTION.HD720
        init_params.camera_fps = 30

        if self.zed.open(init_params) != sl.ERROR_CODE.SUCCESS:
            logger.error("Не удалось инициализировать ZED-камеру")
            return

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        temp_video_path = os.path.join(self.output_folder, f"temp_{timestamp}.avi")
        final_video_path = os.path.join(self.output_folder, f"zed_recording_{timestamp}.mp4")
        
        logger.info(f"Запись начата: {final_video_path}")

        runtime_params = sl.RuntimeParameters()
        image_zed = sl.Mat()
        frame_count = 0
        target_fps = 10
        frame_time = 1.0 / target_fps
        last_frame_time = time.time()

        # Получаем первый кадр для инициализации
        if self.zed.grab(runtime_params) == sl.ERROR_CODE.SUCCESS:
            self.zed.retrieve_image(image_zed, sl.VIEW.LEFT)
            image_np = image_zed.get_data()
            
            if image_np.shape[2] == 4:
                image_np = cv2.cvtColor(image_np, cv2.COLOR_BGRA2BGR)
            
            height, width = image_np.shape[:2]
            fourcc = cv2.VideoWriter_fourcc(*'MJPG')
            self.video_writer = cv2.VideoWriter(temp_video_path, fourcc, target_fps, (width, height))
            
            if not self.video_writer.isOpened():
                logger.error("Не удалось открыть VideoWriter")
                self.zed.close()
                return

        self.is_recording = True

        try:
            while not self.stop_recording_flag:
                current_time = time.time()
                
                if current_time - last_frame_time >= frame_time:
                    if self.zed.grab(runtime_params) == sl.ERROR_CODE.SUCCESS:
                        self.zed.retrieve_image(image_zed, sl.VIEW.LEFT)
                        image_np = image_zed.get_data()
                        
                        if image_np.shape[2] == 4:
                            image_np = cv2.cvtColor(image_np, cv2.COLOR_BGRA2BGR)

                        self.video_writer.write(image_np)
                        frame_count += 1
                        
                        if frame_count % 30 == 0:  # Логируем каждые 30 кадров
                            logger.debug(f"Записано кадров: {frame_count}")
                            
                        last_frame_time = current_time
                        
        finally:
            if self.video_writer:
                self.video_writer.release()
                logger.info(f"Временное видео сохранено: {temp_video_path}")
                logger.info(f"Всего кадров: {frame_count}")
                
                # Конвертируем в совместимый формат
                logger.info("Конвертация видео...")
                if self.convert_to_compatible_format(temp_video_path, final_video_path):
                    os.remove(temp_video_path)
                    logger.info(f"Временный файл удален")
                else:
                    logger.error("Ошибка конвертации, временный файл сохранен")
            
            self.zed.close()
            self.is_recording = False
            logger.info("Запись остановлена")
    
    def stop_recording(self):
        """Остановка записи"""
        if not self.is_recording:
            logger.warning("Нет активной записи")
            return False
        
        logger.info("Остановка записи видео...")
        self.stop_recording_flag = True
        if self.recording_thread:
            self.recording_thread.join(timeout=5.0)
        return True

class CarController:
    def __init__(self):
        self.forward_speed = forward_speed
        self.back_speed = back_speed
        self.arduino = None
        self.last_command_time = 0
        
        port = find_arduino_port()
        if port is None:
            logger.error("Arduino не найден, управление движением недоступно")
            self.arduino = None
            return
        
        self.arduino = None
        try:
            self.arduino = serial.Serial(port, Baud, timeout=1)
            time.sleep(2)  # Ждем инициализации Arduino
            self.stop()
            time.sleep(0.5)
            logger.info(f"Arduino успешно подключен к {port}")
        except Exception as e:
            logger.error(f"Ошибка подключения к {port}: {e}")
            self.arduino = None
    
    def set_speeds(self, forward, back):
        """Установка новых значений скорости"""
        self.forward_speed = forward
        self.back_speed = back
        logger.info(f"Скорость установлена: вперед={forward}, назад={back}")
    
    def update(self, speed, steering):
        if not self.arduino:
            logger.warning("Arduino не подключен, команда игнорируется")
            return

        motor_value = base_speed
        if speed > 0:
            motor_value = self.forward_speed
        elif speed < 0:
            motor_value = self.back_speed
        
        steer_value = base_rotation
        if steering < 0:
            steer_value = left_rotation
        elif steering > 0:
            steer_value = right_rotation
        
        command = f"{motor_value},{steer_value}\n"
        try:
            self.arduino.write(command.encode('utf-8'))
            self.last_command_time = time.time()
            logger.debug(f"Команда движения: {command.strip()}")
        except serial.SerialException as e:
            logger.error(f"Ошибка отправки: {e}")
            self.arduino = None
    
    def stop(self):
        if self.arduino:
            try:
                self.arduino.write("90,90\n".encode('utf-8'))
                self.last_command_time = time.time()
                logger.debug("Команда: СТОП")
            except:
                pass
    
    def check_stop(self):
        """Проверка и остановка если давно не было команд"""
        if self.arduino and time.time() - self.last_command_time > 1.0:
            self.stop()
    
    def close(self):
        self.stop()
        time.sleep(0.1)
        if self.arduino:
            self.arduino.close()
        logger.info("Arduino отключен")

def main():
    # Создаем UDP сокет
    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    sock.settimeout(0.5)  # Таймаут для проверки остановки
    sock.bind((UDP_IP, UDP_PORT))
    logger.info(f"Сервер слушает порт {UDP_PORT}")
    
    # Инициализируем контроллеры
    car = CarController()
    recorder = VideoRecorder()
    
    last_command_time = time.time()
    running = True
    
    logger.info("Сервер готов к приему команд:")
    logger.info("  - Движение: 'скорость,поворот' (например '1,0' - вперед, '-1,-1' - назад+влево)")
    logger.info("  - Скорость: 'speed:100,80' - установить скорость вперед=100, назад=80")
    logger.info("  - R - начать запись видео")
    logger.info("  - C - остановить запись видео")
    logger.info("  - Q - выход из программы")
    logger.info(f"Текущая скорость: вперед={car.forward_speed}, назад={car.back_speed}")
    
    try:
        while running:
            try:
                data, addr = sock.recvfrom(1024)
                command = data.decode('utf-8').strip()
                last_command_time = time.time()
                
                logger.info(f"Получена команда: '{command}' от {addr}")
                
                # Обработка команд
                if command == "Q":
                    logger.info("Получена команда Q - выход")
                    running = False
                    break
                    
                elif command == "R":
                    logger.info("Получена команда R - начать запись")
                    if not recorder.is_recording:
                        recorder.start_recording()
                    else:
                        logger.info("Запись уже идет")
                        
                elif command == "C":
                    logger.info("Получена команда C - остановить запись")
                    if recorder.is_recording:
                        recorder.stop_recording()
                    else:
                        logger.info("Нет активной записи")
                
                elif command.startswith("speed:"):
                    # Команда установки скорости: speed:100,80
                    try:
                        speed_str = command[6:]  # Убираем "speed:"
                        forward, back = map(int, speed_str.split(','))
                        car.set_speeds(forward, back)
                    except (ValueError, IndexError) as e:
                        logger.error(f"Ошибка парсинга команды скорости: {e}")
                        
                else:
                    # Пытаемся распарсить как команду движения (speed,steering)
                    try:
                        speed, steering = map(int, command.split(','))
                        car.update(speed, steering)
                        logger.debug(f"Движение: скорость={speed}, поворот={steering}")
                    except ValueError:
                        logger.warning(f"Неизвестная команда: {command}")
                    
            except socket.timeout:
                # Проверяем, не пора ли остановить машину
                car.check_stop()
                    
    except KeyboardInterrupt:
        logger.info("Остановка сервера...")
    finally:
        # Останавливаем запись если она идет
        if recorder.is_recording:
            logger.info("Остановка записи перед выходом...")
            recorder.stop_recording()
        
        # Отключаем Arduino
        car.close()
        sock.close()
        logger.info("Сервер остановлен")

if __name__ == "__main__":
    main()
