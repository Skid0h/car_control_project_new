import time
import math
import serial
import threading
import serial.tools.list_ports

#Автопоиск ардуино
def find_arduino_port():
    ports = serial.tools.list_ports.comports()
    for port in ports:
        if ('Arduino' in port.description or 'CH340' in port.description or 'USB Serial' in port.description):
            return port.device
       
    for port in ports:
        if port.vid and port.pid:
            if (port.vid == 0x2341) or (port.vid == 0x1A86):
                print(f"Ардуино найдена на порту: {port.device}")
                return port.device
    print('Arduino not found')
    return None

#Функции управления машинкой
class CarController:
    def __init__(self, config):
        self.config = config
        self.lock = threading.Lock()
        self.last_sent_time    = 0                                      # Время отправки последней команды (для command_interval)
        self.last_command_time = 0                                      # Для watchdog
        self.last_sent_cmd     = ""                                     # Последняя отправленная строка команды
        self.arduino = None
        self.current_speed   = 0.0                                      # Показание скорости от Arduino
        self._speed_sample = (0.0, 0.0)
        self._reader_thread  = None                                     # Поток чтения скорости из UART

        port = find_arduino_port()
        if port is None: return
        
        try:
            self.arduino = serial.Serial(port, self.config.baud_rate, timeout=1)
            time.sleep(self.config.arduino_init_delay)
            self.stop()
            time.sleep(self.config.arduino_post_stop_delay)
            self._start_reader()
        except Exception as e:
            self.arduino = None

    def _start_reader(self):
        """Запуск потока чтения строк вида S:<скорость> от Arduino"""
        self._reader_generation = getattr(self, '_reader_generation', 0) + 1
        gen = self._reader_generation
        self._reader_thread = threading.Thread(target=self._read_loop, args=(gen,), daemon=True)
        self._reader_thread.start()

    def _read_loop(self, generation):
        while True:
            ser = self.arduino
            if ser is None or generation != getattr(self, '_reader_generation', 0):
                return  # соединение закрыто или запущен новый поток чтения
            try:
                line = ser.readline().decode('utf-8', errors='ignore').strip()
            except Exception:
                # Не трогаем self.arduino здесь: при переподключении restart()
                # уже создал новый объект, и старый поток не должен его обнулять
                if self.arduino is ser:
                    self.arduino = None
                return
            if not line:
                continue
            # Формат от Arduino: "S:12.34" (или debug-строка "Speed: 12.34")
            if line.startswith(('S:', 'Speed:')):
                try:
                    speed = float(line.split(':', 1)[1])
                except ValueError:
                    continue
                if (math.isfinite(speed) and self.arduino is ser and
                        generation == self._reader_generation):
                    self.current_speed = speed
                    self._speed_sample = (speed, time.monotonic())

    def get_speed(self):
        """Показание скорости от Arduino без пересчёта."""
        speed, received_at = self._speed_sample
        if self.arduino is None or time.monotonic() - received_at >= 0.5:
            return 0.0
        return speed

    def get_speed_sample(self):
        return self._speed_sample if self.arduino is not None else (0.0, 0.0)
   
    def set_speeds(self, forward, back):
        self.config.forward_speed = forward
        self.config.back_speed = back
   
    def update(self, speed, steering):
        if not self.arduino: return
        speed_clamped = max(-1.0, min(1.0, float(speed)))
        motor_value = self.config.neutral_speed
        if speed_clamped > 0: 
            motor_value = int(self.config.neutral_speed + (self.config.forward_speed - self.config.neutral_speed) * speed_clamped)
        elif speed_clamped < 0: 
            motor_value = int(self.config.neutral_speed + (self.config.back_speed - self.config.neutral_speed) * abs(speed_clamped))
        self.update_pwm(motor_value, steering)

    def update_pwm(self, motor_value, steering):
        if not self.arduino:
            return
        motor_value = int(motor_value)
        motor_value = max(self.config.esc_min, min(self.config.esc_max, motor_value))
       
        steering_clamped = max(-1.0, min(1.0, float(steering)))
        steer_value = int(self.config.center_steering - (steering_clamped * self.config.steering_range))
        # Ограничиваем steer_value в рабочий диапазон сервопривода
        steer_value = max(self.config.center_steering - self.config.steering_range, 
                          min(self.config.center_steering + self.config.steering_range, steer_value))
       
        command = f"<{motor_value},{steer_value}>"
        current_time = time.time()
        should_send = False

        if self.last_sent_cmd != command:
            should_send = True
        elif (current_time - self.last_sent_time) > self.config.command_interval:
            should_send = True

        if should_send:
            with self.lock:
                try:
                    self.arduino.write(command.encode('utf-8'))
                    self.last_sent_cmd = command
                    self.last_sent_time = current_time
                    self.last_command_time = current_time 
                except: self.arduino = None
    
    def stop(self):
        if not self.arduino: return
        with self.lock:
            try:
                cmd = f"<{self.config.neutral_speed},{self.config.center_steering}>"
                self.arduino.write(cmd.encode('utf-8'))
                self.last_sent_cmd = cmd
                self.last_command_time = time.time()
            except: pass
   
    def check_stop(self):
        if self.arduino and time.time() - self.last_command_time > self.config.watchdog_timeout: self.stop()
        
    def close(self):
        self.stop()
        self._speed_sample = (0.0, 0.0)
        self.current_speed = 0.0
        time.sleep(self.config.arduino_close_delay)
        if self.arduino: self.arduino.close()
        self.arduino = None
    
    def restart(self):
        """Перезагрузка подключения Arduino"""
        self.close()
        time.sleep(0.5)
        port = find_arduino_port()
        if port is None: 
            self.arduino = None
            return
        try:
            self.arduino = serial.Serial(port, self.config.baud_rate, timeout=1)
            time.sleep(self.config.arduino_init_delay)
            self.stop()
            time.sleep(self.config.arduino_post_stop_delay)
            self._start_reader()
        except Exception as e:
            self.arduino = None
