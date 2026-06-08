import time
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

        port = find_arduino_port()
        if port is None: return
        
        try:
            self.arduino = serial.Serial(port, self.config.baud_rate, timeout=1)
            time.sleep(self.config.arduino_init_delay)
            self.stop()
            time.sleep(self.config.arduino_post_stop_delay)
        except Exception as e:
            self.arduino = None
   
    def set_speeds(self, forward, back):
        self.config.forward_speed = forward
        self.config.back_speed = back
   
    def update(self, speed, steering):
        if not self.arduino: return
        speed_clamped = max(-1.0, min(1.0, float(speed)))
        motor_value = self.config.neutral_speed
        if speed_clamped > 0: motor_value = int(self.config.neutral_speed + (self.config.forward_speed - self.config.neutral_speed) * speed_clamped)
        elif speed_clamped < 0: motor_value = int(self.config.neutral_speed + (self.config.back_speed - self.config.neutral_speed) * abs(speed_clamped))
       
        steering_clamped = max(-1.0, min(1.0, float(steering)))
        steer_value = int(self.config.center_steering - (steering_clamped * self.config.steering_range))
        steer_value = max(0, min(270, steer_value))
       
        command = f"<{motor_value},{steer_value}>"
        current_time = time.time()
        if command != self.last_sent_cmd or (current_time - self.last_sent_time) > self.config.command_interval:
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
        time.sleep(self.config.arduino_close_delay)
        if self.arduino: self.arduino.close()