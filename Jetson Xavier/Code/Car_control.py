
import time
import serial
import threading
from Code.Arduino_auto_find import find_arduino_port

#Функции управления машинкой

class CarController:
    def __init__(self, config):
        self.config = config
        self.car = config['car']
        self.timing = config['timing']
        self.lock = threading.Lock()
        self.forward_speed = self.car['forward_speed']
        self.back_speed = self.car['back_speed']
        self.neutral_speed = self.car['neutral_speed']
        self.center_steering = self.car['center_steering']
        self.command_interval  = self.car.get('command_interval', 0.1)  
        self.last_sent_time    = 0                                      # Время отправки последней команды (для command_interval)
        self.watchdog_timeout  = self.car.get('watchdog_timeout', 0.4)  
        self.last_command_time = 0                                      # Для watchdog
        self.steering_range    = self.car['steering_range']
        self.last_sent_cmd     = ""                                     # Последняя отправленная строка команды
        self.arduino = None

        port = find_arduino_port()
        if port is None: return
        
        try:
            self.arduino = serial.Serial(port, self.car['baud_rate'], timeout=1)
            time.sleep(self.timing['arduino_init_delay'])
            self.stop()
            time.sleep(self.timing['arduino_post_stop_delay'])
        except Exception as e:
            self.arduino = None
   
    def set_speeds(self, forward, back):
        self.forward_speed = forward
        self.back_speed = back
   
    def update(self, speed, steering):
        if not self.arduino: return
        speed_clamped = max(-1.0, min(1.0, float(speed)))
        motor_value = self.neutral_speed
        if speed_clamped > 0: motor_value = int(self.neutral_speed + (self.forward_speed - self.neutral_speed) * speed_clamped)
        elif speed_clamped < 0: motor_value = int(self.neutral_speed + (self.back_speed - self.neutral_speed) * abs(speed_clamped))
       
        steering_clamped = max(-1.0, min(1.0, float(steering)))
        steer_value = int(self.center_steering - (steering_clamped * self.steering_range))
        steer_value = max(0, min(270, steer_value))
       
        command = f"<{motor_value},{steer_value}>"
        current_time = time.time()
        if command != self.last_sent_cmd or (current_time - self.last_sent_time) > self.command_interval:
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
                cmd = f"<{self.neutral_speed},{self.center_steering}>"
                self.arduino.write(cmd.encode('utf-8'))
                self.last_sent_cmd = cmd
                self.last_command_time = time.time()
            except: pass
   
    def check_stop(self):
        if self.arduino and time.time() - self.last_command_time > self.watchdog_timeout: self.stop()
        
    def close(self):
        self.stop()
        time.sleep(self.timing['arduino_close_delay'])
        if self.arduino: self.arduino.close()