"""Запускается на Jetson, принимает команды управления с JETSON_IP + UDP_PORT"""

import socket
import serial
import serial.tools.list_ports
import time
import logging

logging.basicConfig(level=logging.DEBUG, format='%(asctime)s [%(levelname)s] %(message)s')
logger = logging.getLogger(__name__)

UDP_IP = "192.168.137.50"
UDP_PORT = 5005

ARDUINO_PORT = '/dev/ttyUSB0'
Baud = 9600

base_speed = 90
forward_speed = 98
back_speed = 82

base_rotation = 90
right_rotation = 30
left_rotation = 150

def find_arduino_port():
    ports = serial.tools.list_ports.comports()
    
    for port in ports:
        if 'Arduino' in port.description or 'CH340' in port.description or 'USB Serial' in port.description:
            return port.device
        if port.vid and port.pid:
            if (port.vid == 0x2341) or (port.vid == 0x1A86):
                return port.device
    
    for port in ports:
        try:
            test_serial = serial.Serial(port.device, Baud, timeout=1)
            time.sleep(1)
            test_serial.write("90,90\n".encode('utf-8'))
            test_serial.close()
            return port.device
        except:
            continue
    
    return ARDUINO_PORT

class CarController:
    def __init__(self, port=None):
        if port is None:
            port = find_arduino_port()
        
        self.arduino = None
        try:
            self.arduino = serial.Serial(port, Baud, timeout=1)
            time.sleep(2)
            self.stop()
            time.sleep(0.5)
            logger.info(f"Arduino подключен к {port}")
        except Exception as e:
            logger.error(f"Ошибка подключения к {port}: {e}")
            self.arduino = None
    
    def update(self, speed, steering):
        if not self.arduino:
            return

        motor_value = base_speed
        if speed > 0:
            motor_value = forward_speed
        elif speed < 0:
            motor_value = back_speed
        
        steer_value = base_rotation
        if steering < 0:
            steer_value = left_rotation
        elif steering > 0:
            steer_value = right_rotation
        
        command = f"{motor_value},{steer_value}\n"
        self.arduino.write(command.encode('utf-8'))
        logger.debug(f"Команда: {command.strip()}")
    
    def stop(self):
        if self.arduino:
            self.arduino.write("90,90\n".encode('utf-8'))
    
    def close(self):
        self.stop()
        time.sleep(0.1)
        if self.arduino:
            self.arduino.close()
        logger.info("Arduino отключен")

def main():
    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    sock.bind((UDP_IP, UDP_PORT))
    logger.info(f"Сервер слушает порт {UDP_PORT}")
    
    car = CarController()
    
    try:
        while True:
            data, addr = sock.recvfrom(1024)
            command = data.decode('utf-8')
            
            try:
                speed, steering = map(int, command.split(','))
                car.update(speed, steering)
            except ValueError:
                logger.warning(f"Неверный формат: {command}")
                
    except KeyboardInterrupt:
        logger.info("Остановка сервера...")
    finally:
        car.close()
        sock.close()

if __name__ == "__main__":
    main()
