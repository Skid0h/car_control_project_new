import time
import pygame
import serial
import serial.tools.list_ports
import logging

Tick_rate = 10
Baud = 9600

# Значения в микросекундах
base_speed = 1500
forward_speed = 1600
back_speed = 1400

base_rotation = 130
right_rotation = 90
left_rotation = 170

# Настройка логирования
logging.basicConfig(level=logging.DEBUG, format='%(asctime)s [%(levelname)s] %(message)s')
logger = logging.getLogger(__name__)

# Автонахождение arduino
def find_arduino_port():
    """Автоматически находит порт с Arduino"""
    ports = serial.tools.list_ports.comports()
    
    for port in ports:
        if ('Arduino' in port.description or
            'CH340' in port.description or
            'USB Serial' in port.description):
            logger.info(f"Найден Arduino на порту {port.device} ({port.description})")
            return port.device
    logger.error("Arduino не найден")
    return None
        
Arduino_port = find_arduino_port()

class CarController:
    def __init__(self, port = Arduino_port):
        try:
            self.arduino = serial.Serial(port, Baud, timeout=1)
            time.sleep(2)  # Даем Arduino время на перезагрузку после подключения
            self.stop()    # Начальная остановка
            time.sleep(0.5)
            logger.info(f"Arduino подключен к {port}")
        except Exception as e:
            logger.error(f"Ошибка подключения к {port}: {e}")
            raise
    
    def update(self, speed, steering):
        motor_value = base_speed
        if speed > 0:
            motor_value = forward_speed
        elif speed < 0:
            motor_value = back_speed
        
        steer_value = base_rotation
        if steering < 0:
            steer_value = left_rotation  # Лево 
        elif steering > 0:
            steer_value = right_rotation # Право 
        
        # Отправка команды
        command = f"{motor_value},{steer_value}\n"
        self.arduino.write(command.encode())
        logger.debug(f"Команда: {command.strip()}")
    
    def stop(self):
        # Отправляем правильную нейтраль (1500,90)
        command = f"{base_speed},{base_rotation}\n"
        self.arduino.write(command.encode())
        logger.debug("Команда остановки отправлена")
    
    def close(self):
        self.stop()
        time.sleep(0.1)
        self.arduino.close()
        logger.info("Arduino отключен")

def main():
    global forward_speed, back_speed
    
    pygame.init()
    
    # Создание окна
    screen_width = 500
    screen_height = 400
    screen = pygame.display.set_mode((screen_width, screen_height))
    pygame.display.set_caption("Управление машиной (Точная настройка)")
    
    font = pygame.font.SysFont('Arial', 20)
    small_font = pygame.font.SysFont('Arial', 16)
    
    car = None
    connection_status = ""
    connection_color = (255, 0, 0)  
    
    try:
        car = CarController(Arduino_port)
        connection_status = f"Arduino Подключено: {Arduino_port}"
        connection_color = (0, 255, 0) 
    except Exception as e:
        connection_status = f"Ошибка подключения: {e}"
        connection_color = (255, 0, 0)
        car = None
    
    controls = [
        "УПРАВЛЕНИЕ:",
        "Стрелка ВВЕРХ    - Вперед",
        "Стрелка ВНИЗ     - Назад",
        "Стрелка ВЛЕВО    - Поворот налево",
        "Стрелка ВПРАВО   - Поворот направо",
        "1                - УВЕЛИЧИТЬ скорость (можно зажать)",
        "2                - УМЕНЬШИТЬ скорость (можно зажать)",
        "Q                - Выход",
    ]
    
    current_speed = "СТОП"
    current_steering = "ПРЯМО"
    last_key_pressed = None
    debug_info = ""
    
    # Таймер для контроля скорости изменения (чтобы прибавляло по 1 в секунду)
    speed_change_timer = 0
    
    logger.info("Программа управления машиной запущена")
    
    running = True
    clock = pygame.time.Clock()
    
    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            elif event.type == pygame.KEYDOWN:
                key_name = pygame.key.name(event.key)
                last_key_pressed = key_name
                
                if key_name not in ['1', '2']:
                    logger.info(f"Нажата клавиша: {key_name}")
                
                if event.key == pygame.K_q:
                    running = False
        
        keys = pygame.key.get_pressed()
        
        # --- ИЗМЕНЕНИЕ СКОРОСТИ С ЗАЖАТИЕМ (1 РАЗ В СЕКУНДУ) ---
        if keys[pygame.K_1]:  # Увеличиваем скорость
            if speed_change_timer == 0:
                forward_speed += 1
                back_speed -= 1
            
            speed_change_timer += 1
            if speed_change_timer >= Tick_rate:  # Прошла ровно 1 секунда
                speed_change_timer = 0
                
        elif keys[pygame.K_2]:  # Уменьшаем скорость
            if speed_change_timer == 0:
                forward_speed -= 1
                back_speed += 1
                
            speed_change_timer += 1
            if speed_change_timer >= Tick_rate:  # Прошла ровно 1 секунда
                speed_change_timer = 0
        else:
            # Сбрасываем таймер, если кнопки 1 и 2 отпущены
            speed_change_timer = 0
            
        # Предохранители
        forward_speed = max(1500, min(2000, forward_speed))
        back_speed = max(1000, min(1500, back_speed))
        # --------------------------------------------------------
        
        speed = 0
        if keys[pygame.K_UP]:
            speed = 1
            current_speed = "ВПЕРЕД"
        elif keys[pygame.K_DOWN]:
            speed = -1
            current_speed = "НАЗАД"
        else:
            current_speed = "СТОП"
        
        steering = 0
        if keys[pygame.K_LEFT]:
            steering = -1
            current_steering = "ЛЕВО"
        elif keys[pygame.K_RIGHT]:
            steering = 1
            current_steering = "ПРАВО"
        else:
            current_steering = "ПРЯМО"
              
        if car:
            car.update(speed, steering)
        
        debug_info = f"Скорость: {speed}, Руление: {steering}"
        if speed > 0:
            debug_info += f" | Мотор: {forward_speed} (вперед)" 
        elif speed < 0:
            debug_info += f" | Мотор: {back_speed} (назад)"
        else:
            debug_info += f" | Мотор: {base_speed} (стоп)"
            
        settings_info = f"Настройки скорости: Вперед = {forward_speed}, Назад = {back_speed}"
        
        # Отрисовка
        screen.fill((30, 30, 40))
        
        status_text = font.render(connection_status, True, connection_color)
        screen.blit(status_text, (20, 20))
        
        y_offset = 60
        state_text = font.render(f"Скорость: {current_speed}", True, (255, 255, 255))
        screen.blit(state_text, (20, y_offset))
        
        steer_text = font.render(f"Руль: {current_steering}", True, (255, 255, 255))
        screen.blit(steer_text, (20, y_offset + 30))
        
        if last_key_pressed:
            key_text = font.render(f"Последняя клавиша: {last_key_pressed.upper()}", True, (255, 200, 100))
            screen.blit(key_text, (20, y_offset + 60))
        
        debug_text = small_font.render(debug_info, True, (150, 255, 150))
        screen.blit(debug_text, (20, y_offset + 90))
        
        settings_text = small_font.render(settings_info, True, (100, 200, 255))
        screen.blit(settings_text, (20, y_offset + 110))
        
        y_offset = 200
        for i, control in enumerate(controls):
            control_text = small_font.render(control, True, (200, 200, 200))
            screen.blit(control_text, (20, y_offset + i * 22))
        
        pygame.display.flip()
        clock.tick(Tick_rate)
    
    if car:
        car.close()
        logger.info("Соединение с Arduino закрыто")
    
    pygame.quit()
    logger.info("Программа завершена")

if __name__ == "__main__":
    main()
