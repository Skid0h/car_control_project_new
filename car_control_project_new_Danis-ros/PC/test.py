"""Запускается на ПК, отправляет команды управления на JETSON_IP + UDP_PORT"""

import pygame
import socket
import logging

logging.basicConfig(level=logging.INFO, format='%(message)s')
logger = logging.getLogger(__name__)

# Настройки
JETSON_IP = "192.168.137.50"  
UDP_PORT = 5005
Tick_rate = 20

def main():
    pygame.init()
    screen = pygame.display.set_mode((500, 350))
    pygame.display.set_caption(f"Пульт: {JETSON_IP}")
    font = pygame.font.SysFont('Arial', 20, bold=True)
    small_font = pygame.font.SysFont('Arial', 16)
    
    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    last_udp_cmd = ""
    last_system_cmd = ""
    
    controls = [
        "УПРАВЛЕНИЕ ДВИЖЕНИЕМ:",
        "Стрелки          - Движение и поворот",
        "",
        "УПРАВЛЕНИЕ ЗАПИСЬЮ:",
        "R                - НАЧАТЬ запись видео",
        "C                - ОСТАНОВИТЬ запись",
        "Q                - ВЫЙТИ из программы",
        "",
        f"Связь: {JETSON_IP}:{UDP_PORT}"
    ]
    
    running = True
    clock = pygame.time.Clock()
    
    while running:
        system_cmd = ""
        
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            
            # Обработка нажатий клавиш для системных команд
            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_r:
                    system_cmd = "R"
                    logger.info(f"Отправлено: {system_cmd} - НАЧАТЬ ЗАПИСЬ")
                elif event.key == pygame.K_c:
                    system_cmd = "C"
                    logger.info(f"Отправлено: {system_cmd} - ОСТАНОВИТЬ ЗАПИСЬ")
                elif event.key == pygame.K_q:
                    system_cmd = "Q"
                    logger.info(f"Отправлено: {system_cmd} - ВЫХОД")
                    running = False
        
        keys = pygame.key.get_pressed()
        
        # Логика движения
        speed = 1 if keys[pygame.K_UP] else (-1 if keys[pygame.K_DOWN] else 0)
        steering = -1 if keys[pygame.K_LEFT] else (1 if keys[pygame.K_RIGHT] else 0)
        
        udp_cmd = f"{speed},{steering}"
        
        # Отправка команды движения только при изменении
        if udp_cmd != last_udp_cmd:
            try:
                sock.sendto(udp_cmd.encode('utf-8'), (JETSON_IP, UDP_PORT))
                logger.info(f"Движение: {udp_cmd}")
                last_udp_cmd = udp_cmd
            except Exception as e:
                logger.error(f"Ошибка сети: {e}")
        
        # Отправка системной команды (R, C, Q)
        if system_cmd and system_cmd != last_system_cmd:
            try:
                sock.sendto(system_cmd.encode('utf-8'), (JETSON_IP, UDP_PORT))
                last_system_cmd = system_cmd
            except Exception as e:
                logger.error(f"Ошибка сети: {e}")

        # Отрисовка интерфейса
        screen.fill((30, 30, 30))
        
        y_offset = 20
        for i, line in enumerate(controls):
            color = (200, 200, 200)
            if "ЗАПИСЬ" in line:
                color = (255, 200, 0)
            elif "ВЫЙТИ" in line:
                color = (255, 100, 100)
            screen.blit(small_font.render(line, True, color), (20, y_offset + i*22))
        
        # Текущее состояние
        status_y = y_offset + len(controls)*22 + 10
        screen.blit(font.render(f"Движение: [{udp_cmd}]", True, (255, 255, 0)), (20, status_y))
        
        if last_system_cmd in ['R', 'C']:
            status_text = "ЗАПИСЬ АКТИВНА" if last_system_cmd == 'R' else "ЗАПИСЬ ОСТАНОВЛЕНА"
            status_color = (255, 0, 0) if last_system_cmd == 'R' else (0, 255, 0)
            screen.blit(font.render(f"Статус: {status_text}", True, status_color), (20, status_y + 30))
        
        pygame.display.flip()
        clock.tick(Tick_rate)
        
    sock.close()
    pygame.quit()
    logger.info("Программа завершена")

if __name__ == "__main__":
    main()
