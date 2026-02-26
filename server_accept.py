import pygame
import socket
import logging

logging.basicConfig(level=logging.INFO, format='%(message)s')
logger = logging.getLogger(__name__)

# Настройки
JETSON_IP = "10.160.140.176"  
UDP_PORT = 5005
Tick_rate = 20

def main():
    pygame.init()
    screen = pygame.display.set_mode((400, 300))
    pygame.display.set_caption(f"Пульт: {JETSON_IP}")
    font = pygame.font.SysFont('Arial', 20, bold=True)
    small_font = pygame.font.SysFont('Arial', 16)
    
    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    last_udp_cmd = ""
    
    controls = [
        "УПРАВЛЕНИЕ:",
        "Стрелки          - Движение и поворот",
        "Q                - Выход",
    ]
    
    running = True
    clock = pygame.time.Clock()
    
    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT or (event.type == pygame.KEYDOWN and event.key == pygame.K_q):
                running = False
        
        keys = pygame.key.get_pressed()
        
        # Логика кнопок (1, -1 или 0)
        speed = 1 if keys[pygame.K_UP] else (-1 if keys[pygame.K_DOWN] else 0)
        steering = -1 if keys[pygame.K_LEFT] else (1 if keys[pygame.K_RIGHT] else 0)
        
        udp_cmd = f"{speed},{steering}"
        
        # Отправка только при изменении состояния
        if udp_cmd != last_udp_cmd:
            try:
                sock.sendto(udp_cmd.encode('utf-8'), (JETSON_IP, UDP_PORT))
                logger.info(f"Отправлено по Wi-Fi: {udp_cmd}")
                last_udp_cmd = udp_cmd
            except Exception as e:
                logger.error(f"Ошибка сети: {e}")

        # Отрисовка интерфейса
        screen.fill((30, 30, 30))
        screen.blit(font.render(f"Связь: {JETSON_IP}:{UDP_PORT}", True, (0, 255, 255)), (20, 20))
        for i, line in enumerate(controls):
            screen.blit(small_font.render(line, True, (200, 200, 200)), (20, 80 + i*25))
        screen.blit(font.render(f"Состояние: [{udp_cmd}]", True, (255, 255, 0)), (20, 180))
        
        pygame.display.flip()
        clock.tick(Tick_rate)
        
    sock.close()
    pygame.quit()

if __name__ == "__main__":
    main()