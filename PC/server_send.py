"""Запускается на ПК, отправляет команды управления.
  Слушает обратную телеметрию от сервера для синхронизации состояний.
"""

import pygame
import socket
import logging
import json
import time

logging.basicConfig(level=logging.INFO, format='%(message)s')   
logger = logging.getLogger(__name__)

# Настройки
JETSON_IP = "192.168.137.50"  
UDP_PORT = 5005
Tick_rate = 20  # 20 пакетов в секунду (надежное постоянное управление)

def main():
   pygame.init()
   screen = pygame.display.set_mode((550, 700)) # Немного увеличили высоту окна
   pygame.display.set_caption(f"Улучшенный Пульт: {JETSON_IP}")
   font = pygame.font.SysFont('Arial', 20, bold=True)
   small_font = pygame.font.SysFont('Arial', 16)
   
   sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
   sock.setblocking(False)  # Делаем сокет неблокирующим для мгновенного чтения ответов
   
   # Состояния, которые мы получаем от сервера (Телеметрия)
   server_mode = "MANUAL"
   server_recording = False
   server_cam_connected = False
   server_fwd_speed = 1570 # Нейтраль 1500, слегка вперед
   server_bck_speed = 1430 # Нейтраль 1500, слегка назад
   server_msg = ""
   last_telemetry_time = 0
   
   # Локальные данные для отправки (в формате ШИМ)
   current_forward_speed = 1570
   current_back_speed = 1430
   
   controls = [
       "РЕЖИМЫ РАБОТЫ:",
       "A                - Включить АВТОМАТИЧЕСКИЙ режим",
       "S                - Вернуться в РУЧНОЙ режим",
       "",
       "УПРАВЛЕНИЕ ДВИЖЕНИЕМ (Ручной режим):",
       "Стрелки          - Движение и поворот",
       "",
       "УПРАВЛЕНИЕ ЗАПИСЬЮ:",
       "R                - НАЧАТЬ запись видео",
       "C                - ОСТАНОВИТЬ запись",
       "",
       "УПРАВЛЕНИЕ СКОРОСТЬЮ (ШИМ):",
       "1                - УМЕНЬШИТЬ скорость (ближе к 1500)",
       "2                - УВЕЛИЧИТЬ скорость (дальше от 1500)",
       "",
       "ВЫХОД:",
       "Q                - ВЫЙТИ из программы",
       "",
       f"Узел связи: {JETSON_IP}:{UDP_PORT}"
   ]
   
   running = True
   clock = pygame.time.Clock()
   
   while running:
       system_cmd = ""
       
       # 1. ОБРАБОТКА ВВОДА С КЛАВИАТУРЫ
       for event in pygame.event.get():
           if event.type == pygame.QUIT:
               running = False
           
           if event.type == pygame.KEYDOWN:
               if event.key == pygame.K_a:
                   system_cmd = "A"
               elif event.key == pygame.K_s:
                   system_cmd = "S"
               elif event.key == pygame.K_r:
                   system_cmd = "R"
               elif event.key == pygame.K_c:
                   system_cmd = "C"
               elif event.key == pygame.K_q:
                   system_cmd = "Q"
                   running = False
               elif event.key == pygame.K_1:
                   # Уменьшаем скорость (двигаем значения ближе к нейтрали 1500)
                   current_forward_speed = max(1500, current_forward_speed - 1)
                   current_back_speed = min(1500, current_back_speed + 1)
                   system_cmd = f"speed:{current_forward_speed},{current_back_speed}"
               elif event.key == pygame.K_2:
                   # Увеличиваем скорость (двигаем значения к краям 2000 и 1000)
                   current_forward_speed = min(2000, current_forward_speed + 1)
                   current_back_speed = max(1000, current_back_speed - 1)
                   system_cmd = f"speed:{current_forward_speed},{current_back_speed}"
       
       # 2. ОТПРАВКА ДАННЫХ НА СЕРВЕР
       if system_cmd:
           try:
               sock.sendto(system_cmd.encode('utf-8'), (JETSON_IP, UDP_PORT))
           except Exception: pass

       keys = pygame.key.get_pressed()
       speed = 1 if keys[pygame.K_UP] else (-1 if keys[pygame.K_DOWN] else 0)
       steering = -1 if keys[pygame.K_LEFT] else (1 if keys[pygame.K_RIGHT] else 0)
       
       udp_cmd = f"{speed},{steering}"
       try:
           sock.sendto(udp_cmd.encode('utf-8'), (JETSON_IP, UDP_PORT))
       except Exception: pass


       # 3. ЧТЕНИЕ ТЕЛЕМЕТРИИ ОТ СЕРВЕРА
       try:
           while True:
               data, _ = sock.recvfrom(2048)
               telemetry = json.loads(data.decode('utf-8'))
               
               server_mode = telemetry.get('mode', 'MANUAL')
               server_recording = telemetry.get('rec', False)
               server_cam_connected = telemetry.get('cam_connected', False)
               server_fwd_speed = telemetry.get('fwd', current_forward_speed)
               server_bck_speed = telemetry.get('bck', current_back_speed)
               server_msg = telemetry.get('msg', '')
               last_telemetry_time = time.time()
               
       except BlockingIOError:
           pass
       except json.JSONDecodeError:
           pass

       # 4. ОТРИСОВКА ИНТЕРФЕЙСА
       screen.fill((30, 30, 30))
       y_offset = 20
       
       for i, line in enumerate(controls):
           color = (200, 200, 200)
           if "РЕЖИМЫ" in line or "A " in line or "S " in line:
               color = (150, 200, 255)
           elif "ЗАПИСЬ" in line:
               color = (255, 200, 0)
           elif "СКОРОСТЬ" in line:
               color = (100, 255, 100)
           elif "ВЫХОД" in line:
               color = (255, 100, 100)
           screen.blit(small_font.render(line, True, color), (20, y_offset + i*20))
       
       status_y = y_offset + len(controls)*20 + 15
       
       # Статус связи
       is_connected = (time.time() - last_telemetry_time) < 1.0
       conn_text = "СВЯЗЬ: ОК" if is_connected else "ОШИБКА СВЯЗИ (Нет ответа от Jetson)"
       conn_color = (0, 255, 0) if is_connected else (255, 0, 0)
       screen.blit(font.render(conn_text, True, conn_color), (20, status_y))
       
       # Режим
       mode_text = "АВТОМАТИЧЕСКИЙ" if server_mode == "AUTO" else "РУЧНОЙ"
       mode_color = (255, 100, 255) if server_mode == "AUTO" else (100, 255, 255)
       screen.blit(font.render(f"РЕАЛЬНЫЙ РЕЖИМ: {mode_text}", True, mode_color), (20, status_y + 35))
       
       # Статус подключения камеры
       cam_conn_text = "ОК (ZED найдена)" if server_cam_connected else "ОТКЛЮЧЕНА / ОШИБКА"
       cam_conn_color = (0, 255, 0) if server_cam_connected else (255, 50, 50)
       screen.blit(font.render(f"СТАТУС КАМЕРЫ: {cam_conn_text}", True, cam_conn_color), (20, status_y + 65))
       
       # Статус записи
       rec_text = "ЗАПИСЬ ИДЕТ" if server_recording else "ЗАПИСЬ ОСТАНОВЛЕНА"
       rec_color = (255, 0, 0) if server_recording else (150, 150, 150)
       screen.blit(font.render(f"ВИДЕО ЗАПИСЬ: {rec_text}", True, rec_color), (20, status_y + 95))
       
       # Подтвержденные скорости
       speed_y = status_y + 135
       screen.blit(font.render(f"СКОРОСТЬ (ШИМ, Сервер):", True, (100, 255, 100)), (20, speed_y))
       screen.blit(font.render(f"  Вперед: {server_fwd_speed} мкс", True, (0, 255, 0)), (40, speed_y + 25))
       screen.blit(font.render(f"  Назад:  {server_bck_speed} мкс", True, (255, 100, 100)), (40, speed_y + 50))

       # Вывод системных сообщений
       if server_msg:
           warn_color = (255, 200, 0) if int(time.time() * 4) % 2 == 0 else (255, 50, 50)
           screen.blit(font.render(f"СИСТЕМНОЕ СООБЩЕНИЕ: {server_msg}", True, warn_color), (20, speed_y + 90))
       
       pygame.display.flip()
       clock.tick(Tick_rate)
       
   sock.close()
   pygame.quit()
   logger.info("Программа завершена")

if __name__ == "__main__":
   main()
