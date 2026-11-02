"""
Управление машинкой через UART (Arduino + ESC + серво руля).

Формат пакета: "<motor,steer>", например "<1500,90>".

Главное правило этого модуля: на провод не уходит ни одно значение, которое
прошивка отбросила бы. В Main_code.ino пакет принимается только если
ESC_MIN <= motor <= ESC_MAX И STEERING_MIN <= steer <= STEERING_MAX, иначе он
отбрасывается ЦЕЛИКОМ (вместе с рулём) и не обновляет failsafe-таймер -
через 500 мс машина аварийно останавливается. Поэтому мотор и руль здесь
клампятся по esc_min/esc_max/steer_min/steer_max из конфига, а set_speeds()
не позволяет задать forward_speed/back_speed вне этих пределов.
"""

import logging
import threading
import time

import serial
import serial.tools.list_ports

logger = logging.getLogger(__name__)

# Подстроки и VID, по которым ищем Arduino, если порт не задан в конфиге явно.
_PORT_HINTS = ('Arduino', 'CH340', 'CP210', 'FTDI', 'USB Serial', 'ttyUSB', 'ttyACM')
_KNOWN_VIDS = (0x2341, 0x1A86, 0x0403, 0x10C4)


def find_arduino_port(preferred=""):
    """Возвращает имя порта или None.

    Если в конфиге задан arduino_port - используется он и только он: на Jetson
    часто несколько USB-UART устройств, и автопоиск по подстроке 'USB Serial'
    может выбрать не то.
    """
    ports = list(serial.tools.list_ports.comports())
    if preferred:
        for port in ports:
            if port.device == preferred:
                logger.info("Arduino: порт из конфига %s найден.", preferred)
                return port.device
        logger.error("Arduino: заданный в конфиге порт '%s' не найден. Доступны: %s",
                     preferred, [p.device for p in ports] or 'нет устройств')
        return None

    for port in ports:
        description = port.description or ''
        if any(hint.lower() in description.lower() for hint in _PORT_HINTS):
            logger.info("Arduino найдена на %s (%s).", port.device, description)
            return port.device

    for port in ports:
        if port.vid in _KNOWN_VIDS:
            logger.info("Arduino найдена на %s по VID=%04X PID=%s.",
                        port.device, port.vid, port.pid)
            return port.device

    logger.warning("Arduino не найдена. Доступные порты: %s",
                   [p.device for p in ports] or 'нет устройств')
    return None


class CarController:
    """Отправка команд приводам, watchdog и переподключение порта."""

    def __init__(self, config):
        self.config = config
        self.lock = threading.RLock()
        self._connect_lock = threading.Lock()   # одна попытка подключения за раз
        self.last_sent_time = 0.0        # время последней фактической отправки (для command_interval)
        self.last_command_time = 0.0     # для watchdog
        self.last_sent_cmd = ""          # последняя отправленная строка
        self.last_error = ""             # текст последней ошибки - уходит в телеметрию
        self.arduino = None
        self._next_retry_time = 0.0
        self._open()

    # ------------------------------------------------------------------
    # Подключение
    # ------------------------------------------------------------------
    def _open(self):
        """Открывает порт и ставит машину в нейтраль. Возвращает True при успехе.

        Задержки инициализации (~2.5 с) выполняются БЕЗ удержания self.lock,
        иначе поток управления всё это время не смог бы отправить ни одной команды.
        """
        port = find_arduino_port(self.config.arduino_port)
        if port is None:
            with self.lock:
                self.arduino = None
            self.last_error = 'Arduino: порт не найден'
            return False

        try:
            handle = serial.Serial(port, self.config.baud_rate, timeout=1)
        except Exception as exc:
            self.last_error = 'Arduino: не открыть %s: %s' % (port, exc)
            logger.error("%s", self.last_error)
            return False

        # Arduino после подачи питания ~2 с сидит в загрузчике: первые байты теряются.
        time.sleep(self.config.arduino_init_delay)
        with self.lock:
            self.arduino = handle
            self.last_error = ''
        self.stop()
        time.sleep(self.config.arduino_post_stop_delay)
        logger.info("Arduino: подключение установлено (%s @ %s).", port, self.config.baud_rate)
        return True

    def _connect(self):
        """Одна попытка подключения, защищённая от параллельного входа."""
        if not self._connect_lock.acquire(blocking=False):
            return False
        try:
            return self._open()
        finally:
            self._connect_lock.release()

    def try_reconnect(self):
        """Периодическая попытка вернуть связь. Вызывается из основного цикла.

        Arduino может появиться в системе позже Jetson (например после подачи
        питания на борт), и без этого машина оставалась бы неподключённой
        до ручной команды F.
        """
        if self.arduino is not None:
            return True
        now = time.time()
        if now < self._next_retry_time:
            return False
        self._next_retry_time = now + max(0.5, self.config.arduino_retry_interval)
        logger.info("Arduino: попытка переподключения...")
        return self._connect()

    def restart(self):
        """Полное переподключение по команде оператора."""
        self.close()
        time.sleep(0.5)
        ok = self._connect()
        if not ok:
            logger.error("Arduino: переподключение не удалось.")
        return ok
    def close(self):
        self.stop()
        time.sleep(self.config.arduino_close_delay)
        with self.lock:
            if self.arduino is not None:
                try:
                    self.arduino.close()
                except Exception as exc:
                    logger.warning("Arduino: ошибка закрытия порта: %s", exc)
                self.arduino = None
        self.last_sent_cmd = ""

    @property
    def connected(self):
        return self.arduino is not None

    # ------------------------------------------------------------------
    # Значения привода
    # ------------------------------------------------------------------
    def _clamp_motor(self, value):
        return int(max(self.config.esc_min, min(self.config.esc_max, round(value))))

    def _clamp_steer(self, value):
        return int(max(self.config.steer_min, min(self.config.steer_max, round(value))))

    def set_speeds(self, forward, back):
        """Устанавливает ШИМ полного газа вперёд/назад.

        Значения приводятся к допустимому диапазону прошивки и к правильной
        стороне нейтрали: forward не может быть ниже neutral, back не может
        быть выше neutral. Иначе положительная команда газа поехала бы назад.
        Принятые значения попадают в телеметрию, поэтому пульт видит реальность.
        """
        with self.lock:
            neutral = self.config.neutral_speed
            esc_min = self.config.esc_min
            esc_max = self.config.esc_max
            try:
                fwd = int(forward)
                bck = int(back)
            except (TypeError, ValueError):
                logger.warning("speed: нечисловые значения %r/%r, проигнорировано.", forward, back)
                return False

            fwd_clamped = max(neutral, min(esc_max, fwd))
            bck_clamped = min(neutral, max(esc_min, bck))
            if fwd_clamped != fwd or bck_clamped != bck:
                logger.warning("speed: запрошено %d/%d, принято %d/%d (пределы %d..%d, нейтраль %d).",
                               fwd, bck, fwd_clamped, bck_clamped, esc_min, esc_max, neutral)
            self.config.forward_speed = fwd_clamped
            self.config.back_speed = bck_clamped
            return True

    def _motor_from_speed(self, speed_clamped):
        neutral = self.config.neutral_speed
        if speed_clamped > 0:
            raw = neutral + (self.config.forward_speed - neutral) * speed_clamped
        elif speed_clamped < 0:
            raw = neutral + (self.config.back_speed - neutral) * abs(speed_clamped)
        else:
            raw = neutral
        return self._clamp_motor(raw)

    # ------------------------------------------------------------------
    # Команды
    # ------------------------------------------------------------------
    def update(self, speed, steering):
        """speed в [-1; 1] (отрицательный = назад), steering в [-1; 1]."""
        with self.lock:
            if self.arduino is None:
                return False

            try:
                speed_value = float(speed)
                steering_value = float(steering)
            except (TypeError, ValueError):
                logger.warning("update: нечисловые значения %r/%r, команда пропущена.", speed, steering)
                return False

            speed_clamped = max(-1.0, min(1.0, speed_value))
            steering_clamped = max(-1.0, min(1.0, steering_value))

            motor_value = self._motor_from_speed(speed_clamped)
            steer_value = self._clamp_steer(
                self.config.center_steering - steering_clamped * self.config.steering_range)

            command = "<%d,%d>" % (motor_value, steer_value)
            now = time.time()
            repeated = (command == self.last_sent_cmd)
            if repeated and (now - self.last_sent_time) <= self.config.command_interval:
                # Ту же команду чаще command_interval слать незачем.
                self.last_command_time = now
                return True

            return self._write(command, now)

    def stop(self):
        """Нейтраль мотора и руль в центр. Отправляется безусловно, без дедупликации."""
        with self.lock:
            if self.arduino is None:
                return False
            command = "<%d,%d>" % (self._clamp_motor(self.config.neutral_speed),
                                   self._clamp_steer(self.config.center_steering))
            return self._write(command, time.time())

    def _write(self, command, now):
        """Вызывается только под self.lock."""
        try:
            self.arduino.write(command.encode('utf-8'))
        except Exception as exc:
            self.arduino = None
            self.last_error = 'UART: %s' % exc
            self.last_sent_cmd = ""
            logger.error("Arduino: запись '%s' не удалась, соединение закрыто: %s", command, exc)
            return False
        self.last_sent_cmd = command
        self.last_sent_time = now
        self.last_command_time = now
        return True

    def check_stop(self):
        """Watchdog: если команд не было дольше watchdog_timeout - остановка.

        last_command_time обновляется при каждой фактической отправке, в том числе
        из потока управления в авто-режиме. Если поток управления или детекции
        зависнет, команды перестанут приходить и машина встанет здесь,
        раньше, чем сработает failsafe Arduino.
        """
        with self.lock:
            if self.arduino is None:
                return False
            if time.time() - self.last_command_time > self.config.watchdog_timeout:
                logger.warning("Watchdog: команд не было %.2f с - остановка.",
                               time.time() - self.last_command_time)
                return self._write("<%d,%d>" % (self._clamp_motor(self.config.neutral_speed),
                                                self._clamp_steer(self.config.center_steering)),
                                   time.time())
        return False