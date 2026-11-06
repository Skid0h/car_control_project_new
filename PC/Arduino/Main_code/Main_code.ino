#include <Servo.h>

// ==================================================
// НАСТРОЙКИ
// ==================================================

// Пины
const int MOTOR_PIN = 6;              // ESC двигателя
const int STEERING_PIN = 8;           // Сервопривод руля
const int SPEED_SENSOR_PIN = A0;      // Датчик скорости (аналоговый вход)

// Serial
const long SERIAL_BAUD = 9600;        // Скорость UART

// ESC
const int ESC_MIN = 1400;             // Максимальный реверс
const int ESC_NEUTRAL = 1500;         // Нейтраль
const int ESC_MAX = 1600;             // Максимальный газ
const int ESC_REVERSE_DELAY = 150;    // Пауза при смене направления, мс

// Failsafe
const unsigned long FAILSAFE_TIMEOUT = 500;  // Потеря связи, мс

// Руль
const int STEERING_MIN = 0;            // Минимальный угол
const int STEERING_MAX = 180;          // Максимальный угол
const int STEERING_CENTER = 90;        // Центральное положение
const int STEERING_STEP = 3;           // Минимальный физический шаг, градусы
const unsigned long STEERING_INTERVAL = 20; // Интервал движения, мс

// Датчик скорости
const int Td = 10;                     // Интервал опроса датчика, мс
const double WHEEL_LENGTH = 1;         // Длина окружности колеса, м (1 = скорость в оборотах/с)
const int THRESHOLD_VALUE = 800;       // Порог резкого падения сигнала
const unsigned long LED_PULSE = 50;    // Длительность вспышки светодиода, мс
const bool SPEED_DEBUG = true;         // true = печатать "Speed: ..." в Serial

// UART-пакет
const byte BUFFER_SIZE = 32;

// ==================================================
// ПЕРЕМЕННЫЕ
// ==================================================

Servo motorESC;
Servo steeringServo;

int lastMotorValue = ESC_NEUTRAL;

int currentSteering = STEERING_CENTER;
int targetSteering = STEERING_CENTER;

unsigned long lastCommandTime = 0;
unsigned long lastSteeringUpdate = 0;

bool failsafeActive = false;
bool newData = false;

char receivedChars[BUFFER_SIZE];

// Датчик скорости
int lastAnalogValue = THRESHOLD_VALUE;
unsigned long lastImpulseTime = 0;     // Время прошлого импульса
unsigned long lastSpeedCheck = 0;      // Время прошлого опроса датчика
unsigned long ledOnTime = 0;           // Когда включили светодиод
bool firstPulse = true;                // Флаг: был ли уже первый импульс
bool ledIsOn = false;                  // Флаг состояния светодиода

// ==================================================
// SETUP
// ==================================================

void setup() {
  Serial.begin(SERIAL_BAUD);

  motorESC.attach(MOTOR_PIN, ESC_MIN, ESC_MAX);
  steeringServo.attach(STEERING_PIN);

  pinMode(SPEED_SENSOR_PIN, INPUT);
  pinMode(LED_BUILTIN, OUTPUT);

  stopCar();
}

// ==================================================
// LOOP
// ==================================================

void loop() {

  updateSpeedSensor();

  recvWithStartEndMarkers();

  if (newData) {
    parseData();
    newData = false;
  }

  updateSteering();

  if ((millis() - lastCommandTime > FAILSAFE_TIMEOUT) &&
      !failsafeActive) {

    stopCar();
    failsafeActive = true;
  }
}

// ==================================================
// ДАТЧИК СКОРОСТИ
// ==================================================

// Проверка оборота: если разница большая, значит, был оборот
// (сигнал упал с ~800 до ~1)
bool rot(int current, int previous) {
  return (previous - current >= THRESHOLD_VALUE - 1);
}

// Неблокирующее обновление датчика (вместо delay(Td) из тестового кода)
void updateSpeedSensor() {

  unsigned long now = millis();

  // Гасим светодиод через LED_PULSE мс после импульса (без delay)
  if (ledIsOn && (now - ledOnTime >= LED_PULSE)) {
    digitalWrite(LED_BUILTIN, LOW);
    ledIsOn = false;
  }

  // Опрашиваем датчик не чаще, чем раз в Td мс
  if (now - lastSpeedCheck < Td) {
    return;
  }
  lastSpeedCheck = now;

  int analogValue = analogRead(SPEED_SENSOR_PIN);

  if (rot(analogValue, lastAnalogValue)) {

    if (firstPulse) {
      // Самый первый импульс: просто запоминаем время, разницы нет
      lastImpulseTime = now;
      firstPulse = false;
    } else {
      // Разница во времени между этим и прошлым импульсом
      unsigned long deltaTime = now - lastImpulseTime;

      if (deltaTime > 0) {
        // Скорость = 1 оборот / (время в секундах)
        double revPerSec = 1000.0 / deltaTime;
        double speed = revPerSec * WHEEL_LENGTH;

        if (SPEED_DEBUG) {
          Serial.print("Speed: ");
          Serial.println(speed);
        }

        // Вспышка светодиода для наглядности
        digitalWrite(LED_BUILTIN, HIGH);
        ledIsOn = true;
        ledOnTime = now;
      }
    }

    // Обновляем время "прошлого" импульса для следующего раза
    lastImpulseTime = now;
  }

  lastAnalogValue = analogValue;
}

// ==================================================
// UART
// ==================================================

void recvWithStartEndMarkers() {

  static bool recvInProgress = false;
  static byte index = 0;

  const char startMarker = '<';
  const char endMarker = '>';

  while (Serial.available() > 0 && !newData) {

    char rc = Serial.read();

    if (recvInProgress) {

      if (rc != endMarker) {

        receivedChars[index] = rc;
        index++;

        if (index >= BUFFER_SIZE) {
          index = BUFFER_SIZE - 1;
        }

      } else {

        receivedChars[index] = '\0';

        recvInProgress = false;
        index = 0;
        newData = true;
      }

    } else if (rc == startMarker) {

      recvInProgress = true;
    }
  }
}

// ==================================================
// ПАРСИНГ КОМАНДЫ
// ==================================================

void parseData() {

  char *token;

  token = strtok(receivedChars, ",");

  if (token == NULL) {
    return;
  }

  int motorValue = atoi(token);

  token = strtok(NULL, ",");

  if (token == NULL) {
    return;
  }

  int steeringValue = atoi(token);

  if (motorValue >= ESC_MIN &&
      motorValue <= ESC_MAX &&
      steeringValue >= STEERING_MIN &&
      steeringValue <= STEERING_MAX) {

    lastCommandTime = millis();
    failsafeActive = false;

    sendMotorCommand(motorValue);

    targetSteering = steeringValue;
  }
}

// ==================================================
// ПЛАВНОЕ УПРАВЛЕНИЕ РУЛЁМ
// ==================================================

void updateSteering() {

  unsigned long now = millis();

  if (now - lastSteeringUpdate < STEERING_INTERVAL) {
    return;
  }

  if (currentSteering == targetSteering) {
    return;
  }

  int difference = targetSteering - currentSteering;

  if (abs(difference) < STEERING_STEP) {
    currentSteering = targetSteering;
  }
  else if (difference > 0) {
    currentSteering += STEERING_STEP;
  }
  else {
    currentSteering -= STEERING_STEP;
  }

  steeringServo.write(currentSteering);

  lastSteeringUpdate = now;
}

// ==================================================
// ESC
// ==================================================

void sendMotorCommand(int newMotorValue) {

  bool sameDirection =
    (lastMotorValue >= ESC_NEUTRAL && newMotorValue >= ESC_NEUTRAL) ||
    (lastMotorValue <= ESC_NEUTRAL && newMotorValue <= ESC_NEUTRAL);

  if (sameDirection) {

    motorESC.writeMicroseconds(newMotorValue);
    lastMotorValue = newMotorValue;

    return;
  }

  motorESC.writeMicroseconds(ESC_NEUTRAL);
  lastMotorValue = ESC_NEUTRAL;

  delay(ESC_REVERSE_DELAY);

  motorESC.writeMicroseconds(newMotorValue);
  lastMotorValue = newMotorValue;
}

// ==================================================
// ОСТАНОВКА
// ==================================================

void stopCar() {

  motorESC.writeMicroseconds(ESC_NEUTRAL);
  steeringServo.write(STEERING_CENTER);

  lastMotorValue = ESC_NEUTRAL;

  currentSteering = STEERING_CENTER;
  targetSteering = STEERING_CENTER;

  lastSteeringUpdate = millis();
}
