#include <Servo.h>

// ==================================================
// НАСТРОЙКИ
// ==================================================

// Пины
const int MOTOR_PIN = 6;              // ESC двигателя
const int STEERING_PIN = 8;           // Сервопривод руля

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

// ==================================================
// SETUP
// ==================================================

void setup() {
  Serial.begin(SERIAL_BAUD);

  motorESC.attach(MOTOR_PIN, ESC_MIN, ESC_MAX);
  steeringServo.attach(STEERING_PIN);

  stopCar();
}

// ==================================================
// LOOP
// ==================================================

void loop() {

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
