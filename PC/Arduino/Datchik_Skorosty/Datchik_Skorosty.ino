#define analogInPin A0
#define Td 10
#define wheelLength 1
#define thresholdValue 800

// Глобальные переменные (живут между циклами loop)
int lastAnalogValue = thresholdValue;
unsigned long lastImpulseTime = 0; // Время прошлого импульса
bool firstPulse = true;            // Флаг: был ли уже первый импульс

void setup() {
  Serial.begin(9600);
  pinMode(analogInPin, INPUT);
  pinMode(LED_BUILTIN, OUTPUT);
}

// Твоя логика проверки: если разница большая, значит, был оборот
bool rot(int current, int previous) {
  // Проверяем, упало ли значение резко (было ~800, стало ~1)
  return (previous - current >= thresholdValue - 1);
}

void loop() {
  int analogValue = analogRead(analogInPin); // ИСПРАВЛЕНО: только один аргумент

  if (rot(analogValue, lastAnalogValue)) {
    unsigned long currentTime = millis(); // Фиксируем время СЕЙЧАС

    if (firstPulse) {
      // Это самый первый импульс. Просто запоминаем время, разницы нет.
      lastImpulseTime = currentTime;
      firstPulse = false;
    } else {
      // Считаем разницу во времени между этим и прошлым импульсом
      unsigned long deltaTime = currentTime - lastImpulseTime;

      if (deltaTime > 0) {
        // Скорость = 1 оборот / (время в секундах)
        double revPerSec = 1000.0 / deltaTime; 
        double speed = revPerSec * wheelLength;
        
        Serial.print("Speed: ");
        Serial.println(speed);
        
        // Мигаем светодиодом для наглядности
        digitalWrite(LED_BUILTIN, HIGH);
        delay(50);
        digitalWrite(LED_BUILTIN, LOW);
      }
    }

    // Обновляем время "прошлого" импульса для следующего раза
    lastImpulseTime = currentTime;
  }

  lastAnalogValue = analogValue; // ИСПРАВЛЕНО: присваивание значения
  delay(Td);
}
