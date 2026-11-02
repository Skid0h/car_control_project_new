const int inputPin = A2;
const int outputPin = 9;  // D9 поддерживает PWM

void setup() {
  Serial.begin(9600);
  pinMode(outputPin, OUTPUT);
}

void loop() {
  int raw = analogRead(inputPin);        // значение 0–1023
  float voltage = raw * (5.0 / 1023.0); // напряжение на A1
  int pwm = map(raw, 0, 1023, 0, 255);  // значение для PWM

  analogWrite(outputPin, pwm);

  Serial.print("A1: ");
  Serial.print(voltage, 2);
  Serial.println(" V");
  Serial.println(raw);

  delay(200);
}
