// Companion firmware for the ROS 2 Jetson. Calibrate constants on YOUR vehicle.
// Existing <motor_us,steer_degrees> plus optional <motor_us,steer_degrees,S> STOP.
#include <Servo.h>
#include <stdlib.h>
#include <string.h>
#include <ctype.h>

Servo motor;
Servo steering;
const int NEUTRAL = 1500, REVERSE = 1420, FORWARD = 1570;
const int CENTER = 100, STEER_MIN = 70, STEER_MAX = 130;
const unsigned long WATCHDOG_MS = 500, STEER_PERIOD_MS = 50;
const unsigned long RELEASE_MS = 1200;  // release servo torque after safe return
const int MAX_STEP = 3;  // 60 command-degrees/s, at most 20 changes/s
char input[32];
byte used = 0;
bool receiving = false;
bool stopping = true;
int angle = CENTER, target = CENTER, motorValue = NEUTRAL;
unsigned long lastPacket = 0, lastSteer = 0, stopSince = 0;

void stopMotor() {
  if (motorValue != NEUTRAL) motor.writeMicroseconds(NEUTRAL);
  motorValue = NEUTRAL;
  if (!stopping) stopSince = millis();
  stopping = true;
  target = CENTER;  // slew to the calibrated centre, then detach
}

bool number(char *text, long &result) {
  if (!*text) return false;
  for (char *p = text; *p; ++p) if (!isdigit((unsigned char)*p)) return false;
  char *end;
  result = strtol(text, &end, 10);
  return *end == '\0';
}

void packet() {
  char *comma = strchr(input, ',');
  if (!comma) { stopMotor(); return; }
  char *mode = strchr(comma + 1, ',');
  bool safeStop = false;
  if (mode) {
    safeStop = mode[1] == 'S' && mode[2] == '\0';
    *mode = '\0';
    if (!safeStop) { stopMotor(); return; }
  }
  *comma = '\0';
  long m, s;
  if (!number(input, m) || !number(comma + 1, s) ||
      m < REVERSE || m > FORWARD || s < STEER_MIN || s > STEER_MAX ||
      (safeStop && m != NEUTRAL)) {
    stopMotor();
    return;  // invalid packets never feed watchdog
  }
  lastPacket = millis();
  if (safeStop) {
    stopMotor();  // ignore stale steering angle encoded in STOP packet
    return;
  }
  stopping = false;
  if (!steering.attached()) {
    steering.attach(10);
    steering.write(angle);  // stored centre after detach
  }
  if ((int)m != motorValue) motor.writeMicroseconds((int)m);
  motorValue = (int)m;
  target = (int)s;
}

void setup() {
  Serial.begin(9600);
  motor.writeMicroseconds(NEUTRAL);
  steering.write(CENTER);
  motor.attach(9, 1000, 2000);
  steering.attach(10);
  motor.writeMicroseconds(NEUTRAL);
  steering.write(CENTER);
  lastPacket = lastSteer = stopSince = millis();
}

void loop() {
  // Bound parsing so watchdog and steering updates run under Serial floods.
  for (byte budget = 0; budget < 64 && Serial.available(); ++budget) {
    char c = Serial.read();
    if (c == '<') { used = 0; receiving = true; }
    else if (receiving && c == '>') {
      input[used] = '\0';
      receiving = false;
      packet();
    } else if (receiving) {
      if (used < sizeof(input) - 1) input[used++] = c;
      else { receiving = false; used = 0; stopMotor(); }
    }
  }
  unsigned long now = millis();
  if (now - lastPacket > WATCHDOG_MS) stopMotor();
  if (now - lastSteer >= STEER_PERIOD_MS) {
    lastSteer = now;
    int delta = constrain(target - angle, -MAX_STEP, MAX_STEP);
    if (delta != 0) {
      angle += delta;
      if (steering.attached()) steering.write(angle);
    }
  }
  if (stopping && angle == CENTER && steering.attached() &&
      now - stopSince >= RELEASE_MS) {
    steering.detach();  // stop 50 Hz pulses and holding torque
  }
}
