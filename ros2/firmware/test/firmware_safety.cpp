// Deterministic host simulation of the real sketch's parser, watchdog and Servo calls.
#include <cassert>
#include <string>
#include <cstdint>
using byte = unsigned char;
unsigned long simulated_ms = 0;
unsigned long millis() { return simulated_ms; }
template <class T> T constrain(T value, T low, T high) {
  return value < low ? low : (value > high ? high : value);
}
struct FakeSerial {
  std::string pending;
  void begin(int) {}
  int available() const { return static_cast<int>(pending.size()); }
  char read() { char c = pending.front(); pending.erase(0, 1); return c; }
  void feed(const char *text) { pending += text; }
} Serial;
#include "../car_controller/car_controller.ino"

void advance(unsigned long until) {
  while (simulated_ms < until) {
    simulated_ms += 10;
    loop();
  }
}

int main() {
  setup();
  assert(steering.attached());
  Serial.feed("<1570,70>");
  loop();
  for (int i = 1; i <= 4; ++i) {
    advance(200 * i);
    Serial.feed("<1570,70>");
    loop();
  }
  assert(motorValue == 1570 && angle == 70 && steering.attached());
  // STOP immediately removes ESC torque, slowly returns the wheel to centre.
  simulated_ms = 810;
  Serial.feed("<1500,70,S>");
  loop();
  assert(motorValue == 1500 && target == CENTER && steering.attached());
  advance(860);
  assert(angle == 73);
  advance(1310);
  assert(angle == CENTER);
  advance(2020);
  assert(!steering.attached());  // no continuous holding torque
  // Neutral manual steering remains possible when the third field is absent.
  simulated_ms = 2030;
  Serial.feed("<1500,120>");
  loop();
  assert(steering.attached() && !stopping && target == 120);
  advance(2300);
  assert(angle > CENTER);
  // Loss of serial heartbeat causes the same return and release.
  advance(2540);
  assert(stopping && motorValue == NEUTRAL && target == CENTER);
  advance(3800);
  assert(angle == CENTER && !steering.attached());
  // Malformed or disallowed packets cannot revive motor or keep watchdog alive.
  Serial.feed("<1570,120,S><9999,90><1500,90,X>");
  loop();
  assert(stopping && motorValue == NEUTRAL && !steering.attached());
}
