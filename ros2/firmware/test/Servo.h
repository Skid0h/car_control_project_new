#pragma once
class Servo {
 public:
  bool enabled = false;
  int last = 90;
  int pulse = 1500;
  int attach(int, int = 544, int = 2400) { enabled = true; return 1; }
  void detach() { enabled = false; }
  bool attached() const { return enabled; }
  void write(int value) { last = value; }
  void writeMicroseconds(int value) { pulse = value; }
};
