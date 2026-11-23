// QI-VLA USB-serial arm firmware (Arduino Uno/Nano/Mega or ESP32) for
// `python -m src.bridge.server --driver serial --port COM3 [--cal calib.json]`.
// Protocol: one JSON object per line.
//   host -> board: {"cmd":"q","deg":[d0..d5]}  |  {"cmd":"stop"}
//   board -> host: {"deg":[d0..d5]}   (last commanded, slew-limited servo angles)
// Calibration (radians -> degrees) lives on the PC side (SerialDriver), so this
// sketch only slews servos safely. Libraries: ArduinoJson v7, Servo.
// NOT compiled in this repository's CI — verify on your hardware first.
#include <ArduinoJson.h>
#include <Servo.h>

const int PINS[6] = {3, 5, 6, 9, 10, 11};
const float MAX_DEG_PER_S = 90.0;     // slew limit per servo
const unsigned long WATCHDOG_MS = 500;

Servo servo[6];
float cmd[6] = {90, 90, 90, 90, 90, 90}, now_[6] = {90, 90, 90, 90, 90, 90};
unsigned long lastCmd = 0, lastTick = 0, lastReport = 0;

void setup() {
  Serial.begin(115200);
  for (int j = 0; j < 6; j++) { servo[j].attach(PINS[j]); servo[j].write((int)now_[j]); }
}

void loop() {
  if (Serial.available()) {
    String line = Serial.readStringUntil('\n');
    JsonDocument m;
    if (!deserializeJson(m, line)) {
      const char* c = m["cmd"] | "";
      if (!strcmp(c, "q") && m["deg"].size() == 6) {
        for (int j = 0; j < 6; j++) cmd[j] = constrain((float)m["deg"][j], 0.0f, 180.0f);
        lastCmd = millis();
      } else if (!strcmp(c, "stop")) {
        for (int j = 0; j < 6; j++) cmd[j] = now_[j];
      }
    }
  }
  unsigned long t = millis();
  if (t - lastTick >= 20) {
    float dt = (t - lastTick) / 1000.0; lastTick = t;
    if (t - lastCmd > WATCHDOG_MS) for (int j = 0; j < 6; j++) cmd[j] = now_[j];
    for (int j = 0; j < 6; j++) {
      now_[j] += constrain(cmd[j] - now_[j], -MAX_DEG_PER_S * dt, MAX_DEG_PER_S * dt);
      servo[j].write((int)(now_[j] + 0.5));
    }
  }
  if (t - lastReport >= 50) {
    lastReport = t;
    Serial.print("{\"deg\":[");
    for (int j = 0; j < 6; j++) { Serial.print(now_[j], 1); if (j < 5) Serial.print(','); }
    Serial.println("]}");
  }
}
