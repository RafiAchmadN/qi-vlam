// QI-VLA ESP32 Wi-Fi arm firmware — speaks "qi-vla-bridge/1" directly, so the
// phone app can drive a 6-servo hobby arm (SO-100/SO-101-style geometry or any
// MG996R/SG90 desktop arm) with NO PC in between.
//
// Libraries (Arduino Library Manager): "WebSockets" by Markus Sattler (Links2004),
// "ArduinoJson" v7 by Benoit Blanchon, "ESP32Servo" by Kevin Harrington.
// Board: any ESP32 dev board. NOT compiled/flashed in this repository's CI —
// verify on your hardware, with the arm unpowered first (see README.md).
//
// App: Koneksi -> "QI-VLA Bridge", URL ws://<ESP32-IP>:8765/ws -> Hubungkan -> Armed.
#include <WiFi.h>
#include <WebSocketsServer.h>
#include <ArduinoJson.h>
#include <ESP32Servo.h>

const char* WIFI_SSID = "YOUR_WIFI";       // <-- edit
const char* WIFI_PASS = "YOUR_PASSWORD";   // <-- edit

// ---- calibration: servo_deg = CENTER + SIGN * SCALE * degrees(q) ----------
const int   PINS[6]   = {13, 12, 14, 27, 26, 25};
const float CENTER[5] = {90, 90, 90, 90, 90};
const float SIGN[5]   = {1, 1, 1, 1, 1};
const float SCALE[5]  = {1, 1, 1, 1, 1};
const float GRIP_CLOSED = 30, GRIP_OPEN = 90;
// joint limits (rad), identical to src/robot/arm.py
const float QMIN[6] = {-2.0, -1.6, -0.2, -1.9, -2.8, 0.0};
const float QMAX[6] = { 2.0,  1.6,  2.6,  1.9,  2.8, 1.0};
const float QD_MAX[6] = {1.5, 1.5, 1.5, 2.0, 2.0, 2.0};   // rad/s
const float Q_HOME[6] = {0.0, -0.2, 1.6, 1.1, 0.0, 1.0};
const uint32_t WATCHDOG_MS = 500;   // no command for this long -> hold pose

WebSocketsServer ws(8765);
Servo servo[6];
float qCmd[6], qNow[6];
bool estop = false, holding = true;
int controller = -1;
uint32_t lastCmdMs = 0, lastTickMs = 0, lastStateMs = 0;

float toDeg(int j, float q) {
  if (j == 5) return GRIP_CLOSED + (GRIP_OPEN - GRIP_CLOSED) * constrain(q, 0.0f, 1.0f);
  return constrain(CENTER[j] + SIGN[j] * SCALE[j] * q * 57.29578f, 0.0f, 180.0f);
}

void sendInfo(uint8_t id, const char* role) {
  JsonDocument d;
  d["type"] = "info"; d["protocol"] = "qi-vla-bridge/1"; d["robot"] = "esp32-servo-arm"; d["role"] = role;
  String s; serializeJson(d, s); ws.sendTXT(id, s);
}

void sendError(uint8_t id, const char* msg) {
  JsonDocument d; d["type"] = "error"; d["message"] = msg;
  String s; serializeJson(d, s); ws.sendTXT(id, s);
}

void onWs(uint8_t id, WStype_t type, uint8_t* payload, size_t len) {
  if (type == WStype_CONNECTED) {
    bool ctrl = controller < 0;
    if (ctrl) controller = id;
    sendInfo(id, ctrl ? "controller" : "observer");
  } else if (type == WStype_DISCONNECTED) {
    if (id == controller) { controller = -1; for (int j = 0; j < 6; j++) qCmd[j] = qNow[j]; holding = true; }
  } else if (type == WStype_TEXT) {
    JsonDocument m;
    if (deserializeJson(m, payload, len)) return;
    const char* t = m["type"] | "";
    if (!strcmp(t, "estop")) { estop = true; for (int j = 0; j < 6; j++) qCmd[j] = qNow[j]; holding = true; }
    else if (!strcmp(t, "estop_release") && id == controller) estop = false;
    else if (!strcmp(t, "joint_command")) {
      if (id != controller) { sendError(id, "observer: read-only"); return; }
      if (estop) { sendError(id, "rejected (e-stop latched)"); return; }
      JsonArray q = m["q"];
      if (q.size() != 6) { sendError(id, "rejected (need 6 joints)"); return; }
      for (int j = 0; j < 6; j++) qCmd[j] = constrain((float)q[j], QMIN[j], QMAX[j]);
      lastCmdMs = millis(); holding = false;
    }
  }
}

void setup() {
  Serial.begin(115200);
  for (int j = 0; j < 6; j++) { qCmd[j] = qNow[j] = Q_HOME[j]; servo[j].attach(PINS[j]); servo[j].write(toDeg(j, qNow[j])); }
  WiFi.begin(WIFI_SSID, WIFI_PASS);
  while (WiFi.status() != WL_CONNECTED) { delay(300); Serial.print("."); }
  Serial.printf("\nQI-VLA arm ready: ws://%s:8765/ws\n", WiFi.localIP().toString().c_str());
  ws.begin(); ws.onEvent(onWs);
}

void loop() {
  ws.loop();
  uint32_t now = millis();
  if (now - lastTickMs >= 20) {               // 50 Hz servo update with slew-rate limit
    float dt = (now - lastTickMs) / 1000.0f; lastTickMs = now;
    if (!holding && now - lastCmdMs > WATCHDOG_MS) { for (int j = 0; j < 6; j++) qCmd[j] = qNow[j]; holding = true; }
    for (int j = 0; j < 6; j++) {
      float step = constrain(qCmd[j] - qNow[j], -QD_MAX[j] * dt, QD_MAX[j] * dt);
      qNow[j] += step; servo[j].write(toDeg(j, qNow[j]));
    }
  }
  if (now - lastStateMs >= 60) {              // ~16 Hz state (commanded servo pose; hobby servos have no feedback)
    lastStateMs = now;
    JsonDocument d; d["type"] = "state"; d["estop"] = estop; d["t"] = now;
    JsonArray q = d["q"].to<JsonArray>(); for (int j = 0; j < 6; j++) q.add(qNow[j]);
    String s; serializeJson(d, s); ws.broadcastTXT(s);
  }
}
