// Live robot connections. All transports expose the same interface:
//   connect() -> Promise, close(), sendJoints(q[6]), estop(), onState(cb(q)), onStatus(cb(text, ok))
//
// 1. BridgeTransport   — "qi-vla-bridge/1" JSON over WebSocket. Spoken by the
//    Python bridge (src/bridge/server.py: mock / USB-serial servo arms) AND by
//    the ESP32 Wi-Fi firmware (firmware/esp32_wifi_arm), so a phone can drive
//    a hobby arm directly with no PC.
// 2. RosbridgeTransport — rosbridge v2 protocol (ROS 1/ROS 2 rosbridge_suite):
//    publishes trajectory_msgs/JointTrajectory, subscribes sensor_msgs/JointState.
//    Works with any ros2_control arm (SO-101/LeRobot ROS drivers, myCobot, UR, ...).
import { JOINT_NAMES } from "../robot/kinematics.js";

class Base {
  constructor() { this.stateCb = () => {}; this.statusCb = () => {}; this.ws = null; this.robotInfo = null; }
  onState(cb) { this.stateCb = cb; }
  onStatus(cb) { this.statusCb = cb; }
  get connected() { return !!this.ws && this.ws.readyState === 1; }
  _open(url, onOpen) {
    return new Promise((resolve, reject) => {
      let ws;
      try { ws = new WebSocket(url); } catch (e) { reject(e); return; }
      this.ws = ws;
      const timer = setTimeout(() => { ws.close(); reject(new Error("timeout koneksi (4 s)")); }, 4000);
      ws.onopen = () => { clearTimeout(timer); onOpen(); this.statusCb("terhubung", true); resolve(); };
      ws.onerror = () => { clearTimeout(timer); this.statusCb("error koneksi", false); reject(new Error("websocket error")); };
      ws.onclose = () => { this.statusCb("terputus", false); };
      ws.onmessage = (ev) => { try { this._onMessage(JSON.parse(ev.data)); } catch { /* ignore non-JSON */ } };
    });
  }
  close() { if (this.ws) this.ws.close(); this.ws = null; }
  _send(obj) { if (this.connected) this.ws.send(JSON.stringify(obj)); }
}

export class BridgeTransport extends Base {
  constructor(url) { super(); this.url = url; this.seq = 0; }
  connect() { return this._open(this.url, () => this._send({ type: "hello", protocol: "qi-vla-bridge/1", client: "qi-vla-app" })); }
  _onMessage(m) {
    if (m.type === "state" && Array.isArray(m.q)) this.stateCb(m.q);
    else if (m.type === "info") { this.robotInfo = m; this.statusCb(`terhubung: ${m.robot || "robot"}`, true); }
    else if (m.type === "error") this.statusCb(`robot: ${m.message}`, false);
  }
  sendJoints(q) { this._send({ type: "joint_command", q, seq: this.seq++, t: Date.now() }); }
  estop() { this._send({ type: "estop" }); }
  releaseEstop() { this._send({ type: "estop_release" }); }
}

export class RosbridgeTransport extends Base {
  constructor(url, { cmdTopic = "/arm_controller/joint_trajectory", stateTopic = "/joint_states", jointNames = JOINT_NAMES, horizon = 0.1 } = {}) {
    super();
    Object.assign(this, { url, cmdTopic, stateTopic, jointNames, horizon });
  }
  connect() {
    return this._open(this.url, () => {
      this._send({ op: "advertise", topic: this.cmdTopic, type: "trajectory_msgs/msg/JointTrajectory" });
      this._send({ op: "subscribe", topic: this.stateTopic, type: "sensor_msgs/msg/JointState", throttle_rate: 50 });
    });
  }
  _onMessage(m) {
    if (m.op !== "publish" || m.topic !== this.stateTopic || !m.msg) return;
    const { name, position } = m.msg;
    if (!name || !position) return;
    const q = this.jointNames.map((n) => { const i = name.indexOf(n); return i >= 0 ? position[i] : 0; });
    this.lastMeasured = q;
    this.stateCb(q);
  }
  sendJoints(q) {
    const sec = Math.floor(this.horizon), nsec = Math.round((this.horizon - sec) * 1e9);
    this._send({
      op: "publish", topic: this.cmdTopic,
      msg: { joint_names: this.jointNames, points: [{ positions: q.slice(0, this.jointNames.length), velocities: [], accelerations: [], effort: [], time_from_start: { sec, nanosec: nsec } }] },
    });
  }
  // ROS has no generic e-stop topic: hold the last MEASURED pose (stops motion
  // for position-controlled arms). Wire your robot's real e-stop separately.
  estop() { if (this.lastMeasured) this.sendJoints(this.lastMeasured); }
  releaseEstop() {}
}
