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
  constructor() { this.stateCb = () => {}; this.statusCb = () => {}; this.sceneCb = () => {}; this.roverCb = () => {}; this.ws = null; this.robotInfo = null; }
  /** mobile robots: cb({pose:[x,y,yaw] (odom frame), vw:[v,w], scan:[36 ranges] | null}) */
  onRoverState(cb) { this.roverCb = cb; }
  onState(cb) { this.stateCb = cb; }
  /** simulated robots (bridge --driver mujoco) also stream object poses */
  onScene(cb) { this.sceneCb = cb; }
  get isSim() { return !!this.robotInfo?.sim; }
  sceneReset() {}
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
    else if (m.type === "scene") this.sceneCb(m);
    else if (m.type === "rover_state") this.roverCb({ pose: m.pose, vw: m.vw, scan: m.scan || null, map: m.map });
    else if (m.type === "info") { this.robotInfo = m; this.statusCb(`terhubung: ${m.robot || "robot"}${m.sim ? ` (${m.env || m.map})` : ""}`, true); }
    else if (m.type === "error") this.statusCb(`robot: ${m.message}`, false);
  }
  sendJoints(q) { this._send({ type: "joint_command", q, seq: this.seq++, t: Date.now() }); }
  estop() { this._send({ type: "estop" }); }
  releaseEstop() { this._send({ type: "estop_release" }); }
  sceneReset(env, seed) { this._send({ type: "scene_reset", env, seed }); }
  sendTwist(v, w) { this._send({ type: "twist_command", v, w, seq: this.seq++ }); }
}

export class RosbridgeTransport extends Base {
  constructor(url, { cmdTopic = "/arm_controller/joint_trajectory", stateTopic = "/joint_states", jointNames = JOINT_NAMES, horizon = 0.1,
    rover = false, cmdVelTopic = "/cmd_vel", odomTopic = "/odom", scanTopic = "/scan", beams = 36 } = {}) {
    super();
    Object.assign(this, { url, cmdTopic, stateTopic, jointNames, horizon, rover, cmdVelTopic, odomTopic, scanTopic, beams, lastScan: null });
  }
  connect() {
    return this._open(this.url, () => {
      if (this.rover) {
        // TurtleBot3 / Nav2 conventions (ROS 2 Humble: geometry_msgs/Twist on /cmd_vel)
        this._send({ op: "advertise", topic: this.cmdVelTopic, type: "geometry_msgs/msg/Twist" });
        this._send({ op: "subscribe", topic: this.odomTopic, type: "nav_msgs/msg/Odometry", throttle_rate: 50 });
        this._send({ op: "subscribe", topic: this.scanTopic, type: "sensor_msgs/msg/LaserScan", throttle_rate: 100 });
        return;
      }
      this._send({ op: "advertise", topic: this.cmdTopic, type: "trajectory_msgs/msg/JointTrajectory" });
      this._send({ op: "subscribe", topic: this.stateTopic, type: "sensor_msgs/msg/JointState", throttle_rate: 50 });
    });
  }
  /** LaserScan (any angle_min/increment/size) -> `beams` ranges at robot-frame angles k*2pi/beams */
  resampleScan(msg) {
    const out = new Array(this.beams);
    const n = msg.ranges.length;
    for (let k = 0; k < this.beams; k++) {
      const a = (k * 2 * Math.PI) / this.beams;
      let i = Math.round((((a - msg.angle_min) % (2 * Math.PI)) + 2 * Math.PI) % (2 * Math.PI) / msg.angle_increment);
      i = Math.min(n - 1, Math.max(0, i));
      const r = msg.ranges[i];
      out[k] = Number.isFinite(r) && r >= msg.range_min ? Math.min(r, msg.range_max) : msg.range_max;
    }
    return out;
  }
  _onMessage(m) {
    if (this.rover && m.op === "publish" && m.msg) {
      if (m.topic === this.scanTopic) { this.lastScan = this.resampleScan(m.msg); return; }
      if (m.topic === this.odomTopic) {
        const p = m.msg.pose.pose, q = p.orientation, tw = m.msg.twist.twist;
        const yaw = Math.atan2(2 * (q.w * q.z + q.x * q.y), 1 - 2 * (q.y * q.y + q.z * q.z));
        this.roverCb({ pose: [p.position.x, p.position.y, yaw], vw: [tw.linear.x, tw.angular.z], scan: this.lastScan });
      }
      return;
    }
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
  sendTwist(v, w) {
    this._send({ op: "publish", topic: this.cmdVelTopic, msg: { linear: { x: v, y: 0, z: 0 }, angular: { x: 0, y: 0, z: w } } });
  }
  // ROS has no generic e-stop topic: hold the last MEASURED pose (stops motion
  // for position-controlled arms) / publish zero velocity for mobile bases.
  // Wire your robot's real e-stop separately.
  estop() { if (this.rover) this.sendTwist(0, 0); else if (this.lastMeasured) this.sendJoints(this.lastMeasured); }
  releaseEstop() {}
}
