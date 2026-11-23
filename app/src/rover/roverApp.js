// Mobile-robot mode of the app: language goal -> place, A* on a lidar-updated
// costmap, local controller (learned dense/TTNS policy trained in MuJoCo, or
// the pure-pursuit expert), lidar barrier shield, kinematic simulation or a
// live robot (bridge `--robot rover`, or ROS /cmd_vel + /odom + /scan).
import { loadMapIndex, loadMap, buildGrid, NavExpert, RoverModel, navFeatures, shieldVW, groundPlace, sampleNavScene, wrap } from "./rover.js";
import { forwardLayers } from "../vla/policy.js";
import { rng } from "../robot/envs.js";

export class NavPolicy {
  constructor(spec) {
    if (spec.format !== "qi-vla-policy/1" || spec.kind !== "nav") throw new Error("not a navigation policy");
    Object.assign(this, { spec, name: spec.method, params: spec.params });
  }
  static async load(url) { const r = await fetch(url); if (!r.ok) throw new Error(`nav policy ${r.status}`); return new NavPolicy(await r.json()); }
  act(robot, pose, scan, target, goal, vw) {
    const y = forwardLayers(this.spec.layers, navFeatures(robot, pose, scan, target, goal, vw));
    return { v: Math.max(-1, Math.min(1, y[0])) * robot.v_max, w: Math.max(-1, Math.min(1, y[1])) * robot.w_max };
  }
}

export class RoverApp {
  constructor({ log }) {
    Object.assign(this, { log, index: null, map: null, grid: null, robot: null, model: null, expert: null, goal: null,
      place: null, t: 0, status: "idle", policies: {}, policyKey: "nav_ttns", shieldOn: true, unknownCount: 0,
      unknown: [], live: null, liveOrigin: null, measured: null, lastScan: null, lastFront: null, source: "-", lat: 0, stable: 0,
      stats: { shieldSteps: 0, steps: 0 } });
  }
  async init() {
    this.index = await loadMapIndex();
    this.robot = this.index.robot;
    for (const k of ["ttns", "dense"]) {
      try { this.policies[`nav_${k}`] = await NavPolicy.load(`./models/nav_${k}.json`); }
      catch { this.log("info", `policy navigasi ${k} belum ada (latih: python -m src.sim.mobile.train_nav train)`); }
    }
    await this.setMap(this.index.default);
  }
  async setMap(id, seed = 7) {
    this.map = await loadMap(id);
    this.grid = buildGrid(this.map, this.robot.radius + 0.06);
    this.reset(seed);
  }
  reset(seed = Date.now()) {
    const s = sampleNavScene(this.map, this.grid, this.robot, rng(seed), null, this.unknownCount);
    this.unknown = s.unknown;
    this.model = new RoverModel(this.robot, this.map, s.start, this.unknown);
    this.status = "idle"; this.expert = null; this.goal = null; this.place = null;
    this.log("info", `peta "${this.map.name}": robot di (${s.start[0].toFixed(2)}, ${s.start[1].toFixed(2)})${this.unknown.length ? `, ${this.unknown.length} rintangan tak dikenal (tidak ada di peta)` : ""}`);
  }
  run(text) {
    const p = groundPlace(text, this.map);
    if (!p) { this.log("error", `tempat tidak dikenal — coba: ${this.map.places.map((q) => q.names_id[0]).join(", ")}`); return false; }
    this.place = p; this.goal = p.pos.slice();
    this.expert = new NavExpert(this.robot, this.grid, this.goal);
    this.status = "running"; this.t = 0; this.stable = 0; this.stats = { shieldSteps: 0, steps: 0 };
    this.log("info", `rencana: navigasi ke ${p.names_id[0]} (${p.pos.map((v) => v.toFixed(2)).join(", ")}) · A* + ${this.policyKey === "expert" ? "pure pursuit" : this.policyKey}`);
    return true;
  }
  stop() { this.status = "idle"; this.sendZero(); }
  sendZero() { if (this.live?.connected && this.live.armed) this.live.transport.sendTwist(0, 0); }

  /** live robot pose in MAP frame: odometry composed with the pose the robot had in the app at connect time */
  onLiveState(st) {
    if (st.map) { // simulated rover on the bridge: pose is already in the map frame
      this.measured = st.pose.slice();
      if (st.scan) this.lastScan = st.scan;
      return;
    }
    if (!this.liveOrigin) { this.liveOrigin = { map: this.model.pose.slice(), odom: st.pose.slice() }; this.log("info", "pose robot nyata disinkronkan ke pose di peta (odom → map)"); }
    const o = this.liveOrigin, d = [st.pose[0] - o.odom[0], st.pose[1] - o.odom[1]], r = o.map[2] - o.odom[2];
    this.measured = [o.map[0] + Math.cos(r) * d[0] - Math.sin(r) * d[1], o.map[1] + Math.sin(r) * d[0] + Math.cos(r) * d[1], wrap(st.pose[2] + r)];
    if (st.scan) this.lastScan = st.scan;
  }
  pose() { return this.live?.connected && this.measured ? this.measured : this.model.pose; }
  scan() { return this.live?.connected && this.lastScan ? this.lastScan : this.model.scan(); }

  tick(dt, estop) {
    const pose = this.pose(), scan = this.scan();
    this.curScan = scan;
    let v = 0, w = 0;
    if (this.status === "running" && !estop) {
      this.t += dt;
      const t0 = performance.now();
      const ex = this.expert.act(pose, scan, this.t);
      if (this.policyKey === "expert" || !this.policies[this.policyKey]) { ({ v, w } = ex); this.source = "pure pursuit"; }
      else { ({ v, w } = this.policies[this.policyKey].act(this.robot, pose, scan, ex.target, this.goal, this.model.last)); this.source = this.policyKey; }
      this.lat = performance.now() - t0;
      if (this.shieldOn) {
        const s = shieldVW(this.robot, v, w, scan);
        if (s.clipped) { this.stats.shieldSteps++; this.source = "shield lidar"; }
        v = s.v; w = s.w; this.lastFront = s.front;
      }
      this.stats.steps++;
      const d = Math.hypot(pose[0] - this.goal[0], pose[1] - this.goal[1]);
      this.stable = d < 0.2 ? this.stable + 1 : 0;
      if (this.stable >= 5) { this.status = "done"; v = 0; w = 0; this.log("info", `✔ sampai di ${this.place.names_id[0]} dalam ${this.t.toFixed(1)} s (shield aktif ${(100 * this.stats.shieldSteps / Math.max(1, this.stats.steps)).toFixed(0)}% langkah)`); }
      else if (this.t > 90) { this.status = "failed"; v = 0; w = 0; this.log("error", "✖ timeout navigasi"); }
    }
    this.model.step(v, w, dt);  // sim mode: the robot; live mode: keeps last command for features
    if (this.live?.connected && this.live.armed && !estop) this.live.transport.sendTwist(v, w);
    return { v, w };
  }
  distToGoal() { const p = this.pose(); return this.goal ? Math.hypot(p[0] - this.goal[0], p[1] - this.goal[1]) : null; }
}
