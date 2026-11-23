// End-to-end pick-and-place policy trained in MuJoCo physics on the
// simulated dataset (src/sim/train_pickplace.py). Unlike the skill agent
// (which sequences sub-goals and uses the small reach policy), this one model
// decides the whole motion and when to open/close the gripper, from:
//   joints, gripper, tip, cube position, place position (from the grounded
//   instruction) — features identical to src/sim/features.py (parity-tested).
import { fk, QD_MAX, Q_HOME } from "../robot/kinematics.js";
import { forwardLayers } from "./policy.js";

export const CUBE = 0.03;
export const MAX_WINDUP = 0.25;

export function ppFeatures(q, obj, goal, goalIsCube) {
  const tip = fk(q);
  const place = [goal[0], goal[1], goal[2] + CUBE / 2];
  return [q[0], q[1], q[2], q[3], q[5], tip[0], tip[1], tip[2],
    5 * (obj[0] - tip[0]), 5 * (obj[1] - tip[1]), 5 * (obj[2] - tip[2]),
    5 * (place[0] - tip[0]), 5 * (place[1] - tip[1]), 5 * (place[2] - tip[2]),
    10 * (obj[2] - CUBE / 2), goalIsCube ? 1 : 0, 1];
}

/** Wrist roll that makes the fingers close on two cube faces (see src/sim/expert.py). */
export function rollForYaw(q0, yaw) {
  const a = q0 - yaw + Math.PI / 4;
  const m = Math.PI / 2;
  return ((a % m) + m) % m - Math.PI / 4;
}

export class PickPlacePolicy {
  constructor(spec) {
    if (spec.format !== "qi-vla-policy/1" || spec.kind !== "pickplace") throw new Error("not a pick-place policy");
    this.spec = spec; this.name = spec.method; this.params = spec.params;
  }
  static async load(url) {
    const r = await fetch(url);
    if (!r.ok) throw new Error(`policy fetch failed: ${r.status}`);
    return new PickPlacePolicy(await r.json());
  }
  /** -> {qd: rad/s for joints 0..3, gripper: 0 closed | 1 open} */
  act(q, obj, goal, goalIsCube) {
    const y = forwardLayers(this.spec.layers, ppFeatures(q, obj, goal, goalIsCube));
    return { qd: y.slice(0, 4).map((v, i) => Math.max(-1, Math.min(1, v)) * QD_MAX[i]), gripper: y[4] > 0 ? 1 : 0, raw: y };
  }
}

/**
 * Runs one pick-and-place instruction with the end-to-end policy.
 * world: { objectPose(name) -> {pos:[x,y,z], yaw} | null, goalSupport(name, kind) -> [x,y,z] | null }
 */
export class PickPlaceRunner {
  constructor(policy, { log = () => {}, timeout = 30 } = {}) {
    this.policy = policy; this.log = log; this.timeout = timeout; this.status = "idle";
  }
  start(obj, goal, goalKind) {
    Object.assign(this, { obj, goal, goalKind, t: 0, stable: 0, roll: null, status: "running" });
    this.log("step", `→ policy end-to-end: ${obj} → ${goal} (${goalKind})`);
  }
  stop() { this.status = "idle"; }
  /** returns {qd, gripper, roll} or null when not running */
  tick(q, dt, world) {
    if (this.status !== "running") return null;
    this.t += dt;
    const o = world.objectPose(this.obj), g = world.goalSupport(this.goal, this.goalKind);
    if (!o || !g) { this.fail(`objek ${o ? this.goal : this.obj} tidak ditemukan`); return null; }
    const a = this.policy.act(q, o.pos, g, this.goalKind === "cube");
    if (a.gripper > 0.5 && o.pos[2] < CUBE / 2 + 0.01) this.roll = rollForYaw(q[0], o.yaw || 0);
    const done = world.isSuccess(this.obj, this.goal, this.goalKind);
    this.stable = done ? this.stable + 1 : 0;
    if (this.stable >= 10) { this.status = "done"; this.log("info", "✔ instruksi selesai (policy end-to-end)"); return null; }
    if (this.t > this.timeout) { this.fail("timeout policy end-to-end"); return null; }
    return { qd: a.qd, gripper: a.gripper, roll: this.roll };
  }
  fail(msg) { this.status = "failed"; this.log("error", `✖ ${msg}`); }
}

export { Q_HOME };
