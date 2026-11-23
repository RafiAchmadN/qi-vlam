// Demonstration recorder: every executed pick-and-place instruction (by the
// skill agent, the end-to-end policy, or a human via the joint sliders)
// becomes a training episode in format qi-vla-demos/1. Convert with
// scripts/import_app_demos.py into the same LeRobot-style dataset the
// MuJoCo collector writes, then train with src/sim/train_pickplace.py.
//
// env_state layout (32) == src/sim/envs.py::env_state_vector:
//   obj xyz yaw (4) | goal support xyz (3) | goal_is_cube (1) |
//   objects blue, cyan, green, orange, red xyz yaw (20, zeros if absent) | pads purple, yellow xy (4)

export const CUBE_ORDER = ["blue", "cyan", "green", "orange", "red"];
export const PAD_ORDER = ["purple", "yellow"];

export function envState(objects, obj, goal, goalKind) {
  const pose = (n) => objects[n] ? [...objects[n].pos, objects[n].yaw || 0] : [0, 0, 0, 0];
  const o = pose(obj);
  let g = [0, 0, 0];
  if (objects[goal]) g = goalKind === "cube" ? [objects[goal].pos[0], objects[goal].pos[1], objects[goal].pos[2] + 0.015] : [objects[goal].pos[0], objects[goal].pos[1], 0];
  const out = [...o, ...g, goalKind === "cube" ? 1 : 0];
  for (const c of CUBE_ORDER) out.push(...pose(c));
  for (const p of PAD_ORDER) out.push(...(objects[p] ? objects[p].pos.slice(0, 2) : [0, 0]));
  return out;
}

export class Recorder {
  constructor() { this.enabled = false; this.episodes = []; this.cur = null; }
  begin(meta) { if (this.enabled) this.cur = { ...meta, frames: [], t0: Date.now() }; }
  frame(f) { if (this.cur && this.cur.frames.length < 2000) this.cur.frames.push(f); }
  end(success) {
    if (!this.cur) return null;
    const ep = { ...this.cur, success: !!success, duration_s: (Date.now() - this.cur.t0) / 1000 };
    delete ep.t0;
    this.cur = null;
    if (ep.frames.length >= 5) this.episodes.push(ep);
    return ep;
  }
  get active() { return !!this.cur; }
  clear() { this.episodes = []; this.cur = null; }
  payload(mode) {
    return { format: "qi-vla-demos/1", app: "qi-vla-robot", mode, created: new Date().toISOString(), fps: 20, episodes: this.episodes };
  }
  stats() {
    const n = this.episodes.length, ok = this.episodes.filter((e) => e.success).length;
    return { n, ok, frames: this.episodes.reduce((s, e) => s + e.frames.length, 0) };
  }
}
