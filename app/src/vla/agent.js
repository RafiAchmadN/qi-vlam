// Skill executor: turns a grounded plan into a stream of end-effector
// sub-goals. Each control tick: perception-updated goal -> learned policy ->
// safety shield -> joint command. Object positions come from the vision
// module, re-queried at the start of every skill (closed-loop grounding).
import { fk, dist, Q_HOME } from "../robot/kinematics.js";
import { CUBE } from "../robot/scene.js";

const ABOVE = 0.09, LIFT = 0.12, TOL = 0.008, SUBGOAL_TIMEOUT = 8.0;

export class Agent {
  constructor({ perceive, log }) {
    this.perceive = perceive; // () => detections {color:{x,y}}
    this.log = log;
    this.queue = []; this.current = null; this.status = "idle";
    this.memory = {}; // last good detection per object (the arm occludes objects while descending)
  }
  start(plan) { this.queue = this.expand(plan); this.current = null; this.memory = {}; this.status = this.queue.length ? "running" : "idle"; }
  stop() { this.queue = []; this.current = null; this.status = "idle"; }

  // plan skills -> primitive steps (resolved lazily so perception is fresh)
  expand(plan) {
    const steps = [];
    for (const s of plan) {
      if (s.skill === "pick") {
        steps.push({ kind: "gripper", open: true, label: "buka gripper" });
        steps.push({ kind: "move", ref: s.object, dz: ABOVE, align: s.object, label: `di atas ${s.object}` });
        steps.push({ kind: "move", ref: s.object, dz: 0, grasp: true, useMemory: true, align: s.object, label: `turun ke ${s.object}` });
        steps.push({ kind: "gripper", open: false, attach: s.object, label: "genggam" });
        steps.push({ kind: "move", ref: null, lift: LIFT, label: "angkat" });
      } else if (s.skill === "place") {
        steps.push({ kind: "move", ref: s.target, dz: ABOVE, label: `di atas ${s.target}` });
        steps.push({ kind: "move", ref: s.target, dz: s.onCube ? CUBE + 0.004 : CUBE / 2 + 0.004, place: true, useMemory: true, label: `turunkan di ${s.target}` });
        steps.push({ kind: "gripper", open: true, release: true, label: "lepas" });
        steps.push({ kind: "move", ref: null, lift: LIFT, label: "mundur ke atas" });
      } else if (s.skill === "reach") {
        steps.push({ kind: "move", ref: s.target, dz: 0.05, label: `dekati ${s.target}` });
      } else if (s.skill === "home") {
        steps.push({ kind: "move", fixed: fk(Q_HOME), label: "posisi home" });
        steps.push({ kind: "gripper", open: true, label: "buka gripper" });
      } else if (s.skill === "gripper") {
        steps.push({ kind: "gripper", open: s.open, label: s.open ? "buka gripper" : "tutup gripper" });
      }
    }
    return steps;
  }

  resolveGoal(step, q, world) {
    if (step.fixed) return step.fixed;
    if (step.lift) { const p = fk(q); return [p[0], p[1], step.lift]; }
    let d = step.useMemory ? this.memory[step.ref] : null;
    if (!d) {
      d = this.perceive()[step.ref];
      if (!d) return null;
      this.memory[step.ref] = d;
    }
    let z;
    if (step.grasp) z = CUBE / 2;                         // grasp at cube center
    else if (step.place) z = step.dz;                     // already absolute height above support
    else z = step.dz + (world.kindOf(step.ref) === "cube" ? CUBE : 0);
    return [d.x, d.y, z];
  }

  /** advance one tick; returns {goal, gripper} or null if idle */
  tick(q, dt, world) {
    if (this.status !== "running") return null;
    if (!this.current) {
      const step = this.queue.shift();
      if (!step) { this.status = "done"; this.log("info", "✔ instruksi selesai"); return null; }
      step.t = 0;
      if (step.kind === "move") {
        step.goal = this.resolveGoal(step, q, world);
        if (!step.goal) { this.fail(`objek '${step.ref}' tidak terdeteksi kamera`); return null; }
      }
      this.current = step;
      this.log("step", `→ ${step.label}`);
    }
    const s = this.current;
    s.t += dt;
    if (s.kind === "gripper") {
      const g = s.open ? 1 : 0;
      if (s.t >= 0.6) {
        if (s.attach) world.tryAttach(s.attach);
        if (s.release) world.release();
        this.current = null;
      }
      return { goal: null, gripper: g };
    }
    if (dist(fk(q), s.goal) < TOL) { this.current = null; return { goal: s.goal, gripper: null, align: s.align }; }
    if (s.t > SUBGOAL_TIMEOUT) { this.fail(`timeout saat '${s.label}'`); return null; }
    return { goal: s.goal, gripper: null, align: s.align };
  }

  fail(msg) { this.log("error", `✖ ${msg}`); this.stop(); this.status = "failed"; }
}
