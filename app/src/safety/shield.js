// Runtime assurance for the learned policy (Simplex architecture):
//   learned controller (compressed VLA policy)  --\
//                                                  >-- switch --> joint command
//   verified fallback (damped-least-squares IK) --/      ^
//                                                         |
//   monitor: limits, workspace, Lyapunov V(e) = e^T P e --+
//
// P comes from the SDP certificate solved in Python for the identified
// closed-loop error model (formal for that linear model only). Online, a
// sustained rise of V means the learned policy is not behaving like the
// certified model, so control is handed to the fallback until V decreases.
import { QD_MAX, Q_MIN, Q_MAX, WORKSPACE_MIN, WORKSPACE_MAX, fk, ikStep } from "../robot/kinematics.js";

export class SafetyShield {
  constructor({ P, dt = 0.05, riseTolerance = 0.005, patience = 4, recover = 10,
    excursion = 1.3, vFloor = 1e-6 } = {}) {
    this.P = P || [[1, 0, 0], [0, 1, 0], [0, 0, 1]];
    this.dt = dt;
    this.riseTolerance = riseTolerance; // relative per-step rise counted as a violation
    this.patience = patience;           // consecutive violations before fallback
    this.recover = recover;             // consecutive decreases before handing back
    this.excursion = excursion;         // V > excursion * min(V) since goal start -> fallback
    this.vFloor = vFloor;               // ignore V below ~(1 mm)^2: sensor/quantization noise
    this.reset();
    this.estop = false;
    this.events = [];
  }
  reset() {
    this.prevV = null; this.rises = 0; this.fallback = false; this.calm = 0; this.lastV = 0; this.minV = Infinity;
  }
  V(e) {
    const P = this.P;
    let s = 0;
    for (let i = 0; i < 3; i++) for (let j = 0; j < 3; j++) s += e[i] * P[i][j] * e[j];
    return s;
  }
  log(kind, msg) {
    this.events.push({ t: Date.now(), kind, msg });
    if (this.events.length > 200) this.events.shift();
  }
  /** Returns {qd, source, V, clipped} — qd for joints 0..3. */
  filter(q, target, qdLearned) {
    if (this.estop) return { qd: [0, 0, 0, 0], source: "estop", V: this.lastV, clipped: true };
    const ee = fk(q);
    const e = [target[0] - ee[0], target[1] - ee[1], target[2] - ee[2]];
    const V = this.V(e);
    // Lyapunov monitor (decrease condition, with tolerance for near-zero V)
    const active = V > this.vFloor;
    if (active && this.prevV !== null && V > this.prevV * (1 + this.riseTolerance)) this.rises++;
    else this.rises = 0;
    const excursed = active && V > this.excursion * this.minV;
    this.minV = Math.min(this.minV, V);
    if (!this.fallback && (this.rises >= this.patience || excursed)) {
      this.fallback = true; this.calm = 0;
      this.log("fallback", excursed
        ? "V melampaui 1.3× minimumnya → kontrol diambil IK fallback"
        : `V naik ${this.patience}x berturut-turut → kontrol diambil IK fallback`);
    }
    if (this.fallback) {
      this.calm = V < (this.prevV ?? V) ? this.calm + 1 : 0;
      if (this.calm >= this.recover) {
        this.fallback = false; this.rises = 0;
        this.log("recover", "V turun stabil → kontrol kembali ke policy terkompresi");
      }
    }
    this.prevV = V; this.lastV = V;
    let qd = this.fallback || !qdLearned ? ikStep(q, target) : qdLearned.slice(0, 4);
    let clipped = false;
    // velocity limits
    qd = qd.map((v, i) => {
      const c = Math.max(-QD_MAX[i], Math.min(QD_MAX[i], v));
      if (c !== v || !Number.isFinite(v)) clipped = true;
      return Number.isFinite(c) ? c : 0;
    });
    // joint limits + workspace (table) — scale the step down until it is valid
    for (let k = 0; k < 6; k++) {
      const qn = q.slice();
      for (let i = 0; i < 4; i++) qn[i] = Math.max(Q_MIN[i], Math.min(Q_MAX[i], q[i] + qd[i] * this.dt));
      const p = fk(qn);
      const below = p[2] < WORKSPACE_MIN[2];
      const outside = p.some((v, i) => v < WORKSPACE_MIN[i] - 0.03 || v > WORKSPACE_MAX[i] + 0.03);
      if (!below && !outside) break;
      clipped = true;
      qd = qd.map((v) => v * 0.5);
      if (k === 5) { qd = [0, 0, 0, 0]; this.log("block", "langkah ditolak: keluar workspace / tabrak meja"); }
    }
    return { qd, source: this.fallback || !qdLearned ? "fallback-ik" : "policy", V, clipped };
  }
}
