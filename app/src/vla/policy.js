// Runs the exported (dense or TTNS-compressed) reach policy on-device.
// Format: qi-vla-policy/1, written by src/robot/train_policy.py.
import { fk } from "../robot/kinematics.js";

const gelu = (x) => 0.5 * x * (1 + Math.tanh(0.7978845608028654 * (x + 0.044715 * x * x * x)));

function matvec(W, x) { // W: out x in
  const out = new Float64Array(W.length);
  for (let i = 0; i < W.length; i++) {
    const row = W[i];
    let s = 0;
    for (let j = 0; j < row.length; j++) s += row[j] * x[j];
    out[i] = s;
  }
  return out;
}

export function forwardLayers(layers, x) {
  let h = Float64Array.from(x);
  for (const L of layers) {
    let y;
    if (L.type === "dense") y = matvec(L.W, h);
    else if (L.type === "ttns") y = matvec(L.left, matvec(L.root, matvec(L.right, h))); // (left·root·right) x
    else throw new Error(`unknown layer ${L.type}`);
    for (let i = 0; i < y.length; i++) {
      y[i] += L.b[i];
      if (L.act === "gelu_tanh") y[i] = gelu(y[i]);
      else if (L.act === "tanh") y[i] = Math.tanh(y[i]);
    }
    h = y;
  }
  return Array.from(h);
}

export function features(q, target) {
  const ee = fk(q);
  return [q[0], q[1], q[2], q[3], target[0], target[1], target[2],
    (target[0] - ee[0]) * 5, (target[1] - ee[1]) * 5, (target[2] - ee[2]) * 5, 1];
}

export class ReachPolicy {
  constructor(spec) {
    if (spec.format !== "qi-vla-policy/1") throw new Error("unsupported policy format");
    this.spec = spec;
    this.name = spec.method;
    this.params = spec.params;
  }
  static async load(url) {
    const r = await fetch(url);
    if (!r.ok) throw new Error(`policy fetch failed: ${r.status}`);
    return new ReachPolicy(await r.json());
  }
  /** joint velocities (rad/s) for q[0..3] */
  act(q, target) {
    const y = forwardLayers(this.spec.layers, features(q, target));
    return y.map((v, i) => v * this.spec.qd_max[i]);
  }
}

/** Optional remote policy (server-side inference over HTTP, e.g. a bigger VLA). */
export class RemotePolicy {
  constructor(baseUrl) { this.baseUrl = baseUrl.replace(/\/$/, ""); this.name = "remote"; this.params = null; }
  async act(q, target, instruction = "") {
    const r = await fetch(`${this.baseUrl}/act`, {
      method: "POST", headers: { "content-type": "application/json" },
      body: JSON.stringify({ q, target, instruction }),
    });
    if (!r.ok) throw new Error(`remote policy ${r.status}`);
    return (await r.json()).qd;
  }
}
