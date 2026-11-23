// Node built-in test runner: `npm test` (no browser, no WebGL needed).
import { test } from "node:test";
import assert from "node:assert/strict";
import { readFileSync } from "node:fs";
import { fk, ikStep, jacobian, dist, Q_HOME, clampQ } from "../src/robot/kinematics.js";
import { forwardLayers, ReachPolicy } from "../src/vla/policy.js";
import { ground } from "../src/vla/language.js";
import { SafetyShield } from "../src/safety/shield.js";
import { detect, rgb2hsv } from "../src/vla/vision.js";

const load = (f) => JSON.parse(readFileSync(new URL(`../public/models/${f}`, import.meta.url)));

test("JS forward kinematics matches Python canonical model (src/robot/arm.py)", () => {
  const v = load("fk_vectors.json");
  v.q.forEach((q, i) => v.ee[i].forEach((e, k) => assert.ok(Math.abs(fk(q)[k] - e) < 1e-9)));
});

test("analytic Jacobian matches finite differences", () => {
  const q = [0.3, 0.2, 1.0, 0.5, 0, 1], J = jacobian(q);
  for (let j = 0; j < 4; j++) {
    const a = q.slice(), b = q.slice(); a[j] += 1e-6; b[j] -= 1e-6;
    const fa = fk(a), fb = fk(b);
    for (let i = 0; i < 3; i++) assert.ok(Math.abs(J[i][j] - (fa[i] - fb[i]) / 2e-6) < 1e-6);
  }
});

test("IK fallback converges to reachable targets", () => {
  for (const t of [[0.2, 0.05, 0.05], [0.15, -0.15, 0.1], [0.27, 0.1, 0.02]]) {
    let q = Q_HOME.slice();
    for (let k = 0; k < 150; k++) { const qd = ikStep(q, t); for (let i = 0; i < 4; i++) q[i] += qd[i] * 0.05; q = clampQ(q); }
    assert.ok(dist(fk(q), t) < 0.005, `target ${t} err ${dist(fk(q), t)}`);
  }
});

for (const name of ["dense", "ttns"]) {
  test(`exported ${name} policy: JS forward == PyTorch forward`, () => {
    const spec = load(`policy_${name}.json`);
    spec.test_vectors.x.forEach((x, i) => {
      const y = forwardLayers(spec.layers, x);
      y.forEach((v, k) => assert.ok(Math.abs(v - spec.test_vectors.y[i][k]) < 1e-4, `${v} vs ${spec.test_vectors.y[i][k]}`));
    });
  });
  test(`${name} policy reaches targets in closed loop (JS)`, () => {
    const p = new ReachPolicy(load(`policy_${name}.json`));
    let q = Q_HOME.slice(); const t = [0.22, -0.08, 0.04];
    for (let k = 0; k < 100; k++) { const qd = p.act(q, t); for (let i = 0; i < 4; i++) q[i] += qd[i] * 0.05; q = clampQ(q); }
    assert.ok(dist(fk(q), t) < 0.01);
  });
}

test("language grounding: Indonesian and English pick-and-place", () => {
  const scene = { cubes: ["red", "green", "blue"], pads: ["yellow", "purple"] };
  let g = ground("ambil kubus merah taruh di pad kuning", scene);
  assert.deepEqual(g.plan, [{ skill: "pick", object: "red" }, { skill: "place", target: "yellow", onCube: false }]);
  g = ground("Pick up the blue cube and place it on the purple pad", scene);
  assert.equal(g.plan[0].object, "blue"); assert.equal(g.plan[1].target, "purple");
  g = ground("tumpuk kubus hijau di atas kubus biru", scene);
  assert.deepEqual(g.plan[1], { skill: "place", target: "blue", onCube: true });
  assert.equal(ground("home", scene).plan[0].skill, "home");
  assert.equal(ground("ambil kubus ungu", { cubes: [], pads: [] }).ok, false);
  assert.equal(ground("nyanyikan lagu", scene).ok, false);
});

test("safety shield: blocks table collision, clips velocity, latches e-stop", () => {
  const s = new SafetyShield();
  const q = Q_HOME.slice(); q[1] = 1.2; q[2] = 1.2; q[3] = 0.6; // tip near the table
  const r = s.filter(q, [0.2, 0, 0.0], [0, 50, 50, 50]);
  assert.ok(r.clipped);
  const qn = q.slice(); for (let i = 0; i < 4; i++) qn[i] += r.qd[i] * 0.05;
  assert.ok(fk(qn)[2] >= 0.005 - 1e-9 || r.qd.every((v) => v === 0));
  s.estop = true;
  assert.deepEqual(s.filter(q, [0.2, 0, 0.1], [1, 1, 1, 1]).qd, [0, 0, 0, 0]);
});

test("safety shield: Lyapunov monitor hands control to IK when V keeps rising", () => {
  const s = new SafetyShield({ patience: 3 });
  let q = Q_HOME.slice(); const t = [0.2, 0.1, 0.06];
  let src = "";
  for (let k = 0; k < 10; k++) {
    const ee = fk(q); const away = [ee[0] - t[0], ee[1] - t[1], ee[2] - t[2]];
    // adversarial "policy": move AWAY from the target
    const J = jacobian(q); const bad = [0, 1, 2, 3].map((j) => 3 * (J[0][j] * away[0] + J[1][j] * away[1] + J[2][j] * away[2]));
    const r = s.filter(q, t, bad); src = r.source;
    for (let i = 0; i < 4; i++) q[i] += r.qd[i] * 0.05; q = clampQ(q);
  }
  assert.equal(src, "fallback-ik");
});

test("vision: HSV segmentation recovers object centroid in world coords", () => {
  assert.deepEqual(rgb2hsv(255, 0, 0).map((v) => Math.round(v * 100) / 100), [0, 1, 1]);
  const W = 160, H = 128, px = new Uint8Array(W * H * 4).fill(40);
  const b = { xMin: -0.02, xMax: 0.38, yMin: -0.25, yMax: 0.25 };
  // paint a red blob centred on world (0.2, 0.1)
  const v0 = Math.round(((0.2 - b.xMin) / 0.4) * H - 0.5), u0 = Math.round(((b.yMax - 0.1) / 0.5) * W - 0.5);
  for (let v = v0 - 3; v <= v0 + 3; v++) for (let u = u0 - 3; u <= u0 + 3; u++) { const i = (v * W + u) * 4; px[i] = 230; px[i + 1] = 30; px[i + 2] = 30; }
  const d = detect(px, W, H, b).red;
  assert.ok(Math.abs(d.x - 0.2) < 0.005 && Math.abs(d.y - 0.1) < 0.005, JSON.stringify(d));
});
