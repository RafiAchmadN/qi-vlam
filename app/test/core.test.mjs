// Node built-in test runner: `npm test` (no browser, no WebGL needed).
import { test } from "node:test";
import assert from "node:assert/strict";
import { readFileSync } from "node:fs";
import { fk, ikStep, jacobian, dist, Q_HOME, clampQ } from "../src/robot/kinematics.js";
import { forwardLayers, ReachPolicy } from "../src/vla/policy.js";
import { ground } from "../src/vla/language.js";
import { SafetyShield } from "../src/safety/shield.js";
import { detect, rgb2hsv, HUE_RANGES } from "../src/vla/vision.js";

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

// ------------------------------------------------------------ environments, pick-place, data
import { existsSync } from "node:fs";
import { sampleLayout, rng, insideXY, sampleVisualDR } from "../src/robot/envs.js";
import { ppFeatures, rollForYaw, PickPlacePolicy } from "../src/vla/pickplace.js";
import { envState, Recorder } from "../src/data/recorder.js";
import { inObstacle } from "../src/safety/shield.js";

const env = (id) => JSON.parse(readFileSync(new URL(`../public/envs/${id}.json`, import.meta.url)));
const ENV_IDS = JSON.parse(readFileSync(new URL("../public/envs/index.json", import.meta.url))).envs;

test("environment layouts: separated, inside spawn region, clear of obstacles", () => {
  for (const id of ENV_IDS) {
    const e = env(id);
    for (let s = 1; s < 40; s++) {
      const L = sampleLayout(e, rng(s));
      const pts = [...Object.values(L.cubes), ...Object.values(L.pads)].map((o) => o.pos);
      assert.equal(pts.length, 5);
      pts.forEach((p, i) => pts.slice(0, i).forEach((q) => assert.ok(Math.hypot(p[0] - q[0], p[1] - q[1]) >= e.spawn.min_sep - 1e-9)));
      for (const p of pts) for (const o of e.obstacles) if (!o.visual_only) assert.ok(!insideXY(p, o));
    }
  }
  const v = sampleVisualDR(JSON.parse(readFileSync(new URL("../public/envs/randomization.json", import.meta.url))), rng(3));
  assert.ok(v.light > 0.6 && v.light < 1.4);
});

test("shield blocks motion into an environment obstacle", () => {
  const sh = new SafetyShield({});
  const q = [0, 0.6, 1.0, 1.2, 0, 1];
  const tip = fk(q);
  sh.obstacles = [{ name: "box", pos: [tip[0] + 0.05, tip[1], tip[2]], size: [0.06, 0.2, 0.2] }];
  assert.ok(inObstacle([tip[0] + 0.05, tip[1], tip[2]], sh.obstacles));
  const fwd = ikStep(q, [tip[0] + 0.08, tip[1], tip[2]]);
  const out = sh.limit(q, fwd.map((v) => v * 1));
  const q2 = q.slice(); for (let i = 0; i < 4; i++) q2[i] += out.qd[i] * 0.05;
  assert.ok(out.clipped);
  assert.ok(!inObstacle(fk(q2), sh.obstacles, 0.02));
});

test("recorder env_state layout matches src/sim/envs.py::env_state_vector (32 values)", () => {
  const objs = { red: { pos: [0.2, 0, 0.015], yaw: 0.1 }, green: { pos: [0.1, 0.1, 0.015], yaw: 0 }, blue: { pos: [0.15, -0.1, 0.015], yaw: 0 },
    yellow: { pos: [0.25, 0.1, 0.001] }, purple: { pos: [0.25, -0.1, 0.001] } };
  const s = envState(objs, "red", "blue", "cube");
  assert.equal(s.length, 32);
  assert.deepEqual(s.slice(0, 8), [0.2, 0, 0.015, 0.1, 0.15, -0.1, 0.03, 1]);
  assert.deepEqual(s.slice(28), [0.25, -0.1, 0.25, 0.1]);
  assert.deepEqual(s.slice(12, 16), [0, 0, 0, 0]); // cyan slot absent
  const r = new Recorder(); r.enabled = true; r.begin({ instruction: "x" });
  for (let i = 0; i < 6; i++) r.frame({ q: [0, 0, 0, 0, 0, 1] });
  r.end(true);
  assert.equal(r.payload("sim").format, "qi-vla-demos/1");
  assert.deepEqual(r.stats(), { n: 1, ok: 1, frames: 6 });
});

const PP = new URL("../public/models/pickplace_ttns.json", import.meta.url);
test("pick-place policy: JS features, roll and forward pass match Python export", { skip: !existsSync(PP) }, () => {
  const spec = JSON.parse(readFileSync(PP));
  for (const c of spec.feature_vectors.cases) {
    const x = ppFeatures(c.q, c.obj, c.goal, c.goal_is_cube);
    c.x.forEach((v, i) => assert.ok(Math.abs(x[i] - v) < 1e-5, `feature ${i}`));
    assert.ok(Math.abs(rollForYaw(c.q[0], c.obj[3]) - c.roll) < 1e-9);
  }
  spec.test_vectors.x.forEach((x, k) => forwardLayers(spec.layers, x).forEach((v, i) => assert.ok(Math.abs(v - spec.test_vectors.y[k][i]) < 1e-4)));
  const p = new PickPlacePolicy(spec);
  const a = p.act([0, -0.2, 1.6, 1.1, 0, 1], [0.2, 0, 0.015], [0.25, 0.1, 0], false);
  assert.equal(a.qd.length, 4);
});

test("vision separates the seven object colors; grounding knows new colors and shape words", () => {
  const hex = { red: 0xe53935, orange: 0xfa7317, yellow: 0xfdd835, green: 0x43a047, cyan: 0x00bfa5, blue: 0x1e88e5, purple: 0x8e24aa };
  for (const [c, h] of Object.entries(hex)) {
    const [hh] = rgb2hsv((h >> 16) & 255, (h >> 8) & 255, h & 255);
    const hits = Object.entries(HUE_RANGES).filter(([, rs]) => rs.some(([a, b]) => hh >= a && hh <= b)).map(([k]) => k);
    assert.deepEqual(hits, [c], `${c} hue ${hh.toFixed(1)}`);
  }
  const g = ground("ambil tabung oranye taruh di pad ungu", { cubes: ["orange", "cyan", "red"], pads: ["purple", "yellow"] });
  assert.ok(g.ok); assert.equal(g.plan[0].object, "orange"); assert.equal(g.plan[1].target, "purple");
  const h = ground("stack the cyan can on the red cube", { cubes: ["orange", "cyan", "red"], pads: [] });
  assert.equal(h.plan[1].target, "red"); assert.ok(h.plan[1].onCube);
});

// ------------------------------------------------------------ mobile robot
import { boxes as mapBoxes, raycast as rayJS, buildGrid, planPath, navFeatures, shieldVW, groundPlace, Costmap, RoverModel, NavExpert } from "../src/rover/rover.js";
const MAPIDX = JSON.parse(readFileSync(new URL("../public/maps/index.json", import.meta.url)));
const mapOf = (id) => JSON.parse(readFileSync(new URL(`../public/maps/${id}.json`, import.meta.url)));

test("rover: A* connects every pair of places on every map; grounding finds places (ID/EN)", () => {
  for (const id of MAPIDX.maps) {
    const m = mapOf(id), g = buildGrid(m, MAPIDX.robot.radius + 0.06);
    for (const a of m.places) for (const b of m.places) if (a !== b) assert.ok(planPath(g, a.pos, b.pos), `${id}: ${a.id}->${b.id}`);
  }
  const home = mapOf("home");
  assert.equal(groundPlace("tolong pergi ke dapur", home).id, "kitchen");
  assert.equal(groundPlace("go to the charging station", home).id, "dock");
  assert.equal(groundPlace("ke kamar tidur", home).id, "bedroom");
  assert.equal(groundPlace("terbang ke bulan", home), null);
});

test("rover: lidar shield bounds forward speed, expert reaches a goal in the kinematic sim", () => {
  const R = MAPIDX.robot, scan = new Array(36).fill(3.5); scan[0] = R.radius + 0.1;
  const s = shieldVW(R, 0.26, 0.4, scan);
  assert.ok(s.clipped && s.v < 0.05 && s.w === 0.4);
  const m = mapOf("office"), g = buildGrid(m, R.radius + 0.06);
  const goal = m.places.find((p) => p.id === "meeting_room").pos;
  const rob = new RoverModel(R, m, [1.2, 1.9, 0]), ex = new NavExpert(R, g, goal);
  let t = 0;
  for (let k = 0; k < 1600 && !ex.done; k++, t += 0.05) { const sc = rob.scan(); const a = ex.act(rob.pose, sc, t); const sh = shieldVW(R, a.v, a.w, sc); rob.step(sh.v, sh.w); }
  assert.ok(ex.done, "expert did not reach the meeting room");
  assert.equal(rob.collisions, 0);
  const cm = new Costmap(g, R); assert.ok(cm.update([1.2, 1.9, 0], new Array(36).fill(0.5)));
});

const NAV = new URL("../public/models/nav_ttns.json", import.meta.url);
test("rover: JS ray casting, features and shield match the Python export", { skip: !existsSync(NAV) }, () => {
  const spec = JSON.parse(readFileSync(NAV));
  for (const c of spec.parity_cases) {
    const B = mapBoxes(mapOf(c.map));
    rayJS(B, ...c.pose).forEach((r, k) => assert.ok(Math.abs(r - c.scan[k]) < 1e-5, `beam ${k}`));
    navFeatures(MAPIDX.robot, c.pose, c.scan, c.target, c.goal, c.vw).forEach((v, i) => assert.ok(Math.abs(v - c.x[i]) < 1e-5, `feature ${i}`));
    const s = shieldVW(MAPIDX.robot, 0.26, 0.5, c.scan);
    assert.ok(Math.abs(s.v - c.shield[0]) < 1e-9 && Math.abs(s.w - c.shield[1]) < 1e-9);
  }
  spec.test_vectors.x.forEach((x, k) => forwardLayers(spec.layers, x).forEach((v, i) => assert.ok(Math.abs(v - spec.test_vectors.y[k][i]) < 1e-4)));
});

const VIS = new URL("../public/models/real_cnn_ttns.json", import.meta.url);
test("real-data vision policy: JS CNN runtime matches PyTorch on a raw real frame", { skip: !existsSync(VIS) }, async () => {
  const { VisionPolicy } = await import("../src/real/visionPolicy.js");
  const spec = JSON.parse(readFileSync(VIS));
  const p = new VisionPolicy(spec), tv = spec.test_vector, S = spec.size;
  const rgb = Buffer.from(tv.rgb_b64, "base64"), rgba = new Uint8ClampedArray(S * S * 4);
  for (let i = 0; i < S * S; i++) { rgba[i * 4] = rgb[i * 3]; rgba[i * 4 + 1] = rgb[i * 3 + 1]; rgba[i * 4 + 2] = rgb[i * 3 + 2]; rgba[i * 4 + 3] = 255; }
  const y = p.forward(rgba, tv.state_norm, tv.task);
  y.forEach((r, j) => r.forEach((v, i) => assert.ok(Math.abs(v - tv.y[j][i]) < 1e-4, `out ${j},${i}`)));
  const idx = JSON.parse(readFileSync(new URL("../public/real/index.json", import.meta.url)));
  assert.ok(idx.episodes.length >= 1);
});
