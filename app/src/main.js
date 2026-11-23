import { RobotScene, VISION_BOUNDS, CUBE } from "./robot/scene.js";
import { JOINT_NAMES, Q_HOME, Q_MIN, Q_MAX, clampQ, fk, dist, ikStep } from "./robot/kinematics.js";
import { detect, drawDetections } from "./vla/vision.js";
import { ground } from "./vla/language.js";
import { ReachPolicy, RemotePolicy } from "./vla/policy.js";
import { Agent } from "./vla/agent.js";
import { SafetyShield } from "./safety/shield.js";
import { BridgeTransport, RosbridgeTransport } from "./connect/transports.js";

const $ = (id) => document.getElementById(id);
const DT = 0.05; // 20 Hz control, same as training
const CUBES = ["red", "green", "blue"], PADS = ["yellow", "purple"];

// ---------------------------------------------------------------- state
const app = {
  mode: "sim", q: Q_HOME.slice(), measured: null, held: null, target: null,
  policies: {}, policyKey: "ttns", remote: null, transport: null, armed: false,
  synced: false, speed: 1, lastDets: {}, lastPx: null, perturb: false, shieldOn: true,
};

// ---------------------------------------------------------------- log
const logLines = [];
function log(kind, msg) {
  const t = new Date().toLocaleTimeString();
  logLines.push(`[${t}] ${kind.padEnd(7)} ${msg}`);
  if (logLines.length > 300) logLines.shift();
  $("log").textContent = logLines.slice().reverse().join("\n");
  if (kind === "step" || kind === "error" || kind === "info") $("planView").textContent = msg;
}

// ---------------------------------------------------------------- scene
const scene = new RobotScene($("view"));
function resize() {
  const c = $("view").parentElement;
  scene.resize(c.clientWidth, c.clientHeight);
}
window.addEventListener("resize", resize);
resize();

function randomScene(seed = Date.now()) {
  let s = seed % 2147483647; const rnd = () => ((s = (s * 16807) % 2147483647) / 2147483647);
  scene.clearObjects();
  const placed = [];
  const spot = () => {
    for (let k = 0; k < 200; k++) {
      const r = 0.14 + rnd() * 0.12, a = -0.9 + rnd() * 1.8;
      const p = [r * Math.cos(a), r * Math.sin(a)];
      if (placed.every((o) => Math.hypot(o[0] - p[0], o[1] - p[1]) > 0.075)) { placed.push(p); return p; }
    }
    return [0.2, 0];
  };
  CUBES.forEach((c) => scene.addCube(c, spot()));
  PADS.forEach((c) => scene.addPad(c, spot()));
  app.held = null;
  log("info", "scene diacak: 3 kubus (merah/hijau/biru), 2 pad (kuning/ungu)");
}

// world interface used by the agent (sim physics: kinematic grasp)
const world = {
  kindOf: (name) => scene.objects[name]?.kind,
  tryAttach(name) {
    const o = scene.objects[name];
    if (o && dist(fk(app.q), o.pos) < 0.025) { app.held = name; log("info", `kubus ${name} tergenggam`); }
    else log("error", `genggam gagal: ${name} tidak di antara jari`);
  },
  release() {
    if (!app.held) return;
    const o = scene.objects[app.held];
    // drop onto the highest support under the cube (pad / another cube / table)
    let z = CUBE / 2;
    for (const other of Object.values(scene.objects)) {
      if (other === o || other.kind !== "cube") continue;
      if (Math.hypot(other.pos[0] - o.pos[0], other.pos[1] - o.pos[1]) < CUBE * 0.8) z = Math.max(z, other.pos[2] + CUBE);
    }
    o.pos[2] = z;
    log("info", `kubus ${app.held} dilepas`);
    app.held = null;
  },
};

function perceive() {
  app.lastPx = scene.captureVision();
  const dets = detect(app.lastPx, scene.visionW, scene.visionH, VISION_BOUNDS);
  // a held cube is occluded by / attached to the gripper: don't use it as a goal
  if (app.held) delete dets[app.held];
  app.lastDets = dets;
  return dets;
}

const agent = new Agent({ perceive, log });
const shield = new SafetyShield({ dt: DT });

// ---------------------------------------------------------------- policies
async function loadPolicies() {
  for (const k of ["ttns", "dense"]) {
    try {
      app.policies[k] = await ReachPolicy.load(`./models/policy_${k}.json`);
    } catch (e) { log("error", `gagal memuat policy ${k}: ${e.message}`); }
  }
  const p = app.policies.ttns || app.policies.dense;
  if (p?.spec.lyapunov?.P) {
    shield.P = p.spec.lyapunov.P;
    const L = p.spec.lyapunov;
    log("info", `sertifikat Lyapunov: ${L.certified ? "feasible" : "TIDAK feasible"}, ρ(A)=${L.spectral_radius.toFixed(3)}, R²=${L.r_squared_heldout.toFixed(3)}`);
  }
  renderModelInfo();
}

function renderModelInfo() {
  const d = app.policies.dense, t = app.policies.ttns;
  const row = (p) => p ? `${p.name}: ${p.params.toLocaleString()} parameter · reach ${p.spec.closed_loop.reach_success_pct.toFixed(0)}% · error median ${p.spec.closed_loop.final_err_mm_median.toFixed(2)} mm` : "";
  const ratio = d && t ? ` · kompresi ${(d.params / t.params).toFixed(2)}x` : "";
  $("modelInfo").innerHTML = `${row(t)}<br>${row(d)}${ratio}<br>Sertifikat SDP berlaku untuk model galat linear hasil identifikasi; di aplikasi V(e)=eᵀPe dipakai sebagai monitor runtime + fallback IK (Simplex), bukan bukti formal untuk lengan nonlinear.`;
}

let lastLatency = 0;
async function policyAct(q, goal) {
  const t0 = performance.now();
  let qd;
  if (app.policyKey === "ik") qd = null;
  else if (app.policyKey === "remote") {
    try { qd = await app.remote.act(q, goal, $("cmdText").value); } catch (e) { log("error", e.message); qd = null; }
  } else qd = app.policies[app.policyKey]?.act(q, goal) ?? null;
  lastLatency = performance.now() - t0;
  if (qd && app.perturb) qd = qd.map((v) => v + (Math.random() - 0.5) * 3.0);
  return qd;
}

// ---------------------------------------------------------------- control loop
let manualGoal = null, gripperCmd = 1, busy = false;
async function controlTick() {
  if (busy) return; busy = true;
  try {
    const q = app.q;
    const out = agent.tick(q, DT, world);
    if (out?.gripper !== null && out?.gripper !== undefined) gripperCmd = out.gripper;
    const goal = out?.goal || manualGoal;
    app.target = goal;
    if (goal) {
      if (goal !== controlTick.lastGoal) { shield.reset(); controlTick.lastGoal = goal; }
      const learned = await policyAct(q, goal);
      const res = app.shieldOn ? shield.filter(q, goal, learned)
        : { qd: learned || ikStep(q, goal), source: learned ? "policy (tanpa shield)" : "ik", V: shield.V([0, 0, 0]) };
      const qn = q.slice();
      for (let i = 0; i < 4; i++) qn[i] += res.qd[i] * DT * app.speed;
      app.q = clampQ(qn);
      $("hudSource").textContent = shield.estop ? "E-STOP" : res.source;
      $("hudV").textContent = (res.V * 1e4).toFixed(2);
      $("hudErr").textContent = (1000 * dist(fk(app.q), goal)).toFixed(1);
      $("hudLat").textContent = lastLatency.toFixed(2);
      if (manualGoal && dist(fk(app.q), manualGoal) < 0.004) manualGoal = null;
    }
    if (!shield.estop) app.q[5] += Math.sign(gripperCmd - app.q[5]) * Math.min(Math.abs(gripperCmd - app.q[5]), 2.0 * DT);
    if (app.held) { const o = scene.objects[app.held]; const p = fk(app.q); o.pos = [p[0], p[1], p[2]]; }
    if (app.mode === "live" && app.transport?.connected && app.armed && app.synced && !shield.estop) app.transport.sendJoints(app.q);
    flushShieldEvents();
  } finally { busy = false; }
}
let shieldSeen = 0;
function flushShieldEvents() {
  while (shieldSeen < shield.events.length) { const e = shield.events[shieldSeen++]; log("safety", e.msg); }
  if (shield.events.length >= 200) shieldSeen = shield.events.length;
}

setInterval(controlTick, DT * 1000);
let frame = 0;
function animate() {
  scene.render(app.q, { ghostQ: app.mode === "live" ? app.measured : null, target: app.target });
  if (frame++ % 6 === 0) { // refresh camera thumbnail at ~10 Hz
    const px = scene.captureVision();
    const dets = detect(px, scene.visionW, scene.visionH, VISION_BOUNDS);
    drawDetections($("camView"), px, scene.visionW, scene.visionH, dets);
  }
  requestAnimationFrame(animate);
}

// ---------------------------------------------------------------- UI wiring
document.querySelectorAll(".tabs button").forEach((b) => b.addEventListener("click", () => {
  document.querySelectorAll(".tabs button").forEach((x) => x.classList.toggle("on", x === b));
  document.querySelectorAll(".panel").forEach((p) => p.classList.toggle("hidden", p.id !== `tab-${b.dataset.tab}`));
}));

const EXAMPLES = [
  "ambil kubus merah taruh di pad kuning", "pick up the blue cube and place it on the purple pad",
  "tumpuk kubus hijau di atas kubus biru", "sentuh pad ungu", "home",
];
EXAMPLES.forEach((t) => {
  const b = document.createElement("button"); b.textContent = t; b.type = "button";
  b.onclick = () => { $("cmdText").value = t; runInstruction(t); };
  $("chips").appendChild(b);
});

function runInstruction(text) {
  if (shield.estop) { log("error", "E-STOP aktif — tekan STOP lagi untuk melepas"); return; }
  perceive();
  const visible = Object.keys(app.lastDets);
  const g = ground(text, { cubes: CUBES.filter((c) => visible.includes(c) || c === app.held), pads: PADS.filter((c) => visible.includes(c)) });
  if (!g.ok) { log("error", g.error); return; }
  manualGoal = null;
  log("info", `rencana: ${g.plan.map((s) => s.skill + (s.object ? `(${s.object})` : "") + (s.target ? `→${s.target}` : "")).join(" · ")}`);
  agent.start(g.plan);
}
$("cmdForm").addEventListener("submit", (e) => { e.preventDefault(); runInstruction($("cmdText").value); });
$("btnReset").onclick = () => { agent.stop(); randomScene(); };
$("btnHome").onclick = () => runInstruction("home");
$("speed").oninput = (e) => { app.speed = parseFloat(e.target.value); };

$("estop").onclick = () => {
  shield.estop = !shield.estop;
  $("estop").classList.toggle("latched", shield.estop);
  if (shield.estop) { agent.stop(); manualGoal = null; app.transport?.estop(); log("safety", "E-STOP ditekan: semua gerakan dihentikan"); }
  else {
    app.transport?.releaseEstop();
    // re-sync to the measured pose so motion resumes from where the robot really is
    if (app.mode === "live" && app.measured) app.q = clampQ(app.measured.slice(0, 6));
    log("safety", "E-STOP dilepas");
  }
};

// manual joint sliders (joint-space jog, still filtered by limits)
JOINT_NAMES.forEach((n, i) => {
  const d = document.createElement("div"); d.className = "slider";
  d.innerHTML = `<span>${n}</span><input type="range" min="${Q_MIN[i]}" max="${Q_MAX[i]}" step="0.01" value="${Q_HOME[i]}" id="j${i}"><span id="jv${i}"></span>`;
  $("sliders").appendChild(d);
  d.querySelector("input").addEventListener("input", (e) => {
    if (shield.estop) return;
    agent.stop(); manualGoal = null;
    const v = parseFloat(e.target.value);
    if (i === 5) gripperCmd = v; else app.q[i] = v;
    app.q = clampQ(app.q);
    if (fk(app.q)[2] < 0.005) { app.q[i] -= Math.sign(v - Q_HOME[i]) * 0.02; log("safety", "jog ditolak: ujung gripper menyentuh meja"); }
  });
});
setInterval(() => JOINT_NAMES.forEach((_, i) => {
  $(`jv${i}`).textContent = app.q[i].toFixed(2);
  const s = $(`j${i}`); if (document.activeElement !== s) s.value = app.q[i];
}), 200);

// tap on table (desktop: click) -> reach that point
$("view").addEventListener("dblclick", () => { const p = fk(app.q); manualGoal = [p[0], p[1], 0.12]; });

function setMode(m) {
  app.mode = m;
  $("modeSim").classList.toggle("on", m === "sim");
  $("modeLive").classList.toggle("on", m === "live");
  if (m === "live") {
    document.querySelector('.tabs button[data-tab="conn"]').click();
    log("info", "mode Robot Nyata: hubungkan ke bridge/ESP32/rosbridge, lalu centang Armed");
  } else {
    app.armed = false; $("armed").checked = false;
    log("info", "mode Simulasi");
  }
}
$("modeSim").onclick = () => setMode("sim");
$("modeLive").onclick = () => setMode("live");

$("connType").onchange = (e) => {
  const ros = e.target.value === "ros";
  $("rosOpts").classList.toggle("hidden", !ros);
  $("connUrl").value = ros ? "ws://192.168.1.10:9090" : "ws://192.168.1.10:8765/ws";
};
function setPill(text, ok) { const p = $("connPill"); p.textContent = text; p.className = `pill ${ok ? "ok" : "bad"}`; }
$("btnConnect").onclick = async () => {
  app.transport?.close();
  const url = $("connUrl").value.trim();
  app.transport = $("connType").value === "ros"
    ? new RosbridgeTransport(url, { cmdTopic: $("rosCmd").value, stateTopic: $("rosState").value })
    : new BridgeTransport(url);
  app.synced = false; app.armed = false; $("armed").checked = false;
  app.transport.onStatus((t, ok) => setPill(t, ok));
  app.transport.onState((q) => {
    app.measured = q;
    if (!app.synced) { app.q = clampQ(q.slice(0, 6).concat(Q_HOME.slice(q.length))); app.synced = true; log("info", "pose perintah disinkronkan ke pose robot terukur"); }
  });
  try { await app.transport.connect(); log("info", `terhubung ke ${url}`); setMode("live"); }
  catch (e) { log("error", `koneksi gagal: ${e.message}`); setPill("gagal", false); }
};
$("btnDisconnect").onclick = () => { app.transport?.close(); app.transport = null; app.armed = false; $("armed").checked = false; setPill("offline", false); };
$("armed").onchange = (e) => {
  if (e.target.checked && !app.synced) { e.target.checked = false; log("error", "belum ada state dari robot — tidak bisa Armed"); return; }
  app.armed = e.target.checked; log("safety", app.armed ? "ARMED: perintah dikirim ke robot" : "disarmed");
};

$("policySel").onchange = (e) => {
  app.policyKey = e.target.value;
  $("remoteRow").classList.toggle("hidden", app.policyKey !== "remote");
  if (app.policyKey === "remote") app.remote = new RemotePolicy($("remoteUrl").value);
  log("info", `policy aktif: ${app.policyKey}`);
};
$("remoteUrl").onchange = (e) => { app.remote = new RemotePolicy(e.target.value); };
$("shieldOn").onchange = (e) => { app.shieldOn = e.target.checked; log("safety", `shield ${app.shieldOn ? "ON" : "OFF (hanya untuk eksperimen!)"}`); };
$("perturb").onchange = (e) => { app.perturb = e.target.checked; log("info", `gangguan aksi ${app.perturb ? "ON" : "OFF"}`); };

randomScene(42);
loadPolicies();
animate();
// for debugging / e2e tests (tick = one 50 ms control step, lets tests run faster than real time)
window.__qivla = { app, agent, shield, scene, runInstruction, perceive, randomScene, tick: controlTick };
