import { RobotScene, VISION_BOUNDS, CUBE } from "./robot/scene.js";
import { JOINT_NAMES, Q_HOME, Q_MIN, Q_MAX, clampQ, fk, dist, ikStep } from "./robot/kinematics.js";
import { loadEnvIndex, loadEnv, loadRandomization, rng, sampleVisualDR, sampleLayout } from "./robot/envs.js";
import { detect, drawDetections } from "./vla/vision.js";
import { ground } from "./vla/language.js";
import { ReachPolicy, RemotePolicy } from "./vla/policy.js";
import { PickPlacePolicy, PickPlaceRunner, rollForYaw, MAX_WINDUP } from "./vla/pickplace.js";
import { Agent } from "./vla/agent.js";
import { SafetyShield } from "./safety/shield.js";
import { BridgeTransport, RosbridgeTransport } from "./connect/transports.js";
import { Recorder, envState } from "./data/recorder.js";
import { RoverApp } from "./rover/roverApp.js";
import { RoverScene, drawScan } from "./rover/roverScene.js";
import { renderResearch } from "./ui/research.js";
import { RealApp } from "./real/realApp.js";

const $ = (id) => document.getElementById(id);
const DT = 0.05; // 20 Hz control, same as training
const PADS = ["yellow", "purple"];
const graspables = () => app.env?.cubes || ["red", "green", "blue"];

// ---------------------------------------------------------------- state
const app = {
  mode: "sim", q: Q_HOME.slice(), measured: null, held: null, target: null,
  policies: {}, policyKey: "ttns", remote: null, transport: null, armed: false,
  synced: false, speed: 1, lastDets: {}, lastPx: null, perturb: false, shieldOn: true,
  envs: [], envId: "lab_table", env: null, dr: null, visualDR: false, remoteScene: false, robot: "arm",
  task: null, // {obj, goal, kind, instruction} of the running pick-place instruction
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
const roverScene = new RoverScene(scene.renderer, $("view"));
const rover = new RoverApp({ log: (k, m) => log(app.robot === "rover" ? k : "rover", m) });
const real = new RealApp({ log: (k, m) => log(k, m) });
function resize() {
  const c = $("view").parentElement;
  scene.resize(c.clientWidth, c.clientHeight);
  roverScene.resize(c.clientWidth, c.clientHeight);
}
window.addEventListener("resize", resize);
resize();

const shield = new SafetyShield({ dt: DT });

function randomScene(seed = Date.now()) {
  if (app.mode === "live" && app.transport?.isSim) { // the physics twin owns the scene
    app.transport.sceneReset(app.envId, seed % 100000);
    log("info", `scene twin MuJoCo diacak (${app.envId})`);
    return;
  }
  const r = rng(seed);
  if (app.env) scene.applyEnv(app.env, app.visualDR ? sampleVisualDR(app.dr, r) : null);
  scene.clearObjects();
  const env = app.env || { cubes: graspables(), pads: PADS, obstacles: [], spawn: { r: [0.14, 0.26], a: [-0.9, 0.9], min_sep: 0.075 } };
  const lay = sampleLayout(env, r);
  for (const [c, o] of Object.entries(lay.cubes)) scene.addCube(c, o.pos, o.yaw, o.shape);
  for (const [c, o] of Object.entries(lay.pads)) scene.addPad(c, o.pos);
  app.held = null;
  log("info", `scene diacak di "${env.name || "meja"}": ${Object.entries(lay.cubes).map(([c, o]) => `${o.shape === "cylinder" ? "tabung" : "kubus"} ${c}`).join(", ")}${app.visualDR ? " + domain randomization visual" : ""}`);
  refreshArmChips();
}

async function setEnv(id) {
  try {
    app.env = await loadEnv(id);
    app.envId = id;
    if ($("envSel").value !== id) $("envSel").value = id;
    shield.obstacles = app.env.obstacles;
    $("envInfo").innerHTML = `<b>${app.env.name}</b> — ${app.env.description}<br>Dinamika (dipakai di simulasi fisika MuJoCo): lag servo ${(app.env.dynamics.servo_tau * 1000).toFixed(0)} ms · latensi ${app.env.dynamics.latency_steps} langkah · backlash ${(app.env.dynamics.backlash * 1000).toFixed(1)} mrad · noise sensor ${(app.env.dynamics.q_noise * 1000).toFixed(1)} mrad`;
    randomScene(42);
  } catch (e) { log("error", `gagal memuat lingkungan ${id}: ${e.message}`); }
}

// ---------------------------------------------------------------- world model
// Object poses: the app's own simulation, or (live + bridge --driver mujoco)
// the mirrored physics twin.
const world = {
  kindOf: (name) => scene.objects[name]?.kind,
  tryAttach(name) {
    if (app.remoteScene) return; // physics decides whether the grasp holds
    const o = scene.objects[name];
    if (o && dist(fk(app.q), o.pos) < 0.025) { app.held = name; log("info", `kubus ${name} tergenggam`); }
    else log("error", `genggam gagal: ${name} tidak di antara jari`);
  },
  release() {
    if (app.remoteScene || !app.held) return;
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
  objectPose(name) {
    const o = scene.objects[name];
    if (!o) return null;
    const d = app.lastDets[name]; // camera xy when visible (z and yaw from the world model)
    return { pos: d && app.held !== name ? [d.x, d.y, o.pos[2]] : o.pos.slice(), yaw: o.yaw || 0 };
  },
  goalSupport(name, kind) {
    const o = scene.objects[name];
    if (!o) return null;
    return kind === "cube" ? [o.pos[0], o.pos[1], o.pos[2] + CUBE / 2] : [o.pos[0], o.pos[1], 0];
  },
  isSuccess(obj, goal, kind) {
    const o = scene.objects[obj], g = scene.objects[goal];
    if (!o || !g || app.held === obj || app.q[5] < 0.75) return false;
    const dxy = Math.hypot(o.pos[0] - g.pos[0], o.pos[1] - g.pos[1]);
    return kind === "cube" ? dxy < 0.02 && Math.abs(o.pos[2] - (g.pos[2] + CUBE)) < 0.01 : dxy < 0.03 && o.pos[2] < CUBE / 2 + 0.01;
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
const recorder = new Recorder();
let runner = null;

// ---------------------------------------------------------------- policies
async function loadPolicies() {
  for (const k of ["ttns", "dense"]) {
    try {
      app.policies[k] = await ReachPolicy.load(`./models/policy_${k}.json`);
    } catch (e) { log("error", `gagal memuat policy ${k}: ${e.message}`); }
    try {
      app.policies[`pp_${k}`] = await PickPlacePolicy.load(`./models/pickplace_${k}.json`);
    } catch (e) { log("info", `policy pick-place ${k} belum ada (latih: python -m src.sim.train_pickplace)`); }
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
  const pp = (p) => {
    if (!p) return "";
    const e = p.spec.evaluation;
    const f = (k) => e?.[k] ? `${e[k].success_mean.toFixed(0)}±${e[k].success_std.toFixed(0)}%` : "?";
    return `<br>${p.name} (pick-place, dilatih di MuJoCo): ${p.params.toLocaleString()} parameter · sukses: lingkungan latih ${f("train_envs")}, pabrik (baru) ${f("heldout_env")}, servo aus ${f("hard_dynamics")}`;
  };
  $("modelInfo").innerHTML = `${row(t)}<br>${row(d)}${ratio}${pp(app.policies.pp_ttns)}${pp(app.policies.pp_dense)}<br>Sertifikat SDP berlaku untuk model galat linear hasil identifikasi; di aplikasi V(e)=eᵀPe dipakai sebagai monitor runtime + fallback IK (Simplex), bukan bukti formal untuk lengan nonlinear. Policy pick-place end-to-end hanya dibatasi oleh batas keras (sendi, kecepatan, meja, rintangan).`;
}

let lastLatency = 0;
async function policyAct(q, goal) {
  const t0 = performance.now();
  let qd;
  if (app.policyKey === "ik" || app.policyKey.startsWith("pp_")) qd = null;
  else if (app.policyKey === "remote") {
    try { qd = await app.remote.act(q, goal, $("cmdText").value); } catch (e) { log("error", e.message); qd = null; }
  } else qd = app.policies[app.policyKey]?.act(q, goal) ?? null;
  lastLatency = performance.now() - t0;
  if (qd && app.perturb) qd = qd.map((v) => v + (Math.random() - 0.5) * 3.0);
  return qd;
}

// ---------------------------------------------------------------- control loop
let manualGoal = null, gripperCmd = 1, busy = false, rollCmd = null;
const FINGER_ON_CUBE = 0.015 / 0.024; // normalized gripper reading when closed on a 30 mm cube
const nearCube = (q) => { const t = fk(q); return Object.values(scene.objects).some((o) => o.kind === "cube" && dist(t, o.pos) < 0.02); };
function controlInput() {
  // live robot: feed back the MEASURED pose (closes the loop through real
  // servo lag/droop); simulation: the commanded pose is the state.
  if (app.mode === "live" && app.synced && app.measured) return app.measured.slice(0, 6).concat(Q_HOME.slice(app.measured.length)).map((v, i) => i === 5 ? app.q[5] : v);
  return app.q;
}

async function controlTick() {
  if (app.robot === "real") {
    if (real.playing && ++controlTick.realDiv % 2 === 0) realStep();  // 10 Hz = dataset rate
    return;
  }
  if (app.robot === "rover") {
    const out = rover.tick(DT, shield.estop);
    $("hudSource").textContent = shield.estop ? "E-STOP" : rover.status === "running" ? rover.source : rover.status;
    $("hudV").textContent = `${out.v.toFixed(2)} / ${out.w.toFixed(2)}`;
    const d = rover.distToGoal();
    $("hudErr").textContent = d === null ? "–" : d.toFixed(2);
    $("hudLat").textContent = rover.lat.toFixed(2);
    return;
  }
  if (busy) return; busy = true;
  try {
    const qIn = controlInput();
    let qd = null, source = "", V = null;
    if (runner?.status === "running") {
      const t0 = performance.now();
      const out = runner.tick(qIn, DT, world);
      lastLatency = performance.now() - t0;
      if (out) {
        let raw = out.qd;
        if (app.perturb) raw = raw.map((v) => v + (Math.random() - 0.5) * 3.0);
        qd = app.shieldOn ? shield.limit(qIn, raw).qd : raw;
        gripperCmd = out.gripper;
        rollCmd = out.roll;
        source = shield.estop ? "E-STOP" : "policy e2e";
        app.target = null;
      }
    } else {
      const out = agent.tick(qIn, DT, world);
      if (out?.gripper !== null && out?.gripper !== undefined) gripperCmd = out.gripper;
      if (out?.align && scene.objects[out.align]) rollCmd = rollForYaw(qIn[0], scene.objects[out.align].yaw || 0);
      const goal = out?.goal || manualGoal;
      app.target = goal;
      if (goal) {
        if (goal !== controlTick.lastGoal) { shield.reset(); controlTick.lastGoal = goal; }
        const learned = await policyAct(qIn, goal);
        const res = app.shieldOn ? shield.filter(qIn, goal, learned)
          : { qd: learned || ikStep(qIn, goal), source: learned ? "policy (tanpa shield)" : "ik", V: shield.V([0, 0, 0]) };
        qd = res.qd; source = shield.estop ? "E-STOP" : res.source; V = res.V;
        if (manualGoal && dist(fk(qIn), manualGoal) < 0.004) manualGoal = null;
      }
    }
    if (qd) {
      const qn = app.q.slice();
      for (let i = 0; i < 4; i++) qn[i] = Math.max(qIn[i] - MAX_WINDUP, Math.min(qIn[i] + MAX_WINDUP, app.q[i] + qd[i] * DT * app.speed));
      app.q = clampQ(qn);
      $("hudSource").textContent = source;
      $("hudV").textContent = V === null ? "–" : (V * 1e4).toFixed(2);
      $("hudErr").textContent = app.target ? (1000 * dist(fk(qIn), app.target)).toFixed(1) : "–";
      $("hudLat").textContent = lastLatency.toFixed(2);
    }
    if (!shield.estop) {
      app.q[5] += Math.sign(gripperCmd - app.q[5]) * Math.min(Math.abs(gripperCmd - app.q[5]), 2.0 * DT);
      // fingers stop on a cube between them (as in physics: 15 mm half-width / 24 mm travel)
      if (!app.remoteScene && (app.held || nearCube(app.q))) app.q[5] = Math.max(app.q[5], FINGER_ON_CUBE);
      if (rollCmd !== null) app.q[4] += Math.max(-2 * DT, Math.min(2 * DT, rollCmd - app.q[4]));
    }
    // kinematic grasp for the end-to-end policy (it only commands the gripper)
    if (runner?.status === "running" && !app.remoteScene) {
      const o = scene.objects[runner.obj];
      if (!app.held && gripperCmd < 0.5 && app.q[5] <= FINGER_ON_CUBE + 1e-6 && o && dist(fk(app.q), o.pos) < 0.025) { app.held = runner.obj; log("info", `kubus ${runner.obj} tergenggam`); }
      else if (app.held && gripperCmd > 0.5 && app.q[5] > FINGER_ON_CUBE + 0.1) world.release();
    }
    if (app.held && !app.remoteScene) { const o = scene.objects[app.held]; const p = fk(app.q); o.pos = [p[0], p[1], p[2]]; }
    if (app.mode === "live" && app.transport?.connected && app.armed && app.synced && !shield.estop) app.transport.sendJoints(app.q);
    recordTick(qIn, qd);
    flushShieldEvents();
  } finally { busy = false; }
}

// ---------------------------------------------------------------- recording
function recordTick(qIn, qd) {
  if (!recorder.active || !app.task) return;
  const running = runner?.status === "running" || agent.status === "running";
  if (running) {
    recorder.frame({
      q: qIn.slice(0, 6), q_cmd: app.q.slice(), qd: (qd || [0, 0, 0, 0]).slice(0, 4), gripper: gripperCmd > 0.5 ? 1 : 0,
      env_state: envState(scene.objects, app.task.obj, app.task.goal, app.task.kind),
    });
  } else {
    const ok = world.isSuccess(app.task.obj, app.task.goal, app.task.kind);
    const ep = recorder.end(ok);
    app.task = null;
    if (ep) log("info", `demo terekam: ${ep.frames.length} frame, ${ok ? "sukses" : "gagal"}`);
    renderRecorder();
  }
}
function renderRecorder() {
  const s = recorder.stats();
  $("recStats").textContent = `${s.n} episode (${s.ok} sukses) · ${s.frames} frame`;
}

let shieldSeen = 0;
function flushShieldEvents() {
  while (shieldSeen < shield.events.length) { const e = shield.events[shieldSeen++]; log("safety", e.msg); }
  if (shield.events.length >= 200) shieldSeen = shield.events.length;
}

setInterval(controlTick, DT * 1000);
let frame = 0;
function animate() {
  if (app.robot === "real") { requestAnimationFrame(animate); return; }
  if (app.robot === "rover") {
    roverScene.render({ pose: rover.model.pose, ghost: rover.live?.connected ? rover.measured : null, scan: rover.curScan || rover.model.scan(),
      path: rover.status === "running" ? rover.expert?.path : null, goal: rover.goal });
    if (frame++ % 6 === 0) drawScan($("camView"), rover.curScan || rover.model.scan(), 3.5, rover.lastFront);
    requestAnimationFrame(animate);
    return;
  }
  scene.render(app.q, { ghostQ: app.mode === "live" ? app.measured : null, target: app.target });
  if (frame++ % 6 === 0) { // refresh camera thumbnail at ~10 Hz
    const px = scene.captureVision();
    const dets = detect(px, scene.visionW, scene.visionH, VISION_BOUNDS);
    if (app.held) delete dets[app.held];
    app.lastDets = dets;
    drawDetections($("camView"), px, scene.visionW, scene.visionH, dets);
  }
  requestAnimationFrame(animate);
}

// ---------------------------------------------------------------- UI wiring
document.querySelectorAll(".tabs button").forEach((b) => b.addEventListener("click", () => {
  document.querySelectorAll(".tabs button").forEach((x) => x.classList.toggle("on", x === b));
  document.querySelectorAll(".panel").forEach((p) => p.classList.toggle("hidden", p.id !== `tab-${b.dataset.tab}`));
  if (b.dataset.tab === "research" && !renderResearch.done) { renderResearch.done = true; renderResearch($("research")); }
}));

const ID_COLOR = { red: "merah", green: "hijau", blue: "biru", orange: "oranye", cyan: "tosca", yellow: "kuning", purple: "ungu" };
const EXAMPLES = ["home"];
function setChips(list) {
  $("chips").innerHTML = "";
  list.forEach((t) => {
    const b = document.createElement("button"); b.textContent = t; b.type = "button";
    b.onclick = () => { $("cmdText").value = t; runInstruction(t); };
    $("chips").appendChild(b);
  });
}
/** Example instructions built from the objects that are actually on the table now. */
function refreshArmChips() {
  if (app.robot === "rover") return;
  const objs = Object.values(scene.objects);
  const g = objs.filter((o) => o.kind === "cube"), p = objs.filter((o) => o.kind === "pad");
  const noun = (o, lang) => (o.shape === "cylinder" ? (lang === "id" ? "tabung" : "cylinder") : (lang === "id" ? "kubus" : "cube"));
  const out = [];
  if (g[0] && p[0]) out.push(`ambil ${noun(g[0], "id")} ${ID_COLOR[g[0].color]} taruh di pad ${ID_COLOR[p[0].color]}`);
  if (g[1] && p[1]) out.push(`pick up the ${g[1].color} ${noun(g[1], "en")} and place it on the ${p[1].color} pad`);
  const base = g.find((o, i) => i > 0 && o.shape !== "cylinder");
  if (g[0] && base && base !== g[0]) out.push(`tumpuk ${noun(g[0], "id")} ${ID_COLOR[g[0].color]} di atas kubus ${ID_COLOR[base.color]}`);
  if (p[1]) out.push(`sentuh pad ${ID_COLOR[p[1].color]}`);
  out.push("home");
  setChips(out);
}
setChips(EXAMPLES);

function runInstruction(text) {
  if (shield.estop) { log("error", "E-STOP aktif — tekan STOP lagi untuk melepas"); return; }
  if (app.robot === "rover") { rover.run(text); return; }
  perceive();
  const visible = Object.keys(app.lastDets);
  const g = ground(text, { cubes: graspables().filter((c) => visible.includes(c) || c === app.held), pads: PADS.filter((c) => visible.includes(c)) });
  if (!g.ok) { log("error", g.error); return; }
  manualGoal = null; rollCmd = null;
  runner?.stop(); agent.stop();
  if (recorder.active) recorder.end(false);
  const pick = g.plan.find((s) => s.skill === "pick"), place = g.plan.find((s) => s.skill === "place");
  app.task = pick && place ? { obj: pick.object, goal: place.target, kind: place.onCube ? "cube" : "pad", instruction: text } : null;
  if (app.task) recorder.begin({ instruction: text, env: app.envId, mode: app.mode, policy: app.policyKey });
  const pp = app.policies[app.policyKey];
  if (app.policyKey.startsWith("pp_") && pp && app.task) {
    runner = new PickPlaceRunner(pp, { log });
    runner.start(app.task.obj, app.task.goal, app.task.kind);
    return;
  }
  log("info", `rencana: ${g.plan.map((s) => s.skill + (s.object ? `(${s.object})` : "") + (s.target ? `→${s.target}` : "")).join(" · ")}`);
  agent.start(g.plan);
}
$("cmdForm").addEventListener("submit", (e) => { e.preventDefault(); runInstruction($("cmdText").value); });
$("btnReset").onclick = () => { if (app.robot === "rover") { rover.reset(); roverScene.setUnknown(rover.unknown); return; } agent.stop(); runner?.stop(); randomScene(); };
$("btnHome").onclick = () => runInstruction("home");
$("speed").oninput = (e) => { app.speed = parseFloat(e.target.value); };

$("estop").onclick = () => {
  shield.estop = !shield.estop;
  $("estop").classList.toggle("latched", shield.estop);
  if (shield.estop) { agent.stop(); runner?.stop(); rover.stop(); manualGoal = null; app.transport?.estop(); log("safety", "E-STOP ditekan: semua gerakan dihentikan"); }
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
    agent.stop(); runner?.stop(); manualGoal = null;
    const v = parseFloat(e.target.value);
    if (i === 5) gripperCmd = v; else if (i === 4) { rollCmd = null; app.q[i] = v; } else app.q[i] = v;
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
    if (app.remoteScene) { app.remoteScene = false; randomScene(); }
    log("info", "mode Simulasi");
  }
}
$("modeSim").onclick = () => setMode("sim");
$("modeLive").onclick = () => setMode("live");

// mirror the physics twin's scene (bridge --driver mujoco)
function mirrorScene(m) {
  if (app.mode !== "live") return;
  if (app.robot === "rover") {
    if (m.map && m.map !== rover.map?.id) setMap(m.map).then(() => { rover.unknown = m.unknown || []; roverScene.setUnknown(rover.unknown); });
    else if (m.unknown) { rover.unknown = m.unknown; roverScene.setUnknown(m.unknown); }
    return;
  }
  if (!app.remoteScene) { app.remoteScene = true; app.held = null; scene.clearObjects(); log("info", `scene dicerminkan dari twin MuJoCo (${m.env})`); }
  if (m.env && m.env !== app.envId) { $("envSel").value = m.env; loadEnv(m.env).then((e) => { app.env = e; app.envId = m.env; shield.obstacles = e.obstacles; scene.applyEnv(e); }); }
  for (const [c, p] of Object.entries(m.objects || {})) {
    if (!scene.objects[c]) scene.addCube(c, [p[0], p[1], p[2]], p[3], (m.shapes || {})[c] || "cube");
    Object.assign(scene.objects[c], { pos: [p[0], p[1], p[2]], yaw: p[3] });
  }
  for (const [c, p] of Object.entries(m.pads || {})) if (!scene.objects[c]) scene.addPad(c, p);
  for (const c of Object.keys(scene.objects)) if (!(c in (m.objects || {})) && !(c in (m.pads || {}))) { scene.scene.remove(scene.objects[c].mesh); delete scene.objects[c]; }
  if (!mirrorScene.chipsDone) { mirrorScene.chipsDone = true; refreshArmChips(); }
}

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
    ? new RosbridgeTransport(url, { cmdTopic: $("rosCmd").value, stateTopic: $("rosState").value, rover: app.robot === "rover" })
    : new BridgeTransport(url);
  rover.live = app.robot === "rover" ? { transport: app.transport, get connected() { return app.transport?.connected; }, get armed() { return app.armed; } } : null;
  rover.liveOrigin = null; rover.measured = null; rover.lastScan = null;
  app.transport.onRoverState((st) => { rover.onLiveState(st); app.synced = true; });
  app.synced = false; app.armed = false; $("armed").checked = false;
  app.transport.onStatus((t, ok) => setPill(t, ok));
  app.transport.onScene(mirrorScene);
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

// ---------------------------------------------------------------- robot type
function setRobot(kind) {
  app.robot = kind;
  const isRover = kind === "rover", isReal = kind === "real";
  document.querySelectorAll(".rover-only").forEach((e) => e.classList.toggle("hidden", !isRover));
  document.querySelectorAll(".arm-only").forEach((e) => e.classList.toggle("hidden", isRover || isReal));
  document.querySelectorAll(".real-only").forEach((e) => e.classList.toggle("hidden", !isReal));
  document.querySelectorAll(".arm-rover").forEach((e) => e.classList.toggle("hidden", isReal));
  $("realStage").classList.toggle("hidden", !isReal);
  $("chips").classList.toggle("hidden", isReal);
  if (isReal) { initReal(); real.playing = false; $("hudVLabel").textContent = "MAE sendi (unit data)"; $("hudErrLabel").textContent = "gripper cocok"; $("camLabel").textContent = "frame kamera nyata 64×64"; return; }
  scene.controls.enabled = !isRover; roverScene.controls.enabled = isRover;
  $("hudVLabel").textContent = isRover ? "v (m/s) / ω (rad/s)" : "Lyapunov V";
  $("hudErrLabel").textContent = isRover ? "jarak ke tujuan (m)" : "error (mm)";
  $("camLabel").textContent = isRover ? "lidar 360° · tampak atas" : "kamera atas · deteksi HSV";
  if (isRover) setChips(rover.map ? rover.map.places.slice(0, 4).map((p, i) => i % 2 ? `go to the ${p.names_en[0]}` : `pergi ke ${p.names_id[0]}`) : []);
  else refreshArmChips();
  $("cmdText").placeholder = isRover ? "mis. pergi ke dapur / go to the charging station" : "mis. ambil kubus merah taruh di pad kuning";
  agent.stop(); runner?.stop(); rover.stop();
  log("info", isRover ? `mode mobile robot (${rover.robot?.name || "rover"})` : "mode lengan 6-DOF");
}
async function setMap(id) {
  await rover.setMap(id);
  $("mapSel").value = id;
  const m = rover.map;
  $("mapInfo").innerHTML = `<b>${m.name}</b> — ${m.description}<br>Tempat: ${m.places.map((p) => p.names_id[0]).join(", ")}<br>Dinamika (MuJoCo): lag motor ${(m.dynamics.motor_tau * 1000).toFixed(0)} ms · latensi ${m.dynamics.latency_steps} · gesekan lantai ${m.dynamics.floor_friction} · noise lidar ${(m.dynamics.lidar_noise * 100).toFixed(1)} cm`;
  roverScene.setMap(m); roverScene.setUnknown(rover.unknown);
  if (app.robot === "rover") setRobot("rover");
}
$("robotSel").onchange = (e) => setRobot(e.target.value);

// ---------------------------------------------------------------- real-data replay
async function initReal() {
  if (initReal.done) return;
  try {
    await real.init();
    initReal.done = true;
    $("realEpSel").innerHTML = real.index.episodes.map((e) => `<option value="${e.file}">${e.robot.toUpperCase()} · ${e.task} · ep ${e.episode}</option>`).join("");
    for (const o of $("realModelSel").options) o.disabled = !real.models[o.value];
    $("realModelSel").value = real.modelKey;
    await real.loadEpisode($("realEpSel").value);
    drawReal();
  } catch (e) { log("error", `data nyata tidak tersedia: ${e.message}`); }
}
function drawReal() {
  if (!real.ep) return;
  const i = Math.min(real.t, real.frames.length - 1), fr = real.frames[i];
  const g = $("realFrame").getContext("2d"); g.imageSmoothingEnabled = false; g.drawImage(fr.img, 0, 0);
  const cv = $("camView").getContext("2d"); cv.imageSmoothingEnabled = false; cv.drawImage(fr.img, 0, 0, 160, 128);
  $("realCaption").textContent = `${real.ep.robot.toUpperCase()} nyata · "${real.ep.task}" · frame ${real.t}/${real.frames.length}`;
  const m = real.metrics();
  $("hudSource").textContent = real.modelKey;
  $("hudV").textContent = m ? m.mae.toFixed(2) : "–";
  $("hudErr").textContent = m ? `${(100 * m.grip).toFixed(0)}%` : "–";
  $("hudLat").textContent = m ? m.latP50.toFixed(1) : "–";
  $("realStats").textContent = m ? `${m.n} langkah · MAE sendi ${m.mae.toFixed(2)} (unit dataset) · gripper cocok ${(100 * m.grip).toFixed(0)}% · inferensi median ${m.latP50.toFixed(1)} ms di perangkat ini · model ${real.model.params.toLocaleString()} parameter` : "tekan Putar: model menghitung aksi dari setiap frame nyata";
  $("realChart").replaceChildren(real.chart());
}
function realStep() { real.step(); drawReal(); if (real.status === "done") { real.playing = false; $("realPlay").textContent = "Putar"; log("info", `replay selesai: ${$("realStats").textContent}`); } }
$("realPlay").onclick = () => { if (real.status === "done") real.reset(); real.playing = !real.playing; $("realPlay").textContent = real.playing ? "Jeda" : "Putar"; };
$("realStep").onclick = () => { real.playing = false; $("realPlay").textContent = "Putar"; realStep(); };
$("realReset").onclick = () => { real.reset(); $("realPlay").textContent = "Putar"; drawReal(); };
$("realEpSel").onchange = async (e) => { await real.loadEpisode(e.target.value); $("realPlay").textContent = "Putar"; drawReal(); };
$("realModelSel").onchange = async (e) => {
  real.modelKey = e.target.value; const was = real.playing; real.reset();
  await real.loadEpisode($("realEpSel").value); drawReal(); real.playing = was;
  log("info", `model data nyata: ${real.modelKey} (${real.model.params.toLocaleString()} parameter)`);
};
controlTick.realDiv = 0;
$("mapSel").onchange = (e) => setMap(e.target.value);
$("unkSel").onchange = (e) => { rover.unknownCount = parseInt(e.target.value, 10); rover.reset(); roverScene.setUnknown(rover.unknown); };
$("btnMapReset").onclick = () => { rover.reset(); roverScene.setUnknown(rover.unknown); };
$("navPolicySel").onchange = (e) => { rover.policyKey = e.target.value; log("info", `pengendali navigasi: ${rover.policyKey}`); };

$("policySel").onchange = (e) => {
  app.policyKey = e.target.value;
  $("remoteRow").classList.toggle("hidden", app.policyKey !== "remote");
  if (app.policyKey === "remote") app.remote = new RemotePolicy($("remoteUrl").value);
  if (app.policyKey.startsWith("pp_") && !app.policies[app.policyKey]) log("error", "model pick-place belum tersedia di app/public/models");
  log("info", `policy aktif: ${app.policyKey}`);
};
$("remoteUrl").onchange = (e) => { app.remote = new RemotePolicy(e.target.value); };
$("shieldOn").onchange = (e) => { app.shieldOn = e.target.checked; rover.shieldOn = e.target.checked; log("safety", `shield ${app.shieldOn ? "ON" : "OFF (hanya untuk eksperimen!)"}`); };
$("perturb").onchange = (e) => { app.perturb = e.target.checked; log("info", `gangguan aksi ${app.perturb ? "ON" : "OFF"}`); };

// environment + data tab
$("envSel").onchange = (e) => {
  if (app.mode === "live" && app.transport?.isSim) { app.envId = e.target.value; app.transport.sceneReset(e.target.value); return; }
  setEnv(e.target.value);
};
$("visualDR").onchange = (e) => { app.visualDR = e.target.checked; randomScene(); };
$("btnEnvReset").onclick = () => { agent.stop(); runner?.stop(); randomScene(); };
$("recOn").onchange = (e) => { recorder.enabled = e.target.checked; log("info", `rekam demo ${recorder.enabled ? "ON" : "OFF"}`); };
$("recClear").onclick = () => { recorder.clear(); renderRecorder(); };
$("recDownload").onclick = () => {
  const blob = new Blob([JSON.stringify(recorder.payload(app.mode))], { type: "application/json" });
  const a = document.createElement("a");
  a.href = URL.createObjectURL(blob); a.download = `qivla_demos_${Date.now()}.json`; a.click();
  setTimeout(() => URL.revokeObjectURL(a.href), 2000);
};
$("recUpload").onclick = async () => {
  const base = $("recUrl").value.replace(/\/$/, "");
  try {
    const r = await fetch(`${base}/episodes`, { method: "POST", headers: { "content-type": "application/json" }, body: JSON.stringify(recorder.payload(app.mode)) });
    const j = await r.json();
    if (!r.ok) throw new Error(j.error || r.status);
    log("info", `demo terkirim ke bridge: ${j.saved} (${j.episodes} episode, ${j.frames} frame)`);
  } catch (e) { log("error", `kirim demo gagal: ${e.message}`); }
};

async function init() {
  try {
    const idx = await loadEnvIndex();
    app.dr = await loadRandomization();
    for (const id of idx.envs) {
      const e = await loadEnv(id);
      const o = document.createElement("option"); o.value = id; o.textContent = e.name; $("envSel").appendChild(o);
    }
    await setEnv(idx.default);
  } catch (e) { log("error", `lingkungan tidak dimuat: ${e.message}`); randomScene(42); }
  loadPolicies();
  try {
    await rover.init();
    for (const id of rover.index.maps) { const m = await loadMapMeta(id); const o = document.createElement("option"); o.value = id; o.textContent = m.name; $("mapSel").appendChild(o); }
    await setMap(rover.index.default);
  } catch (e) { log("error", `peta mobile robot tidak dimuat: ${e.message}`); }
  renderRecorder();
  animate();
}
const loadMapMeta = (id) => fetch(`./maps/${id}.json`).then((r) => r.json());
init();
// for debugging / e2e tests (tick = one 50 ms control step, lets tests run faster than real time)
window.__qivla = { app, agent, shield, scene, runInstruction, perceive, randomScene, setEnv, recorder, get runner() { return runner; }, tick: controlTick, rover, roverScene, setRobot, setMap, real, realStep };
