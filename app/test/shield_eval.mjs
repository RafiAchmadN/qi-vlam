// Closed-loop evaluation of the on-device stack exactly as the app runs it
// (JS policy + SafetyShield), written to ../results/app_shield_eval.json.
//   node test/shield_eval.mjs
// Conditions: {dense, ttns} x {nominal, action-noise} x {shield on, off}.
import { readFileSync, writeFileSync, mkdirSync } from "node:fs";
import { SafetyShield } from "../src/safety/shield.js";
import { fk, dist, clampQ, Q_HOME, WORKSPACE_MIN } from "../src/robot/kinematics.js";
import { ReachPolicy } from "../src/vla/policy.js";

const N = 300, STEPS = 100, DT = 0.05;
function rng(seed) { let s = seed; return () => ((s = (s * 16807) % 2147483647) / 2147483647); }

function run(policy, { noise, shieldOn, seed }) {
  const rnd = rng(seed);
  let reached = 0, fallbackEpisodes = 0, tableHits = 0, steps = [];
  for (let e = 0; e < N; e++) {
    const sh = new SafetyShield({ P: policy.spec.lyapunov.P });
    let q = Q_HOME.slice(); q[0] += (rnd() - 0.5) * 1.2;
    const t = [0.1 + rnd() * 0.18, -0.2 + rnd() * 0.4, 0.02 + rnd() * 0.18];
    let used = false, hit = false, firstReach = -1;
    for (let k = 0; k < STEPS; k++) {
      let a = policy.act(q, t);
      if (noise) a = a.map((v) => v + (rnd() - 0.5) * 2 * noise);
      const r = shieldOn ? sh.filter(q, t, a) : { qd: a, source: "policy" };
      if (r.source !== "policy") used = true;
      for (let i = 0; i < 4; i++) q[i] += r.qd[i] * DT;
      q = clampQ(q);
      if (fk(q)[2] < WORKSPACE_MIN[2]) hit = true;
      if (firstReach < 0 && dist(fk(q), t) < 0.01) firstReach = k + 1;
    }
    if (dist(fk(q), t) < 0.01) reached++;
    if (used) fallbackEpisodes++;
    if (hit) tableHits++;
    if (firstReach > 0) steps.push(firstReach);
  }
  steps.sort((a, b) => a - b);
  return {
    reach_success_pct: (100 * reached) / N, monitor_intervention_pct: (100 * fallbackEpisodes) / N,
    table_collision_pct: (100 * tableHits) / N, steps_to_reach_median: steps[Math.floor(steps.length / 2)] ?? null,
  };
}

const out = { n_episodes: N, steps: STEPS, dt: DT, conditions: {} };
for (const name of ["dense", "ttns"]) {
  const p = new ReachPolicy(JSON.parse(readFileSync(new URL(`../public/models/policy_${name}.json`, import.meta.url))));
  for (const noise of [0, 1.5]) for (const shieldOn of [true, false]) {
    const key = `${name}|noise=${noise}|shield=${shieldOn ? "on" : "off"}`;
    out.conditions[key] = run(p, { noise, shieldOn, seed: 4242 });
    console.log(key.padEnd(32), JSON.stringify(out.conditions[key]));
  }
}
out.note = "Same seeds across conditions (paired). noise = uniform action noise amplitude in rad/s added to the policy output.";
mkdirSync(new URL("../../results/", import.meta.url), { recursive: true });
writeFileSync(new URL("../../results/app_shield_eval.json", import.meta.url), JSON.stringify(out, null, 2));
