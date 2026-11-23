// Environment presets ("places") shared with Python (src/sim/envs.py):
// app/public/envs/*.json. The app renders them; Python turns the same files
// into MuJoCo scenes and training data.

export async function loadEnvIndex(base = "./envs") {
  const r = await fetch(`${base}/index.json`);
  if (!r.ok) throw new Error(`envs index ${r.status}`);
  return r.json();
}

export async function loadEnv(id, base = "./envs") {
  const r = await fetch(`${base}/${id}.json`);
  if (!r.ok) throw new Error(`env ${id}: ${r.status}`);
  return r.json();
}

export async function loadRandomization(base = "./envs") {
  const r = await fetch(`${base}/randomization.json`);
  return r.ok ? r.json() : null;
}

/** Seeded PRNG (Park-Miller), same as the app's scene randomizer. */
export function rng(seed) {
  let s = Math.max(1, Math.floor(seed) % 2147483647);
  return () => ((s = (s * 16807) % 2147483647) / 2147483647);
}

const uni = (r, [a, b]) => a + (b - a) * r();

/** Visual domain randomization: lighting, table tint, camera mis-calibration. */
export function sampleVisualDR(dr, r) {
  if (!dr) return null;
  const v = dr.visual;
  return {
    light: uni(r, v.light.scale),
    tableTint: uni(r, v.table_tint.add),
    camShift: [uni(r, v.camera_shift_m.add), uni(r, v.camera_shift_m.add)],
    camRot: uni(r, v.camera_rot_rad.add),
  };
}

export function insideXY(p, ob, margin = 0) {
  return Math.abs(p[0] - ob.pos[0]) < ob.size[0] / 2 + margin && Math.abs(p[1] - ob.pos[1]) < ob.size[1] / 2 + margin;
}

/** Random non-overlapping object layout inside the env's spawn region. */
export function sampleLayout(env, r) {
  const { spawn } = env;
  const solid = env.obstacles.filter((o) => !o.visual_only);
  for (let attempt = 0; attempt < 50; attempt++) {
    const placed = [];
    const spot = () => {
      for (let k = 0; k < 60; k++) {
        const rad = uni(r, spawn.r), a = uni(r, spawn.a);
        const p = [rad * Math.cos(a), rad * Math.sin(a)];
        if (placed.some((o) => Math.hypot(o[0] - p[0], o[1] - p[1]) < spawn.min_sep)) continue;
        if (solid.some((ob) => insideXY(p, ob, 0.04))) continue;
        placed.push(p);
        return p;
      }
      return null;
    };
    const cubes = {}, pads = {};
    let ok = true;
    const [lo, hi] = env.n_objects || [env.cubes.length, env.cubes.length];
    const n = Math.max(1, lo + Math.floor(r() * (hi - lo + 1)) - Math.floor(attempt / 20));
    const pool = env.cubes.slice().sort(() => r() - 0.5).slice(0, n);
    const chosen = env.cubes.filter((c) => pool.includes(c));
    for (const c of chosen) { const p = spot(); if (!p) { ok = false; break; } cubes[c] = { pos: p, yaw: uni(r, [-Math.PI / 4, Math.PI / 4]), shape: (env.shapes || {})[c] || "cube" }; }
    if (ok) for (const c of env.pads) { const p = spot(); if (!p) { ok = false; break; } pads[c] = { pos: p }; }
    if (ok) return { cubes, pads };
  }
  throw new Error(`cannot lay out env ${env.id}`);
}
