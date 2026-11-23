// Mobile robot (differential-drive rover) core: map geometry, 2-D lidar ray
// casting, occupancy grid + A*, pure-pursuit expert, learned-policy features
// and the lidar safety shield. Port of src/sim/mobile/{geometry,nav}.py —
// raycast, navFeatures and shieldVW are parity-tested against Python.

export async function loadMapIndex(base = "./maps") {
  const r = await fetch(`${base}/index.json`);
  if (!r.ok) throw new Error(`maps index ${r.status}`);
  return r.json();
}
export async function loadMap(id, base = "./maps") {
  const r = await fetch(`${base}/${id}.json`);
  if (!r.ok) throw new Error(`map ${id}: ${r.status}`);
  return r.json();
}

export const wrap = (a) => ((((a + Math.PI) % (2 * Math.PI)) + 2 * Math.PI) % (2 * Math.PI)) - Math.PI;

/** [[xmin, ymin, xmax, ymax], ...] for walls + obstacles */
export function boxes(map) {
  return [...map.walls, ...map.obstacles].map((b) => [b.pos[0] - b.size[0] / 2, b.pos[1] - b.size[1] / 2, b.pos[0] + b.size[0] / 2, b.pos[1] + b.size[1] / 2]);
}

export function raycast(B, x, y, yaw, n = 36, rmax = 3.5) {
  const out = new Array(n);
  for (let k = 0; k < n; k++) {
    const a = yaw + (k * 2 * Math.PI) / n, dx = Math.cos(a), dy = Math.sin(a);
    let best = Infinity;
    for (const b of B) {
      const tx1 = (b[0] - x) / dx, tx2 = (b[2] - x) / dx, ty1 = (b[1] - y) / dy, ty2 = (b[3] - y) / dy;
      const tmin = Math.max(Math.min(tx1, tx2), Math.min(ty1, ty2));
      const tmax = Math.min(Math.max(tx1, tx2), Math.max(ty1, ty2));
      if (Number.isFinite(tmin) && tmax >= Math.max(tmin, 0)) best = Math.min(best, tmin > 0 ? tmin : 0);
    }
    out[k] = Math.min(best, rmax);
  }
  return out;
}

export function clearance(B, x, y) {
  let m = Infinity;
  for (const b of B) {
    const dx = Math.max(b[0] - x, 0, x - b[2]), dy = Math.max(b[1] - y, 0, y - b[3]);
    m = Math.min(m, Math.hypot(dx, dy));
  }
  return m;
}

export function buildGrid(map, inflate, res = 0.05) {
  const B = boxes(map);
  const nx = Math.round(map.size[0] / res), ny = Math.round(map.size[1] / res);
  const occ = new Uint8Array(nx * ny), cost = new Float32Array(nx * ny);
  for (let j = 0; j < ny; j++) for (let i = 0; i < nx; i++) {
    const d = clearance(B, (i + 0.5) * res, (j + 0.5) * res);
    occ[j * nx + i] = d < inflate ? 1 : 0;
    cost[j * nx + i] = Math.max(0.35 - d, 0) * 8;
  }
  return { occ, cost, nx, ny, res };
}

const cellOf = (g, x, y) => [Math.min(g.ny - 1, Math.max(0, Math.floor(y / g.res))), Math.min(g.nx - 1, Math.max(0, Math.floor(x / g.res)))];
const ptOf = (g, c) => [(c[1] + 0.5) * g.res, (c[0] + 0.5) * g.res];

function nearestFree(g, c) {
  if (!g.occ[c[0] * g.nx + c[1]]) return c;
  let best = null, bd = Infinity;
  for (let j = 0; j < g.ny; j++) for (let i = 0; i < g.nx; i++) {
    if (g.occ[j * g.nx + i]) continue;
    const d = (j - c[0]) ** 2 + (i - c[1]) ** 2;
    if (d < bd) { bd = d; best = [j, i]; }
  }
  return best;
}

function los(g, a, b) {
  const n = Math.max(Math.abs(b[0] - a[0]), Math.abs(b[1] - a[1])) * 2 + 1;
  for (let k = 0; k <= n; k++) {
    const u = k / n, j = Math.round(a[0] + (b[0] - a[0]) * u), i = Math.round(a[1] + (b[1] - a[1]) * u);
    if (g.occ[j * g.nx + i] || g.cost[j * g.nx + i] > 0.8) return false;
  }
  return true;
}

/** 8-connected A* with clearance cost + line-of-sight shortcutting (as geometry.plan_path). */
export function planPath(g, start, goal) {
  const s = nearestFree(g, cellOf(g, start[0], start[1])), t = nearestFree(g, cellOf(g, goal[0], goal[1]));
  if (!s || !t) return null;
  const key = (c) => c[0] * g.nx + c[1];
  const gs = new Map([[key(s), 0]]), came = new Map(), closed = new Set();
  const open = [[0, s]];
  const moves = [[-1, 0, 1], [1, 0, 1], [0, -1, 1], [0, 1, 1], [-1, -1, 1.4142], [-1, 1, 1.4142], [1, -1, 1.4142], [1, 1, 1.4142]];
  while (open.length) {
    let bi = 0;
    for (let k = 1; k < open.length; k++) if (open[k][0] < open[bi][0]) bi = k;
    const [, c] = open.splice(bi, 1)[0];
    const kc = key(c);
    if (kc === key(t)) break;
    if (closed.has(kc)) continue;
    closed.add(kc);
    for (const [dy, dx, w] of moves) {
      const n = [c[0] + dy, c[1] + dx];
      if (n[0] < 0 || n[0] >= g.ny || n[1] < 0 || n[1] >= g.nx || g.occ[key(n)]) continue;
      const ng = gs.get(kc) + w + g.cost[key(n)];
      if (ng < (gs.get(key(n)) ?? Infinity)) {
        gs.set(key(n), ng); came.set(key(n), c);
        open.push([ng + Math.hypot(n[0] - t[0], n[1] - t[1]), n]);
      }
    }
  }
  if (!gs.has(key(t))) return null;
  const cells = [t];
  while (key(cells[cells.length - 1]) !== key(s)) cells.push(came.get(key(cells[cells.length - 1])));
  cells.reverse();
  const out = [cells[0]];
  let i = 0;
  while (i < cells.length - 1) {
    let j = cells.length - 1;
    while (j > i + 1 && !los(g, cells[i], cells[j])) j--;
    out.push(cells[j]); i = j;
  }
  const pts = out.map((c) => ptOf(g, c));
  pts[0] = [start[0], start[1]]; pts[pts.length - 1] = [goal[0], goal[1]];
  return pts;
}

export function lookahead(path, x, y, L = 0.4) {
  let best = Infinity, bi = 0, bu = 0;
  for (let i = 0; i < path.length - 1; i++) {
    const a = path[i], b = path[i + 1], ab = [b[0] - a[0], b[1] - a[1]];
    const u = Math.max(0, Math.min(1, ((x - a[0]) * ab[0] + (y - a[1]) * ab[1]) / Math.max(ab[0] ** 2 + ab[1] ** 2, 1e-9)));
    const d = Math.hypot(a[0] + u * ab[0] - x, a[1] + u * ab[1] - y);
    if (d < best) { best = d; bi = i; bu = u; }
  }
  if (path.length < 2) return path[0].slice();
  let cur = [path[bi][0] + bu * (path[bi + 1][0] - path[bi][0]), path[bi][1] + bu * (path[bi + 1][1] - path[bi][1])];
  let rem = L;
  for (let i = bi; i < path.length - 1; i++) {
    const seg = [path[i + 1][0] - cur[0], path[i + 1][1] - cur[1]], ln = Math.hypot(...seg);
    if (ln >= rem) return [cur[0] + (seg[0] / Math.max(ln, 1e-9)) * rem, cur[1] + (seg[1] / Math.max(ln, 1e-9)) * rem];
    rem -= ln; cur = path[i + 1].slice();
  }
  return path[path.length - 1].slice();
}

/** == src/sim/mobile/nav.py::nav_features (25 values) */
export function navFeatures(robot, pose, scan, target, goal, vw) {
  const [x, y, yaw] = pose, rmax = robot.lidar.range_max;
  const out = [];
  for (let k = 0; k < scan.length; k += 2) out.push(Math.max(0, Math.min(1, Math.min(scan[k], scan[k + 1]) / rmax)));
  const dx = target[0] - x, dy = target[1] - y, b = wrap(Math.atan2(dy, dx) - yaw);
  out.push(Math.min(Math.hypot(dx, dy), 1), Math.sin(b), Math.cos(b));
  out.push(Math.min(Math.hypot(goal[0] - x, goal[1] - y), 3) / 3);
  out.push(vw[0] / robot.v_max, vw[1] / robot.w_max, 1);
  return out;
}

/** == nav.py::shield_vw — lidar barrier speed bound, turning always allowed */
export function shieldVW(robot, v, w, scan, dStop = 0.08, horizon = 0.6) {
  const n = scan.length, k = Math.max(1, Math.round((n * 45) / 360));
  let front = Infinity, back = Infinity;
  for (let i = 0; i <= k; i++) front = Math.min(front, scan[i]);
  for (let i = n - k; i < n; i++) front = Math.min(front, scan[i]);
  for (let i = n / 2 - k; i <= n / 2 + k; i++) back = Math.min(back, scan[i]);
  const vf = Math.max(0, (front - robot.radius - dStop) / horizon), vb = Math.max(0, (back - robot.radius - dStop) / horizon);
  const vs = Math.max(-vb, Math.min(vf, v));
  return { v: vs, w, clipped: Math.abs(vs - v) > 1e-6, front };
}

/** Static map + lidar-marked obstacles (== nav.py::Costmap, Nav2-style layering). */
export class Costmap {
  constructor(grid, robot) {
    this.g = grid; this.marks = new Uint8Array(grid.occ.length);
    const ri = Math.ceil((robot.radius + 0.06) / grid.res);
    this.disc = [];
    for (let dy = -ri; dy <= ri; dy++) for (let dx = -ri; dx <= ri; dx++) if (dx * dx + dy * dy <= ri * ri) this.disc.push([dy, dx]);
  }
  update(pose, scan, rmax = 2.5) {
    const g = this.g, [x, y, yaw] = pose, n = scan.length;
    let changed = false;
    for (let k = 0; k < n; k++) {
      const r = scan[k];
      if (r >= rmax) continue;
      const a = yaw + (2 * Math.PI * k) / n, px = x + (r + 0.03) * Math.cos(a), py = y + (r + 0.03) * Math.sin(a);
      const c = cellOf(g, px, py), ci = c[0] * g.nx + c[1];
      if (g.occ[ci] && g.cost[ci] > 0) continue;
      for (const [dy, dx] of this.disc) {
        const j = c[0] + dy, i = c[1] + dx;
        if (j < 0 || j >= g.ny || i < 0 || i >= g.nx) continue;
        if (!this.marks[j * g.nx + i]) { this.marks[j * g.nx + i] = 1; changed = true; }
      }
    }
    return changed;
  }
  grid() {
    const occ = this.g.occ.slice();
    for (let i = 0; i < occ.length; i++) if (this.marks[i]) occ[i] = 1;
    return { ...this.g, occ };
  }
  blocked(path) { return path.slice(1).some((p) => { const c = cellOf(this.g, p[0], p[1]); return this.marks[c[0] * this.g.nx + c[1]]; }); }
}

/** Pure pursuit on the A* path re-planned on the costmap (== nav.py::NavExpert). */
export class NavExpert {
  constructor(robot, grid, goal) { Object.assign(this, { robot, grid, goal, path: null, tPlan: -1e9, done: false, costmap: new Costmap(grid, robot) }); }
  target(pose, t, scan = null) {
    const [x, y] = pose;
    const changed = scan ? this.costmap.update(pose, scan) : false;
    const since = t - this.tPlan;
    let need = !this.path || since > 4.0;
    if (!need && since > 1.0) need = Math.min(...this.path.map((p) => Math.hypot(p[0] - x, p[1] - y))) > 0.5 || (changed && this.costmap.blocked(this.path));
    if (need) {
      this.tPlan = t;
      const p = planPath(this.costmap.grid(), [x, y], this.goal) || planPath(this.grid, [x, y], this.goal);
      if (p) this.path = p;
    }
    return this.path ? lookahead(this.path, x, y, 0.4) : this.goal.slice();
  }
  act(pose, scan, t) {
    const R = this.robot, tgt = this.target(pose, t, scan), [x, y, yaw] = pose;
    const gd = Math.hypot(this.goal[0] - x, this.goal[1] - y);
    if (gd < 0.12) { this.done = true; return { v: 0, w: 0, target: tgt }; }
    const a = wrap(Math.atan2(tgt[1] - y, tgt[0] - x) - yaw);
    const w = Math.max(-R.w_max, Math.min(R.w_max, 2.5 * a));
    let v = R.v_max * Math.max(Math.cos(a), 0) ** 2 * Math.min(1, gd / 0.5 + 0.2);
    const n = scan.length, k = Math.floor(n / 8);
    let front = Infinity;
    for (let i = 0; i <= k; i++) front = Math.min(front, scan[i]);
    for (let i = n - k; i < n; i++) front = Math.min(front, scan[i]);
    v *= Math.max(0, Math.min(1, (front - R.radius - 0.05) / 0.35));
    return { v, w, target: tgt };
  }
}

/** Kinematic rover (unicycle + motor lag) used by the app's simulation mode. */
export class RoverModel {
  constructor(robot, map, pose, unknown = []) {
    const B = boxes(map).concat(unknown.map((o) => [o.pos[0] - o.size[0] / 2, o.pos[1] - o.size[1] / 2, o.pos[0] + o.size[0] / 2, o.pos[1] + o.size[1] / 2]));
    Object.assign(this, { robot, map, B, pose: pose.slice(), cmd: [0, 0], last: [0, 0], collisions: 0, tau: map.dynamics.motor_tau || 0.08 });
  }
  scan() { return raycast(this.B, this.pose[0], this.pose[1], this.pose[2], this.robot.lidar.beams, this.robot.lidar.range_max); }
  step(v, w, dt = 0.05) {
    const R = this.robot;
    v = Math.max(-R.v_max, Math.min(R.v_max, v)); w = Math.max(-R.w_max, Math.min(R.w_max, w));
    this.last = [v, w];
    const a = Math.min(dt / this.tau, 1);
    this.cmd = [this.cmd[0] + (v - this.cmd[0]) * a, this.cmd[1] + (w - this.cmd[1]) * a];
    const [x, y, yaw] = this.pose;
    const nx = x + this.cmd[0] * Math.cos(yaw) * dt, ny = y + this.cmd[0] * Math.sin(yaw) * dt;
    if (clearance(this.B, nx, ny) > R.radius) { this.pose[0] = nx; this.pose[1] = ny; } else this.collisions++;
    this.pose[2] = wrap(yaw + this.cmd[1] * dt);
  }
}

/** "pergi ke dapur" / "go to the kitchen" -> place id (longest alias match). */
export function groundPlace(text, map) {
  const t = ` ${text.toLowerCase().normalize("NFKD").replace(/[^a-z0-9\s]/g, " ").replace(/\s+/g, " ").trim()} `;
  let best = null, bl = 0;
  for (const p of map.places) for (const n of [...p.names_id, ...p.names_en]) {
    if (t.includes(` ${n} `) && n.length > bl) { best = p; bl = n.length; }
  }
  if (!best && /\b(home|pulang|dock|charge|cas)\b/.test(t)) best = map.places.find((p) => p.id === "dock");
  return best;
}

/** Random start pose + unknown obstacles on the route (as geometry.sample_nav_episode). */
export function sampleNavScene(map, grid, robot, rnd, goalId = null, nUnknown = 0) {
  const B = boxes(map);
  for (let tries = 0; tries < 300; tries++) {
    const x = 0.3 + rnd() * (map.size[0] - 0.6), y = 0.3 + rnd() * (map.size[1] - 0.6);
    if (grid.occ[cellOf(grid, x, y)[0] * grid.nx + cellOf(grid, x, y)[1]] || clearance(B, x, y) < 0.25) continue;
    const place = goalId ? map.places.find((p) => p.id === goalId) : map.places[Math.floor(rnd() * map.places.length)];
    if (Math.hypot(place.pos[0] - x, place.pos[1] - y) < 1.0) continue;
    const path = planPath(grid, [x, y], place.pos);
    if (!path) continue;
    const unknown = [];
    const L = path.slice(1).reduce((s, p, i) => s + Math.hypot(p[0] - path[i][0], p[1] - path[i][1]), 0);
    for (let k = 0; k < 40 * nUnknown && unknown.length < nUnknown; k++) {
      let s = (0.25 + 0.5 * rnd()) * L, i = 0;
      while (i < path.length - 2 && s > Math.hypot(path[i + 1][0] - path[i][0], path[i + 1][1] - path[i][1])) { s -= Math.hypot(path[i + 1][0] - path[i][0], path[i + 1][1] - path[i][1]); i++; }
      const seg = Math.max(Math.hypot(path[i + 1][0] - path[i][0], path[i + 1][1] - path[i][1]), 1e-9), u = Math.min(1, s / seg);
      const c = [path[i][0] + u * (path[i + 1][0] - path[i][0]) + (rnd() - 0.5) * 0.24, path[i][1] + u * (path[i + 1][1] - path[i][1]) + (rnd() - 0.5) * 0.24];
      if (Math.hypot(c[0] - x, c[1] - y) < 0.6 || Math.hypot(c[0] - place.pos[0], c[1] - place.pos[1]) < 0.6 || clearance(B, c[0], c[1]) < 0.25) continue;
      unknown.push({ name: `unknown${unknown.length}`, pos: c, size: [0.15 + rnd() * 0.2, 0.15 + rnd() * 0.2], height: 0.5, color: "#8a8f96" });
    }
    return { start: [x, y, (rnd() - 0.5) * 2 * Math.PI], place, unknown };
  }
  throw new Error("cannot sample a navigation scene");
}
