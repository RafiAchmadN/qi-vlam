// "Riset" tab: the experiment results behind the app, drawn as dependency-free
// SVG charts from app/public/results/summary.json (scripts/export_app_results.py).
const NS = "http://www.w3.org/2000/svg";
const PAL = ["var(--c1)", "var(--c2)", "var(--c3)", "var(--c4)", "var(--c5)", "var(--c6)"];
const SETS = { train_envs: "lingkungan latih", heldout_env: "pabrik (baru)", hard_dynamics: "servo aus",
  train_maps: "peta latih", heldout_map: "RS (baru)", unknown_obstacles: "rintangan tak dikenal" };

function el(tag, attrs = {}, text = null) {
  const e = document.createElementNS(NS, tag);
  for (const [k, v] of Object.entries(attrs)) e.setAttribute(k, v);
  if (text !== null) e.textContent = text;
  return e;
}
const fmt = (v, d = 0) => (v === null || v === undefined || Number.isNaN(v) ? "–" : Number(v).toFixed(d));

/** Grouped bar chart: groups = [{label, values:[...], err:[...]}], series = [names] */
export function groupedBars(groups, series, { max = 100, unit = "%", height = 190, ticks = [0, 25, 50, 75, 100] } = {}) {
  const W = 560, H = height, L = 34, B = 42, T = 8, R = 6;
  const svg = el("svg", { viewBox: `0 0 ${W} ${H}`, class: "chart", role: "img" });
  const ih = H - B - T, gw = (W - L - R) / groups.length, bw = Math.min(18, (gw - 10) / series.length);
  for (const t of ticks) {
    const y = T + ih * (1 - t / max);
    svg.append(el("line", { x1: L, x2: W - R, y1: y, y2: y, class: t ? "grid" : "axis" }));
    svg.append(el("text", { x: L - 4, y: y + 3, "text-anchor": "end" }, `${t}${unit}`));
  }
  groups.forEach((g, gi) => {
    const x0 = L + gi * gw + (gw - bw * series.length) / 2;
    g.values.forEach((v, si) => {
      if (v === null || v === undefined) return;
      const h = ih * Math.max(0, Math.min(1, v / max)), x = x0 + si * bw, y = T + ih - h;
      const r = el("rect", { x, y, width: bw - 2, height: Math.max(h, 0.5), rx: 2, fill: PAL[si % PAL.length] });
      r.append(el("title", {}, `${g.label} · ${series[si]}: ${fmt(v, 1)}${unit}${g.err ? ` ± ${fmt(g.err[si], 1)}` : ""}`));
      svg.append(r);
      if (g.err && g.err[si]) {
        const e = ih * Math.min(g.err[si], max) / max, cx = x + (bw - 2) / 2;
        svg.append(el("line", { x1: cx, x2: cx, y1: y - e, y2: Math.min(T + ih, y + e), stroke: "var(--text-2)", "stroke-width": 1 }));
      }
    });
    const lbl = el("text", { x: L + gi * gw + gw / 2, y: H - B + 14, "text-anchor": "middle" }, g.label);
    svg.append(lbl);
    if (g.sub) svg.append(el("text", { x: L + gi * gw + gw / 2, y: H - B + 26, "text-anchor": "middle", fill: "var(--muted)" }, g.sub));
  });
  return svg;
}

function legend(series) {
  const d = document.createElement("div"); d.className = "legend";
  d.innerHTML = series.map((s, i) => `<span><i style="background:${PAL[i % PAL.length]}"></i>${s}</span>`).join("");
  return d;
}
function card(title, ...children) {
  const c = document.createElement("div"); c.className = "card";
  const h = document.createElement("h4"); h.textContent = title; c.append(h, ...children.filter(Boolean));
  return c;
}
function note(text) { const p = document.createElement("p"); p.className = "note"; p.textContent = text; return p; }
function table(head, rows, hl = () => false) {
  const t = document.createElement("table"); t.className = "data";
  t.innerHTML = `<thead><tr>${head.map((h) => `<th>${h}</th>`).join("")}</tr></thead><tbody>${rows.map((r, i) => `<tr class="${hl(r, i) ? "hl" : ""}">${r.map((c) => `<td>${c}</td>`).join("")}</tr>`).join("")}</tbody>`;
  return t;
}

const NAMES = { dense_dr: "dense + DR", dense_nodr: "dense tanpa DR", healing_ctrl: "kontrol healing", ttns_dr: "TTNS + heal",
  ttns_noheal: "TTNS tanpa heal", dense_dr_noshield: "dense, tanpa shield" };

export async function renderResearch(root, url = "./results/summary.json") {
  let R;
  try { const r = await fetch(url); if (!r.ok) throw new Error(r.status); R = await r.json(); }
  catch (e) { root.innerHTML = `<p class="muted">Hasil belum tersedia (${e.message}). Jalankan <code>python scripts/export_app_results.py</code>.</p>`; return; }
  root.innerHTML = "";
  const kpis = document.createElement("div"); kpis.className = "kpis";
  const k = (v, label) => `<div class="kpi"><b>${v}</b><span>${label}</span></div>`;
  const pp = R.pickplace?.summary, nav = R.nav?.summary;
  const items = [];
  if (pp?.dense_dr && pp?.ttns_dr) items.push(k(`${fmt(pp.dense_dr.params / pp.ttns_dr.params, 1)}×`, "lebih kecil (TTNS, pick-place) dengan sukses setara"));
  if (pp?.dense_dr && pp?.dense_nodr) items.push(k(`+${fmt(pp.dense_dr.heldout_env.success_mean - pp.dense_nodr.heldout_env.success_mean, 0)} pt`, "efek domain randomization di lingkungan baru"));
  if (R.shield) { const c = R.shield.conditions; const on = c["ttns|noise=1.5|shield=on"], off = c["ttns|noise=1.5|shield=off"]; if (on && off) items.push(k(`${off.table_collision_pct}% → ${on.table_collision_pct}%`, "tabrakan meja dengan shield Lyapunov (gangguan)")); }
  if (nav?.ttns_dr) items.push(k(`${fmt(nav.ttns_dr.unknown_obstacles.success_mean, 0)}%`, "navigasi TTNS dengan rintangan tak dikenal"));
  kpis.innerHTML = items.join(""); root.append(kpis);

  if (pp) {
    const order = ["dense_dr", "dense_nodr", "healing_ctrl", "ttns_dr", "ttns_noheal"].filter((m) => pp[m]);
    const sets = ["train_envs", "heldout_env", "hard_dynamics"];
    const ex = R.pickplace.expert;
    const groups = [{ label: "expert", sub: "referensi", values: sets.map((s) => ex[s].success_pct) },
      ...order.map((m) => ({ label: NAMES[m] || m, sub: `${(pp[m].params / 1000).toFixed(0)}k param`, values: sets.map((s) => pp[m][s].success_mean), err: sets.map((s) => pp[m][s].success_std) }))];
    root.append(card(`Lengan: pick-and-place di fisika MuJoCo (${R.pickplace.config.seeds} seed)`, legend(sets.map((s) => SETS[s])), groupedBars(groups, sets.map((s) => SETS[s]), { height: 210 }),
      note(`Dataset: ${R.pickplace.dataset.episodes} episode, ${R.pickplace.dataset.frames.toLocaleString()} frame. ${R.pickplace.scope}`)));
  }
  for (const key of ["real", "real_koch"]) {
    const rb = R[key];
    if (!rb) continue;
    const s = rb.summary, ms = Object.keys(s);
    const ds = Object.entries(rb.datasets).map(([n, d]) => `${d.robot.toUpperCase()}: ${d.task} (${d.episodes} ep)`).join(" · ");
    root.append(card(`Data robot NYATA — ${key === "real" ? "SO-100/SO-101" : "Koch v1.1"} (${rb.config.seeds} seed)`,
      legend(["galat gerakan (1 = diam)", "nMSE aksi berikutnya"]),
      groupedBars(ms.map((m) => ({ label: m.replace("cnn_", "").replace("head_", "h:"), values: [s[m].motion_nmse.mean, s[m].nmse_first.mean], err: [s[m].motion_nmse.std, s[m].nmse_first.std] })),
        ["gerakan", "aksi berikutnya"], { max: 1, unit: "", ticks: [0, 0.25, 0.5, 0.75, 1], height: 200 }),
      table(["model", "param", "KB", "galat gerakan", "MAE sendi", "gripper", "ms"], ms.map((m) => [m, s[m].params.toLocaleString(), fmt(s[m].storage_kb, 0),
        fmt(s[m].motion_nmse.mean, 3), fmt(s[m].joint_mae.mean, 2), `${fmt(100 * s[m].grip_acc.mean, 1)}%`, fmt(s[m].latency_ms.mean, 2)]), (r) => r[0] === "cnn_ttns"),
      note(`${ds}. ${rb.scope}`)));
  }
  if (R.zoo) {
    const z = R.zoo.summary, ms = Object.keys(z);
    const sets = ["train_envs", "heldout_env", "hard_dynamics"];
    root.append(card(`Perbandingan arsitektur (model zoo, ${R.zoo.config.seeds} seed)`, legend(sets.map((s) => SETS[s])),
      groupedBars(ms.map((m) => ({ label: m.replace("head_", "h:"), values: sets.map((s) => z[m][s].success_mean), err: sets.map((s) => z[m][s].success_std) })), sets.map((s) => SETS[s]), { height: 200 }),
      table(["model", "param", "KB", "ms/langkah", "latih", "baru", "aus"], ms.map((m) => [m, z[m].params.toLocaleString(), fmt(z[m].storage_kb, 0), fmt(z[m].latency_ms, 2),
        `${fmt(z[m].train_envs.success_mean)}%`, `${fmt(z[m].heldout_env.success_mean)}%`, `${fmt(z[m].hard_dynamics.success_mean)}%`]),
      (r) => r[0] === "ttns"),
      note(R.zoo.notes)));
  }
  if (nav) {
    const sets = ["train_maps", "heldout_map", "hard_dynamics", "unknown_obstacles"];
    const ex = R.nav.expert;
    root.append(card(`Mobile robot: navigasi berbahasa (${R.nav.config.seeds} seed)`,
      table(["pengendali", "param", ...sets.map((s) => SETS[s])], [["pure pursuit", "–", ...sets.map((s) => `${fmt(ex[s].success_pct)}%`)],
        ...Object.entries(nav).map(([m, v]) => [NAMES[m] || m, v.params.toLocaleString(), ...sets.map((s) => `${fmt(v[s].success_mean)}% · ${fmt(v[s].collision_mean)}% tabrak`)])]),
      note("Benchmark navigasi ini jenuh (semua pengendali ~100%): planner A* + costmap + shield memikul beban; yang terbukti adalah kompresi 2,9× tanpa kehilangan performa, bukan peringkat pengendali.")));
  }
  if (R.shield) {
    const c = R.shield.conditions;
    const keys = ["dense|noise=1.5|shield=off", "dense|noise=1.5|shield=on", "ttns|noise=1.5|shield=off", "ttns|noise=1.5|shield=on"].filter((x) => c[x]);
    root.append(card("Shield Lyapunov + fallback IK di bawah gangguan aksi", legend(["reach sukses", "tabrakan meja", "intervensi monitor"]),
      groupedBars(keys.map((x) => ({ label: x.split("|")[0], sub: x.endsWith("on") ? "shield ON" : "shield OFF", values: [c[x].reach_success_pct, c[x].table_collision_pct, c[x].monitor_intervention_pct] })), ["reach", "tabrakan", "intervensi"], { height: 180 }),
      note(R.shield.note)));
  }
  if (R.libero) {
    const L = R.libero, ks = Object.keys(L).filter((x) => !x.startsWith("head_"));
    const mx = Math.max(...ks.map((x) => L[x].nmse)) * 1.15;
    root.append(card("LIBERO (data robot nyata, offline): galat prediksi aksi nMSE",
      groupedBars(ks.map((x) => ({ label: x.replace("_chi", " χ"), sub: L[x].ratio ? `${fmt(L[x].ratio, 1)}×` : "", values: [L[x].nmse] })), ["nMSE"], { max: Math.ceil(mx * 10) / 10, unit: "", ticks: [0, 0.25, 0.5, 0.75, 1].map((t) => Math.round(t * Math.ceil(mx * 10)) / 10) }),
      note("Lebih kecil lebih baik. Tanpa gambar; bukan success rate simulator LIBERO.")));
  }
}
