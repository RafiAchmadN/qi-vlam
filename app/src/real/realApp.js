// "Data robot nyata" mode: replays held-out episodes recorded on PHYSICAL
// SO-100/SO-101 arms. Every 100 ms the on-device vision policy (trained on the
// real demonstrations, optionally TTNS-compressed) looks at the real camera
// frame + measured joints and computes the next joint targets; the app shows
// them against what the human teleoperator actually commanded.
import { VisionPolicy } from "./visionPolicy.js";

const NS = "http://www.w3.org/2000/svg";
const JOINTS = ["pan", "lift", "elbow", "wrist flex", "wrist roll", "gripper"];

function decodePng(b64, size) {
  return new Promise((resolve, reject) => {
    const img = new Image();
    img.onload = () => {
      const c = document.createElement("canvas"); c.width = size; c.height = size;
      const g = c.getContext("2d", { willReadFrequently: true });
      g.drawImage(img, 0, 0);
      resolve({ img, rgba: g.getImageData(0, 0, size, size).data });
    };
    img.onerror = reject;
    img.src = `data:image/png;base64,${b64}`;
  });
}

export class RealApp {
  constructor({ log }) {
    Object.assign(this, { log, index: null, models: {}, modelKey: null, ep: null, frames: [], t: 0, playing: false,
      chunks: [], pred: [], lat: [], status: "idle" });
  }
  async init() {
    const r = await fetch("./real/index.json");
    if (!r.ok) throw new Error(`real/index.json ${r.status}`);
    this.index = await r.json();
    for (const k of ["cnn_ttns", "cnn_mlp"]) {
      try { this.models[k] = await VisionPolicy.load(`./models/real_${k}.json`); } catch { /* optional */ }
    }
    this.modelKey = this.models.cnn_ttns ? "cnn_ttns" : Object.keys(this.models)[0];
    if (!this.modelKey) throw new Error("no real-data model exported (python -m src.real.export_web)");
  }
  get model() { return this.models[this.modelKey]; }
  async loadEpisode(file) {
    const r = await fetch(`./real/${file}`);
    this.ep = await r.json();
    this.frames = await Promise.all(this.ep.frames.map((b) => decodePng(b, this.model.size)));
    this.reset();
    this.log("info", `episode nyata: ${this.ep.robot.toUpperCase()} · "${this.ep.task}" · ${this.frames.length} frame @ ${this.ep.fps} Hz`);
  }
  reset() { this.t = 0; this.chunks = []; this.pred = []; this.lat = []; this.status = "idle"; this.playing = false; }

  /** One control step on frame t: model -> temporally-ensembled action. */
  step() {
    if (!this.ep || this.t >= this.frames.length) { this.playing = false; this.status = "done"; return null; }
    const m = this.model, t0 = performance.now();
    const { chunk } = m.act(this.frames[this.t].rgba, this.ep.state[this.t], this.ep.dataset);
    this.chunks = [...this.chunks, chunk].slice(-m.k);
    // ACT temporal ensembling: prediction for "now" from each past chunk, oldest weighted most
    let wsum = 0; const a = new Array(6).fill(0);
    this.chunks.forEach((c, j) => {
      const off = this.chunks.length - 1 - j;
      if (off >= c.length) return;
      const w = Math.exp(-0.1 * j); wsum += w;
      for (let i = 0; i < 6; i++) a[i] += w * c[off][i];
    });
    const act = a.map((v) => v / wsum);
    this.lat.push(performance.now() - t0);
    this.pred.push(act);
    this.t++;
    this.status = this.t >= this.frames.length ? "done" : "running";
    return act;
  }
  metrics() {
    const n = this.pred.length;
    if (!n) return null;
    let mae = 0, grip = 0;
    const g = this.model.spec.norm[this.ep.dataset].action, mid = 0.5 * (g.min[5] + g.max[5]);
    for (let i = 0; i < n; i++) {
      const p = this.pred[i], y = this.ep.action[i];
      for (let j = 0; j < 5; j++) mae += Math.abs(p[j] - y[j]) / (5 * n);
      grip += ((p[5] > mid) === (y[5] > mid)) / n;
    }
    const lat = this.lat.slice().sort((x, y) => x - y);
    return { mae, grip, latP50: lat[Math.floor(n / 2)], n };
  }
  /** SVG: predicted (accent) vs teleoperator (grey) trajectory per joint */
  chart() {
    const W = 560, H = 240, rows = 3, cols = 2, pw = W / cols, ph = H / rows;
    const svg = document.createElementNS(NS, "svg");
    svg.setAttribute("viewBox", `0 0 ${W} ${H}`); svg.setAttribute("class", "chart");
    if (!this.ep) return svg;
    const T = this.ep.action.length;
    for (let j = 0; j < 6; j++) {
      const ox = (j % cols) * pw, oy = Math.floor(j / cols) * ph;
      const ys = this.ep.action.map((a) => a[j]).concat(this.pred.map((a) => a[j]));
      const lo = Math.min(...ys), hi = Math.max(...ys), span = Math.max(hi - lo, 1e-3);
      const X = (i) => ox + 6 + (i / Math.max(T - 1, 1)) * (pw - 12), Y = (v) => oy + 16 + (1 - (v - lo) / span) * (ph - 24);
      const line = (arr, color, wdt) => {
        if (arr.length < 2) return;
        const pl = document.createElementNS(NS, "polyline");
        pl.setAttribute("points", arr.map((a, i) => `${X(i).toFixed(1)},${Y(a[j]).toFixed(1)}`).join(" "));
        pl.setAttribute("fill", "none"); pl.setAttribute("stroke", color); pl.setAttribute("stroke-width", wdt);
        svg.append(pl);
      };
      const lbl = document.createElementNS(NS, "text"); lbl.setAttribute("x", ox + 6); lbl.setAttribute("y", oy + 11); lbl.textContent = JOINTS[j];
      svg.append(lbl);
      line(this.ep.action, "var(--c6)", 2.5);
      line(this.pred, "var(--accent)", 1.6);
      if (this.t > 0) {
        const cur = document.createElementNS(NS, "line");
        cur.setAttribute("x1", X(this.t - 1)); cur.setAttribute("x2", X(this.t - 1)); cur.setAttribute("y1", oy + 14); cur.setAttribute("y2", oy + ph - 6);
        cur.setAttribute("stroke", "var(--warm)"); cur.setAttribute("stroke-dasharray", "2 2"); svg.append(cur);
      }
    }
    return svg;
  }
}
