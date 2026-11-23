// On-device runtime of the vision policy trained on REAL robot data
// (src/real/models.py::Policy "cnn_mlp"/"cnn_ttns", exported by src/real/export_web.py).
// Pure JS: conv (stride/pad) + exact GELU, spatial-softmax keypoints, mean pool,
// then the same dense/TTNS head layers as the other app policies.
import { forwardLayers } from "../vla/policy.js";

// erf approximation (Abramowitz-Stegun 7.1.26, |err| < 1.5e-7) for exact GELU
function erf(x) {
  const s = Math.sign(x); x = Math.abs(x);
  const t = 1 / (1 + 0.3275911 * x);
  const y = 1 - (((((1.061405429 * t - 1.453152027) * t) + 1.421413741) * t - 0.284496736) * t + 0.254829592) * t * Math.exp(-x * x);
  return s * y;
}
const gelu = (x) => 0.5 * x * (1 + erf(x / Math.SQRT2));

/** Flatten nested conv weights once: W[o][c][ky][kx] -> Float32Array (o, c, ky, kx). */
function packConv(L) {
  const O = L.W.length, C = L.W[0].length, K = L.W[0][0].length;
  const w = new Float32Array(O * C * K * K);
  let i = 0;
  for (let o = 0; o < O; o++) for (let c = 0; c < C; c++) for (let y = 0; y < K; y++) for (let x = 0; x < K; x++) w[i++] = L.W[o][c][y][x];
  return { w, b: Float32Array.from(L.b), O, C, K, s: L.stride, p: L.pad };
}

/** conv2d + GELU on CHW Float32Array via im2col + GEMM: long contiguous inner loops
 *  (Ho*Wo) instead of short strided ones — ~5-10x faster in JS engines. */
function conv(inp, C, H, Wd, P) {
  const { w, b, O, K, s, p } = P;
  const Ho = Math.floor((H + 2 * p - K) / s) + 1, Wo = Math.floor((Wd + 2 * p - K) / s) + 1, N = Ho * Wo;
  const CKK = C * K * K;
  if (!P.cols || P.cols.length !== CKK * N) P.cols = new Float32Array(CKK * N);
  const cols = P.cols;
  for (let c = 0; c < C; c++) for (let ky = 0; ky < K; ky++) for (let kx = 0; kx < K; kx++) {
    const r = ((c * K + ky) * K + kx) * N, base = c * H * Wd;
    for (let y = 0; y < Ho; y++) {
      const iy = y * s - p + ky, rr = r + y * Wo;
      if (iy < 0 || iy >= H) { cols.fill(0, rr, rr + Wo); continue; }
      const rb = base + iy * Wd;
      for (let x = 0, ix = kx - p; x < Wo; x++, ix += s) cols[rr + x] = ix >= 0 && ix < Wd ? inp[rb + ix] : 0;
    }
  }
  const out = new Float32Array(O * N);
  for (let o = 0; o < O; o++) {
    const ob = o * N, wb = o * CKK;
    out.fill(b[o], ob, ob + N);
    for (let k = 0; k < CKK; k++) {
      const wv = w[wb + k], cb = k * N;
      if (wv === 0) continue;
      for (let n = 0; n < N; n++) out[ob + n] += wv * cols[cb + n];
    }
    for (let n = ob; n < ob + N; n++) out[n] = gelu(out[n]);
  }
  return { t: out, C: O, H: Ho, W: Wo };
}

export class VisionPolicy {
  constructor(spec) {
    if (spec.format !== "qi-vla-vision/1") throw new Error("not a qi-vla-vision/1 model");
    Object.assign(this, { spec, name: spec.kind, params: spec.params, k: spec.k, size: spec.size });
    this.convs = spec.conv.map(packConv);
  }
  static async load(url) { const r = await fetch(url); if (!r.ok) throw new Error(`vision model ${r.status}`); return new VisionPolicy(await r.json()); }

  /** rgba: Uint8ClampedArray size*size*4 (row 0 = top). Returns K x 6 normalized chunk. */
  forward(rgba, stateNorm, taskIndex) {
    const S = this.size;
    let f = { t: new Float32Array(3 * S * S), C: 3, H: S, W: S };
    for (let i = 0; i < S * S; i++) for (let c = 0; c < 3; c++) f.t[c * S * S + i] = rgba[i * 4 + c] / 255;
    for (const P of this.convs) f = conv(f.t, f.C, f.H, f.W, P);
    const feats = [], hw = f.H * f.W, ys = [], xs = [];
    for (let i = 0; i < f.H; i++) ys.push(f.H > 1 ? -1 + (2 * i) / (f.H - 1) : 0);
    for (let j = 0; j < f.W; j++) xs.push(f.W > 1 ? -1 + (2 * j) / (f.W - 1) : 0);
    const ky = [], kx = [], mean = [];
    for (let c = 0; c < f.C; c++) {
      let mx = -Infinity, sum = 0, sy = 0, sx = 0, m = 0;
      for (let i = 0; i < hw; i++) mx = Math.max(mx, f.t[c * hw + i]);
      for (let i = 0; i < f.H; i++) for (let j = 0; j < f.W; j++) {
        const v = f.t[c * hw + i * f.W + j], e = Math.exp(v - mx);
        sum += e; sy += e * ys[i]; sx += e * xs[j]; m += v;
      }
      ky.push(sy / sum); kx.push(sx / sum); mean.push(m / hw);
    }
    feats.push(...ky, ...kx, ...mean, ...stateNorm);
    for (let t = 0; t < this.spec.names.length; t++) feats.push(t === taskIndex ? 1 : 0);
    const y = forwardLayers(this.spec.head, feats);
    const out = [];
    for (let j = 0; j < this.k; j++) out.push(y.slice(j * 6, j * 6 + 6));
    return out;
  }

  /** Full step in dataset units: returns {action (6), chunk (K x 6)} */
  act(rgba, state, dataset) {
    const n = this.spec.norm[dataset], ti = this.spec.names.indexOf(dataset);
    const sn = state.map((v, i) => (v - n.state.mean[i]) / n.state.std[i]);
    const ch = this.forward(rgba, sn, ti).map((a) => a.map((v, i) => v * n.action.std[i] + n.action.mean[i]));
    return { action: ch[0], chunk: ch };
  }
}
