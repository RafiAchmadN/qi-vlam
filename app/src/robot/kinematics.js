// JS port of src/robot/arm.py — keep the two in lock-step.
// test/kinematics.test.mjs checks this file against vectors exported by Python.

export const JOINT_NAMES = ["shoulder_pan", "shoulder_lift", "elbow_flex", "wrist_flex", "wrist_roll", "gripper"];
export const H0 = 0.10, L1 = 0.116, L2 = 0.135, L3 = 0.100;
export const Q_MIN = [-2.0, -1.6, -0.2, -1.9, -2.8, 0.0];
export const Q_MAX = [2.0, 1.6, 2.6, 1.9, 2.8, 1.0];
export const QD_MAX = [1.5, 1.5, 1.5, 2.0, 2.0, 2.0];
export const Q_HOME = [0.0, -0.2, 1.6, 1.1, 0.0, 1.0];
export const WORKSPACE_MIN = [0.08, -0.25, 0.005];
export const WORKSPACE_MAX = [0.32, 0.25, 0.30];

export function fk(q) {
  const a1 = q[1], a2 = q[1] + q[2], a3 = q[1] + q[2] + q[3];
  const r = L1 * Math.sin(a1) + L2 * Math.sin(a2) + L3 * Math.sin(a3);
  const z = H0 + L1 * Math.cos(a1) + L2 * Math.cos(a2) + L3 * Math.cos(a3);
  return [r * Math.cos(q[0]), r * Math.sin(q[0]), z];
}

/** World positions of base, shoulder, elbow, wrist, tip — for rendering. */
export function jointPositions(q) {
  const pts = [[0, 0], [0, H0]];
  let a = 0, r = 0, z = H0;
  for (const [L, dq] of [[L1, q[1]], [L2, q[2]], [L3, q[3]]]) {
    a += dq; r += L * Math.sin(a); z += L * Math.cos(a);
    pts.push([r, z]);
  }
  return pts.map(([rr, zz]) => [rr * Math.cos(q[0]), rr * Math.sin(q[0]), zz]);
}

export function jacobian(q) {
  const a1 = q[1], a2 = q[1] + q[2], a3 = q[1] + q[2] + q[3];
  const r = L1 * Math.sin(a1) + L2 * Math.sin(a2) + L3 * Math.sin(a3);
  const dr = [L1 * Math.cos(a1) + L2 * Math.cos(a2) + L3 * Math.cos(a3),
    L2 * Math.cos(a2) + L3 * Math.cos(a3), L3 * Math.cos(a3)];
  const dz = [-(L1 * Math.sin(a1) + L2 * Math.sin(a2) + L3 * Math.sin(a3)),
    -(L2 * Math.sin(a2) + L3 * Math.sin(a3)), -L3 * Math.sin(a3)];
  const c = Math.cos(q[0]), s = Math.sin(q[0]);
  return [
    [-r * s, dr[0] * c, dr[1] * c, dr[2] * c],
    [r * c, dr[0] * s, dr[1] * s, dr[2] * s],
    [0, dz[0], dz[1], dz[2]],
  ];
}

function solve3(A, b) { // Cramer's rule, A 3x3
  const det = (m) => m[0][0] * (m[1][1] * m[2][2] - m[1][2] * m[2][1])
    - m[0][1] * (m[1][0] * m[2][2] - m[1][2] * m[2][0])
    + m[0][2] * (m[1][0] * m[2][1] - m[1][1] * m[2][0]);
  const D = det(A);
  return [0, 1, 2].map((i) => det(A.map((row, r) => row.map((v, c) => (c === i ? b[r] : v)))) / D);
}

/** Damped-least-squares IK velocity step — the verified fallback controller.
 *  Same math as arm.ik_step (without the null-space pitch term's pinv, which
 *  is replaced by a projected gradient; differences are < 1e-3 rad/s typical). */
export function ikStep(q, target, pitch = 2.8, gain = 1.0, damping = 0.02, dt = 0.05) {
  const J = jacobian(q);
  const ee = fk(q);
  const e = [0, 1, 2].map((i) => target[i] - ee[i]);
  const JJt = [0, 1, 2].map((i) => [0, 1, 2].map((j) =>
    J[i].reduce((s, v, k) => s + v * J[j][k], 0) + (i === j ? damping * damping : 0)));
  const y = solve3(JJt, e.map((v) => (gain * v / dt) * 0.2));
  const qd = [0, 1, 2, 3].map((k) => J[0][k] * y[0] + J[1][k] * y[1] + J[2][k] * y[2]);
  // secondary task: keep tool pitch, projected so it doesn't disturb position
  const g = [0, 1, 1, 1].map((v) => v * (pitch - (q[1] + q[2] + q[3])) * 2.0);
  const Jg = [0, 1, 2].map((i) => J[i].reduce((s, v, k) => s + v * g[k], 0));
  const z = solve3(JJt, Jg);
  for (let k = 0; k < 4; k++) qd[k] += g[k] - (J[0][k] * z[0] + J[1][k] * z[1] + J[2][k] * z[2]);
  return qd.map((v, k) => Math.max(-QD_MAX[k], Math.min(QD_MAX[k], v)));
}

export const clampQ = (q) => q.map((v, i) => Math.max(Q_MIN[i], Math.min(Q_MAX[i], v)));
export const inWorkspace = (p) => p.every((v, i) => v >= WORKSPACE_MIN[i] - 1e-9 && v <= WORKSPACE_MAX[i] + 1e-9);
export const dist = (a, b) => Math.hypot(a[0] - b[0], a[1] - b[1], a[2] - b[2]);
