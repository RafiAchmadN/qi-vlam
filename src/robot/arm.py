"""Canonical kinematic model of the 6-servo desktop arm used by the QI-VLA app.

The geometry is a simplified SO-100/SO-101-style arm (the open-source LeRobot
arm): shoulder_pan, shoulder_lift, elbow_flex, wrist_flex, wrist_roll, gripper.
Only the first four joints affect end-effector *position*; wrist_roll and the
gripper are handled by the task planner.

This module is the single source of truth: `app/src/robot/kinematics.js`
re-implements exactly the same equations, and `tests/test_robot.py` checks the
JS port against vectors exported from here (`app/public/models/fk_vectors.json`).

Convention (all angles in radians):
    q0 = shoulder_pan   rotation about world z
    q1 = shoulder_lift  0 = upper arm pointing straight up, + leans forward
    q2 = elbow_flex     relative to upper arm
    q3 = wrist_flex     relative to forearm
    r  = L1 sin(q1) + L2 sin(q1+q2) + L3 sin(q1+q2+q3)
    z  = H0 + L1 cos(q1) + L2 cos(q1+q2) + L3 cos(q1+q2+q3)
    x  = r cos(q0),  y = r sin(q0)

These are nominal link lengths, not a calibrated model of any specific robot;
the bridge (`src/bridge/drivers.py`) maps them to real servo units with a
per-joint offset/sign/scale calibration table.
"""

from __future__ import annotations

import numpy as np

JOINT_NAMES = ["shoulder_pan", "shoulder_lift", "elbow_flex", "wrist_flex",
               "wrist_roll", "gripper"]
H0 = 0.10   # base -> shoulder_lift axis height (m)
L1 = 0.116  # upper arm
L2 = 0.135  # forearm
L3 = 0.100  # wrist_flex axis -> gripper tip
LINKS = (H0, L1, L2, L3)

# Joint limits (rad); gripper in [0 closed, 1 open] normalized units.
Q_MIN = np.array([-2.0, -1.6, -0.2, -1.9, -2.8, 0.0])
Q_MAX = np.array([2.0, 1.6, 2.6, 1.9, 2.8, 1.0])
QD_MAX = np.array([1.5, 1.5, 1.5, 2.0, 2.0, 2.0])  # rad/s velocity limits
Q_HOME = np.array([0.0, -0.2, 1.6, 1.1, 0.0, 1.0])

# Reachable table workspace (m) the planner / safety shield allow.
WORKSPACE_MIN = np.array([0.08, -0.25, 0.005])
WORKSPACE_MAX = np.array([0.32, 0.25, 0.30])


def fk(q: np.ndarray) -> np.ndarray:
    """End-effector (gripper tip) position for joint vector(s) q[..., >=4]."""
    q = np.asarray(q, dtype=np.float64)
    q0, q1, q2, q3 = q[..., 0], q[..., 1], q[..., 2], q[..., 3]
    a1, a2, a3 = q1, q1 + q2, q1 + q2 + q3
    r = L1 * np.sin(a1) + L2 * np.sin(a2) + L3 * np.sin(a3)
    z = H0 + L1 * np.cos(a1) + L2 * np.cos(a2) + L3 * np.cos(a3)
    return np.stack([r * np.cos(q0), r * np.sin(q0), z], axis=-1)


def joint_positions(q: np.ndarray) -> np.ndarray:
    """World positions of base, shoulder, elbow, wrist, tip (5, 3) for rendering."""
    q0, q1, q2, q3 = q[0], q[1], q[2], q[3]
    pts = [(0.0, 0.0), (0.0, H0)]
    a, r, z = 0.0, 0.0, H0
    for L, dq in ((L1, q1), (L2, q2), (L3, q3)):
        a += dq
        r += L * np.sin(a)
        z += L * np.cos(a)
        pts.append((r, z))
    return np.array([[r * np.cos(q0), r * np.sin(q0), z] for r, z in pts])


def jacobian(q: np.ndarray) -> np.ndarray:
    """Analytic 3x4 position Jacobian d fk / d q[0:4]."""
    q0, q1, q2, q3 = q[0], q[1], q[2], q[3]
    a1, a2, a3 = q1, q1 + q2, q1 + q2 + q3
    r = L1 * np.sin(a1) + L2 * np.sin(a2) + L3 * np.sin(a3)
    dr = np.array([L1 * np.cos(a1) + L2 * np.cos(a2) + L3 * np.cos(a3),
                   L2 * np.cos(a2) + L3 * np.cos(a3),
                   L3 * np.cos(a3)])
    dz = -np.array([L1 * np.sin(a1) + L2 * np.sin(a2) + L3 * np.sin(a3),
                    L2 * np.sin(a2) + L3 * np.sin(a3),
                    L3 * np.sin(a3)])
    c, s = np.cos(q0), np.sin(q0)
    J = np.zeros((3, 4))
    J[0, 0], J[1, 0] = -r * s, r * c
    J[0, 1:], J[1, 1:], J[2, 1:] = dr * c, dr * s, dz
    return J


def ik_step(q: np.ndarray, target: np.ndarray, pitch: float | None = None,
            gain: float = 1.0, damping: float = 0.02, dt: float = 0.05) -> np.ndarray:
    """One damped-least-squares IK velocity step (the 'expert' controller).

    Returns joint velocities for q[0:4] (rad/s), clipped to QD_MAX. If `pitch`
    is given, a secondary task keeps the tool pitch (q1+q2+q3) near it, which
    produces natural top-down grasps (pitch = pi).
    """
    J = jacobian(q)
    e = target - fk(q)
    JJt = J @ J.T + damping ** 2 * np.eye(3)
    qd = J.T @ np.linalg.solve(JJt, gain * e / dt * 0.2)
    if pitch is not None:
        # null-space projection of the pitch objective
        N = np.eye(4) - np.linalg.pinv(J) @ J
        g = np.array([0.0, 1.0, 1.0, 1.0]) * (pitch - (q[1] + q[2] + q[3]))
        qd = qd + N @ (2.0 * g)
    return np.clip(qd, -QD_MAX[:4], QD_MAX[:4])


def clamp_q(q: np.ndarray) -> np.ndarray:
    return np.clip(q, Q_MIN, Q_MAX)


def policy_features(q: np.ndarray, target: np.ndarray) -> np.ndarray:
    """Observation vector fed to the learned reach policy (11-dim).

    q0..q3 (4) | target xyz (3) | 5 * (target - ee) (3) | constant 1 (1)
    Kept identical to `app/src/vla/policy.js::features`.
    """
    ee = fk(q)
    return np.concatenate([q[:4], target, (target - ee) * 5.0, [1.0]]).astype(np.float32)
