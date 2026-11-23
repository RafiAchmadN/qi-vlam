"""Observation features of the learned pick-and-place policy.

Kept identical to `app/src/vla/pickplace.js::ppFeatures` (parity-tested), so a
model trained here on simulated data runs unchanged on the phone.

    q0..q3 (4) | gripper (1) | tip xyz (3) | 5*(obj - tip) (3) |
    5*(place - tip) (3) | 10*(obj_z - CUBE/2) (1) | goal_is_cube (1) | 1 (1)   = 17

`place` is where the cube centre must end up: the support point (pad centre on
the table, or top of the goal cube) + half a cube. Everything is in the robot
base frame; positions come from the perception module (camera detections in
the app, noisy ground truth in simulation).
"""

from __future__ import annotations

import numpy as np

from src.robot import arm

CUBE = 0.03
PP_DIM = 17
PP_OUT = 5  # qd0..qd3 / QD_MAX (4), gripper command in [-1 close, +1 open] (1)


def pickplace_features(q, obj, goal, goal_is_cube) -> np.ndarray:
    q = np.asarray(q, dtype=np.float64)
    obj = np.asarray(obj, dtype=np.float64)[:3]
    place = np.asarray(goal, dtype=np.float64)[:3] + np.array([0.0, 0.0, CUBE / 2])
    tip = arm.fk(q)
    return np.concatenate([q[:4], [q[5]], tip, 5.0 * (obj - tip), 5.0 * (place - tip),
                           [10.0 * (obj[2] - CUBE / 2), 1.0 if goal_is_cube else 0.0, 1.0]]
                          ).astype(np.float32)


def batch_features(state: np.ndarray, env_state: np.ndarray) -> np.ndarray:
    """Vectorised `pickplace_features` over dataset rows.

    state: (N, 6) measured joints; env_state: (N, >=8) = obj xyz yaw (4),
    goal xyz (3), goal_is_cube (1), ...
    """
    q = state.astype(np.float64)
    tip = arm.fk(q)
    obj = env_state[:, 0:3].astype(np.float64)
    place = env_state[:, 4:7].astype(np.float64) + np.array([0.0, 0.0, CUBE / 2])
    return np.concatenate([q[:, :4], q[:, 5:6], tip, 5.0 * (obj - tip), 5.0 * (place - tip),
                           10.0 * (obj[:, 2:3] - CUBE / 2), env_state[:, 7:8].astype(np.float64),
                           np.ones((len(q), 1))], axis=1).astype(np.float32)
