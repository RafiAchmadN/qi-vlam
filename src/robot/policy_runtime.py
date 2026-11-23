"""NumPy runtime for exported `qi-vla-policy/1` JSON policies (no torch needed).

Same math as `app/src/vla/policy.js`; used by the bridge's /act endpoint so a
PC (or a robot's onboard computer) can serve the policy to thin clients.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List, Sequence

import numpy as np

from src.robot import arm


def _gelu_tanh(x: np.ndarray) -> np.ndarray:
    return 0.5 * x * (1.0 + np.tanh(0.7978845608028654 * (x + 0.044715 * x ** 3)))


class PolicyRuntime:
    def __init__(self, spec: Dict) -> None:
        if spec.get("format") != "qi-vla-policy/1":
            raise ValueError("unsupported policy format")
        self.spec = spec
        self.layers = []
        for L in spec["layers"]:
            if L["type"] == "dense":
                mats = [np.asarray(L["W"])]
            elif L["type"] == "ttns":
                mats = [np.asarray(L["right"]), np.asarray(L["root"]), np.asarray(L["left"])]
            else:
                raise ValueError(f"unknown layer type {L['type']}")
            self.layers.append((mats, np.asarray(L["b"]), L.get("act")))
        self.qd_max = np.asarray(spec["qd_max"])

    @classmethod
    def load(cls, path: str | Path) -> "PolicyRuntime":
        return cls(json.loads(Path(path).read_text()))

    def forward(self, x: Sequence[float]) -> np.ndarray:
        h = np.asarray(x, dtype=np.float64)
        for mats, b, act in self.layers:
            for M in mats:          # TTNS: right, then root, then left (never forms W)
                h = M @ h
            h = h + b
            if act == "gelu_tanh":
                h = _gelu_tanh(h)
            elif act == "tanh":
                h = np.tanh(h)
        return h

    def act(self, q: Sequence[float], target: Sequence[float]) -> List[float]:
        x = arm.policy_features(np.asarray(q, dtype=np.float64), np.asarray(target, dtype=np.float64))
        return (self.forward(x) * self.qd_max).tolist()
