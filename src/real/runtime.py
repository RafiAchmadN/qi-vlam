"""NumPy runtime for exported real-data vision policies (format qi-vla-vision/1).

The SAME committed file (app/public/models/real_cnn_ttns.json) runs in the app
(JS, app/src/real/visionPolicy.js) and on a PC driving the physical arm
(src/real/deploy.py --model *.json) — no PyTorch needed at deployment.
"""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Dict, List

import numpy as np

from src.robot.policy_runtime import PolicyRuntime

_erf = np.vectorize(math.erf, otypes=[np.float64])


def gelu_exact(x: np.ndarray) -> np.ndarray:
    try:
        from scipy.special import erf
        return (0.5 * x * (1.0 + erf(x / np.float32(math.sqrt(2.0))))).astype(np.float32)
    except ImportError:  # pragma: no cover
        return 0.5 * x * (1.0 + _erf(x / math.sqrt(2.0)))


def conv2d(x: np.ndarray, W: np.ndarray, b: np.ndarray, s: int, p: int) -> np.ndarray:
    """x (C,H,W), W (O,C,K,K) -> (O,Ho,Wo) via im2col + matmul."""
    C, H, Wd = x.shape
    O, _, K, _ = W.shape
    xp = np.pad(x, ((0, 0), (p, p), (p, p)))
    Ho, Wo = (H + 2 * p - K) // s + 1, (Wd + 2 * p - K) // s + 1
    sc, sh, sw = xp.strides
    cols = np.lib.stride_tricks.as_strided(xp, (C, K, K, Ho, Wo), (sc, sh, sw, sh * s, sw * s))
    return (W.reshape(O, -1) @ cols.reshape(C * K * K, Ho * Wo)).reshape(O, Ho, Wo) + b[:, None, None]


class VisionRuntime:
    def __init__(self, spec: Dict) -> None:
        if spec.get("format") != "qi-vla-vision/1":
            raise ValueError("expected a qi-vla-vision/1 model")
        self.spec, self.k, self.size, self.names = spec, spec["k"], spec["size"], spec["names"]
        self.convs = [(np.asarray(c["W"], np.float32), np.asarray(c["b"], np.float32), c["stride"], c["pad"])
                      for c in spec["conv"]]
        self.head = PolicyRuntime({"format": "qi-vla-policy/1", "layers": spec["head"], "qd_max": [1.0]})
        self.norm = spec["norm"]

    @classmethod
    def load(cls, path) -> "VisionRuntime":
        return cls(json.loads(Path(path).read_text(encoding="utf-8")))

    def forward(self, rgb: np.ndarray, state_norm, task: int) -> np.ndarray:
        """rgb (S,S,3) uint8 -> (K, 6) normalized chunk."""
        x = np.asarray(rgb, np.float32).transpose(2, 0, 1) / np.float32(255.0)
        for W, b, s, p in self.convs:
            x = gelu_exact(conv2d(x, W, b, s, p))
        c, h, w = x.shape
        flat = x.reshape(c, h * w)
        e = np.exp(flat - flat.max(1, keepdims=True))
        pr = (e / e.sum(1, keepdims=True)).reshape(c, h, w)
        ys, xs = np.linspace(-1, 1, h), np.linspace(-1, 1, w)
        feats = np.concatenate([(pr.sum(2) * ys).sum(1), (pr.sum(1) * xs).sum(1), x.mean((1, 2), dtype=np.float64),
                                np.asarray(state_norm, np.float64), np.eye(len(self.names))[task]])
        return self.head.forward(feats).reshape(self.k, 6)


class TorchLike:
    """Adapter so deploy.Controller can drive a VisionRuntime like the torch model."""

    def __init__(self, rt: VisionRuntime) -> None:
        self.rt, self.k = rt, rt.k

    def eval(self):
        return self

    def __call__(self, img, state, task):
        import torch
        rgb = (img[0].permute(1, 2, 0).numpy() * 255.0).round().astype(np.uint8)
        y = self.rt.forward(rgb, state[0].numpy(), int(task[0].argmax()))
        return torch.from_numpy(y.astype(np.float32))[None]


def as_checkpoint(path: str) -> Dict:
    """Load an exported JSON as a deploy.Controller checkpoint."""
    rt = VisionRuntime.load(path)
    return {"kind": rt.spec["kind"] + "(json)", "k": rt.k, "names": rt.names, "norm": rt.norm, "size": rt.size,
            "model": TorchLike(rt)}


def chunk_list(y: np.ndarray) -> List[List[float]]:
    return y.round(6).tolist()
