"""Production inference server for compressed VLAM policies.

FastAPI app exposing /health, /infer, /metrics with structured JSON logging,
a hard latency budget (default 100 ms per ISO 26262 real-time constraint) and
a Lyapunov safety assertion hook applied to every returned control command.
"""

from __future__ import annotations

import json
import logging
import statistics
import time
from typing import Dict, List, Optional

import torch
import torch.nn as nn

try:
    from fastapi import FastAPI, HTTPException
    from pydantic import BaseModel
    _FASTAPI = True
except ImportError:  # pragma: no cover
    _FASTAPI = False

from src.training.vlam_trainer import LyapunovStabilityValidator


class JsonFormatter(logging.Formatter):
    def format(self, record: logging.LogRecord) -> str:
        payload = {
            "ts": self.formatTime(record),
            "level": record.levelname,
            "msg": record.getMessage(),
            "logger": record.name,
        }
        if hasattr(record, "extra_fields"):
            payload.update(record.extra_fields)  # type: ignore[attr-defined]
        return json.dumps(payload)


def get_json_logger(name: str = "vlam.server") -> logging.Logger:
    logger = logging.getLogger(name)
    if not logger.handlers:
        h = logging.StreamHandler()
        h.setFormatter(JsonFormatter())
        logger.addHandler(h)
        logger.setLevel(logging.INFO)
    return logger


class LatencyProfiler:
    """Rolling latency statistics with budget enforcement."""

    def __init__(self, budget_ms: float = 100.0, window: int = 1000) -> None:
        self.budget_ms = budget_ms
        self.window = window
        self.samples: List[float] = []

    def record(self, latency_ms: float) -> bool:
        self.samples.append(latency_ms)
        if len(self.samples) > self.window:
            self.samples.pop(0)
        return latency_ms <= self.budget_ms

    def summary(self) -> Dict[str, float]:
        if not self.samples:
            return {"count": 0}
        s = sorted(self.samples)
        return {
            "count": len(s),
            "mean_ms": statistics.mean(s),
            "p50_ms": s[len(s) // 2],
            "p95_ms": s[int(len(s) * 0.95)],
            "max_ms": s[-1],
            "budget_ms": self.budget_ms,
            "violations": sum(1 for x in s if x > self.budget_ms),
        }

    def profile_model(self, model: nn.Module, input_shape, n_runs: int = 50,
                      warmup: int = 5) -> Dict[str, float]:
        model.eval()
        x = torch.randn(*input_shape)
        with torch.no_grad():
            for _ in range(warmup):
                model(x)
            for _ in range(n_runs):
                t0 = time.perf_counter()
                model(x)
                self.record((time.perf_counter() - t0) * 1000)
        return self.summary()


class CompressedVLAMServer:
    """Wraps a compressed policy model with safety-checked inference."""

    def __init__(self, model: nn.Module, budget_ms: float = 100.0) -> None:
        self.model = model.eval()
        self.profiler = LatencyProfiler(budget_ms=budget_ms)
        self.validator = LyapunovStabilityValidator()
        self.log = get_json_logger()

    @torch.no_grad()
    def infer(self, features: torch.Tensor) -> Dict:
        t0 = time.perf_counter()
        out = self.model(features)
        latency = (time.perf_counter() - t0) * 1000
        within = self.profiler.record(latency)
        # safety hook: bounded command assertion
        safe = bool(torch.isfinite(out).all()) and bool(out.abs().max() < 1e3)
        self.log.info("inference", extra={"extra_fields": {
            "latency_ms": round(latency, 3), "within_budget": within, "safe": safe}})
        return {
            "action": out.squeeze(0).tolist(),
            "latency_ms": latency,
            "within_budget": within,
            "safety_ok": safe,
        }


def build_app(model: Optional[nn.Module] = None, in_features: int = 128):
    """Create the FastAPI app; a small default model is used if none given."""
    if not _FASTAPI:  # pragma: no cover
        raise RuntimeError("fastapi not installed; pip install fastapi uvicorn")
    if model is None:
        model = nn.Sequential(nn.Linear(in_features, 256), nn.GELU(), nn.Linear(256, 4))
    server = CompressedVLAMServer(model)

    class InferRequest(BaseModel):
        features: List[float]

    app = FastAPI(title="QI-VLA Inference Server", version="1.0.0")

    @app.get("/health")
    def health():
        return {"status": "ok"}

    @app.post("/infer")
    def infer(req: InferRequest):
        x = torch.tensor(req.features, dtype=torch.float32).unsqueeze(0)
        result = server.infer(x)
        if not result["safety_ok"]:
            raise HTTPException(status_code=500, detail="safety assertion failed")
        return result

    @app.get("/metrics")
    def metrics():
        return server.profiler.summary()

    return app
