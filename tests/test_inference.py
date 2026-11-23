import torch
import torch.nn as nn

from src.inference.model_server import CompressedVLAMServer, LatencyProfiler


class TestLatencyProfiler:
    def test_record_and_summary(self):
        p = LatencyProfiler(budget_ms=100)
        assert p.record(50.0) is True
        assert p.record(150.0) is False
        s = p.summary()
        assert s["count"] == 2 and s["violations"] == 1

    def test_profile_model(self):
        p = LatencyProfiler(budget_ms=100)
        s = p.profile_model(nn.Linear(64, 8), (1, 64), n_runs=10)
        assert s["count"] == 10
        assert s["p95_ms"] < 100  # tiny model on CPU


class TestServer:
    def test_infer_contract(self):
        server = CompressedVLAMServer(nn.Linear(16, 4))
        out = server.infer(torch.randn(1, 16))
        assert set(out) >= {"action", "latency_ms", "within_budget", "safety_ok"}
        assert len(out["action"]) == 4
        assert out["safety_ok"]

    def test_safety_flag_on_nan(self):
        class Bad(nn.Module):
            def forward(self, x):
                return x[:, :4] * float("nan")

        server = CompressedVLAMServer(Bad())
        assert server.infer(torch.randn(1, 16))["safety_ok"] is False
