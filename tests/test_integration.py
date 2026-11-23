import copy

import torch

from src.compression.tensor_network import compress_model
from src.compression.quantization import quantize_model_int8
from src.evaluation.benchmark import make_proxy_vlam, make_task, task_accuracy
from src.evaluation.safety_suite import run_safety_suite
from src.evaluation.safety_baseline_comparison import run_safety_baseline_comparison
from src.inference.model_server import CompressedVLAMServer
from src.training.vlam_trainer import VLAMTrainer


def test_full_pipeline(tmp_path):
    """Train -> compress -> compare vs INT8 -> serve. End-to-end on CPU."""
    batch_fn = make_task(dim=128, seed=0)
    model = make_proxy_vlam(dim=128, depth=2, seed=0)
    VLAMTrainer(model, max_steps=150, seed=0).train(batch_fn)
    base_acc = task_accuracy(model, batch_fn)

    mps_model, report = compress_model(copy.deepcopy(model), bond_dimension=24)
    VLAMTrainer(mps_model, max_steps=200, lr=1e-3, seed=0).train(batch_fn)  # healing
    mps_acc = task_accuracy(mps_model, batch_fn)
    assert report.overall_ratio >= 2.0, "compression target >=2x"
    assert base_acc - mps_acc <= 0.05 * max(base_acc, 1e-9) + 0.05

    int8_model = quantize_model_int8(copy.deepcopy(model))
    assert task_accuracy(int8_model, batch_fn) > 0

    server = CompressedVLAMServer(mps_model)
    out = server.infer(torch.randn(1, 128))
    assert out["within_budget"] and out["safety_ok"]


def test_safety_suite(tmp_path):
    summary = run_safety_suite(n_episodes=20, seed=0,
                               out_path=str(tmp_path / "safety.json"))
    assert summary["safe_pct"] >= 50.0


def test_safety_baseline_comparison(tmp_path):
    """Heuristic-clipping baseline vs our SDP certificate: at a clearly stable gain the SDP
    must certify; at a clearly unstable gain it must not — the certificate
    tracks the true spectral-radius condition by construction."""
    summary = run_safety_baseline_comparison(
        gains=(0.5, 1.3), n_episodes=5, T=10, seed=0,
        out_path=str(tmp_path / "sbc.json"))
    rows = {r["gain"]: r for r in summary["sweep"]}
    assert rows[0.5]["truly_stable"] and rows[0.5]["sdp_certified"]
    assert not rows[1.3]["truly_stable"] and not rows[1.3]["sdp_certified"]


def test_ablation_smoke(tmp_path):
    from src.evaluation.ablation import run_ablation
    res = run_ablation(bond_dims=[8], seed=0, out_path=str(tmp_path / "p.json"))
    assert "healing_only_control_accuracy" in res
    assert len(res["pareto"]) == 2  # mps + ttns


def test_rl_experiment_smoke(tmp_path, monkeypatch):
    import src.evaluation.rl_experiment as rx
    monkeypatch.setattr(rx, "_HAS_PQC", False)  # keep fast, skip circuit sim
    res = rx.run_rl_experiment(n_seeds=1, max_rollouts=200,
                               out_path=str(tmp_path / "rl.json"))
    assert "mlp" in res and "surrogate" in res


def test_pqc_noise_sensitivity_smoke(tmp_path):
    import pytest
    try:
        from src.evaluation.pqc_noise_sensitivity import run_noise_sensitivity
    except Exception:
        pytest.skip("pennylane unavailable")
    res = run_noise_sensitivity(train_rollouts=100, n_eval_states=4,
                                noise_levels=(0.0, 0.05),
                                out_path=str(tmp_path / "noise.json"))
    assert len(res["sweep"]) == 2
    assert res["sweep"][0]["reward_retained_pct"] == 100.0


def test_benchmark_smoke(tmp_path):
    from src.evaluation.benchmark import run_benchmark
    res = run_benchmark(n_seeds=1, bond_dimension=16, dim=64, train_steps=30,
                        out_path=str(tmp_path / "r.json"))
    assert "total_training_flops_reduction_pct" in res["training_efficiency"]


def test_app_factory():
    from src.inference.app import create_app
    assert create_app().title
