"""Robot stack: kinematics, exported-policy runtime, bridge safety, LIBERO loader."""

import json
import os
from pathlib import Path

import numpy as np
import pytest

from src.robot import arm
from src.robot.policy_runtime import PolicyRuntime
from src.bridge.drivers import JointCal, MockDriver, SerialConfig, SerialDriver, make_driver
from src.bridge.server import BridgeCore, build_app

MODELS = Path("app/public/models")


def test_jacobian_matches_finite_difference():
    q = np.array([0.3, 0.2, 1.0, 0.5, 0.0, 1.0])
    J = arm.jacobian(q)
    for j in range(4):
        d = np.zeros(6)
        d[j] = 1e-6
        assert np.allclose(J[:, j], (arm.fk(q + d) - arm.fk(q - d)) / 2e-6, atol=1e-6)


def test_ik_expert_reaches_workspace_targets():
    rng = np.random.default_rng(0)
    for _ in range(20):
        q, t = arm.Q_HOME.copy(), rng.uniform([0.1, -0.2, 0.02], [0.28, 0.2, 0.2])
        for _ in range(150):
            q[:4] += arm.ik_step(q, t, pitch=2.8) * 0.05
            q = arm.clamp_q(q)
        assert np.linalg.norm(arm.fk(q) - t) < 5e-3


def test_joint_positions_end_at_fk():
    q = np.array([0.4, 0.3, 1.2, 0.8, 0, 1])
    assert np.allclose(arm.joint_positions(q)[-1], arm.fk(q))


def test_fk_vectors_export_consistent():
    v = json.loads((MODELS / "fk_vectors.json").read_text())
    assert np.allclose(arm.fk(np.asarray(v["q"])), np.asarray(v["ee"]), atol=1e-9)


@pytest.mark.parametrize("name", ["dense", "ttns"])
def test_policy_runtime_matches_torch_export(name):
    p = PolicyRuntime.load(MODELS / f"policy_{name}.json")
    tv = p.spec["test_vectors"]
    for x, y in zip(tv["x"], tv["y"]):
        assert np.allclose(p.forward(x), y, atol=1e-4)
    q, t = arm.Q_HOME.copy(), np.array([0.2, -0.05, 0.05])
    for _ in range(100):
        q[:4] += np.asarray(p.act(q, t)) * 0.05
        q = arm.clamp_q(q)
    assert np.linalg.norm(arm.fk(q) - t) < 0.01


def test_ttns_export_is_smaller_and_certified():
    d = json.loads((MODELS / "policy_dense.json").read_text())
    t = json.loads((MODELS / "policy_ttns.json").read_text())
    assert t["params"] * 2 < d["params"]
    assert t["lyapunov"]["certified"] and t["lyapunov"]["spectral_radius"] < 1


def test_policy_runtime_rejects_unknown_format():
    with pytest.raises(ValueError):
        PolicyRuntime({"format": "other"})


def test_bridge_core_clamps_steps_and_limits():
    core = BridgeCore(MockDriver(noise=0.0))
    q0 = core.driver.read()
    out = core.command([5.0, 5.0, 5.0, 5.0, 5.0, 5.0], now=0.0)
    step = np.abs(np.asarray(out) - np.asarray(q0))
    assert np.all(step <= arm.QD_MAX * core.dt * 1.5 + 1e-9)
    assert core.stats["clamped"] == 1
    assert core.command([float("nan")] * 6, now=0.02) is None
    assert core.command([0.0] * 3, now=0.04) is None


def test_bridge_core_watchdog_and_estop():
    core = BridgeCore(MockDriver(noise=0.0), watchdog_s=0.5)
    core.command(arm.Q_HOME.tolist(), now=0.0)
    core.tick(now=0.2)
    assert core.stats["watchdog_trips"] == 0
    core.tick(now=0.8)
    assert core.stats["watchdog_trips"] == 1 and core.holding
    core.trigger_estop()
    assert core.command(arm.Q_HOME.tolist(), now=1.0) is None
    core.release_estop()
    assert core.command(arm.Q_HOME.tolist(), now=1.1) is not None


def test_mock_driver_tracks_command():
    d = MockDriver(tau=0.01, noise=0.0)
    target = arm.Q_HOME.copy()
    target[0] = 0.05
    d.write(target.tolist())
    import time
    time.sleep(0.15)
    assert abs(d.read()[0] - 0.05) < 1e-3
    d.stop()


def test_serial_calibration_roundtrip():
    cfg = SerialConfig(cal=[JointCal(90, 1, 1), JointCal(80, -1, 1), JointCal(95, 1, 1.1),
                            JointCal(90, 1, 1), JointCal(90, 1, 1)])
    drv = SerialDriver(cfg)
    q = [0.2, -0.3, 1.0, 0.4, 0.1, 0.5]
    assert np.allclose(drv.from_deg(drv.to_deg(q)), q, atol=1e-6)
    assert drv.read() == drv.last_q  # no port open -> last known
    with pytest.raises(ValueError):
        make_driver("nope")


def test_bridge_websocket_and_act_endpoint():
    from fastapi.testclient import TestClient
    app = build_app(MockDriver(noise=0.0), str(MODELS / "policy_ttns.json"), static_dir=None)
    with TestClient(app) as c:
        assert c.get("/health").json()["status"] == "ok"
        assert c.get("/info").json()["policy"] == "ttns"
        r = c.post("/act", json={"q": arm.Q_HOME.tolist(), "target": [0.2, 0.0, 0.05]}).json()
        assert len(r["qd"]) == 4
        with c.websocket_connect("/ws") as ws:
            info = json.loads(ws.receive_text())
            assert info["type"] == "info" and info["role"] == "controller"
            with c.websocket_connect("/ws") as obs:
                assert json.loads(obs.receive_text())["role"] == "observer"
                obs.send_text(json.dumps({"type": "joint_command", "q": arm.Q_HOME.tolist()}))
                msgs = [json.loads(obs.receive_text()) for _ in range(5)]
                assert any(m.get("message") == "observer: read-only" for m in msgs)
            ws.send_text(json.dumps({"type": "joint_command", "q": arm.Q_HOME.tolist()}))
            ws.send_text(json.dumps({"type": "estop"}))
            ws.send_text(json.dumps({"type": "joint_command", "q": arm.Q_HOME.tolist()}))
            seen = [json.loads(ws.receive_text()) for _ in range(8)]
            assert any(m["type"] == "state" for m in seen)
            assert any(m["type"] == "error" for m in seen)


def _write_fake_libero(root: Path, n_eps: int = 8, T: int = 12):
    import pyarrow as pa
    import pyarrow.parquet as pq
    (root / "data" / "chunk-000").mkdir(parents=True)
    (root / "meta").mkdir(parents=True)
    rng = np.random.default_rng(0)
    rows = {"observation.state": [], "action": [], "episode_index": [], "task_index": [], "frame_index": []}
    for e in range(n_eps):
        for t in range(T):
            rows["observation.state"].append(rng.normal(size=8).astype(np.float32).tolist())
            rows["action"].append(np.r_[rng.normal(size=6), [1.0 if t > T // 2 else -1.0]].astype(np.float32).tolist())
            rows["episode_index"].append(e)
            rows["task_index"].append(e % 2 + (30 if e >= 4 else 0))
            rows["frame_index"].append(t)
    pq.write_table(pa.table(rows), root / "data" / "chunk-000" / "file-000.parquet")
    pq.write_table(pa.table({"task_index": [0, 1, 30, 31],
                             "__index_level_0__": ["put the bowl on the plate", "turn on the stove",
                                                   "pick up the black bowl", "open the drawer"]}),
                   root / "meta" / "tasks.parquet")


def test_libero_loader_split_and_features(tmp_path):
    from src.vla_data.libero import build_features, episode_split, load_libero, suite_of
    from src.vla_data.text import encode
    _write_fake_libero(tmp_path)
    d = load_libero(str(tmp_path))
    assert d.state.shape == (96, 8) and d.n_episodes == 8
    sp = episode_split(d, seed=0)
    for a in ("train", "val", "test"):
        for b in ("train", "val", "test"):
            if a < b:
                assert not set(d.episode[sp[a]]) & set(d.episode[sp[b]])  # no episode leakage
    X = build_features(d)
    assert X.shape == (96, 80)
    assert np.allclose(X[d.frame == 0, 8:16], 0)  # causal velocity is zero at t=0
    assert suite_of(3) == "libero_10" and suite_of(35) == "libero_spatial" and suite_of(99) == "unknown"
    assert abs(np.linalg.norm(encode("pick up the milk")) - 1) < 1e-6
    assert np.allclose(encode(""), 0)
    with pytest.raises(FileNotFoundError):
        load_libero(str(tmp_path / "missing"))


@pytest.mark.skipif(not Path("data/libero/meta/tasks.parquet").exists(),
                    reason="LIBERO not downloaded (python scripts/download_libero.py)")
def test_libero_benchmark_smoke(tmp_path, monkeypatch):
    import src.evaluation.libero_benchmark as lb
    monkeypatch.setattr(lb, "_PENNYLANE", False)
    out = lb.run(n_seeds=1, steps=30, heal_steps=10, configs=(("ttns", 64),), head_steps=5,
                 head_train_frames=512, out_path=str(tmp_path / "r.json"))
    assert {"dense", "healing_ctrl", "int8", "ttns_chi64", "head_tanh"} <= set(out["summary"])


def test_train_policy_pipeline_smoke(tmp_path):
    from src.robot.train_policy import main
    out = main(out_dir=str(tmp_path / "models"), n_episodes=30, steps=60, heal_steps=20,
               n_eval=5, n_lyap=10, results_path=str(tmp_path / "app_policy.json"))
    assert out["params_ttns"] < out["params_dense"]
    for name in ("dense", "ttns"):
        p = PolicyRuntime.load(tmp_path / "models" / f"policy_{name}.json")
        x, y = p.spec["test_vectors"]["x"][0], p.spec["test_vectors"]["y"][0]
        assert np.allclose(p.forward(x), y, atol=1e-4)
