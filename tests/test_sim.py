"""Simulation layer: environment presets, MuJoCo physics, expert, dataset,
Gymnasium env, URDF, bridge MuJoCo driver, pick-place training pipeline."""

import json
from pathlib import Path

import numpy as np
import pytest

mujoco = pytest.importorskip("mujoco")

from src.robot import arm  # noqa: E402
from src.sim.envs import env_index, hard_dynamics, load_env, sample_episode  # noqa: E402
from src.sim.expert import PickPlaceExpert, roll_for_yaw, run_expert, wrap_quarter  # noqa: E402
from src.sim.features import batch_features, pickplace_features  # noqa: E402
from src.sim.physics import ArmSim  # noqa: E402

ENVS = env_index()["envs"]


@pytest.mark.parametrize("env_id", ENVS)
def test_mjcf_tip_matches_canonical_fk(env_id):
    sim = ArmSim(sample_episode(env_id, 0))
    m, d = sim.model, sim.data
    rng = np.random.default_rng(1)
    for _ in range(20):
        q = rng.uniform(arm.Q_MIN, arm.Q_MAX)
        for a, v in zip(sim.qadr, q[:5]):
            d.qpos[a] = v
        mujoco.mj_forward(m, d)
        assert np.allclose(d.site_xpos[sim.tip], arm.fk(q), atol=1e-9)


def test_urdf_tip_matches_canonical_fk_and_committed_file_is_current():
    from src.sim.urdf import build_urdf, urdf_fk
    xml = build_urdf()
    rng = np.random.default_rng(2)
    for _ in range(30):
        q = rng.uniform(arm.Q_MIN, arm.Q_MAX)
        assert np.allclose(urdf_fk(xml, dict(zip(arm.JOINT_NAMES, q))), arm.fk(q), atol=1e-9)
    committed = Path("ros2/qi_vla_ros/urdf/qi_vla_arm.urdf").read_text(encoding="utf-8")
    assert committed.replace("\r\n", "\n") == xml, "regenerate: python -m src.sim.urdf"


@pytest.mark.parametrize("env_id", ENVS)
def test_episode_sampling_valid_and_deterministic(env_id):
    spec = load_env(env_id)
    for s in range(30):
        a, b = sample_episode(env_id, s), sample_episode(env_id, s)
        assert a.to_json() == b.to_json()
        pts = [v[:2] for v in a.cubes.values()] + list(a.pads.values())
        for i in range(len(pts)):
            for j in range(i):
                assert np.hypot(pts[i][0] - pts[j][0], pts[i][1] - pts[j][1]) >= spec["spawn"]["min_sep"] - 1e-9
        for p in pts:
            for o in spec["obstacles"]:
                if not o.get("visual_only"):
                    assert not (abs(p[0] - o["pos"][0]) < o["size"][0] / 2 and abs(p[1] - o["pos"][1]) < o["size"][1] / 2)
        assert a.obj != a.goal and a.instruction
    r = sample_episode(env_id, 3, randomize=True).dynamics
    assert r != load_env(env_id)["dynamics"]


def test_wrist_roll_wrap():
    assert abs(wrap_quarter(np.pi / 2 + 0.1) - 0.1) < 1e-12
    assert -np.pi / 4 <= roll_for_yaw(1.3, -0.7) < np.pi / 4


@pytest.mark.parametrize("env_id,seed", [("lab_table", 0), ("kitchen_counter", 1), ("warehouse_shelf", 2)])
def test_physics_expert_picks_and_places_with_friction(env_id, seed):
    sim = ArmSim(sample_episode(env_id, seed, randomize=True))
    r = run_expert(sim, max_steps=500)
    assert r["success"], r["phase"]
    assert r["collision_steps"] == 0


def test_servo_latency_and_lag_slow_the_response():
    def rise(dyn):
        sim = ArmSim(sample_episode("lab_table", 0), dynamics=dyn)
        tgt = arm.Q_HOME.copy()
        tgt[0] = 0.4
        for k in range(40):
            sim.step(tgt)
            if sim.q_true()[0] > 0.2:
                return k
        return 99
    fast = rise(dict(servo_tau=0.02, latency_steps=0))
    slow = rise(hard_dynamics(load_env("lab_table")))
    assert slow > fast + 2


def test_features_vectorised_equals_scalar():
    rng = np.random.default_rng(0)
    q = rng.uniform(arm.Q_MIN, arm.Q_MAX, size=(5, 6)).astype(np.float32)
    env = rng.uniform(-0.2, 0.3, size=(5, 24)).astype(np.float32)
    env[:, 7] = [0, 1, 0, 1, 0]
    B = batch_features(q, env)
    for i in range(5):
        assert np.allclose(B[i], pickplace_features(q[i], env[i, :4], env[i, 4:7], env[i, 7] > 0.5), atol=1e-5)


def test_collect_writes_lerobot_style_dataset(tmp_path):
    from src.sim.collect import collect
    from src.sim.train_pickplace import load_dataset
    from src.vla_data.libero import load_libero
    out = collect(str(tmp_path / "ds"), envs=("lab_table",), episodes=2, workers=1, dart=0.1)
    assert out["episodes_kept"] >= 1 and out["frames"] > 50
    info = json.loads((tmp_path / "ds" / "meta" / "info.json").read_text())
    assert info["codebase_version"] == "v3.0" and info["features"]["observation.state"]["shape"] == [6]
    d = load_libero(str(tmp_path / "ds"))
    assert d.state.shape[1] == 6 and d.action.shape[1] == 6 and d.n_episodes == out["episodes_kept"]
    X, Y, ep = load_dataset(str(tmp_path / "ds"))
    assert X.shape == (out["frames"], 17) and np.all(np.abs(Y) <= 1.0 + 1e-6)


def test_import_app_demos_roundtrip(tmp_path):
    import subprocess
    import sys
    frame = {"q": arm.Q_HOME.tolist(), "q_cmd": arm.Q_HOME.tolist(), "qd": [0.1, 0, 0, 0], "gripper": 1,
             "env_state": [0.2, 0.0, 0.015, 0.0, 0.25, 0.1, 0.0, 0.0] + [0.0] * 24}
    payload = {"format": "qi-vla-demos/1", "mode": "sim",
               "episodes": [{"env": "kitchen_counter", "instruction": "ambil kubus merah taruh di pad kuning",
                             "success": True, "frames": [frame] * 12}]}
    f = tmp_path / "d.json"
    f.write_text(json.dumps(payload))
    r = subprocess.run([sys.executable, "scripts/import_app_demos.py", str(f), "--out", str(tmp_path / "ds")],
                       capture_output=True, text=True)
    assert r.returncode == 0, r.stderr
    from src.sim.train_pickplace import load_dataset
    X, Y, _ = load_dataset(str(tmp_path / "ds"))
    assert X.shape == (12, 17) and np.allclose(Y[0, 0], 0.1 / arm.QD_MAX[0])


def test_gymnasium_env_api():
    import gymnasium as gym
    import src.sim.gym_env  # noqa: F401  (registers the id)
    env = gym.make("QiVla/PickPlace-v0", env_id="kitchen_counter")
    obs, info = env.reset(seed=0)
    assert set(obs) == {"state", "environment_state", "features"} and info["instruction"]
    assert env.observation_space.contains(obs)
    for _ in range(5):
        obs, r, term, trunc, info = env.step(env.action_space.sample())
        assert np.isfinite(r)
    from gymnasium.utils.env_checker import check_env
    check_env(env.unwrapped, skip_render_check=True)


def test_bridge_mujoco_driver_scene_and_demo_upload(tmp_path):
    from fastapi.testclient import TestClient
    from src.bridge.drivers import MujocoDriver, make_driver
    from src.bridge.server import build_app
    drv = MujocoDriver("warehouse_shelf", seed=0, realtime=False)
    assert isinstance(make_driver("mujoco", env_id="lab_table"), MujocoDriver)
    drv.write([0.3] + arm.Q_HOME[1:].tolist())
    for _ in range(30):
        drv.advance()
    assert abs(drv.read()[0] - 0.3) < 0.02
    app = build_app(drv, None, static_dir=None, demo_dir=str(tmp_path / "demos"))
    with TestClient(app) as c:
        assert c.get("/info").json()["sim"] is True
        with c.websocket_connect("/ws") as ws:
            assert json.loads(ws.receive_text())["env"] == "warehouse_shelf"
            ws.send_text(json.dumps({"type": "scene_reset", "env": "kitchen_counter", "seed": 5}))
            msgs = [json.loads(ws.receive_text()) for _ in range(6)]
            scene = [m for m in msgs if m["type"] == "scene"]
            from src.sim.envs import GRASPABLE
            assert scene and scene[0]["env"] == "kitchen_counter"
            assert set(scene[0]["objects"]) <= set(GRASPABLE) and 3 <= len(scene[0]["objects"]) <= 5
        bad = c.post("/episodes", json={"format": "nope"})
        assert bad.status_code == 400
        ok = c.post("/episodes", json={"format": "qi-vla-demos/1", "episodes": [{"frames": [{}] * 3}]}).json()
        assert ok["episodes"] == 1 and Path(ok["saved"]).exists()


def test_train_pickplace_pipeline_smoke(tmp_path):
    from src.sim.collect import collect
    from src.sim.train_pickplace import run
    collect(str(tmp_path / "dr"), envs=("lab_table",), episodes=2, workers=1)
    out = run(str(tmp_path / "dr"), str(tmp_path / "none"), seeds=1, steps=20, heal_steps=5, chi=8, hidden=64,
              n_train=1, n_heldout=1, n_hard=1, out_models=str(tmp_path / "models"),
              out_json=str(tmp_path / "r.json"), workers=1)
    assert {"dense_dr", "healing_ctrl", "ttns_dr", "ttns_noheal"} <= set(out["summary"])
    assert out["summary"]["ttns_dr"]["params"] < out["summary"]["dense_dr"]["params"]
    spec = json.loads((tmp_path / "models" / "pickplace_ttns.json").read_text())
    from src.robot.policy_runtime import PolicyRuntime
    rt = PolicyRuntime(spec)
    for x, y in zip(spec["test_vectors"]["x"], spec["test_vectors"]["y"]):
        assert np.allclose(rt.forward(x), y, atol=1e-4)


@pytest.mark.skipif(not Path("app/public/models/pickplace_ttns.json").exists(), reason="pick-place policy not trained")
def test_exported_pickplace_policy_parity_and_certificate_free_scope():
    from src.robot.policy_runtime import PolicyRuntime
    spec = json.loads(Path("app/public/models/pickplace_ttns.json").read_text())
    rt = PolicyRuntime(spec)
    for x, y in zip(spec["test_vectors"]["x"], spec["test_vectors"]["y"]):
        assert np.allclose(rt.forward(x), y, atol=1e-4)
    for c in spec["feature_vectors"]["cases"]:
        assert np.allclose(pickplace_features(c["q"], c["obj"], c["goal"], c["goal_is_cube"]), c["x"], atol=1e-5)
    assert spec["kind"] == "pickplace" and "evaluation" in spec


def test_model_zoo_architectures_train_and_act():
    import torch
    from src.sim.model_zoo import ZooPolicy, build, chunk_targets, model_stats, train_model
    from src.sim.features import PP_DIM
    g = torch.Generator().manual_seed(0)
    X = torch.randn(256, PP_DIM, generator=g)
    Y = torch.tanh(torch.randn(256, 5, generator=g))
    ep = np.repeat(np.arange(8), 32)
    assert chunk_targets(Y.numpy(), ep).shape == (256, 5, 5)
    base = train_model("mlp", X, Y, ep, 0, steps=3)
    for kind in ("mlp_small", "resmlp", "act_chunk", "diffusion", "ttns", "int8", "head_tanh", "head_surrogate"):
        m = train_model(kind, X, Y, ep, 0, steps=3, base=base, heal_steps=2, head_steps=2, head_frames=64)
        sd = m.state_dict()
        m2 = build(kind, {})
        m2.load_state_dict(sd)                     # workers rebuild the skeleton and load weights
        pol = ZooPolicy(kind, m2)
        for _ in range(3):
            y = pol(X[0].numpy())
        assert y.shape == (5,) and np.all(np.isfinite(y))
        st = model_stats(kind, m2, X[0].numpy())
        assert st["params"] > 0 and st["latency_ms"] > 0
