"""Mobile robot layer: maps, ray casting vs MuJoCo rangefinders, planning,
costmap with unknown obstacles, physics, expert, shield, Gym env, bridge
rover driver, navigation training pipeline."""

import json
from pathlib import Path

import numpy as np
import pytest

pytest.importorskip("mujoco")

from src.sim.mobile.geometry import (boxes, build_grid, clearance, episode_boxes, grid_for,  # noqa: E402
                                     load_map, map_index, plan_path, raycast, robot_spec,
                                     sample_nav_episode)
from src.sim.mobile.nav import Costmap, NavExpert, nav_features, run_nav, shield_vw  # noqa: E402
from src.sim.mobile.physics import KinematicRover, RoverSim  # noqa: E402

MAPS = map_index()["maps"]


@pytest.mark.parametrize("map_id", MAPS)
def test_all_places_mutually_reachable(map_id):
    m, g = load_map(map_id), grid_for(map_id)
    for a in m["places"]:
        assert not g.occ[g.cell(*a["pos"])], f"{a['id']} inside inflated obstacle"
        for b in m["places"]:
            if a is not b:
                assert plan_path(g, a["pos"], b["pos"]) is not None, (a["id"], b["id"])


def test_mujoco_lidar_matches_2d_raycast():
    ep = sample_nav_episode("office", 3, randomize=False, unknown=2)
    sim = RoverSim(ep)
    x, y, yaw = sim.pose()
    assert np.allclose(sim.scan(), raycast(episode_boxes(load_map("office"), ep), x, y, yaw), atol=2e-3)


def test_planned_path_is_collision_free():
    m = load_map("warehouse")
    B, g = boxes(m), grid_for("warehouse")
    p = plan_path(g, (0.6, 0.6), (4.0, 5.25))
    P = np.asarray(p)
    for a, b in zip(P[:-1], P[1:]):
        for u in np.linspace(0, 1, 20):
            q = a + u * (b - a)
            assert clearance(B, *q) > robot_spec()["radius"]


def test_unknown_obstacles_not_in_static_map_but_detected_by_costmap():
    ep = sample_nav_episode("home", 5, randomize=False, unknown=3)
    assert ep.unknown, "expected unknown obstacles"
    o = ep.unknown[0]
    g = grid_for("home")
    assert not g.occ[g.cell(*o["pos"])]            # planner's map does not know it
    sim = KinematicRover(ep)
    cm = Costmap(g)
    # look at the obstacle from 0.6 m away
    sim.x, sim.y = o["pos"][0] - 0.6, o["pos"][1]
    sim.yaw = 0.0
    if clearance(sim.B, sim.x, sim.y) > 0.15:
        cm.update(sim.pose(), sim.scan())
        assert cm.marks[g.cell(o["pos"][0] - o["size"][0] / 2, o["pos"][1])]


def test_shield_limits_forward_speed_near_wall():
    R = robot_spec()
    scan = np.full(36, 3.5)
    scan[0] = R["radius"] + 0.1        # wall 10 cm ahead of the bumper
    v, w, hit = shield_vw(0.26, 0.4, scan)
    assert hit and v < 0.05 and w == 0.4
    v2, _, hit2 = shield_vw(-0.2, 0.0, scan)  # reversing away is allowed
    assert not hit2 and v2 == -0.2


@pytest.mark.parametrize("map_id,seed,unknown", [("home", 0, 0), ("office", 2, 2)])
def test_physics_expert_navigates(map_id, seed, unknown):
    sim = RoverSim(sample_nav_episode(map_id, seed, randomize=True, unknown=unknown))
    r = run_nav(sim)
    assert r["success"] and not r["collision"], r


def test_physics_rover_motion_and_slip():
    ep = sample_nav_episode("home", 0, randomize=False)
    sim = RoverSim(ep)
    for _ in range(40):
        sim.step(0.2, 0.0)
    d = np.hypot(sim.pose()[0] - ep.start[0], sim.pose()[1] - ep.start[1])
    assert 0.25 < d < 0.42                         # motor lag: less than 0.4 m commanded
    slip = RoverSim(ep, dynamics={"floor_friction": 0.05})
    for _ in range(40):
        slip.step(0.26, 1.5)
    assert slip.collision_steps >= 0               # runs on a near-frictionless floor without blowing up
    assert np.all(np.isfinite(slip.pose()))


def test_nav_features_shape_and_ranges():
    x = nav_features((1.0, 1.0, 0.3), np.full(36, 2.0), (1.5, 1.2), (3.0, 3.0), (0.1, -0.5))
    assert x.shape == (25,) and np.all(np.abs(x) <= 1.0 + 1e-6)


def test_gym_nav_env_api():
    import gymnasium as gym
    import src.sim.mobile.gym_env  # noqa: F401
    env = gym.make("QiVla/Nav-v0", map_id="home", backend="kinematic", unknown=1)
    obs, info = env.reset(seed=1)
    assert info["instruction"] and obs["scan"].shape == (36,) and obs["features"].shape == (25,)
    for _ in range(5):
        obs, r, term, trunc, inf = env.step(env.action_space.sample())
        assert np.isfinite(r)
    from gymnasium.utils.env_checker import check_env
    check_env(env.unwrapped, skip_render_check=True)


def test_bridge_rover_driver_twist_watchdog_and_protocol():
    from fastapi.testclient import TestClient
    from src.bridge.drivers import RoverDriver, make_driver
    from src.bridge.server import build_app
    d = RoverDriver("warehouse", seed=1, realtime=False, watchdog_s=0.2)
    assert isinstance(make_driver("rover-kinematic", map_id="home"), RoverDriver)
    y0 = d.sim.pose()[2]
    d.twist(0.0, 1.0)
    for _ in range(10):
        d.advance()
    assert abs(d.sim.pose()[2] - y0) > 0.1
    assert d.twist(float("nan"), 0.0) is None
    import time
    time.sleep(0.3)
    d.advance()
    assert d.cmd == (0.0, 0.0) and d.watchdog_trips == 1
    app = build_app(d, None, static_dir=None)
    with TestClient(app) as c:
        assert c.get("/info").json()["kind"] == "rover"
        with c.websocket_connect("/ws") as ws:
            assert json.loads(ws.receive_text())["role"] == "controller"
            ws.send_text(json.dumps({"type": "twist_command", "v": 0.1, "w": 0.0}))
            ws.send_text(json.dumps({"type": "estop"}))
            ws.send_text(json.dumps({"type": "twist_command", "v": 0.1, "w": 0.0}))
            msgs = [json.loads(ws.receive_text()) for _ in range(6)]
            st = [m for m in msgs if m["type"] == "rover_state"]
            assert st and len(st[0]["scan"]) == 36
            assert any(m["type"] == "error" for m in msgs)
            assert d.cmd == (0.0, 0.0)


def test_rover_urdf_committed_is_current():
    from src.sim.urdf import build_rover_urdf
    committed = Path("ros2/qi_vla_ros/urdf/qi_vla_rover.urdf").read_text(encoding="utf-8")
    assert committed.replace("\r\n", "\n") == build_rover_urdf()


def test_nav_training_pipeline_smoke(tmp_path):
    from src.sim.mobile.train_nav import collect, load_nav, train
    out = collect(str(tmp_path / "nav"), maps=("home",), episodes=2, workers=1, unknown=0)
    assert out["episodes"] >= 1
    X, Y, ep = load_nav(str(tmp_path / "nav"))
    assert X.shape[1] == 25 and Y.shape[1] == 2 and np.all(np.abs(Y) <= 1)
    r = train(str(tmp_path / "nav"), str(tmp_path / "none"), seeds=1, steps=20, heal_steps=5, chi=8, hidden=64,
              n_train=1, n_heldout=1, n_hard=1, n_unknown=1, max_t=2.0, out_models=str(tmp_path),
              out_json=str(tmp_path / "r.json"), workers=1)
    assert {"dense_dr", "ttns_dr", "healing_ctrl", "ttns_noheal", "dense_dr_noshield"} <= set(r["summary"])
    spec = json.loads((tmp_path / "nav_ttns.json").read_text())
    from src.robot.policy_runtime import PolicyRuntime
    rt = PolicyRuntime(spec)
    for x, y in zip(spec["test_vectors"]["x"], spec["test_vectors"]["y"]):
        assert np.allclose(rt.forward(x), y, atol=1e-4)
    for c in spec["parity_cases"]:
        assert np.allclose(nav_features(c["pose"], c["scan"], c["target"], c["goal"], c["vw"]), c["x"], atol=1e-5)
