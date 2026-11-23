"""Real-robot data pipeline on a synthetic LeRobot-v3 dataset (a real mp4 video
is encoded on the fly, so CI needs no download): decode + alignment, training
of every model family, replay deployment loop, web export + parity."""

import json
from pathlib import Path

import numpy as np
import pytest

av = pytest.importorskip("av")
pa = pytest.importorskip("pyarrow")
import pyarrow.parquet as pq  # noqa: E402

FPS, EPS, T, CAM = 30, 4, 45, "observation.images.top"


def make_dataset(root: Path, name: str = "fake_so100") -> str:
    base = root / name
    (base / "meta" / "episodes" / "chunk-000").mkdir(parents=True)
    (base / "data" / "chunk-000").mkdir(parents=True)
    vdir = base / "videos" / CAM / "chunk-000"
    vdir.mkdir(parents=True)
    rng = np.random.default_rng(0)
    rows, eps = {k: [] for k in ("observation.state", "action", "timestamp", "frame_index", "episode_index", "index", "task_index")}, []
    with av.open(str(vdir / "file-000.mp4"), "w") as c:
        s = c.add_stream("mpeg4", rate=FPS)
        s.width, s.height, s.pix_fmt = 96, 96, "yuv420p"
        g = 0
        for e in range(EPS):
            target = rng.uniform(-50, 50, 2)            # object position -> visible as a bright square
            for t in range(T):
                img = np.full((96, 96, 3), 40, np.uint8)
                cx, cy = int(48 + target[0] * 0.6), int(48 + target[1] * 0.6)
                img[cy - 6:cy + 6, cx - 6:cx + 6] = (230, 60, 60)
                for p in s.encode(av.VideoFrame.from_ndarray(img, format="rgb24")):
                    c.mux(p)
                q = np.r_[target * (t / T), [0, 0, 0], [30.0 if t < T // 2 else 5.0]]
                rows["observation.state"].append(q.astype(np.float32).tolist())
                rows["action"].append((q + np.r_[target * (1 / T), [0, 0, 0, 0]]).astype(np.float32).tolist())
                rows["timestamp"].append(t / FPS)
                rows["frame_index"].append(t)
                rows["episode_index"].append(e)
                rows["index"].append(g)
                rows["task_index"].append(0)
                g += 1
            eps.append({"episode_index": e, f"videos/{CAM}/chunk_index": 0, f"videos/{CAM}/file_index": 0,
                        f"videos/{CAM}/from_timestamp": e * T / FPS, f"videos/{CAM}/to_timestamp": (e + 1) * T / FPS,
                        "length": T})
        for p in s.encode():
            c.mux(p)
    pq.write_table(pa.table(rows), base / "data" / "chunk-000" / "file-000.parquet")
    pq.write_table(pa.Table.from_pylist(eps), base / "meta" / "episodes" / "chunk-000" / "file-000.parquet")
    pq.write_table(pa.table({"task_index": [0], "task": ["put the red square in the box"]}), base / "meta" / "tasks.parquet")
    (base / "meta" / "info.json").write_text(json.dumps({
        "codebase_version": "v3.0", "fps": FPS,
        "video_path": "videos/{video_key}/chunk-{chunk_index:03d}/file-{file_index:03d}.mp4"}))
    return name


@pytest.fixture(scope="module")
def fake(tmp_path_factory):
    import src.real.data as D
    root = tmp_path_factory.mktemp("real")
    name = make_dataset(root)
    D.MAIN_CAMERA[name] = CAM
    D.ROBOT_OF[name] = "so100"
    old = D.ROOT
    D.ROOT = root
    yield root, name
    D.ROOT = old


def test_prepare_aligns_video_frames(fake):
    import src.real.data as D
    root, name = fake
    d = D.prepare(name, size=32, hz=10, root=root)
    assert d.img.shape == (EPS * T // 3, 32, 32, 3) and len(d.episodes) == EPS
    # the bright square moves with the episode's target: frames of different episodes differ
    e0, e1 = d.img[d.episode == 0][0].astype(int), d.img[d.episode == 1][0].astype(int)
    assert np.abs(e0 - e1).mean() > 1.0
    again = D.prepare(name, size=32, hz=10, root=root)   # cache hit
    assert np.array_equal(again.img, d.img)
    ch = D.chunk_actions(d.action, d.episode, 4)
    assert ch.shape == (len(d.action), 4, 6)


def test_train_all_models_replay_and_export(fake, tmp_path, monkeypatch):
    import src.real.data as D
    import src.real.train_real as TR
    root, name = fake
    orig = D.prepare
    monkeypatch.setattr(TR, "prepare", lambda n, size=64: orig(n, size=size, root=root))
    monkeypatch.setattr(TR, "available", lambda: [name])
    out = TR.run([name], seeds=1, steps=6, heal_steps=2, head_steps=2,
                 models=["state_mlp", "cnn_mlp", "act_tf", "healing_ctrl", "cnn_ttns", "cnn_ttns_noheal",
                         "cnn_mps", "cnn_int8", "head_tanh", "head_surrogate"],
                 out_json=str(tmp_path / "r.json"), out_models=str(tmp_path / "m"))
    s = out["summary"]
    assert s["cnn_ttns"]["params"] < s["cnn_mlp"]["params"]
    for k in ("nmse_first", "motion_nmse", "joint_mae", "grip_acc"):
        assert np.isfinite(s["cnn_mlp"][k]["mean"])
    import src.real.deploy as DP
    real_prepare = TR.prepare                       # the root-bound lambda
    monkeypatch.setattr(D, "prepare", real_prepare)  # deploy/export import it from src.real.data at call time
    res = DP.replay(str(tmp_path / "m" / "cnn_ttns.pt"), name, episodes=1)
    assert res["episodes"] and res["latency_ms_p95"] > 0 and np.isfinite(res["joint_mae"])
    import src.real.export_web as EW
    monkeypatch.setattr(EW, "prepare", real_prepare)
    info = EW.export(str(tmp_path / "m" / "cnn_ttns.pt"), out_dir=str(tmp_path / "web"), episodes=1, frames=5)
    spec = json.loads(Path(info["model"]).read_text())
    assert spec["format"] == "qi-vla-vision/1" and spec["head"][0]["type"] == "ttns" and "test_vector" in spec
    ep = json.loads((tmp_path / "web" / "real" / json.loads((tmp_path / "web" / "real" / "index.json").read_text())["episodes"][0]["file"]).read_text())
    assert len(ep["frames"]) == 5 and len(ep["py_pred_first"]) == 5


def test_controller_safety_clamps():
    import torch
    from src.real.deploy import Controller
    from src.real.models import Policy

    class Big(torch.nn.Module):          # always predicts a huge jump
        k = 3

        def forward(self, img, st, tk):
            return torch.full((1, 3, 6), 50.0)
    norm = {"state": {"mean": [0] * 6, "std": [1] * 6}, "action": {"mean": [0] * 6, "std": [1] * 6, "min": [-10] * 6, "max": [10] * 6}}
    ck = {"model": Big(), "names": ["d"], "norm": {"d": norm}, "size": 8, "k": 3}
    c = Controller(ck, "d", max_delta=2.0)
    a = c.act(np.zeros((8, 8, 3), np.uint8), np.zeros(6))["action"]
    assert np.allclose(a, 2.0)                      # limited to max_delta from the current pose
    for _ in range(10):
        a = c.act(np.zeros((8, 8, 3), np.uint8), np.zeros(6))["action"]
    assert np.allclose(a, 10.0)                     # and never beyond the recorded max
    assert Policy("cnn_mlp", 2, 3)(torch.zeros(1, 3, 64, 64), torch.zeros(1, 6), torch.zeros(1, 2)).shape == (1, 3, 6)
