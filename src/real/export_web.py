"""Export a real-data vision policy (+ a few held-out real episodes) for the app.

    python -m src.real.export_web --model models/real/cnn_ttns.pt

Writes
  app/public/models/real_<kind>.json   format qi-vla-vision/1: conv encoder weights, head layers
                                       (dense or TTNS factors, same layer format as qi-vla-policy/1),
                                       per-dataset normalization, chunk size
  app/public/real/index.json           list of exported episodes
  app/public/real/<dataset>_<ep>.json  64x64 JPEG frames (base64), measured joints, teleop actions,
                                       and the PYTHON model's predictions (JS parity test)
Only cnn_mlp-family models (incl. TTNS/MPS-compressed heads) are exportable.
"""

from __future__ import annotations

import argparse
import base64
import io
import json
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

from src.compression.tensor_network import TTNSLinear
from src.real.data import episode_split, prepare


def _r(t) -> list:
    return np.asarray(t.detach().numpy(), dtype=np.float64).round(6).tolist()


def export_head(head: nn.Sequential) -> list:
    out, mods = [], list(head)
    for i, m in enumerate(mods):
        act = "gelu_tanh" if i + 1 < len(mods) and isinstance(mods[i + 1], nn.GELU) else None
        if isinstance(m, nn.Linear):
            out.append({"type": "dense", "W": _r(m.weight), "b": _r(m.bias), "act": act})
        elif isinstance(m, TTNSLinear):
            out.append({"type": "ttns", "right": _r(m.right), "root": _r(m.root), "left": _r(m.left), "b": _r(m.bias), "act": act})
        elif not isinstance(m, nn.GELU):
            raise ValueError(f"cannot export layer {type(m).__name__} (export cnn_mlp / cnn_ttns)")
    return out


def export(model_path: str, out_dir: str = "app/public", episodes: int = 1, frames: int = 60) -> dict:
    ck = torch.load(model_path, weights_only=False)
    m = ck["model"].eval()
    if ck["kind"] not in ("cnn_mlp", "cnn_ttns") or not hasattr(m, "enc"):
        raise ValueError("export supports cnn_mlp / cnn_ttns checkpoints")
    convs = [c for c in m.enc.net if isinstance(c, nn.Conv2d)]
    spec = {"format": "qi-vla-vision/1", "kind": ck["kind"], "k": ck["k"], "size": ck["size"], "names": ck["names"],
            "norm": ck["norm"], "params": int(sum(p.numel() for p in m.parameters())),
            "conv": [{"W": _r(c.weight), "b": _r(c.bias), "stride": c.stride[0], "pad": c.padding[0]} for c in convs],
            "head": export_head(m.head),
            "note": "conv: GELU (exact erf) after every conv; features = [spatial-softmax x/y (2*128), mean-pool (128)] "
                    "+ normalized joints (6) + task one-hot; head output = K x 6 normalized actions"}
    # raw-pixel test vector for the JS parity test (Node has no image decoder)
    d0 = prepare(ck["names"][0], size=ck["size"])
    n0 = ck["norm"][ck["names"][0]]
    st0 = (d0.state[0] - np.asarray(n0["state"]["mean"], np.float32)) / np.asarray(n0["state"]["std"], np.float32)
    with torch.no_grad():
        y0 = m(torch.from_numpy(d0.img[:1]).permute(0, 3, 1, 2).float() / 255.0, torch.from_numpy(st0[None]),
               torch.nn.functional.one_hot(torch.tensor([0]), len(ck["names"])).float())[0]
    spec["test_vector"] = {"rgb_b64": base64.b64encode(d0.img[0].tobytes()).decode(), "state_norm": st0.round(6).tolist(),
                           "task": 0, "y": y0.numpy().round(6).tolist()}
    out = Path(out_dir)
    (out / "models").mkdir(parents=True, exist_ok=True)
    (out / "models" / f"real_{ck['kind']}.json").write_text(json.dumps(spec), encoding="utf-8")
    from PIL import Image
    index = []
    (out / "real").mkdir(parents=True, exist_ok=True)
    for ti, name in enumerate(ck["names"]):
        d = prepare(name, size=ck["size"])
        _, _, test = episode_split(d.episode, 0)
        n = ck["norm"][name]
        sm, ss = np.asarray(n["state"]["mean"], np.float32), np.asarray(n["state"]["std"], np.float32)
        task = torch.nn.functional.one_hot(torch.tensor([ti]), len(ck["names"])).float()
        for e in sorted(test)[:episodes]:
            sel = np.nonzero(d.episode == e)[0][:frames]
            imgs, preds = [], []
            for i in sel:   # PNG = lossless: the browser decodes exactly the pixels the model saw
                buf = io.BytesIO()
                Image.fromarray(d.img[i]).save(buf, format="PNG", optimize=True)
                imgs.append(base64.b64encode(buf.getvalue()).decode())
            with torch.no_grad():
                x = torch.from_numpy(d.img[sel]).permute(0, 3, 1, 2).float() / 255.0
                st = torch.from_numpy((d.state[sel] - sm) / ss)
                preds = m(x, st, task.repeat(len(sel), 1))[:, 0].numpy()
            ep = {"dataset": name, "robot": n["robot"], "task": n["task"], "episode": int(e), "fps": d.fps,
                  "frames": imgs, "state": d.state[sel].round(4).tolist(), "action": d.action[sel].round(4).tolist(),
                  "py_pred_first": preds.round(5).tolist()}
            fn = f"{name}_{int(e)}.json"
            (out / "real" / fn).write_text(json.dumps(ep), encoding="utf-8")
            index.append({"file": fn, "dataset": name, "robot": n["robot"], "task": n["task"], "episode": int(e), "frames": len(sel)})
    (out / "real" / "index.json").write_text(json.dumps({"model": f"models/real_{ck['kind']}.json", "episodes": index}, indent=1), encoding="utf-8")
    return {"model": str(out / "models" / f"real_{ck['kind']}.json"), "episodes": len(index)}


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--model", default="models/real/cnn_ttns.pt")
    p.add_argument("--episodes", type=int, default=1)
    a = p.parse_args()
    print(export(a.model, episodes=a.episodes))


if __name__ == "__main__":
    main()
