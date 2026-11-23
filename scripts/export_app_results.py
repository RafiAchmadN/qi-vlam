"""Compact the experiment results in results/*.json into the single file the
app's "Riset" tab reads (app/public/results/summary.json).

    python scripts/export_app_results.py
"""

from __future__ import annotations

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def load(name):
    p = ROOT / "results" / name
    return json.loads(p.read_text(encoding="utf-8")) if p.exists() else None


def main() -> None:
    out = {"format": "qi-vla-results/1"}
    pp = load("sim_pickplace.json")
    if pp and "summary" in pp:
        out["pickplace"] = {"summary": pp["summary"], "expert": pp["expert_reference"], "dataset": pp["dataset"],
                            "config": pp["config"], "scope": pp["scope"]}
    nav = load("sim_nav.json")
    if nav and "summary" in nav:
        out["nav"] = {"summary": nav["summary"], "expert": nav["expert_reference"], "dataset": nav["dataset"],
                      "config": nav["config"], "scope": nav["scope"]}
    zoo = load("model_zoo.json")
    if zoo and "summary" in zoo:
        out["zoo"] = {"summary": zoo["summary"], "config": zoo["config"], "notes": zoo["notes"], "dataset": zoo["dataset"]}
    for key, fname in (("real", "real_benchmark.json"), ("real_koch", "real_benchmark_koch.json")):
        rb = load(fname)
        if rb and "summary" in rb:
            out[key] = {"summary": {k: {kk: vv for kk, vv in v.items() if kk != "per_dataset"} | {"per_dataset": v["per_dataset"]}
                                    for k, v in rb["summary"].items()},
                        "datasets": rb["datasets"], "config": rb["config"], "scope": rb["scope"]}
    sh = load("app_shield_eval.json")
    if sh:
        out["shield"] = {"conditions": sh["conditions"], "n_episodes": sh["n_episodes"], "note": sh["note"]}
    lib = load("libero_benchmark.json")
    if lib:
        out["libero"] = {k: {"nmse": v["nmse"]["mean"], "nmse_std": v["nmse"]["std"],
                             "ratio": v.get("param_ratio_vs_dense", {}).get("mean"),
                             "latency_ms": v.get("latency_ms_p50", {}).get("mean")}
                         for k, v in lib["summary"].items()}
    ap = load("app_policy.json")
    if ap:
        out["app_policy"] = {k: ap[k] for k in ("params_dense", "params_ttns", "param_ratio", "closed_loop", "lyapunov") if k in ap}
    dst = ROOT / "app" / "public" / "results" / "summary.json"
    dst.parent.mkdir(parents=True, exist_ok=True)
    dst.write_text(json.dumps(out, separators=(",", ":")), encoding="utf-8")
    print(dst, round(dst.stat().st_size / 1024, 1), "KB", sorted(out))


if __name__ == "__main__":
    main()
