"""QI-VLA bridge: connects the app (phone/browser) to a real or mock robot.

    python -m src.bridge.server --driver mock                 # no hardware
    python -m src.bridge.server --driver serial --port COM3 --cal calib.json
    python -m src.bridge.server --driver mujoco --env kitchen_counter   # physics digital twin
    python -m src.bridge.server --driver rover --map warehouse --unknown 2   # mobile robot (MuJoCo)

Endpoints
  WS   /ws       protocol "qi-vla-bridge/1" (same as the ESP32 firmware)
  POST /act      server-side policy inference {q, target} -> {qd}
  GET  /health, /info
  GET  /         the built web app (app/dist), if present — open it from any browser

Protocol (JSON text frames)
  app -> bridge: {"type":"hello"} | {"type":"joint_command","q":[6 floats],"seq":n}
                 {"type":"estop"} | {"type":"estop_release"}
                 {"type":"scene_reset","env":id,"seed":n}      (simulated drivers only)
                 {"type":"twist_command","v":m/s,"w":rad/s}   (mobile robots)
  bridge -> app: {"type":"info",...} | {"type":"state","q":[...],"t":ms,"estop":bool}
                 {"type":"scene","objects":{color:[x,y,z,yaw]},"pads":{color:[x,y]},"env":id,...}
                 {"type":"rover_state","pose":[x,y,yaw],"vw":[v,w],"scan":[36],"map":id}  (mobile)
                 {"type":"error","message":...}

Bridge-side safety (defense in depth — the app's shield is NOT trusted alone):
  * joint-limit clamp and per-tick step limit (QD_MAX * dt * 1.5)
  * watchdog: no command for `watchdog_s` -> driver.stop() (hold pose)
  * e-stop latch, released only by an explicit estop_release
  * one controlling client; later clients are read-only observers
"""

from __future__ import annotations

import argparse
import asyncio
import json
import time
from contextlib import asynccontextmanager
from pathlib import Path
from typing import List, Optional

import numpy as np
from fastapi import Body, FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import JSONResponse
from pydantic import BaseModel

from src.bridge.drivers import Driver, make_driver
from src.robot import arm
from src.robot.policy_runtime import PolicyRuntime

PROTOCOL = "qi-vla-bridge/1"


class BridgeCore:
    """Transport-independent safety + state logic (unit-testable without sockets)."""

    def __init__(self, driver: Driver, rate_hz: float = 50.0, watchdog_s: float = 0.5) -> None:
        self.driver = driver
        self.dt = 1.0 / rate_hz
        self.watchdog_s = watchdog_s
        self.estop = False
        self.last_cmd_t: Optional[float] = None
        self.last_sent: Optional[np.ndarray] = None
        self.holding = True
        self.stats = {"commands": 0, "clamped": 0, "watchdog_trips": 0, "rejected": 0}

    def command(self, q: List[float], now: Optional[float] = None) -> Optional[List[float]]:
        now = time.monotonic() if now is None else now
        if self.estop:
            self.stats["rejected"] += 1
            return None
        qa = np.asarray(q, dtype=np.float64)
        if qa.shape != (6,) or not np.all(np.isfinite(qa)):
            self.stats["rejected"] += 1
            return None
        safe = arm.clamp_q(qa)
        ref = self.last_sent if self.last_sent is not None else np.asarray(self.driver.read())
        span = (now - self.last_cmd_t) if self.last_cmd_t is not None else self.dt
        max_step = arm.QD_MAX * max(min(span, 0.2), self.dt) * 1.5
        safe = np.clip(safe, ref - max_step, ref + max_step)
        if not np.allclose(safe, qa):
            self.stats["clamped"] += 1
        self.driver.write(safe.tolist())
        self.last_sent, self.last_cmd_t, self.holding = safe, now, False
        self.stats["commands"] += 1
        return safe.tolist()

    def tick(self, now: Optional[float] = None) -> None:
        now = time.monotonic() if now is None else now
        if not self.holding and self.last_cmd_t is not None and now - self.last_cmd_t > self.watchdog_s:
            self.driver.stop()
            self.holding = True
            self.last_sent = None
            self.stats["watchdog_trips"] += 1

    def trigger_estop(self) -> None:
        self.estop = True
        self.driver.stop()
        self.holding, self.last_sent = True, None

    def release_estop(self) -> None:
        self.estop = False


class ActRequest(BaseModel):
    q: List[float]
    target: List[float]
    instruction: str = ""


def build_app(driver: Driver, policy_path: Optional[str] = None,
              static_dir: Optional[str] = None, watchdog_s: float = 0.5,
              demo_dir: str = "data/app_demos") -> FastAPI:
    is_rover = getattr(driver, "kind", "arm") == "rover"
    core = BridgeCore(driver, watchdog_s=watchdog_s)
    policy = PolicyRuntime.load(policy_path) if policy_path and Path(policy_path).exists() else None
    state = {"controller": None, "clients": set()}

    @asynccontextmanager
    async def lifespan(_app: FastAPI):
        driver.connect()

        async def loop() -> None:
            k = 0
            while True:
                if is_rover:
                    if k % 3 == 0 and state["clients"]:
                        msgs = [json.dumps(dict(driver.read_state(), type="rover_state", estop=core.estop))]
                        if k % 25 == 0:
                            msgs.append(json.dumps(dict(driver.scene(), type="scene")))
                        for ws in list(state["clients"]):
                            try:
                                for m_ in msgs:
                                    await ws.send_text(m_)
                            except Exception:
                                state["clients"].discard(ws)
                    k += 1
                    await asyncio.sleep(core.dt)
                    continue
                core.tick()
                if k % 3 == 0 and state["clients"]:  # ~16 Hz state broadcast at 50 Hz tick
                    msg = json.dumps({"type": "state", "q": [round(v, 5) for v in driver.read()],
                                      "t": int(time.time() * 1000), "estop": core.estop})
                    msgs = [msg]
                    if k % 12 == 0 and hasattr(driver, "scene"):  # ~4 Hz scene mirror (simulated drivers)
                        msgs.append(json.dumps(dict(driver.scene(), type="scene")))
                    for ws in list(state["clients"]):
                        try:
                            for m_ in msgs:
                                await ws.send_text(m_)
                        except Exception:  # client went away mid-send
                            state["clients"].discard(ws)
                k += 1
                await asyncio.sleep(core.dt)
        task = asyncio.create_task(loop())
        try:
            yield
        finally:
            task.cancel()
            driver.stop()
            driver.close()

    app = FastAPI(title="QI-VLA Bridge", version="0.1.0", lifespan=lifespan)
    # The APK's WebView (origin http://localhost) and dev servers call /act
    # cross-origin. The bridge is a LAN tool with no auth: run it only on a
    # trusted network (see README "Keamanan").
    from fastapi.middleware.cors import CORSMiddleware
    app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])

    @app.get("/health")
    def health():
        return {"status": "ok", "protocol": PROTOCOL, "estop": core.estop, "stats": core.stats}

    @app.get("/info")
    def info():
        return dict(driver.info(), protocol=PROTOCOL, policy=policy.spec["method"] if policy else None)

    @app.post("/act")
    def act(req: ActRequest):
        if policy is None:
            return JSONResponse({"error": "no policy loaded (--policy)"}, status_code=503)
        t0 = time.perf_counter()
        qd = policy.act(req.q, req.target)
        return {"qd": qd, "latency_ms": (time.perf_counter() - t0) * 1000, "policy": policy.spec["method"]}

    @app.post("/episodes")
    def save_episodes(payload: dict = Body(...)):
        """Store demonstrations recorded in the app (format qi-vla-demos/1); convert
        to a training dataset with `python scripts/import_app_demos.py`."""
        if payload.get("format") != "qi-vla-demos/1" or not isinstance(payload.get("episodes"), list):
            return JSONResponse({"error": "expected format qi-vla-demos/1"}, status_code=400)
        d = Path(demo_dir)
        d.mkdir(parents=True, exist_ok=True)
        f = d / f"demos_{time.strftime('%Y%m%d_%H%M%S')}_{len(list(d.glob('*.json')))}.json"
        f.write_text(json.dumps(payload), encoding="utf-8")
        return {"saved": str(f), "episodes": len(payload["episodes"]),
                "frames": sum(len(e.get("frames", [])) for e in payload["episodes"])}

    @app.websocket("/ws")
    async def ws_endpoint(ws: WebSocket):
        await ws.accept()
        state["clients"].add(ws)
        is_ctrl = state["controller"] is None
        if is_ctrl:
            state["controller"] = ws
        await ws.send_text(json.dumps(dict(driver.info(), type="info", protocol=PROTOCOL,
                                           role="controller" if is_ctrl else "observer")))
        try:
            while True:
                try:
                    m = json.loads(await ws.receive_text())
                except ValueError:
                    continue
                t = m.get("type")
                if t == "estop":  # any client may stop the robot
                    core.estop = True
                    driver.stop()
                    if not is_rover:
                        core.trigger_estop()
                elif t == "twist_command" and is_rover:
                    if state["controller"] is not ws:
                        await ws.send_text(json.dumps({"type": "error", "message": "observer: read-only"}))
                    elif core.estop or driver.twist(float(m.get("v", 0.0)), float(m.get("w", 0.0))) is None:
                        await ws.send_text(json.dumps({"type": "error", "message": "rejected (e-stop latched or bad command)"}))
                elif t == "estop_release" and state["controller"] is ws:
                    core.release_estop()
                elif t == "scene_reset" and state["controller"] is ws and hasattr(driver, "reset_scene"):
                    try:
                        driver.reset_scene(m.get("map") or m.get("env"), m.get("seed"))
                        core.last_sent = None
                        await ws.send_text(json.dumps(dict(driver.scene(), type="scene")))
                    except (FileNotFoundError, ValueError) as e:
                        await ws.send_text(json.dumps({"type": "error", "message": f"scene_reset: {e}"}))
                elif t == "joint_command":
                    if state["controller"] is not ws:
                        await ws.send_text(json.dumps({"type": "error", "message": "observer: read-only"}))
                    elif core.command(m.get("q", [])) is None:
                        await ws.send_text(json.dumps({"type": "error", "message":
                                                       "rejected (e-stop latched or bad command)"}))
        except WebSocketDisconnect:
            pass
        finally:
            state["clients"].discard(ws)
            if state["controller"] is ws:
                state["controller"] = None
                driver.stop()  # controller gone -> hold pose

    if static_dir and Path(static_dir).is_dir():
        from fastapi.staticfiles import StaticFiles
        app.mount("/", StaticFiles(directory=static_dir, html=True), name="app")
    return app


def main() -> None:
    import uvicorn

    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--driver", default="mock", choices=["mock", "serial", "mujoco", "rover", "rover-kinematic"])
    p.add_argument("--map", default="home", help="map for --driver rover (app/public/maps)")
    p.add_argument("--unknown", type=int, default=0, help="obstacles not in the map (rover)")
    p.add_argument("--env", default="lab_table", help="environment preset for --driver mujoco")
    p.add_argument("--randomize", action="store_true", help="domain-randomized dynamics (mujoco)")
    p.add_argument("--port", default=None, help="serial port, e.g. COM3 or /dev/ttyUSB0")
    p.add_argument("--cal", default=None, help="calibration JSON for the serial driver")
    p.add_argument("--policy", default="app/public/models/policy_ttns.json")
    p.add_argument("--static", default="app/dist")
    p.add_argument("--host", default="0.0.0.0")
    p.add_argument("--http-port", type=int, default=8765)
    a = p.parse_args()
    app = build_app(make_driver(a.driver, a.port, a.cal, env_id=a.env, randomize=a.randomize,
                                map_id=a.map, unknown=a.unknown),
                    a.policy, a.static)
    uvicorn.run(app, host=a.host, port=a.http_port)


if __name__ == "__main__":
    main()
