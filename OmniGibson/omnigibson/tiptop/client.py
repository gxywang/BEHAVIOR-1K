"""Blocking websocket client for `tiptop-server` (one observation in, one full plan out)."""

import json
import logging
import time
import urllib.request

from websockets.sync.client import connect

import numpy as np
from omnigibson.tiptop.protocol import packb, parse_plan, unpackb

log = logging.getLogger(__name__)


class TiptopPlanningError(RuntimeError):
    pass


class TiptopClient:
    """Speaks tiptop's protocol: msgpack-numpy metadata on connect, msgpack-numpy request, JSON text response."""

    def __init__(
        self,
        host: str = "localhost",
        port: int = 8765,
        connect_retries: int = 12,
        retry_wait_s: float = 5.0,
        expected_robot_type: str | None = "panda",
        expected_dof: int | None = 7,
    ):
        self.host, self.port = host, port
        self.connect_retries, self.retry_wait_s = connect_retries, retry_wait_s
        self.expected_robot_type, self.expected_dof = expected_robot_type, expected_dof
        self.metadata: dict | None = None
        self.last_response: dict | None = None

    @property
    def uri(self) -> str:
        return f"ws://{self.host}:{self.port}"

    def health(self, timeout_s: float = 3.0) -> bool:
        """GET /health on the websocket port answers once the server finished its cuRobo warm-up."""
        try:
            with urllib.request.urlopen(f"http://{self.host}:{self.port}/health", timeout=timeout_s) as resp:
                return resp.status == 200
        except Exception:
            return False

    def wait_for_server(self, timeout_s: float = 600.0) -> None:
        t0 = time.time()
        while not self.health():
            if time.time() - t0 > timeout_s:
                raise TiptopPlanningError(f"tiptop-server at {self.uri} did not become healthy within {timeout_s}s")
            log.info(f"waiting for tiptop-server at {self.uri} ...")
            time.sleep(5.0)

    def fetch_metadata(self) -> dict:
        """Connect, read the server's metadata frame and disconnect (embodiment details before capturing)."""
        ws = connect(self.uri, compression=None, max_size=None, open_timeout=30.0, close_timeout=5.0)
        try:
            self.metadata = unpackb(ws.recv(timeout=60.0))
        finally:
            ws.close()
        log.info(f"tiptop-server metadata: {self.metadata}")
        return self.metadata

    def check_embodiment(self) -> None:
        """Validate robot_type / dof against what this client will execute; adopt dof from the embodiment metadata."""
        emb = self.metadata.get("embodiment") or {}
        if self.expected_dof is None and emb.get("joint_names"):
            self.expected_dof = len(emb["joint_names"])
        self._check_metadata({})

    def _check_metadata(self, request: dict) -> None:
        """Refuse to execute plans made for another embodiment; warn when gt_* keys would be ignored."""
        robot_type, dof = self.metadata.get("robot_type"), self.metadata.get("dof")
        if self.expected_robot_type and robot_type not in (None, self.expected_robot_type):
            raise TiptopPlanningError(
                f"tiptop-server plans for robot_type={robot_type!r} but this client executes on {self.expected_robot_type!r}; "
                f"start the server with the matching --config tiptop/config/tiptop_sim_*.yml"
            )
        if self.expected_dof and dof not in (None, self.expected_dof):
            raise TiptopPlanningError(f"tiptop-server dof={dof} does not match the {self.expected_dof}-DoF arm")
        if robot_type is None:
            log.warning("server metadata has no robot_type (older tiptop); cannot verify the embodiment")
        if "gt_masks" in request and not self.metadata.get("gt_detections_supported", False):
            log.warning("server does not advertise gt_detections_supported; gt_* keys will be ignored")

    def plan(self, request: dict, timeout_s: float = 900.0) -> dict:
        """Send one request and return the server response with a parsed plan under response['plan'].

        Reconnects for every request, as the reference client does, so the server handler starts fresh.
        """
        last_error = None
        for attempt in range(self.connect_retries):
            try:
                ws = connect(self.uri, compression=None, max_size=None, open_timeout=30.0, close_timeout=5.0)
                break
            except OSError as e:
                last_error = e
                log.warning(f"connect to {self.uri} failed ({e}); retry {attempt + 1}/{self.connect_retries}")
                time.sleep(self.retry_wait_s)
        else:
            raise TiptopPlanningError(f"could not connect to {self.uri}: {last_error}")
        try:
            self.metadata = unpackb(ws.recv(timeout=60.0))
            log.info(f"tiptop-server metadata: {self.metadata}")
            self._check_metadata(request)
            payload = packb(request)
            t0 = time.time()
            log.info(f"sending {len(payload) / 1e6:.1f} MB request, task={request['task']!r}")
            ws.send(payload)
            raw = ws.recv(timeout=timeout_s)
        finally:
            ws.close()
        if isinstance(raw, bytes):
            raw = raw.decode(errors="replace")
        try:
            response = json.loads(raw)
        except json.JSONDecodeError as e:
            # On an unhandled exception the server sends str(e) as a text frame and closes with code 1011
            raise TiptopPlanningError(f"server raised before answering: {raw[:500]!r}") from e
        response["client_roundtrip_s"] = time.time() - t0
        self.last_response = response
        if not response.get("success"):
            raise TiptopPlanningError(f"planning failed: {response.get('error')} (save_dir={response.get('save_dir')})")
        response["plan"] = parse_plan(response["plan"])
        return response


class SimStateStream:
    """Mirrors the simulator into the planner's Rerun view over a second websocket connection to the same port.

    The server's ``_sim_state_stream`` is the other end. Messages are msgpack with numpy arrays:
      sim_scene  {"type": "sim_scene", "objects": {name: {"vertices": (N, 3) f32 in the object's own frame,
                  "faces": (M, 3) i32, "pose": (4, 4) f32 base-frame pose, "kind": "object" | "context"}}}
                 once per connection: the simulator's own meshes (task objects, and furniture named as context)
      sim_state  {"type": "sim_state", "t": simulated seconds, "q": (dof,) f32 planned joints in the server's joint
                  order, "q_gripper": finger opening (m), "objects": {name: (4, 4) f32 base-frame pose},
                  "images": {name: JPEG bytes}}   every ``every`` env steps; images in every ``image_every``-th one
    Names are the simulator's own (task instance names, ``candle_2``); the server keeps perception's names for its
    hulls and matches nothing by name. ``attach`` once per session; ``TiptopSim.step`` calls ``on_step``. A failure
    never reaches the episode: a dropped connection is retried every ``reconnect_s`` (``max_failures`` times in a
    row, then the mirror is off), an error while reading the simulator turns the mirror off at once.
    """

    def __init__(
        self,
        host: str,
        port: int = 8765,
        every: int = 2,
        image_every: int = 3,
        reconnect_s: float = 5.0,
        max_failures: int = 3,
    ):
        self.uri = f"ws://{host}:{port}"
        self.every = max(1, int(every))
        self.image_every = max(1, int(image_every))
        self.reconnect_s = reconnect_s
        self.max_failures = max_failures
        self.ws = None
        self.sim = None
        self.sent = 0
        self.calls = 0
        self.failures = 0  # connection attempts failed in a row
        self.disabled = False  # no mirror on the server, too many failed connections, or the simulator side failed
        self._last_attempt = 0.0

    def attach(self, sim) -> bool:
        """Connect, send the simulator's meshes and hook into ``sim.step``; False if not connected (yet)."""
        self.sim = sim
        ok = self._connect()
        sim.state_stream = None if self.disabled else self
        return ok

    def _connect(self) -> bool:
        self._last_attempt = time.time()
        try:
            self.ws = connect(self.uri, compression=None, max_size=None, open_timeout=3.0, close_timeout=2.0)
            metadata = unpackb(self.ws.recv(timeout=10.0))
            if not metadata.get("sim_state_supported"):
                log.warning("tiptop-server does not accept sim_state messages; not streaming sim state")
                self.close()
                self.disabled = True
                return False
            scene = self.sim.stream_scene()
            self.ws.send(packb({"type": "sim_scene", "objects": scene}))
            self.failures = 0
            log.info(
                f"streaming sim state to {self.uri}: {len(scene)} meshes, state every {self.every} step(s), "
                f"images in every {self.image_every}. message"
            )
            return True
        except Exception as e:
            self.close()
            self.failures += 1
            if self.failures >= self.max_failures:
                self.disabled = True
                log.warning(f"sim state stream unavailable ({e}); giving up after {self.failures} attempts")
            else:
                log.warning(f"sim state stream unavailable ({e}); retrying in {self.reconnect_s:.0f} s")
            return False

    def on_step(self, sim) -> None:
        self.calls += 1
        if self.disabled or (self.calls - 1) % self.every:
            return
        if self.ws is None and (time.time() - self._last_attempt < self.reconnect_s or not self._connect()):
            return
        try:
            msg = {
                "type": "sim_state",
                "t": float(sim.sim_time),
                "q": np.asarray(sim.q_arm(), dtype=np.float32),
                "q_gripper": float(sim.q_fingers()[0]),
                "objects": sim.object_poses_base_mats(),
            }
            if self.sent % self.image_every == 0:
                msg["images"] = sim.stream_images()
            payload = packb(msg)
        except Exception as e:  # noqa: BLE001 - the mirror must never end an episode
            log.warning(f"sim state mirror off: reading the simulator failed ({e})")
            self.close()
            self.disabled = True
            return
        try:
            self.ws.send(payload)
            self.sent += 1
        except Exception as e:  # noqa: BLE001
            log.warning(f"sim state stream dropped after {self.sent} messages ({e}); reconnecting")
            self.close()
            self._last_attempt = time.time()

    def close(self) -> None:
        if self.ws is not None:
            try:
                self.ws.close()
            except Exception:
                pass
            self.ws = None
