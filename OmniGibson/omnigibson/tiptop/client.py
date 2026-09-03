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

    def _check_metadata(self, request: dict) -> None:
        """Refuse to execute plans made for another embodiment; warn when gt_* keys would be ignored."""
        robot_type, dof = self.metadata.get("robot_type"), self.metadata.get("dof")
        if self.expected_robot_type and robot_type not in (None, self.expected_robot_type):
            raise TiptopPlanningError(
                f"tiptop-server plans for robot_type={robot_type!r} but this client executes on {self.expected_robot_type!r}; "
                f"start the server with --config tiptop/config/tiptop_sim_panda.yml"
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
    """Streams the simulator's robot joints and object poses to the server so its Rerun view follows the sim.

    Protocol (additive, tiptop fork >= 6a790df+): after the metadata frame, the client sends msgpack dicts
    {"type": "sim_state", "t": s, "q": (7,) arm rad, "q_gripper": finger opening m,
     "objects": {label: (4,4) base-frame transform relative to the pose at capture}}
    until it closes the connection. The server logs them into its current Rerun recording. Any failure only
    disables the stream; execution never depends on it.
    """

    def __init__(self, host: str, port: int = 8765, every: int = 1):
        self.uri = f"ws://{host}:{port}"
        self.every = max(1, int(every))
        self.ws = None
        self.sent = 0
        self.calls = 0

    def open(self) -> bool:
        try:
            self.ws = connect(self.uri, compression=None, max_size=None, open_timeout=10.0, close_timeout=2.0)
            metadata = unpackb(self.ws.recv(timeout=30.0))
            if not metadata.get("sim_state_supported"):
                log.warning("tiptop-server does not accept sim_state messages; not streaming sim state")
                self.close()
                return False
            log.info(f"streaming sim state to {self.uri} every {self.every} step(s)")
            return True
        except Exception as e:
            log.warning(f"sim state stream unavailable ({e})")
            self.ws = None
            return False

    def send(self, t: float, q_arm, q_gripper: float, objects: dict) -> None:
        self.calls += 1
        if self.ws is None or (self.calls - 1) % self.every:
            return
        msg = {
            "type": "sim_state",
            "t": float(t),
            "q": np.asarray(q_arm, dtype=np.float32),
            "q_gripper": float(q_gripper),
            "objects": {k: np.asarray(v, dtype=np.float32) for k, v in objects.items()},
        }
        try:
            self.ws.send(packb(msg))
            self.sent += 1
        except Exception as e:
            log.warning(f"sim state stream stopped after {self.sent} messages ({e})")
            self.close()

    def close(self) -> None:
        if self.ws is not None:
            try:
                self.ws.close()
            except Exception:
                pass
            self.ws = None
