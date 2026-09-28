"""The websocket tape (WEEK4_PLAN 3.3, 5.5; W4-A2): every frame the bench exchanges with the planner servers,
recorded, logged or served back without a server.

One frame is one connection to a planner: the metadata the server sends on connect, the one request the client sends
(a legacy plan request, a ``move``, a ``skill`` or a ``reach``; none for ``fetch_metadata``) and the server's answer.
``b1k.bridge.client.connect`` is the one door every one of them goes through (``plan``, ``skill``, ``reach``,
``fetch_metadata``, ``move``, and the SimStateStream, whose frames are passed through and never taped), so patching
it is the whole hook; ``TiptopClient.health`` is patched in the replay modes so ``wait_for_server`` returns at once
without a server. Each frame is tagged with the StepLedger's owner and call id in flight and the op it carries.

Modes (``--wstape``):
- ``record``: every frame, whole (``index.jsonl`` plus ``frames/NNNNN.msgpack`` under the tape directory);
- ``log``: the index alone: per-field digests of every request, digests of every response;
- ``replay``: frames served by index; each request compared field by field with the recorded one, every field but
  VOLATILE; the first mismatch raises TapeDiverged, a BaseException, because bench.py swallows every Exception per
  round (bench.py:426) and a diverged replay must stop the instance, not the round;
- ``replay-log``: served by index, every diff logged (``wstape_replay.jsonl`` in the run's out dir), never stops;
- ``replay-live``: served by index until the first mismatch, or until frame ``live_at``, then real connections; the
  frames after the switch are recorded whole under ``<log_dir>/wstape_live``.

The seed stamp: every legacy plan request gets ``request["seed"] = 2300 + 1000 * replicate + k``, where k is the
index the server's own counter would give it: ``_run_pipeline`` increments ``_request_n`` once per plan request and
once per skill request that reaches it (tiptop_websocket_server.py:1026-1033, registry.dispatch), and the server
takes an explicit seed over its own (line 1028). k is counted per server (host:port): the right-arm press port is
its own server with its own counter. With replicate 0 the stamps equal what planner.log prints ("request k seeded
with 230k") for a server started with --seed 2300. The stamp is applied in every mode, so a replay compares it like
any other field: a k that drifts is a divergence, not a blind spot.

VOLATILE = {"rgb", "views[*].rgb"} is fixed (W4-P: the three RGB arrays differ across processes and reach no
sim-side decision). It is never extended; any other differing field is a divergence.
"""

from __future__ import annotations

import hashlib
import json
import logging
import threading
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from b1k.bridge import client as bridge_client
from b1k.bridge.protocol import packb, unpackb

log = logging.getLogger("omnigibson.tiptop.wstape")

MODES = ("off", "log", "record", "replay", "replay-log", "replay-live")
REPLAY_MODES = ("replay", "replay-log", "replay-live")
VOLATILE = frozenset({"rgb", "views[*].rgb"})  # fixed; never extended (WEEK4_PLAN 5.1)
SEED_BASE, SEED_STRIDE = 2300, 1000
STREAM_TYPES = ("sim_scene", "sim_state")
PIPELINE_OPS = ("plan", "skill")  # what the server's _request_n counts
OWN_SOLVE_SKILLS = ("articulate",)  # served by the skill's own solve, never through _run_pipeline
BUILD_REFUSALS = ("unsupported", "not_visible")  # registry.dispatch refuses these before _run_pipeline (phase None)
ROOT = Path(__file__).resolve().parents[4]  # the checkout this process runs (run.check_imports' root)
LIVE_RETRY_WAIT_S, LIVE_CONNECT_RETRIES = 5.0, 12  # TiptopClient._open's retries


class TapeDiverged(BaseException):
    """A replayed request differs from the tape at ``path`` (the first differing field), at frame ``index``."""

    def __init__(self, path: str, index: int, detail: str = "", op: str = ""):
        self.path, self.index, self.detail, self.op = path, index, detail, op
        super().__init__(f"tape diverged at frame {index} ({op}): {path}: {detail}")


class TapeExhausted(TapeDiverged):
    """The run asked for one frame more than the tape holds."""

    def __init__(self, index: int, op: str = ""):
        super().__init__("<end of tape>", index, f"the tape has {index} frames", op)


def seed_for(replicate: int, k: int) -> int:
    return SEED_BASE + SEED_STRIDE * int(replicate) + int(k)


def op_of(request) -> str:
    """What a sent request is: ``plan`` (a legacy planning request has no type), ``move``, ``skill``, ``reach``,
    ``stream`` (the SimStateStream's messages) or ``metadata`` (nothing sent: fetch_metadata)."""
    if request is None:
        return "metadata"
    kind = request.get("type") if isinstance(request, dict) else None
    if kind in STREAM_TYPES:
        return "stream"
    return str(kind) if kind else "plan"


def server_of(uri: str) -> str:
    return uri.split("://", 1)[-1].rstrip("/")


def is_volatile(path: str) -> bool:
    if path in VOLATILE:
        return True
    starred = _star(path)
    return starred in VOLATILE


def _star(path: str) -> str:
    out, i = [], 0
    while i < len(path):
        if path[i] == "[":
            j = path.index("]", i)
            out.append("[*]")
            i = j + 1
        else:
            out.append(path[i])
            i += 1
    return "".join(out)


def _join(path: str, key) -> str:
    if isinstance(key, int):
        return f"{path}[{key}]"
    return f"{path}.{key}" if path else str(key)


def _is_array(x) -> bool:
    return isinstance(x, (np.ndarray, np.generic))


def diff_fields(recorded, sent, path: str = "") -> list:
    """Every field where ``sent`` differs from ``recorded``, as (path, detail), VOLATILE paths left out, in the
    tape's key order then the replay's extra keys. Arrays are bit-exact (dtype, shape, bytes); floats by value with
    nan equal to nan."""
    if path and is_volatile(path):
        return []
    a, b = recorded, sent
    if isinstance(a, dict) and isinstance(b, dict):
        out = []
        for k in a:
            if k not in b:
                if not is_volatile(_join(path, k)):
                    out.append((_join(path, k), "missing in the replay"))
            else:
                out.extend(diff_fields(a[k], b[k], _join(path, k)))
        for k in b:
            if k not in a and not is_volatile(_join(path, k)):
                out.append((_join(path, k), "not on the tape"))
        return out
    if isinstance(a, (list, tuple)) and isinstance(b, (list, tuple)):
        if len(a) != len(b):
            return [(path or "<root>", f"length {len(a)} on the tape, {len(b)} in the replay")]
        out = []
        for i, (x, y) in enumerate(zip(a, b)):
            out.extend(diff_fields(x, y, _join(path, i)))
        return out
    if _is_array(a) or _is_array(b):
        if not (_is_array(a) and _is_array(b)):
            return [(path or "<root>", f"{type(a).__name__} on the tape, {type(b).__name__} in the replay")]
        a, b = np.asarray(a), np.asarray(b)
        if a.dtype != b.dtype:
            return [(path or "<root>", f"dtype {a.dtype} on the tape, {b.dtype} in the replay")]
        if a.shape != b.shape:
            return [(path or "<root>", f"shape {a.shape} on the tape, {b.shape} in the replay")]
        if a.tobytes() == b.tobytes():
            return []
        try:
            n = int(np.count_nonzero(a != b)) if a.dtype != object else "?"
            worst = (
                float(np.nanmax(np.abs(a.astype(np.float64) - b.astype(np.float64)))) if a.dtype.kind in "fiub" else "?"
            )
            return [(path or "<root>", f"{n} of {a.size} elements differ (max abs {worst})")]
        except Exception:  # noqa: BLE001
            return [(path or "<root>", f"{a.size} elements, bytes differ")]
    if isinstance(a, float) and isinstance(b, float) and a != a and b != b:
        return []
    numbers = isinstance(a, (int, float)) and isinstance(b, (int, float)) and isinstance(a, bool) == isinstance(b, bool)
    if type(a) is not type(b) and not numbers:
        return [(path or "<root>", f"{type(a).__name__} {a!r} on the tape, {type(b).__name__} {b!r} in the replay")]
    if a != b:
        return [(path or "<root>", f"{_short(a)} on the tape, {_short(b)} in the replay")]
    return []


def _short(x) -> str:
    s = repr(x)
    return s if len(s) <= 80 else s[:77] + "..."


def _feed(h, x) -> None:
    if isinstance(x, dict):
        h.update(b"{")
        for k in sorted(x, key=repr):
            _feed(h, k)
            h.update(b":")
            _feed(h, x[k])
        h.update(b"}")
    elif isinstance(x, (list, tuple)):
        h.update(b"[")
        for v in x:
            _feed(h, v)
            h.update(b",")
        h.update(b"]")
    elif _is_array(x):
        arr = np.asarray(x)
        h.update(f"<{arr.dtype}{arr.shape}>".encode())
        h.update(np.ascontiguousarray(arr).tobytes())
    elif isinstance(x, bytes):
        h.update(x)
    else:
        h.update(repr(x).encode())


def digest(x) -> str:
    h = hashlib.sha256()
    _feed(h, x)
    return h.hexdigest()[:16]


def field_digests(request) -> dict:
    """{path: digest} one level into the request, and one level into each view (``views[i].depth``)."""
    if not isinstance(request, dict):
        return {}
    out = {}
    for k, v in request.items():
        if k == "views" and isinstance(v, list):
            for i, view in enumerate(v):
                if isinstance(view, dict):
                    for vk, vv in view.items():
                        out[f"views[{i}].{vk}"] = digest(vv)
                else:
                    out[f"views[{i}]"] = digest(view)
        else:
            out[str(k)] = digest(v)
    return out


@dataclass
class Frame:
    index: int
    op: str = "metadata"
    owner: str = "unowned"
    call_id: str | None = None
    server: str = ""
    k: int | None = None
    seed: int | None = None
    metadata: dict | None = None
    request: bytes | None = None  # the payload as sent (stamped)
    response: object = None  # the raw answer frame (str or bytes)
    wall_s: float | None = None
    live: bool = False
    recv_error: tuple | None = None  # (which recv: 0 metadata, 1 answer; module, type, message) when it raised
    send_error: tuple | None = None  # (module, type, message) when the request's send raised: the server never saw it

    def request_dict(self):
        return None if self.request is None else unpackb(self.request)

    def response_json(self):
        raw = self.response
        if raw is None:
            return None
        if isinstance(raw, bytes):
            raw = raw.decode(errors="replace")
        try:
            return json.loads(raw)
        except (ValueError, TypeError):
            return None

    def row(self) -> dict:
        req, ans = self.request_dict(), self.response_json()
        ok = None
        if isinstance(ans, dict):
            ok = ans.get("success") if "success" in ans else ans.get("ok")
        return {
            "i": self.index,
            "op": self.op,
            "owner": self.owner,
            "call_id": self.call_id,
            "server": self.server,
            "k": self.k,
            "seed": self.seed,
            "task": req.get("task") if isinstance(req, dict) else None,
            "skill": req.get("skill") if isinstance(req, dict) else None,
            "request_sha": None if self.request is None else hashlib.sha256(self.request).hexdigest()[:16],
            "fields": field_digests(req),
            "response_sha": None if self.response is None else digest(self.response),
            "response_ok": ok,
            "response_code": ans.get("code") if isinstance(ans, dict) else None,
            "response_error": (ans.get("error") or None) if isinstance(ans, dict) else None,
            "wall_s": self.wall_s,
            "live": self.live,
            "recv_error": None if self.recv_error is None else list(self.recv_error),
            "send_error": None if self.send_error is None else list(self.send_error),
        }


# ---------------------------------------------------------------- the tape on disk


class TapeDir:
    """A recorded tape: ``index.jsonl`` (one row per frame) and, for a record, ``frames/NNNNN.msgpack``."""

    def __init__(self, path):
        self.path = Path(path)
        self.index = self.path / "index.jsonl"
        self.frames = self.path / "frames"

    def rows(self) -> list:
        if not self.index.exists():
            return []
        return [json.loads(line) for line in self.index.read_text().splitlines() if line.strip()]

    def __len__(self) -> int:
        return len(self.rows())

    @property
    def whole(self) -> bool:
        return self.frames.is_dir() and any(self.frames.glob("*.msgpack"))

    def frame_path(self, i: int) -> Path:
        return self.frames / f"{i:05d}.msgpack"

    def load(self, i: int) -> Frame | None:
        p = self.frame_path(i)
        if not p.exists():
            return None
        d = unpackb(p.read_bytes())
        response = d.get("response")
        if d.get("response_text") and isinstance(response, bytes):
            response = response.decode()
        return Frame(
            i,
            d["op"],
            d.get("owner", "unowned"),
            d.get("call_id"),
            d.get("server", ""),
            d.get("k"),
            d.get("seed"),
            d.get("metadata"),
            d.get("request"),
            response,
            d.get("wall_s"),
            bool(d.get("live", False)),
            None if d.get("recv_error") is None else tuple(d["recv_error"]),
            None if d.get("send_error") is None else tuple(d["send_error"]),
        )

    def append(self, frame: Frame, whole: bool) -> dict:
        self.path.mkdir(parents=True, exist_ok=True)
        row = frame.row()
        with open(self.index, "a") as f:
            f.write(json.dumps(row, default=_json_default) + "\n")
        if whole:
            self.frames.mkdir(exist_ok=True)
            response, text = frame.response, False
            if isinstance(response, str):
                response, text = response.encode(), True
            self.frame_path(frame.index).write_bytes(
                packb(
                    {
                        "op": frame.op,
                        "owner": frame.owner,
                        "call_id": frame.call_id,
                        "server": frame.server,
                        "k": frame.k,
                        "seed": frame.seed,
                        "metadata": frame.metadata,
                        "request": frame.request,
                        "response": response,
                        "response_text": text,
                        "wall_s": frame.wall_s,
                        "live": frame.live,
                        "recv_error": None if frame.recv_error is None else list(frame.recv_error),
                        "send_error": None if frame.send_error is None else list(frame.send_error),
                    }
                )
            )
        return row


def _recorded_error(err: tuple) -> Exception:
    """The exception a recorded recv raised: ``err`` is (which recv, module, type, message)."""
    _, module, name, message = err
    return rebuilt_error(module, name, message)


def rebuilt_error(module: str, name: str, message: str) -> Exception:
    """A recorded exception as its own class with its recorded message, so the client's per-round handling records
    the same ``Type: message`` it did: cls(message) when that constructor takes a message (TimeoutError), else the
    class built without its constructor (websockets' ConnectionClosedError(rcvd, sent) takes no message), a subclass
    of the same name whose str is the message when even its __str__ needs the missing fields; RuntimeError naming it
    only when the class is gone."""
    try:
        import importlib

        cls = getattr(importlib.import_module(module), name)
    except Exception:  # noqa: BLE001
        cls = None
    if not (isinstance(cls, type) and issubclass(cls, Exception)):
        return RuntimeError(f"{module}.{name}: {message}")
    try:
        e = cls(message)
        if str(e) == message:
            return e
    except Exception:  # noqa: BLE001 - a constructor with its own signature
        pass
    e = cls.__new__(cls)
    Exception.__init__(e, message)
    try:
        if str(e) == message:
            return e
    except Exception:  # noqa: BLE001 - a __str__ that reads fields the constructor would have set
        pass
    same = type(cls.__name__, (cls,), {"__module__": cls.__module__, "__str__": lambda self, m=message: m})
    e = same.__new__(same)
    Exception.__init__(e, message)
    return e


def _json_default(x):
    if isinstance(x, (np.integer,)):
        return int(x)
    if isinstance(x, (np.floating,)):
        return float(x)
    if isinstance(x, np.ndarray):
        return x.tolist()
    return str(x)


# ---------------------------------------------------------------- the connections


class _Recording:
    """A real connection whose frame is taped on close (record, log, and the live tail of replay-live)."""

    def __init__(self, tape: WsTape, uri: str, ws, live: bool = False):
        self.tape, self.uri, self.ws = tape, uri, ws
        self.frame = Frame(-1, server=server_of(uri), owner=tape.owner, call_id=tape.call_id, live=live)
        self.n_recv, self.stream, self.taped, self.t0 = 0, False, False, time.time()

    def recv(self, timeout=None, **kw):
        try:
            raw = self.ws.recv(timeout=timeout, **kw) if timeout is not None else self.ws.recv(**kw)
        except Exception as e:
            if self.n_recv <= 1 and not self.stream:  # the frame never came (a timeout): the replay raises the same
                self.frame.recv_error = (self.n_recv, type(e).__module__, type(e).__name__, str(e))
            raise
        if self.n_recv == 0:
            self.frame.metadata = unpackb(raw) if isinstance(raw, bytes) else raw
        elif self.n_recv == 1 and not self.stream:
            self.frame.response = raw
        self.n_recv += 1
        return raw

    def send(self, payload, **kw):
        if self.stream:
            return self.ws.send(payload, **kw)
        request = unpackb(payload) if isinstance(payload, bytes) else None
        op = op_of(request)
        if op == "stream":
            self.stream = True
            return self.ws.send(payload, **kw)
        self.frame.op = op
        payload = self.tape._stamp(self.frame, request, payload)
        self.frame.request = payload
        try:
            return self.ws.send(payload, **kw)
        except Exception as e:  # the server never saw it: taped, raised again on replay, its k not counted
            self.frame.send_error = (type(e).__module__, type(e).__name__, str(e))
            self.tape._unstamp(self.frame)
            raise

    def close(self, *a, **k):
        try:
            return self.ws.close(*a, **k)
        finally:
            if not self.stream and not self.taped:
                self.taped = True
                self.frame.wall_s = round(time.time() - self.t0, 3)
                self.tape._taped(self.frame)

    def __enter__(self):
        return self

    def __exit__(self, *a):
        self.close()


class _Replaying:
    """A connection served from the tape; goes live (a real connection from here on) under replay-live."""

    def __init__(self, tape: WsTape, uri: str, connect_kwargs: dict):
        self.tape, self.uri, self.kwargs = tape, uri, connect_kwargs
        self.n_recv, self.stream, self.sent, self.done = 0, False, False, False
        self.frame: Frame | None = None
        self.real = None  # the live connection after a switch
        self.op = "metadata"
        self.diffs: list = []
        self.owner, self.call_id = tape.owner, tape.call_id

    def recv(self, timeout=None, **kw):
        if self.real is not None:
            return self.real.recv(timeout=timeout, **kw)
        i = self.tape.index
        if self.n_recv == 0:
            self.n_recv += 1
            frame = self.tape.dir.load(i)
            if frame is not None:  # the planner the run talks to must be the one the tape's frame came from
                conflict = self.tape._pair(frame.server, server_of(self.uri), frame.op)
                if conflict is not None:
                    self.diffs = [conflict]
                    self.tape.mismatches += 1
                    if self.tape.mode == "replay":
                        raise TapeDiverged(conflict[0], i, conflict[1], "metadata")
                    if self.tape.mode == "replay-live":
                        self._switch(i, [conflict])
                        return self.real.recv(timeout=timeout, **kw)
                    log.warning(f"wstape frame {i}: {conflict[1]}")
            if frame is not None and frame.recv_error is not None and frame.recv_error[0] == 0:
                self.frame, self.done = frame, True
                self.tape._served(self, frame)  # the connection ends here, as it did on the tape
                raise _recorded_error(frame.recv_error)
            frame = frame or self.tape.dir.load(len(self.tape.dir) - 1)
            if frame is None or frame.metadata is None:
                raise TapeExhausted(i, "metadata")
            return packb(self.tape._rerooted(frame.metadata))
        if self.frame is None:
            raise TapeExhausted(i, self.op)
        self.n_recv += 1
        if self.frame.recv_error is not None and self.frame.recv_error[0] == 1:
            raise _recorded_error(self.frame.recv_error)
        return self.frame.response

    def send(self, payload, **kw):
        if self.real is not None:
            return self.real.send(payload, **kw)
        if self.stream:
            return None
        request = unpackb(payload) if isinstance(payload, bytes) else None
        self.op = op_of(request)
        if self.op == "stream":
            self.stream = True
            return None
        tape, i = self.tape, self.tape.index
        want = tape.dir.load(i)
        probe = Frame(i, self.op, server=server_of(self.uri))
        payload = tape._stamp(probe, request, payload)  # the stamp is a field like any other
        self.sent = True
        if want is None:
            if tape.mode == "replay-live":
                tape.mismatches += 1
                return self._go_live(payload, probe, i, [("<end of tape>", f"the tape has {i} frames")])
            raise TapeExhausted(i, self.op)
        self.frame = want
        if want.op != self.op:
            self.diffs = [("op", f"{want.op} on the tape, {self.op} in the replay")]
        else:
            self.diffs = diff_fields(want.request_dict(), request)
        forced = tape.live_at is not None and i >= tape.live_at
        if tape.mode == "replay-live" and (self.diffs or forced):
            tape.mismatches += bool(self.diffs)
            return self._go_live(payload, probe, i, self.diffs)
        if self.diffs:
            tape.mismatches += 1
            if tape.mode == "replay":
                path, detail = self.diffs[0]
                raise TapeDiverged(path, i, detail, self.op)
            log.warning(
                f"wstape frame {i} ({self.op}): {len(self.diffs)} field(s) differ from the tape; first {self.diffs[0][0]}: "
                f"{self.diffs[0][1]}"
            )
        if want.send_error is not None:  # the recorded send failed: the server never saw it, so neither does the k
            tape._unstamp(probe)
            self.done = True
            tape._served(self, want)
            raise rebuilt_error(*want.send_error)
        return None

    def _switch(self, i: int, diffs: list) -> None:
        """Real connections from here on, before this connection's request is known (a metadata-stage mismatch):
        the live connection's own recv answers the client, and its send stamps as any live frame does."""
        tape = self.tape
        why = f"frame {i} (metadata) differs at {diffs[0][0]}: {diffs[0][1]}"
        log.warning(f"wstape: switching to live connections: {why}")
        tape.live, tape.switched_at, tape.switch_reason = True, i, why
        tape._log_switch(i, "metadata", diffs)
        ws = tape._live_connect(self.uri, self.kwargs, i)
        self.real = _Recording(tape, self.uri, ws, live=True)
        self.real.frame.owner, self.real.frame.call_id = self.owner, self.call_id

    def _go_live(self, payload, probe: Frame, i: int, diffs: list):
        """Real connections from here on. This request goes out as stamped (its k is already counted), so the
        recording connection sends it raw instead of stamping it a second time."""
        tape = self.tape
        why = (
            f"frame {i} forced live (--wstape-live-at {tape.live_at})"
            if not diffs
            else (f"frame {i} ({self.op}) differs at {diffs[0][0]}: {diffs[0][1]}")
        )
        log.warning(f"wstape: switching to live connections: {why}")
        tape.live, tape.switched_at, tape.switch_reason = True, i, why
        tape._log_switch(i, self.op, diffs)
        ws = tape._live_connect(self.uri, self.kwargs, i)
        self.real = _Recording(tape, self.uri, ws, live=True)
        frame = self.real.frame
        frame.owner, frame.call_id = self.owner, self.call_id
        self.real.recv(timeout=60.0)  # the real server's metadata; the client already has the tape's
        tape._check_live(frame.metadata, i)
        frame.op, frame.request, frame.k, frame.seed = self.op, payload, probe.k, probe.seed
        return ws.send(payload)

    def close(self, *a, **k):
        if self.real is not None:
            return self.real.close(*a, **k)
        if self.done or self.stream or (not self.sent and self.n_recv == 0):
            return None
        self.done = True
        if not self.sent:  # a metadata-only connection (fetch_metadata): the tape's frame here must be one too
            tape, i = self.tape, self.tape.index
            want = tape.dir.load(i)
            self.frame = want
            if want is None:
                self.diffs = self.diffs or [("<end of tape>", f"the tape has {i} frames")]
            elif want.op != "metadata":
                self.diffs = self.diffs or [("op", f"{want.op} on the tape, metadata in the replay")]
            if self.diffs:
                tape.mismatches += 1
                log.warning(f"wstape frame {i} (metadata): {self.diffs[0][1]}")
                tape._served(self, self.frame)
                if tape.mode == "replay":  # strict: the first mismatch stops the instance, here as at a send
                    raise TapeDiverged(self.diffs[0][0], i, self.diffs[0][1], "metadata")
                if tape.mode == "replay-live":  # the next connection is live: the tape no longer describes the run
                    tape.live, tape.switched_at = True, i
                    tape.switch_reason = f"frame {i} (metadata) differs at {self.diffs[0][0]}: {self.diffs[0][1]}"
                    log.warning(f"wstape: switching to live connections: {tape.switch_reason}")
                return None
        self.tape._served(self, self.frame)
        return None

    def __enter__(self):
        return self

    def __exit__(self, *a):
        self.close()


# ---------------------------------------------------------------- the tape


class WsTape:
    """See the module docstring. ``path``: the tape directory (written by record and log, read by the replays);
    ``log_dir``: where a replay's own log goes (``wstape_replay.jsonl``) and the live tail (``wstape_live``)."""

    def __init__(self, mode: str, path, replicate: int = 0, live_at: int | None = None, ledger=None, log_dir=None):
        if mode not in MODES or mode == "off":
            raise ValueError(f"--wstape {mode!r}: one of {MODES[1:]}")
        self.mode, self.replicate, self.live_at = mode, int(replicate), live_at
        self.dir = TapeDir(path)
        self.log_dir = Path(log_dir) if log_dir is not None else self.dir.path.parent
        self.ledger = ledger
        self.index = 0  # frames taped or served so far, across every connection and instance
        self.k: dict = {}  # server -> pipeline requests sent so far (the server's _request_n)
        self.frames: list = []  # the rows written (record/log) or served (replay); every mode
        self.mismatches = 0
        self.live = False
        self.switched_at: int | None = None
        self.switch_reason: str | None = None
        self.stamped: list = []  # (server, k, seed) per stamped legacy request
        self.servers: dict = {}  # replay: the tape's server -> the replay's, paired at first sight (a port may move)
        self.rerooted: dict | None = None  # replay: the served metadata's module root, and the root it was mapped to
        self.live_imports: list = []  # replay-live: the live planner's modules, checked against this checkout
        self._installed = False
        self._lock = threading.Lock()
        self.real_connect = bridge_client.connect
        self._orig_health = bridge_client.TiptopClient.health
        self.replay_log = self.log_dir / "wstape_replay.jsonl"
        self.live_dir = TapeDir(self.log_dir / "wstape_live")
        if mode in REPLAY_MODES:
            if not self.dir.whole:
                raise FileNotFoundError(
                    f"--wstape {mode}: {self.dir.path} holds no recorded frames (a 'log' tape has digests only; "
                    f"replay needs a 'record')"
                )
        elif self.dir.index.exists():
            raise FileExistsError(f"--wstape {mode}: {self.dir.index} exists; a tape is never overwritten")

    # ---------------------------------------------------------------- the patch
    @property
    def owner(self) -> str:
        return self.ledger.current if self.ledger is not None else "unowned"

    @property
    def call_id(self):
        return getattr(self.ledger, "call_id", None)

    def install(self) -> "WsTape":
        if self._installed:
            return self
        self.real_connect = bridge_client.connect
        self._orig_health = bridge_client.TiptopClient.health
        bridge_client.connect = self._connect
        if self.mode in REPLAY_MODES:
            tape, orig = self, self._orig_health

            def health(client, timeout_s: float = 3.0) -> bool:
                return True if not tape.live else orig(client, timeout_s)

            bridge_client.TiptopClient.health = health
        self._installed = True
        log.info(f"wstape {self.mode}: {self.dir.path} ({len(self.dir)} frames on it), replicate {self.replicate}")
        return self

    def uninstall(self) -> None:
        if not self._installed:
            return
        bridge_client.connect = self.real_connect
        bridge_client.TiptopClient.health = self._orig_health
        self._installed = False

    def __enter__(self):
        return self.install()

    def __exit__(self, *a):
        self.uninstall()

    def _connect(self, uri, **kwargs):
        with self._lock:
            if self.mode in REPLAY_MODES and not self.live:
                return _Replaying(self, uri, kwargs)
            ws = self.real_connect(uri, **kwargs)
            return _Recording(self, uri, ws, live=self.live)

    # ---------------------------------------------------------------- the stamp
    def _stamp(self, frame: Frame, request, payload: bytes) -> bytes:
        """The seed stamp on a legacy plan request; k counted per server. Skill requests carry their own seed."""
        server = frame.server
        if frame.op == "plan" and isinstance(request, dict):
            k = self.k.get(server, 0) + 1
            self.k[server] = k
            seed = seed_for(self.replicate, k)
            if request.get("seed") not in (None, seed):
                log.warning(f"wstape: request {k} to {server} carried seed {request['seed']}; stamped {seed}")
            request["seed"] = seed
            frame.k, frame.seed = k, seed
            self.stamped.append((server, k, seed))
            return packb(request)
        return payload

    def _unstamp(self, frame: Frame) -> None:
        """A stamped request whose send failed never reached the server: its k is given back."""
        if frame.op == "plan" and frame.k is not None and self.k.get(frame.server) == frame.k:
            self.k[frame.server] = frame.k - 1
            if self.stamped and self.stamped[-1] == (frame.server, frame.k, frame.seed):
                self.stamped.pop()

    def _count_skill(self, frame: Frame, server: str | None = None) -> None:
        """A skill request that reached the pipeline counts toward the server's k, as the plan requests do.
        ``server``: the connection's own (a replay on another port than the recording's counts under the port it
        talks to, the key _stamp uses); else the frame's."""
        if frame.op != "skill" or frame.send_error is not None:
            return
        req = frame.request_dict() or {}
        ans = frame.response_json() or {}
        refused = not ans.get("ok", False) and ans.get("phase") is None and ans.get("code") in BUILD_REFUSALS
        if req.get("skill") in OWN_SOLVE_SKILLS or refused:
            return
        key = server or frame.server
        frame.k = self.k[key] = self.k.get(key, 0) + 1

    # ---------------------------------------------------------------- replay: which planner, which checkout
    def _pair(self, taped: str, now: str, op: str = ""):
        """The replay talks to the planner the tape's frame came from: each server on the tape is paired with the
        replay's server it is first served on (a replay may run on another port); a later frame of that server on
        another one, or another server on its partner, is a mismatch ("server", detail). None when it pairs."""
        if not taped:
            return None
        paired = self.servers.setdefault(taped, now)
        other = next((t for t, n in self.servers.items() if n == now and t != taped), None)
        if paired != now or other is not None:
            return (
                "server",
                f"{taped} on the tape (paired with {paired}); the replay talks to {now}"
                + (f", paired with {other}" if other else ""),
            )
        return None

    def _rerooted(self, metadata):
        """A replay has no planner: the tape is it, and its metadata names the recording's checkout. Its module
        paths are mapped onto this checkout (the part from ``/tiptop/`` on kept), logged and summarised, so
        run.check_imports judges the replay's own code and a tape replays from another snapshot of the same code."""
        modules = (metadata or {}).get("modules") if isinstance(metadata, dict) else None
        if not modules:
            return metadata
        root = str(ROOT)
        out, moved = {}, None
        for name, path in modules.items():
            if isinstance(path, str) and "/tiptop/" in path and not path.startswith(root + "/"):
                recorded, rest = path.split("/tiptop/", 1)
                out[name] = f"{root}/tiptop/{rest}"
                moved = recorded
            else:
                out[name] = path
        if moved is None:
            return metadata
        if self.rerooted is None:
            self.rerooted = {"from": moved, "to": root}
            log.info(f"wstape: the tape's planner modules name {moved}; served as {root} (a replay runs no planner)")
        return {**metadata, "modules": out}

    def _check_live(self, metadata, i: int) -> None:
        """The live planner a replay-live switched to must run this checkout's tiptop and cutamp (D25), as
        check_imports requires of a planner at connect: logged, summarised, and a TapeDiverged("imports") if not."""
        modules = (metadata or {}).get("modules") or {} if isinstance(metadata, dict) else {}
        files = {name: modules.get(name) for name in ("tiptop", "cutamp")}
        outside = sorted(n for n, p in files.items() if p is None or not Path(p).resolve().is_relative_to(ROOT))
        self.live_imports.append({"frame": i, "modules": files, "ok": not outside})
        log.info("imports (live planner): " + ", ".join(f"{n}={p}" for n, p in files.items()))
        if outside:
            raise TapeDiverged(
                "imports", i, f"the live planner's {', '.join(outside)} not imported from {ROOT}", "metadata"
            )

    def _live_connect(self, uri: str, kwargs: dict, i: int):
        """The real connection a replay-live switches to, retried on OSError as TiptopClient._open retries: the
        server opens its port only once its warm-up is done (tiptop_websocket_server.run), so a fresh planner still
        warming up, or one relaunching after a start-up crash, refuses the connection for a while."""
        last = None
        for attempt in range(LIVE_CONNECT_RETRIES):
            try:
                return self.real_connect(uri, **kwargs)
            except OSError as e:
                last = e
                log.warning(f"wstape: live connect to {uri} failed ({e}); retry {attempt + 1}/{LIVE_CONNECT_RETRIES}")
                time.sleep(LIVE_RETRY_WAIT_S)
        raise last

    # ---------------------------------------------------------------- the records
    def _taped(self, frame: Frame) -> None:
        with self._lock:
            frame.index = self.index
            self.index += 1
            self._count_skill(frame)
            target = self.live_dir if frame.live else self.dir
            row = target.append(frame, whole=self.mode != "log" or frame.live)
            self.frames.append(row)
        log.info(
            f"wstape frame {frame.index} ({frame.op}, {frame.owner}, {frame.server}"
            + (f", k {frame.k}, seed {frame.seed}" if frame.k else "")
            + f"): {'live ' if frame.live else ''}"
            f"{'recorded' if self.mode != 'log' or frame.live else 'logged'}"
        )

    def _served(self, conn: _Replaying, frame: Frame | None) -> None:
        with self._lock:
            i = self.index
            self.index += 1
            row = {
                "i": i,
                "op": conn.op,
                "owner": conn.owner,
                "call_id": conn.call_id,
                "server": server_of(conn.uri),
                "tape_op": None if frame is None else frame.op,
                "matched": not conn.diffs,
                "n_diffs": len(conn.diffs),
                "diffs": [{"path": p, "detail": d} for p, d in conn.diffs],
                "live": False,
            }
            if frame is not None:
                self._count_skill(frame, server_of(conn.uri))
            self.frames.append(row)
            self.log_dir.mkdir(parents=True, exist_ok=True)
            with open(self.replay_log, "a") as f:
                f.write(json.dumps(row, default=_json_default) + "\n")
        log.info(
            f"wstape frame {i} ({conn.op}) served from the tape"
            + ("" if not conn.diffs else f"; {len(conn.diffs)} diff(s), first {conn.diffs[0][0]}")
        )

    def _log_switch(self, i: int, op: str, diffs: list) -> None:
        """The frame a replay-live switched at, in the replay's own log (``live``: true, ``matched``: whether its
        request agreed with the tape, a forced switch). The frame itself is taped live, under wstape_live."""
        row = {
            "i": i,
            "op": op,
            "owner": self.owner,
            "call_id": self.call_id,
            "server": None,
            "tape_op": None,
            "matched": not diffs,
            "n_diffs": len(diffs),
            "diffs": [{"path": p, "detail": d} for p, d in diffs],
            "live": True,
            "switched": True,
        }
        self.log_dir.mkdir(parents=True, exist_ok=True)
        with open(self.replay_log, "a") as f:
            f.write(json.dumps(row, default=_json_default) + "\n")

    def summary(self) -> dict:
        return {
            "mode": self.mode,
            "path": str(self.dir.path),
            "replicate": self.replicate,
            "frames": self.index,
            "on_tape": len(self.dir),
            "k": dict(self.k),
            "stamped": [{"server": s, "k": k, "seed": seed} for s, k, seed in self.stamped],
            "mismatches": self.mismatches,
            "live": self.live,
            "switched_at": self.switched_at,
            "switch_reason": self.switch_reason,
            "servers": dict(self.servers),
            "rerooted": self.rerooted,
            "live_imports": list(self.live_imports),
            "replay_log": str(self.replay_log) if self.mode in REPLAY_MODES else None,
        }
