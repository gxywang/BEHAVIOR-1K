"""Measurement off the simulator's truth for the instruments (WEEK4_PLAN 3.4, W4-A2): the GripperWatch and the state
digest. Privileged reads, so they live here; the bench passes them in and host/ never names this package.

GripperWatch sees every env step's action through the StepLedger's observers and records each open-to-closed
gripper command per arm as a ``close`` event (and each closed-to-open as an ``open``), with the ledger's owner and
call id at that step. A command is closed below 0 (MultiFingerGripperController, smooth mode: +1 open, -1 closed).
While a close is pending, every closed step reads the robot's grasp assist into the row: +1 a weld (IsGraspingState
TRUE), -1 closed on air (FALSE), 0 unknown. It settles -- the reading is final -- once the command has held one
value for ``settle_steps`` steps (the executor's gripper hold: a creeping close ramps the command down and then
holds CLOSE 25 steps before it logs is_grasping, so the settled reading is taken on the same step as the executor's),
or at the last closed step before the command opens again, or when the watch is finished. ``via`` names who issued
the close: ``executor`` for a plan's gripper event (PlanExecutor.set_gripper under execute, the close the executor
logs with is_grasping=), ``executor.start`` for its start_gripper (not logged so), ``sim`` for any other (a closed-fist
drawer pull, a sticky grasp), read off the call stack at the close. Measurement only: nothing it reads steers
anything.

``state_digest(sim, knowledge)`` is what the Runner tape takes before and after every write: the step count, the
env.step count, the teleports, the hand record in order, a hash of the robot's base pose and joint positions, a hash
of every tracked object's pose, and the knowledge source's memory length and hash. Two digests that differ name the
first place two runs stopped being the same episode (scripts/tape_diff.py).
"""

from __future__ import annotations

import hashlib
import sys

import numpy as np

# The executor holds a gripper command for gripper_hold_steps (25) env steps before it reads is_grasping.
SETTLE_STEPS = 25
ARMS = ("left", "right")
EXECUTOR = "b1k.bridge.executor"  # PlanExecutor's module: its set_gripper issues a plan's gripper events


class GripperWatch:
    """See the module docstring. ``events`` rows are counters.py's gripper.jsonl schema: step, arm, event,
    is_grasping, owner, call_id (plus ``settled_at``: the step the final reading was taken on, None while pending, and
    a close's ``via``). An ``open`` row's is_grasping is read after the opening step."""

    def __init__(self, sim, ledger, settle_steps: int = SETTLE_STEPS):
        self.sim, self.ledger, self.settle_steps = sim, ledger, int(settle_steps)
        self.events: list[dict] = []
        self._closed: dict = {}  # arm -> whether the last command seen was closed
        self._pending: dict = {}  # arm -> [row, the command value, steps it has held it, step of the reading]
        ledger.observers.append(self.on_step)

    def on_step(self, action, sim) -> None:
        commands = self._commands(action, sim)
        if commands is None:
            return
        for arm, value in commands.items():
            closed = value < 0
            was = self._closed.get(arm)
            self._closed[arm] = closed
            if closed and was is False:
                row = self._row(arm, "close")
                row["via"] = self._issuer()
                self.events.append(row)
                self._pending[arm] = [row, value, 0, None]
            if closed and arm in self._pending:
                pending = self._pending[arm]
                pending[2] = pending[2] + 1 if value == pending[1] else 1
                pending[1] = value
                pending[0]["is_grasping"] = self._read(arm)  # the latest closed reading; final once settled
                pending[3] = self._now()
                if pending[2] >= self.settle_steps:
                    self._settle(arm)
            elif not closed and was is True:
                self._settle(arm)  # its last closed step's reading: the one before the hand let go
                row = self._row(arm, "open")
                row["is_grasping"] = self._read(arm)
                row["settled_at"] = row["step"]
                self.events.append(row)

    def finish(self) -> list[dict]:
        for arm in list(self._pending):
            self._settle(arm)
        return self.events

    @property
    def closes(self) -> list[dict]:
        return [e for e in self.events if e["event"] == "close"]

    def _now(self) -> int:
        return int(getattr(self.sim, "n_steps", 0))

    def _row(self, arm: str, event: str) -> dict:
        return {
            "step": self._now(),
            "arm": arm,
            "event": event,
            "is_grasping": None,
            "owner": self.ledger.current,
            "call_id": self.ledger.call_id,
            "settled_at": None,
            "via": None,
        }

    @staticmethod
    def _issuer() -> str:
        """Who issued the command this env step carries: PlanExecutor.set_gripper under start_gripper, under
        anything else (execute), or not the executor at all."""
        f, via = sys._getframe(1), "sim"
        while f is not None:
            code, module = f.f_code, f.f_globals.get("__name__")
            if module == EXECUTOR and code.co_name == "set_gripper":
                via = "executor"
            elif module == EXECUTOR and code.co_name == "start_gripper" and via == "executor":
                return "executor.start"
            elif via == "executor" and module == EXECUTOR and code.co_name == "execute":
                return via
            f = f.f_back
        return via

    def _settle(self, arm: str) -> None:
        """The pending close's latest reading is final; ``settled_at`` is the step it was taken on."""
        pending = self._pending.pop(arm, None)
        if pending is not None:
            pending[0]["settled_at"] = pending[3]

    def _read(self, arm: str) -> int:
        """The assist's word on ``arm``: IsGraspingState's own values, TRUE 1, UNKNOWN 0, FALSE -1 (0 when it
        cannot be read)."""
        try:
            return int(self.sim.robot.is_grasping(arm))
        except Exception:  # noqa: BLE001 - a robot without the assist: unknown
            return 0

    @staticmethod
    def _commands(action, sim) -> dict | None:
        """{arm: command} from the action of one env step: the R1Pro's gripper groups by the robot's action index."""
        robot = getattr(sim, "robot", None)
        idx = getattr(robot, "controller_action_idx", None)
        name = getattr(robot, "name", None)
        if idx is None or not isinstance(action, dict) or name not in action:
            return None
        a = action[name]
        out = {}
        for arm in ARMS:
            group = idx.get(f"gripper_{arm}")
            if group is None:
                continue
            values = np.asarray(a[group]).reshape(-1)
            if values.size:
                out[arm] = float(values[0])
        return out


# ---------------------------------------------------------------- the state digest


def _bytes(x) -> bytes:
    if hasattr(x, "detach"):
        x = x.detach().cpu().numpy()
    return np.ascontiguousarray(np.asarray(x, dtype=np.float64)).tobytes()


def _feed(h, x) -> None:
    """Hash a value structurally: containers by shape, arrays by dtype and bytes, scalars by repr."""
    if isinstance(x, dict):
        h.update(b"{")
        for k in sorted(x, key=repr):
            _feed(h, k)
            h.update(b":")
            _feed(h, x[k])
        h.update(b"}")
    elif isinstance(x, (list, tuple, set, frozenset)):
        h.update(b"[")
        for v in sorted(x, key=repr) if isinstance(x, (set, frozenset)) else x:
            _feed(h, v)
            h.update(b",")
        h.update(b"]")
    elif hasattr(x, "detach") or isinstance(x, (np.ndarray, np.generic)):
        arr = np.asarray(x.detach().cpu().numpy() if hasattr(x, "detach") else x)
        h.update(f"<{arr.dtype}{arr.shape}>".encode())
        h.update(np.ascontiguousarray(arr).tobytes())
    elif isinstance(x, bytes):
        h.update(x)
    else:
        h.update(repr(x).encode())


def _hex(h) -> str:
    return h.hexdigest()[:16]


def state_digest(sim, knowledge=None) -> dict:
    """See the module docstring. Every part is taken on its own, so a fake without a robot still digests."""
    d = {
        "n_steps": int(getattr(sim, "n_steps", 0)),
        "env_steps": getattr(getattr(sim, "step_env", None), "calls", None),
        "teleports": int(getattr(sim, "teleports", 0)),
        "held": tuple((str(k), str(v)) for k, v in dict(getattr(sim, "held_objects", {}) or {}).items()),
    }
    try:
        pos, quat = sim.robot.get_position_orientation()
        h = hashlib.sha256(_bytes(pos) + _bytes(quat) + _bytes(sim.robot.get_joint_positions()))
        d["robot"] = _hex(h)
    except Exception:  # noqa: BLE001
        d["robot"] = None
    try:
        objects = getattr(sim, "objects", None)
        if objects is None:
            d["objects"] = None
        else:
            h = hashlib.sha256()
            for name in sorted(objects):
                pos, quat = objects[name].get_position_orientation()
                h.update(name.encode() + _bytes(pos) + _bytes(quat))
            d["objects"] = _hex(h)
    except Exception:  # noqa: BLE001
        d["objects"] = None
    # the tracked objects' joints: a drawer that slid open moves no root pose (store_honey's cabinet, j_link_4),
    # so an open or a close is visible here alone
    try:
        objects = getattr(sim, "objects", None)
        if objects is None:
            d["joints"] = None
        else:
            h, n = hashlib.sha256(), 0
            for name in sorted(objects):
                obj = objects[name]
                if int(getattr(obj, "n_dof", 0) or 0) > 0:
                    h.update(name.encode() + _bytes(obj.get_joint_positions()))
                    n += 1
            d["joints"] = (n, _hex(h))
    except Exception:  # noqa: BLE001
        d["joints"] = None
    memory = getattr(knowledge, "seen", None) if knowledge is not None else None
    if memory is None and knowledge is not None:
        memory = getattr(getattr(knowledge, "sim", None), "seen_boxes", None)
    if memory is None:
        d["memory"] = None
    else:
        h = hashlib.sha256()
        _feed(h, memory)
        d["memory"] = (len(memory), _hex(h))
    return d
