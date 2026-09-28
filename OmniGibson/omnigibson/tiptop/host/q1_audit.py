"""The differential audit of the pseudo planner's Q1 shim in the sim (WEEK4_PLAN §3.3 q1_audit.py, §5.4 G3; W4-E).

DualEpisode sits between the Runner (or its TapeRecorder) and the shim: after every pure read the shim answers, it
asks the real Episode the same question at the same instant and records whether the two agree: is_shut, holding,
held_names, support_of, distance, edge_gap, switched_on, goal_already_holds, stance_key, is_floor, floor, has_arm,
and floor_failed_at after each put_down. after_transition and fixture_for are stateful (the knowledge source's
appeared() list, sim.track) and are never compared. The Runner sees exactly the shim's answers and exceptions; a
disagreement is a row in ``mismatches``, never a substitute.

PurityAudit wraps the connector the planner sees: around every 0-step op (b1k.runtime.audit.zero_step: task,
skills, clock, stance_request, propose_stances, check_stances, check, goal_status, holds, abort and every
world.<member>) and around every comparison read DualEpisode makes on the real Episode, it takes the host's state
digest before and after and requires it unchanged (world.appeared and world.fixture_for may change its ``objects``
alone: they track what they name, STATEFUL_OPS), and takes the StepLedger's write-calls total before and after
and requires it unchanged; ``check(ledger)`` also requires the unowned write-calls to be 0. The digest is a callable
the bench passes in (the oracle package's state_digest bound to the sim and the knowledge source): nothing here
reads the simulator, and host/ never names that package.
"""

from __future__ import annotations

import math
from collections import Counter
from contextlib import contextmanager, nullcontext
from typing import Any, Callable, Optional

from b1k.runtime.audit import ConnectorAudit, zero_step


# The name the episode host imports. b1k/runtime/audit.py once defined its verdict under the op's name ``check``, so a
# planner's check(call) through it answered the verdict tuple (found by the episode host's unit twin, W4-E); since
# the week-4 fix pass the b1k class has the op ``check(call)`` and the verdict ``verdict(n0)`` itself.
AuditedConnector = ConnectorAudit

# The reads compared (WEEK4_PLAN §3.3): the Runner's pure questions. sim.n_steps / sim.max_steps are compared after
# every write instead (the ClockView against the Episode's own clock, row 16).
PURE_READS = ("is_shut", "holding", "held_names", "support_of", "distance", "edge_gap", "switched_on",
              "goal_already_holds", "stance_key", "is_floor", "has_arm")
ATTR_READS = ("floor",)
SKIPPED = ("after_transition", "fixture_for")  # stateful: never asked twice
# Their world ops change the digest's ``objects`` (what the sim tracks) by design: appeared() tracks a cut's halves
# (knowledge.appeared), fixture_for tracks the fixture it names (sim.track), exactly as Episode.after_transition and
# Episode.fixture_for do. PurityAudit allows that key alone to change across them; any other change is a violation.
STATEFUL_OPS = ("world.appeared", "world.fixture_for")
WRITES = ("pick", "achieve", "put_down", "open_up", "release", "pour", "dwell", "stand_for", "walk_to_floor")
_MISSING = object()


def same(a: Any, b: Any) -> bool:
    """Equality for the answers compared: floats by value (inf == inf, nan == nan), sequences element-wise, sets as
    sets, everything else by ==."""
    if isinstance(a, float) or isinstance(b, float):
        try:
            fa, fb = float(a), float(b)
        except (TypeError, ValueError):
            return False
        if math.isnan(fa) and math.isnan(fb):
            return True
        return fa == fb or math.isclose(fa, fb, rel_tol=1e-12, abs_tol=0.0)
    if isinstance(a, (set, frozenset)) or isinstance(b, (set, frozenset)):
        try:
            return set(a) == set(b)
        except TypeError:
            return False
    if isinstance(a, (list, tuple)) and isinstance(b, (list, tuple)):
        return len(a) == len(b) and all(same(x, y) for x, y in zip(a, b))
    return a == b


def _shown(x: Any) -> Any:
    """A value as the audit rows carry it (JSON-plain): sets sorted, tuples as lists, others by repr when needed."""
    if isinstance(x, (set, frozenset)):
        return sorted((_shown(v) for v in x), key=repr)
    if isinstance(x, (list, tuple)):
        return [_shown(v) for v in x]
    if isinstance(x, dict):
        return {str(k): _shown(v) for k, v in x.items()}
    if x is None or isinstance(x, (bool, int, float, str)):
        return x
    return repr(x)


def _outcome(fn: Callable[[], Any]) -> tuple:
    """(value, exception): what a read answered, or what it raised (Exception only; a BaseException such as
    EpisodeOver or TapeDiverged propagates)."""
    try:
        return fn(), None
    except Exception as e:  # noqa: BLE001 - the exception is the answer compared (KeyError from distance)
        return None, e


class DualEpisode:
    """See the module docstring. ``shim``: the EpisodeOverConnector the Runner would hold; ``episode``: the bench's
    Episode; ``purity``: the PurityAudit whose ``reading`` brackets every comparison read (None: unbracketed)."""

    def __init__(self, shim, episode, purity: Optional["PurityAudit"] = None):
        object.__setattr__(self, "_shim", shim)
        object.__setattr__(self, "_ep", episode)
        object.__setattr__(self, "_purity", purity)
        object.__setattr__(self, "_rows", [])
        object.__setattr__(self, "_compared", Counter())
        object.__setattr__(self, "_mismatched", Counter())
        object.__setattr__(self, "_writes", 0)

    # -- the record ------------------------------------------------------------------------------------------------
    @property
    def rows(self) -> list:
        return self._rows

    @property
    def mismatches(self) -> list:
        return [r for r in self._rows if not r["equal"]]

    def summary(self) -> dict:
        return {"compared": dict(self._compared), "mismatched": dict(self._mismatched),
                "mismatches": len(self.mismatches), "writes": self._writes}

    def _record(self, member: str, args: tuple, kwargs: dict, shim_out: tuple, ep_out: tuple) -> None:
        (sv, se), (ev, ee) = shim_out, ep_out
        if se is not None or ee is not None:  # the type and the message, as the Runner tape compares them
            equal = se is not None and ee is not None and type(se) is type(ee) and str(se) == str(ee)
        else:
            equal = same(sv, ev)
        self._compared[member] += 1
        if not equal:
            self._mismatched[member] += 1
        self._rows.append({
            "kind": "dual", "member": member, "args": _shown(args), "kwargs": _shown(kwargs), "equal": bool(equal),
            "shim": _shown(sv) if se is None else f"raised {type(se).__name__}: {se}",
            "episode": _shown(ev) if ee is None else f"raised {type(ee).__name__}: {ee}",
            "step": self._step(),
        })

    def _step(self):
        sim = getattr(self._ep, "sim", None)
        return None if sim is None else int(getattr(sim, "n_steps", 0) or 0)

    def _bracket(self, label: str):
        p = self._purity
        return p.reading(label) if p is not None else nullcontext()

    def _episode_answer(self, member: str, args: tuple, kwargs: dict) -> tuple:
        with self._bracket(f"dual.{member}"):
            return _outcome(lambda: getattr(self._ep, member)(*args, **kwargs))

    # -- the Episode surface: the shim's, with the reads compared --------------------------------------------------
    def __getattr__(self, name: str):
        if name.startswith("_"):
            raise AttributeError(name)
        shim = object.__getattribute__(self, "_shim")
        value = getattr(shim, name)  # an absent member raises here, as the Runner's probe expects
        if name in ATTR_READS:
            ep_value = _outcome(lambda: getattr(self._ep, name))
            self._record(name, (), {}, (value, None), ep_value)
            return value
        if name == "floor_failed_at":  # read by the Runner after a failed floor put-down (row 15)
            self._record(name, (), {}, (value, None), (self._ep.__dict__.get(name, _MISSING), None))
            return value
        if not callable(value):
            return value
        if name in PURE_READS:
            return self._compared_read(name, value)
        if name in WRITES:
            return self._write(name, value)
        return value

    def __setattr__(self, name, value):
        raise AttributeError(f"the Runner does not write to its Episode; refusing to set {name}")

    def _compared_read(self, name: str, fn: Callable):
        def read(*args, **kwargs):
            shim_out = _outcome(lambda: fn(*args, **kwargs))
            self._record(name, args, kwargs, shim_out, self._episode_answer(name, args, kwargs))
            if shim_out[1] is not None:
                raise shim_out[1]
            return shim_out[0]

        read.__name__ = name
        return read

    def _write(self, name: str, fn: Callable):
        """A write passes through untouched; afterwards the clocks are compared (row 16), and after a put_down the
        shim's floor_failed_at against the Episode's (row 15)."""

        def write(*args, **kwargs):
            object.__setattr__(self, "_writes", self._writes + 1)
            try:
                return fn(*args, **kwargs)
            finally:
                self._after_write(name)

        write.__name__ = name
        return write

    def _after_write(self, name: str) -> None:
        try:
            self._compare_clock()
            if name == "put_down":
                shim_failed = getattr(self._shim, "floor_failed_at", _MISSING)
                ep_failed = self._ep.__dict__.get("floor_failed_at", _MISSING)
                self._record("floor_failed_at", (), {"after": name}, (shim_failed, None), (ep_failed, None))
        except Exception as e:  # noqa: BLE001 - the comparison must never change what the Runner sees
            self._rows.append({"kind": "dual", "member": f"after.{name}", "args": [], "kwargs": {}, "equal": False,
                               "shim": None, "episode": f"comparison raised {type(e).__name__}: {e}",
                               "step": self._step()})
            self._mismatched[f"after.{name}"] += 1

    def _compare_clock(self) -> None:
        ep_sim = getattr(self._ep, "sim", None)
        if ep_sim is None:
            return
        view = getattr(self._shim, "sim", None)
        for field in ("n_steps", "max_steps"):
            shim_out = _outcome(lambda: getattr(view, field))
            with self._bracket(f"dual.sim.{field}"):
                ep_out = _outcome(lambda: getattr(ep_sim, field))
            self._record(f"sim.{field}", (), {}, shim_out, ep_out)


class PurityAudit:
    """A Connector proxy (see the module docstring): ``digest()`` and ``writes()`` are taken before and after every
    0-step op and every bracketed read; a change is a violation. ``rows``: one per bracket; ``violations``: the
    labels and what changed."""

    def __init__(self, conn: Any, digest: Callable[[], dict], writes: Callable[[], int]):
        self.conn, self.digest, self.writes = conn, digest, writes
        self.rows: list = []
        self.violations: list = []
        self.brackets = 0

    # -- the bracket ------------------------------------------------------------------------------------------------
    @contextmanager
    def reading(self, label: str):
        d0, w0 = self.digest(), self.writes()
        self.brackets += 1
        try:
            yield
        finally:
            d1, w1 = self.digest(), self.writes()
            changed = sorted(k for k in set(d0) | set(d1) if d0.get(k) != d1.get(k))
            row = {"kind": "purity", "op": label, "digest_changed": changed, "writes": int(w1 - w0)}
            if label in STATEFUL_OPS and set(changed) <= {"objects"}:  # the tracked set grew, as legacy's read does
                row["stateful"], changed = bool(changed), []
            self.rows.append(row)
            if changed or w1 != w0:
                self.violations.append(
                    f"{label}: " + (f"digest changed at {changed} ({[(k, d0.get(k), d1.get(k)) for k in changed]})"
                                    if changed else "") + (f"; {w1 - w0} write-call(s)" if w1 != w0 else "")
                )

    def _op(self, op: str, fn: Callable[..., Any], *a, **k) -> Any:
        if not zero_step(op):
            return fn(*a, **k)
        with self.reading(op):
            return fn(*a, **k)

    # -- the Connector, op by op ------------------------------------------------------------------------------------
    def task(self): return self._op("task", self.conn.task)
    def skills(self): return self._op("skills", self.conn.skills)
    def world(self): return _PureWorld(self, self.conn.world())
    def clock(self): return self._op("clock", self.conn.clock)
    def observe(self, req): return self._op("observe", self.conn.observe, req)
    def stance_request(self, call): return self._op("stance_request", self.conn.stance_request, call)
    def propose_stances(self, req, k=8): return self._op("propose_stances", self.conn.propose_stances, req, k)
    def check_stances(self, call, stances): return self._op("check_stances", self.conn.check_stances, call, stances)
    def go_to(self, stance): return self._op("go_to", self.conn.go_to, stance)
    def check(self, call): return self._op("check", self.conn.check, call)
    def start(self, call): return self._op("start", self.conn.start, call)
    def wait(self, *handles): return self._op("wait", self.conn.wait, *handles)
    def run(self, call): return self._op("run", self.conn.run, call)
    def abort(self, handle): return self._op("abort", self.conn.abort, handle)
    def dwell(self, steps, until=()): return self._op("dwell", self.conn.dwell, steps, until)
    def goal_status(self): return self._op("goal_status", self.conn.goal_status)
    def holds(self, fact): return self._op("holds", self.conn.holds, fact)

    # -- the verdict ------------------------------------------------------------------------------------------------
    def summary(self, ledger=None) -> dict:
        unowned = 0
        if ledger is not None:
            try:
                unowned = int(ledger.totals().get("unowned_writes", 0))
            except Exception:  # noqa: BLE001 - a ledger without totals: unknown, reported as such
                unowned = -1
        return {"brackets": self.brackets, "digest_changes": sum(1 for r in self.rows if r["digest_changed"]),
                "writes_in_reads": sum(r["writes"] for r in self.rows), "unowned_writes": unowned,
                "violations": len(self.violations) + (1 if unowned else 0)}

    def verdict(self, ledger=None) -> tuple:
        """(summary, violations): the bracket violations, plus the unowned write-calls when a ledger is given."""
        out = list(self.violations)
        s = self.summary(ledger)
        if s["unowned_writes"]:
            out.append(f"unowned write-calls: {s['unowned_writes']}")
        return s, out


class _PureWorld:
    def __init__(self, audit: PurityAudit, world: Any):
        self._audit, self._world = audit, world

    def __getattr__(self, name: str):
        if name.startswith("_"):
            raise AttributeError(name)
        fn = getattr(self._world, name)
        if not callable(fn):
            return fn

        def member(*a, **k):
            return self._audit._op(f"world.{name}", fn, *a, **k)

        return member
