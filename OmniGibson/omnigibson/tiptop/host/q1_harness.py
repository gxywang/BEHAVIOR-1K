"""The offline equivalence harness of the pseudo planner's Q1 shim (WEEK4_PLAN §3.6, §5.2 E1, §5.3 G2; SPEC §7 Q1
gate item 1): the Runner driven through the REAL connector stack with no simulator, over the strategies suite's fakes
(E1) or over a recorded Runner tape (G2).

``build_episode_connector(ep, members)`` wires what the episode host wires (W4-E), with the null Env and Adapter of
b1k/tests/fakes.py in the simulator's seat: the production Runtime and DirectConnector, EpisodeRegistry over
EPISODE_SPECS with the PARITY profile, the LegacyChannel, EpisodeLegacyBackend and EpisodeNavigator (W4-C) over an
ExecLog of the Episode, a teleport navigator that refuses, a tiptop backend and a planner client that raise when
touched, and ``skillrun.PASSTHROUGH = (EpisodeOver,)`` until ``close()``. The providers read the Episode itself:

  EpisodeWorld   is_open False iff ep.is_shut, else None; holding ("left",) iff ep.holding; held("left") the refs of
                 ep.held_names(), held("right") (); base_pose from ep.stance_key() (10 cm, 15 degrees), None when
                 absent; support_of, distance, edge_gap, switched_on and fixture_for forward to the Episode member of
                 the same name, with the production provider's conversions (OracleWorld: distance's KeyError, an
                 object never perceived, is None); appeared() to after_transition (§4 row 11); objects() the scope's
                 refs; box() unknown (no Runner member reads a box).
  EpisodeScorerOver  holds(fact) = ep.goal_already_holds(pred, *args); an exception or a missing member is None.
  task()         a TaskInfo built on every call: the floor re-read from ep.floor, the arms asked of ep.has_arm when
                 the Episode has it (else ("left",)), the scope the Runner was built with.
  clock()        ep.sim's n_steps and max_steps, read on every access; Clock(0, None, 0) when there is no sim.

Every read the providers make goes through one ``Src``, which knows the Runner's read in flight (``RunnerSide``, the
marker the harness puts between the TapeRecorder and the shim). A read the Runner's own question needs is passed to
the Episode as asked; a read the connector's code makes on its own (the backend's post-call judge, the registry's
advisory precheck and resolve_arm, the shim's dwell clip and its constructor, the backend's step count) is logged in
``Src.extras``. Over a live fake an extra read is answered by the fake (its reads are pure); over a TapeEpisode it is
answered from the tape without consuming it (``peek``), so the Runner's own reads stay served in order, and every
read left unanswerable on the TapeEpisode is one a Runner member asked.

``replay(tape)`` is G2 for one tape: the Runner rebuilt from the header's construction inputs (``construction()``,
which q1_pytest stamps into every header, or the bench's ``options``/``scope``; ``strategy_for`` when the recorded
spec is the one it would build) is run
directly against a TapeEpisode (E0), then as Runner -> TapeRecorder(shim) -> this stack -> ExecLog -> TapeEpisode,
and the second recording must equal the tape.

Offline only: nothing here imports the simulator, and no provider here reads it.
"""

from __future__ import annotations

import copy
import dataclasses
import functools
import hashlib
import inspect
import json
import math
import re
import sys
from collections import Counter
from contextlib import contextmanager
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Optional

import b1k.runtime.skillrun as skillrun
from b1k.connector.goals import Clock, GoalPanel, ScorerChecker, TaskInfo
from b1k.connector.observe import StepObs
from b1k.connector.skills import NavResult
from b1k.connector.types import Belief, ObjRef, Pose2, Provided
from b1k.connector.world import ProvenancePolicy
from b1k.planner.pseudo import tape as tp
from b1k.planner.pseudo.names import bddl_category
from b1k.planner.pseudo.shim import ALL, counts
from b1k.runtime.core import Runtime
from b1k.runtime.direct import DirectConnector
from b1k.runtime.skillrun import Services
from b1k.skills.scripted import ScriptedBackend
from b1k.tests.fakes import Adapter, Env, Observer
from omnigibson.tiptop.host.legacy_channel import LegacyChannel
from omnigibson.tiptop.host.legacy_episode import EPISODE_SPECS, EpisodeLegacyBackend, EpisodeNavigator, EpisodeRegistry
from omnigibson.tiptop.host.routing_profiles import PARITY

WRITES = tp.WRITES  # the Episode calls that move the robot or the clock: what the ExecLog records
NO_CROSSING = ("stance_key",)  # extras peek never answers from an earlier segment: a write may have moved the base
DIAGNOSTIC = ("step", "digest")  # the host's probe fields on a write (bench.py's TapeRecorder): never the Runner's
SRC = "oracle"  # the providers' tag: the Episode's own answers, as the episode host's providers tag theirs


class OfflineOnly(AssertionError):
    """The harness was asked for something only the simulator has (a native backend, the planner server, a
    teleport): under PARITY no call may get there."""


# ---------------------------------------------------------------------------------------- the Runner's read in flight
class Origin:
    """``now``: (member, args, kwargs) of the Runner read the shim is serving, None outside one; ``served``: whether
    a read that IS that question has gone to the Episode already; ``during``: a label for the logs."""

    def __init__(self):
        self.now: Optional[tuple] = None
        self.served = False
        self.during = "init"

    @contextmanager
    def reading(self, member: str, args: tuple, kwargs: dict):
        saved = (self.now, self.served, self.during)
        self.now, self.served, self.during = (member, tuple(args), dict(kwargs)), False, member
        try:
            yield
        finally:
            self.now, self.served, self.during = saved

    @contextmanager
    def writing(self, member: str):
        saved = (self.now, self.served, self.during)
        self.now, self.served, self.during = None, False, member
        try:
            yield
        finally:
            self.now, self.served, self.during = saved


class RunnerSide:
    """What the TapeRecorder holds in place of the shim: the same members, the same answers, the same exceptions
    (hasattr/getattr answer exactly as on the shim), and each Runner read marked on the Origin while the shim serves
    it. It adds nothing the Runner can see."""

    __slots__ = ("_shim", "_origin")

    def __init__(self, shim, origin: Origin):
        object.__setattr__(self, "_shim", shim)
        object.__setattr__(self, "_origin", origin)

    def __getattr__(self, name):
        if name.startswith("_"):
            raise AttributeError(name)
        origin = self._origin
        if name == "sim":
            return _SimSide(getattr(self._shim, "sim"), origin)
        with origin.writing(name) if name in WRITES else origin.reading(name, (), {}):
            value = getattr(self._shim, name)  # a property (floor, floor_failed_at) reads here
        if not callable(value):
            return value
        if name in WRITES:

            @functools.wraps(value)
            def write(*args, **kwargs):
                with origin.writing(name):
                    return value(*args, **kwargs)

            return write

        @functools.wraps(value)
        def read(*args, **kwargs):
            with origin.reading(name, args, kwargs):
                return value(*args, **kwargs)

        return read

    def __setattr__(self, name, value):
        raise AttributeError(f"the Runner does not write to its Episode; refusing to set {name}")


class _SimSide:
    __slots__ = ("_clock", "_origin")

    def __init__(self, clock, origin: Origin):
        object.__setattr__(self, "_clock", clock)
        object.__setattr__(self, "_origin", origin)

    def __getattr__(self, name):
        with self._origin.reading(f"sim.{name}", (), {}):
            return getattr(self._clock, name)

    def __setattr__(self, name, value):
        raise AttributeError("the clock view is read-only")


# ------------------------------------------------------------------------------------------------ the Episode's reads
def _key(member: str, args=(), kwargs=None) -> str:
    return tp.dumps([member, tuple(args), dict(kwargs or {})])


class Src:
    """The one road from the harness's providers to the Episode (a live fake, or a TapeEpisode when ``tape`` is
    given). ``members``: the Episode members the harness mirrors (presence over a tape). See the module docstring
    for which reads reach the Episode and which are the connector's own (``extras``); ``misreads`` are reads made
    while serving a Runner question that are not that question (over a tape they go to the TapeEpisode all the same,
    where they consume or are unanswerable)."""

    def __init__(self, ep, members=ALL, tape: Optional[tp.Tape] = None):
        self.ep, self.members, self.tape = ep, frozenset(members), tape
        self.origin = Origin()
        self.extras: list = []
        self.misreads: list = []
        self.stale: list = []  # extras over a tape that had no answer in their own segment (peek, NO_CROSSING)
        self._index = _segments(tape) if tape is not None else None
        self._head = tape.header if tape is not None else {}

    def has(self, member: str) -> bool:
        if self.tape is not None:
            return member in self.members
        return hasattr(self.ep, member)

    def read(self, member: str, *args, **kwargs):
        return self._get(member, args, kwargs, lambda: getattr(self.ep, member)(*args, **kwargs), False)

    def attr(self, member: str, plumbing: bool = False):
        """``floor`` or ``sim.<name>``. ``plumbing``: a read made on every task() whoever asks (the floor), so it
        reaches the Episode only when it is the Runner's own question."""

        def live():
            if member.startswith("sim."):
                return getattr(self.ep.sim, member[4:])
            return getattr(self.ep, member)

        return self._get(member, (), {}, live, plumbing)

    def _get(self, member, args, kwargs, live, plumbing):
        o = self.origin
        key = (member, tuple(args), dict(kwargs))
        if o.now is not None and (not plumbing or (o.now == key and not o.served)):
            if o.now == key:
                o.served = True
            else:
                self.misreads.append({"member": member, "args": key[1], "kwargs": key[2], "asked": o.now})
            return live()
        self.extras.append({"member": member, "args": key[1], "kwargs": key[2], "during": o.during})
        return live() if self.tape is None else self.peek(member, args, kwargs)

    def peek(self, member: str, args=(), kwargs=None):
        """The tape's answer to a read as of the current segment, consuming nothing: the segment's first record of
        that read, else (sim.n_steps) the last served write's step after it, else the latest earlier record, else
        the header's floor / max_steps, else None. A recorded exception answers None. ``stance_key`` never falls
        back across a write (any write may re-stance: a failed floor put_down's achieve did): with no record in
        its own segment it answers None, listed in ``stale`` so a divergence it causes is named, not silent."""
        seg, k = self.ep.segment, _key(member, args, kwargs)
        recs = self._index[seg].get(k) if seg < len(self._index) else None
        if recs:
            return _ret(recs[0])
        if member == "sim.n_steps":
            for w in reversed(self.tape.writes[:seg]):
                step = w.get("step")
                if step and step[1] is not None:
                    return step[1]
        if member in NO_CROSSING:
            self.stale.append({"member": member, "args": tuple(args), "segment": seg, "during": self.origin.during})
            return None
        for s in range(min(seg, len(self._index)) - 1, -1, -1):
            recs = self._index[s].get(k)
            if recs:
                return _ret(recs[-1])
        return {"floor": self._head.get("floor"), "sim.max_steps": self._head.get("max_steps"),
                "sim.n_steps": 0}.get(member)


def _segments(tape: tp.Tape) -> list:
    out = [{}]
    for r in tape.records:
        if r["kind"] == "write":
            out.append({})
        elif r["kind"] in ("read", "attr"):
            out[-1].setdefault(_key(r["member"], r.get("args", ()), r.get("kwargs", {})), []).append(r)
    return out


def _ret(r: dict):
    return None if "exc" in r else _snap(r.get("ret"))


def ref(name) -> ObjRef:
    """The ref the shim falls back to for a name it has not seen: the same equality on both sides."""
    return ObjRef(name, bddl_category(name), False)


class EpisodeWorld:
    """The WorldView over the Episode (module docstring). ``updates``: every WorldUpdate a skill result applied."""

    def __init__(self, src: Src, scope=()):
        self.src, self.scope = src, tuple(scope)
        self.updates: list = []

    def _b(self, v) -> Belief:
        return Belief(v, SRC, 0)

    def objects(self) -> list:
        return [ref(n) for n in self.scope]

    def held(self, arm) -> Belief:
        if arm != "left":
            return self._b(())
        return self._b(tuple(ref(n) for n in (self.src.read("held_names") or ())))

    def holding(self, o) -> Belief:
        return self._b(("left",) if self.src.read("holding", o.id) else ())

    def is_open(self, o, joint=None) -> Belief:
        shut = self.src.read("is_shut", o.id) if joint is None else self.src.read("is_shut", o.id, joint=joint)
        return self._b(False if shut else None)

    def switched_on(self, o) -> Belief:
        return self._b(self.src.read("switched_on", o.id))

    def support_of(self, o) -> Belief:
        v = self.src.read("support_of", o.id)
        return self._b(None if v is None else ref(v))

    def distance(self, a, b) -> Belief:
        """OracleWorld.distance's rule: the Episode's KeyError (never perceived) is an unknown distance, None; the
        shim turns that back into the Episode's KeyError (shim.distance)."""
        try:
            return self._b(self.src.read("distance", a.id, b.id))
        except KeyError:
            return self._b(None)

    def box(self, o) -> Belief:
        return self._b(None)

    def edge_gap(self, item, support) -> Belief:
        return self._b(self.src.read("edge_gap", item.id, None if support is None else support.id))

    def fixture_for(self, ability, near=None) -> Belief:
        v = self.src.read("fixture_for", ability, near=None if near is None else near.id)
        return self._b(ref(v) if v else None)

    def appeared(self) -> list:
        return [ref(n) for n in (self.src.read("after_transition") or ())]

    def base_pose(self) -> Belief:
        k = self.src.read("stance_key") if self.src.has("stance_key") else None
        if k is None:
            return self._b(None)
        return self._b(Pose2(k[0] * 0.1, k[1] * 0.1, k[2] * (math.pi / 12)))

    def tick(self, obs) -> None:
        pass

    def apply(self, u) -> None:
        self.updates.append(u)


class EpisodeScorerOver:
    """The scorer the ``scorer`` GoalChecker asks: the Episode's own goal_already_holds."""

    def __init__(self, src: Src):
        self.src = src

    def holds(self, fact) -> Optional[bool]:
        try:
            return self.src.read("goal_already_holds", fact.pred, *fact.args)
        except Exception:  # noqa: BLE001 - a missing member, or a predicate the Episode cannot judge: unknown
            return None


class _Arms:
    """TaskInfo.arms as the shim reads it (``arm in task().arms``): one ep.has_arm(arm) per question."""

    def __init__(self, src: Src):
        self.src = src

    def __contains__(self, arm) -> bool:
        return bool(self.src.read("has_arm", arm)) if self.src.has("has_arm") else arm == "left"

    def __iter__(self):
        return iter(a for a in ("left", "right") if a in self)


class LiveClock:
    """Clock over ep.sim, read on every access (the shim's ClockView reads max_steps and step)."""

    shadow_steps = 0

    def __init__(self, src: Src):
        self._src = src

    @property
    def step(self):
        return self._src.attr("sim.n_steps")

    @property
    def max_steps(self):
        return self._src.attr("sim.max_steps")

    def left(self):
        m = self.max_steps
        return None if m is None else m - self.step


# ------------------------------------------------------------------------------------------------- executed calls
class ExecLog:
    """The Episode as the backend and the navigator see it: every attribute passes through, and every executed
    write (tape.WRITES) is recorded with its full args and kwargs and what it returned or raised (``executed(log)``).
    The signature of a write is the Episode's (inspect.signature follows __wrapped__), so the rebuild check binds as
    it would."""

    def __init__(self, ep):
        object.__setattr__(self, "_ep", ep)
        object.__setattr__(self, "_executed", [])  # never an Episode-named attribute: the backend reads ep.records

    def __getattr__(self, name):
        if name.startswith("_"):
            raise AttributeError(name)
        value = getattr(self._ep, name)
        if name not in WRITES or not callable(value):
            return value
        records = self._executed

        @functools.wraps(value)
        def call(*args, **kwargs):
            rec = {"member": name, "args": encode(args), "kwargs": encode(kwargs)}
            try:
                ret = value(*args, **kwargs)
            except BaseException as e:
                rec["exc"] = [type(e).__name__, str(e)]
                records.append(rec)
                raise
            rec["ret"] = encode(ret)
            records.append(rec)
            return ret

        return call

    def __setattr__(self, name, value):
        raise AttributeError(f"the execution log does not write to the Episode; refusing to set {name}")


def executed(log: ExecLog) -> list:
    """The executed Episode calls, in order: {member, args, kwargs, ret | exc}, encoded."""
    return object.__getattribute__(log, "_executed")


@functools.lru_cache(maxsize=None)
def episode_signatures() -> dict:
    """bench.Episode's write signatures without self: what a TapeEpisode's writes bind against in the rebuild check."""
    from omnigibson.tiptop.bench import Episode

    out = {}
    for m in WRITES:
        sig = inspect.signature(getattr(Episode, m))
        out[m] = sig.replace(parameters=list(sig.parameters.values())[1:])
    return out


class TapeFake:
    """A TapeEpisode as the backend and the navigator may see it: its writes (served in order, with the Episode's
    signatures) and a sim that answers n_steps and max_steps from the tape without consuming it; nothing else, so no
    connector read reaches the tape but through Src."""

    def __init__(self, te: tp.TapeEpisode, src: Src):
        object.__setattr__(self, "_te", te)
        object.__setattr__(self, "_src", src)

    def __getattr__(self, name):
        if name in WRITES:
            fn = getattr(self._te, name)

            def call(*args, **kwargs):
                return fn(*args, **kwargs)

            call.__name__, call.__signature__ = name, episode_signatures()[name]
            return call
        if name == "sim" and self._src.has("sim"):
            return _PeekSim(self._src)
        raise AttributeError(name)


class _PeekSim:
    __slots__ = ("_src",)

    def __init__(self, src: Src):
        object.__setattr__(self, "_src", src)

    def __getattr__(self, name):
        if name not in tp.SIM_READS:
            raise AttributeError(name)
        return self._src.attr(f"sim.{name}")


# -------------------------------------------------------------------------------------------- the simulator's seats
class RefusingTeleport:
    """The TeleportNavigator's seat under EpisodeNavigator: a non-legacy stance has nowhere to go offline."""

    requires_sim_clock = True

    def __init__(self):
        self.refused: list = []

    def base_pose(self):
        return Belief(None, SRC, 0)

    def propose(self, req, k: int = 8) -> list:
        return []

    def go_to(self, stance, obs):
        self.refused.append(stance.key)
        return NavResult(False, stance, 0, 0, "the offline harness has no teleport"), obs
        yield

    def apply(self, update) -> None:
        pass


class NoNative:
    """The tiptop backend's seat: PARITY routes nothing here."""

    name = "tiptop"

    def supports(self, call) -> bool:
        return True

    def check(self, call, svc):
        raise OfflineOnly(f"{call.skill} reached the tiptop backend offline")

    def run(self, call, svc, obs):
        raise OfflineOnly(f"{call.skill} reached the tiptop backend offline")


class NoPlanner:
    """The planner server's seat: a legacy route never asks it."""

    def __getattr__(self, name):
        raise OfflineOnly(f"planner.{name} was asked offline")


class _Host:
    """HostHooks and DirectConnector's adapter over the null Env of b1k/tests/fakes.py."""

    def __init__(self):
        self.env, self.adapter = Env(), Adapter()

    def commanded_targets(self) -> dict:
        return {}

    def observe_now(self) -> StepObs:
        return self.adapter.parse(self.env.raw(), 0)

    def env_step(self, a):
        return self.env.step(a)

    def raw(self):
        return self.env.raw()

    def parse(self, raw, step):
        return self.adapter.parse(raw, step)


def members_of(ep) -> frozenset:
    """The Episode members the fake has now (the shim exposes exactly these)."""
    return frozenset(m for m in ALL if hasattr(ep, m))


def members_of_tape(tape: tp.Tape) -> frozenset:
    """The members a tape shows present: probed present, or asked, somewhere on it; a member only ever probed
    absent is absent."""
    present = set()
    for r in tape.records:
        if r["kind"] != "probe" or r["present"]:
            present.add(r["member"])
    return frozenset(m for m in ALL if m in present)


# ------------------------------------------------------------------------------------------------------- the stack
class Harness(SimpleNamespace):
    def runner_side(self, shim) -> RunnerSide:
        return RunnerSide(shim, self.src.origin)

    def close(self) -> None:
        if not self.closed:
            skillrun.PASSTHROUGH = self.saved_passthrough
            self.closed = True

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()

    def violations(self) -> list:
        """What a faithful run never shows: typed/literal mismatches, channel entries nobody took, Runtime steps,
        env steps, teleports asked for."""
        ch, rt = self.channel, self.rt
        out = [f"mismatch {m}" for m in ch.mismatches]
        if ch._entries:  # the channel's own record of entries no backend took
            out.append(f"unconsumed entries {sorted(ch._entries)}")
        if ch._nav:
            out.append(f"unconsumed nav entries {list(ch._nav)}")
        if rt.step or rt.idle_steps or any(rt.charged.values()):
            out.append(f"runtime steps {rt.step} idle {rt.idle_steps} charged {dict(rt.charged)}")
        if self.host.env.n:
            out.append(f"env steps {self.host.env.n}")
        if self.teleport.refused:
            out.append(f"teleports refused {self.teleport.refused}")
        return out

    def diagnostics(self, shim=None) -> dict:
        ch, rt = self.channel, self.rt
        return {
            "mismatches": [encode(m) for m in ch.mismatches],
            "exc_counts": dict(ch.exc_counts),
            "violations": self.violations(),
            "rt": {"step": rt.step, "idle_steps": rt.idle_steps, "charged": dict(rt.charged), "live": sorted(rt.runs),
                   "results": len(rt.results)},
            "calls": dict(Counter(f"{r['skill']}/{r['backend']}/{r['status']}" for r in rt.log)),
            "advisory": dict(self.registry.advisory),
            "planner_reads": dict(rt.planner_oracle_reads),
            "extras": dict(Counter(f"{e['during']}>{e['member']}" for e in self.src.extras)),
            "misreads": [encode(m) for m in self.src.misreads],
            "stale_extras": [encode(m) for m in self.src.stale],
            "shim": None if shim is None else counts(shim),
        }


def build_episode_connector(ep, members, *, scope=(), task: str = "t", src: Optional[Src] = None) -> Harness:
    """The real Runtime and DirectConnector over the Episode ``ep`` (a fake, or a TapeFake with its ``src``); see the
    module docstring. The caller closes it (it restores skillrun.PASSTHROUGH)."""
    from omnigibson.tiptop.scene import EpisodeOver  # the harness exception bench hosts pass through

    members = frozenset(members)
    src = src or Src(ep, members)
    execlog = ExecLog(ep)
    host, channel, teleport = _Host(), LegacyChannel(), RefusingTeleport()
    world, scorer = EpisodeWorld(src, scope), EpisodeScorerOver(src)
    scope_refs = tuple(ref(n) for n in scope)

    def task_info() -> TaskInfo:
        floor = src.attr("floor", plumbing=True) if src.has("floor") else None
        return TaskInfo(0, task, (), scope_refs, None if floor is None else ref(floor), _Arms(src), None, SRC)

    def clock():
        return LiveClock(src) if src.has("sim") else Clock(0, None, 0)

    backend = EpisodeLegacyBackend(execlog, channel, host.observe_now)
    registry = EpisodeRegistry(EPISODE_SPECS, {"legacy": backend, "scripted": ScriptedBackend(), "tiptop": NoNative()},
                               PARITY)
    nav = EpisodeNavigator(execlog, channel, teleport)
    svc = Services(world=world, map=None, geometry=None, articulation=None, joints=None, collision=None,
                   grasp=SimpleNamespace(held=lambda arm, obs: Belief(None, "proprio", 0)),
                   buttons=SimpleNamespace(button=lambda o: Provided(None, "map", 0)),
                   goals=GoalPanel({"scorer": ScorerChecker(scorer)}, "scorer"), planner=NoPlanner(), percepts={},
                   provenance=ProvenancePolicy("pseudo"), clock=clock, epochs=lambda: (0, 0))
    saved = skillrun.PASSTHROUGH
    skillrun.PASSTHROUGH = (EpisodeOver,)
    try:
        rt = Runtime(registry, svc, observer=Observer(), navigator=nav, task_info=task_info, host=host)
        conn = DirectConnector(rt, host.env_step, host, host.raw())
    except BaseException:
        skillrun.PASSTHROUGH = saved
        raise
    return Harness(conn=conn, rt=rt, channel=channel, backend=backend, nav=nav, registry=registry, world=world,
                   scorer=scorer, src=src, execlog=execlog, host=host, teleport=teleport, members=members, ep=ep,
                   saved_passthrough=saved, closed=False)


# ------------------------------------------------------------------------------------- logs, states and rebuilding
def encode(x: Any) -> Any:
    """The tape codec's JSON, with anything it cannot carry (a fake's helper object, a function, a class) reduced to
    its type and, for an object, its own sorted vars: comparable across processes (no addresses)."""
    try:
        return tp.encode(x)
    except TypeError:
        pass
    if isinstance(x, (list, tuple)):
        return [encode(v) for v in x]
    if isinstance(x, (set, frozenset)):
        return sorted((encode(v) for v in x), key=_order)
    if isinstance(x, dict):
        return [[encode(k), encode(v)] for k, v in x.items()]
    if isinstance(x, type) or inspect.isroutine(x) or isinstance(x, functools.partial):
        return {"$fn": getattr(x, "__qualname__", type(x).__qualname__)}
    if hasattr(x, "__dict__"):
        return {"$obj": type(x).__qualname__, "vars": {k: encode(v) for k, v in sorted(vars(x).items())}}
    return {"$repr": re.sub(r" at 0x[0-9a-f]+", "", repr(x))}


def _order(v: Any) -> str:
    return json.dumps(v, sort_keys=True)


def _snap(x: Any) -> Any:
    try:
        return copy.deepcopy(x)
    except Exception:  # noqa: BLE001 - an answer that cannot be copied is served as it is
        return x


def state_of(ep) -> dict:
    """The fake's final state: its vars, sorted, encoded."""
    return {k: encode(v) for k, v in sorted(vars(ep).items())}


def state_digest(ep) -> str:
    return hashlib.sha256(json.dumps(state_of(ep), sort_keys=True).encode()).hexdigest()


def construction(strategy) -> dict:
    """The Runner's construction inputs, whole (tape.header's ``inputs`` carries the options only as a digest): the
    spec's fields, the goal, the options as the Runner holds them (committed to a container already: committing is
    idempotent), the attempts and the scope."""
    spec = strategy.spec
    return {"spec": dataclasses.asdict(spec) if dataclasses.is_dataclass(spec) else None,
            "goal": list(strategy.goal), "options": [list(o) for o in strategy.options],
            "attempts": strategy.attempts, "scope": list(strategy.scope)}


def construction_of(header: dict) -> dict:
    """The construction inputs a header carries: q1_pytest's ``construction`` block; else the bench's form (its
    ``options`` and ``scope`` beside ``inputs``, bench.py: the Runner strategy_for built); else, for a one-option
    header whose digest is the goal's, the goal alone. A header with none of them cannot be rebuilt (ValueError)."""
    c, inputs = header.get("construction"), header.get("inputs") or {}
    if c is not None:
        return c
    if header.get("options") is not None:
        return {"spec": None, "goal": inputs.get("goal"), "options": header["options"],
                "attempts": inputs.get("attempts"), "scope": header.get("scope") or inputs.get("scope") or []}
    one = [list(inputs.get("goal") or ())]
    if inputs.get("n_options") != 1 or hashlib.sha256(tp.dumps(one).encode()).hexdigest() != inputs.get(
            "options_sha256"):
        raise ValueError(f"{header.get('instance')}: the header carries no construction inputs, and its options are "
                         f"not its goal alone ({inputs.get('n_options')} options)")
    return {"spec": None, "goal": inputs.get("goal"), "options": one, "attempts": inputs.get("attempts"),
            "scope": inputs.get("scope") or []}


def rebuild_runner(header: dict):
    """The Runner a tape was recorded from (construction_of): strategy_for when the recorded spec is the one it would
    pick for the task (or none was recorded), else Runner over the recorded spec. The rebuilt Runner's inputs must
    equal the header's."""
    from b1k.bridge import strategies as st

    task, c, inputs = header["task"], construction_of(header), header.get("inputs") or {}
    default = st.STRATEGIES.get(task) or st.TaskSpec(task, task.replace("_", " "), press="in_place")
    spec = default if c["spec"] is None else st.TaskSpec(**c["spec"])
    if spec == default:
        runner = st.strategy_for(task, c["goal"], options=c["options"], attempts=c["attempts"], scope=c["scope"])
    else:
        runner = st.Runner(spec, c["goal"], options=c["options"], attempts=c["attempts"], scope=c["scope"])
    got = tp.runner_inputs(runner)
    if inputs and got != inputs:
        raise ValueError(f"{header.get('instance')}: the rebuilt Runner's inputs differ from the header's: {got} != "
                         f"{inputs}")
    return runner


def _exc_classes() -> tuple:
    """What a tape's endings may be, as the bench records them: the Runner's two, EpisodeOver, and TapeDiverged (a
    BaseException, on tape because bench.py's TapeRecorder names it)."""
    from b1k.planner.pseudo.errors import TransferBlocked, Unreachable
    from omnigibson.tiptop.host.wstape import TapeDiverged
    from omnigibson.tiptop.scene import EpisodeOver

    return Unreachable, TransferBlocked, EpisodeOver, TapeDiverged


def _ending(fn) -> tuple:
    try:
        return fn(), None
    except (KeyboardInterrupt, SystemExit):
        raise
    except BaseException as e:  # noqa: BLE001 - the run's ending is what is compared, TapeDiverged included
        return None, e


def without_diagnostics(tape: tp.Tape) -> tp.Tape:
    """The tape with the host's probe fields (DIAGNOSTIC: a write's step and digest) taken off every record: what
    G2 compares, since a replay's recorder has no sim to probe. G4 compares the probes."""
    return tp.Tape(tape.header, [{k: v for k, v in r.items() if k not in DIAGNOSTIC} for r in tape.records])


def _probed(tape: tp.Tape) -> int:
    return sum(1 for r in tape.records if any(k in r for k in DIAGNOSTIC))


def _show(e) -> Optional[list]:
    return None if e is None else [type(e).__name__, str(e)]


def replay(tape: tp.Tape) -> dict:
    """G2 for one tape (WEEK4_PLAN §5.3): E0 (the rebuilt Runner against a TapeEpisode) and the connector replay
    (Runner -> TapeRecorder(shim) -> this stack -> ExecLog -> TapeEpisode). ``ok`` requires, for both: the recording
    equals the tape (the host's step and digest probes aside: ``without_diagnostics``; their count is reported),
    every write served in order, nothing unanswerable; for the connector, no typed/literal mismatch and no
    violation, and the same ending as E0 (a recorded TapeDiverged included: both recorders tape it, as the bench's
    does). Extra reads by the connector's own code are returned in ``extras``, never failed on; an extra over the
    tape that had no answer in its own segment is in ``stale_extras``."""
    from b1k.planner.pseudo.planner import PseudoPlanner

    text, exc = tape.dumps(), _exc_classes()
    header = tape.header
    out = {"instance": header.get("instance"), "task": header.get("task"), "writes": len(tape.writes),
           "records": len(tape.records)}
    try:
        rebuild_runner(header)
    except ValueError as e:  # a header without the construction inputs: this tape cannot be replayed
        return {**out, "ok": False, "error": str(e), "e0": {"ok": False}, "connector": {"ok": False, "extras": {}}}
    # E0
    plain = without_diagnostics(tape)
    out["probed_records"] = _probed(tape)
    te0 = tp.TapeEpisode(tp.Tape.loads(text), exc_classes=exc)
    rec0 = tp.Tape(dict(header))
    ret0, e0 = _ending(lambda: rebuild_runner(header).run(tp.TapeRecorder(te0, rec0, exc_classes=exc)))
    d0 = tp.diff(plain, rec0)
    out["e0"] = {"diff": None if d0 is None else [d0.index, d0.kind, encode(d0.a), encode(d0.b)],
                 "served": te0.segment, "unanswerable": encode(te0.unanswerable), "ending": _show(e0)}
    out["e0"]["ok"] = d0 is None and te0.segment == len(tape.writes) and not te0.unanswerable
    # the connector stack
    te1 = tp.TapeEpisode(tp.Tape.loads(text), exc_classes=exc)
    members = members_of_tape(tape)
    src = Src(te1, members, tape=tape)
    scope = construction_of(header)["scope"]
    rec1 = tp.Tape(dict(header))
    with build_episode_connector(TapeFake(te1, src), members, scope=scope, task=header.get("task") or "t",
                                 src=src) as h:
        planner = PseudoPlanner(runner=rebuild_runner(header), channel=h.channel, members=members,
                                tape=lambda s: tp.TapeRecorder(h.runner_side(s), rec1, exc_classes=exc))
        ret1, e1 = _ending(lambda: planner.run(h.conn))
        diag = h.diagnostics(planner.shim)
    d1 = tp.diff(plain, rec1)
    out["connector"] = {"diff": None if d1 is None else [d1.index, d1.kind, encode(d1.a), encode(d1.b)],
                        "served": te1.segment, "unanswerable": encode(te1.unanswerable), "ending": _show(e1),
                        "mismatches": diag["mismatches"], "violations": diag["violations"],
                        "extras": diag["extras"], "misreads": diag["misreads"], "advisory": diag["advisory"],
                        "calls": diag["calls"], "stale_extras": diag["stale_extras"]}
    out["connector"]["ok"] = (d1 is None and te1.segment == len(tape.writes) and not te1.unanswerable
                              and not diag["mismatches"] and not diag["violations"]
                              and _show(e1) == _show(e0))
    out["ok"] = out["e0"]["ok"] and out["connector"]["ok"]
    return out


# --------------------------------------------------------------------------------------------- comparing two sessions
def compare_logs(a, b) -> dict:
    """Per test: identical, or the first key the two session logs (q1_pytest --q1-log) differ at. Files starting
    with '_' are session summaries and are not compared."""
    a, b = Path(a), Path(b)
    names = sorted({p.name for d in (a, b) for p in d.glob("*.json") if not p.name.startswith("_")})
    out = {"tests": len(names), "identical": 0, "differ": {}, "missing": []}
    for n in names:
        pa, pb = a / n, b / n
        if not (pa.exists() and pb.exists()):
            out["missing"].append(n)
            continue
        la, lb = json.loads(pa.read_text()), json.loads(pb.read_text())
        if la == lb:
            out["identical"] += 1
        else:
            out["differ"][n] = _first_difference(la, lb)
    return out


def _first_difference(x, y, path="") -> str:
    if type(x) is not type(y):
        return f"{path}: {type(x).__name__} != {type(y).__name__}"
    if isinstance(x, dict):
        for k in sorted(set(x) | set(y)):
            if x.get(k) != y.get(k):
                return _first_difference(x.get(k), y.get(k), f"{path}.{k}")
    if isinstance(x, list):
        for i, (u, v) in enumerate(zip(x, y)):
            if u != v:
                return _first_difference(u, v, f"{path}[{i}]")
        if len(x) != len(y):
            return f"{path}: length {len(x)} != {len(y)}"
    return f"{path}: {json.dumps(x)[:200]} != {json.dumps(y)[:200]}"


def main(argv=None) -> int:
    """python -m omnigibson.tiptop.host.q1_harness compare DIR_A DIR_B | replay TAPE_OR_DIR ..."""
    argv = list(sys.argv[1:] if argv is None else argv)
    if argv[:1] == ["compare"] and len(argv) == 3:
        r = compare_logs(argv[1], argv[2])
        print(f"logs: {r['tests']} tests, identical {r['identical']}, differ {len(r['differ'])}, "
              f"missing {len(r['missing'])}")
        for n, why in sorted(r["differ"].items()):
            print(f"  DIFFER {n}: {why}")
        for n in r["missing"]:
            print(f"  MISSING {n}")
        return 0 if r["identical"] == r["tests"] and r["tests"] else 1
    if argv[:1] == ["replay"] and len(argv) >= 2:
        paths = [p for a in argv[1:] for p in (sorted(Path(a).rglob("*.json")) if Path(a).is_dir() else [Path(a)])]
        bad = 0
        for p in paths:
            r = replay(tp.Tape.load(p))
            bad += not r["ok"]
            print(f"{'ok  ' if r['ok'] else 'FAIL'} {p.name}: writes {r['writes']} e0 {r['e0']['ok']} connector "
                  f"{r['connector']['ok']} extras {sum(r['connector']['extras'].values())}"
                  + (f" ({r['error']})" if "error" in r else ""))
        print(f"replayed {len(paths)}, ok {len(paths) - bad}")
        return 1 if bad or not paths else 0
    print(main.__doc__)
    return 2


if __name__ == "__main__":
    sys.exit(main())
