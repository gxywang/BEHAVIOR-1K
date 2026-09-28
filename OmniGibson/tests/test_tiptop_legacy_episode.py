"""The episode-mode host pieces of the pseudo planner's Q1 shim (WEEK4_PLAN §3.3, W4-C) without Isaac Sim: the
LegacyChannel, EpisodeLegacyBackend (the literal call executed, the typed one rebuilt and checked, exceptions back
through the channel as the same objects, EpisodeOver through PASSTHROUGH, hands from the record diff, status from
the judge or the literal), EpisodeNavigator, EpisodeRegistry's advisory precheck, EPISODE_SPECS, the routing profiles
and their overlay, TeleportNavigator's legacy guard and BenchHost's sim clock. Fakes: a FakeEpisode with records,
hands and n_steps, and one with no sim at all."""

import dataclasses
from types import SimpleNamespace

import numpy as np
import pytest

import b1k.runtime.skillrun as skillrun
from b1k.bridge.strategies import TransferBlocked, Unreachable
from b1k.connector.goals import GoalPanel
from b1k.connector.observe import StepObs
from b1k.connector.skills import (
    CloseArgs,
    Code,
    IntentArgs,
    NavResult,
    OpenArgs,
    PickArgs,
    PlaceArgs,
    PressArgs,
    Rel,
    Relation,
    ReleaseArgs,
    SkillCall,
    Stance,
    Status,
    WaitArgs,
    WorldUpdate,
)
from b1k.connector.types import AABB, Belief, Fact, ObjRef, Pose2
from b1k.connector.world import ProvenancePolicy
from b1k.runtime.compose import CLOSED
from b1k.runtime.core import Runtime
from b1k.runtime.direct import DirectConnector
from b1k.runtime.skillrun import Services
from b1k.skills.registry import SkillRegistry, load_routing
from b1k.tests.fakes import REFS, TASK, Env, JointWorld, Observer, Scripted, SimV, apple, basket, floor, radio, table
from omnigibson.tiptop.host.bench_host import BenchHost
from omnigibson.tiptop.host.legacy_channel import LegacyChannel
from omnigibson.tiptop.host.legacy_episode import (
    EPISODE_SPECS,
    NO_ENTRY,
    EpisodeLegacyBackend,
    EpisodeNavigator,
    EpisodeRegistry,
)
from omnigibson.tiptop.host.routing_profiles import NATIVE, PARITY, native, overlay
from omnigibson.tiptop.host.teleport_nav import MOVE_TO_STEPS, TeleportNavigator
from omnigibson.tiptop.scene import EpisodeOver

jar = ObjRef("jar.n.01_1", "jar")
cabinet = ObjRef("cabinet.n.01_1", "cabinet", True)
camera = ObjRef("camera.n.01_1", "camera")
tripod = ObjRef("tripod.n.01_1", "tripod")
ALL = [*REFS, jar, cabinet, camera, tripod]
STANCE = Stance("legacy:cabinet.n.01_1", Pose2(0.0, 0.0, 0.0), 0.0, "Episode.stand_for", "oracle")


# ----------------------------------------------------------------------------------------------------------- fakes
class FakeSim:
    def __init__(self, max_steps=None, labels=True):
        self.n_steps, self.teleports, self.max_steps = 0, 0, max_steps
        self.held_objects = {}  # tracked label -> arm, as R1ProSim.hands() reads it
        if labels:  # the sim tracks labels, not BDDL names: bench.py:733's mapping
            self.bddl_names = {"jar_1": jar.id, "apple_1": apple.id, "camera_1": camera.id}

    def hands(self):
        return dict(self.held_objects)


class FakeEpisode:
    """Episode's write methods with their real signatures. Every call is logged with its FULL args and kwargs (an
    ExecLog: bench.Episode.pick's ``into`` included); ``ok`` is what they return, ``raises`` what they raise, after
    stepping the sim as a real round would."""

    def __init__(self, max_steps=None, ok=True, raises=None, sim=True, labels=True):
        if sim:
            self.sim, self.records = FakeSim(max_steps, labels), []
        self.calls, self.ok, self.raises = [], ok, raises

    def _label(self, bddl):
        names = getattr(getattr(self, "sim", None), "bddl_names", {})
        return next((k for k, v in names.items() if v == bddl), bddl)

    def _did(self, method, args, kwargs, steps, record=None):
        self.calls.append((method, args, kwargs))
        if hasattr(self, "sim"):
            self.sim.n_steps += steps
            if record is not None:
                self.records.append({**record, "step": self.sim.n_steps})
        if self.raises is not None:
            raise self.raises
        return self.ok

    def pick(self, bddl, into=None, single_round=False):
        ok = self._did("pick", (bddl,), {"into": into, "single_round": single_round}, 40, {"round": 0, "pick": bddl})
        if ok and hasattr(self, "sim"):
            self.sim.held_objects[self._label(bddl)] = "left"
        return ok

    def achieve(self, atoms, arm="left", floor=None, done=None):
        ok = self._did("achieve", (atoms,), {"arm": arm, "floor": floor, "done": done}, 30, {"round": 1, "atoms": atoms})
        if ok and hasattr(self, "sim") and atoms and atoms[0]["predicate"] in ("ontop", "inside", "nextto"):
            self.sim.held_objects.pop(self._label(atoms[0]["args"][0]), None)  # a placement lets go
        return ok

    def put_down(self, bddl, support, floor=None):
        ok = self._did("put_down", (bddl, support), {"floor": floor}, 30, {"round": 1, "put_down": bddl})
        if ok and hasattr(self, "sim"):
            self.sim.held_objects.pop(self._label(bddl), None)
        return ok

    def open_up(self, name, fraction=None, single_round=False, joint=None):
        return self._did("open_up", (name,), {"fraction": fraction, "single_round": single_round, "joint": joint}, 20,
                         {"open": name, "why": "" if self.ok else "no grasp on the handle"})

    def release(self, steps=45):  # Episode.release returns None
        self._did("release", (), {"steps": steps}, steps)
        if hasattr(self, "sim"):
            self.sim.held_objects.clear()

    def pour(self, item, target):
        return self._did("pour", (item, target), {}, 25)

    def dwell(self, steps):
        left = self.sim.max_steps - self.sim.n_steps if getattr(getattr(self, "sim", None), "max_steps", None) else int(steps)
        steps = max(0, min(int(steps), left))
        self._did("dwell", (steps,), {}, steps)
        return steps

    def stand_for(self, *names):
        if hasattr(self, "sim"):
            self.sim.teleports += 1
        self._did("stand_for", names, {}, 40)
        return {"stance": names}

    def walk_to_floor(self, name):
        if hasattr(self, "sim"):
            self.sim.teleports += 1
        return self._did("walk_to_floor", (name,), {}, 40)


class Host:
    """HostHooks and DirectConnector's env/adapter over the fake env."""

    def __init__(self, env):
        self.env, self.arm_left = env, 0.0

    def commanded_targets(self):
        return {"arm_left": np.full(7, self.arm_left, dtype=np.float32),
                "gripper_left": np.array([CLOSED], dtype=np.float32)}  # fmt: skip

    def observe_now(self):
        return StepObs(0, self.env.p.copy(), {})

    def env_step(self, a):
        return self.env.step(a)

    def raw(self):
        return self.env.raw()

    def parse(self, raw, step):
        return StepObs(step, raw["proprio"], raw)


class NoPlanner:
    """The planner server as a legacy route must see it: never. Every touch is counted in ``touched`` (a handler
    that swallows the AssertionError cannot hide it) and raises."""

    def __init__(self):
        object.__setattr__(self, "touched", [])

    def __getattr__(self, name):
        self.touched.append(name)
        raise AssertionError(f"planner.{name} was asked on a legacy route")


class BoxWorld(JointWorld):
    """Knows the apple's box, so pick_up's StanceRequest has key poses and the reach test would ask the IK service."""

    def box(self, o):
        return self._b(AABB((0.9, 0.0, 0.8), (1.0, 0.1, 0.9)) if o.id == apple.id else None)


class TeleportStub:
    """The TeleportNavigator's seat under EpisodeNavigator: every non-legacy go_to is recorded and refused."""

    requires_sim_clock = True

    def __init__(self):
        self.gone, self.applied = [], []

    def base_pose(self):
        return Belief(Pose2(0.0, 0.0, 0.0), "oracle", 0)

    def propose(self, req, k=8):
        return [Stance("ring:0", Pose2(1.0, 0.0, 0.0), 1.0, "stub", "oracle")]

    def go_to(self, stance, obs):
        self.gone.append(stance.key)
        return NavResult(False, stance, 0, 0, "stub refuses"), obs
        yield

    def apply(self, u):
        self.applied.append(u)


def build(ep, truth=None, facts=(), routing=None, planner=None, world=None, registry=EpisodeRegistry,
          base_pose=None, backends=None):
    """The production Runtime and DirectConnector over EpisodeRegistry, the channel, the backend and the navigator:
    what the connector host wires (WEEK4_PLAN §3.3, W4-E), with the fakes in the seats of the sim."""
    env, ch = Env(), LegacyChannel()
    h = Host(env)
    lb = EpisodeLegacyBackend(ep, ch, h.observe_now)
    tp = TeleportStub()
    nav = EpisodeNavigator(ep, ch, tp, base_pose=base_pose)
    v = SimV(truth)
    svc = Services(world=world or JointWorld(ALL, facts), map=None, geometry=None, articulation=None, joints=None,
                   collision=None, grasp=SimpleNamespace(held=lambda arm, obs: Belief(None, "proprio", 0)),
                   buttons=SimpleNamespace(button=lambda o: Belief(None, "oracle", 0)),
                   goals=GoalPanel({"scorer": v}, "scorer"), planner=planner or NoPlanner(), percepts={},
                   provenance=ProvenancePolicy("pseudo"), clock=lambda: None, epochs=lambda: (0, 0))
    reg = registry(EPISODE_SPECS, backends or {"legacy": lb, "scripted": Scripted()}, routing or PARITY)
    rt = Runtime(reg, svc, observer=Observer(), navigator=nav, task_info=lambda: TASK, host=h)
    conn = DirectConnector(rt, h.env_step, h, h.raw())
    return SimpleNamespace(conn=conn, rt=rt, ch=ch, lb=lb, nav=nav, tp=tp, reg=reg, v=v, env=env, ep=ep)


def put_run(b, cid, entry, call):
    b.ch.put(cid, entry)
    return b.conn.run(dataclasses.replace(call, call_id=cid))


def entry(method, *args, **kwargs):
    return {"method": method, "args": args, "kwargs": kwargs}


ON_TABLE = PlaceArgs(jar, (Relation(Rel.ON, table),))
IN_CABINET = PlaceArgs(jar, (Relation(Rel.IN, cabinet),))


# ---------------------------------------------------------------------------------------------- the literal call
def test_the_literal_call_is_executed_with_its_declared_extras_and_the_rebuild_check_is_silent():
    """Rows 18-24 of WEEK4_PLAN §4: the Runner's call, into=, floor= and arm= included, runs as given; the typed
    call is rebuilt from the SkillCall plus those extras and agrees with it."""
    b = build(FakeEpisode())
    rows = [
        ("q1-1", entry("pick", jar.id, into=cabinet.id), SkillCall("pick_up", PickArgs(jar), arm="left"),
         ("pick", (jar.id,), {"into": cabinet.id, "single_round": False}), True),
        ("q1-2", entry("achieve", [{"predicate": "inside", "args": [jar.id, cabinet.id]}], arm="right"),
         SkillCall("place", IN_CABINET, arm="right"),
         ("achieve", ([{"predicate": "inside", "args": [jar.id, cabinet.id]}],), {"arm": "right", "floor": None, "done": None}), True),
        ("q1-3", entry("put_down", jar.id, floor.id, floor=True),
         SkillCall("place", PlaceArgs(jar, (Relation(Rel.ON, floor),)), arm="left"),
         ("put_down", (jar.id, floor.id), {"floor": True}), True),
        ("q1-4", entry("open_up", cabinet.id, fraction=0.8),
         SkillCall("open", OpenArgs(cabinet, min_fraction=0.8), arm="left"),
         ("open_up", (cabinet.id,), {"fraction": 0.8, "single_round": False, "joint": None}), True),
        ("q1-5", entry("open_up", cabinet.id, fraction=0.0), SkillCall("close", CloseArgs(cabinet), arm="left"),
         ("open_up", (cabinet.id,), {"fraction": 0.0, "single_round": False, "joint": None}), True),
        ("q1-6", entry("achieve", [{"predicate": "toggled_on", "args": [radio.id]}]),
         SkillCall("press", PressArgs(radio, want_on=True), arm="left"),
         ("achieve", ([{"predicate": "toggled_on", "args": [radio.id]}],), {"arm": "left", "floor": None, "done": None}), True),
        ("q1-7", entry("release"), SkillCall("release", ReleaseArgs(), arm="left"),
         ("release", (), {"steps": 45}), None),
        ("q1-8", entry("pour", jar.id, basket.id), SkillCall("intent.pour", IntentArgs(jar, basket, "pour"), arm="left"),
         ("pour", (jar.id, basket.id), {}), True),
        ("q1-9", entry("achieve", [{"predicate": "attached", "args": [camera.id, tripod.id]}], arm="left"),
         SkillCall("intent.attach", IntentArgs(camera, tripod, "attached"), arm="left"),
         ("achieve", ([{"predicate": "attached", "args": [camera.id, tripod.id]}],), {"arm": "left", "floor": None, "done": None}), True),
        ("q1-10", entry("dwell", 30), SkillCall("wait", WaitArgs(30), budget_steps=30),
         ("dwell", (30,), {}), 30),
    ]
    for cid, e, call, literal, ret in rows:
        r = put_run(b, cid, e, call)
        assert b.ep.calls[-1] == literal, (cid, b.ep.calls[-1])
        assert r.status is Status.SUCCEEDED and r.backend == "legacy" and r.requires_sim_clock, (cid, r)
        out = b.ch.outcome(cid)
        assert out == {"executed": True, "value": ret, "exc": None}, (cid, out)
        assert r.evidence["method"] == e["method"] and r.evidence["legacy_return"] == ret, (cid, r.evidence)
        assert b.ch.take(cid) is None, "the entry was taken"
    assert b.ch.mismatches == [], b.ch.mismatches
    assert b.rt.step == 0 and b.rt.idle_steps == 0 and sum(b.rt.charged.values()) == 0, \
        "every legacy call costs 0 Runtime steps: it stepped the sim itself (U0)"
    assert b.lb.steps == b.ep.sim.n_steps == 40 + 30 + 30 + 20 + 20 + 30 + 45 + 25 + 30 + 30
    assert [r["evidence"]["records"] for r in b.rt.log][0] == [{"round": 0, "pick": jar.id, "step": 40}], \
        "the round records the call added, and those alone"
    assert b.rt.log[1]["evidence"]["records"] == [
        {"round": 1, "atoms": [{"predicate": "inside", "args": [jar.id, cabinet.id]}], "step": 70}], \
        "a later call's records are its own, never the earlier calls' as well"
    assert b.ch.outcome("q1-99") is None, "never executed: None"


def test_the_rebuild_check_records_a_typed_call_that_disagrees_with_the_literal_which_still_runs():
    """Mutant: executing the rebuilt call (or rebuilding it from the literal) would hide a shim that typed the wrong
    object. The LITERAL jar is picked; the typed apple is the mismatch's rebuilt side."""
    b = build(FakeEpisode())
    r = put_run(b, "q1-1", entry("pick", jar.id, into=cabinet.id), SkillCall("pick_up", PickArgs(apple), arm="left"))
    assert b.ep.calls == [("pick", (jar.id,), {"into": cabinet.id, "single_round": False})], "the literal ran"
    assert r.status is Status.SUCCEEDED, r
    assert len(b.ch.mismatches) == 1
    m = b.ch.mismatches[0]
    assert m["call_id"] == "q1-1"
    assert m["rebuilt"] == {"method": "pick", "bddl": apple.id, "into": cabinet.id, "single_round": False}
    assert m["literal"] == {"method": "pick", "bddl": jar.id, "into": cabinet.id, "single_round": False}
    # a place whose typed target differs
    put_run(b, "q1-2", entry("achieve", [{"predicate": "ontop", "args": [jar.id, table.id]}]),
            SkillCall("place", PlaceArgs(jar, (Relation(Rel.ON, basket),)), arm="left"))
    assert b.ch.mismatches[-1]["rebuilt"]["atoms"] == [{"predicate": "ontop", "args": [jar.id, basket.id]}]
    assert b.ch.mismatches[-1]["literal"]["atoms"] == [{"predicate": "ontop", "args": [jar.id, table.id]}]
    # an intent whose atom names another tool than the typed call
    put_run(b, "q1-3", entry("achieve", [{"predicate": "attached", "args": [camera.id, tripod.id]}]),
            SkillCall("intent.attach", IntentArgs(jar, tripod, "attached"), arm="left"))
    assert b.ch.mismatches[-1]["rebuilt"]["atoms"] == [{"predicate": "attached", "args": [jar.id, tripod.id]}]
    # a close the typed call sends as an open
    put_run(b, "q1-4", entry("open_up", cabinet.id, fraction=0.0),
            SkillCall("open", OpenArgs(cabinet, min_fraction=0.8), arm="left"))
    assert (b.ch.mismatches[-1]["rebuilt"]["fraction"], b.ch.mismatches[-1]["literal"]["fraction"]) == (0.8, 0.0)
    assert len(b.ch.mismatches) == 4 and b.ep.calls[-1][1] == (cabinet.id,)
    # an intent whose typed TARGET differs from the atom's second argument
    put_run(b, "q1-5", entry("achieve", [{"predicate": "attached", "args": [camera.id, tripod.id]}]),
            SkillCall("intent.attach", IntentArgs(camera, basket, "attached"), arm="left"))
    assert b.ch.mismatches[-1]["rebuilt"]["atoms"] == [{"predicate": "attached", "args": [camera.id, basket.id]}]
    # a pairing the table does not know: an achieve typed as a pick
    put_run(b, "q1-6", entry("achieve", [{"predicate": "ontop", "args": [jar.id, table.id]}]),
            SkillCall("pick_up", PickArgs(jar), arm="left"))
    assert b.ch.mismatches[-1]["rebuilt"] == {"method": "achieve", "skill": "pick_up", "pairing": "unknown"}
    assert len(b.ch.mismatches) == 6


def test_a_wait_is_checked_against_the_literal_dwell_capped_by_what_the_episode_has_left():
    """Row 24: the typed steps are min(literal, max_steps - n_steps) when the sim has a limit, else the literal."""
    b = build(FakeEpisode(max_steps=100))
    b.ep.sim.n_steps = 80
    r = put_run(b, "q1-1", entry("dwell", 600), SkillCall("wait", WaitArgs(20), budget_steps=20))
    assert b.ep.calls == [("dwell", (20,), {})] and r.steps == 20 and b.ch.mismatches == []
    put_run(b, "q1-2", entry("dwell", 600), SkillCall("wait", WaitArgs(600), budget_steps=600))
    assert b.ch.mismatches[-1]["rebuilt"] == {"method": "dwell", "typed_steps": 600}, "the typed call carried 600"
    assert b.ch.mismatches[-1]["literal"]["typed_steps"] == 0, "the literal capped by what is left: 0"
    b2 = build(FakeEpisode())  # no limit: the literal
    put_run(b2, "q1-1", entry("dwell", 600), SkillCall("wait", WaitArgs(600), budget_steps=600))
    assert b2.ch.mismatches == [] and b2.ep.calls == [("dwell", (600,), {})]


# ---------------------------------------------------------------------------------------------------- exceptions
@pytest.mark.parametrize("exc", [TransferBlocked("the hand is full"), Unreachable("jar"), RuntimeError("boom")])
def test_an_exception_comes_back_through_the_channel_as_the_same_object(exc):
    b = build(FakeEpisode(raises=exc), truth={"holding": False, "lifted": False})  # nothing held after the crash
    r = put_run(b, "q1-1", entry("pick", jar.id), SkillCall("pick_up", PickArgs(jar), arm="left"))
    assert b.ch.outcome("q1-1")["exc"] is exc and b.ch.outcome("q1-1")["value"] is None, "the shim re-raises it"
    assert r.status is Status.FAILED and r.detail == str(exc) and r.steps == 40, r
    assert b.ch.exc_counts == {type(exc).__name__: 1}
    # and from the navigator's stand_for
    b.ch.put_nav(entry("stand_for", cabinet.id))
    nr = b.conn.go_to(STANCE)
    assert nr.ok is False and nr.stance.key == STANCE.key and type(exc).__name__ in nr.detail
    assert b.ch.nav_outcome()["exc"] is exc and b.ch.exc_counts[type(exc).__name__] == 2
    assert b.nav.steps == 40 and b.lb.steps == 40, "the steps the failed calls took are counted"


def test_episode_over_escapes_runtime_start_and_go_to_and_the_steps_are_still_counted(monkeypatch):
    monkeypatch.setattr(skillrun, "PASSTHROUGH", (EpisodeOver,))
    over = EpisodeOver("timeout", 40)
    b = build(FakeEpisode(raises=over))
    b.ch.put("q1-1", entry("pick", jar.id))
    with pytest.raises(EpisodeOver) as e:
        b.conn.run(SkillCall("pick_up", PickArgs(jar), arm="left", call_id="q1-1"))
    assert e.value is over and b.ch.outcome("q1-1") is None, "not a result: the episode is over"
    assert b.lb.steps == 40 and b.ep.sim.n_steps == 40, "counted, for the host's U0 ledger"
    assert "q1-1" in b.rt.runs, "the run is still live (U0-a's L): nothing runs after EpisodeOver in an episode"
    b = build(FakeEpisode(raises=over))
    b.ch.put_nav(entry("stand_for", cabinet.id))
    with pytest.raises(EpisodeOver):
        b.conn.go_to(STANCE)
    assert b.nav.steps == 40 and b.ch.nav_outcome() is None
    monkeypatch.setattr(skillrun, "PASSTHROUGH", ())  # not a PASSTHROUGH: a result, as any other exception
    b2 = build(FakeEpisode(raises=EpisodeOver("timeout", 40)), truth={"holding": False, "lifted": False})
    r = put_run(b2, "q1-1", entry("pick", jar.id), SkillCall("pick_up", PickArgs(jar), arm="left"))
    assert r.status is Status.FAILED and isinstance(b2.ch.outcome("q1-1")["exc"], EpisodeOver)


# ------------------------------------------------------------------------------------------- precheck and planner
def test_the_advisory_precheck_never_refuses_and_records_what_it_would_have():
    """A hand the WorldView says is full would refuse a pick (HAND_FULL); the Episode owns that: the call runs, the
    would-refuse is counted. A route to a backend that does not own its preconditions gets the full precheck."""
    b = build(FakeEpisode(), facts=(Fact("holding", (apple.id, "left")),))
    call = SkillCall("pick_up", PickArgs(jar), arm="left")
    assert b.conn.check(call).ok, "check() is ok on every legacy route"
    r = put_run(b, "q1-1", entry("pick", jar.id), call)
    assert r.status is Status.SUCCEEDED and b.ep.calls == [("pick", (jar.id,), {"into": None, "single_round": False})]
    assert b.reg.advisory == {"hand_full": 1}, "once per call: the run's own precheck, not check() before it"
    put_run(b, "q1-2", entry("achieve", [{"predicate": "ontop", "args": [jar.id, table.id]}]),
            SkillCall("place", ON_TABLE, arm="right"))  # the right hand holds nothing: NOT_HOLDING would refuse
    assert b.reg.advisory == {"hand_full": 1, "not_holding": 1} and len(b.ep.calls) == 2
    b.reg.routing["press"] = {"default": "scripted"}  # scripted owns nothing: needs_percept refuses as ever
    r = b.conn.run(SkillCall("press", PressArgs(radio), arm="left"))
    assert (r.status, r.code) == (Status.PRECONDITION_UNMET, Code.PERCEPT_REQUIRED), r


def test_a_legacy_route_makes_no_reach_or_ik_call_where_the_plain_registry_would():
    """pick_up has a StanceRequest (the apple's box is known): SkillRegistry's precheck asks the IK service for the
    reach test; EpisodeRegistry's advisory never does (planner: a stub that raises when touched)."""
    call = SkillCall("pick_up", PickArgs(apple), arm="left")
    b = build(FakeEpisode(), world=BoxWorld(ALL))
    r = put_run(b, "q1-1", entry("pick", apple.id), call)
    assert r.status is Status.SUCCEEDED and b.ep.calls == [("pick", (apple.id,), {"into": None, "single_round": False})]
    assert b.rt.svc.planner.touched == [], "no IK request, not even one a handler swallowed"
    assert not any(k.startswith("error:") for k in b.reg.advisory), b.reg.advisory
    plain = build(FakeEpisode(), world=BoxWorld(ALL), registry=SkillRegistry)  # the control: the stub bites there
    r = put_run(plain, "q1-1", entry("pick", apple.id), call)
    assert (r.status, r.code) == (Status.FAILED, Code.BACKEND_ERROR) and "planner.reach" in r.detail, r
    assert plain.ep.calls == [] and plain.rt.svc.planner.touched == ["reach"], "the plain registry asked it"


def test_an_advisory_check_that_raises_is_counted_and_the_call_still_runs():
    def boom(call, svc):
        raise KeyError("typo")

    specs = {**EPISODE_SPECS, "pick_up": dataclasses.replace(EPISODE_SPECS["pick_up"], check=boom)}
    b = build(FakeEpisode(), registry=lambda sp, be, ro: EpisodeRegistry(specs, be, ro))
    r = put_run(b, "q1-1", entry("pick", jar.id), SkillCall("pick_up", PickArgs(jar), arm="left"))
    assert r.status is Status.SUCCEEDED and b.reg.advisory == {"error:KeyError": 1}, (r, b.reg.advisory)


def test_the_judge_reads_the_observation_after_the_run():
    b = build(FakeEpisode())
    after = StepObs(777, b.env.p.copy(), {})
    b.lb.observe_now = lambda: after
    put_run(b, "q1-1", entry("pick", jar.id), SkillCall("pick_up", PickArgs(jar), arm="left"))
    assert b.v.obs_seen and b.v.obs_seen[-1] is after, "the frames after the legacy call, not the Runtime's before it"


class ZeroStepPour(FakeEpisode):
    def pour(self, item, target):
        return self._did("pour", (item, target), {}, 0)


def test_a_failure_with_no_text_that_never_stepped_is_a_backend_error():
    b = build(ZeroStepPour(ok=False))
    r = put_run(b, "q1-1", entry("pour", jar.id, basket.id), SkillCall("intent.pour", IntentArgs(jar, basket, "pour"),
                                                                       arm="left"))
    assert (r.status, r.code, r.phase, r.steps) == (Status.FAILED, Code.BACKEND_ERROR, None, 0), r


def test_a_bent_relation_is_reported_on_the_result_and_still_runs():
    """SPEC F16's degrade marker, as LegacyBackend names it: under and touching always, an in whose target has no
    compartment floor on the legacy wire; reported, never refused (PARITY runs the literal call)."""
    b = build(FakeEpisode())
    b.lb.has_cavity = lambda target, item: False  # the bench without --inside-region
    r = put_run(b, "q1-1", entry("achieve", [{"predicate": "inside", "args": [jar.id, cabinet.id]}]),
                SkillCall("place", IN_CABINET, arm="left"))
    assert r.status is Status.SUCCEEDED and (r.evidence["degraded_to"], r.evidence["relations"]) == ("on", ["in->on"])
    r = put_run(b, "q1-2", entry("achieve", [{"predicate": "under", "args": [jar.id, table.id]}]),
                SkillCall("place", PlaceArgs(jar, (Relation(Rel.UNDER, table),)), arm="left"))
    assert r.evidence["relations"] == ["under->on"]
    r = put_run(b, "q1-3", entry("achieve", [{"predicate": "ontop", "args": [jar.id, table.id]}]),
                SkillCall("place", ON_TABLE, arm="left"))
    assert "degraded_to" not in r.evidence and b.lb.degraded == 2 and len(b.ep.calls) == 3
    b.lb.has_cavity = lambda target, item: True
    r = put_run(b, "q1-4", entry("achieve", [{"predicate": "inside", "args": [jar.id, cabinet.id]}]),
                SkillCall("place", IN_CABINET, arm="left"))
    assert "degraded_to" not in r.evidence, "a compartment floor on the wire: not bent"


class HandOver(FakeEpisode):
    def achieve(self, atoms, arm="left", floor=None, done=None):
        ok = self._did("achieve", (atoms,), {"arm": arm, "floor": floor, "done": done}, 30, {"round": 1})
        self.sim.held_objects[self._label(jar.id)] = "right"  # the jar changed hands within the call
        return ok


def test_a_label_that_changes_hands_within_a_call_is_released_before_it_is_held():
    ep = HandOver()
    ep.sim.held_objects["jar_1"] = "left"
    b = build(ep)
    r = put_run(b, "q1-1", entry("achieve", [{"predicate": "inside", "args": [jar.id, cabinet.id]}], arm="right"),
                SkillCall("place", IN_CABINET, arm="right"))
    assert r.world_updates == (WorldUpdate("released", jar, "left", source="oracle"),
                               WorldUpdate("held", jar, "right", source="oracle")), r.world_updates


class Rounds(FakeEpisode):
    """achieve adds the scripted round records and returns ``ok`` (tidying_bedroom's rows, G3)."""

    def __init__(self, rounds, **kw):
        super().__init__(**kw)
        self.rounds = rounds

    def achieve(self, atoms, arm="left", floor=None, done=None):
        self.calls.append(("achieve", (atoms,), {"arm": arm}))
        for r in self.rounds:
            self.sim.n_steps += int(r.get("env_steps", 0)) + 100  # the captures and the plan
            self.records.append({**r, "step": self.sim.n_steps})
        return self.ok


NEXT_TO_BED = PlaceArgs(jar, (Relation(Rel.NEXT_TO, table),))


def test_a_call_whose_rounds_never_executed_is_coded_by_its_rounds_error_whatever_the_episode_returned():
    """tidying q1-4: every round failed to plan (No satisfying particles), yet Episode.achieve returned True (the
    held sandal hung beside the bed, and its own nextto test is geometric); the judge says no. Nothing was placed:
    the cause is the planner's, not a wrong placement."""
    ep = Rounds([{"round": 11, "error": "TiptopPlanningError: planning failed: cuTAMP failed to find a plan: No "
                  "satisfying particles found after optimizing all 1 plan(s)"}], ok=True)
    b = build(ep, truth={"nextto": False})
    r = put_run(b, "q1-4", entry("achieve", [{"predicate": "nextto", "args": [jar.id, table.id]}]),
                SkillCall("place", NEXT_TO_BED, arm="left"))
    assert (r.status, r.code, r.phase) == (Status.INFEASIBLE, Code.NO_PLACEMENT, "particles"), r
    assert r.evidence["legacy_ok"] is True


def test_a_call_whose_last_round_executed_is_coded_by_that_execution_not_an_earlier_refusal():
    """tidying q1-7: round 17 was rejected by motion validation after 40 steps, round 18 ran its plan and set the
    sandal down in the wrong place; the Episode returned False and the judge says no: an executed wrong placement."""
    ep = Rounds([{"round": 17, "env_steps": 40, "error": "motion validation rejected PlaceNear(sandal_2, ...)"},
                 {"round": 18, "env_steps": 258}], ok=False)
    b = build(ep, truth={"nextto": False})
    r = put_run(b, "q1-7", entry("achieve", [{"predicate": "nextto", "args": [jar.id, table.id]}]),
                SkillCall("place", NEXT_TO_BED, arm="left"))
    assert (r.status, r.code, r.phase) == (Status.FAILED, Code.PLACED_WRONG, "execute"), r
    ep = Rounds([{"round": 18, "env_steps": 258}, {"round": 19, "env_steps": 40, "error": "motion validation "
                  "rejected PlaceNear(sandal_2, ...)"}], ok=False)
    b = build(ep, truth={"nextto": False})
    r = put_run(b, "q1-8", entry("achieve", [{"predicate": "nextto", "args": [jar.id, table.id]}]),
                SkillCall("place", NEXT_TO_BED, arm="left"))
    assert (r.status, r.code, r.phase) == (Status.INFEASIBLE, Code.EXEC_REFUSED, "motion"), "the last round's own"


# ----------------------------------------------------------------------------------------------- hands and status
def test_the_hand_updates_follow_the_record_diff_with_source_oracle():
    b = build(FakeEpisode())
    r = put_run(b, "q1-1", entry("pick", jar.id, into=cabinet.id), SkillCall("pick_up", PickArgs(jar), arm="left"))
    assert r.world_updates == (WorldUpdate("held", jar, "left", source="oracle"),), r.world_updates
    assert b.rt.svc.world.held("left").value == (jar,), "applied by the Runtime before the planner sees the result"
    r = put_run(b, "q1-2", entry("achieve", [{"predicate": "inside", "args": [jar.id, cabinet.id]}]),
                SkillCall("place", IN_CABINET, arm="left"))
    assert r.world_updates == (WorldUpdate("released", jar, "left", source="oracle"),)
    assert b.rt.svc.world.held("left").value == ()
    b.ep.sim.held_objects["apple_1"] = "right"  # a label the call does not name: its BDDL name, by the sim's map
    r = put_run(b, "q1-3", entry("release"), SkillCall("release", ReleaseArgs(), arm="right"))
    assert r.world_updates == (WorldUpdate("released", apple, "right", source="oracle"),)
    r = put_run(b, "q1-4", entry("open_up", cabinet.id, fraction=0.8), SkillCall("open", OpenArgs(cabinet, min_fraction=0.8)))
    assert r.world_updates == (), "nothing changed hands"


def test_a_fake_with_no_sim_records_or_hands_runs_at_zero_steps_with_no_updates():
    ep = FakeEpisode(sim=False)
    b = build(ep)
    r = put_run(b, "q1-1", entry("pick", jar.id), SkillCall("pick_up", PickArgs(jar), arm="left"))
    assert r.status is Status.SUCCEEDED and r.steps == 0 and r.world_updates == () and r.evidence["records"] == []
    assert ep.calls == [("pick", (jar.id,), {"into": None, "single_round": False})] and b.lb.steps == 0
    b.ch.put_nav(entry("walk_to_floor", floor.id))
    nr = b.conn.go_to(Stance("legacy:" + floor.id, Pose2(0.0, 0.0, 0.0), 0.0, "Episode.walk_to_floor", "oracle"))
    assert nr.ok and nr.shadow_steps == 0 and b.nav.steps == 0


def test_a_call_with_no_entry_is_unsupported_except_a_wait_which_dwells_the_typed_steps():
    b = build(FakeEpisode())
    r = b.conn.run(SkillCall("pick_up", PickArgs(jar), arm="left", call_id="q1-1"))
    assert (r.status, r.code, r.phase, r.steps) == (Status.INFEASIBLE, Code.UNSUPPORTED, "check", 0)
    assert r.detail == NO_ENTRY and b.ep.calls == [] and b.ch.outcome("q1-1") is None
    assert b.conn.dwell(30) == 30 and b.ep.calls == [("dwell", (30,), {})], "conn.dwell: the literal is the typed"
    assert b.rt.step == 0 and b.rt.charged == {"skill": 0, "wait": 0}


def test_a_press_without_a_state_check_is_judged_on_the_literal_atom_it_asked_for():
    """The Runner's achieve([toggled_on(t)]) is typed as a press with want_on None, which effects() leaves unstated:
    the result is judged on that literal atom, so a press that turned its switch on says so in its effects (the
    cook_bacon stove) and one that did not is not a success, whatever the literal returned."""
    b = build(FakeEpisode(ok=True), truth={"toggled_on": True})
    r = put_run(b, "q1-1", entry("achieve", [{"predicate": "toggled_on", "args": [radio.id]}]),
                SkillCall("press", PressArgs(radio, want_on=None), arm="left"))
    assert r.status is Status.SUCCEEDED and r.effects == (Fact("toggled_on", (radio.id,)),), r
    assert len(b.v.obs_seen) == 1, "judged once, after the run"
    b = build(FakeEpisode(ok=True), truth={"toggled_on": False})
    r = put_run(b, "q1-1", entry("achieve", [{"predicate": "toggled_on", "args": [radio.id]}]),
                SkillCall("press", PressArgs(radio, want_on=None), arm="left"))
    assert r.status is not Status.SUCCEEDED and r.effects == () and r.evidence["legacy_ok"] is True, r


def test_an_empty_effects_call_takes_its_status_from_the_literal_return_and_the_error():
    """An intent has nothing the GoalChecker judges: the literal decides."""
    b = build(FakeEpisode(ok=False))
    r = put_run(b, "q1-2", entry("achieve", [{"predicate": "attached", "args": [camera.id, tripod.id]}]),
                SkillCall("intent.attach", IntentArgs(camera, tripod, "attached"), arm="left"))
    assert r.status is Status.FAILED and r.code is Code.PLACED_WRONG and r.phase == "execute", r
    b2 = build(FakeEpisode(ok=True))
    r = put_run(b2, "q1-1", entry("achieve", [{"predicate": "attached", "args": [camera.id, tripod.id]}]),
                SkillCall("intent.attach", IntentArgs(camera, tripod, "attached"), arm="left"))
    assert r.status is Status.SUCCEEDED and r.evidence["legacy_ok"] is True


def test_a_press_is_never_pre_judged_the_literal_call_always_runs():
    """Mutant: LegacyBackend's pre-judge (a device already in the wanted state succeeds in 0 steps) would skip the
    Runner's call and break parity: the Runner pressed, whatever the scorer said before."""
    b = build(FakeEpisode(), truth={"toggled_on": True})
    r = put_run(b, "q1-1", entry("achieve", [{"predicate": "toggled_on", "args": [radio.id]}]),
                SkillCall("press", PressArgs(radio, want_on=True), arm="left"))
    assert b.ep.calls == [("achieve", ([{"predicate": "toggled_on", "args": [radio.id]}],),
                           {"arm": "left", "floor": None, "done": None})], "pressed"
    assert r.steps == 30 and r.status is Status.SUCCEEDED and r.detail == "" and b.ep.sim.n_steps == 30
    assert len(b.v.obs_seen) == 1, "judged once, after the run"


def test_the_status_comes_from_the_judge_and_from_the_literal_only_when_the_judge_cannot():
    """Mutant: a vacuous verdict (None) read as success would call a failed pick succeeded."""
    unjudged = {"holding": None, "lifted": None}
    b = build(FakeEpisode(ok=False), truth=unjudged)
    r = put_run(b, "q1-1", entry("pick", jar.id), SkillCall("pick_up", PickArgs(jar), arm="left"))
    assert r.status is not Status.SUCCEEDED and r.verdicts == {"scorer": None}, r
    assert (r.status, r.code, r.phase) == (Status.FAILED, Code.GRASP_MISSED, "execute"), "no known text, it moved"
    b = build(FakeEpisode(ok=True), truth=unjudged)
    r = put_run(b, "q1-1", entry("pick", jar.id), SkillCall("pick_up", PickArgs(jar), arm="left"))
    assert r.status is Status.SUCCEEDED and r.effects == (Fact("holding", (jar.id, "left")), Fact("lifted", (jar.id,)))
    b = build(FakeEpisode(ok=True), truth={"holding": True, "lifted": False})  # the judge says no: it decides
    r = put_run(b, "q1-1", entry("pick", jar.id), SkillCall("pick_up", PickArgs(jar), arm="left"))
    assert (r.status, r.code, r.phase) == (Status.FAILED, Code.NOT_LIFTED, "verify") and r.evidence["legacy_ok"]
    b = build(FakeEpisode(ok=False), truth={"ontop": True})  # the judge says yes though the legacy code said no
    r = put_run(b, "q1-1", entry("achieve", [{"predicate": "ontop", "args": [jar.id, table.id]}]),
                SkillCall("place", ON_TABLE, arm="left"))
    assert r.status is Status.SUCCEEDED and r.evidence["legacy_ok"] is False
    b = build(FakeEpisode(ok=False), truth={"open": False})  # the round's own text codes the failure
    r = put_run(b, "q1-1", entry("open_up", cabinet.id, fraction=0.8),
                SkillCall("open", OpenArgs(cabinet, min_fraction=0.8), arm="left"))
    assert (r.status, r.code, r.phase) == (Status.INFEASIBLE, Code.NO_MOTION, "check"), r


def test_a_judge_that_raises_counts_as_unknown():
    b = build(FakeEpisode(ok=True))
    b.v.check = lambda goal, obs=None, percept=None: 1 / 0
    r = put_run(b, "q1-1", entry("pick", jar.id), SkillCall("pick_up", PickArgs(jar), arm="left"))
    assert r.status is Status.SUCCEEDED and r.verdicts == {}, r


# ------------------------------------------------------------------------------------------------------ navigation
def test_a_legacy_stance_runs_the_runners_stand_for_and_lands_where_the_base_pose_says():
    landed = [None]
    b = build(FakeEpisode(), base_pose=lambda: landed[0])
    b.ch.put_nav(entry("stand_for", cabinet.id, jar.id))
    stance = Stance("legacy:cabinet.n.01_1,jar.n.01_1", Pose2(0.0, 0.0, 0.0), 0.0, "Episode.stand_for", "oracle")
    epochs = (b.rt.base_epoch, b.rt.trunk_epoch)
    nr = b.conn.go_to(stance)
    assert b.ep.calls == [("stand_for", (cabinet.id, jar.id), {})]
    assert nr == NavResult(True, stance, 0, MOVE_TO_STEPS, f"Episode.stand_for{(cabinet.id, jar.id)}"), nr
    assert b.ch.nav_outcome() == {"executed": True, "value": {"stance": (cabinet.id, jar.id)}, "exc": None}
    assert b.nav.steps == 40 and b.rt.step == 0 and b.rt.charged == {"go_to": 0}, "0 Runtime steps, on the sim clock"
    assert (b.rt.base_epoch, b.rt.trunk_epoch) == (epochs[0] + 1, epochs[1] + 1)
    landed[0] = Belief(Pose2(1.5, 2.0, 0.3), "oracle", 0)
    b.ch.put_nav(entry("walk_to_floor", floor.id))
    nr = b.conn.go_to(Stance("legacy:" + floor.id, Pose2(0.0, 0.0, 0.0), 0.0, "Episode.walk_to_floor", "oracle"))
    assert nr.ok and nr.stance.pose == Pose2(1.5, 2.0, 0.3) and nr.stance.key == "legacy:" + floor.id
    b.ep.ok = False  # walk_to_floor returns False: the floor was not located
    b.ch.put_nav(entry("walk_to_floor", floor.id))
    nr = b.conn.go_to(Stance("legacy:" + floor.id, Pose2(0.0, 0.0, 0.0), 0.0, "Episode.walk_to_floor", "oracle"))
    assert nr.ok is False and b.ch.nav_outcome()["value"] is False and b.ch.nav_outcome()["exc"] is None
    assert b.tp.gone == [], "no teleport went to the TeleportNavigator"
    nr = b.conn.go_to(Stance("ring:0", Pose2(1.0, 0.0, 0.0), 1.0, "a native stance", "oracle"))
    assert b.tp.gone == ["ring:0"] and nr.detail == "stub refuses", "every other key goes to the TeleportNavigator"
    assert b.conn.propose_stances(None)[0].key == "ring:0"


def test_a_nav_entry_that_does_not_name_the_stances_objects_is_a_harness_error():
    b = build(FakeEpisode())
    b.ch.put_nav(entry("stand_for", jar.id))
    nr = b.conn.go_to(STANCE)
    assert nr.ok is False and "is not the stance's" in nr.detail and b.ep.calls == []
    assert b.ch.nav_outcome() is None, "never executed: the shim raises Q1HarnessError on this"
    assert b.ch.mismatches[-1] == {"call_id": "nav", "rebuilt": {"method": "stand_for", "args": [cabinet.id]},
                                   "literal": {"method": "stand_for", "args": [jar.id]}}


def test_a_leftover_nav_entry_after_a_served_nav_leaves_no_stale_outcome():
    """After one stand_for ran, an entry that does not match the next stance (a FIFO out of step) is not executed,
    and the channel holds no outcome for it: the previous nav's value can never answer the next nav."""
    b = build(FakeEpisode())
    b.ch.put_nav(entry("stand_for", cabinet.id))
    assert b.conn.go_to(STANCE).ok and b.ch.nav_outcome()["executed"]
    b.ch.put_nav(entry("stand_for", tripod.id))  # a leftover entry
    nr = b.conn.go_to(Stance("legacy:" + jar.id, Pose2(0.0, 0.0, 0.0), 0.0, "Episode.stand_for", "oracle"))
    assert nr.ok is False and b.ch.nav_outcome() is None and len(b.ep.calls) == 1
    b.ch.put_nav(entry("walk_to_floor", cabinet.id))  # the right names, the wrong method
    nr = b.conn.go_to(STANCE)
    assert nr.ok is False and b.ch.nav_outcome() is None and len(b.ep.calls) == 1
    assert [m["literal"]["method"] for m in b.ch.mismatches] == ["stand_for", "walk_to_floor"]


def test_the_teleport_navigator_refuses_a_legacy_key_without_place_robot():
    sim = SimpleNamespace(n_steps=0, placed=[], looked=[])
    sim.place_robot = lambda *a, **k: pytest.fail("place_robot: a legacy stance would teleport to the map origin")
    sim.look_at = lambda *names: sim.looked.append(names)
    nav = TeleportNavigator(sim)
    nav.looking = (jar.id,)
    with pytest.raises(StopIteration) as done:
        next(nav.go_to(STANCE, "obs"))
    nr, obs = done.value.value
    assert nr == NavResult(False, STANCE, 0, 0, "a legacy stance outside the episode host") and obs == "obs"
    assert sim.placed == [] and sim.looked == [] and nav.steps == 0 and sim.n_steps == 0


# ---------------------------------------------------------------------------------------------- specs and routing
def test_episode_specs_are_specs_plus_the_six_intents_registered_here_alone():
    from b1k.skills.specs import SPECS

    intents = {k for k in EPISODE_SPECS if k.startswith("intent.")}
    assert intents == {"intent.attach", "intent.stamp", "intent.cut", "intent.heat", "intent.aim", "intent.pour"}
    assert {k: v for k, v in EPISODE_SPECS.items() if not k.startswith("intent.")} == SPECS
    assert not any(k.startswith("intent.") for k in SPECS), "never in SPECS"
    for k in intents:
        s = EPISODE_SPECS[k]
        assert s.args_type is IntentArgs and s.info.default_backend == "legacy" and s.info.budget_steps == 1100
        assert s.check is None and s.stance_request is None and not s.info.needs_percept
    b = build(FakeEpisode())
    assert all(b.lb.supports(SkillCall(k, None)) for k in EPISODE_SPECS)
    assert not b.lb.supports(SkillCall("push", None))


def test_parity_resolves_every_episode_skill_to_legacy_with_the_scorer_alone():
    assert PARITY["goal_checker"] == "scorer" and PARITY["goal_checkers_shadow"] == []
    assert {k for k in PARITY if k not in ("goal_checker", "goal_checkers_shadow")} == set(EPISODE_SPECS)
    b = build(FakeEpisode())
    for k in EPISODE_SPECS:
        assert b.reg.backend_for(SkillCall(k, None)) is b.lb, k
    assert b.reg.backend_for(SkillCall("close", CloseArgs(cabinet, joint="j_link_4"))) is b.lb


def test_native_is_routing_yamls_lines_plus_every_intent_on_legacy_with_the_shadows_off():
    cfg = load_routing()
    for k in EPISODE_SPECS:
        assert NATIVE[k] == (cfg[k] if k in cfg else {"default": "legacy"}), k
    assert NATIVE["place"] == cfg["place"] and NATIVE["press"] == cfg["press"] and NATIVE["close"] == cfg["close"]
    assert all(NATIVE[f"intent.{n}"] == {"default": "legacy"} for n in ("attach", "stamp", "cut", "heat", "aim", "pour"))
    assert NATIVE["goal_checker"] == cfg["goal_checker"] and NATIVE["goal_checkers_shadow"] == []
    assert native(shadow=True)["goal_checkers_shadow"] == list(cfg["goal_checkers_shadow"])
    assert set(NATIVE) == set(PARITY)


def test_overlay_routes_a_single_on_place_to_tiptop_and_a_mixed_one_to_legacy():
    tip = SimpleNamespace(name="tiptop", supports=lambda c: True)
    b = build(FakeEpisode(), routing=overlay(PARITY, ["place.on=tiptop"]))
    b.reg.backends["tiptop"] = tip
    assert b.reg.backend_for(SkillCall("place", ON_TABLE)) is tip
    assert b.reg.backend_for(SkillCall("place", PlaceArgs(jar, (Relation(Rel.ON, floor), Relation(Rel.NEXT_TO, table))))) is b.lb
    assert b.reg.backend_for(SkillCall("place", IN_CABINET)) is b.lb
    assert PARITY["place"] == {"default": "legacy"}, "a copy: the profile is untouched"
    over = overlay(PARITY, ["close.prismatic=tiptop", "press=tiptop", "intent.pour=scripted", "place.on=tiptop"])
    assert over["close"] == {"default": "legacy", "by_joint": {"prismatic": "tiptop"}}
    assert over["press"] == {"default": "tiptop"} and over["intent.pour"] == {"default": "scripted"}
    assert over["place"] == {"default": "legacy", "by_relation": {"on": "tiptop"}}
    for bad in ("push=tiptop", "place.above=tiptop", "place.on", "=tiptop"):
        with pytest.raises(ValueError):
            overlay(PARITY, [bad])


# ------------------------------------------------------------------------------------------------------ BenchHost
def test_bench_host_parse_stamps_the_sims_clock_when_asked_and_base_moved_follows_the_teleports():
    p = np.arange(61, dtype=np.float32)
    sim = SimpleNamespace(n_steps=1234, teleports=2)
    host = BenchHost(sim, sim_clock=True)
    assert host.parse({"proprio": p}, 3).step == 1234 and host.parse({"proprio": p}, 0).step == 1234
    assert sim.n_steps == 1234, "the sim's own count is never written"
    assert BenchHost(sim).parse({"proprio": p}, 3).step == 3, "the skill bench keeps the Runtime's step"
    assert host.base_moved() is False, "no teleport since construction"
    sim.teleports += 1
    assert host.base_moved() is True and host.base_moved() is False, "since the previous call"
    sim.teleports += 2
    assert host.base_moved() is True
    assert BenchHost(SimpleNamespace(n_steps=0)).base_moved() is False, "a sim without teleports"
