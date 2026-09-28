"""The connector host on the bench without Isaac Sim (WEEK4_PLAN §3.3 episode_host.py, §5.6 U0's unit twin; W4-E):
the host built over a stepping fake sim (its step_env counts n_steps and raises EpisodeOver at a chosen step, as
R1ProSim.step_env does), a self-stepping fake legacy Episode (every write steps the sim through step_env, so the
StepLedger sees it under the ep.* owner), fake providers over that Episode, and the real Runtime, DirectConnector,
EpisodeRegistry, EpisodeLegacyBackend, EpisodeNavigator, ConnectorAudit, PurityAudit, DualEpisode, PseudoPlanner
and shim. The digest is the oracle package's real state_digest over the fakes.

A clean parity episode passes every verdict; EpisodeOver mid-run is U0-exact both inside a legacy call (L = X = 0)
and inside a native wait (X = 1, L = the live run's steps); and each planted fault FAILS U0: a step in a read op,
a step in the backend's post-judge outside ep.*, a step in Runtime._finish, a step in host.build, and a step taken
with episode_open False. bench.main is run end to end on the same fakes under --runner connector.
"""

import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

import b1k.runtime.skillrun as skillrun
from b1k.connector.goals import Clock, GoalPanel, ScorerChecker, TaskInfo
from b1k.connector.observe import Percept, PerceptInfo, StepObs
from b1k.connector.types import Belief, Fact, ObjRef, Pose2
from b1k.connector.world import ProvenancePolicy
from b1k.planner.pseudo import tape as tp
from b1k.planner.pseudo.shim import ClockView
from b1k.runtime.core import Runtime
from b1k.runtime.skillrun import Services
from b1k.skills.scripted import ScriptedBackend
from omnigibson.tiptop import bench
from omnigibson.tiptop.host import episode_host
from omnigibson.tiptop.host.instruments import StepLedger
from omnigibson.tiptop.host.q1_audit import DualEpisode, PurityAudit, same
from omnigibson.tiptop.oracle import watch
from omnigibson.tiptop.scene import EpisodeOver

JAR, CAB, COUNTER, FLOOR = "jar.n.01_1", "cabinet.n.01_1", "countertop.n.01_1", "floor.n.01_1"
REFS = {JAR: ObjRef(JAR, "jar"), CAB: ObjRef(CAB, "cabinet", True), COUNTER: ObjRef(COUNTER, "countertop", True),
        FLOOR: ObjRef(FLOOR, "floor", True)}
SCOPE = sorted([JAR, CAB, COUNTER])
INSIDE = {"predicate": "inside", "args": [JAR, CAB]}
GOAL = [INSIDE]
OPTIONS = [[INSIDE]]
SPEC = SimpleNamespace(task="store_honey", instruction="put the jar in the cabinet", attempts_per_item=2, plan="transfer")
PLANNERS = {"left": (None, {"embodiment": {}})}


def B(v, source="oracle"):
    return Belief(v, source, 0)


# ------------------------------------------------------------------------------------------------------------ fakes
class FakeSim:
    """R1ProSim as the host and the ledger touch it: step_env counts n_steps while the episode is open and raises
    EpisodeOver at ``over_at`` after counting the step (scene.py:352-366); place_robot teleports and steps."""

    def __init__(self, max_steps=10**6, over_at=None):
        self.n_steps, self.max_steps, self.teleports, self.over_at = 0, max_steps, 0, over_at
        self.episode_open = True
        self.held_objects, self.bddl_names = {}, {JAR: JAR, CAB: CAB}
        self.robot, self.objects, self.last_action = None, {}, None
        self.primary_view, self.extra_views, self.last_gripper = "head", (), 1.0
        self.env = SimpleNamespace(reset=lambda: None, task=SimpleNamespace(success=True))
        self.recorders, self.send_room, self.video_caption = [], False, ""

    def hands(self):
        return dict(self.held_objects)

    def tracked_label(self, name):
        return name

    def floor_name(self):
        return FLOOR

    def step_env(self, action):
        self.last_action = action
        if self.episode_open:
            self.n_steps += 1
            if self.over_at is not None and self.n_steps >= self.over_at:
                raise EpisodeOver("timeout", self.n_steps)
        return {}

    def step(self, k=1):
        for _ in range(k):
            self.step_env({"r": k})

    def place_robot(self, x, y, yaw, note=None):
        self.teleports += 1
        self.step(2)

    def commanded_targets(self):
        return {}

    # what bench.main touches
    def task_scope(self):
        return set(SCOPE)

    def reset_embodiment(self, embodiment):
        pass

    def begin_episode(self, metrics, stop_when_done=True, max_steps=None, name=""):
        self.n_steps, self.episode_open, self.max_steps = 0, True, max_steps or self.max_steps

    def end_episode(self):
        self.episode_open = False
        return self.n_steps

    def mark_goal_initial(self):
        pass

    def goal_status(self):
        return {"satisfied": [], "unsatisfied": [], "total": 1}

    def hold(self, n, gripper=1.0):
        self.step(n)


class FakeEpisode:
    """bench.Episode's members over the FakeSim: the reads answer the sim's record and the fake's own state; the
    writes step the sim through step_env (a self-stepping legacy backend) and keep the record. ``fault``:
    "closed_step" makes dwell step once with the episode closed."""

    def __init__(self, sim, ok=True, fault=None):
        self.sim, self.ok, self.fault = sim, ok, fault
        self.records, self.knowledge = [], SimpleNamespace(seen={})
        self.floor, self.planners, self.spec = FLOOR, {"left": None}, SPEC
        self.shut, self.placed, self.pose = {CAB: True}, False, (0, 0, 0)
        self.refuse = None  # (names, exception): stand_for raises it for those names (Episode.stand_for's Unreachable)

    # -- reads
    def holding(self, bddl):
        return self.sim.tracked_label(bddl) in self.sim.hands()

    def held_names(self):
        return [self.sim.bddl_names[label] for label in self.sim.hands() if label in self.sim.bddl_names]

    def is_shut(self, name):
        return bool(self.shut.get(name))

    def is_floor(self, name):
        return bool(name) and name.startswith("floor.")

    def support_of(self, bddl):
        if bddl == JAR:
            return None if self.holding(JAR) else (CAB if self.placed else COUNTER)
        return self.floor

    def distance(self, a, b):
        if "unseen" in (a, b):
            raise KeyError(a)
        return 0.5

    def edge_gap(self, item, support):
        return float("inf") if support is None else 0.1

    def switched_on(self, name):
        return None

    def goal_already_holds(self, predicate, *args):
        return predicate == "inside" and tuple(args) == (JAR, CAB) and self.placed

    def stance_key(self):
        return self.pose

    def has_arm(self, arm):
        return arm in self.planners

    def after_transition(self):
        return []

    def fixture_for(self, ability, near=None):
        return None

    # -- writes: the sim stepped through step_env
    def _did(self, steps, record):
        self.sim.step(steps)
        self.records.append({**record, "step": self.sim.n_steps})
        return self.ok

    def pick(self, bddl, into=None, single_round=False):
        ok = self._did(40, {"round": 0, "pick": bddl, "into": into})
        if ok:
            self.sim.held_objects[self.sim.tracked_label(bddl)] = "left"
        return ok

    def achieve(self, atoms, arm="left", floor=None, done=None):
        ok = self._did(30, {"round": 1, "atoms": atoms})
        if ok and atoms and atoms[0]["predicate"] in ("ontop", "inside", "nextto"):
            self.sim.held_objects.pop(atoms[0]["args"][0], None)
            self.placed = atoms[0]["predicate"] == "inside"
        return ok

    def put_down(self, bddl, support, floor=None):
        placed = self.achieve([{"predicate": "ontop", "args": [bddl, support]}], floor=floor)
        if not placed and self.is_floor(support):
            self.__dict__.setdefault("floor_failed_at", set()).add(self.stance_key())
        return placed

    def open_up(self, name, fraction=None, single_round=False, joint=None):
        ok = self._did(20, {"open": name, "why": ""})
        if ok:
            self.shut[name] = fraction == 0.0
        return ok

    def release(self, steps=45):
        self.sim.step(steps)
        self.sim.held_objects.clear()
        self.records.append({"release": True, "step": self.sim.n_steps})

    def pour(self, item, target):
        return self._did(25, {"pour": item})

    def dwell(self, steps):
        left = self.sim.max_steps - self.sim.n_steps if self.sim.max_steps else int(steps)
        steps = max(0, min(int(steps), left))
        if self.fault == "closed_step":  # a step taken while the episode is closed: never counted, never owned
            self.sim.episode_open = False
            self.sim.step_env({"r": "closed"})
            self.sim.episode_open = True
        self.sim.step(steps)
        return steps

    def stand_for(self, *names):
        if self.refuse is not None and names == self.refuse[0]:
            raise self.refuse[1]
        self.sim.place_robot(1.0, 0.0, 0.0, note=f"stand_for {names}")
        self.pose = (10, 0, 0)
        return {"stance": names}

    def walk_to_floor(self, name):
        self.stand_for(name)
        return True


class FakeWorld:
    """The providers' WorldView over the FakeEpisode (what OracleWorld(hands="episode") answers from the sim's
    record and the Episode's own reads). ``fault``: "step_in_read" steps the sim inside support_of, "write_in_read"
    writes the hand record inside switched_on (and undoes it), "wrong_holding" answers no arm ever."""

    def __init__(self, ep, sim, fault=None):
        self.ep, self.sim, self.fault, self.updates, self.disagreements = ep, sim, fault, [], 0

    def tick(self, obs):
        pass

    def apply(self, u):
        """As OracleWorld(hands="episode") applies a held / released update: the sim's record, written idempotently
        (a write-call the Runtime makes with no owner of its own). ``fault``: "step_in_apply" steps the sim here,
        "apply_writes" leaves a lasting entry in the record."""
        self.updates.append(u)
        if self.fault == "step_in_apply":
            self.sim.step(1)
        if self.fault == "apply_writes":
            self.sim.held_objects["ghost"] = "right"
        if u.arm is None:
            return
        if u.kind == "held" and u.obj is not None:
            self.sim.held_objects[u.obj.id] = u.arm
        elif u.kind == "released" and u.obj is not None:
            self.sim.held_objects.pop(u.obj.id, None)

    def objects(self):
        return [REFS[n] for n in SCOPE]

    def box(self, o):
        return B(None)

    def held(self, arm):
        return B(tuple(REFS.get(n, ObjRef(n, "x")) for n, a in self.sim.hands().items() if a == arm))

    def holding(self, o):
        if self.fault == "wrong_holding":
            return B(())
        return B(tuple(a for a in ("left", "right") if self.sim.hands().get(o.id) == a))

    def is_open(self, o, joint=None):
        return B(False if self.ep.is_shut(o.id) else None)

    def support_of(self, o):
        if self.fault == "step_in_read":
            self.sim.step(1)
        v = self.ep.support_of(o.id)
        return B(None if v is None else REFS.get(v, ObjRef(v, "x")))

    def distance(self, a, b):
        try:
            return B(self.ep.distance(a.id, b.id))
        except KeyError:
            return B(None)

    def edge_gap(self, item, support):
        return B(self.ep.edge_gap(item.id, None if support is None else support.id))

    def switched_on(self, o):
        if self.fault == "write_in_read":
            self.sim.held_objects["ghost"] = "right"
            self.sim.held_objects.pop("ghost")
        return B(self.ep.switched_on(o.id))

    def fixture_for(self, ability, near=None):
        return B(None)

    def appeared(self):
        return []

    def base_pose(self):
        k = self.ep.stance_key()
        return B(None if k is None else Pose2(k[0] * 0.1, k[1] * 0.1, k[2] * (np.pi / 12)))


class FakeScorer:
    def __init__(self, ep):
        self.ep = ep

    def holds(self, fact):
        if fact.pred == "holding":
            return self.ep.holding(fact.args[0]) and self.ep.sim.hands().get(fact.args[0]) == fact.args[1]
        if fact.pred == "lifted":
            return self.ep.holding(fact.args[0])
        if fact.pred == "hand_empty":
            return not any(a == fact.args[0] for a in self.ep.sim.hands().values())
        if fact.pred == "open":
            return not self.ep.is_shut(fact.args[0])
        if fact.pred in ("inside", "ontop", "nextto"):
            return self.ep.goal_already_holds(fact.pred, *fact.args)
        return None


class NoPlanner:
    def __getattr__(self, name):
        raise AssertionError(f"planner.{name} was asked on a parity run")


class NoNative:
    name = "tiptop"

    def supports(self, call):
        return True

    def check(self, call, svc):
        raise AssertionError(f"{call.skill} reached the tiptop backend")

    def run(self, call, svc, obs):
        raise AssertionError(f"{call.skill} reached the tiptop backend")


class FakeProviders:
    """The providers package as the host loads it: pseudo_services over the Episode and task_info. ``fault``
    "step_in_build" steps the sim while the services are built (inside host.build)."""

    def __init__(self, fault=None):
        self.fault, self.calls = fault, []

    def pseudo_services(self, ep, planner, routing, collision="map", hands="sensor", scorer_scope_only=False,
                        shadows=None):
        self.calls.append({"collision": collision, "hands": hands, "scope_only": scorer_scope_only,
                           "shadows": shadows, "goal_checker": routing.get("goal_checker")})
        sim = ep.sim
        if self.fault == "step_in_build":
            sim.step(1)
        world = FakeWorld(ep, sim, self.fault)
        svc = Services(world=world, map=None, geometry=None, articulation=None, joints=None, collision=None,
                       grasp=SimpleNamespace(held=lambda arm, obs: Belief(None, "proprio", 0)),
                       buttons=SimpleNamespace(button=lambda o: Belief(None, "map", 0)),
                       goals=GoalPanel({"scorer": ScorerChecker(FakeScorer(ep))}, "scorer"), planner=planner,
                       percepts={}, provenance=ProvenancePolicy("pseudo"),
                       clock=lambda: Clock(sim.n_steps, sim.max_steps, 559 * sim.teleports), epochs=lambda: (0, 0))
        return svc, None

    def task_info(self, sim, planners, max_steps, name):
        options = tuple(tuple(Fact(a["predicate"], tuple(a["args"])) for a in opt) for opt in OPTIONS)
        return TaskInfo(1, name, options, tuple(REFS[n] for n in SCOPE), REFS[FLOOR], tuple(planners), max_steps,
                        "oracle")


class FakeHost:
    """BenchHost's seat: env_step steps the sim through step_env (DirectConnector's env step), a StepObs stamped
    with sim.n_steps (sim_clock), base_moved off the teleports."""

    def __init__(self, sim, segmenter=None):
        self.sim, self.segmenter, self.env_wall_s, self.frames_wall_s = sim, segmenter, 0.0, 0.0
        self._teleports = sim.teleports

    def raw(self):
        return {"proprio": np.zeros(61, dtype=np.float32)}

    def env_step(self, a):
        self.sim.step_env({"r": a})
        return self.raw()

    def parse(self, raw, step):
        return StepObs(self.sim.n_steps, raw["proprio"], raw)

    def observe_now(self):
        return self.parse(self.raw(), 0)

    def commanded_targets(self):
        return {}

    def base_moved(self):
        t = self.sim.teleports
        moved, self._teleports = t != self._teleports, t
        return moved


class FakeObserver:
    requires_sim_clock = True

    def __init__(self, sim):
        self.sim, self.steps, self.wall_s, self.views = sim, 0, 0.0, ("head",)

    def observe(self, req, obs):
        self.sim.step(3)
        self.steps += 3
        info = PerceptInfo("", self.sim.n_steps, 0, 0, {o.id: 1.0 for o in req.targets}, "oracle")
        return Percept(info, {}, {}, None), obs
        yield


class FakeTeleport:
    requires_sim_clock = True

    def __init__(self):
        self.steps, self.gone = 0, []

    def base_pose(self):
        return B(Pose2(0.0, 0.0, 0.0))

    def propose(self, req, k=8):
        return []

    def go_to(self, stance, obs):
        self.gone.append(stance.key)
        from b1k.connector.skills import NavResult

        return NavResult(False, stance, 0, 0, "the fake teleport refuses"), obs
        yield

    def apply(self, u):
        pass


class FakeRunner:
    """strategies.Runner as the host builds it (its construction inputs) and as it drives an Episode: the reads
    the shim serves, the writes and the navigation, in one fixed order."""

    def __init__(self, spec, goal, options=None, attempts=None, scope=()):
        self.spec, self.goal = spec, list(goal)
        self.options = [list(o) for o in options] if options else [list(goal)]
        self.attempts = spec.attempts_per_item if attempts is None else int(attempts)
        self.scope, self.seen = list(scope), []

    def run(self, ep):
        s = self.seen
        s.append(("floor", ep.floor))
        s.append(("clock", ep.sim.max_steps, ep.sim.n_steps))
        if ep.is_shut(CAB):
            ep.stand_for(CAB)
            s.append(("open", ep.open_up(CAB, fraction=0.8)))
        ep.stand_for(JAR)
        s.append(("pick", ep.pick(JAR, into=CAB)))
        s.append(("holding", ep.holding(JAR)))
        s.append(("held", ep.held_names()))
        s.append(("support", ep.support_of(JAR)))
        try:
            ep.distance("unseen", JAR)
        except KeyError:
            s.append(("distance", "KeyError"))
        s.append(("gap", ep.edge_gap(JAR, CAB)))
        s.append(("switched", ep.switched_on(CAB)))
        s.append(("stance", ep.stance_key()))
        s.append(("floor?", ep.is_floor(FLOOR)))
        s.append(("arm", ep.has_arm("right")))
        ep.stand_for(CAB)
        s.append(("place", ep.achieve([INSIDE])))
        s.append(("holds", ep.goal_already_holds("inside", JAR, CAB)))
        s.append(("dwell", ep.dwell(30)))
        s.append(("close", ep.open_up(CAB, fraction=0.0)))
        s.append(("clock", ep.sim.max_steps, ep.sim.n_steps))


def strategy_for(task, goal, options=None, attempts=None, scope=(), **kwargs):
    return FakeRunner(SPEC, goal, options=options, attempts=attempts, scope=scope)


def make_args(**over):
    args = SimpleNamespace(runner="connector", routing_profile="parity", route=[], audit=True, shadow=False,
                           attempts_per_item=2, views=["head"], task="put the jar in the cabinet",
                           task_name="store_honey")
    args.__dict__.update(over)
    return args


def fake_backends(ep, host, svc):
    return {"tiptop": NoNative(), "scripted": ScriptedBackend()}


def build(tmp_path, *, fault=None, over_at=None, audit=True, routes=(), tape=False, host_attempts=2, ok=True,
          wstape=None, tape_the_episode=False):
    sim = FakeSim(over_at=over_at)
    ep = FakeEpisode(sim, ok=ok, fault=fault)
    ledger = StepLedger(sim)
    ledger.install_episode(ep)
    providers = FakeProviders(fault)
    args = make_args(audit=audit, route=list(routes))
    strategy = strategy_for("store_honey", GOAL, options=OPTIONS, attempts=host_attempts, scope=SCOPE)
    recorded = tp.Tape(tp.header("store_honey", 301, "connector", "parity", 0, strategy=strategy, floor=FLOOR))
    around = (lambda inner: tp.TapeRecorder(inner, recorded, exc_classes=(), step_probe=lambda: sim.n_steps)) if tape else None
    if tape_the_episode:  # the wiring slip U0-c must see: the tape factory around the bench's Episode, not the shim
        around = lambda inner: tp.TapeRecorder(ep, recorded)  # noqa: E731
    host = episode_host.build(
        ep, sim, PLANNERS, args, providers, strategy_for, strategy, ledger, watch, tape=around, inst_dir=tmp_path,
        max_steps=sim.max_steps, planner_client=NoPlanner(), bench_host=FakeHost(sim), observer=FakeObserver(sim),
        teleport=FakeTeleport(), make_backends=fake_backends, wstape=wstape,
    )
    return SimpleNamespace(host=host, sim=sim, ep=ep, ledger=ledger, providers=providers, strategy=strategy,
                           tape=recorded)


def run(h):
    reason, raised = "strategy finished", None
    try:
        h.host.run()
    except EpisodeOver as e:
        reason, raised = e.reason, e
    return h.host.close(reason, raised), raised


def seen(h) -> list:
    """What the planner's own Runner (built by the factory from conn.task()) saw through the shim."""
    return h.host.planner.last_runner.seen


# ------------------------------------------------------------------------------------------- a clean parity episode
def test_a_clean_parity_episode_passes_every_verdict_and_streams_its_rows(tmp_path):
    h = build(tmp_path, tape=True)
    assert h.host.build_steps == 0 and not h.host.build_digest_changed
    assert h.providers.calls == [{"collision": "map", "hands": "episode", "scope_only": True, "shadows": [],
                                  "goal_checker": "scorer"}], "collision pinned to map, the episode's hands"
    block, raised = run(h)
    assert raised is None and block["reason"] == "strategy finished"
    g3 = block["g3"]
    assert g3["pass"] and block["ok"], json.dumps(block["g3"]) + json.dumps(block["u0b"]) + json.dumps(block["u0a"])
    assert (g3["dual_mismatches"], g3["purity_violations"], g3["typed_literal_mismatches"], g3["renders_outside_ep"]) == (0, 0, 0, 0)
    assert g3["runner_inputs_equal"] and block["runner_inputs"]["equal_at_run"]
    # U0-a: every call was legacy, so the Runtime stepped nothing and charged nothing
    assert block["u0a"] == {"ok": True, "identity": True, "step": 0, "idle_steps": 0,
                            "charged": {"skill": 0, "wait": 0, "go_to": 0}, "charged_sum": 0, "L": 0, "X": 0,
                            "episode_over": False}
    # U0-b: the ledger's owners are the ep.* members alone and sum to the sim's count
    rows = block["ledger"]
    assert set(rows) == {"host.build", "go_to", "apply", "ep.stand_for", "ep.open_up", "ep.pick", "ep.achieve",
                         "ep.dwell"}, sorted(rows)
    assert rows["host.build"]["steps"] == 0 and rows["ep.stand_for"]["steps"] == 6 and rows["ep.pick"]["steps"] == 40
    assert rows["go_to"]["steps"] == 0, "a legacy stance's steps are ep.stand_for's, the innermost owner"
    assert rows["apply"] == {**rows["apply"], "steps": 0, "held_writes": 2}, "the pick's held and the place's released"
    assert block["u0b"]["totals"]["owned_steps"] == h.sim.n_steps == 6 + 20 + 40 + 30 + 30 + 20
    assert block["u0b"]["totals"]["unowned_writes"] == 0
    assert all(block["u0b"]["checks"].values()), block["u0b"]["checks"]
    assert block["u0c"] == {"ok": True, "clock_view": "ClockView", "chain": ["TapeRecorder", "DualEpisode", "EpisodeOverConnector"],
                            "ends_at_shim": True, "no_episode": True}
    assert block["u0d"]["ok"] and block["u0d"]["summary"]["ops"] > 20 and block["u0d"]["summary"]["first_sim_in"] == 0
    assert block["rule2"] == {"ok": True, "rt": {"place_robot": 0, "capture": 0, "look_at": 0}, "outside": {},
                              "d21_debt": {"place_robot": 3, "capture": 0, "look_at": 0}}
    assert block["collision"] == "map" and block["hands"] == "episode" and block["scorer_scope_only"] is True
    # the differential audit compared every pure read the Runner made, and the clocks after every write
    compared = block["dual"]["summary"]["compared"]
    for member in ("is_shut", "holding", "held_names", "support_of", "distance", "edge_gap", "switched_on",
                   "goal_already_holds", "stance_key", "is_floor", "has_arm", "floor", "sim.n_steps", "sim.max_steps"):
        assert compared.get(member, 0) >= 1, (member, compared)
    assert block["dual"]["summary"]["mismatches"] == 0 and block["purity"]["summary"]["brackets"] > 30
    assert block["calls"] == {"open/legacy/succeeded/None": 1, "pick_up/legacy/succeeded/None": 1,
                              "place/legacy/succeeded/None": 1, "wait/legacy/succeeded/None": 1,
                              "close/legacy/succeeded/None": 1}
    assert block["channel_exceptions"] == {} and block["mismatches"] == []
    assert block["planner_oracle_reads"].get("scorer.holds") == 1
    assert block["requests"] is None, "no websocket tape: the requests are unknown, not failed"
    # the Runner saw the shim's answers, which were the Episode's
    assert ("holding", True) in seen(h) and ("support", None) in seen(h)  # in the hand: no support
    assert ("holds", True) in seen(h) and ("dwell", 30) in seen(h) and ("distance", "KeyError") in seen(h)
    assert seen(h)[-1] == ("clock", h.sim.max_steps, h.sim.n_steps) and h.strategy.seen == [], "the host's strategy never ran"
    # the streams
    for name in ("skill_calls.jsonl", "goal_checks.jsonl", "ledger.jsonl", "ledger_calls.jsonl", "audit.jsonl",
                 "connector.json"):
        assert (tmp_path / name).exists(), name
    calls = [json.loads(l) for l in (tmp_path / "skill_calls.jsonl").read_text().splitlines()]
    assert [c["skill"] for c in calls] == ["open", "pick_up", "place", "wait", "close"]
    assert all(c["backend"] == "legacy" and c["requires_sim_clock"] for c in calls)
    audit_rows = [json.loads(l) for l in (tmp_path / "audit.jsonl").read_text().splitlines()]
    assert {r["kind"] for r in audit_rows} == {"op", "purity", "dual"}
    ledger_rows = [json.loads(l) for l in (tmp_path / "ledger.jsonl").read_text().splitlines()]
    assert {r["owner"] for r in ledger_rows} == set(rows)
    # the tape is the Runner's: its writes are the Episode calls, with the shim behind them
    assert [w["member"] for w in h.tape.writes] == ["stand_for", "open_up", "stand_for", "pick", "stand_for", "achieve", "dwell", "open_up"]
    assert h.tape.writes[3]["step"] == [24, 64]
    assert skillrun.PASSTHROUGH == (), "restored at close"


def test_close_is_idempotent_and_the_second_call_returns_the_same_block(tmp_path):
    h = build(tmp_path, audit=False)
    block, _ = run(h)
    assert block["ok"] and block["dual"] is None and block["purity"] is None and block["audit_enabled"] is False
    assert h.host.close("again", None) is block


# ------------------------------------------------------------------------------------------------ EpisodeOver
def test_episode_over_inside_a_legacy_call_is_u0_exact_with_L_and_X_zero(tmp_path):
    """The sim ends inside ep.pick (step 50 of its 40): EpisodeOver passes through the backend (PASSTHROUGH), the
    Runtime and the connector to the host's caller; the pick's steps are the ep.pick owner's."""
    h = build(tmp_path, over_at=50)
    block, raised = run(h)
    assert isinstance(raised, EpisodeOver) and block["reason"] == "timeout" and block["raised"].startswith("EpisodeOver")
    assert block["ok"], json.dumps(block["g3"]) + json.dumps(block["u0b"]["checks"])
    a = block["u0a"]
    assert (a["step"], a["L"], a["X"], a["episode_over"], a["ok"]) == (0, 0, 0, True, True)
    assert block["live_runs"] == {"q1-2": 0}, "the pick is still live, at 0 Runtime steps"
    assert block["live_at_end"] is None
    assert block["ledger"]["ep.pick"]["steps"] == 50 - 24 and block["ledger_calls"][-1]["error"] == "EpisodeOver"
    assert h.sim.n_steps == 50 and block["u0b"]["totals"]["owned_steps"] == 50


def test_episode_over_inside_a_native_wait_counts_X_and_the_live_runs_steps(tmp_path):
    """wait routed to scripted: the Runner's dwell steps through DirectConnector's env_step, which raises EpisodeOver
    on its 4th step: rt.step 4, the run saw 3 observations (L = 3), X = 1."""
    h = build(tmp_path, over_at=6 + 20 + 40 + 30 + 4, routes=["wait=scripted"])
    block, raised = run(h)
    assert isinstance(raised, EpisodeOver)
    a = block["u0a"]
    assert (a["step"], a["idle_steps"], a["charged"].get("wait"), a["L"], a["X"], a["ok"]) == (4, 0, None, 3, 1, True), a
    assert block["over_in_env_step"] == 1 and block["live_at_end"] == {"call_id": "q1-4", "steps": 3}
    assert block["ledger"]["rt"] == {**block["ledger"]["rt"], "steps": 4, "env_step_calls": 4}
    assert block["ok"], json.dumps(block["g3"]) + json.dumps(block["u0b"]["checks"])
    # the same arithmetic without an EpisodeOver reason is a violation: L and X must then be 0
    assert episode_host.u0a(h.host.rt, 3, 1, False)["ok"] is False
    assert episode_host.u0a(h.host.rt, 3, 1, True)["ok"] is True


def test_a_native_wait_that_finishes_is_charged_and_the_ledgers_rt_row_is_the_runtimes_step(tmp_path):
    h = build(tmp_path, routes=["wait=scripted"])
    block, raised = run(h)
    assert raised is None and block["ok"], json.dumps(block["g3"]) + json.dumps(block["u0b"]["checks"])
    assert block["u0a"]["step"] == 30 and block["u0a"]["charged"]["wait"] == 30 and block["ledger"]["rt"]["steps"] == 30
    assert block["calls"]["wait/scripted/succeeded/None"] == 1 and ("dwell", 30) in seen(h)


# ------------------------------------------------------------------------------------------ the planted faults
@pytest.mark.parametrize(
    "fault, failing",
    [
        ("step_in_read", ("u0b", "u0d")),  # a 0-step op (world.support_of) stepped: unowned, and the audit saw it
        ("step_in_post_judge", ("u0b",)),  # the backend's post-judge stepped outside ep.*: unowned
        ("step_in_finish", ("u0b",)),  # Runtime._finish stepped: unowned
        ("step_in_build", ("u0b", "build")),  # host.build stepped
        ("closed_step", ("u0b",)),  # a step with episode_open False
        ("step_in_apply", ("u0b",)),  # a step inside Runtime._finish's world.apply: owned by "apply", never 0
    ],
)
def test_each_planted_fault_fails_u0(tmp_path, monkeypatch, fault, failing):
    h = build(tmp_path, fault=fault if fault in ("step_in_read", "step_in_build", "closed_step", "step_in_apply")
              else None)
    if fault == "step_in_post_judge":
        judge = h.host.rt.svc.goals.judge  # the skills' panel: what EpisodeLegacyBackend judges with, not the planner's

        def stepping_judge(goal, obs=None, percept=None, who="", shadows=True):
            h.sim.step(1)
            return judge(goal, obs, percept, who=who, shadows=shadows)

        monkeypatch.setattr(h.host.rt.svc.goals, "judge", stepping_judge)
    if fault == "step_in_finish":
        finish = Runtime._finish

        def stepping_finish(rt, cid, run):
            h.sim.step(1)
            return finish(rt, cid, run)

        monkeypatch.setattr(Runtime, "_finish", stepping_finish)
    block, raised = run(h)
    assert raised is None
    assert not block["ok"] and not block["g3"]["pass"], fault
    for verdict in failing:
        assert block["g3"][verdict] is False, (fault, verdict, block["g3"])
    if fault == "step_in_build":
        assert block["ledger"]["host.build"]["steps"] == 1 and block["u0b"]["checks"]["host_build_zero"] is False
    elif fault == "closed_step":
        assert block["u0b"]["checks"]["closed_zero"] is False and block["u0b"]["totals"]["closed_env_step_calls"] == 1
    elif fault == "step_in_apply":
        assert block["ledger"]["apply"]["steps"] >= 1 and block["u0b"]["checks"]["apply_zero"] is False
        assert block["u0b"]["checks"]["unowned_steps"] is True, "owned by apply: only apply_zero can see it"
    else:
        assert block["u0b"]["checks"]["unowned_steps"] is False and block["u0b"]["totals"]["unowned_steps"] >= 1
    if fault == "step_in_read":
        assert any("world.support_of" in v for v in block["u0d"]["violations"]), block["u0d"]
        assert block["purity"]["summary"]["digest_changes"] >= 1, "the digest moved across a read"


def test_an_apply_that_changes_the_hand_record_on_a_parity_run_fails_g3(tmp_path):
    """Under PARITY every result's hand updates repeat what the Episode already wrote: an apply that changes the
    record is a write the connector added."""
    h = build(tmp_path, fault="apply_writes")
    block, raised = run(h)
    assert raised is None and block["g3"]["apply_hand_changes"] >= 1 and not block["g3"]["pass"]
    assert block["apply_hand_changes"][0]["after"].get("ghost") == "right"


@pytest.mark.parametrize("owner", ["unowned", "rt", "apply", "refresh"])
def test_a_capture_owned_by_anything_but_the_episode_or_the_planners_services_fails_rule2(tmp_path, owner):
    """A native skill's own code runs unowned (DirectConnector starts and resumes it outside env_step's ``rt``):
    rule 2 reads every owner but ep.*, observe and go_to."""
    h = build(tmp_path, audit=False, routes=("wait=scripted",))
    h.ledger.row(owner).capture = 1
    block, _ = run(h)
    assert block["rule2"]["ok"] is False and block["rule2"]["outside"] == {owner: {"capture": 1}}
    assert not block["g3"]["pass"]
    h = build(tmp_path, audit=False, routes=("wait=scripted",))
    h.ledger.row("observe").capture, h.ledger.row("ep.pick").capture = 1, 1  # the planner's and legacy's own
    block, _ = run(h)
    assert block["rule2"]["ok"] is True and block["rule2"]["d21_debt"]["capture"] == 1


def test_a_render_by_a_skills_code_fails_a_native_run_and_the_planners_capture_does_not(tmp_path):
    h = build(tmp_path, audit=False, routes=("wait=scripted",))
    h.ledger.row("observe").renders = 40  # the planner's capture: reported, never a failure
    block, _ = run(h)
    assert block["g3"]["parity_run"] is False and block["g3"]["renders_by_skills"] == 0 and block["ok"], block["g3"]
    h = build(tmp_path, audit=False, routes=("wait=scripted",))
    h.ledger.row("unowned").renders = 3  # a native skill grabbing a frame
    block, _ = run(h)
    assert block["g3"]["renders_by_skills"] == 3 and not block["ok"]


def test_the_requests_are_this_episodes_frames_alone_and_a_plan_request_is_the_episodes(tmp_path):
    frames = [{"op": "skill", "owner": "unowned"}, {"op": "metadata", "owner": "unowned"}]  # an earlier instance's
    ws = SimpleNamespace(frames=frames, mode="log")
    h = build(tmp_path, wstape=ws)
    assert h.host.frames0 == 2
    frames.append({"op": "plan", "owner": "ep.pick"})
    block, _ = run(h)
    assert block["requests"]["by_type_and_owner"] == {"plan/ep.pick": 1} and block["requests"]["ok"], block["requests"]
    assert block["ok"], block["g3"]
    for bad in ({"op": "skill", "owner": "unowned"}, {"op": "plan", "owner": "unowned"}):
        frames = [{"op": "metadata", "owner": "unowned"}]
        ws = SimpleNamespace(frames=frames, mode="log")
        h = build(tmp_path, wstape=ws)
        frames.append(bad)
        block, _ = run(h)
        assert not block["g3"]["pass"], bad
    assert block["requests"]["plan_move_outside_ep"] == 1


def test_a_runner_that_holds_the_episode_itself_fails_u0c_and_the_connector_floors(tmp_path):
    """The tape factory wrapped around the bench's Episode (the legacy wiring) instead of what the host hands it:
    the Runner never goes through the connector, which U0-c and the audited parity run's floors must see."""
    h = build(tmp_path, tape=True, tape_the_episode=True)
    block, raised = run(h)
    assert raised is None
    assert block["u0c"]["ok"] is False and block["u0c"]["ends_at_shim"] is False and block["u0c"]["no_episode"] is False
    assert block["g3"]["shim_calls"] == 0 and not block["g3"]["pass"]


def test_the_dual_episode_compares_an_exceptions_message_too():
    class Raises:
        def __init__(self, msg):
            self.msg = msg

        def distance(self, a, b):
            raise KeyError(self.msg)

    d = DualEpisode(Raises("floor.n.01_2"), Raises("no distance between floor.n.01_2 and x"))
    with pytest.raises(KeyError):
        d.distance("floor.n.01_2", "x")
    assert len(d.mismatches) == 1, "the same type with another message is not the same answer"
    d = DualEpisode(Raises("floor.n.01_2"), Raises("floor.n.01_2"))
    with pytest.raises(KeyError):
        d.distance("floor.n.01_2", "x")
    assert d.mismatches == []


def test_the_purity_audit_lets_appeared_and_fixture_for_track_what_they_name_and_nothing_else():
    state = {"objects": 0, "robot": 0}

    class World:
        def appeared(self):
            state["objects"] += 1
            return []

        def fixture_for(self, ability, near=None):
            state["objects"] += 1
            return None

        def support_of(self, o):
            state["objects"] += 1
            return None

    conn = SimpleNamespace(world=lambda: World())
    audit = PurityAudit(conn, lambda: dict(state), lambda: 0)
    audit.world().appeared()
    audit.world().fixture_for("heatSource")
    assert audit.violations == [] and [r.get("stateful") for r in audit.rows] == [True, True]
    audit.world().support_of(None)
    assert len(audit.violations) == 1 and "world.support_of" in audit.violations[0]


def test_the_capture_observers_steps_are_counted_when_the_episode_ends_inside_its_capture():
    from b1k.connector.observe import ObserveRequest
    from omnigibson.tiptop.host.capture_observer import CaptureObserver

    class Sim:
        primary_view, extra_views, n_steps = "head", (), 100

        def look_at(self, *names):
            self.n_steps += 2

        def capture(self, task):
            self.n_steps += 5
            raise EpisodeOver("timeout", self.n_steps)

    sim = Sim()
    obs = CaptureObserver(sim, None, None, "task")
    with pytest.raises(EpisodeOver):
        next(obs.observe(ObserveRequest((REFS[JAR],), views=("head",), aim=True), None))
    assert obs.steps == 7, "the look and the capture's settle stepped the sim before EpisodeOver"


def test_a_step_in_a_read_op_without_the_audit_still_fails_u0(tmp_path):
    h = build(tmp_path, fault="step_in_read", audit=False)
    block, _ = run(h)
    assert not block["ok"] and block["g3"]["u0d"] is False and block["g3"]["u0b"] is False


# --------------------------------------------------------------------------------------- the G3 items, one by one
def test_runner_inputs_that_differ_from_the_hosts_strategy_are_flagged_not_raised(tmp_path):
    h = build(tmp_path, host_attempts=3)
    block, raised = run(h)
    assert raised is None
    assert block["runner_inputs"]["equal"] is False and block["g3"]["runner_inputs_equal"] is False
    assert block["runner_inputs"]["planner"]["attempts"] == 2 and block["runner_inputs"]["host"]["attempts"] == 3
    assert not block["ok"] and block["g3"]["u0a"] and block["g3"]["u0b"], "U0 itself still holds"


def test_the_dual_episode_flags_a_shim_answer_that_differs_from_the_episodes(tmp_path):
    h = build(tmp_path, fault="wrong_holding")
    block, _ = run(h)
    d = block["dual"]
    assert d["summary"]["mismatched"] == {"holding": 1} and d["mismatches"][0]["member"] == "holding"
    assert (d["mismatches"][0]["shim"], d["mismatches"][0]["episode"]) == (False, True)
    assert not block["ok"] and block["g3"]["dual_mismatches"] == 1
    assert ("holding", False) in seen(h), "the Runner saw the shim's answer, never a substitute"


def test_the_purity_audit_flags_a_write_call_inside_a_read_op(tmp_path):
    h = build(tmp_path, fault="write_in_read")
    block, _ = run(h)
    p = block["purity"]
    assert p["summary"]["writes_in_reads"] == 2 and any("world.switched_on" in v for v in p["violations"]), p
    assert p["summary"]["unowned_writes"] == 2 and block["u0b"]["checks"]["unowned_writes"] is False
    assert not block["ok"] and block["g3"]["purity_violations"] >= 1


def test_the_purity_audit_brackets_the_episode_side_of_every_dual_read(tmp_path):
    h = build(tmp_path)
    block, _ = run(h)
    ops = [r["op"] for r in h.host.purity.rows]
    assert "dual.is_shut" in ops and "dual.holding" in ops and "dual.sim.n_steps" in ops and "world.holding" in ops
    assert h.host.purity.rows[0]["digest_changed"] == [] and all(r["writes"] == 0 for r in h.host.purity.rows)


def test_close_never_raises(tmp_path, monkeypatch):
    h = build(tmp_path)
    monkeypatch.setattr(h.host.audit, "verdict", lambda n0=None: (_ for _ in ()).throw(RuntimeError("boom")))
    block = h.host.close("strategy finished", None)
    assert block["ok"] is False and block["error"] == "RuntimeError: boom" and block["collision"] == "map"
    assert (tmp_path / "connector.json").exists()
    h2 = build(tmp_path / "unrun")
    block2 = h2.host.close(None, None)  # closed before the planner ever ran: no Runner, no shim
    assert block2["ok"] is False and block2["u0c"]["ok"] is False and block2["runner_inputs"]["equal"] is False
    assert block2["u0a"]["ok"] and block2["u0b"]["ok"], "nothing ran: the step arithmetic holds trivially"


def test_a_transfer_blocked_or_unreachable_comes_back_through_the_channel_as_the_same_object(tmp_path):
    from b1k.bridge.strategies import Unreachable

    h = build(tmp_path, audit=False)
    boom = Unreachable("no stance reaches the cabinet")
    h.ep.refuse = ((CAB,), boom)
    with pytest.raises(Unreachable) as info:
        h.host.run()
    assert info.value is boom
    block = h.host.close("crash: Unreachable", info.value)
    assert block["channel_exceptions"] == {"Unreachable": 1} and block["u0a"]["ok"] and block["u0b"]["ok"]


# ------------------------------------------------------------------------------------------------- q1_audit
def test_same_compares_floats_sequences_and_sets_by_value():
    assert same(float("inf"), float("inf")) and same(float("nan"), float("nan")) and not same(0.1, 0.2)
    assert same([1, "a", 0.5], (1, "a", 0.5)) and not same([1], [1, 2]) and same({(1, 0)}, frozenset({(1, 0)}))
    assert same(None, None) and not same(None, False) and same("x", "x")


def test_the_dual_episode_exposes_the_shims_members_alone_and_never_the_episodes(tmp_path):
    h = build(tmp_path)
    shim = SimpleNamespace(holding=lambda b: True, sim=ClockView(SimpleNamespace(clock=lambda: Clock(3, 9, 0))))
    dual = DualEpisode(shim, h.ep)
    assert dual.holding(JAR) is True and dual.summary()["mismatched"] == {"holding": 1}, "the fake Episode holds nothing"
    assert not hasattr(dual, "is_shut"), "the Episode has is_shut; the shim does not, so the Runner must not see it"
    assert isinstance(dual.sim, ClockView) and dual.sim.n_steps == 3
    with pytest.raises(AttributeError):
        dual.floor = FLOOR


def test_the_purity_audit_reports_the_digest_keys_that_moved():
    state = {"n_steps": 0, "held": ()}
    conn = SimpleNamespace(clock=lambda: state.__setitem__("n_steps", state["n_steps"] + 1), run=lambda call: "ran")
    audit = PurityAudit(conn, lambda: dict(state), lambda: 0)
    audit.run("call")  # a stepping op: never bracketed
    audit.clock()  # a 0-step op that moved the clock
    s, v = audit.verdict()
    assert s == {"brackets": 1, "digest_changes": 1, "writes_in_reads": 0, "unowned_writes": 0, "violations": 1}
    assert v == ["clock: digest changed at ['n_steps'] ([('n_steps', 0, 1)])"]


# --------------------------------------------------------------------------------------------------- bench.main
def pseudo_services(ep, planner, routing, collision="map", hands="sensor", scorer_scope_only=False, shadows=None):
    """This module as bench.main's --providers: the fakes above over the Episode it is given."""
    return FakeProviders().pseudo_services(ep, planner, routing, collision, hands, scorer_scope_only, shadows)


def task_info(sim, planners, max_steps, name):
    return FakeProviders().task_info(sim, planners, max_steps, name)


class MainEpisode(FakeEpisode):
    def __init__(self, sim, args, planners, knowledge, out_dir, spec=None):
        super().__init__(sim)
        self.knowledge, self.planners = knowledge, planners


def run_bench_main(tmp_path, monkeypatch, extra):
    import omnigibson.eval.evaluator as evaluator
    import omnigibson.eval.utils.score_utils as score_utils
    import omnigibson.metrics as metrics
    import omnigibson.tiptop.knowledge as knowledge_mod
    from b1k.bridge import strategies
    from omnigibson.tiptop.host import skillbench

    sims = []

    class Metric:
        def __init__(self, human):
            pass

        def step(self, *a):
            pass

        def aggregate(self, env):
            return {"q_score": {"final": 1.0}}

    def build_sim(args, embodiment, max_steps):
        sims.append(FakeSim())
        return sims[-1]

    monkeypatch.setattr(bench, "setup_logging", lambda: None)
    monkeypatch.setattr(evaluator, "resolve_instance_ids", lambda task, instances, mode: [301 + i for i in instances])
    monkeypatch.setattr(evaluator, "load_task_instance", lambda *a, **k: None)
    monkeypatch.setattr(score_utils, "load_human_stats", lambda task: {"length": 1000})
    monkeypatch.setattr(metrics, "AgentMetric", Metric)
    monkeypatch.setattr(metrics, "TaskMetric", Metric)
    monkeypatch.setattr(knowledge_mod, "make_knowledge", lambda *a, **k: SimpleNamespace(report=lambda: {}, seen={}))
    monkeypatch.setattr(strategies, "strategy_for", strategy_for)
    monkeypatch.setattr(strategies, "task_goal_atoms", lambda sim: list(GOAL))
    monkeypatch.setattr(strategies, "task_goal_options", lambda sim: [list(o) for o in OPTIONS])
    monkeypatch.setattr(bench, "connect_planners", lambda args: (None, {"embodiment": {}}, None, None))
    monkeypatch.setattr(bench, "check_imports", lambda *m: "imports: fake")
    monkeypatch.setattr(bench, "build_r1pro_sim", build_sim)
    monkeypatch.setattr(bench, "apply_embodiment_posture", lambda sim, args, embodiment: sim.hold(3))
    monkeypatch.setattr(bench, "wants_home_torso", lambda spec: False)
    monkeypatch.setattr(bench, "Episode", MainEpisode)
    monkeypatch.setattr(skillbench, "make_backends", fake_backends)
    monkeypatch.setattr(episode_host, "BenchHost", lambda sim, segmenter, sim_clock=False: FakeHost(sim, segmenter))
    monkeypatch.setattr(episode_host, "CaptureObserver", lambda sim, host, segmenter, task: FakeObserver(sim))
    monkeypatch.setattr(episode_host, "TeleportNavigator", lambda sim: FakeTeleport())
    out = tmp_path / "out"
    with pytest.raises(SystemExit) as code:
        bench.main(["--out-dir", str(out), "--task-name", "store_honey", "--instances", "0", "--no-video",
                    "--no-state-stream", "--providers", __name__, *extra])
    assert code.value.code == 0
    return json.loads((out / "json" / "store_honey_301_0.json").read_text()), sims[0], out


def test_bench_main_runs_the_pseudo_planner_over_the_connector_and_writes_the_connector_block(tmp_path, monkeypatch):
    result, sim, out = run_bench_main(tmp_path, monkeypatch, ["--runner", "connector", "--audit", "--runner-tape"])
    assert result["bench"]["reason"] == "strategy finished" and result["success"] is True
    block = result["bench"]["connector"]
    assert block["ok"] and block["g3"]["pass"], json.dumps(block["g3"]) + json.dumps(block["u0b"]["checks"])
    assert block["u0c"]["chain"] == ["TapeRecorder", "DualEpisode", "EpisodeOverConnector"], block["u0c"]
    assert block["calls"] and block["g3"]["shim_calls"] > 0 and block["g3"]["dual_compared"], "through the connector"
    assert block["u0b"]["n0"] == 3, "the posture's steps come before the host is built"
    assert block["ledger"]["host.build"]["steps"] == 0 and block["u0b"]["totals"]["closed_env_step_calls"] == 0
    assert set(result["bench"]) >= {"reason", "max_steps", "wall_time_s", "knowledge", "collision_map", "teleports",
                                    "goal", "video", "rounds", "instruments", "connector"}, "a superset of legacy's"
    assert result["bench"]["rounds"] and result["bench"]["teleports"] == 3
    inst = result["bench"]["instruments"]
    assert inst["runner_tape"]["writes"] == 8 and inst["ledger"]["totals"]["sum_matches_sim"]
    tape = tp.Tape.load(out / "tapes" / "store_honey_301_0.json")
    assert tape.header["runner"] == "connector" and tape.header["ending"] == {"reason": "strategy finished", "raised": None}
    assert [w["member"] for w in tape.writes][:2] == ["stand_for", "open_up"]
    inst_dir = out / "store_honey_301_0"
    assert (inst_dir / "skill_calls.jsonl").exists() and (inst_dir / "audit.jsonl").exists()
    assert (inst_dir / "connector.json").exists() and (inst_dir / "gripper.jsonl").exists()
    assert type(sim.held_objects) is dict and "step_env" not in vars(sim), "the instruments finished at the end"


def test_bench_main_records_an_episode_over_from_a_legacy_call_under_the_connector(tmp_path, monkeypatch):
    original = FakeSim.__init__

    def ending(self, max_steps=10**6, over_at=None):
        original(self, max_steps, over_at=3 + 50)

    monkeypatch.setattr(FakeSim, "__init__", ending)
    result, sim, out = run_bench_main(tmp_path, monkeypatch, ["--runner", "connector"])
    assert result["bench"]["reason"] == "timeout" and result["steps"] == 53
    block = result["bench"]["connector"]
    assert block["ok"] and block["u0a"]["episode_over"] and block["reason"] == "timeout"


def test_the_flags_accept_the_connector_and_refuse_an_audit_outside_parity(capsys):
    minimal = ["--out-dir", "/tmp/x", "--task-name", "store_honey"]
    args = bench.parse_args(minimal + ["--runner", "connector"])
    assert args.runner == "connector" and bench.instrumented(args)
    with pytest.raises(SystemExit):
        bench.parse_args(minimal + ["--runner", "connector", "--audit", "--routing-profile", "native"])
    assert "parity only" in capsys.readouterr().err
