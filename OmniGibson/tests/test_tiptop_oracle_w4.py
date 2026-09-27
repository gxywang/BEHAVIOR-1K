"""W4-O: the oracle providers for the episode host (WEEK4_PLAN 3.4), over stubs of the scene: the episode-mode hands
(the robot's own record, what Episode.holding and held_names read), apply's idempotent writes to it, the GraspSensor
compared and never obeyed, is_open from the JointStateEstimator, refresh_hands over run.note_hands, the clock's
teleport shadow, the scope-only scorer, task_info over task_goal_options, and the skill bench's task_info
pass-through."""

import logging
from types import SimpleNamespace

import numpy as np
import pytest

from b1k.connector.skills import PickArgs, SkillCall, Status, WorldUpdate
from b1k.connector.types import Fact, ObjRef, Provided
from b1k.connector.world import ProvenancePolicy, guarded
from b1k.runtime.direct import DirectConnector
from omnigibson.tiptop.bench import Episode
from omnigibson.tiptop.oracle import world as oracle_world
from omnigibson.tiptop.oracle.world import ARMS, OracleWorld


# ------------------------------------------------------------------------------------------------- the hand record
class RecordSim:
    """The hand record as R1ProSim keeps it: held_objects {tracked label: arm}, bddl_names {label: BDDL name},
    hands() a copy of the record, tracked_label the label of a tracked object and the name itself otherwise."""

    def __init__(self, held=()):
        self.bddl_names = {"jar_1": "jar.n.01_1", "lid_1": "lid.n.01_1", "cup_1": "cup.n.01_1"}
        self.held_objects = dict(held)
        self.n_steps, self.arm = 0, "left"

    def hands(self):
        return dict(self.held_objects)

    def tracked_label(self, name):
        return next((label for label, bddl in self.bddl_names.items() if bddl == name), name)

    def scene_object(self, name):
        return SimpleNamespace(fixed_base=False)


def episode_world(sim, grasp=None, knowledge=None):
    return OracleWorld(SimpleNamespace(sim=sim, knowledge=knowledge), grasp, hands="episode")


def test_episode_mode_hands_are_the_robots_own_record_in_its_insertion_order():
    sim = RecordSim([("jar_1", "left"), ("lid_1", "right"), ("cup_1", "left")])
    w, ep = episode_world(sim), SimpleNamespace(sim=sim)  # ep: what Episode.holding / held_names read
    left, right = w.held("left"), w.held("right")
    assert [o.id for o in left.value] == ["jar.n.01_1", "cup.n.01_1"], "the record's insertion order, BDDL names"
    assert [o.id for o in right.value] == ["lid.n.01_1"] and left.source == right.source == "oracle"
    assert sorted(o.id for a in ARMS for o in w.held(a).value) == sorted(Episode.held_names(ep))
    for bddl in ("jar.n.01_1", "lid.n.01_1", "cup.n.01_1", "bowl.n.01_1"):
        assert bool(w.holding(ObjRef(bddl, bddl.split(".")[0])).value) is Episode.holding(ep, bddl)
    assert w.holding(ObjRef("lid.n.01_1", "lid")).value == ("right",)
    assert w.holding(ObjRef("bowl.n.01_1", "bowl")).value == (), "not held: an empty tuple, never None"
    sim.held_objects.clear()
    assert w.held("left").value == () and w.held("right").value == () and Episode.held_names(ep) == []
    sim.held_objects.update([("cup_1", "left"), ("jar_1", "left")])  # one hand: the ids are held_names in its order
    assert [o.id for a in ARMS for o in w.held(a).value] == Episode.held_names(ep) == ["cup.n.01_1", "jar.n.01_1"]


def test_the_grasp_sensor_is_compared_with_the_record_and_never_obeyed():
    sim = RecordSim([("jar_1", "left")])
    sensed = {"left": False, "right": True}
    w = episode_world(sim, SimpleNamespace(held=lambda arm, obs: Provided(sensed[arm], "proprio", 0)))
    assert [o.id for o in w.held("left").value] == ["jar.n.01_1"], "the sensor's empty reading cleared the record"
    assert w.held("right").value == (), "the sensor's held reading put something in the record"
    assert w.disagreements == 2 and sim.held_objects == {"jar_1": "left"}
    sensed["left"], sensed["right"] = True, None
    w.held("left"), w.held("right")
    assert w.disagreements == 2, "an agreeing and an unknown reading count nothing"
    assert w.holding(ObjRef("jar.n.01_1", "jar")).value == ("left",)


def test_apply_writes_the_record_and_a_repeat_changes_nothing():
    sim = RecordSim()
    w = episode_world(sim)
    jar, cup, lid = (ObjRef(n, n.split(".")[0]) for n in ("jar.n.01_1", "cup.n.01_1", "lid.n.01_1"))
    w.apply(WorldUpdate("held", jar, "left"))
    w.apply(WorldUpdate("held", jar, "left"))
    assert sim.held_objects == {"jar_1": "left"}
    w.apply(WorldUpdate("held", lid, "right"))
    w.apply(WorldUpdate("held", cup, "left"))
    w.apply(WorldUpdate("held", jar, "left"))  # a repeat keeps the record's order too
    before = [("jar_1", "left"), ("lid_1", "right"), ("cup_1", "left")]
    assert list(sim.held_objects.items()) == before
    for u in (
        WorldUpdate("moved", jar, "left"),
        WorldUpdate("held", None, "left"),
        WorldUpdate("held", jar, None),
        WorldUpdate("joint", lid, joint="j", value=0.3),
        WorldUpdate("released", jar, None),
    ):
        w.apply(u)
    assert list(sim.held_objects.items()) == before, "nothing else writes it"
    w.apply(WorldUpdate("released", jar, "left"))
    w.apply(WorldUpdate("released", jar, "left"))
    assert sim.held_objects == {"lid_1": "right", "cup_1": "left"}
    w.apply(WorldUpdate("released", None, "left"))  # every entry of that arm
    w.apply(WorldUpdate("released", None, "left"))
    assert sim.held_objects == {"lid_1": "right"}
    assert w.hands == {"left": set(), "right": set()}, "the sensor mode's ledger is untouched in episode mode"


def test_the_default_is_the_sensor_mode_and_a_wrong_mode_is_refused():
    grasp = SimpleNamespace(held=lambda arm, obs: Provided(True, "proprio", 0))
    w = OracleWorld(SimpleNamespace(sim=RecordSim([("jar_1", "left")])), grasp)
    assert w.mode == "sensor" and w.held("left").value is None, "sensor mode: the record is not read"
    w.apply(WorldUpdate("held", ObjRef("jar.n.01_1", "jar"), "left"))
    assert w.sim.held_objects == {"jar_1": "left"} and w.hands["left"] == {ObjRef("jar.n.01_1", "jar")}
    with pytest.raises(ValueError):
        OracleWorld(SimpleNamespace(sim=RecordSim()), grasp, hands="record")


# ------------------------------------------------------------------------------------------- is_open (rule 6)
def test_is_open_takes_the_live_value_from_the_joint_state_estimator(monkeypatch):
    from omnigibson.tiptop.oracle.joints import OracleJoints

    q = {"j1": 0.1, "j2": 0.0}
    obj = SimpleNamespace(
        fixed_base=True, joints={n: SimpleNamespace(get_state=lambda n=n: (np.array([q[n]]), 0)) for n in q}
    )
    frames = [{"name": "j1", "lower": 0.0, "upper": 0.4, "position": 0.1, "closed": 0.0},
              {"name": "j2", "lower": 0.0, "upper": 0.4, "position": 0.0, "closed": 0.0}]  # fmt: skip
    monkeypatch.setattr(oracle_world, "openable_joints", lambda o: [dict(j) for j in frames])
    sim = SimpleNamespace(scene_object=lambda n: obj, n_steps=7)
    cab = ObjRef("cabinet.n.01_1", "cabinet", True)
    old = OracleWorld(SimpleNamespace(sim=sim), None)  # the position openable_joints read
    new = OracleWorld(SimpleNamespace(sim=sim), None, joints=OracleJoints(sim))  # the estimator's, the same value
    for j in (None, "j1", "j2"):
        assert new.is_open(cab, j).value is old.is_open(cab, j).value
        assert new.open_fraction(cab, j).value == pytest.approx(old.open_fraction(cab, j).value)
    assert (new.is_open(cab).value, new.open_fraction(cab).value) == (True, pytest.approx(0.25))
    assert new.is_open(cab).source == old.is_open(cab).source == "oracle", "the estimator's source"
    q["j1"] = 0.0  # the drawer shuts; openable_joints' frames still carry the position they were read with
    assert new.is_open(cab).value is False and new.open_fraction(cab).value == 0.0, "the estimator's value"
    assert old.is_open(cab).value is True, "the old read answers from the frames' position"
    q["j2"] = 0.4
    assert new.is_open(cab, "j2").value is True and new.open_fraction(cab).value == pytest.approx(1.0)
    est = SimpleNamespace(value=lambda o, joint: Provided(q[joint], "proprio", 7))  # a proprio estimator's source
    assert OracleWorld(SimpleNamespace(sim=sim), None, joints=est).is_open(cab).source == "proprio"
    monkeypatch.setattr(oracle_world, "openable_joints", lambda o: [])
    assert new.is_open(cab).value is None and new.is_open(cab).source == "oracle", "nothing that opens: unknown"


# ------------------------------------------------------------------------------------------- refresh_hands
class RefreshSim(RecordSim):
    """What run.note_hands with no atoms touches: localization's box against the hand at the origin, grasp_sensed,
    check_hands. ``step_on_check``: a stub that steps where it must not."""

    def __init__(self, held, step_on_check=False):
        super().__init__(held)
        self.step_on_check, self.checked = step_on_check, 0

    def grasp_sensed(self, arm=None):
        return True

    def check_hands(self):
        self.checked += 1
        if self.step_on_check:
            self.n_steps += 1

    def base_to_world(self, p):
        return np.asarray(p, dtype=np.float64)

    def eef_pose_base(self, arm):
        return np.eye(4)


BOXES = {
    "jar.n.01_1": {"lo": (5.0, 5.0, 5.0), "hi": (6.0, 6.0, 6.0)},  # far from the hand: it left
    "lid.n.01_1": {"lo": (-0.1, -0.1, -0.1), "hi": (0.1, 0.1, 0.1)},
}  # around the hand: still held
KNOWLEDGE = SimpleNamespace(localize=lambda *names: {n: BOXES[n] for n in names})


def test_refresh_hands_pops_what_localization_says_left_the_hand_and_never_steps():
    from omnigibson.tiptop import oracle

    sim = RefreshSim([("jar_1", "left"), ("lid_1", "right")])
    w = episode_world(sim, knowledge=KNOWLEDGE)
    assert w.refresh_hands() == ["jar_1"]
    assert sim.held_objects == {"lid_1": "right"} and sim.n_steps == 0 and sim.checked == 1
    assert w.refresh_hands() == [] and sim.held_objects == {"lid_1": "right"}
    policy = ProvenancePolicy("pseudo")
    assert oracle.refresh_hands(guarded(w, policy, "world")) == [], "the module-level entry, through the guard"
    stepping = RefreshSim([("jar_1", "left")], step_on_check=True)
    with pytest.raises(RuntimeError, match="stepped the sim"):
        episode_world(stepping, knowledge=KNOWLEDGE).refresh_hands()


# ------------------------------------------------------------------------------- pseudo_services: clock, hands, scorer
def scene_sim(teleports=0):
    cab = SimpleNamespace(fixed_base=True, category="cabinet", name="cab_1")
    return SimpleNamespace(
        n_steps=9,
        max_steps=1000,
        teleports=teleports,
        scene_object=lambda n: cab,
        robot=None,
        task_scope=lambda: {},
        held_objects={},
        bddl_names={},
        env=SimpleNamespace(
            scene=SimpleNamespace(objects=[]), task=SimpleNamespace(object_scope={"floor.n.01_1": None})
        ),
    )


def test_the_clocks_shadow_is_the_teleports_at_the_human_move_to_cost():
    from omnigibson.tiptop.host.teleport_nav import MOVE_TO_STEPS
    from omnigibson.tiptop.oracle import pseudo_services

    sim = scene_sim(teleports=3)
    ep = SimpleNamespace(sim=sim, is_floor=lambda n: n.startswith("floor."))
    svc, _ = pseudo_services(ep, "planner", {"goal_checker": "scorer"})
    c = svc.clock()
    assert (c.step, c.max_steps, c.shadow_steps) == (9, 1000, 3 * 559) and c.left() == 1000 - 9 - 1677
    sim.teleports, sim.n_steps = 4, 20
    assert (svc.clock().step, svc.clock().shadow_steps) == (20, 4 * MOVE_TO_STEPS) == (20, 2236), "read live"


def test_pseudo_services_builds_the_episode_hosts_world_scorer_and_shadows():
    from omnigibson.tiptop.oracle import pseudo_services
    from omnigibson.tiptop.oracle.joints import OracleJoints

    ep = SimpleNamespace(sim=scene_sim(), is_floor=lambda n: n.startswith("floor."))
    routing = {"goal_checker": "scorer", "goal_checkers_shadow": ["perception"]}
    svc, _ = pseudo_services(ep, "planner", routing)
    assert svc.world.mode == "sensor" and svc.goals.checkers["scorer"].scorer.scope_only is False, "the bench's"
    assert list(svc.goals.checkers) == ["scorer", "perception"] and isinstance(svc.world.joints, OracleJoints)
    host, _ = pseudo_services(ep, "planner", routing, hands="episode", scorer_scope_only=True, shadows=[])
    assert host.world.mode == "episode" and host.goals.checkers["scorer"].scorer.scope_only is True
    assert list(host.goals.checkers) == ["scorer"], "shadows overrides routing's list"
    assert routing["goal_checkers_shadow"] == ["perception"], "routing itself is untouched"
    vlm, _ = pseudo_services(ep, "planner", {"goal_checker": "scorer"}, shadows=["vlm"])
    assert list(vlm.goals.checkers) == ["scorer", "vlm"]


def test_scope_only_answers_none_for_a_relation_on_a_name_the_task_does_not_scope():
    from omnigibson.object_states import Inside, NextTo, OnTop
    from omnigibson.tiptop.oracle.goals import EpisodeScorer

    scope, burner = ("apple.n.01_1", "table.n.02_1"), object()
    apple_obj = SimpleNamespace(
        states={s: SimpleNamespace(get_value=lambda other: other is burner) for s in (OnTop, NextTo, Inside)}
    )

    def holds(pred, *names):  # R1ProSim.holds: the task's evaluator raises KeyError on a name outside its scope
        for n in names:
            if n not in scope:
                raise KeyError(n)
        return True

    sim = SimpleNamespace(
        robot=None,
        holds=holds,
        env=SimpleNamespace(task=SimpleNamespace(object_scope=scope)),
        scene_object=lambda n: {"apple.n.01_1": apple_obj, "burner_mdanhg_0": burner}.get(n, object()),
    )
    today, scoped = EpisodeScorer(sim), EpisodeScorer(sim, scope_only=True)
    assert today.scope_only is False
    for pred in ("ontop", "nextto", "inside"):
        f = Fact(pred, ("apple.n.01_1", "burner_mdanhg_0"))
        assert today.holds(f) is True, "the bench's answer: the state BDDL's predicate reads"
        assert scoped.holds(f) is None, "scope only: sim.holds raises KeyError, which is None (legacy's False)"
        assert scoped.holds(Fact(pred, ("apple.n.01_1", "table.n.02_1"))) is True, "a scoped name: the evaluator"
    assert (
        scoped.holds(Fact("open", ("burner_mdanhg_0",))) is None
        and today.holds(Fact("open", ("burner_mdanhg_0",))) is None
    )


# ------------------------------------------------------------------------------------------------- task_info
def head(*terms):
    """A compiled ground goal atom as option_atoms reads it: a bddl HEAD with its terms."""
    from bddl.condition_evaluation import HEAD

    h = HEAD.__new__(HEAD)
    h.terms = list(terms)
    return h


class StubTask:
    def __init__(self, options, instance=301):
        self._options, self.reads, self.activity_instance_id = options, 0, instance

    @property
    def ground_goal_state_options(self):
        self.reads += 1
        return self._options


def task_sim(options, instance=301):
    fixed = {"cabinet.n.01_1", "floor.n.01_1"}
    return SimpleNamespace(
        env=SimpleNamespace(task=StubTask(options, instance)),
        scene_object=lambda n: SimpleNamespace(fixed_base=n in fixed),
        task_scope=lambda: {"jar.n.01_1": None, "cabinet.n.01_1": None},
        floor_name=lambda: "floor.n.01_1",
    )


def test_task_info_is_task_goal_options_as_facts_plus_floor_arms_and_scope(caplog):
    from b1k.bridge.strategies import task_goal_atoms, task_goal_options
    from omnigibson.eval.utils.eval_utils import TASK_NAMES_TO_INDICES
    from omnigibson.tiptop.oracle import task_info, taskinfo  # task_info at module level: a --providers host's

    options = [[head("inside", "jar.n.01_1", "cabinet.n.01_1"), head("not", "open", "cabinet.n.01_1")],
               [head("ontop", "jar.n.01_1", "cabinet.n.01_1")]]  # fmt: skip
    sim, planners = task_sim(options), {"left": ("client", {}), "right": ("press", {})}
    with caplog.at_level(logging.WARNING):
        ti = task_info(sim, planners, 11519, "store_honey")
    assert (ti.task_id, ti.name, ti.source) == (TASK_NAMES_TO_INDICES["store_honey"], "store_honey", "oracle")
    assert ti.goal_options == ((Fact("inside", ("jar.n.01_1", "cabinet.n.01_1")), Fact("open", ("cabinet.n.01_1",), False)),
                               (Fact("ontop", ("jar.n.01_1", "cabinet.n.01_1")),))  # fmt: skip
    assert ti.goal_options == tuple(tuple(taskinfo.atom_to_fact(a) for a in o) for o in task_goal_options(sim))
    assert [taskinfo.atom_to_fact(a) for a in task_goal_atoms(sim)] == list(ti.goal_options[0]), (
        "option 0 is the goal the host-built strategy runs"
    )
    assert [r.id for r in ti.scope] == ["cabinet.n.01_1", "jar.n.01_1"], "sorted(task_scope())"
    assert (ti.scope[0].fixed, ti.scope[1].fixed, ti.scope[0].category) == (True, False, "cabinet")
    assert ti.floor == ObjRef("floor.n.01_1", "floor", True) and ti.arms == ("left", "right") and ti.max_steps == 11519
    assert "capped" not in caplog.text
    reads = sim.env.task.reads
    again = task_info(sim, ["left"], 500, "store_honey")
    assert again.goal_options is ti.goal_options and again.scope is ti.scope and again.floor == ti.floor
    assert sim.env.task.reads == reads, "cached per (task, instance)"
    assert (again.arms, again.max_steps) == (("left",), 500), "the arms and max_steps are the call's"
    other = task_info(task_sim(options[:1], instance=302), ("left",), 500, "store_honey")
    assert len(other.goal_options) == 1, "another instance is read afresh"
    with caplog.at_level(logging.WARNING):
        assert task_info(task_sim(options), ("left",), None, "not_a_challenge_task").task_id == -1
    assert "not a challenge task" in caplog.text


def test_a_capped_read_is_the_same_random0_sample_and_is_logged(caplog):
    from b1k.bridge.strategies import GOAL_OPTIONS_READ, task_goal_atoms, task_goal_options
    from omnigibson.tiptop.oracle import taskinfo

    options = [[head("inside", f"candle.n.01_{i}", "basket.n.01_1")] for i in range(GOAL_OPTIONS_READ + 1)]
    sim = task_sim(options)
    with caplog.at_level(logging.WARNING):
        ti = taskinfo.task_info(sim, ("left",), None, "assembling_gift_baskets")
    assert len(ti.goal_options) == GOAL_OPTIONS_READ == 20000
    assert ti.goal_options == tuple(tuple(taskinfo.atom_to_fact(a) for a in o) for o in task_goal_options(sim)), (
        "the function's own Random(0) sample: called, never re-sampled"
    )
    assert list(ti.goal_options[0]) != [taskinfo.atom_to_fact(a) for a in task_goal_atoms(sim)]
    assert "capped at 20000" in caplog.text


# ------------------------------------------------------------------------------------- the bench's pass-through
class Host:  # the skill bench test's bench host over the fakes' Env
    def __init__(self, env):
        self.env, self.arm_left, self.env_wall_s = env, 0.0, 0.0

    def commanded_targets(self):
        from b1k.runtime.compose import CLOSED

        return {"arm_left": np.full(7, self.arm_left, dtype=np.float32),
                "gripper_left": np.array([CLOSED], dtype=np.float32)}  # fmt: skip

    def observe_now(self):
        from b1k.connector.observe import StepObs

        self.env.p[3:10] = self.arm_left
        return StepObs(0, self.env.p.copy(), {})

    def env_step(self, a):
        return self.env.step(a)

    def raw(self):
        return self.env.raw()

    def parse(self, raw, step):
        from b1k.connector.observe import StepObs

        return StepObs(step, raw["proprio"], raw)


class FakeEpisode:
    def __init__(self, host):
        self.sim, self.host, self.records = SimpleNamespace(n_steps=0), host, []

    def pick(self, name, into=None, **kw):
        self.sim.n_steps += 40
        self.host.arm_left = 0.9
        return True


class SimClockObserver:
    requires_sim_clock = True

    def observe(self, req, obs):
        from b1k.connector.observe import Percept, PerceptInfo

        return Percept(PerceptInfo("", 0, 0, 0, {}, "oracle"), {}, {}, None), obs
        yield {}


def bench_trial(**kw):
    from b1k.tests.fakes import Env, apple, make_rt
    from omnigibson.tiptop.host import skillbench
    from omnigibson.tiptop.host.legacy_skills import LegacyBackend

    host = Host(Env())
    backends = {"legacy": LegacyBackend(FakeEpisode(host), host.observe_now, single_round=True)}
    r, rt, calls, wall = skillbench.run_trial(host, make_rt().svc, backends, {"pick_up": {"default": "legacy"}},
                                              SimClockObserver(), SkillCall("pick_up", PickArgs(apple), arm="left", seed=3),
                                              **kw)  # fmt: skip
    return r, rt, calls, DirectConnector(rt, host.env_step, host, host.raw())


def test_the_bench_trial_hands_its_runtime_the_task_info_and_its_results_are_unchanged():
    from b1k.tests.fakes import TASK
    from omnigibson.tiptop.host import skillbench

    r, rt, calls, conn = bench_trial(task_info=lambda: TASK)
    assert r.status is Status.SUCCEEDED and r.seed == 3 and r.backend == "legacy" and len(calls) == 1
    assert (rt.step, rt.idle_steps, sum(rt.charged.values())) == (0, 0, 0) and skillbench.u0(rt)
    assert conn.task() is TASK and rt.query("task", None) is TASK, "conn.task() is no longer None"
    r0, rt0, calls0, conn0 = bench_trial()
    assert conn0.task() is None, "the default is today's"
    assert (r0.status, r0.steps, r0.backend, len(calls0)) == (r.status, r.steps, r.backend, 1)
    assert (rt0.step, rt0.idle_steps, dict(rt0.charged)) == (rt.step, rt.idle_steps, dict(rt.charged))
