"""The skill bench v0 (SPEC §8 Track A2, §9) without Isaac Sim: R1ProSim's step_action and last-action cache, the
bench host, the CaptureObserver, the legacy backend (smoke group 5's legacy half), single_round, the bench's trial
and its step invariant (U0), and the oracle providers over fakes of the scene."""

from pathlib import Path
from types import MethodType, SimpleNamespace

import numpy as np
import pytest
import torch as th
import yaml

from b1k.connector.goals import GoalPanel
from b1k.connector.observe import ObserveRequest, StepObs
from b1k.connector.skills import (
    Code,
    OpenArgs,
    PickArgs,
    PlaceArgs,
    Rel,
    Relation,
    SkillCall,
    Status,
    WorldUpdate,
)
from b1k.connector.types import Fact, ObjRef, Provided
from b1k.connector.world import ProvenancePolicy, link_pose
from b1k.observation import PROPRIO_SLICES
from b1k.runtime.compose import ACTION_SLICES, CLOSED
from b1k.runtime.direct import DirectConnector
from b1k.tests.fakes import Env, Scripted, SimV, apple, basket, make_rt, table
from omnigibson.tiptop.host import skillbench
from omnigibson.tiptop.host.bench_host import BenchHost
from omnigibson.tiptop.host.capture_observer import CaptureObserver
from omnigibson.tiptop.host.legacy_skills import LegacyBackend, classify

ROOT = Path(__file__).resolve().parents[2]


# ------------------------------------------------------------------------------------------ R1ProSim and the host
def test_step_action_places_each_runtime_group_on_the_robots_controller_and_reads_it_back():
    from omnigibson.tiptop.r1pro import R1ProSim

    order = ("base", "trunk", "arm_right", "gripper_right", "arm_left", "gripper_left")  # not ACTION_SLICES' order
    sizes = {g: s.stop - s.start for g, s in ACTION_SLICES.items()}
    idx, k = {}, 0
    for g in order:
        idx[g], k = th.arange(k, k + sizes[g]), k + sizes[g]
    sent = []
    sim = SimpleNamespace(robot=SimpleNamespace(name="r1", controller_action_idx=idx, action_dim=23), arm="left",
                          other_arm="right", last_action=None, step_env=lambda a: sent.append(a) or "obs")  # fmt: skip
    assert R1ProSim.commanded_targets(sim) == {}  # nothing stepped yet
    a23 = np.arange(23, dtype=np.float32) / 10.0
    a23[ACTION_SLICES["gripper_left"]], a23[ACTION_SLICES["gripper_right"]] = CLOSED, 1.0
    assert R1ProSim.step_action(sim, a23) == "obs"
    sim.last_action = sent[0]
    back = R1ProSim.commanded_targets(sim)
    for g, s in ACTION_SLICES.items():
        assert np.allclose(back[g], a23[s]), g
    assert sent[0]["r1"][idx["arm_left"]].tolist() == pytest.approx(a23[ACTION_SLICES["arm_left"]].tolist())
    assert (sim.last_gripper, sim.other_gripper) == (CLOSED, 1.0), "R1ProSim's own steps keep the hands after it"


def test_every_env_step_is_kept_as_the_last_action():
    from omnigibson.tiptop.scene import TiptopSim

    stepped = []
    sim = SimpleNamespace(
        env=SimpleNamespace(step=lambda a: stepped.append(a) or ({}, 0.0, False, False, {})),
        episode_open=True, n_steps=0, metrics=[], state_stream=None, recorders=[], stop_when_done=False,
        robot=SimpleNamespace(name="r1"), action=lambda q, g: {"r1": ("q", q, g)},
    )  # fmt: skip
    sim.step_env = MethodType(TiptopSim.step_env, sim)
    TiptopSim.step(sim, [0.1], -1.0)
    assert sim.last_action == {"r1": ("q", [0.1], -1.0)} and stepped == [sim.last_action] and sim.n_steps == 1
    sim.step_env({"r1": "whole"})
    assert sim.last_action == {"r1": "whole"} and sim.n_steps == 2


def fake_robot():
    dims = {k: s.stop - s.start for k, s in PROPRIO_SLICES.items()}
    values = {k: th.full((n,), float(i)) for i, (k, n) in enumerate(dims.items())}
    values["joint_qpos"] = th.zeros(40)  # fields the eval layout leaves out are ignored
    return SimpleNamespace(_get_proprioception_dict=lambda: values)


def test_the_bench_host_hands_the_runtime_the_eval_proprio_layout_and_what_r1pro_last_commanded():
    stepped = []
    sim = SimpleNamespace(robot=fake_robot(), n_steps=7, step_action=stepped.append,
                          commanded_targets=lambda: {"arm_left": np.ones(7)})  # fmt: skip
    host = BenchHost(sim)
    p = host.proprio()
    assert p.shape == (61,) and p.dtype == np.float32
    for i, s in enumerate(PROPRIO_SLICES.values()):
        assert (p[s] == i).all()
    raw = host.env_step(np.zeros(23))
    assert len(stepped) == 1 and (raw["proprio"] == p).all() and host.env_wall_s >= 0.0
    obs = host.observe_now()
    assert obs.step == 7 and (obs.proprio == p).all()
    assert host.parse(raw, 3).step == 3 and host.commanded_targets()["arm_left"].sum() == 7


def test_the_capture_observer_runs_on_the_sim_clock_and_returns_the_planners_percept():
    host = SimpleNamespace(observe_now=lambda: StepObs(40, np.arange(61, dtype=np.float32), {}))
    request = {"view_name": "head", "rgb": "img", "views": [{"name": "left_wrist", "rgb": "wimg"}]}
    extras = {"views": {"left_wrist": {}}}
    sim = SimpleNamespace(capture=lambda task: (request, extras), tracked_label=lambda n: n.replace(".n.01_", "_"))
    masks = {"head": {"apple_1": np.ones((2, 2), bool), "basket_1": np.zeros((2, 2), bool)},
             "left_wrist": {"apple_1": np.zeros((2, 2), bool), "basket_1": np.zeros((2, 2), bool)}}  # fmt: skip
    segmenter = SimpleNamespace(masks=lambda labels, rq, ex: Provided(masks, "oracle", 40))
    obs = CaptureObserver(sim, host, segmenter, "t")
    assert obs.requires_sim_clock
    gen = obs.observe(ObserveRequest((apple, basket)), None)
    with pytest.raises(StopIteration) as done:
        next(gen)  # ends before its first yield: 0 Runtime steps
    percept, after = done.value.value
    assert after.step == 40 and percept.info.visible == {apple.id: 1.0, basket.id: 0.0}
    assert percept.info.source == "oracle" and set(percept.views) == {"head", "left_wrist"}
    assert (percept.q["trunk"] == np.arange(61)[PROPRIO_SLICES["trunk_qpos"]]).all()
    with pytest.raises(NotImplementedError):
        next(obs.observe(ObserveRequest((apple,), aim=False), None))


# ------------------------------------------------------------------------------------ the legacy backend (group 5)
class FakeSim:
    n_steps = 0


class FakeEpisode:
    def __init__(self, host, picks=True, record=None):
        self.sim, self.host, self.picks, self.records, self.record = FakeSim(), host, picks, [], record

    def pick(self, name, into=None, **kw):
        self.kw = kw
        self.sim.n_steps += 40
        self.host.arm_left = 0.9  # R1ProSim.step drove arm_left inside the run
        if self.record:
            self.records.append(self.record)
        return self.picks

    def achieve(self, atoms, arm="left"):
        self.sim.n_steps += 30
        self.atoms = atoms
        return True


class Host:
    def __init__(self, env):
        self.env, self.arm_left, self.env_wall_s = env, 0.0, 0.0

    def commanded_targets(self):
        return {"arm_left": np.full(7, self.arm_left, dtype=np.float32),
                "gripper_left": np.array([CLOSED], dtype=np.float32)}  # fmt: skip

    def observe_now(self):
        self.env.p[3:10] = self.arm_left
        return StepObs(0, self.env.p.copy(), {})

    # DirectConnector's env and adapter, for run_trial
    def env_step(self, a):
        return self.env.step(a)

    def raw(self):
        return self.env.raw()

    def parse(self, raw, step):
        return StepObs(step, raw["proprio"], raw)


def legacy_rt(env, strict=frozenset(), truth=None, host=True, single_round=True, **ep):
    h = Host(env)
    episode = FakeEpisode(h, **ep)
    lb = LegacyBackend(episode, h.observe_now, has_cavity=lambda o: False, strict_relations=strict,
                       single_round=single_round)  # fmt: skip
    routing = {"pick_up": {"default": "legacy"}, "place": {"default": "legacy"}, "wait": {"default": "scripted"}}
    v = SimV(truth)
    return make_rt(backends={"legacy": lb, "scripted": Scripted()}, routing=routing, verifier=v,
                   host=h if host else None), episode, v  # fmt: skip


def test_a_legacy_call_costs_no_runtime_step_reseeds_the_latch_and_is_judged_after_it_ran():
    env, acts = Env(), []
    rt, ep, v = legacy_rt(env, truth={"under": False})
    conn = DirectConnector(rt, lambda a: (acts.append(a.copy()), env.step(a))[1], Host(env), env.raw())
    pick = conn.run(SkillCall("pick_up", PickArgs(apple), arm="left"))
    assert not acts, "a legacy call costs 0 Runtime env steps (it stepped the sim itself)"
    assert pick.steps == 40 and pick.requires_sim_clock and pick.status is Status.SUCCEEDED
    assert float(v.obs_seen[0].proprio[3]) == np.float32(0.9), "the verdict is judged on the frames AFTER the run"
    assert ep.kw == {"single_round": True} and pick.evidence["single_round"] is True
    under = conn.run(SkillCall("place", PlaceArgs(apple, (Relation(Rel.UNDER, table),))))
    inside = conn.run(SkillCall("place", PlaceArgs(apple, (Relation(Rel.IN, basket),))))
    conn.dwell(2)
    assert np.allclose(acts[-1][7:14], 0.9), "arm_left continues from where the legacy run left it (probe P2)"
    assert acts[-1][14] == CLOSED
    assert (under.status, under.code) == (Status.FAILED, Code.PLACED_WRONG), under
    assert under.evidence["degraded_to"] == "on" and under.evidence["legacy_ok"] is True
    assert inside.evidence.get("degraded_to") == "on", "in -> a target with no cavity is bent onto on(): flagged"
    env3 = Env()
    rt3, ep3, _ = legacy_rt(env3, strict=frozenset({Rel.UNDER}), single_round=False)
    conn3 = DirectConnector(rt3, env3.step, Host(env3), env3.raw())
    assert conn3.run(SkillCall("place", PlaceArgs(apple, (Relation(Rel.UNDER, table),)))).code is Code.UNSUPPORTED
    assert ep3.sim.n_steps == 0, "UNDER is strict: nothing ran for it"
    conn3.run(SkillCall("pick_up", PickArgs(apple), arm="left"))
    assert ep3.kw == {}, "the pseudo planner's shim keeps today's multi-round pick"
    env4 = Env()
    rt4, _, _ = legacy_rt(env4, host=False)
    with pytest.raises(RuntimeError, match="HostHooks"):
        DirectConnector(rt4, env4.step, Host(env4), env4.raw()).run(SkillCall("pick_up", PickArgs(apple)))


def test_a_legacy_failure_takes_its_code_from_the_rounds_own_error_text():
    env = Env()
    record = {"round": 0, "error": "TiptopPlanningError: planning failed: cuTAMP failed to find a plan: No satisfying "
                                   "particles found after 3 attempts"}  # fmt: skip
    rt, _, _ = legacy_rt(env, truth={"holding": False}, picks=False, record=record)
    r = DirectConnector(rt, env.step, Host(env), env.raw()).run(SkillCall("pick_up", PickArgs(apple), arm="left"))
    assert (r.status, r.code, r.phase) == (Status.INFEASIBLE, Code.NO_GRASP, "particles")
    assert r.evidence["records"] == [record] and r.verdicts == {"scorer": False}


@pytest.mark.parametrize("why, skill, expected", [
    ("TiptopPlanningError: planning failed: cuTAMP failed to find a plan: Motion planning failed", "pick_up",
     (Status.INFEASIBLE, Code.NO_MOTION, "motion")),
    ("No satisfying particles", "place", (Status.INFEASIBLE, Code.NO_PLACEMENT, "particles")),
    ("motion validation rejected Pick(can_1, grasp1, q1): ('left_realsense_link', 'floors_1')", "pick_up",
     (Status.INFEASIBLE, Code.EXEC_REFUSED, "motion")),
    ("GoalNotVisible: goal objects ['bacon_1'] are not visible in any view", "pick_up",
     (Status.PRECONDITION_UNMET, Code.NOT_VISIBLE, "check")),
    ("no stance in front of electric_refrigerator.n.01_1 lets the arm reach its handle and pull", "open",
     (Status.INFEASIBLE, Code.NO_STANCE_HERE, "check")),
    ("no joint of cabinet.n.01_1 can be taken hold of", "open", (Status.INFEASIBLE, Code.NO_FEATURE, "check")),
    (None, "pick_up", (Status.FAILED, Code.GRASP_MISSED, "execute")),
    ("the joint did not move", "open", (Status.FAILED, Code.STALLED, "execute")),
])  # fmt: skip
def test_the_one_classifier_maps_todays_error_text_to_codes(why, skill, expected):
    assert classify(why, skill) == expected


# ------------------------------------------------------------------------------------------------ single_round
def test_a_single_round_pick_neither_stands_nor_pushes_and_runs_one_round():
    from omnigibson.tiptop.bench import Episode

    calls = []
    sim = SimpleNamespace(arm="left", side_entry=lambda *a: False, push_face=lambda b: calls.append("push") or 1,
                          held_objects={})  # fmt: skip
    ep = SimpleNamespace(
        sim=sim, rounds=3, args=SimpleNamespace(grasping_mode="assisted"), support_of=lambda b: None,
        is_floor=lambda n: False, stand_for=lambda b: calls.append("stand"), reaches_floor=lambda b: False,
        plan_and_execute=lambda atoms, floor=False: calls.append("round"), holding=lambda b: False,
        achieve=lambda atoms: calls.append("push round"),
    )  # fmt: skip
    assert Episode.pick(ep, "apple.n.01_1", single_round=True) is False
    assert calls == ["round"]
    calls.clear()
    Episode.pick(ep, "apple.n.01_1")
    assert calls.count("stand") == 3 and calls.count("round") == 3, "today's multi-round pick is unchanged"


def test_a_single_round_open_pulls_from_where_it_stands_with_no_fallback_stance(monkeypatch):
    import omnigibson.tiptop.articulation as articulation
    from omnigibson.tiptop.bench import Episode

    opened, pushed = [], []
    joint = {"name": "j_link_4", "lower": 0.0, "upper": 0.39, "position": 0.3, "closed": 0.0}
    monkeypatch.setattr(articulation, "openable_joints", lambda obj: [joint])
    sim = SimpleNamespace(
        arm="left", n_steps=5, scene_object=lambda n: None,
        open_container=lambda arm, name, **kw: opened.append(kw) or {"opened": False, "why": "stance rejected"},
        push_joint=lambda arm, name, j, target, stand=True: pushed.append(stand) or {"reached": True},
    )  # fmt: skip
    ep = SimpleNamespace(sim=sim, spec=None, records=[], is_shut=lambda n: True,
                         stand_for=lambda n: pytest.fail("single_round stood for the container"))  # fmt: skip
    assert Episode.open_up(ep, "cabinet.n.01_1", single_round=True) is False
    assert [kw["stand"] for kw in opened] == [False] and ep.records[-1]["single_round"] is True
    assert Episode.open_up(ep, "cabinet.n.01_1", 0.0, single_round=True) is True and pushed == [False]


# ---------------------------------------------------------------------------------------- the trial and U0
class Tip:  # a native-shaped backend: plans from the planner's Percept, then yields its motion
    name = "tiptop"

    def supports(self, c):
        return True

    def check(self, c, s):
        from b1k.connector.skills import Precheck

        return Precheck(True)

    def run(self, call, svc, obs):
        from b1k.connector.skills import SkillResult

        assert svc.percepts[call.percept] is not None
        for _ in range(3):
            obs = yield {"arm_left": np.full(7, 0.3, dtype=np.float32)}
        return SkillResult(call.call_id, call.skill, "tiptop", Status.SUCCEEDED, None, "", "", (), {"scorer": True},
                           "scorer")  # fmt: skip


class SimClockObserver:  # the bench's CaptureObserver shape
    requires_sim_clock = True

    def __init__(self):
        self.n = 0

    def observe(self, req, obs):
        from b1k.connector.observe import Percept, PerceptInfo

        self.n += 1
        return Percept(PerceptInfo("", 0, 0, 0, {}, "oracle"), {}, {}, None), obs
        yield {}


def trial(backend, observer=None):
    host = Host(Env())
    ep = FakeEpisode(host)
    base = make_rt()  # its Services, rebuilt by run_trial around the bench's specs and backends
    backends = {"legacy": LegacyBackend(ep, host.observe_now, single_round=True), "tiptop": Tip()}
    routing = {"pick_up": {"default": backend}}
    return skillbench.run_trial(host, base.svc, backends, routing, observer or SimClockObserver(),
                                SkillCall("pick_up", PickArgs(apple), arm="left", seed=3))  # fmt: skip


def test_a_legacy_trial_is_one_call_with_no_observe_and_holds_the_step_invariant():
    observer = SimClockObserver()
    r, rt, calls, wall = trial("legacy", observer)
    assert r.status is Status.SUCCEEDED and r.seed == 3 and observer.n == 0, "legacy captures in its own run"
    assert (rt.step, rt.idle_steps, sum(rt.charged.values())) == (0, 0, 0) and skillbench.u0(rt)
    assert skillbench.latch_gap(rt) < 1e-6, "re-seeded from what the legacy run last commanded: no snap-back"
    assert len(calls) == 1 and wall >= 0.0
    case = {"id": "c", "call": SkillCall("pick_up", PickArgs(apple)), "expect": {"status": "succeeded"}}
    row = skillbench.row(case, 0, 3, r, rt, 40, 2.0, 0.5)
    assert row["u0"] and row["steps"] == 40 and row["planning_wall_s"] == 1.5 and row["verdicts"] == {"scorer": True}
    s = skillbench.summarize(case, [row, dict(row, status="failed", code="grasp_missed", u0=False)])
    assert (s["n"], s["succeeded"], s["rate"], s["expect_met"], s["u0_all"]) == (2, 1, 0.5, 1, False)


def test_a_native_trial_observes_once_and_every_env_step_is_charged_to_it():
    observer = SimClockObserver()
    r, rt, _, _ = trial("tiptop", observer)
    assert r.status is Status.SUCCEEDED and observer.n == 1
    assert rt.charged == {"observe": 0, "skill": 3} and rt.step == 3 and rt.idle_steps == 0 and skillbench.u0(rt)


def test_the_step_invariant_fails_on_an_idle_step_or_an_uncharged_one():
    rt = SimpleNamespace(step=3, idle_steps=0, charged={"skill": 3})
    assert skillbench.u0(rt)
    assert not skillbench.u0(SimpleNamespace(step=3, idle_steps=1, charged={"skill": 2}))
    assert not skillbench.u0(SimpleNamespace(step=3, idle_steps=0, charged={"skill": 2}))


def test_the_latch_gap_is_how_far_the_next_action_would_snap_an_arm():
    p = np.zeros(61, dtype=np.float32)
    p[PROPRIO_SLICES["arm_left_qpos"]] = 0.9
    target = np.zeros(23, dtype=np.float32)
    rt = SimpleNamespace(obs=StepObs(0, p, {}), latch=SimpleNamespace(target=target))
    assert skillbench.latch_gap(rt) == pytest.approx(0.9)


def test_the_bench_cases_load_as_skill_calls_and_a_setup_the_bench_cannot_do_is_refused(tmp_path):
    bench = ROOT / "tiptop/b1k/skills/bench"
    cases = {c["id"]: c for f in ("pick_up.yaml", "open.yaml") for c in skillbench.load_cases(bench / f)}
    assert set(cases) == {"pick_up_freeze_fruit_apple", "pick_up_store_honey_jar", "open_store_honey_drawer"}
    apple_case = cases["pick_up_freeze_fruit_apple"]
    assert (apple_case["task"], apple_case["instance"], apple_case["seeds"]) == ("freeze_fruit", 301, [0, 1, 2, 3, 4])
    assert apple_case["call"] == SkillCall("pick_up", PickArgs(ObjRef("apple.n.01_2", "apple")), arm="left")
    drawer = cases["open_store_honey_drawer"]["call"]
    assert isinstance(drawer.args, OpenArgs) and drawer.args.target.fixed and drawer.args.joint == "j_link_4"
    only = skillbench.load_cases(bench / "pick_up.yaml", ids=["pick_up_store_honey_jar"])
    assert [c["id"] for c in only] == ["pick_up_store_honey_jar"]
    bad = tmp_path / "bad.yaml"
    first = yaml.safe_load((bench / "pick_up.yaml").read_text())[0]
    bad.write_text(yaml.safe_dump([{**first, "setup": {"held": "apple.n.01_2"}}]))
    with pytest.raises(ValueError, match="not on the bench yet"):
        skillbench.load_cases(bad)


def test_the_one_call_planner_observes_only_when_the_skill_asks_for_a_percept():
    seen = []
    conn = SimpleNamespace(
        check=lambda call: SimpleNamespace(code=Code.PERCEPT_REQUIRED if call.percept is None else None),
        observe=lambda req: seen.append(req) or SimpleNamespace(id="c1"), run=lambda call: call,
    )  # fmt: skip
    call = SkillCall("place", PlaceArgs(apple, (Relation(Rel.ON, table), Relation(Rel.NEXT_TO, basket))))
    ran = skillbench.one_call(conn, call)
    assert ran.percept == "c1" and seen[0].targets == (apple, table, basket)
    conn.check = lambda call: SimpleNamespace(code=None)
    assert skillbench.one_call(conn, call).percept is None and len(seen) == 1


# -------------------------------------------------------------------------------------------- the oracle providers
def test_the_oracle_world_holds_what_the_skills_said_while_the_fingers_say_held(monkeypatch):
    from omnigibson.tiptop.oracle import world as oracle_world

    held = {"left": True}
    grasp = SimpleNamespace(held=lambda arm, obs: Provided(held.get(arm, False), "proprio", 0))
    sim = SimpleNamespace(scene_object=lambda n: SimpleNamespace(fixed_base=n.startswith("cabinet")))
    supports = {"jar.n.01_1": "box.n.01_1", "box.n.01_1": "cabinet.n.01_1", "cabinet.n.01_1": "floor.n.01_1"}
    inside = lambda p, a, b: b.startswith("cabinet")
    ep = SimpleNamespace(sim=sim, support_of=supports.get, is_floor=lambda n: n.startswith("floor"),
                         is_shut=lambda n: True, goal_already_holds=inside)  # fmt: skip
    w = oracle_world.OracleWorld(ep, grasp)
    assert w.held("left").value is None, "closed on something no skill named: unknown"
    jar = ObjRef("jar.n.01_1", "jar")
    w.apply(WorldUpdate("held", jar, "left"))
    assert w.held("left").value == (jar,) and w.holding(jar).value == ("left",)
    held["left"] = False
    assert w.held("left").value == () and w.held("left").source == "proprio"
    held["left"] = True
    assert w.held("left").value is None, "an empty reading cleared the record"
    assert [o.id for o in w.enclosed_by(ObjRef("box.n.01_1", "box")).value] == ["cabinet.n.01_1"]
    joints = [{"name": "j1", "lower": 0.0, "upper": 0.4, "position": 0.1, "closed": 0.0},
              {"name": "j2", "lower": 0.0, "upper": 0.4, "position": 0.0, "closed": 0.0}]  # fmt: skip
    monkeypatch.setattr(oracle_world, "openable_joints", lambda obj: joints)
    cab = ObjRef("cabinet.n.01_1", "cabinet", True)
    assert w.is_open(cab).value is True and w.is_open(cab, "j2").value is False
    assert w.open_fraction(cab).value == pytest.approx(0.25) and w.is_open(cab).source == "oracle"
    monkeypatch.setattr(oracle_world, "openable_joints", lambda obj: [])
    assert w.is_open(cab).value is None, "nothing that opens: unknown, not shut"


def test_the_scorer_answers_holding_and_hand_empty_from_the_grasp_and_the_rest_from_bddl():
    from omnigibson.controllers import IsGraspingState
    from omnigibson.tiptop.oracle.goals import EpisodeScorer

    apple_obj = object()
    robot = SimpleNamespace(is_grasping=lambda arm, obj=None: IsGraspingState.TRUE
                            if arm == "left" and obj in (None, apple_obj) else IsGraspingState.FALSE)  # fmt: skip
    sim = SimpleNamespace(robot=robot, scene_object=lambda n: apple_obj if n == "apple.n.01_1" else object(),
                          holds=lambda p, *a: {"ontop": True}[p])  # fmt: skip
    s = EpisodeScorer(sim)
    assert s.holds(Fact("holding", ("apple.n.01_1", "left"))) is True
    assert s.holds(Fact("holding", ("apple.n.01_1", "right"))) is False
    assert s.holds(Fact("hand_empty", ("left",))) is False and s.holds(Fact("hand_empty", ("right",))) is True
    assert s.holds(Fact("ontop", ("apple.n.01_1", "table.n.02_1"))) is True
    assert s.holds(Fact("levitating", ("apple.n.01_1",))) is None, "a predicate it cannot judge"


def test_gravity_turns_a_lid_shut_and_leaves_a_door_and_a_drawer_alone():
    from omnigibson.tiptop.oracle.articulation import gravity

    hinge = np.zeros(3)
    assert gravity("revolute", (1, 0, 0), hinge, (0, 0.2, 0), opens_up=True) == "falls_shut"  # a lid, hinged at back
    assert gravity("revolute", (1, 0, 0), hinge, (0, 0.2, 0), opens_up=False) == "falls_open"
    assert gravity("revolute", (0, 0, 1), hinge, (0.4, 0, 0.5), opens_up=True) == "neutral"  # a door
    assert gravity("prismatic", (1, 0, 0), hinge, (0, 0.2, 0), opens_up=True) == "neutral"


def box_vertices(lo, hi) -> np.ndarray:
    return np.array([[x, y, z] for x in (lo[0], hi[0]) for y in (lo[1], hi[1]) for z in (lo[2], hi[2])], float)


def test_voxels_cover_every_hull_and_a_panel_thinner_than_a_voxel():
    from omnigibson.tiptop.oracle.mapbuild import voxelize

    cube, panel = box_vertices((0, 0, 0), (0.1, 0.1, 0.1)), box_vertices((0.2, 0, 0), (0.205, 0.1, 0.1))
    origin, occ = voxelize([cube, panel], 0.02)
    assert origin == (0.0, 0.0, 0.0) and occ.shape == (11, 5, 5)
    assert occ[:5].all(), "the cube's 5 x 5 x 5 voxels"
    assert occ[10].all() and not occ[5:10].any(), "the 5 mm panel fills its one layer; nothing between"
    origin, occ = voxelize([box_vertices((-0.05, -0.05, -0.05), (0.05, 0.05, 0.05))], 0.02)
    assert origin == pytest.approx((-0.06, -0.06, -0.06)) and occ.shape == (6, 6, 6) and occ.all()


def test_a_moving_link_is_mapped_at_its_closed_value_and_posed_back_by_the_estimated_one(monkeypatch):
    import omnigibson.utils.usd_utils as usd_utils
    from omnigibson.tiptop.oracle import mapbuild

    body = box_vertices((0.0, 0.0, 0.0), (0.4, 0.4, 0.4))
    drawer_live = box_vertices((0.4, 0.1, 0.1), (0.5, 0.3, 0.3)) + [0.2, 0.0, 0.0]  # pulled out 0.2 along +x
    geoms = lambda v: {"g": SimpleNamespace(prim=v)}
    obj = SimpleNamespace(
        get_position_orientation=lambda: (th.tensor([1.0, 2.0, 0.0]), th.tensor([0.0, 0.0, 0.0, 1.0])),
        links={"base_link": SimpleNamespace(collision_meshes=geoms(body + [1.0, 2.0, 0.0])),
               "link_1": SimpleNamespace(collision_meshes=geoms(drawer_live + [1.0, 2.0, 0.0])),
               "handle_meta": SimpleNamespace(visual_only=True, collision_meshes={})},
    )  # fmt: skip
    joint = {"name": "j1", "kind": "prismatic", "axis": np.array([1.0, 0, 0]), "origin": np.array([1.4, 2.2, 0.2]),
             "lower": 0.0, "upper": 0.3, "position": 0.2, "closed": 0.0, "link": "link_1"}  # fmt: skip
    monkeypatch.setattr(mapbuild, "openable_joints", lambda o: [joint])
    monkeypatch.setattr(usd_utils, "mesh_prim_to_trimesh_mesh", lambda prim, **kw: SimpleNamespace(vertices=prim))
    ref = ObjRef("cabinet.n.01_1", "cabinet", True)
    piece = mapbuild.build_piece(obj, ref, 0.02)
    assert set(piece.links) == {"body", "link_1"} and piece.joints[0].closed_end == "lower"
    grid = piece.links["link_1"]
    lo = np.asarray(grid.origin)
    assert lo == pytest.approx((0.4, 0.1, 0.1)), "the drawer's voxels are where it is when shut, in the body frame"
    assert piece.links["body"].occupied.shape == (20, 20, 20)
    T = link_pose(piece, "link_1", {"j1": 0.2})  # the estimator's value poses it back where the sim has it
    first = T @ np.r_[np.asarray(grid.origin) + 0.01, 1.0]
    assert first[:3] == pytest.approx((1.0 + 0.61, 2.0 + 0.11, 0.11))


def test_the_pseudo_map_serves_fixed_furniture_once_per_scene_tagged_map(monkeypatch):
    from omnigibson.tiptop.oracle import mapbuild

    robot = SimpleNamespace(fixed_base=False, category="agent", name="robot_r1")
    cab = SimpleNamespace(fixed_base=True, category="bottom_cabinet", name="bottom_cabinet_slgzfc_0")
    floor = SimpleNamespace(fixed_base=True, category="floors", name="floors_1")
    wall = SimpleNamespace(fixed_base=True, category="walls", name="walls_2")
    apple_obj = SimpleNamespace(fixed_base=False, category="apple", name="apple_7")
    by = {"cabinet.n.01_1": cab, "apple.n.01_2": apple_obj, "walls_2": wall}
    sim = SimpleNamespace(robot=robot, n_steps=4, task_scope=lambda: {"cabinet.n.01_1": cab, "apple.n.01_2": apple_obj},
                          env=SimpleNamespace(scene=SimpleNamespace(objects=[robot, cab, floor, wall, apple_obj])),
                          scene_object=by.get)  # fmt: skip
    built = []
    monkeypatch.setattr(mapbuild, "build_piece", lambda obj, ref, res: built.append(ref) or ("piece", ref.id))
    m = mapbuild.PseudoMap(sim)
    assert [o.id for o in m.furniture()] == ["cabinet.n.01_1", "walls_2"]
    got = m.piece(ObjRef("cabinet.n.01_1", "cabinet"))
    assert got == Provided(("piece", "cabinet.n.01_1"), "map", 4)
    m.piece(ObjRef("cabinet.n.01_1", "cabinet"))
    assert len(built) == 1, "built once, then served from the scene's cache"
    assert m.piece(ObjRef("apple.n.01_2", "apple")).value is None, "a movable is not map data"
    assert mapbuild.pseudo_map(sim) is mapbuild.pseudo_map(sim)


def test_the_pseudo_stack_counts_every_oracle_read_and_judges_with_the_scorer(monkeypatch):
    from omnigibson.tiptop.oracle import pseudo_services

    cab = SimpleNamespace(aabb=(th.tensor([0.0, 0.0, 0.0]), th.tensor([1.0, 0.5, 0.9])), fixed_base=True)
    sim = SimpleNamespace(n_steps=9, max_steps=None, scene_object=lambda n: cab, robot=None,
                          task_scope=lambda: {}, env=SimpleNamespace(scene=SimpleNamespace(objects=[])))  # fmt: skip
    routing = {"goal_checker": "scorer", "goal_checkers_shadow": []}
    svc, segmenter = pseudo_services(SimpleNamespace(sim=sim), "planner", routing)
    assert isinstance(svc.goals, GoalPanel) and svc.goals.primary == "scorer" and svc.planner == "planner"
    top = svc.geometry.top_support(ObjRef("cabinet.n.01_1", "cabinet", True))
    assert top.value.z == pytest.approx(0.9) and top.source == "oracle"
    assert svc.provenance.take() == {"geometry.top_support": 1}, "counted in pseudo"
    assert isinstance(svc.provenance, ProvenancePolicy) and segmenter.sim is sim
