"""The skill bench v0 (SPEC §8 Track A2, §9) without Isaac Sim: R1ProSim's step_action and last-action cache, the
bench host, the CaptureObserver, the legacy backend (smoke group 5's legacy half), single_round, the bench's trial
and its step invariant (U0), and the oracle providers over fakes of the scene."""

import dataclasses
from pathlib import Path
from types import MethodType, SimpleNamespace

import numpy as np
import pytest
import torch as th
import yaml

from b1k.connector.goals import GoalPanel
from b1k.connector.observe import ObserveRequest, StepObs
from b1k.connector.skills import (
    Binding,
    CloseArgs,
    Code,
    OpenArgs,
    PickArgs,
    PlaceArgs,
    PressArgs,
    Rel,
    Relation,
    ReleaseArgs,
    SkillCall,
    Status,
    WaitArgs,
    WorldUpdate,
)
from b1k.connector.types import Belief, Fact, ObjRef, Pose2, Provided
from b1k.connector.world import ProvenancePolicy, link_pose
from b1k.observation import PROPRIO_SLICES, CameraView
from b1k.runtime.compose import ACTION_SLICES, CLOSED
from b1k.runtime.direct import DirectConnector
from b1k.tests.fakes import REFS, Env, JointWorld, Scripted, SimV, apple, basket, make_rt, radio, table
from omnigibson.tiptop.host import overview, skillbench
from omnigibson.tiptop.host.bench_host import BenchHost, Frames
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
    names = {"trunk": [f"torso_joint{k}" for k in range(1, 5)],
             "left": [f"left_arm_joint{k}" for k in range(1, 8)], "right": [f"right_arm_joint{k}" for k in range(1, 8)]}
    robot = SimpleNamespace(name="r1", controller_action_idx=idx, action_dim=23, trunk_joint_names=names["trunk"],
                            arm_joint_names={"left": names["left"], "right": names["right"]})
    sim = SimpleNamespace(robot=robot, arm="left", other_arm="right", last_action=None, other_gripper=1.0,
                          posture={j: 0.0 for j in names["right"]}, planned_joints=names["trunk"] + names["left"],
                          step_env=lambda a: sent.append(a) or "obs")  # fmt: skip
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
    own = R1ProSim.action(sim, np.zeros(11), CLOSED)["r1"]  # R1ProSim's own next step (a legacy run, a restore)
    assert own[idx["arm_right"]].tolist() == pytest.approx(a23[ACTION_SLICES["arm_right"]].tolist()), \
        "the idle arm stays where the Runtime put it: no snap back to the pre-run posture"
    assert set(sim.posture) == set(names["right"]), "only the locked joints are latched"


def test_a_capture_after_the_camera_moved_waits_past_the_frames_from_before_the_move(monkeypatch):
    """w2s2 apple_1 t0: the first capture after a 5 cm stance change rendered the old view twice, 'converged after 4
    renders' on it, and the new pose deprojected it: 84 points on the other apple, closed on air."""
    from omnigibson.tiptop import r1pro
    from omnigibson.tiptop.r1pro import R1ProSim

    old, new, frames = th.zeros(2, 2, 3), th.full((2, 2, 3), 90.0), []
    q = th.tensor([0.0, 0.0, 0.0, 1.0])
    shadow = SimpleNamespace(pose=(th.zeros(3), q), renders=0)
    shadow.get_position_orientation = lambda: shadow.pose
    shadow.set_position_orientation = lambda position, orientation: setattr(shadow, "pose", (position, orientation))
    shadow.get_obs = lambda: ({"rgb": old if shadow.renders <= 4 else new}, {})  # 4 renders of the old view
    render = lambda: setattr(shadow, "renders", shadow.renders + 1)  # noqa: E731
    monkeypatch.setattr(r1pro, "og", SimpleNamespace(sim=SimpleNamespace(render=render)))
    monkeypatch.setattr(r1pro, "_intrinsics", lambda sensor: np.eye(3))
    cam = SimpleNamespace(get_position_orientation=lambda: (th.tensor([0.054, 0.0, 0.0]), q))
    sim = SimpleNamespace(robot_cams={"head": cam}, view_sensor=lambda name: shadow)
    obs, _ = R1ProSim._capture_obs(sim, "head")
    assert (obs["rgb"] == new).all() and shadow.renders == 8, "two agreeing frames of the new view, not the ghost"
    shadow.renders = 0  # the camera where it was: the first two agreeing frames are the view
    obs, _ = R1ProSim._capture_obs(sim, "head")
    assert shadow.renders == 4 and (obs["rgb"] == old).all()


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
    assert host.parse(raw, 3).sensors is None and host.observe_now().sensors is None, \
        "no segmenter yet (before the trial's providers exist): no frames"


def test_a_bench_step_obs_carries_the_cameras_current_frames_rendered_only_when_a_checker_reads_them():
    """SPEC §8 gate item 5: at the end of a run the GoalPanel judges the robot cameras' current images (the head and
    both wrists, where they stand), with the oracle segmenter's per-view masks of every tracked object, keyed by BDDL
    name and source-tagged. Reading sensors is not a capture: no sim step, no aim; and nothing renders until read."""
    from b1k.perception.verifier import frames

    n, rendered, asked = 4, [], []
    frame = {"rgb": np.zeros((n, n, 3), np.uint8), "depth": np.ones((n, n), np.float32),
             "intrinsics": np.eye(3, dtype=np.float32), "world_from_cam": np.eye(4, dtype=np.float32), "robot_mask": None}
    masks = {v: {"jar_1": np.eye(n, dtype=bool) if v == "left_wrist" else np.zeros((n, n), bool),
                 "cabinet_1": np.zeros((n, n), bool)} for v in ("head", "left_wrist", "right_wrist")}  # fmt: skip

    def segment(labels, request, extras):
        asked.append((labels, request["view_name"], [v["name"] for v in request["views"]], sorted(extras["views"])))
        return Provided(masks, "oracle", 9)

    sim = SimpleNamespace(robot=fake_robot(), n_steps=9, primary_view="head",
                          extra_views=("left_wrist", "right_wrist", "head_left"),
                          robot_cam_names={"head": "h", "left_wrist": "l", "right_wrist": "r"},
                          view_frame=lambda name: rendered.append(name) or (dict(frame), {"seg_instance": None}),
                          objects={"jar_1": 1, "cabinet_1": 2},
                          bddl_names={"jar_1": "jar.n.01_1", "cabinet_1": "cabinet.n.01_1"})  # fmt: skip
    host = BenchHost(sim, segmenter=SimpleNamespace(masks=segment))
    obs = host.parse(host.raw(), 3)
    assert isinstance(obs.sensors, Frames) and rendered == [] and asked == [], "made every step, rendered when read"
    assert list(obs.sensors.views) == ["head", "left_wrist", "right_wrist"], \
        "the cameras where they stand: a turned head view (head_left needs a torso ramp) is not one of them"
    assert rendered == ["head", "left_wrist", "right_wrist"] and sim.n_steps == 9, "one render per camera, no sim step"
    assert asked == [(["jar_1", "cabinet_1"], "head", ["left_wrist", "right_wrist"], ["left_wrist", "right_wrist"])], \
        "the segmenter over every tracked object in every view, in the capture's request shape"
    assert all(isinstance(v, CameraView) for v in obs.sensors.views.values())
    assert obs.sensors.masks.source == "oracle" and set(obs.sensors.masks.value) == {"jar.n.01_1", "cabinet.n.01_1"}, \
        "the primary view's masks, keyed by BDDL name as the goal atoms name the objects, source-tagged"
    assert obs.sensors.view_masks.source == "oracle" and obs.sensors.view_masks.value["left_wrist"]["jar.n.01_1"].any()
    assert [v.name for v, m, s in frames(obs.sensors)] == ["head", "left_wrist", "right_wrist"], \
        "the PerceptionVerifier reads it as it reads a Percept: every view"
    obs.sensors.views
    assert rendered == ["head", "left_wrist", "right_wrist"] and host.frames_wall_s > 0.0, "cached; its seconds counted"
    assert host.observe_now().sensors is not obs.sensors, "a new carrier per observation: it renders now, not then"


def view(name: str, n: int = 2) -> dict:  # one view as R1ProSim.capture's request carries it
    return {"name": name, "rgb": np.zeros((n, n, 3), np.uint8), "depth": np.ones((n, n), np.float32),
            "intrinsics": np.eye(3), "world_from_cam": np.eye(4)}  # fmt: skip


def capture_sim(n: int = 2) -> SimpleNamespace:
    """R1ProSim's capture (the primary view's request, a wrist view in it) and the oracle segmenter's masks: the apple
    in the head view only."""
    request = {**view("head", n), "view_name": "head", "views": [view("left_wrist", n)]}
    request.pop("name")
    extras = {"views": {"left_wrist": {}}}
    masks = {"head": {"apple_1": np.eye(n, dtype=bool), "basket_1": np.zeros((n, n), bool)},
             "left_wrist": {"apple_1": np.zeros((n, n), bool), "basket_1": np.zeros((n, n), bool)}}  # fmt: skip
    sim = SimpleNamespace(tracked_label=lambda n: n.replace(".n.01_", "_"), masks=masks, n_steps=0, looked=[],
                          segmenter=SimpleNamespace(masks=lambda labels, rq, ex: Provided(masks, "oracle", 40)),
                          primary_view="head", extra_views=("left_wrist",))  # fmt: skip

    def capture(task):
        sim.n_steps += 61  # R1ProSim.capture steps the sim itself
        return request, extras

    sim.capture, sim.look_at = capture, lambda *names: sim.looked.append(names)
    return sim


def observed(gen):
    with pytest.raises(StopIteration) as done:
        next(gen)
    return done.value.value


def test_the_capture_observer_runs_on_the_sim_clock_and_returns_the_planners_percept():
    host = SimpleNamespace(observe_now=lambda: StepObs(40, np.arange(61, dtype=np.float32), {}))
    sim = capture_sim()
    sim.masks["left_wrist"]["basket_1"][0, 0] = True  # the basket: only the wrist view sees it
    obs = CaptureObserver(sim, host, sim.segmenter, "t")
    assert obs.requires_sim_clock and obs.views == ("head", "left_wrist")
    gen = obs.observe(ObserveRequest((apple, basket), views=obs.views), None)
    with pytest.raises(StopIteration) as done:
        next(gen)  # ends before its first yield: 0 Runtime steps
    percept, after = done.value.value
    assert after.step == 40 and percept.info.visible == {apple.id: 1.0, basket.id: 0.0}, \
        "visible is read off the masks the Percept carries (the head's): the wrist-only basket is not"
    assert sim.looked == [(apple.id, basket.id)], "the capture framed the targets"
    assert obs.steps == 61, "the capture's own sim steps, for the bench's U0 check"
    assert percept.info.source == "oracle" and list(percept.views) == ["head", "left_wrist"]
    assert all(isinstance(v, CameraView) for v in percept.views.values()), "the Percept's view type"
    assert percept.masks.value == {apple.id: sim.masks["head"]["apple_1"], basket.id: sim.masks["head"]["basket_1"]}
    assert percept.masks.source == "oracle", "the primary view's masks, keyed by ObjRef.id"
    assert percept.view_masks.value == {"head": percept.masks.value,
                                        "left_wrist": {apple.id: sim.masks["left_wrist"]["apple_1"],
                                                       basket.id: sim.masks["left_wrist"]["basket_1"]}}, \
        "every view's masks too, keyed the same way, so a builder can send the wrist views"
    assert percept.view_masks.source == "oracle" and percept.view_masks.value["left_wrist"][basket.id][0, 0]
    assert (percept.q["trunk"] == np.arange(61)[PROPRIO_SLICES["trunk_qpos"]]).all() and obs.wall_s > 0.0
    percept, _ = observed(obs.observe(ObserveRequest((apple,), views=obs.views, context=(basket,)), None))
    assert sim.looked[-1] == (apple.id,), "the context is masked, never aimed at"
    assert set(percept.masks.value) == {apple.id, basket.id} and percept.view_masks.value["left_wrist"][basket.id][0, 0]
    with pytest.raises(NotImplementedError, match="captures"):  # R1ProSim captures what it was built with
        next(obs.observe(ObserveRequest((apple,), views=("head",)), None))
    with pytest.raises(NotImplementedError, match="captures"):
        next(obs.observe(ObserveRequest((apple,), views=obs.views, look_at=(1.0, 0.0, 0.8)), None))


def test_observe_without_aim_captures_from_where_the_head_is_and_moves_nothing(monkeypatch):
    from omnigibson.tiptop.host import capture_observer

    host = SimpleNamespace(observe_now=lambda: StepObs(40, np.arange(61, dtype=np.float32), {}))
    sim = capture_sim()
    request, extras = sim.capture("t")  # what the aimed capture renders (the fake counts 61 steps: aimed only)
    sim.n_steps, rendered = 0, []
    sim.capture = lambda task: pytest.fail("observe(aim=False) went through R1ProSim.capture: it aims and swings")
    monkeypatch.setattr(capture_observer, "TiptopSim",
                        SimpleNamespace(capture=lambda s, task: rendered.append((s, task)) or (request, extras)))
    obs = CaptureObserver(sim, host, sim.segmenter, "t")
    gen = obs.observe(ObserveRequest((apple, basket), views=obs.views, aim=False), None)
    with pytest.raises(StopIteration) as done:
        next(gen)
    percept, after = done.value.value
    assert rendered == [(sim, "t")], "TiptopSim.capture: the views as the cameras stand"
    assert sim.looked == [] and obs.steps == 0 and sim.n_steps == 0, "no look, no swing, no head turn: 0 sim steps"
    assert list(percept.views) == ["head", "left_wrist"] and percept.info.visible == {apple.id: 1.0, basket.id: 0.0}
    assert set(percept.view_masks.value) == {"head", "left_wrist"} and after.step == 40


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

    def open_up(self, name, fraction=None, single_round=False, joint=None):
        self.opened = (name, fraction, joint, single_round)
        return True

    def release(self):  # Episode.release returns nothing
        self.sim.n_steps += 45


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
    lb = LegacyBackend(episode, h.observe_now, has_cavity=lambda target, item: False, strict_relations=strict,
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


def test_a_legacy_success_names_what_the_hand_holds_so_the_next_precheck_agrees():
    from b1k.skills.specs import hand_empty, holding_obj
    from omnigibson.tiptop.oracle.world import OracleWorld

    env = Env()
    rt, ep, _ = legacy_rt(env)
    grasp = SimpleNamespace(held=lambda arm, obs: Provided(arm == "left", "proprio", 0))  # the left fingers closed
    world = OracleWorld(SimpleNamespace(sim=None), grasp)
    rt.svc = dataclasses.replace(rt.svc, world=world)
    conn = DirectConnector(rt, env.step, Host(env), env.raw())
    pick = conn.run(SkillCall("pick_up", PickArgs(apple), arm="left"))
    assert pick.world_updates == (WorldUpdate("held", apple, "left"),)
    place = SkillCall("place", PlaceArgs(apple, (Relation(Rel.ON, table),)), arm="left")
    assert holding_obj(place, rt.svc).ok, "a place after a legacy pick finds the apple in the hand"
    assert conn.run(place).world_updates == (WorldUpdate("released", apple, "left"),)
    assert world.hands["left"] == set()


def test_a_legacy_release_that_opened_the_hand_succeeds_when_nothing_judges_it():
    env = Env()
    rt, ep, _ = legacy_rt(env, truth={"hand_empty": None})  # goal_checker none, or a checker that cannot judge
    r = LegacyBackend(ep, Host(env).observe_now)
    gen = r.run(SkillCall("release", ReleaseArgs(), arm="left"), rt.svc, None)
    with pytest.raises(StopIteration) as done:
        next(gen)
    result = done.value.value
    assert (result.status, result.evidence["legacy_ok"], result.steps) == (Status.SUCCEEDED, True, 45), result
    assert result.world_updates == (WorldUpdate("released", None, "left"),)
    from omnigibson.tiptop.host.legacy_skills import FAILED_AS

    assert FAILED_AS["release"] is Code.BLOCKED, "a release that did not let go is not a wrong placement"


def test_a_legacy_open_takes_the_calls_joint_and_fraction():
    env = Env()
    rt, ep, _ = legacy_rt(env)
    call = SkillCall("open", OpenArgs(ObjRef("cabinet.n.01_1", "cabinet", True), joint="j_link_2", min_fraction=0.5))
    gen = LegacyBackend(ep, Host(env).observe_now, single_round=True).run(call, rt.svc, None)
    with pytest.raises(StopIteration):
        next(gen)
    assert ep.opened == ("cabinet.n.01_1", 0.5, "j_link_2", True)


def test_the_benchs_tiptop_pick_refuses_what_its_planner_cannot_plan_before_planning():
    tiptop = skillbench.make_backends(FakeEpisode(Host(Env())), Host(Env()), SimpleNamespace())["tiptop"]
    call = SkillCall("pick_up", PickArgs(apple), arm="left", freeze_trunk=True)
    assert tiptop.check(call, None).code is Code.UNSUPPORTED, "r1pro_left moves the torso"


def test_the_bench_finds_its_backends_by_file_so_a_new_skill_never_edits_it():
    backends = skillbench.make_backends(FakeEpisode(Host(Env())), Host(Env()), SimpleNamespace())
    tiptop, scripted = backends["tiptop"], backends["scripted"]
    assert {"pick_up", "open", "close"} <= set(tiptop.builders), "builders/pick.py and builders/articulate.py"
    assert set(tiptop.checks) >= {"pick_up", "open"}, "a builder's check goes with it"
    assert set(tiptop.stops) >= {"open", "close"} and set(tiptop.updates) >= {"open", "close"}, \
        "and its stall stop and world updates (articulate.stop_state, articulate.world_updates)"
    assert tiptop.supports(SkillCall("open", OpenArgs(basket))) and not tiptop.supports(SkillCall("wait", None))
    assert scripted.supports(SkillCall("wait", None)) and scripted.name == "scripted"
    assert skillbench.SPECS is __import__("b1k.skills.specs", fromlist=["SPECS"]).SPECS, "one SPECS, in b1k"


def test_the_bench_flags_an_in_the_legacy_wire_bends_onto_on():
    geometry = SimpleNamespace(cavity=lambda t, item: Provided("floor" if t == basket else None, "oracle", 0))
    ep = SimpleNamespace(sim=SimpleNamespace(send_inside=False))
    legacy = skillbench.make_backends(ep, Host(Env()), SimpleNamespace(geometry=geometry))["legacy"]
    into = lambda t: SkillCall("place", PlaceArgs(apple, (Relation(Rel.IN, t),)))
    assert legacy._bent(into(basket)) == [Rel.IN], "without --inside-region every in() lands on the hull top"
    ep.sim.send_inside = True
    assert legacy._bent(into(basket)) == [] and legacy._bent(into(table)) == [Rel.IN]


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
    ("TiptopPlanningError: the start posture is in self-collision (the human's idle right arm; cuRobo self-collision "
     "cost 4.01)", "place", (Status.INFEASIBLE, Code.NO_MOTION, "motion")),  # the server's S1 precheck: no place ran
])  # fmt: skip
def test_the_one_classifier_maps_todays_error_text_to_codes(why, skill, expected):
    assert classify(why, skill) == expected


def test_a_legacy_crash_before_any_sim_step_is_a_backend_fault_not_a_wrong_placement():
    """brisket_legacy: Episode.achieve raised ValueError (the burner is not a task object) before the sim stepped, and
    the row read placed_wrong / execute for a place that never happened."""
    env = Env()
    rt, ep, _ = legacy_rt(env, truth={"ontop": False})

    def boom(atoms, arm="left"):
        raise ValueError("burner.n.01_1 is not a task object")

    ep.achieve = boom
    r = DirectConnector(rt, env.step, Host(env), env.raw()).run(SkillCall("place", PlaceArgs(apple, (Relation(Rel.ON, table),)), arm="left"))
    assert (r.status, r.code, r.phase, r.steps) == (Status.FAILED, Code.BACKEND_ERROR, None, 0), r
    assert "not a task object" in r.detail


def test_a_legacy_press_on_a_device_already_in_the_wanted_state_succeeds_in_zero_steps():
    """SPEC 6.5 on the legacy lane too: the dispatch pressed toggled_on unconditionally and flipped the device."""
    env = Env()
    rt, ep, _ = legacy_rt(env, truth={"toggled_on": True})
    rt.registry.routing["press"] = {"default": "legacy"}
    r = DirectConnector(rt, env.step, Host(env), env.raw()).run(
        SkillCall("press", PressArgs(ObjRef("radio.n.01_1", "radio"), want_on=True), arm="left"))
    assert (r.status, r.steps, r.detail) == (Status.SUCCEEDED, 0, "already in the wanted state"), r
    assert ep.sim.n_steps == 0 and not hasattr(ep, "atoms"), "achieve was never called"
    rt2, ep2, _ = legacy_rt(Env(), truth={"toggled_on": False})
    rt2.registry.routing["press"] = {"default": "legacy"}
    r2 = DirectConnector(rt2, Env().step, Host(Env()), Env().raw()).run(
        SkillCall("press", PressArgs(ObjRef("radio.n.01_1", "radio"), want_on=True), arm="left"))
    assert ep2.atoms == [{"predicate": "toggled_on", "args": ["radio.n.01_1"]}] and ep2.sim.n_steps == 30


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
    ep.spec = SimpleNamespace(opens={"cabinet.n.01_1": {"joint": "j_link_4", "fraction": 0.8, "height": 0.5}})
    Episode.open_up(ep, "cabinet.n.01_1", 0.5, single_round=True, joint="j_link_2")
    assert {k: opened[-1][k] for k in ("joint", "fraction", "height")} == {"joint": "j_link_2", "fraction": 0.5,
                                                                          "height": 0.5}, "the call wins over the hint"
    assert Episode.open_up(ep, "cabinet.n.01_1", 0.0, single_round=True, joint="j_link_2") is True
    assert pushed == [False], "a close of another joint pushes nothing"


def test_a_door_pulled_short_is_pushed_on_from_the_same_stance_in_a_single_round(monkeypatch):
    from omnigibson.tiptop import r1pro

    door = {"name": "j0", "kind": "revolute", "lower": 0.0, "upper": 1.5, "position": 0.0, "closed": 0.0,
            "link": "door"}  # fmt: skip
    reads = iter([[door], [dict(door, position=0.3)]])  # before the pull, and after it: short of 1.2
    monkeypatch.setattr(r1pro, "openable_joints", lambda obj: next(reads))
    pushed = []
    hand = SimpleNamespace(get_position_orientation=lambda: (th.zeros(3), th.tensor([0.0, 0.0, 0.0, 1.0])))
    run = {"grasp": {"joint": door, "travel": 1.2, "kind": "bar"}, "plan": {"reached": 5}, "waypoints": 5, "why": "",
           "held": True, "stance": None}  # fmt: skip
    sim = SimpleNamespace(scene_object=lambda n: object(), OPEN=1.0, robot=SimpleNamespace(eef_links={"left": hand}),
                          container_grasps=lambda *a, **k: ["g"], _drive_joint=lambda *a, stand: run,
                          push_joint=lambda arm, name, j, target, stand=True: pushed.append(stand) or {"position": 1.2})
    out = r1pro.R1ProSim.open_container(sim, "left", "fridge.n.01_1", fraction=0.8, stand=False)
    assert pushed == [False], "the push after a short pull chose a stance of its own: a base teleport in the skill"
    assert out["opened"] is True


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
    assert skillbench.row(case, 0, 3, r, rt, 40, 2.0, 0.5, 1.2)["planning_wall_s"] == 0.3, "the capture is not planning"
    bound = dataclasses.replace(r, binding=(Binding("robot_to_world", 3, 256),))
    assert skillbench.row(case, 0, 3, bound, rt, 40, 2.0, 0.5)["binding"] == [
        {"constraint": "robot_to_world", "satisfied": 3, "of": 256}]
    opened = dataclasses.replace(r, world_updates=(WorldUpdate("joint", basket, joint="j_link_4", value=0.31,
                                                               source="oracle"),))
    (u,) = skillbench.row(case, 0, 3, opened, rt, 40, 2.0, 0.5)["world_updates"]
    assert (u["kind"], u["joint"], u["value"], u["source"]) == ("joint", "j_link_4", 0.31, "oracle"), \
        "an open's joint value and its source reach the row (SPEC 6.6: theta and its source)"
    s = skillbench.summarize(case, [row, dict(row, status="failed", code="grasp_missed", u0=False)])
    assert (s["n"], s["succeeded"], s["rate"], s["expect_met"], s["u0_all"]) == (2, 1, 0.5, 1, False)
    assert (s["shadow_judged"], s["agree"]) == (0, 0), "no shadow checker judged: nothing agreed (None is no agreement)"
    shadow = [dict(row, verdicts={"scorer": True, "perception": None}), dict(row, verdicts={"scorer": True, "perception": True}),
              dict(row, verdicts={"scorer": True, "perception": False})]
    s = skillbench.summarize(case, shadow)
    assert (s["shadow_judged"], s["agree"]) == (2, 1), "two judged, one agrees; the None row counts in neither"


def test_a_native_trial_observes_once_and_every_env_step_is_charged_to_it():
    observer = SimClockObserver()
    r, rt, _, _ = trial("tiptop", observer)
    assert r.status is Status.SUCCEEDED and observer.n == 1
    assert rt.charged == {"observe": 0, "skill": 3} and rt.step == 3 and rt.idle_steps == 0 and skillbench.u0(rt)


class World(JointWorld):  # what the pick builder reads besides the hands: where the base stands, what the apple is on
    def base_pose(self):
        return Belief(Pose2(0.0, 0.0, 0.0), "oracle", 0)

    def support_of(self, o):
        return Belief(table, "oracle", 0)


class Planner:  # the planner server: records the request, answers a two-waypoint approach, the grasp and a lift
    def __init__(self):
        self.asked = []

    def skill(self, req):
        from tiptop.skills.wire import WIRE, PlanStep, SkillResponse, StopRule

        self.asked.append(req)
        names = tuple(f"torso_joint{k}" for k in range(1, 5)) + tuple(f"left_arm_joint{k}" for k in range(1, 8))
        plan = (PlanStep("trajectory", "approach", positions=((0.1,) * 11, (0.2,) * 11), dt=1 / 30),
                PlanStep("gripper", "grasp", gripper="creep", stop=StopRule("width", 0.006, "grasp_missed")),
                PlanStep("trajectory", "lift", positions=((0.3,) * 11,), dt=1 / 30))  # fmt: skip
        return SkillResponse(WIRE, True, None, "lift", "", (), names, plan, (("holding", "apple_1"),), {})


def test_a_native_pick_on_the_bench_plans_from_the_capture_and_plays_the_plan():
    env, planner, sim = Env(), Planner(), capture_sim(4)
    host = Host(env)
    base = make_rt(planner=planner)
    svc = dataclasses.replace(base.svc, world=World(REFS))
    observer = CaptureObserver(sim, host, sim.segmenter, "t")
    call = SkillCall("pick_up", PickArgs(apple), arm="left", seed=3, backend="tiptop")
    r, rt, _, _ = skillbench.run_trial(host, svc, skillbench.make_backends(FakeEpisode(host), host, svc),
                                       {"pick_up": {"default": "legacy"}}, observer, call)  # fmt: skip
    assert (r.status, r.backend, r.phase) == (Status.SUCCEEDED, "tiptop", "lift"), r
    (req,) = planner.asked
    assert req.skill == "pick" and req.seed == 3 and req.observation["view_name"] == "head"
    assert req.observation["gt_labels"] == ["apple_1"] and np.array_equal(req.observation["gt_masks"][0], np.eye(4))
    assert rt.charged == {"observe": 0, "skill": r.steps} and r.steps > 0 and skillbench.u0(rt)


def test_a_trial_starts_from_what_the_setup_left_commanded_so_a_held_hand_stays_closed_through_a_dwell():
    host, acts = Host(Env()), []  # Host.commanded_targets: gripper_left CLOSED, as a demo restore leaves it
    host.env_step = lambda a: (acts.append(a.copy()), host.env.step(a))[1]
    svc = make_rt().svc
    r, rt, _, _ = skillbench.run_trial(host, svc, skillbench.make_backends(FakeEpisode(host), host, svc),
                                       {"wait": {"default": "scripted"}}, SimClockObserver(),
                                       SkillCall("wait", WaitArgs(2)))  # fmt: skip
    assert r.status is Status.SUCCEEDED and len(acts) == 2 and skillbench.u0(rt)
    assert all(a[ACTION_SLICES["gripper_left"]] == CLOSED for a in acts), \
        "Latch opens both hands at its first observation; the trial starts from what the setup left commanded"


def test_the_step_invariant_fails_on_an_idle_step_or_an_uncharged_one():
    rt = SimpleNamespace(step=3, idle_steps=0, charged={"skill": 3})
    assert skillbench.u0(rt)
    assert skillbench.u0(rt, sim_steps=64, observe_steps=61), "the sim: the run's 3 steps and the capture's 61"
    assert not skillbench.u0(rt, sim_steps=65, observe_steps=61), "a step nothing charged: something planned on it"
    assert not skillbench.u0(SimpleNamespace(step=3, idle_steps=1, charged={"skill": 2}))
    assert not skillbench.u0(SimpleNamespace(step=3, idle_steps=0, charged={"skill": 2}))


def test_a_native_row_fails_u0_on_a_sim_step_nobody_charged():
    r, rt, _, _ = trial("tiptop")
    case = {"id": "c", "call": SkillCall("pick_up", PickArgs(apple))}
    assert skillbench.row(case, 0, 3, r, rt, rt.step + 61, 2.0, 0.5, 1.0, 61)["u0"]
    assert not skillbench.row(case, 0, 3, r, rt, rt.step + 62, 2.0, 0.5, 1.0, 61)["u0"]


def test_the_setup_frames_the_calls_objects_after_the_teleport(monkeypatch):
    events = []
    monkeypatch.setattr(skillbench, "apply_embodiment_posture", lambda sim, args, emb: events.append("torso"))
    monkeypatch.setattr(skillbench, "aim_overview", lambda sim, *pose, target=None: events.append(("aim", target)))
    sim = SimpleNamespace(OPEN=1.0, place_robot=lambda *pose, note: events.append("place"),
                          look_at=lambda *names: events.append(("look", names)),
                          hold=lambda n, g: events.append("hold"))  # fmt: skip
    og = SimpleNamespace(sim=SimpleNamespace(dump_state=lambda serialized: "physics"))
    case = {"id": "c", "setup": {"robot_pose": [1.0, 2.0, 0.0]},
            "call": SkillCall("place", PlaceArgs(apple, (Relation(Rel.IN, basket),)))}
    skillbench.setup(og, sim, SimpleNamespace(settle_steps=3), case, {})
    assert events == ["torso", "place", ("aim", apple.id), ("look", (apple.id, basket.id)), "hold"], \
        "place_robot clears the look target: the capture would aim at the default point, not at the objects; the " \
        "overview camera is aimed at the call's object after the teleport"
    events.clear()
    case["call"] = SkillCall("release", ReleaseArgs())
    skillbench.setup(og, sim, SimpleNamespace(settle_steps=3), case, {})
    assert events == ["torso", "place", ("aim", None), "hold"], \
        "a call that names no object frames nothing (look_at() would raise); the overview looks at the workspace"
    events.clear()
    sim.scene_object = lambda name: SimpleNamespace(joints={"j_link_4": SimpleNamespace(
        set_pos=lambda v: events.append(("joint", name, v)))})
    case["setup"]["joint_states"] = {"cabinet.n.01_1": {"j_link_4": 0.31}}
    skillbench.setup(og, sim, SimpleNamespace(settle_steps=3), case, {})
    assert events == ["torso", "place", ("aim", None), ("joint", "cabinet.n.01_1", 0.31), "hold"], \
        "a drawer to close starts open: the joint set after the teleport, the settle propagates it"


def test_the_overview_camera_takes_the_first_eye_that_sees_the_robot_and_the_target(monkeypatch):
    """VIDEO_FINDINGS 9: the shoulder eye sat behind a pillar at the jar stance and looked at a window at the drawer
    stance. Each candidate eye is raycast at the robot's chest and at the call's target; the first clear one wins."""
    aimed, walls = [], {}
    key = lambda p: tuple(round(float(v), 2) for v in p)

    def raytest(start, end):  # a wall between some eyes and what they look at
        body = walls.get((key(start), key(end)))
        return {"hit": True, "distance": 0.5, "rigidBody": body} if body else {"hit": False}

    monkeypatch.setattr(overview, "raytest", raytest)
    jar = SimpleNamespace(prim_path="/World/scene_0/jar_7", aabb_center=th.tensor([2.0, 2.3, 0.9]))
    sim = SimpleNamespace(overview_view="shoulder", aim_overview=lambda eye, look: aimed.append((key(eye), key(look))),
                          robot=SimpleNamespace(prim_path="/World/scene_0/robot0"), scene_object=lambda n: jar)  # fmt: skip
    shoulder, right, chest, look = (-0.5, 3.1, 1.7), (-0.5, 0.9, 1.7), (1.0, 2.0, 1.1), (2.0, 2.3, 0.9)  # base (1, 2, yaw 0)
    overview.aim_overview(sim, 1.0, 2.0, 0.0, target="jar.n.01_1")
    assert aimed[-1] == (shoulder, look), "every ray clear: the --overview eye as place_robot puts it, at the target"
    walls[(shoulder, look)] = "/World/scene_0/walls_pillar"
    overview.aim_overview(sim, 1.0, 2.0, 0.0, target="jar.n.01_1")
    assert aimed[-1] == (right, look), "a pillar hides the jar from the shoulder eye: the next eye, over the right"
    walls[(right, chest)] = "/World/scene_0/window_3"
    overview.aim_overview(sim, 1.0, 2.0, 0.0, target="jar.n.01_1")
    assert aimed[-1][0] == (2.15, 2.75, 1.35), "a window between that eye and the robot: the eye ahead-left"
    walls[(shoulder, look)], walls[(right, chest)] = jar.prim_path, sim.robot.prim_path
    overview.aim_overview(sim, 1.0, 2.0, 0.0, target="jar.n.01_1")
    assert aimed[-1] == (shoulder, look), "a hit on the target's own body, or on the robot's toward its chest, is clear"
    walls[(shoulder, look)] = "/World/scene_0/walls_pillar"
    for e in overview.OVERVIEW_EYES:
        walls[(key((1.0 + e[0], 2.0 + e[1], e[2])), chest)] = "/World/scene_0/walls"
    overview.aim_overview(sim, 1.0, 2.0, 0.0, target="jar.n.01_1")
    assert aimed[-1] == (shoulder, look), "every eye blocked: the --overview eye, as before"
    overview.aim_overview(sim, 1.0, 2.0, 0.0)
    assert aimed[-1] == (shoulder, (1.7, 2.0, 0.55)), "no target named (a release): the view's own workspace point"
    assert skillbench.aim_overview is overview.aim_overview, "the bench's setup aims through it (harness-only: HARNESS_SETUP)"


class DemoSim:
    """R1ProSim as a demo setup sees it: the planned left arm, an idle right arm with a nominal posture, both grippers,
    the task scope (BDDL name -> object) and the fingers the fake robot reports."""

    OPEN = 1.0
    overview_view = "shoulder"

    def aim_overview(self, eye, target):
        self.events.append(("overview", tuple(round(v, 3) for v in eye), tuple(round(v, 3) for v in target)))

    def scene_object(self, name):
        return SimpleNamespace(prim_path=f"/World/{name}", aabb_center=th.tensor([1.5, 2.0, 0.9]))

    def __init__(self, finger_sum=0.02, phantom_qvel=0.0):
        self.arm, self.other_arm, self.n_steps = "left", "right", 0
        self.robot = SimpleNamespace(arm_joint_names={a: [f"{a}_arm_joint{k}" for k in range(1, 8)]
                                                      for a in ("left", "right")}, prim_path="/World/robot0")  # fmt: skip
        self.posture = {f"right_arm_joint{k}": 0.0 for k in range(1, 8)} | {"left_gripper_finger_joint1": 0.05}
        self.locked_nominal = {f"right_arm_joint{k}": 0.0 for k in range(1, 8)}
        self.stance_ready, self.last_gripper, self.other_gripper, self.held_objects = None, 1.0, 1.0, {}
        self.env, self.events, self.finger_sum, self.phantom_qvel = "env", [], finger_sum, phantom_qvel
        self.scope = {"log.n.01_2": SimpleNamespace(name="log_176"), "block.n.01_1": SimpleNamespace(name="block_9")}

    def task_scope(self):
        return self.scope

    def tracked_label(self, bddl):
        return bddl.replace(".n.01_", "_")

    def q_arm(self):
        return np.full(11, 0.3)

    def look_at(self, *names):
        self.events.append(("look", names))

    def hold(self, n, gripper):
        self.events.append(("hold", n, gripper))
        self.last_gripper, self.n_steps = gripper, self.n_steps + n

    def commanded_targets(self):
        return {"base": np.zeros(3), "trunk": np.zeros(4), "arm_left": np.zeros(7), "arm_right": np.zeros(7),
                "gripper_left": np.array([self.last_gripper]), "gripper_right": np.array([self.other_gripper])}

    def proprio(self):
        p = np.zeros(61, dtype=np.float32)
        p[PROPRIO_SLICES["arm_right_qpos"]] = np.arange(7) * 0.1 + 0.5  # the human's right arm, as restored
        p[PROPRIO_SLICES["gripper_left_qpos"]] = self.finger_sum / 2  # the left fingers, settled on the log or on air
        p[PROPRIO_SLICES["gripper_left_qvel"]] = self.phantom_qvel  # what PhysX says they do meanwhile
        return p


def demo_case(held={"left": "log_176"}):
    return {"id": "d", "instance": 188, "mode": "train", "setup": {"robot_pose": [1.0, 2.0, 0.0], "held": dict(held)},
            "demo": {"snapshot": "snap.json"},
            "call": SkillCall("place", PlaceArgs(ObjRef("log.n.01_2", "log"), (Relation(Rel.ON, basket),)))}


@pytest.fixture
def demo_env(monkeypatch, tmp_path):
    restored = []
    monkeypatch.setattr(skillbench.demo_cases, "restore",
                        lambda env, snap, inst, mode: restored.append((env, snap, inst, mode)))
    monkeypatch.setattr(skillbench.demo_cases, "held", lambda robot: {"left": "log_176", "right": None})
    monkeypatch.setattr(overview, "raytest", lambda start, end: {"hit": False})  # every overview eye clear
    (tmp_path / "snap.json").write_text('{"log_176": {"pos": [1, 2, 3]}}')
    og = SimpleNamespace(sim=SimpleNamespace(dump_state=lambda serialized: "physics"))
    return og, restored, tmp_path


def test_a_demo_case_setup_restores_the_snapshot_adopts_the_human_posture_and_verifies_the_hold(demo_env):
    og, restored, tmp_path = demo_env
    sim = DemoSim()
    host = SimpleNamespace(proprio=sim.proprio, observe_now=lambda: StepObs(0, sim.proprio(), {}))
    case = demo_case()
    case["demo"]["snapshot"] = str(tmp_path / "snap.json")
    state, held = skillbench.setup(og, sim, SimpleNamespace(settle_steps=3, mode="public_test"), case, {}, host)
    assert restored == [("env", {"log_176": {"pos": [1, 2, 3]}}, 188, "train")], \
        "demo_cases.restore: instance + snapshot, of the case's mode (a demo's is a training instance)"
    assert held == {"left": case["call"].args.obj}, "setup.held names the sim object; the hand holds the call's ObjRef"
    assert sim.posture["right_arm_joint3"] == pytest.approx(0.7), \
        "the idle arm is locked where the human left it: no hold or capture drives it to the nominal posture"
    assert sim.locked_nominal["right_arm_joint7"] == pytest.approx(1.1), "and restore_locked_arm has nothing to undo"
    assert sim.posture["left_gripper_finger_joint1"] == 0.05, "only the idle arm's joints change"
    assert sim.stance_ready == [0.3] * 11, "the planned arm's ready posture is where it stands"
    assert (sim.last_gripper, sim.other_gripper) == (CLOSED, 1.0) and sim.held_objects == {"log_2": "left"}
    assert sim.events == [("overview", (-0.5, 3.1, 1.7), (1.5, 2.0, 0.9)), ("look", ("log.n.01_2", basket.id)),
                          ("hold", 3, CLOSED)], \
        "the overview camera over the shoulder as place_robot puts it (the restore put the base down, no teleport), " \
        "at the call's object; the settle keeps the hand closed"
    assert state[1]["last_gripper"] == CLOSED and state[1]["held_objects"] == {"log_2": "left"}
    assert skillbench.pick_arm(case["call"], held) == "left", "the hand that holds the object places it"
    assert skillbench.pick_arm(SkillCall("pick_up", PickArgs(apple)), held) == "right", "a free hand picks"
    assert skillbench.pick_arm(SkillCall("pick_up", PickArgs(apple), arm="left"), held) == "left", "the call's arm wins"
    assert skillbench.pick_arm(SkillCall("pick_up", PickArgs(apple)), {}) == "left"


def test_a_demo_case_whose_hand_does_not_hold_after_the_restore_fails_its_setup_with_the_reason(demo_env):
    og, restored, tmp_path = demo_env
    sim = DemoSim(finger_sum=0.0)  # the fingers closed on air: the grasp did not come back
    host = SimpleNamespace(proprio=sim.proprio, observe_now=lambda: StepObs(0, sim.proprio(), {}))
    case = demo_case()
    case["demo"]["snapshot"] = str(tmp_path / "snap.json")
    case["instance"], case["mode"] = 301, "public_test"  # a manip run's stance: its snapshot is of a public_test instance
    with pytest.raises(skillbench.CaseSetupError, match=r"\['left'\] hand does not hold \['log.n.01_2'\].*proprio"):
        skillbench.setup(og, sim, SimpleNamespace(settle_steps=3, mode="train"), case, {}, host)
    assert restored == [("env", {"log_176": {"pos": [1, 2, 3]}}, 301, "public_test")], \
        "it did restore first, the case's own mode over the bench's; the proprio GraspSensor refused what it found"
    refs = skillbench.held_refs(sim, {"right": "wicker_basket_92"}, case["call"])
    assert refs == {"right": ObjRef("wicker_basket_92", "wicker_basket", False)}, "an object outside the task scope"
    block = skillbench.held_refs(sim, {"right": "block_9"}, case["call"])["right"]
    assert block == ObjRef("block.n.01_1", "block", False), "a task object the call does not name: its BDDL name"


def test_a_hold_whose_welded_finger_reports_a_phantom_velocity_verifies_over_a_window(demo_env):
    """make_microwave_popcorn (W2-P probe): the fingers unchanged at [0.0497, 0.0] m while PhysX reports 5 cm/s, so
    the sensor's one-shot velocity test never resolved and 15 held setups were refused. The setup reads the sensor
    over its window, one physics step apart, and the sensor settles by position."""
    og, restored, tmp_path = demo_env
    sim = DemoSim(finger_sum=0.0497, phantom_qvel=0.05)
    host = SimpleNamespace(proprio=sim.proprio, observe_now=lambda: StepObs(sim.n_steps, sim.proprio(), {}))
    case = demo_case()
    case["demo"]["snapshot"] = str(tmp_path / "snap.json")
    state, held = skillbench.setup(og, sim, SimpleNamespace(settle_steps=3, mode="public_test"), case, {}, host)
    assert held == {"left": case["call"].args.obj}
    assert sim.events[-3:] == [("hold", 3, CLOSED), ("hold", 1, CLOSED), ("hold", 1, CLOSED)], \
        "the settle, then one step per further reading until the window was full"


def test_the_self_test_spread_is_the_widest_range_over_the_trials():
    p = np.zeros(61)
    rows = [{"proprio": p.tolist(), "objects": {"log_176": [0.0, 0.0, 0.0], "apple_1": [1.0, 0.0, 0.0]}},
            {"proprio": (p + np.eye(61)[PROPRIO_SLICES["arm_left_qpos"].start + 2] * 0.002).tolist(),
             "objects": {"log_176": [0.0, 0.0, 0.0], "apple_1": [1.0, 0.003, 0.004]}}]  # fmt: skip
    s = skillbench.spread(rows)
    assert s["proprio"]["arm_left"] == pytest.approx(0.002) and s["proprio"]["trunk"] == 0.0
    assert set(s["proprio"]) == {"trunk", "arm_left", "arm_right", "gripper_left", "gripper_right"}
    assert s["objects_m"] == {"log_176": 0.0, "apple_1": pytest.approx(0.005)}
    args = skillbench.parse_args(["--out-dir", "o", "--case", "c.yaml", "--setup-only", "--selftest", "5"])
    assert args.setup_only and args.selftest == 5


def test_every_trial_restores_the_command_state_the_physics_state_does_not_carry():
    events = []
    og = SimpleNamespace(sim=SimpleNamespace(dump_state=lambda serialized: "physics",
                                             load_state=lambda st, serialized: events.append(("load", st))))
    sim = SimpleNamespace(OPEN=1.0, posture={"right_arm_joint4": 0.0}, look_target=np.array([0.6, 0.1, 0.8]),
                          stance_ready=None, seen_boxes={"x": 1}, last_gripper=1.0, other_gripper=CLOSED,
                          held_objects={"log_2": "right"})  # fmt: skip
    sim.hold = lambda n, g: events.append(("hold", dict(sim.posture), g))
    sim.begin_episode = lambda name: events.append(("begin", name)) or setattr(sim, "held_objects", {})
    state = skillbench.snapshot(og, sim)
    sim.posture["right_arm_joint4"] = -1.9  # trial 0 tucked the idle arm (tuck_idle_arm), or a ramp stopped
    sim.look_target, sim.stance_ready = None, [0.1] * 11
    sim.last_gripper, sim.other_gripper, sim.held_objects = CLOSED, 1.0, {}  # trial 0 picked with the left, dropped
    skillbench.restore(og, sim, state, "t1")
    assert sim.posture == {"right_arm_joint4": 0.0} and np.allclose(sim.look_target, [0.6, 0.1, 0.8])
    assert sim.stance_ready is None and sim.seen_boxes == {}
    assert events == [("load", "physics"), ("hold", {"right_arm_joint4": 0.0}, 1.0), ("begin", "t1")], \
        "the restore's own step commands the restored posture and gripper, not the last trial's"
    assert (sim.last_gripper, sim.other_gripper) == (1.0, CLOSED), "a demo's held right hand stays closed"
    assert sim.held_objects == {"log_2": "right"}, "begin_episode cleared the hand record; the setup's hands refill it"
    bare = SimpleNamespace(OPEN=1.0, seen_boxes={}, hold=lambda n, g: events.append(("bare", g)),
                           begin_episode=lambda name: None)  # a sim that never had a gripper command
    skillbench.restore(og, bare, skillbench.snapshot(og, bare), "t2")
    assert events[-1] == ("bare", 1.0), "no command recorded: open"


def test_the_latch_gap_is_how_far_the_next_action_would_snap_an_arm():
    p = np.zeros(61, dtype=np.float32)
    p[PROPRIO_SLICES["arm_left_qpos"]] = 0.9
    target = np.zeros(23, dtype=np.float32)
    rt = SimpleNamespace(obs=StepObs(0, p, {}), latch=SimpleNamespace(target=target))
    assert skillbench.latch_gap(rt) == pytest.approx(0.9)


def test_the_bench_cases_load_as_skill_calls_and_a_setup_the_bench_cannot_do_is_refused(tmp_path):
    bench = ROOT / "tiptop/b1k/skills/bench"
    cases = {c["id"]: c for f in ("pick_up.yaml", "open.yaml", "close.yaml") for c in skillbench.load_cases(bench / f)}
    assert {"pick_up_freeze_fruit_apple", "pick_up_store_honey_jar", "open_store_honey_drawer",
            "open_store_batteries_drawer", "close_store_honey_drawer", "open_storing_food_fancyy_door2",
            "close_storing_food_fancyy_door2"} <= set(cases)
    shut = cases["close_store_honey_drawer"]
    assert isinstance(shut["call"].args, CloseArgs) and shut["setup"]["joint_states"] == {"cabinet.n.01_1": {"j_link_4": 0.31}}
    door = cases["close_storing_food_fancyy_door2"]  # a door: the same case form, the joint in radians
    assert door["call"].args.joint == "j_door2" and door["setup"]["joint_states"] == {"cabinet.n.01_3": {"j_door2": 0.7}}
    assert cases["open_store_batteries_drawer"]["call"].args.joint == "j_link_5"
    apple_case = cases["pick_up_freeze_fruit_apple"]
    assert (apple_case["task"], apple_case["instance"], apple_case["seeds"]) == ("freeze_fruit", 301, [0, 1, 2, 3, 4])
    assert apple_case["call"] == SkillCall("pick_up", PickArgs(ObjRef("apple.n.01_2", "apple")), arm="left")
    drawer = cases["open_store_honey_drawer"]["call"]
    assert isinstance(drawer.args, OpenArgs) and drawer.args.target.fixed and drawer.args.joint == "j_link_4"
    only = skillbench.load_cases(bench / "pick_up.yaml", ids=["pick_up_store_honey_jar"])
    assert [c["id"] for c in only] == ["pick_up_store_honey_jar"]
    bad = tmp_path / "bad.yaml"
    first = yaml.safe_load((bench / "pick_up.yaml").read_text())[0]
    bad.write_text(yaml.safe_dump([{**first, "setup": {"object_poses": {}}}]))
    with pytest.raises(ValueError, match="not on the bench yet"):
        skillbench.load_cases(bad)
    bad.write_text(yaml.safe_dump([{**first, "setup": {"held": {"left": "apple_2"}}}]))
    with pytest.raises(ValueError, match="needs demo.snapshot"):
        skillbench.load_cases(bad)
    bad.write_text(yaml.safe_dump([{**{k: v for k, v in first.items() if k != "mode"}, "demo": {"snapshot": "s.json"}}]))
    with pytest.raises(ValueError, match="names the mode"):
        skillbench.load_cases(bad), "a mode-less snapshot case fell back to --mode (public_test) over a train snapshot"


def test_the_demo_cases_load_with_their_snapshot_mode_and_hands():
    bench = ROOT / "tiptop/b1k/skills/bench"
    groups = ("pick_up", "place_on", "place_in", "open_drawer", "open_door", "press")
    cases = [c for g in groups for c in skillbench.load_cases(bench / f"demo_{g}.yaml")]
    assert len(cases) == 45 and len({c["id"] for c in cases}) == 45, "all 45 on the bench"
    assert all(c["mode"] == "train" and Path(c["demo"]["snapshot"]).is_file() for c in cases), \
        "every demo case restores a snapshot the bench ships (a path relative to the case file, resolved)"
    held = [c for c in cases if (c["setup"].get("held"))]
    assert len(held) == 22 and all(set(c["setup"]["held"]) <= {"left", "right"} for c in held)
    log = next(c for c in cases if c["id"] == "place_on.chopping_wood.e8933.f625")
    assert log["setup"]["held"] == {"left": "log_176"} and isinstance(log["call"].args, PlaceArgs)
    assert log["call"].args.obj.id == "log.n.01_2" and log["call"].arm is None
    assert log["demo"]["arms"]["left"][0] == pytest.approx(-0.6772) and len(log["demo"]["fingers"]["right"]) == 2
    pizza = next(c for c in yaml.safe_load((bench / "demo_place_on.yaml").read_text()) if c["task"] == "make_pizza")
    assert "gpu_dynamics" not in pizza and "skip" not in pizza, \
        "with the evaluator's flags (rules disabled, GPU dynamics off) the scene loads and restores (fix pass " \
        "2026-09-26, week2/fix/pizza_setup); the flag was the PhysX GPU crash it was skipped for"
    skipped = [{**pizza, "id": "x", "skip": "why"}]
    with_skip = yaml.safe_dump(skipped)
    tmp = bench / "..tmp_skip.yaml"
    try:
        tmp.write_text(with_skip)
        assert skillbench.load_cases(tmp) == [], "a skip: case is left out"
    finally:
        tmp.unlink()


def test_the_place_gate_cases_are_the_positive_control_and_the_crouch_reset_to_ready():
    """W3-A2: the positive control legacy passes (tidying_bedroom's book onto the nightstand), and chopping_wood's
    demo situation with the harness's posture reset, since S1 refuses the human's crouch (3.89)."""
    bench = ROOT / "tiptop/b1k/skills/bench"
    cases = {c["id"]: c for c in skillbench.load_cases(bench / "place_on.yaml")}
    tidy, chop = cases["place_on.tidying_bedroom.e3675.f5216"], cases["place_on.chopping_wood.e8933.f625.ready"]
    assert Path(tidy["demo"]["snapshot"]).is_file() and tidy["setup"]["held"] == {"left": "hardback_188"}
    demo = next(c for c in skillbench.load_cases(bench / "demo_place_on.yaml") if c["id"] == chop["id"][: -len(".ready")])
    assert chop["setup"] == {**demo["setup"], "ready": "left"} and chop["demo"]["snapshot"] == demo["demo"]["snapshot"]


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


def test_the_one_call_planner_observes_the_movables_near_a_pick_beside_it_and_aims_at_the_object_alone():
    """w2s2: the Percept masked the apple alone, so the bowl it lies in was in no world and the exempt support never
    reached cuTAMP; week-3 g1 apple_1 t0 and t3: the approach turned that unseen bowl over and closed on air. The
    movables near the object are observed beside it; far ones and furniture (the map room's) are not."""
    from b1k.connector.types import AABB

    seen = []
    boxes = {apple.id: AABB((0.0, 0.0, 0.9), (0.08, 0.08, 0.98)), basket.id: AABB((0.1, 0.0, 0.88), (0.3, 0.2, 1.0)),
             "radio.n.01_1": AABB((1.0, 1.0, 0.9), (1.1, 1.1, 1.0)), table.id: AABB((-1, -1, 0), (1, 1, 0.88))}
    world = SimpleNamespace(box=lambda o: Belief(boxes.get(o.id), "oracle", 0),
                            objects=lambda: [apple, basket, radio, table])  # fmt: skip
    conn = SimpleNamespace(
        check=lambda call: SimpleNamespace(code=Code.PERCEPT_REQUIRED if call.percept is None else None),
        observe=lambda req: seen.append(req) or SimpleNamespace(id="c1"), run=lambda call: call, world=lambda: world,
    )  # fmt: skip
    skillbench.one_call(conn, SkillCall("pick_up", PickArgs(apple)))
    assert seen[-1].targets == (apple,) and seen[-1].context == (basket,), "2 cm away: framed; the radio, the table not"
    boxes.pop(apple.id)  # never localized: nothing to measure from
    skillbench.one_call(conn, SkillCall("pick_up", PickArgs(apple)))
    assert seen[-1].context == ()
    skillbench.one_call(conn, SkillCall("place", PlaceArgs(apple, (Relation(Rel.ON, table),))))
    assert seen[-1].context == (), "a place frames its own target"


# -------------------------------------------------------------------------------------------- the oracle providers
def test_the_oracle_segmenter_gives_a_touching_neighbours_contact_band_to_the_neighbour():
    """w2s2 freeze_fruit: masked alone, apple_1 took 18 pixels of the apple_2 it leans on (within gt_masks' 8 mm of
    its surface), which made its cloud 2 cm taller: the pick got side grasps through the bowl (apple_1 t3)."""
    import trimesh

    from omnigibson.tiptop.gt_masks import masks_from_geometry
    from omnigibson.tiptop.oracle.segmenter import OracleSegmenter

    r, n, f = 0.04, 128, 400.0
    centres = {"apple_1": np.array([-r, 0.0, 0.6]), "apple_2": np.array([r, 0.0, 0.6])}  # touching, side by side
    far = {"plate_1": np.array([1.0, 0.0, 0.6])}
    k = np.array([[f, 0.0, n / 2], [0.0, f, n / 2], [0.0, 0.0, 1.0]])
    u, v = np.meshgrid(np.arange(n) - n / 2, np.arange(n) - n / 2)
    d = np.stack([u / f, v / f, np.ones_like(u)], -1)
    d /= np.linalg.norm(d, axis=-1, keepdims=True)
    t = np.full((n, n), np.inf)
    for c in centres.values():  # the camera at the origin looking +z: the nearer sphere hit per pixel
        b = (d @ c) ** 2 - (c @ c - r * r)
        t = np.where(b > 0, np.minimum(t, d @ c - np.sqrt(np.clip(b, 0, None))), t)
    depth = np.where(np.isfinite(t), t * d[..., 2], 0.0)
    boxes = {**centres, **far}
    objects = {l: SimpleNamespace(aabb=(th.tensor(c - r), th.tensor(c + r))) for l, c in boxes.items()}
    meshed = []

    def object_meshes(labels):
        meshed.append(list(labels))
        return {l: trimesh.creation.icosphere(4, r).apply_translation(boxes[l]) for l in labels}

    sim = SimpleNamespace(objects=objects, n_steps=7, object_meshes=object_meshes,
                          oracle_masks=lambda view, ex, labels, meshes: np.stack(list(masks_from_geometry(
                              view["depth"], view["intrinsics"], np.eye(4), {l: meshes[l] for l in labels}).values())))
    request = {"view_name": "head", "depth": depth, "intrinsics": k}
    alone = masks_from_geometry(depth, k, np.eye(4), object_meshes(["apple_1"]))["apple_1"]
    got = OracleSegmenter(sim).masks(["apple_1"], request, {}).value["head"]
    band = alone & ~got["apple_1"]
    assert band.sum() > 0 and (u[band] > 0).all(), "the pixels apple_1 took alone are apple_2's, right of the contact"
    assert set(got) == {"apple_1"} and meshed[-1] == ["apple_1", "apple_2"], "the neighbour masked too, not answered"


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
    held["left"] = None  # the fingers still move, or a trial's first step before the sensor's window is full
    assert w.held("left").value == (jar,), "an unknown reading leaves the record standing (the toolbox demo place: " \
        "5/5 NOT_HOLDING at step 0 while the setup had just verified the hold)"
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


def test_the_oracle_segmenter_masks_nothing_for_a_target_no_tracked_object_stands_for():
    """w2s3 tripod and smoke detector: a place onto the floor or an untracked table ended the observe in
    object_meshes ("no tracked object for labels ['floor.n.01_1']"). A place's target is the map's support, never a
    segmented object: an empty mask, and the tracked labels theirs."""
    from omnigibson.tiptop.oracle.segmenter import OracleSegmenter

    meshed = []
    sim = SimpleNamespace(objects={"apple_1": SimpleNamespace(aabb=(th.zeros(3), th.ones(3)))}, obstacles={}, n_steps=3,
                          object_meshes=lambda labels: meshed.append(list(labels)) or {l: l for l in labels},
                          oracle_masks=lambda view, ex, labels, meshes: np.ones((len(labels), 2, 2), bool))  # fmt: skip
    sim.tracked_object = lambda label: sim.objects.get(label) or sim.obstacles.get(label)
    request = {"view_name": "head", "depth": np.ones((2, 2)), "intrinsics": np.eye(3)}
    got = OracleSegmenter(sim).masks(["apple_1", "floor.n.01_1"], request, {}).value["head"]
    assert meshed == [["apple_1"]] and got["apple_1"].all() and not got["floor.n.01_1"].any()
    assert not OracleSegmenter(sim).masks(["floor.n.01_1"], request, {}).value["head"]["floor.n.01_1"].any()


def test_the_scorer_answers_holding_and_hand_empty_from_the_grasp_and_the_rest_from_bddl(monkeypatch):
    from omnigibson.controllers import IsGraspingState
    from omnigibson.tiptop.oracle.goals import EpisodeScorer
    from omnigibson.utils import usd_utils

    apple_obj, touching = SimpleNamespace(scene=SimpleNamespace(idx=0)), []
    monkeypatch.setattr(usd_utils, "RigidContactAPI", SimpleNamespace(
        is_in_contact=lambda scene_idx, query_set, with_set, ignore_set, current_only: bool(touching)
        and query_set == [apple_obj] and ignore_set == [robot]))
    robot = SimpleNamespace(is_grasping=lambda arm, obj=None: IsGraspingState.TRUE
                            if arm == "left" and obj in (None, apple_obj) else IsGraspingState.FALSE)  # fmt: skip
    sim = SimpleNamespace(robot=robot, scene_object=lambda n: apple_obj if n == "apple.n.01_1" else object(),
                          holds=lambda p, *a: {"ontop": True}[p])  # fmt: skip
    s = EpisodeScorer(sim)
    assert s.holds(Fact("holding", ("apple.n.01_1", "left"))) is True
    assert s.holds(Fact("holding", ("apple.n.01_1", "right"))) is False
    assert s.holds(Fact("lifted", ("apple.n.01_1",))) is True
    touching.append("board")  # week 2's knife: the fingers on the blade, the knife still on its board
    assert s.holds(Fact("holding", ("apple.n.01_1", "left"))) is True
    assert s.holds(Fact("lifted", ("apple.n.01_1",))) is False, "held, not lifted clear: not picked up"
    touching.clear()
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


def test_the_mesh_room_is_todays_physical_meshes_posed_in_the_map_frame():
    from b1k.skills.tiptop.builders import pick
    from omnigibson.tiptop.oracle.meshroom import MeshRoom

    pose = (th.tensor([1.0, 3.0, 0.5]), th.tensor([0.0, 0.0, 0.0, 1.0]))

    def body(category, fixed, lo, hi):
        return SimpleNamespace(get_position_orientation=lambda: pose, category=category, fixed_base=fixed,
                               aabb=(th.tensor(lo), th.tensor(hi)), name=f"{category}_0")

    shelf = body("shelf", True, [0.8, 2.8, 0.0], [1.2, 3.2, 1.0])
    others = {"apple_1": body("apple", False, [0.9, 2.9, 0.9], [1.0, 3.0, 1.0]),  # a task movable: a perceived hull
              "lawn_0": body("lawn", True, [-9.0, -9.0, 0.0], [9.0, 9.0, 0.01]),  # the ground, not furniture
              "far_0": body("shelf", True, [9.0, 9.0, 0.0], [9.5, 9.5, 1.0])}  # past the radius
    mesh = {"vertices": np.zeros((3, 3), np.float32), "faces": np.zeros((1, 3), np.int32), "kind": "obstacle",
            "pose": np.eye(4, dtype=np.float32), "fixed_base": True}  # fmt: skip
    asked = []
    sim = SimpleNamespace(n_steps=4, obstacles={}, room_collision_scene=lambda: {n: mesh for n in sim.obstacles},
                          robot=None, task_scope=lambda: {})  # fmt: skip

    def nearby_obstacles(collision_map=False):
        asked.append(collision_map)
        sim.obstacles = {"shelf_0": shelf, **others}  # as R1ProSim: registered here, read by room_collision_scene

    sim.nearby_obstacles = nearby_obstacles
    got = MeshRoom(sim).room((0.0, 0.0, 0.0), 3.0)
    (entry,) = got.value  # the map room's bodies only: the A/B changes the room's form, nothing else
    assert (got.source, got.step, asked) == ("oracle", 4, [True])
    assert entry["name"] == "shelf_0" and entry["vertices"] is mesh["vertices"] and entry["kind"] == "obstacle"
    assert np.allclose(entry["pose"][:3, 3], [1.0, 3.0, 0.5]), "the map (world) frame, not the base frame"
    world = World(REFS)
    world.base_pose = lambda: Belief(Pose2(1.0, 2.0, np.pi / 2), "oracle", 0)
    rt = make_rt()
    rt.svc = dataclasses.replace(rt.svc, world=world, collision=MeshRoom(sim))
    (moved,) = pick.room(rt.svc)
    assert np.allclose(moved["pose"][:3, 3], [1.0, 0.0, 0.5]), "the pick builder moves it into the base frame"


def test_the_pseudo_stack_counts_every_oracle_read_and_judges_with_the_scorer(monkeypatch):
    from omnigibson.tiptop.oracle import pseudo_services

    from b1k.connector.world import FurniturePiece, VoxelGrid
    from omnigibson.tiptop.oracle import mapbuild

    cab = SimpleNamespace(aabb=(th.tensor([0.0, 0.0, 0.0]), th.tensor([1.0, 0.5, 0.9])), fixed_base=True,
                          category="cabinet", name="cab_1")
    wall = SimpleNamespace(aabb=(th.tensor([0.0, 0.0, 0.0]), th.tensor([1.0, 0.5, 2.9])), fixed_base=True,
                           category="floors", name="floor_1")  # no map piece: NOT_FURNITURE
    sim = SimpleNamespace(n_steps=9, max_steps=None, scene_object=lambda n: cab if "cabinet" in n else wall, robot=None,
                          task_scope=lambda: {}, env=SimpleNamespace(scene=SimpleNamespace(objects=[])))  # fmt: skip
    occupied = np.zeros((50, 25, 45), dtype=bool)
    occupied[:, :, :40] = True  # the cabinet's body: top at 0.80; its AABB top is 0.90 (a rail, say)
    occupied[:2, :, :] = True
    mapbuild._MAPS[id(sim)] = mapbuild.PseudoMap(sim)  # the per-sim cache is keyed by id(): a collected fake's id can recur
    mapbuild._MAPS[id(sim)].pieces["cab_1"] = FurniturePiece(ObjRef("cabinet.n.01_1", "cabinet", True), tuple(map(tuple, np.eye(4))),
                                                             {"body": VoxelGrid(0.02, (0.0, 0.0, 0.0), occupied)}, ())
    routing = {"goal_checker": "scorer", "goal_checkers_shadow": []}
    svc, segmenter = pseudo_services(SimpleNamespace(sim=sim), "planner", routing)
    assert isinstance(svc.goals, GoalPanel) and svc.goals.primary == "scorer" and svc.planner == "planner"
    top = svc.geometry.top_support(ObjRef("cabinet.n.01_1", "cabinet", True))
    assert top.value.z == pytest.approx(0.8) and top.source == "map", \
        "the map's modal top (SPEC 6.2), never the AABB's maximum: bed_1's headboard put the pillow 6.5 cm high"
    top = svc.geometry.top_support(ObjRef("floors.n.01_1", "floors", True))
    assert top.value.z == pytest.approx(2.9) and top.source == "oracle", "no piece in the map: the AABB, tagged oracle"
    assert svc.provenance.take() == {"geometry.top_support": 1}, "the oracle read is counted in pseudo; the map read is not privileged"
    assert isinstance(svc.provenance, ProvenancePolicy) and segmenter.sim is sim
    sim.nearby_obstacles, sim.obstacles, sim.room_collision_scene = lambda collision_map: None, {}, dict
    mesh, _ = pseudo_services(SimpleNamespace(sim=sim), "planner", routing, collision="mesh")
    assert mesh.collision.room((0.0, 0.0, 0.0), 3.0) == Provided([], "oracle", 9)
    assert mesh.provenance.take() == {"collision.room": 1}, "the mesh room is oracle and counted"


def test_the_oracle_base_pose_carries_the_bases_height_over_the_floor():
    """Pose2.z (W2-S2 6a21afb25): the room moved into the base frame sits where the cloud has it, 5 mm down for a
    standing R1Pro; nothing pinned the z the OracleWorld reads."""
    from omnigibson.tiptop.oracle.world import OracleWorld

    robot = SimpleNamespace(get_position_orientation=lambda: (th.tensor([1.0, 2.0, 0.005]), th.tensor([0.0, 0.0, 0.0, 1.0])))
    world = OracleWorld(SimpleNamespace(sim=SimpleNamespace(robot=robot)), grasp=None)
    pose = world.base_pose().value
    assert (pose.x, pose.y, pose.yaw) == (1.0, 2.0, 0.0) and pose.z == pytest.approx(0.005)


def test_a_demo_restore_loads_the_instance_in_the_cases_mode(monkeypatch):
    """demo_cases.restore(..., mode): a manip run's stance snapshot is of a public_test instance (bb0680248); the
    only tests of that change replaced restore with a lambda, so the argument never reached load_task_instance."""
    import omnigibson.eval.evaluator as evaluator
    from omnigibson.tiptop.host import demo_cases

    loaded, resets = [], []
    monkeypatch.setattr(evaluator, "load_task_instance", lambda env, robot, inst, mode: loaded.append((inst, mode)))
    monkeypatch.setattr("omnigibson.utils.python_utils.recursively_convert_to_torch", lambda x: x)
    env = SimpleNamespace(reset=lambda: resets.append(1), robots=[SimpleNamespace(name="r1")],
                          scene=SimpleNamespace(object_registry=lambda k, name: SimpleNamespace(
                              load_state=lambda st, serialized: None)))  # fmt: skip
    demo_cases.restore(env, {}, 301, "public_test")
    demo_cases.restore(env, {}, 188)
    assert loaded == [(301, "public_test"), (188, "train")] and len(resets) == 4
