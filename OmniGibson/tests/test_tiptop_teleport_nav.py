"""The bench's TeleportNavigator (SPEC D5, D20; check_stances week 3) without Isaac Sim: ring candidates where the
footprint is free, nearest first at the preferred standoff; go_to as R1ProSim.place_robot on the sim clock, charged
the human move-to mean in shadow steps; and the one-call planner reaching by check_stances before its observe."""

import math
from types import SimpleNamespace

import numpy as np
import pytest
import torch as th

from b1k.connector.skills import Code, NavResult, PickArgs, Precheck, SkillCall, Stance, StanceRequest
from b1k.connector.types import ObjRef, Pose2
from omnigibson.tiptop.host import skillbench
from omnigibson.tiptop.host.teleport_nav import MOVE_TO_STEPS, STANDOFFS, TeleportNavigator

jar = ObjRef("jar.n.01_1", "jar")
JAR = SimpleNamespace(aabb=(th.tensor([1.95, 1.95, 0.85]), th.tensor([2.05, 2.05, 0.95])), aabb_center=th.tensor([2.0, 2.0, 0.9]))


def sim(blocked=lambda x, y: False, refuse=False, search=None):
    """A scene whose footprint test refuses ``blocked`` spots, a robot at the origin, a legacy stance search that
    finds ``search`` ((score, x, y, yaw, ...) or None), and a teleport that steps the sim 40 times (its fold and
    unfold ramps)."""
    s = SimpleNamespace(n_steps=0, placed=[], objects={jar.id: JAR}, searched=[])
    s.robot = SimpleNamespace(get_position_orientation=lambda: (th.zeros(3), th.tensor([0.0, 0.0, 0.0, 1.0])))
    s.scene_aabbs = lambda: ["boxes"]
    s.scene_object = lambda name: s.objects[name]
    s.asked = []

    def footprint_free(x, y, ignore, aabbs=None, yaw=None, arms=True, reaching=()):
        s.asked.append((round(x, 3), round(y, 3), round(yaw, 3), aabbs, tuple(reaching), arms))
        return (False, "overlaps counter", 0.0) if blocked(x, y) else (True, "", 0.4)

    def place_robot(x, y, yaw, note="", min_unfold=0.0):
        from omnigibson.tiptop.r1pro import BasePlacementCollision

        s.n_steps += 40
        if refuse:
            raise BasePlacementCollision("base destination rejected: base_link intersects counter", obstacle="counter")
        s.placed.append((x, y, yaw, note, min_unfold))

    s.xy_radius = lambda name: 0.07
    s.best_base_pose = lambda points, **kw: (s.searched.append((points, kw)), (search, {}))[1]

    s._footprint_free, s.place_robot = footprint_free, place_robot
    return s


def test_propose_rings_the_reach_point_where_the_footprint_is_free_nearest_first():
    s = sim(blocked=lambda x, y: y > 2.35)  # a counter beyond the jar: no standing on its far side
    nav = TeleportNavigator(s)
    req = StanceRequest((jar,), ((2.0, 2.0, 0.9),))
    out = nav.propose(req, k=8)
    assert len(out) == 8 and all(isinstance(st, Stance) and st.source == "oracle" for st in out)
    d = [math.hypot(st.pose.x - 2.0, st.pose.y - 2.0) for st in out]
    assert np.allclose(d, STANDOFFS[0]), "the preferred standoff first: 12 spots around it, 8 wanted"
    assert all(st.pose.y <= 2.35 for st in out), "none where the footprint test refused"
    here = [math.hypot(st.pose.x, st.pose.y) for st in out]
    assert here == sorted(here), "nearest to where the base stands first"
    for st in out:
        facing = math.atan2(2.0 - st.pose.y, 2.0 - st.pose.x)
        assert min(abs((st.pose.yaw - facing - off + math.pi) % (2 * math.pi) - math.pi) for off in (0, math.pi / 6, -math.pi / 6)) < 1e-9
    assert all(a[3] == ["boxes"] and a[4] == (JAR,) and a[5] is False for a in s.asked), \
        "one scene reading for the whole search; the arm may rest in what it reaches for; the arms are not tested " \
        "at their working posture (the teleport folds them and checks the landing itself)"
    few = TeleportNavigator(sim(blocked=lambda x, y: y > 1.9)).propose(req, k=8)
    assert len(few) == 8 and len({round(math.hypot(st.pose.x - 2.0, st.pose.y - 2.0), 2) for st in few}) > 1, \
        "the next standoffs fill in when the preferred ring has too few free spots"
    assert TeleportNavigator(sim(blocked=lambda x, y: True)).propose(req) == []


def test_propose_leads_with_the_legacy_stance_search_which_frames_the_target_and_the_rings_fill_in():
    s = sim(search=(0.8, 1.25, 2.1, -3.0, [0.75], [0.1], 0.3))
    out = TeleportNavigator(s).propose(StanceRequest((jar,), ((2.0, 2.0, 0.9),)), k=8)
    assert len(out) == 8 and out[0].key == "search:0" and out[0].pose == Pose2(1.25, 2.1, -3.0, 0.0)
    assert all(st.key.startswith("ring:") for st in out[1:]), "the rings after it, to k"
    ((points, kw),) = s.searched
    assert np.allclose(points, [[2.0, 2.0]]) and kw["reaching"] == [JAR] and kw["support_z"] == [pytest.approx(0.85)], \
        "asked as place_robot_for asks it: the targets' box centres, their bottoms as the support the camera must see"
    assert kw["half_widths"] == [0.07] and np.allclose(kw["boxes"][0][1], [2.05, 2.05, 0.95])
    assert TeleportNavigator(sim()).propose(StanceRequest((jar,), ((2.0, 2.0, 0.9),)), k=8)[0].key.startswith("ring:"), \
        "no stance from the search: the rings alone"


def test_go_to_teleports_on_the_sim_clock_and_charges_the_move_to_mean_in_shadow_steps():
    s = sim()
    nav = TeleportNavigator(s)
    assert nav.requires_sim_clock
    stance = Stance("ring:0.60:0", Pose2(1.5, 2.0, 0.3, 0.005), 1.0, "test", "oracle")
    gen = nav.go_to(stance, "obs")
    with pytest.raises(StopIteration) as done:
        next(gen)  # ends before its first yield: 0 Runtime steps
    result, obs = done.value.value
    assert result == NavResult(True, stance, 0, MOVE_TO_STEPS) and obs == "obs"
    assert s.placed == [(1.5, 2.0, 0.3, "go_to ring:0.60:0", 0.0)] and nav.steps == 40, \
        "the teleport's own sim steps; the arm unfolds as far as it can there (no minimum: UNFOLD_MIN refused every " \
        "stance the IK service reached from, at a jar on a counter and a wall switch over a cabinet)"
    refused = TeleportNavigator(sim(refuse=True))
    with pytest.raises(StopIteration) as done:
        next(refused.go_to(stance, "obs"))
    result, _ = done.value.value
    assert not result.ok and "counter" in result.detail and result.shadow_steps == 0 and refused.steps == 40
    assert nav.base_pose().value == Pose2(0.0, 0.0, 0.0, 0.0) and nav.base_pose().source == "oracle"


class Conn:
    """A connector out of reach: check says NO_STANCE_HERE until go_to, then PERCEPT_REQUIRED, then the run."""

    def __init__(self, proposals=3):
        self.events, self.here, self.proposals = [], False, proposals
        self.rt = SimpleNamespace(results={})

    def check(self, call):
        if not self.here:
            return Precheck(False, Code.NO_STANCE_HERE, stance=StanceRequest((jar,), ((2.0, 2.0, 0.9),)))
        return Precheck(False, Code.PERCEPT_REQUIRED) if call.percept is None else Precheck(True)

    def propose_stances(self, req, k=8):
        return [Stance(f"s{i}", Pose2(float(i), 0.0, 0.0), 1.0, "t", "oracle") for i in range(self.proposals)]

    def check_stances(self, call, stances):
        self.events.append("check_stances")
        return [0.0, 1.0, 1.0][: len(stances)]

    def go_to(self, stance):
        self.events.append(("go_to", stance.key))
        self.here = True
        return NavResult(True, stance, 0, MOVE_TO_STEPS)

    def observe(self, req):
        self.events.append("observe")
        return SimpleNamespace(id="p1")

    def run(self, call):
        self.events.append(("run", call.percept))
        return "result"


def test_a_trials_teleports_are_rows_of_their_own_with_the_landings_refusal():
    stance = Stance("ring:0.60:0", Pose2(1.5, 2.0, 0.3, 0.005), 1.0, "test", "oracle")
    results = {"c1": NavResult(False, stance, 0, 0, "base destination rejected: base_link intersects counter"),
               "c2": NavResult(True, stance, 0, MOVE_TO_STEPS), "c3": "a skill result"}
    rows = skillbench.go_to_rows(results)
    assert [r["ok"] for r in rows] == [False, True] and rows[0]["detail"].endswith("intersects counter")
    assert rows[1] == {"key": "ring:0.60:0", "pose": {"__type__": "Pose2", "x": 1.5, "y": 2.0, "yaw": 0.3, "z": 0.005},
                       "ok": True, "shadow_steps": MOVE_TO_STEPS, "detail": ""}


def test_the_one_call_planner_reaches_by_check_stances_before_it_observes_only_for_a_reach_case():
    c = Conn()
    assert skillbench.one_call(c, SkillCall("pick_up", PickArgs(jar), arm="left"), reach=True) == "result"
    assert c.events == ["check_stances", ("go_to", "s1"), "observe", ("run", "p1")], \
        "the IK service ranks the proposals, the best is driven to, then the observe from there"
    c = Conn()
    assert skillbench.one_call(c, SkillCall("pick_up", PickArgs(jar), arm="left")) == "result"
    assert c.events == [("run", None)], "no reach: the case's stance is the setup's, and check says NO_STANCE_HERE"
    c = Conn(proposals=0)
    assert skillbench.one_call(c, SkillCall("pick_up", PickArgs(jar), arm="left"), reach=True) == "result"
    assert c.events == ["check_stances", ("run", None)], \
        "nowhere to stand: the trial is not an exception; the run answers NO_STANCE_HERE from where it is"
