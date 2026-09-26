"""Two hands on the skill bench (SPEC §5.6, §6.4, §8 week 2): hold(left, "here") + observe(aim=False) + press(right)
as one bench trial, its U0 in the concurrent form, and the bridge half of S1: adopting the other arm's planner locks
the idle arm where it stands instead of refusing a posture off the nominal."""

from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch as th

from b1k.connector.observe import Percept, PerceptInfo, StepObs
from b1k.connector.skills import HoldArgs, Precheck, PressArgs, SkillCall, SkillResult, Status
from b1k.connector.types import Fact, ObjRef
from b1k.runtime.compose import ACTION_SLICES, CLOSED, OPEN
from b1k.skills.scripted import ScriptedBackend
from b1k.tests.fakes import REFS, Env, JointWorld, make_rt, radio
from omnigibson.tiptop.host import skillbench
from omnigibson.tiptop.r1pro import R1ProSim

ROOT = Path(__file__).resolve().parents[2]


# ------------------------------------------------------------------------------------------------ S1, bridge half
def test_adopting_the_other_arms_planner_locks_the_idle_arm_where_it_stands():
    """The human's left arm holds the radio 0.8 rad off r1pro_right's nominal left lock (turning_on_radio ep 168):
    the request sends the measured lock and the server takes it (S1), so the bridge adopts it instead of refusing."""
    names = [f"torso_joint{k}" for k in range(1, 5)]
    names += [f"{a}_arm_joint{k}" for a in ("left", "right") for k in range(1, 8)]
    names += ["left_gripper_finger_joint1", "left_gripper_finger_joint2"]
    q = dict.fromkeys(names, 0.0) | {"torso_joint1": 1.024, "torso_joint2": -1.448, "torso_joint3": -0.522,
                                     "left_arm_joint1": -0.823, "left_arm_joint4": -0.915, "left_arm_joint6": 0.996,
                                     "left_gripper_finger_joint1": 0.04}
    nominal = {"torso_joint1": 1.025, "torso_joint2": -1.45, "torso_joint3": -0.47, "torso_joint4": 0.0,
               "left_arm_joint1": -1.6312, "left_arm_joint2": 0.2636, "left_arm_joint3": -1.812,
               "left_arm_joint4": -1.4576,
               "left_arm_joint5": -0.0508, "left_arm_joint6": -0.3727, "left_arm_joint7": -1.3193,
               "left_gripper_finger_joint1": 0.05, "left_gripper_finger_joint2": 0.05}
    right = [f"right_arm_joint{k}" for k in range(1, 8)]
    emb = {"arm": "right", "robot_type": "r1pro_right", "locked_joints": nominal, "joint_names": right,
           "q_home": [-1.1189, -0.7095, 1.7969, -1.8904, 0.3492, 0.5079, 1.1452]}
    sim = SimpleNamespace(
        arm="left", other_arm="right", joint_index={n: i for i, n in enumerate(names)},
        robot=SimpleNamespace(get_joint_positions=lambda: th.tensor([q[n] for n in names], dtype=th.float64),
                              gripper_control_idx={"right": th.tensor([20, 21])}),
        arm_idx=th.arange(11), gripper_idx=th.tensor([18, 19]), mirror_arm_idx=None, mirror_gripper_idx=None,
        last_gripper=CLOSED, other_gripper=OPEN, stance_ready=[0.1] * 11,
    )  # fmt: skip
    R1ProSim.adopt_embodiment(sim, emb)  # until 2026-09-26: RuntimeError "locks left_arm_joint1 ... off by 0.808"
    assert (sim.arm, sim.other_arm, sim.planned_joints) == ("right", "left", right)
    assert sim.posture == {j: q[j] for j in nominal if "finger" not in j}, "idle arm and torso held where they are"
    assert sim.locked_nominal == sim.posture, "the lock is the measured posture: nothing for restore_locked_arm to undo"
    assert (sim.last_gripper, sim.other_gripper) == (OPEN, CLOSED), "the left keeps holding; the right resumes its own"
    assert sim.q_home == emb["q_home"] and sim.stance_ready is None


# ------------------------------------------------------------------------------------------- the two-handed trial
class Host:
    """The bench host over the fakes' env, as a demo restore leaves it: the left hand commanded closed on the radio."""

    def __init__(self, env):
        self.env, self.acts, self.env_wall_s = env, [], 0.0

    def commanded_targets(self):
        return {"arm_left": np.zeros(7, dtype=np.float32), "gripper_left": np.array([CLOSED], dtype=np.float32)}

    def observe_now(self):
        return StepObs(self.env.n, self.env.p.copy(), {})

    def env_step(self, a):
        self.acts.append(a.copy())
        return self.env.step(a)

    def raw(self):
        return self.env.raw()

    def parse(self, raw, step):
        return StepObs(step, raw["proprio"], raw)


class Press:  # a native-shaped right-arm press: plans from the planner's Percept, then yields its motion
    name = "tiptop"

    def supports(self, c):
        return True

    def check(self, c, s):
        return Precheck(True)

    def run(self, call, svc, obs):
        assert call.arm == "right" and svc.percepts[call.percept] is not None
        for _ in range(3):
            obs = yield {"arm_right": np.full(7, 0.3, dtype=np.float32)}
        return SkillResult(call.call_id, "press", "tiptop", Status.SUCCEEDED, None, "stroke", "", (), {"scorer": True},
                           "scorer")  # fmt: skip


class Observer:  # the bench's CaptureObserver shape: on the sim clock, ends before its first yield
    requires_sim_clock = True

    def __init__(self):
        self.aims = []

    def observe(self, req, obs):
        self.aims.append(req.aim)
        return Percept(PerceptInfo("", 0, 0, 0, {}, "oracle"), {}, {}, None), obs
        yield {}


def two_hands(lease):
    host, observer = Host(Env()), Observer()
    svc = make_rt().svc.__class__(**{**vars(make_rt().svc),
                                     "world": JointWorld(REFS, {Fact("holding", (radio.id, "left"))})})
    backends = {"scripted": ScriptedBackend(), "tiptop": Press()}
    routing = {"hold": {"default": "scripted"}, "press": {"default": "tiptop"}}
    press = SkillCall("press", PressArgs(radio), arm="right", seed=1)
    r, rt, calls, _ = skillbench.run_trial(host, svc, backends, routing, observer, press, lease)
    return r, rt, calls, host, observer


def test_the_two_handed_trial_holds_the_left_hand_through_an_unaimed_observe_and_the_right_press():
    lease = SkillCall("hold", HoldArgs(radio, pose="here"), arm="left")
    r, rt, calls, host, observer = two_hands(lease)
    assert r.status is Status.SUCCEEDED and observer.aims == [False], "a lease holds the trunk read-only: no aim"
    held = next(c for c in calls if c["skill"] == "hold")
    assert (held["backend"], held["status"], held["steps"]) == ("scripted", "aborted", 3), "still holding at the abort"
    assert rt.step == 3 and rt.charged == {"observe": 0, "skill": 6}, "the lease is charged on every step too"
    assert not skillbench.u0(rt) and skillbench.u0(rt, lease_steps=held["steps"]), "U0 in its concurrent form"
    assert all(a[ACTION_SLICES["gripper_left"]] == CLOSED for a in host.acts) and len(host.acts) == 3
    assert all(a[ACTION_SLICES["arm_right"]] == pytest.approx(0.3) for a in host.acts)
    case = {"id": "c", "call": SkillCall("press", PressArgs(radio)), "lease": lease}
    row = skillbench.row(case, 0, 1, r, rt, 3, 2.0, 0.5, lease=held)
    assert row["u0"] and row["lease"] == {"skill": "hold", "backend": "scripted", "status": "aborted",
                                          "code": "aborted", "steps": 3}
    assert skillbench.summarize(case, [row])["lease_kept"] == 1
    lost = dict(row, lease=dict(held, status="failed", code="hold_lost"))
    assert skillbench.summarize(case, [row, lost])["lease_kept"] == 1


def test_without_a_lease_the_one_call_planner_aims_as_before():
    r, rt, calls, host, observer = two_hands(None)
    assert r.status is Status.SUCCEEDED and observer.aims == [True] and [c["skill"] for c in calls] == ["press"]
    assert rt.charged == {"observe": 0, "skill": 3} and skillbench.u0(rt)


def test_the_two_hands_cases_load_with_their_lease():
    cases = skillbench.load_cases(ROOT / "tiptop/b1k/skills/bench/two_hands.yaml")
    assert cases and all(c["task"] == "turning_on_radio" and c["setup"]["held"] == {"left": "radio_89"} for c in cases)
    for c in cases:
        assert c["call"].skill == "press" and c["call"].arm == "right" and c["call"].args.want_on is True
        assert c["lease"] == SkillCall("hold", HoldArgs(c["call"].args.target, pose="here"), arm="left"), \
            "the lease holds the very ObjRef the press names, so the setup's held record satisfies its precheck"
        assert isinstance(c["call"].args.target, ObjRef) and Path(c["demo"]["snapshot"]).is_file()
