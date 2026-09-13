"""How the executor behaves against an arm that cannot reach its target. No simulator.

An arm that meets furniture used to keep being commanded: the rest of the trajectory, and then another 90 steps of
converge() leaning on a target it could not reach. The trajectory now stops; converge still spends its budget, but
records what it spent it on, because the error is a maximum over every joint and a plateau can be one joint
stalled while the rest are still closing. The stub below is a joint that moves toward its target until it is
"blocked", after which it stays put however hard it is commanded.
"""

import numpy as np
import pytest

from omnigibson.tiptop.executor import PlanExecutor


class StubSim:
    """One joint. It follows a command by ``speed`` per step until ``wall``, and never passes it."""

    CLOSE, OPEN = -1.0, 1.0
    last_gripper = 1.0
    arm = "left"
    planned_joints = ("j",)
    dt = 1 / 30

    def __init__(self, wall=np.inf, speed=0.05, start=0.0):
        self.q = np.array([float(start)], dtype=np.float32)
        self.wall, self.speed, self.steps = wall, speed, 0

    def step(self, q_arm, gripper):
        self.steps += 1
        target = float(np.asarray(q_arm).reshape(-1)[0])
        move = np.clip(target - float(self.q[0]), -self.speed, self.speed)
        self.q = np.array([min(float(self.q[0]) + move, self.wall)], dtype=np.float32)

    def q_arm(self):
        return self.q.copy()


def executor(sim):
    return PlanExecutor(sim)


def test_converge_records_that_it_spent_the_whole_budget_on_a_jam():
    sim = StubSim(wall=0.2)  # the joint jams at 0.2; the target is 1.0
    ex = executor(sim)
    err = ex.converge(np.array([1.0], dtype=np.float32), tol=0.01, max_steps=40)
    assert err == pytest.approx(0.8, abs=0.02), "it reports the error it was stuck at"
    trace = ex.last_converge
    assert trace["capped"] is True and trace["steps"] == 40
    # it stopped getting closer early and then pushed for the rest: that gap is what a later exit rule needs
    assert trace["last_improving_step"] < 10


def test_converge_still_reaches_a_target_it_can_reach():
    sim = StubSim()
    err = executor(sim).converge(np.array([1.0], dtype=np.float32), tol=0.01, max_steps=500)
    assert err < 0.01
    assert sim.q[0] == pytest.approx(1.0, abs=0.01)


def test_converge_keeps_settling_a_joint_that_is_still_creeping():
    # 9% of gripper events follow a segment that ended 0.01-0.05 rad short and used the whole budget; cutting
    # those settles is the regression a no-progress exit would cause, so a slow creep must still reach its target
    sim = StubSim(speed=0.002)
    ex = executor(sim)
    err = ex.converge(np.array([0.05], dtype=np.float32), tol=0.005, max_steps=500)
    assert err < 0.005
    assert ex.last_converge["capped"] is False


# --------------------------------------------------------------- the command leash
def test_the_leash_does_nothing_to_a_command_the_arm_is_following():
    from omnigibson.tiptop.executor import leash

    measured = np.array([0.0, 1.0, -0.5])
    target = np.array([0.01, 1.02, -0.49])  # healthy tracking error is 0.000-0.019 rad
    assert np.allclose(leash(target, measured), target, atol=1e-6)


def test_the_leash_bounds_how_far_ahead_of_a_stuck_joint_the_command_can_run():
    from omnigibson.tiptop.executor import EXEC_LEASH, leash

    measured = np.array([0.0, 0.0])
    target = np.array([2.0, -2.0])  # a segment that has run a long way past a jammed joint
    out = leash(target, measured)
    assert np.allclose(out, [EXEC_LEASH, -EXEC_LEASH])


def test_a_leashed_command_still_reaches_its_target_when_the_arm_is_free():
    sim = StubSim(speed=0.05)
    ex = executor(sim)
    err = ex.converge(np.array([1.0], dtype=np.float32), tol=0.01, max_steps=500)
    assert err < 0.01, "the leash advances with the arm, so a free joint still arrives"
