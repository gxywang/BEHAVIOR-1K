"""The executor's two ways of stopping a push, against a scripted arm. No simulator.

Both exist because an arm that meets furniture used to keep being commanded: the rest of the trajectory, and then
another 90 steps of converge() leaning on a target it could not reach. The stub below is a joint that moves toward
its target until it is "blocked", after which it stays put however hard it is commanded.
"""

import numpy as np
import pytest

from omnigibson.tiptop.executor import CONVERGE_PATIENCE, PlanExecutor


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


def test_converge_stops_when_the_arm_stops_getting_closer():
    sim = StubSim(wall=0.2)  # the joint jams at 0.2; the target is 1.0
    err = executor(sim).converge(np.array([1.0], dtype=np.float32), tol=0.01, max_steps=500)
    assert err == pytest.approx(0.8, abs=0.02), "it should report the error it was stuck at"
    assert sim.steps < 60, f"it pushed for {sim.steps} steps against a jam; the patience is {CONVERGE_PATIENCE}"


def test_converge_still_reaches_a_target_it_can_reach():
    sim = StubSim()
    err = executor(sim).converge(np.array([1.0], dtype=np.float32), tol=0.01, max_steps=500)
    assert err < 0.01
    assert sim.q[0] == pytest.approx(1.0, abs=0.01)


def test_converge_is_not_tripped_by_slow_but_steady_progress():
    sim = StubSim(speed=0.002)  # slower than CONVERGE_NO_PROGRESS per step would be, but always improving
    err = executor(sim).converge(np.array([0.05], dtype=np.float32), tol=0.005, max_steps=500)
    assert err < 0.005, "steady progress must not count as no progress"
