"""Sticky contact seeking must not strike a target while open or advance during the grasp window."""
from types import MethodType, SimpleNamespace

import numpy as np
import pytest
import torch as th
import trimesh

from omnigibson.tiptop.r1pro import R1ProSim, STICKY_STANDOFF


class ContactArm:
    """One joint whose fingertip contacts a target at 1 cm; the assist takes several stationary steps."""
    dt = 1 / 30
    planned_joints = ["left_arm_joint1"]
    joint_index = {"left_arm_joint1": 0}
    OPEN, CLOSE = 1.0, -1.0

    def __init__(self, attach_after=10, query_failure=False):
        self.q = np.zeros(1)
        self.posture, self.n_steps, self.last_gripper = {}, 0, self.OPEN
        self.commands, self.checks, self.ik_goals = [], [], []
        self.touch_steps, self.attach_after, self.query_failure = 0, attach_after, query_failure
        self.target = SimpleNamespace(name="book", links={"body": SimpleNamespace(prim_path="/book/body")})
        self.robot = SimpleNamespace(
            get_joint_positions=lambda: th.tensor(self.q),
            _ag_obj_in_hand={"left": None},
            _find_gripper_contacts=self.contacts,
            finger_link_names={"left": ["left_gripper_finger_link1", "left_gripper_finger_link2"]},
        )
        self.ramp_to = MethodType(R1ProSim.ramp_to, self)
        self.ik = SimpleNamespace(solve=self.solve, fk=lambda q: (np.array([q[0], 0, 0]), np.array([0, 0, 0, 1])))

    def q_arm(self):
        return self.q.copy()

    def contacts(self, arm):
        if self.query_failure:
            raise RuntimeError("contact sensor unavailable")
        return ({"/book/body"} if self.q[0] >= 0.01 - 1e-9 else set()), {}

    def step(self, q, gripper):
        self.q = np.asarray(q).copy()
        self.last_gripper = gripper
        self.n_steps += 1
        self.commands.append((float(self.q[0]), gripper))
        if self.q[0] >= 0.01 - 1e-9 and gripper == self.CLOSE:
            self.touch_steps += 1
            if self.attach_after is not None and self.touch_steps >= self.attach_after:
                self.robot._ag_obj_in_hand["left"] = self.target

    def hold(self, n, gripper, q_arm=None):
        for _ in range(n):
            self.step(self.q if q_arm is None else q_arm, gripper)

    def _motion_finger_ranges(self, measured, gripper):
        return {"finger": (0.05, 0.0)} if gripper != self.last_gripper else {}

    def ramp_collision(self, names, start, goal, allowed_contacts=None, gripper=None):
        self.checks.append((list(start), list(goal), allowed_contacts, gripper))
        if max(goal, default=0) >= 0.01 and "left_gripper_finger_link1" not in (allowed_contacts or {}).get("book", ()):
            return "left_gripper_finger_link1", "book", 0
        return None

    def _targets_from(self, joints, q):
        return q

    def solve(self, goal, quat, **kwargs):
        self.ik_goals.append(np.asarray(goal).copy())
        return [goal[0]]


def test_open_precontact_collision_refuses_before_any_motor_command():
    arm = ContactArm()
    hit = arm.ramp_to([0.02], {}, arm.OPEN, 15)
    assert hit[0] == "left_gripper_finger_link1 intersects book"
    assert arm.commands == []


@pytest.mark.parametrize("attach_after", [10, None])
def test_sticky_contact_stops_arm_before_waiting_for_attachment_and_never_presses_deeper(attach_after):
    arm = ContactArm(attach_after=attach_after)
    held = R1ProSim._sticky_close_on(arm, "left", arm.ik, arm.target, np.eye(4), [1, 0, 0], [0], arm.planned_joints)
    assert held is (attach_after is not None)
    assert arm.commands
    assert all(command == arm.CLOSE for _, command in arm.commands)
    # Closure happens at the standoff with no target exception; terminal checks exempt only active fingers.
    assert arm.checks[0][0] == arm.checks[0][1] == [0.0]
    assert not arm.checks[0][2]
    for _, _, allowed, _ in arm.checks[1:]:
        assert allowed == {"book": set(arm.robot.finger_link_names["left"])}
    contact = next(i for i, (q, _) in enumerate(arm.commands) if q >= 0.01 - 1e-9)
    assert len(arm.commands) > contact + 1  # the grasp window is observed while stationary
    assert all(q == arm.commands[contact][0] for q, _ in arm.commands[contact:])
    assert max(float(goal[0]) for goal in arm.ik_goals) < 0.013


def test_contact_seeking_without_contact_observations_refuses_before_motor_commands():
    arm = ContactArm(query_failure=True)
    hit = arm.ramp_to([0.005], {}, arm.CLOSE, 0, stop_on_contact="left")
    assert hit[0] == "contact observation unavailable"
    assert arm.commands == []


def test_sticky_precontact_keeps_target_collision_and_uses_physical_bottom_of_hollow_object():
    # Four walls and a floor: the AABB top centre is empty, and the first real top-facing surface is the floor.
    floor = trimesh.creation.box([0.2, 0.2, 0.01])
    floor.apply_translation([0.7, 0, 0.5])
    walls = []
    for x, y, size in [(0.605, 0, [0.01, 0.2, 0.1]), (0.795, 0, [0.01, 0.2, 0.1]),
                       (0.7, -0.095, [0.2, 0.01, 0.1]), (0.7, 0.095, [0.2, 0.01, 0.1])]:
        wall = trimesh.creation.box(size)
        wall.apply_translation([x, y, 0.55])
        walls.append(wall)
    physical = trimesh.util.concatenate([floor, *walls])
    target = SimpleNamespace(name="bowl", aabb=tuple(th.tensor(v) for v in physical.bounds))
    calls = []

    def solution(*args, **kwargs):
        calls.append(("solution", np.asarray(args[3]), kwargs))
        return [0.2]

    sim = SimpleNamespace(
        scene_object=lambda name: target, collision_mesh_world=lambda obj: physical,
        to_base=lambda p, q: (p, q), base_pose=lambda: (th.zeros(3), th.tensor([0., 0., 0., 1.])),
        arm_ik=lambda *a, **k: None, ik_joint_names=lambda *a, **k: ["joint"],
        robot=SimpleNamespace(get_joint_positions=lambda: th.zeros(1), _ag_obj_in_hand={"left": target}),
        joint_index={"joint": 0}, scene_aabbs=lambda: [], _press_solution=solution,
        grasp_target=lambda *a, **k: (np.zeros(3), np.eye(3)), reach_plan=lambda *a, **k: [[0.2]],
        path_hits_scene=lambda *a, **k: [], _targets_from=lambda names, q: q, posture={}, OPEN=1., CLOSE=-1.,
        q_arm=lambda: np.zeros(1),
        ramp_to=lambda *a, **k: calls.append(("ramp", a, k)),
        _sticky_close_on=lambda *a: calls.append(("close", a)) or True,
        lift_held=lambda *a: calls.append(("lift",)) or True,
    )
    assert R1ProSim.press_grasp(sim, "left", "bowl") is True
    assert calls[0][0] == "ramp" and calls[0][1][2] == sim.CLOSE  # the hand closes first, in free space
    assert calls[1][0] == "solution"
    assert calls[1][1][2] == pytest.approx(0.505, abs=1e-6)
    assert calls[1][2]["press"] == -STICKY_STANDOFF
    assert calls[2][0] == "ramp" and calls[2][1][2] == sim.CLOSE  # and travels closed
    assert not calls[2][2].get("allowed_contacts")
    assert calls[3][0] == "close"


@pytest.mark.parametrize("planned", [True, False])
def test_no_straight_way_in_asks_the_planner_and_seeks_contact_only_after_a_completed_path(planned):
    book = trimesh.creation.box([0.04, 0.2, 0.25])
    book.apply_translation([0.7, 0.0, 1.0])
    target = SimpleNamespace(name="book", aabb=tuple(th.tensor(v) for v in book.bounds))
    calls = []
    sim = SimpleNamespace(
        scene_object=lambda name: target, collision_mesh_world=lambda obj: book,
        to_base=lambda p, q: (p, q), base_pose=lambda: (th.zeros(3), th.tensor([0., 0., 0., 1.])),
        arm_ik=lambda *a, **k: None, ik_joint_names=lambda *a, **k: ["joint"],
        robot=SimpleNamespace(get_joint_positions=lambda: th.zeros(1), _ag_obj_in_hand={"left": target}),
        joint_index={"joint": 0}, scene_aabbs=lambda: [], _press_solution=lambda *a, **k: [0.2],
        grasp_target=lambda *a, **k: (np.zeros(3), np.eye(3)),
        reach_plan=lambda *a, **k: [],  # between the boards: no straight candidate clears the scene
        path_hits_scene=lambda *a, **k: [], _targets_from=lambda names, q: q, posture={}, OPEN=1., CLOSE=-1.,
        q_arm=lambda: np.zeros(1), ramp_to=lambda *a, **k: None,
        planned_approach=lambda where, quat, note: calls.append(("planned", note)) or planned,
        _sticky_close_on=lambda *a: calls.append(("close",)) or True,
        lift_held=lambda *a: calls.append(("lift",)) or True,
    )
    assert R1ProSim.press_grasp(sim, "left", "book") is planned
    assert calls[0][0] == "planned"
    assert (("close",) in calls) is planned  # never press toward a book the hand did not reach
