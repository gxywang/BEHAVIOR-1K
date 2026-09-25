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


def test_converge_stops_loading_a_stationary_jam():
    sim = StubSim(wall=0.2)  # the joint jams at 0.2; the target is 1.0
    ex = executor(sim)
    err = ex.converge(np.array([1.0], dtype=np.float32), tol=0.01, max_steps=40)
    assert err == pytest.approx(0.8, abs=0.02), "it reports the error it was stuck at"
    trace = ex.last_converge
    assert trace["stalled"] is True and trace["steps"] < 40
    assert trace["capped"] is False
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


def test_a_crawling_planner_segment_is_played_faster_but_never_slower():
    """Over half of what the planner returns is a crawl, and every waypoint of it costs a simulator step.

    Measured over 1862 saved planner trajectory segments: the peak joint speed has a median of 0.33 rad/s and 54%
    peak below 0.5, while the fastest tenth already reaches 1.52. Scaling a segment's clock changes how fast its
    own path is played and nothing else -- not the path, so not collision or reachability (2026-09-14).
    """
    from omnigibson.tiptop.executor import TRAJECTORY_MIN_SPEED, plan_dt

    # a crawl: one joint moving 0.01 rad per 0.1 s step = 0.1 rad/s
    crawl = {"positions": [[0.0], [0.01], [0.02]], "dt": 0.1}
    assert TRAJECTORY_MIN_SPEED == 0.0
    assert plan_dt(crawl) == crawl["dt"], "execution preserves the planner's timing"
    faster = plan_dt(crawl, floor=1.0)
    assert faster < crawl["dt"], "a crawl is played faster"
    assert np.isclose(faster, 0.01), "explicit speed-floor opt-in still works"

    # already at or above the floor: untouched, so the executor never outruns the planner's own pace
    brisk = {"positions": [[0.0], [0.2], [0.4]], "dt": 0.1}  # 2 rad/s
    assert plan_dt(brisk, floor=TRAJECTORY_MIN_SPEED) == brisk["dt"]

    # degenerate segments are returned as they are rather than divided by zero
    assert plan_dt({"positions": [[0.0]], "dt": 0.1}, floor=1.0) == 0.1
    assert plan_dt({"positions": [[0.0], [0.1]], "dt": 0.0}, floor=1.0) == 0.0
    assert plan_dt({"positions": [[0.0], [0.0]], "dt": 0.1}, floor=1.0) == 0.1


# --------------------------------------------------------------- the leash on bridge ramps
class StubArm:
    """One planned joint that stops dead at ``wall`` however hard it is commanded.

    ``ramp_to`` is called unbound against this, so it needs only what the method itself touches.
    """

    dt = 1 / 30
    planned_joints = ("j",)
    joint_index = {"j": 0}
    n_steps = 0

    def __init__(self, wall: float = 0.2, speed: float = 0.2):
        import torch as th

        self.q, self.wall, self.speed, self.posture = [0.0], wall, speed, {}
        self.leads = []  # how far ahead of the arm each command was sent
        self.robot = type("R", (), {"get_joint_positions": lambda _s: th.tensor(self.q, dtype=th.float32)})()

    def step(self, q_arm, gripper):
        cmd = float(np.asarray(q_arm).reshape(-1)[0])
        self.leads.append(cmd - self.q[0])
        self.q = [min(self.q[0] + float(np.clip(cmd - self.q[0], -self.speed, self.speed)), self.wall)]

    def ramp_collision(self, names, start, goal, allowed_contacts=None, gripper=None):
        return None  # these tests isolate tracking/leashing; scene preflight has geometry tests

    def hold(self, n, gripper, q_arm=None):
        for _ in range(n):  # the settle steps lean too, so they must be measured, not skipped
            self.step(self.q if q_arm is None else q_arm, gripper)


def _ramp(leashed: bool):
    from omnigibson.tiptop.r1pro import TRAVEL_MAX_JOINT_VEL, R1ProSim

    sim = StubArm()
    blocked = R1ProSim.ramp_to(
        sim, [2.0], {}, 0.0, 12, note="fold for travel", max_vel=TRAVEL_MAX_JOINT_VEL, leashed=leashed
    )
    return sim, blocked


def test_a_bridge_ramp_into_furniture_is_leashed_like_a_planned_segment():
    """91% of measured arm-vs-world contact is on these ramps; the leash bounds how hard they lean."""
    from omnigibson.tiptop.executor import EXEC_LEASH

    sim, blocked = _ramp(leashed=True)
    assert blocked is not None, "the ramp still notices it is blocked"
    assert max(sim.leads) <= EXEC_LEASH + 1e-6, f"it leaned {max(sim.leads):.3f} rad past the arm"


def test_the_leash_does_not_change_when_a_ramp_calls_itself_blocked():
    """Detection runs on the path target, not the leashed command, so the block fires at the same step."""
    assert _ramp(leashed=True)[1] == _ramp(leashed=False)[1]


def test_an_unleashed_ramp_still_gets_to_push():
    """The grasp press and the drawer pull exist to load a joint and opt out."""
    from omnigibson.tiptop.executor import EXEC_LEASH

    sim, _ = _ramp(leashed=False)
    assert max(sim.leads) > EXEC_LEASH


class ExecutionSim(StubSim):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.finger_commands = []
        self.arm_commands = []
        self.robot = type("Robot", (), {"is_grasping": lambda self: False})()

    def step(self, q_arm, gripper):
        self.finger_commands.append(gripper)
        self.arm_commands.append(np.asarray(q_arm).copy())
        super().step(q_arm, gripper)

    def q_fingers(self):
        return np.array([self.finger_commands[-1] if self.finger_commands else self.last_gripper])

    def eef_pose_base(self, arm):
        return np.eye(4)

    def finger_travel(self):
        return 0.05  # r1pro: each finger joint runs 0 (closed) .. 0.05 m (open)


def motion(positions):
    return {"type": "trajectory", "label": "Place(book, box)", "positions": positions, "dt": 0.1}


def test_plan_start_mismatch_does_not_home_or_release():
    sim = ExecutionSim(start=0.4)
    sim.last_gripper = sim.CLOSE
    result = PlanExecutor(sim).execute({"q_init": [0.0], "gripper_init": "open", "steps": []})
    assert result["completed"] is False
    assert "start" in result["error"]
    assert sim.steps == 0
    assert not sim.finger_commands


def test_blocked_place_stops_before_release_or_later_segments():
    sim = ExecutionSim(wall=0.2)
    sim.last_gripper = sim.CLOSE
    plan = {
        "q_init": [0.0],
        "steps": [
            motion(np.linspace(0, 1.0, 30)[:, None]),
            {"type": "gripper", "label": "Place(book, box)", "action": "open"},
            motion([[1.0], [0.0]]),
        ],
    }
    result = PlanExecutor(sim).execute(plan)
    assert result["completed"] is False
    assert result["trajectories"][0]["fell_behind"]
    assert result["gripper_events"] == []
    assert set(sim.finger_commands) == {sim.CLOSE}
    assert sim.steps < 40
    assert np.array_equal(sim.arm_commands[-1], sim.q_arm()), "abort must cancel the residual push command"
    assert result["env_steps"] == sim.steps


def test_discontinuous_plan_is_not_bridged_with_an_unchecked_move():
    sim = ExecutionSim()
    result = PlanExecutor(sim).execute({"q_init": [0.0], "steps": [motion([[0.4], [0.5]])]})
    assert result["completed"] is False
    assert sim.steps == 0


def test_successful_place_follows_motion_then_releases():
    sim = ExecutionSim()
    sim.last_gripper = sim.CLOSE
    result = PlanExecutor(sim, gripper_hold_steps=2).execute(
        {
            "q_init": [0.0],
            "steps": [
                motion(np.linspace(0, 0.4, 20)[:, None]),
                {"type": "gripper", "label": "Place(book, box)", "action": "open"},
            ],
        }
    )
    assert result["completed"] is True
    assert len(result["gripper_events"]) == 1
    assert sim.q[0] == pytest.approx(0.4, abs=0.01)
    assert result["env_steps"] == sim.steps


@pytest.mark.parametrize("failure", ["collision", "unavailable"])
def test_full_body_validation_refuses_a_planned_trajectory_before_motor_commands(failure):
    sim = ExecutionSim()

    def validate(positions, gripper, label):
        assert positions[-1][0] == pytest.approx(0.2)
        if failure == "unavailable":
            raise ValueError("no physical collision geometry")
        return ("left_gripper_finger_link1", "box", 4)

    sim.validate_motion = validate
    result = PlanExecutor(sim).execute({"q_init": [0.0], "steps": [motion([[0.0], [0.2]])]})
    assert result["completed"] is False
    assert "validation" in result["error"]
    assert sim.steps == 0


def test_a_pick_closes_the_fingers_at_a_creep_then_holds_closed():
    """The assisted weld needs 0.3 s of two-finger contact; a close at the joint speed cap (8 mm per step) wedged
    flat objects out of a corner pinch in 3-5 steps. The close ramps the width command so each finger travels at
    most GRASP_CLOSE_STEP per env step, then holds CLOSE for the usual hold steps."""
    from omnigibson.tiptop.executor import GRASP_CLOSE_STEP

    sim = ExecutionSim()
    result = PlanExecutor(sim, gripper_hold_steps=5).execute(
        {"q_init": [0.0], "steps": [{"type": "gripper", "label": "Pick(book, g1, q1)", "action": "close"}]}
    )
    assert result["completed"] is True
    commands = np.array(sim.finger_commands[:-15])  # the 15 settling steps after the plan repeat the last command
    assert sim.CLOSE < commands[0] < sim.OPEN and commands[-5:].tolist() == [sim.CLOSE] * 5
    assert (np.diff(commands) <= 0).all(), "monotonic: the fingers never reopen on the way down"
    travel = np.diff(np.concatenate([[sim.OPEN], commands])) * sim.finger_travel() / 2  # a command of 2 spans the range
    assert np.abs(travel).max() <= GRASP_CLOSE_STEP + 1e-9
    assert len(commands) == pytest.approx(sim.finger_travel() / GRASP_CLOSE_STEP + 5, abs=1.5)
    assert sim.finger_commands[-1] == sim.CLOSE
    fist = ExecutionSim()  # a press closes the empty hand in one step, as before
    PlanExecutor(fist, gripper_hold_steps=5).execute(
        {"q_init": [0.0], "steps": [{"type": "gripper", "label": "Push(button)", "action": "close"}]}
    )
    assert set(fist.finger_commands) == {sim.CLOSE}


def test_legs_after_a_planned_grasp_or_release_keep_that_objects_finger_allowance():
    sim = ExecutionSim()
    labels = []
    sim.validate_motion = lambda positions, gripper, label: labels.append(label)
    pick = "Pick(plate_1, grasp1, q1)"
    result = PlanExecutor(sim).execute({"q_init": [0.0], "steps": [
        {**motion([[0.0], [0.2]]), "label": pick},
        {"type": "gripper", "label": pick, "action": "close"},
        {**motion([[0.2], [0.3]]), "label": "GoToInitial(q0)"},
        {"type": "gripper", "label": "Place(plate_1, sink)", "action": "open"},
        {**motion([[0.3], [0.1]]), "label": "GoToInitial(q0)"},
    ]})
    assert result["completed"] is True
    trajectory_labels = [label for label in labels if "GoToInitial" in label]
    # the picked object keeps its finger allowance until the Place opens; the retreat after it keeps the Place's
    assert trajectory_labels == [f"{pick} then GoToInitial(q0)", "Place(plate_1, sink) then GoToInitial(q0)"]


@pytest.mark.parametrize("initial", [True, False])
def test_gripper_opening_is_validated_before_release(initial):
    sim = ExecutionSim()
    sim.last_gripper = sim.CLOSE
    checked = []

    def validate(positions, gripper, label):
        checked.append((np.asarray(positions), gripper, label))
        return ("left_gripper_finger_link1", "container_wall", 1)

    sim.validate_motion = validate
    plan = {"q_init": [0.0], "steps": []}
    if initial:
        plan["gripper_init"] = "open"
    else:
        plan["steps"] = [{"type": "gripper", "label": "Place(book, box)", "action": "open"}]
    result = PlanExecutor(sim).execute(plan)
    assert result["completed"] is False
    assert checked[0][1] == sim.OPEN
    assert result["gripper_events"] == []
    assert sim.steps == 0



@pytest.mark.parametrize("leashed", [True, False])
def test_zero_settle_bridge_tracking_abort_cancels_residual_target_without_changing_gripper(leashed):
    from omnigibson.tiptop.r1pro import R1ProSim, TRAVEL_MAX_JOINT_VEL

    class RecordingArm(StubArm):
        def __init__(self):
            super().__init__()
            self.commands = []

        def step(self, q_arm, gripper):
            self.commands.append((np.asarray(q_arm).copy(), gripper))
            super().step(q_arm, gripper)

    sim = RecordingArm()
    blocked = R1ProSim.ramp_to(
        sim, [2.0], {}, -1.0, 0, note="zero-settle waypoint", max_vel=TRAVEL_MAX_JOINT_VEL, leashed=leashed
    )
    assert blocked is not None
    assert len(sim.commands) == blocked[1] + 1  # exactly one cancellation after the last path command
    assert sim.commands[-2][0][0] > sim.q[0]
    assert sim.commands[-1][0] == pytest.approx(sim.q)
    assert {gripper for _, gripper in sim.commands} == {-1.0}


def test_a_placenear_release_keeps_the_released_objects_finger_allowance():
    """cuTAMP labels inside-round releases PlaceNear(...); "Place(" does not prefix-match it."""
    sim = ExecutionSim()
    labels = []
    sim.validate_motion = lambda positions, gripper, label: labels.append(label)
    release = "PlaceNear(bowl_1, grasp0, pose2, sink_1, bowl_1, q1)"
    result = PlanExecutor(sim).execute({"q_init": [0.0], "steps": [
        {**motion([[0.0], [0.2]]), "label": release},
        {"type": "gripper", "label": release, "action": "open"},
        {**motion([[0.2], [0.1]]), "label": "GoToInitial(q0)"},
    ]})
    assert result["completed"] is True
    assert labels[-1] == f"{release} then GoToInitial(q0)"
    assert release in labels  # its own leg is not relabelled


def test_the_motion_audit_gives_a_placenear_target_finger_allowance():
    import re

    source = open(__import__("omnigibson.tiptop.r1pro", fromlist=["x"]).__file__).read()
    pattern = re.search(r're\.match\(r"(\^\(Pick[^"]+)"', source)[1]
    assert re.match(pattern, "PlaceNear(bowl_1, grasp0, pose2, sink_1, bowl_1, q1)")[2].strip() == "bowl_1"


def test_do_execute_keeps_hold_for_a_stamp_or_a_pour_and_blocks_the_grasp_assist_for_a_push(monkeypatch, tmp_path):
    """T3 (run.do_execute): a stamp or a pour plays the plan cut by keep_holding (no gripper open, the approach
    reversed) so the tool stays in the hand; a push, like a press, runs with the grasp assist blocked (a sticky
    finger would weld the book it slides); a placement does neither."""
    from types import SimpleNamespace

    import omnigibson.tiptop.executor as executor
    import omnigibson.tiptop.run as run

    played, calls = [], []

    class Executor:
        close_eef = None

        def __init__(self, sim, gripper_hold_steps=0):
            pass

        def execute(self, plan):
            played.append([s["type"] for s in plan["steps"]])
            return {"completed": True}

    monkeypatch.setattr(executor, "PlanExecutor", Executor)
    monkeypatch.setattr(run, "note_hands", lambda *a, **k: None)
    leg = lambda a, b: {"type": "trajectory", "label": "Place(x, g, p, y, q)", "positions": np.linspace(a, b, 3)[:, None],
                        "velocities": None, "dt": 0.02}  # fmt: skip
    plan = {"version": "1.1.0", "q_init": np.zeros(1), "gripper_init": "closed", "steps": [
        leg(0.0, 1.0), leg(1.0, 1.5), {"type": "gripper", "label": "Place(x, g, p, y, q)", "action": "open"}, leg(1.5, 0.0)]}  # fmt: skip
    sim = SimpleNamespace(n_steps=0, arm="left", held_objects={}, object_poses_world=lambda: {},
                          block_grasping=lambda arm: calls.append(("block", arm)), unblock_grasping=lambda: calls.append("unblock"))  # fmt: skip
    args = SimpleNamespace(goal="", activity=False, gripper_hold_steps=0, no_video=True, grasping_mode="sticky")
    atom = lambda p, *a: {"predicate": p, "args": list(a)}
    run.do_execute(sim, args, tmp_path, plan, "t", atoms=[atom("stamp", "brush_1", "shoe_1")], record=False)
    run.do_execute(sim, args, tmp_path, plan, "t", atoms=[atom("pour", "cheese_1", "dough_1")], record=False)
    assert played == [["trajectory"] * 4] * 2 and calls == []
    run.do_execute(sim, args, tmp_path, plan, "t", atoms=[atom("push", "comic_book_3")], record=False)
    assert played[-1] == ["trajectory", "trajectory", "gripper", "trajectory"] and calls == [("block", "left"), "unblock"]
    run.do_execute(sim, args, tmp_path, plan, "t", atoms=[atom("on", "book_1", "table")], record=False)
    assert played[-1] == played[-2] and len(calls) == 2
    args.grasping_mode = "physical"  # no assist to block
    run.do_execute(sim, args, tmp_path, plan, "t", atoms=[atom("push", "comic_book_3")], record=False)
    assert len(calls) == 2
