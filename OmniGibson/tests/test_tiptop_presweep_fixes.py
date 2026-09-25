"""Pre-sweep fixes of 2026-09-22/23: falls into unloaded rooms, fold/unfold, locked-arm drift, sticky lift, floor
put-downs and container regions. Stubs only; no simulator."""

from types import MethodType, SimpleNamespace

import numpy as np
import pytest
import torch as th
import trimesh

from omnigibson.tiptop.r1pro import R1ProSim


def _stance_sim(room, loaded):
    floor = SimpleNamespace(name="floors_living", category="floors")
    return SimpleNamespace(
        base_box=lambda: np.array([[-0.4, -0.35, 0.04], [0.25, 0.35, 0.4]]),
        base_pose=lambda: (th.tensor([0.0, 0.0, 0.0]), th.tensor([0.0, 0.0, 0.0, 1.0])),
        scene_aabbs=lambda: [(floor, np.array([-10.0, -10.0, -0.1]), np.array([10.0, 10.0, 0.0]))],
        env=SimpleNamespace(scene=SimpleNamespace(
            seg_map=SimpleNamespace(get_room_instance_by_point=lambda p: room), load_room_instances=loaded)),
        robot=object(),
    )


def test_a_stance_in_a_room_the_scene_did_not_load_is_refused():
    free, why, _ = R1ProSim._footprint_free(_stance_sim("bedroom_0", ["living_room_0"]), 1.0, 1.0, ignore=(), arms=False)
    assert not free and "did not load" in why
    free, why, _ = R1ProSim._footprint_free(_stance_sim("living_room_0", ["living_room_0"]), 1.0, 1.0, ignore=(), arms=False)
    assert free, why
    free, why, _ = R1ProSim._footprint_free(_stance_sim("bedroom_0", None), 1.0, 1.0, ignore=(), arms=False)
    assert free, why  # a full scene loads every room


def test_a_floor_covering_is_ground_only_while_it_is_flat():
    """bringing_in_wood 301: the search spared every paver, the landing check (ground by height) refused the base on
    a 9.5 cm paver kerb, and 16 candidates went that way; a flat paver is still stood on (2026-09-23)."""

    def with_paver(top):
        sim = _stance_sim("living_room_0", None)
        floor, paver = sim.scene_aabbs(), SimpleNamespace(name="paver_kerb", category="paver")
        sim.scene_aabbs = lambda: floor + [(paver, np.array([0.8, 0.8, 0.0]), np.array([1.2, 1.2, top]))]
        return sim

    free, why, _ = R1ProSim._footprint_free(with_paver(0.095), 1.0, 1.0, ignore=(), arms=False)
    assert not free and why == "overlaps paver_kerb"
    free, why, _ = R1ProSim._footprint_free(with_paver(0.03), 1.0, 1.0, ignore=(), arms=False)
    assert free, why


def test_a_base_that_dropped_below_the_floor_did_not_settle_even_when_level():
    sim = SimpleNamespace(base_pose=lambda: (th.tensor([1.0, 2.0, -0.38]), th.tensor([0.0, 0.0, 0.0, 1.0])))
    level, why = R1ProSim.settled_level(sim, 1.0, 2.0)
    assert not level and "dropped 0.38 m" in why and sim.last_settle["drop_m"] == pytest.approx(0.38)
    sim.base_pose = lambda: (th.tensor([1.0, 2.0, 0.006]), th.tensor([0.0, 0.0, 0.0, 1.0]))
    assert R1ProSim.settled_level(sim, 1.0, 2.0) == (True, "")


def test_an_arm_already_folded_unfolds_to_the_ready_posture_not_to_the_fold():
    ramps = []
    sim = SimpleNamespace(
        q_home=[1.0, -1.4, -0.4, 0.0, -1.6, 0.3, -1.8], planned_joints=["torso_joint1", "torso_joint2", "torso_joint3",
        "torso_joint4", "left_arm_joint1", "left_arm_joint2", "left_arm_joint3"],
        q_arm=lambda: np.array([1.0, -1.4, -0.4, 0.0, 0.0, 0.0, 0.0]), posture={}, last_gripper=1.0,
        ramp_to=lambda *a, **k: ramps.append(a) or None,
    )
    assert R1ProSim.fold_for_travel(sim) == sim.q_home
    sim.q_arm = lambda: np.array([1.0, -1.4, -0.4, 0.0, -1.6, 0.3, -1.8])
    assert R1ProSim.fold_for_travel(sim) == [1.0, -1.4, -0.4, 0.0, -1.6, 0.3, -1.8]  # working posture: itself


@pytest.mark.parametrize("holding", [False, True])
def test_a_drifted_empty_locked_arm_is_driven_back_before_the_capture_and_a_holding_one_is_not(holding):
    ramps = []
    names = ["right_arm_joint1", "right_arm_joint2", "torso_joint1"]
    sim = SimpleNamespace(
        locked_nominal={"right_arm_joint1": 0.0, "right_arm_joint2": 0.0, "torso_joint1": 1.0},
        hands=lambda: {"radio_1": "right"} if holding else {},
        robot=SimpleNamespace(get_joint_positions=lambda: th.tensor([0.4, 0.0, 1.2]), arm_names=("left", "right"),
                              arm_joint_names={"left": [], "right": names[:2]}),
        joint_index={n: i for i, n in enumerate(names)}, posture={"right_arm_joint1": 0.4, "right_arm_joint2": 0.0},
        q_arm=lambda: np.zeros(2), last_gripper=1.0, ramp_to=lambda *a, **k: ramps.append(a) or None,
    )
    R1ProSim.restore_locked_arm(sim)
    if holding:
        assert ramps == []
    else:
        assert ramps and ramps[0][1]["right_arm_joint1"] == 0.0 and "torso_joint1" not in ramps[0][1]


def test_only_the_carried_object_may_touch_what_it_rested_on():
    from omnigibson.tiptop.collision import CARRIED, JointPathCollision

    model = JointPathCollision.__new__(JointPathCollision)
    model.links = np.array(["left_gripper_link"])
    model.local_centres = np.array([[0.0, 0.0, 1.0]])
    model.radii = np.array([0.02])
    model.self_ignore = set()
    model.buffer, model.self_buffer = 0.0, {}
    model.attachment_ignore = {"left_gripper_link": {"left_gripper_link"}}
    model.fk = SimpleNamespace(fk=lambda q, link: (np.array([q[0], 0.0, 0.0]), np.array([0.0, 0.0, 0.0, 1.0])))
    model.motion_bounds = lambda links, centres: np.ones((len(links), 3))
    model.bounds = np.ones((1, 3))
    shelf = trimesh.creation.box([0.2, 0.2, 0.02])
    shelf.apply_translation([0.0, 0.0, 0.93])
    book = ("left_gripper_link", np.array([[0.0, 0.0, 0.94]]), np.array([0.02]))  # rests on the shelf
    assert model.check([0.0], [0.0], np.eye(4), [("shelf", shelf)], attachments=[book])[:2] == ("attached_object_left", "shelf")
    assert model.check([0.0], [0.0], np.eye(4), [("shelf", shelf)], {"shelf": {CARRIED}}, attachments=[book]) is None
    shelf.apply_translation([0.0, 0.0, 0.06])  # now the gripper itself is in the shelf: never excused
    assert model.check([0.0], [0.0], np.eye(4), [("shelf", shelf)], {"shelf": {CARRIED}}, attachments=[book]) is not None


def test_a_floor_goal_gets_a_floor_slab_and_a_region_that_failed_is_not_offered_again():
    sim = SimpleNamespace(
        base_pose=lambda: (th.tensor([2.0, 3.0, 0.01]), th.tensor([0.0, 0.0, 0.0, 1.0])),
        scene_aabbs=lambda: [(SimpleNamespace(category="floors"), np.array([0.0, 0.0, -0.1]), np.array([5.0, 5.0, 0.0]))],
        inside_region=lambda item, container: {"dims": [0.3, 0.3, 0.02], "pose": [0.5, 0, 0.7, 1, 0, 0, 0]},
        label_of=lambda bddl: bddl.split(".")[0] + "_1",
        region_refused={("plate.n.04_2", "sink.n.01_1")},
    )
    sim.floor_surface = MethodType(R1ProSim.floor_surface, sim)
    atoms = [{"predicate": "ontop", "args": ["plate.n.04_1", "floor.n.01_1"]},
             {"predicate": "inside", "args": ["plate.n.04_2", "sink.n.01_1"]},
             {"predicate": "inside", "args": ["bowl.n.01_1", "tub.n.01_1"]}]
    out = R1ProSim.inside_regions(sim, atoms)
    assert set(out) == {"table", "tub_1"}  # the floor slab under the planner's support label; the sink refused
    slab = out["table"]
    assert slab["pose"][2] + slab["dims"][2] / 2 == pytest.approx(-0.01)  # top face at the floor, base frame


def test_an_opening_stance_the_destination_check_refuses_is_reported_not_raised():
    from omnigibson.tiptop.r1pro import BasePlacementCollision
    import inspect

    source = inspect.getsource(R1ProSim._drive_joint)  # the stance is taken here, for open_container and push_joint
    assert "except RuntimeError" in source and issubclass(BasePlacementCollision, RuntimeError)


def test_button_hints_describe_the_button_of_every_toggle_goal():
    """r1pro called self.button_label, a function that had moved to b1k.bridge.protocol: every task with a
    toggled_on goal (radio, lights, fax, modem, scanner) died describing its button (smoke run 2026-09-22)."""
    sim = SimpleNamespace(
        button_world=lambda bddl: (np.array([1.0, 0.0, 1.0]), np.array([-1.0, 0.0, 0.0]), 0.01),
        to_base=lambda p, q: (p, q), label_of=lambda bddl: "radio_receiver_1",
    )
    out = R1ProSim.button_hints(sim, [{"predicate": "toggled_on", "args": ["radio_receiver.n.01_1"]}])
    assert list(out) == ["radio_receiver_1_button"] and out["radio_receiver_1_button"]["radius"] == 0.01
    off = R1ProSim.button_hints(sim, [{"predicate": "not", "args": ["toggled_on", "radio_receiver.n.01_1"]}])
    assert list(off) == ["radio_receiver_1_button"]


def _placement_sim(blocked_beyond):
    """A robot whose straight unfold at any stance is clear up to ``blocked_beyond`` of the way (joint 0 of [0, 0]
    toward the targets)."""
    checks, unfolds = [], []

    def collide(x, y, yaw, then=None):
        checks.append(then)
        end = None if then is None else (then[-1] if np.ndim(then) == 2 else then)
        return ("left_gripper_finger_link1", "coffee_table", 5) if end is not None and end[0] > blocked_beyond else None

    sim = SimpleNamespace(
        fold_for_travel=lambda: [1.0, 1.0], q_arm=lambda: [0.0, 0.0], base_placement_collision=collide,
        move_base=lambda *a: None, aim_overview=lambda *a: None, overview_view="shoulder",
        unfold_after_travel=unfolds.append, log_teleport_contacts=lambda: None,
        held_objects={}, _fold_blocked=False, planned_joints=["j1", "j2"], arm="left", level_held=lambda arm: set(),
    )
    sim.unfold_reach = MethodType(R1ProSim.unfold_reach, sim)
    return sim, checks, unfolds


def test_a_stance_is_taken_with_as_much_of_the_unfold_as_stays_clear(monkeypatch):
    """turning_on_radio 2026-09-22: the 0.65 m stance was refused because the READY hand would land on the coffee
    table the radio stands on; the 0.75 m stance it took instead reached no grasp."""
    from omnigibson.tiptop.r1pro import BasePlacementCollision

    monkeypatch.setattr("omnigibson.tiptop.r1pro.gm", SimpleNamespace(HEADLESS=True))
    sim, checks, unfolds = _placement_sim(blocked_beyond=0.6)
    R1ProSim.place_robot(sim, 1.0, 0.0, 0.0, min_unfold=0.5)
    assert checks[0] is None  # the landing posture first
    assert unfolds == [[0.5, 0.5]]  # then out as far as stays clear, not refused for the full unfold
    unfolds.clear()
    with pytest.raises(BasePlacementCollision) as refused:
        R1ProSim.place_robot(sim, 1.0, 0.0, 0.0, min_unfold=0.75)
    assert refused.value.unfold == 0.5 and unfolds == []  # refused BEFORE the teleport
    checks.clear()
    R1ProSim.place_robot(sim, 1.0, 0.0, 0.0, unfold=False)  # an opening stance stays folded
    assert checks == [None] and unfolds == []


def test_a_folded_arm_is_no_stance_but_the_furthest_partial_unfold_is():
    """setup_a_bar 2026-09-22: standing with the arm left in the fold, every capture ramp was refused and the
    wrist cameras saw nothing; so a stance where it cannot come out at all is refused outright."""
    from omnigibson.tiptop.r1pro import BasePlacementCollision

    target = SimpleNamespace(aabb_center=th.tensor([0.0, 0.0, 0.5]), aabb=(th.zeros(3), th.ones(3)))
    reach = {1.0: 0.25, 2.0: 0.0, 3.0: 0.25, 4.0: 0.0}  # x of each candidate -> how far the arm unfolds there
    placed = []

    def best(*args, **kwargs):
        x = float(len(kwargs["avoid"]) % 4 + 1)
        return (0, x, 0.0, 0.0, [0.5], [0.0], 0.1), {}

    def place(x, y, yaw, note="", min_unfold=0.0, **kwargs):
        placed.append((x, min_unfold))
        if reach[x] < min_unfold or reach[x] == 0.0:
            raise BasePlacementCollision("short", unfold=reach[x])
        return {"x": x, "y": y, "yaw": yaw}

    sim = SimpleNamespace(
        scene_object=lambda name: target, grasped_labels=lambda: {}, objects={},
        robot_cam=SimpleNamespace(get_position_orientation=lambda: (th.tensor([0.0, 0.0, 1.4]), th.tensor([0.0, 0, 0, 1]))),
        camera_floor_distance=lambda z: 0.4, best_base_pose=best, hands=lambda: {}, xy_radius=lambda name: 0.1, place_robot=place,
        base_placement_collision=lambda x, y, yaw: None, hidden_from_here=lambda names: {}, to_base=lambda *args: args,
        look_at=lambda *names: None,
    )
    result = R1ProSim.place_robot_for(sim, "target")
    assert result["x"] == 1.0  # of the two 25% stances, the better-scored (earlier) one
    assert placed[-1] == (1.0, 0.25)
    reach.update({1.0: 0.0, 3.0: 0.0})
    with pytest.raises(RuntimeError, match="after 8 candidates"):  # nowhere the arm comes out: unreachable
        R1ProSim.place_robot_for(sim, "target")


def test_unfold_reach_finds_the_longest_clear_fraction_before_the_teleport():
    sim, checks, _ = _placement_sim(blocked_beyond=0.6)
    fraction, legs, why = R1ProSim.unfold_reach(sim, [1.0, 2.0], 0, 0, 0)
    assert (fraction, legs) == (0.5, [[0.5, 1.0]]) and "coffee_table" in why
    sim, checks, _ = _placement_sim(blocked_beyond=-1.0)
    assert R1ProSim.unfold_reach(sim, [1.0, 1.0], 0, 0, 0)[:2] == (0.0, None) and len(checks) == 4


def test_planned_standoff_asks_for_the_presolved_configuration_in_planner_joint_order():
    from omnigibson.tiptop.r1pro import R1ProSim

    asked, held = [], []
    sim = SimpleNamespace(
        arm="left", planned_joints=["torso_joint1", "left_arm_joint1"], OPEN=1.0,
        hold=lambda n, g: held.append(g),
        planned_approach=lambda where, quat, note, goal_q=None: asked.append((note, goal_q)) or True,
    )
    ik = SimpleNamespace(fk=lambda q, frame: (np.zeros(3), np.array([0.0, 0, 0, 1])))
    joints_of = ["left_arm_joint1", "torso_joint1"]  # a different order from the planner's
    assert R1ProSim.planned_standoff(sim, "left", ik, joints_of, [0.3, 0.7], "cabinet")
    assert asked == [("reach the standoff of cabinet", [0.7, 0.3])] and held == [1.0]
    asked.clear()
    sim.planned_approach = lambda where, quat, note, goal_q=None: asked.append(goal_q) or goal_q is None
    assert R1ProSim.planned_standoff(sim, "left", ik, joints_of, [0.3, 0.7], "cabinet")
    assert asked == [[0.7, 0.3], None]  # the pre-solved configuration first, then any configuration at the pose
    assert not R1ProSim.planned_standoff(sim, "right", ik, joints_of, [0.3, 0.7], "cabinet")
    # the fingers as the approach will have them: a bar's jaw, not reopened (the planner plans with them as they are)
    assert R1ProSim.planned_standoff(sim, "left", ik, joints_of, [0.3, 0.7], "cabinet", 0.1) and held[-1] == 0.1


def test_return_to_ready_ramps_then_asks_the_planner_for_the_ready_configuration():
    from omnigibson.tiptop.r1pro import R1ProSim

    asked = []

    def sim_with(ramp_result, held=None):
        sim = SimpleNamespace(
            q_home=[1.0, 2.0], q_arm=lambda: [0.0, 0.0], posture={}, last_gripper=0.0, arm="left", level=set(),
            robot=SimpleNamespace(_ag_obj_in_hand={"left": held}),
            ramp_to=lambda *args, **kwargs: ramp_result,
            planned_approach=lambda where, quat, note, goal_q=None: asked.append(goal_q) or True,
        )
        sim.level_held = MethodType(R1ProSim.level_held, sim)
        return sim

    assert R1ProSim.return_to_ready(sim_with(None)) and asked == []
    assert R1ProSim.return_to_ready(sim_with(("left_gripper_link intersects booth", 0, 0.0)))
    assert asked == [[1.0, 2.0]]  # the planner is asked for the ready CONFIGURATION, not a pose
    # holding something: the move request cannot carry it (it would be a room static the fingers start in)
    assert not R1ProSim.return_to_ready(sim_with(("attached_object_left intersects table", 0, 0.0), held=object()))
    assert asked == [[1.0, 2.0]]
    home = SimpleNamespace(q_home=[1.0, 2.0], q_arm=lambda: [1.0, 2.0])
    assert R1ProSim.return_to_ready(home)  # already there: nothing moves
    # E-level: a load carried level (a plate under a pizza) is not turned toward the ready posture past 20 deg
    ramps = []
    sim = sim_with(None, held=object())
    sim.level, sim.hands, sim.ramp_to = {"plate_1"}, lambda: {"plate_1": "left"}, lambda *a, **k: ramps.append(a) or None
    sim.planned_joints, sim.joint_index = ["j1", "j2"], {"j1": 0, "j2": 1}
    sim.robot.get_joint_positions = lambda: th.tensor([0.0, 0.0])
    flat, rolled = [0.0, 0.0, 0.0, 1.0], [np.sin(np.pi / 4), 0.0, 0.0, np.cos(np.pi / 4)]  # the hand rolled 90 deg
    yawed = [0.0, 0.0, np.sin(np.pi / 4), np.cos(np.pi / 4)]  # turned about the vertical: the load stays level
    sim.arm_ik = lambda arm, frame=None, with_torso=False: SimpleNamespace(
        arm_joints=["j1", "j2"], fk=lambda q: (np.zeros(3), rolled if list(q) == [1.0, 2.0] else flat))
    assert not R1ProSim.return_to_ready(sim) and ramps == []
    sim.arm_ik = lambda arm, frame=None, with_torso=False: SimpleNamespace(
        arm_joints=["j1", "j2"], fk=lambda q: (np.zeros(3), yawed if list(q) == [1.0, 2.0] else flat))
    assert R1ProSim.return_to_ready(sim) and len(ramps) == 1


def test_switching_planners_brings_the_first_arm_back_to_ready_first():
    from omnigibson.tiptop.bench import Episode

    order = []
    sim = SimpleNamespace(
        arm="left",
        return_to_ready=lambda note: order.append("ready") or True,
        adopt_embodiment=lambda emb: order.append(("adopt", emb["robot_type"])),
    )
    ep = SimpleNamespace(sim=sim, planners={"left": ("L", {}), "right": ("R", {"embodiment": {"robot_type": "r1pro_right"}})})
    Episode.use_arm(ep, "right")
    assert order == ["ready", ("adopt", "r1pro_right")] and sim.move_planner == "R"


def test_a_dry_candidate_search_still_takes_the_furthest_partial_unfold():
    from omnigibson.tiptop.r1pro import BasePlacementCollision

    target = SimpleNamespace(aabb_center=th.tensor([0.0, 0.0, 0.5]), aabb=(th.zeros(3), th.ones(3)))
    placed = []

    def best(*args, **kwargs):  # two stances, then the search runs dry
        n = len(kwargs["avoid"])
        return ((0, float(n + 1), 0.0, 0.0, [0.5], [0.0], 0.1), {}) if n < 2 else (None, {"overlaps desk": 9})

    def place(x, y, yaw, note="", min_unfold=0.0, **kwargs):
        placed.append((x, min_unfold))
        if min_unfold > 0.25:
            raise BasePlacementCollision("short", unfold=0.25)
        return {"x": x, "y": y, "yaw": yaw}

    sim = SimpleNamespace(
        scene_object=lambda name: target, grasped_labels=lambda: {}, objects={},
        robot_cam=SimpleNamespace(get_position_orientation=lambda: (th.tensor([0.0, 0.0, 1.4]), th.tensor([0.0, 0, 0, 1]))),
        camera_floor_distance=lambda z: 0.4, best_base_pose=best, hands=lambda: {}, xy_radius=lambda name: 0.1, place_robot=place,
        base_placement_collision=lambda x, y, yaw: None, hidden_from_here=lambda names: {}, to_base=lambda *args: args,
        look_at=lambda *names: None,
    )
    assert R1ProSim.place_robot_for(sim, "target")["x"] == 1.0
    assert placed[-1] == (1.0, 0.25)


def test_captures_at_a_partway_stance_return_to_that_stance_not_to_the_unreachable_ready_posture(monkeypatch):
    monkeypatch.setattr("omnigibson.tiptop.r1pro.gm", SimpleNamespace(HEADLESS=True))
    sim, checks, unfolds = _placement_sim(blocked_beyond=0.6)
    sim.stance_ready = None
    R1ProSim.place_robot(sim, 1.0, 0.0, 0.0, min_unfold=0.5)
    assert sim.stance_ready == [0.5, 0.5]
    sim, checks, unfolds = _placement_sim(blocked_beyond=2.0)  # a clear stance: back to q_home
    sim.stance_ready = [0.1, 0.1]
    R1ProSim.place_robot(sim, 1.0, 0.0, 0.0, min_unfold=0.5)
    assert sim.stance_ready is None
    captured = []
    sim = SimpleNamespace(stance_ready=[0.5, 0.5], q_home=[1.0, 1.0], look_arm=None,
                          _capture_views=lambda task, ready: captured.append(ready) or ({}, {}))
    R1ProSim._capture_with_motion(sim, "task")
    assert captured == [[0.5, 0.5]]


def test_an_unfold_the_straight_line_cannot_make_is_tried_elbow_first():
    """From the travel fold the hand hangs low beside the base; straight out it sweeps forward at coffee-table
    height, elbow first it rises before it travels (turning_on_radio's 0.65 m stance, 2026-09-22)."""
    names = ["torso_joint1", "left_arm_joint1", "left_arm_joint2", "left_arm_joint3", "left_arm_joint4"]
    checked = []

    def collide(x, y, yaw, then=None):
        checked.append(then)
        legs = [then] if np.ndim(then) == 1 else then
        elbow_raised_first = len(legs) == 2 and legs[0][4] != 0.0 and legs[0][1] == 0.0
        return None if elbow_raised_first else ("left_gripper_finger_link1", "coffee_table", 3)

    sim = SimpleNamespace(q_arm=lambda: [0.2, 0.0, 0.0, 0.0, 0.0], planned_joints=names, base_placement_collision=collide)
    fraction, legs, why = R1ProSim.unfold_reach(sim, [0.2, 1.0, 1.0, 1.0, 1.0], 0, 0, 0)
    assert fraction == 1.0 and len(legs) == 2
    assert legs[0] == [0.2, 0.0, 0.0, 0.0, 1.0] and legs[1] == [0.2, 1.0, 1.0, 1.0, 1.0]
    assert len(checked) == 2  # straight, then elbow first; the torso-first/arm-first copies are not re-checked


def _pick_ep(attached_first, target, releases=True, grasping_mode="sticky"):
    from omnigibson.tiptop.bench import Episode

    calls = []

    def hold(n, gripper=None):
        calls.append(("hold", gripper))
        if releases:
            sim.robot._ag_obj_in_hand["left"] = None

    sim = SimpleNamespace(
        arm="left", n_steps=0, OPEN=1.0, held_objects={}, push_face=lambda bddl: None,
        robot=SimpleNamespace(_ag_obj_in_hand={"left": attached_first}),
        press_grasp=lambda arm, bddl, spare=(): False, scene_object=lambda name: target, hold=hold,
        return_to_ready=lambda note, allowed_contacts=None: calls.append(("ready", note, allowed_contacts)) or True,
        retreat_contacts=lambda bddl=None: {"floors_1": {"left_gripper_link"}, "target": {"left_gripper_link"}},
    )
    ep = SimpleNamespace(
        sim=sim, rounds=1, records=[], planners={"left": ("planner", {})},
        args=SimpleNamespace(grasping_mode=grasping_mode),
        stand_for=lambda bddl: None, plan_and_execute=lambda atoms, floor=False: None, reaches_floor=lambda b: False,
        support_of=lambda b: None, holding=lambda b: b in sim.held_objects,
        note_pressed_grasp=lambda b, after=None: calls.append(("note", b)) or sim.held_objects.update({b: "left"}),
    )
    ep.pick = MethodType(Episode.pick, ep)
    return ep, calls


def test_a_target_the_assist_took_on_the_way_in_is_recorded_not_an_episode_ending_block():
    """assembling_gift_baskets 2026-09-22: the hand closes before it approaches, the assist took the candle on the way
    in, the close at the precontact pose was refused, and "an unconfirmed grasp on pillar_candle_88" ended the
    episode at step 1031 of 39090."""
    candle = SimpleNamespace(name="pillar_candle_88")
    ep, calls = _pick_ep(attached_first=candle, target=candle)
    assert ep.pick("candle.n.01_1") is True
    assert ("note", "candle.n.01_1") in calls


def test_another_body_the_assist_holds_is_let_go_and_the_arm_comes_back_before_the_next_object():
    from b1k.bridge.strategies import TransferBlocked

    ep, calls = _pick_ep(attached_first=SimpleNamespace(name="desk_12"), target=SimpleNamespace(name="pen_3"))
    assert ep.pick("pen.n.01_1") is False
    assert calls[0] == ("hold", 1.0) and calls[-1][0] == "ready"
    assert calls[-1][2] == {"floors_1": {"left_gripper_link"}, "target": {"left_gripper_link"}}  # it may leave them
    ep, calls = _pick_ep(attached_first=SimpleNamespace(name="desk_12"), target=SimpleNamespace(name="pen_3"),
                         releases=False)
    with pytest.raises(TransferBlocked, match="unconfirmed grasp on desk_12"):  # a hand that will not let go
        ep.pick("pen.n.01_1")


def test_the_landing_check_gives_the_floor_no_extra_clearance():
    """laying_tile_floors 2026-09-22: the hand low after a floor pick sat 2 cm off the floor at every destination and
    TELEPORT_CLEARANCE refused all of them."""
    from omnigibson.tiptop.r1pro import TELEPORT_CLEARANCE

    seen = []
    model = SimpleNamespace(
        joint_names=["j"], check_polyline=lambda path, tf, obstacles, allowed, attachments=(), clearance=0.0:
            seen.append((sorted(n for n, _ in obstacles), clearance)),
    )
    floor, desk = SimpleNamespace(name="floors_1", category="floors"), SimpleNamespace(name="desk_1", category="desk")
    paver = SimpleNamespace(name="paver_1", category="paver")  # outdoor ground under another name
    low, high = np.array([0.0, 0.0, -0.1]), np.array([1.0, 1.0, 0.8])
    sim = SimpleNamespace(
        _motion_collision_model=lambda: model, robot=SimpleNamespace(get_joint_positions=lambda: th.zeros(1),
                                                                     _ag_obj_in_hand={}),
        joint_index={"j": 0}, planned_joints=["j"],
        _motion_obstacles=lambda held, allowed, base=None, only=None: [("floors_1", None), ("desk_1", None), ("paver_1", None)],
        scene_aabbs=lambda: [(floor, low, high), (desk, low, high), (paver, low, np.array([1.0, 1.0, 0.02]))],
    )
    R1ProSim.base_placement_collision(sim, 1.0, 0.0, 0.0)
    assert seen == [(["desk_1"], TELEPORT_CLEARANCE), (["floors_1", "paver_1"], 0.0)]


def test_a_hand_backing_out_may_touch_only_the_target_and_the_floor_with_its_hand_links():
    floor, desk = SimpleNamespace(name="floors_1", category="floors"), SimpleNamespace(name="desk_1", category="desk")
    bottle = SimpleNamespace(name="bottle_228", category="bottle")
    sim = SimpleNamespace(
        arm="left", robot=SimpleNamespace(finger_link_names={"left": ["left_gripper_finger_link1", "left_gripper_finger_link2"]}),
        scene_aabbs=lambda: [(floor, None, None), (desk, None, None), (bottle, None, None)],
        scene_object=lambda name: bottle,
    )
    from omnigibson.tiptop.collision import CARRIED

    hand = {"left_gripper_link", "left_gripper_finger_link1", "left_gripper_finger_link2"}
    # what the hand carries may leave the floor too (a knocked-over bottle lifted off it)
    assert R1ProSim.retreat_contacts(sim, "bottle.n.01_1") == {"floors_1": hand | {CARRIED}, "bottle_228": hand}
    assert R1ProSim.retreat_contacts(sim) == {"floors_1": hand | {CARRIED}}  # the desk stays an obstacle


def test_the_travel_fold_brings_the_torso_home_and_the_unfold_goes_to_ready():
    """re_shelving_library_books 2026-09-23: after a high-shelf place the torso stood raised; folding only the arm and
    unfolding back to that posture swept the wrist camera through the furniture at 175 stances in a row."""
    ramped = []
    sim = SimpleNamespace(
        q_home=[1.0, -1.4, -1.6, 0.3], planned_joints=["torso_joint1", "torso_joint2", "left_arm_joint1", "left_arm_joint2"],
        q_arm=lambda: [1.3, -1.0, -0.5, 0.9], posture={}, last_gripper=1.0,
        ramp_to=lambda targets, *a, **k: ramped.append(list(targets)),
    )
    unfold_to = R1ProSim.fold_for_travel(sim)
    assert ramped == [[1.0, -1.4, 0.0, 0.0]]  # arm folded, torso at home, not left raised
    assert unfold_to == [1.0, -1.4, -1.6, 0.3]  # the ready posture, not the high-shelf one it came from


def test_a_carrying_robot_takes_any_landing_clear_stance_whatever_the_unfold():
    """putting_away_toys 2026-09-23: with a toy in the hand every stance at the toy box was refused at 0% unfold."""
    target = SimpleNamespace(aabb_center=th.tensor([0.0, 0.0, 0.5]), aabb=(th.zeros(3), th.ones(3)))
    asked = []

    def place(x, y, yaw, note="", min_unfold=0.0, **kwargs):
        asked.append(min_unfold)
        return {"x": x, "y": y, "yaw": yaw}

    sim = SimpleNamespace(
        scene_object=lambda name: target, grasped_labels=lambda: {}, objects={},
        robot_cam=SimpleNamespace(get_position_orientation=lambda: (th.tensor([0.0, 0.0, 1.4]), th.tensor([0.0, 0, 0, 1]))),
        camera_floor_distance=lambda z: 0.4, xy_radius=lambda name: 0.1, place_robot=place,
        best_base_pose=lambda *a, **k: ((0, 1.0, 0.0, 0.0, [0.5], [0.0], 0.1), {}), hands=lambda: {"toy_1": "left"},
        base_placement_collision=lambda x, y, yaw: None, hidden_from_here=lambda names: {}, to_base=lambda *args: args,
        look_at=lambda *names: None,
    )
    R1ProSim.place_robot_for(sim, "target")
    assert asked == [0.0]
    sim.hands = lambda: {}
    R1ProSim.place_robot_for(sim, "target")
    assert asked[-1] == 0.5  # empty-handed, a folded-arm stance is still refused


def test_an_object_the_assist_takes_on_the_way_back_is_let_go():
    """setup_a_bar 302 2026-09-23: the return to ready pushed the closed hand into a can, the assist took it, and the
    unrecorded attachment made every later stance collide."""
    stray = SimpleNamespace(name="can_of_soda_58")
    ep, calls = _pick_ep(attached_first=None, target=SimpleNamespace(name="can_of_soda_3"))

    def back(note, allowed_contacts=None):
        calls.append(("ready", note))
        ep.sim.robot._ag_obj_in_hand["left"] = stray  # taken during the way back
        return False

    ep.sim.return_to_ready = back
    assert ep.pick("can__of__soda.n.01_3") is False
    assert calls[-1] == ("hold", 1.0)  # opened after the way back


def test_a_sticky_pick_ends_at_the_ready_posture_carrying():
    """putting_shoes_on_rack 2026-09-23: 10 sticky carries searched their destination stance from the crouch the
    grasp left (head camera 0.75-1.04 m) and 9 found none; the 2 planned carries, from the ready posture, both did."""
    shoe = SimpleNamespace(name="gym_shoe_77")
    ep, calls = _pick_ep(attached_first=shoe, target=shoe)
    ep.sim.press_grasp = lambda arm, bddl, spare=(): True
    assert ep.pick("gym_shoe.n.01_1") is True
    assert calls[-2] == ("note", "gym_shoe.n.01_1")
    assert calls[-1][:2] == ("ready", "ready posture with gym_shoe.n.01_1 in hand")
    assert calls[-1][2] == {"floors_1": {"left_gripper_link"}, "target": {"left_gripper_link"}}  # it may leave them


def test_under_assisted_grasping_every_pick_is_a_planner_round_and_the_press_never_runs():
    """The evaluator welds only a grasp whose finger-to-finger ray hits the object; the sticky press closes the hand
    before it touches, so under assisted it can take nothing. The jaw gate that skipped the planner round for a
    10 cm candle is gone with it: the planner's part grasps are what a bulky object gets (A-assist, 2026-09-24)."""
    candle = SimpleNamespace(name="pillar_candle_88")
    for mode, pressed, rounds in (("assisted", [], 2), ("sticky", ["candle.n.01_1"], 1)):
        planned, presses = [], []
        ep, calls = _pick_ep(attached_first=None, target=candle, grasping_mode=mode)
        ep.rounds = 2
        ep.sim.press_grasp = lambda arm, bddl, spare=(): presses.append(bddl) or True
        ep.plan_and_execute = lambda atoms, floor=False: planned.append(atoms)
        assert ep.pick("candle.n.01_1") is (mode == "sticky")
        assert planned == [[{"predicate": "holding", "args": ["candle.n.01_1"]}]] * rounds
        assert presses == pressed  # the fallback still runs after a failed round under sticky


def test_a_lift_refused_by_the_robots_own_body_lets_go_where_the_object_rests():
    """laying_tile 301/302/303 (2026-09-23): the post-grasp lift was refused at sample 0 against base_link, the tile
    stayed in the hand, and no motion ever left that start: 0 env steps after the grasp, TransferBlocked."""
    from omnigibson.tiptop.run import DROP_STEPS

    def sim_with(stopped):
        holds = []
        tile = SimpleNamespace(name="ceramic_tile_186", aabb=(th.zeros(3), th.ones(3)))
        sim = SimpleNamespace(
            robot=SimpleNamespace(get_joint_positions=lambda: th.zeros(1)), joint_index={"j": 0},
            rests_against=lambda obj: set(), retreat_contacts=lambda: {}, objects={"tile_1": tile}, held_objects={},
            _targets_from=lambda names, q: q, posture={}, CLOSE=-1.0, OPEN=1.0,
            _motion_collision_model=lambda: SimpleNamespace(links=np.array(["base_link", "torso_link4"])),
            ramp_to=lambda *a, **k: stopped, hold=lambda n, gripper: holds.append((n, gripper)),
        )
        return sim, tile, holds

    ik = SimpleNamespace(fk=lambda q: (np.zeros(3), np.array([0.0, 0.0, 0.0, 1.0])), solve=lambda *a, **k: [0.1])
    sim, tile, holds = sim_with(("attached_object_left intersects base_link", 0, 0.0, 0))
    assert R1ProSim.lift_held(sim, ik, tile, [0.0, 0.0, -1.0], ["j"], "from above") is False
    assert holds == [(DROP_STEPS, 1.0)] and sim.held_objects == {}  # let go, and the one-motion hold record is gone
    for stopped in (("attached_object_left intersects pillar_candle_85", 0, 0.0, 0),  # a neighbour: the carry goes on
                    ("left_gripper_finger_link1 intersects base_link", 0, 0.0, 200),  # partway: the start was clear
                    ("left_arm_joint4", 12, 0.2)):  # stopped while moving: the object is off its support
        sim, tile, holds = sim_with(stopped)
        assert R1ProSim.lift_held(sim, ik, tile, [0.0, 0.0, -1.0], ["j"], "from above") is False
        assert holds == []


def test_a_look_configuration_that_swings_into_the_robot_is_re_solved_on_the_abducted_branch(monkeypatch):
    """bringing_in_wood 301/303 (2026-09-23): Lula's branch from the ready seed rolled the upper arm across the head
    camera and the fingers stopped on zed_link; the same hand pose from the abducted seed (302) swung out clear."""
    monkeypatch.setattr("omnigibson.tiptop.r1pro.look_pose", lambda target, shoulder, side, offset: (offset, None))
    monkeypatch.setattr("omnigibson.tiptop.r1pro.link_pose_for_camera", lambda eye, quat, cam: (eye, quat))
    joints = [f"left_arm_joint{i}" for i in range(1, 8)]
    seeds, swings = [], []
    ik = SimpleNamespace(solve=lambda pos, quat, seed: seeds.append(list(seed)) or [seed[1]] * 7)
    sim = SimpleNamespace(
        robot=SimpleNamespace(arm_joint_names={"left": joints}, arm_link_names={"left": ["left_arm_link1"]},
                              get_joint_positions=lambda: th.zeros(7),
                              links={"left_arm_link1": SimpleNamespace(get_position_orientation=lambda: (0, 0))}),
        joint_index={j: i for i, j in enumerate(joints)}, arm_ik=lambda arm: ik, scene_aabbs=lambda: [],
        to_base=lambda pos, quat: (th.zeros(3), None), camera_in_link={"left": np.eye(4)},
        links_in_base_box=lambda arm, ik, q: [], links_before_camera=lambda arm, ik, q, target: [],
        swing_collision=lambda arm, here, goal: swings.append(list(goal))
        or (("left_gripper_finger_link1", "zed_link") if goal[1] == 0.0 else None),
        path_contacts=lambda arm, ik, here, goal, aabbs: (0, []),
    )
    assert R1ProSim.wrist_look(sim, "left", np.array([0.6, 0.0, 0.85])) == [2.0] * 7
    assert seeds[0][1] == 0.0 and seeds[1][1] == 2.0  # the ready seed, then the abducted branch of the same offset
    assert swings == [[0.0] * 7, [2.0] * 7]  # the first branch was checked and refused, the second checked and taken,
    # and the six candidates ranked behind them were never swing-checked (about 1 s each)


def test_a_locked_arm_whose_straight_way_back_is_refused_goes_elbow_first():
    """putting_shoes_on_rack 302 (2026-09-23): the straight line back crossed the shoe in the other hand, nothing
    else was tried, and 8 rounds were refused ('Robot locked joints differ from collision model')."""
    names = ["right_arm_joint1", "right_arm_joint2", "right_arm_joint3", "right_arm_joint4", "torso_joint1"]
    now = dict(zip(names, [-0.57, -0.1, 0.57, -1.7, 1.0]))
    ramps = []

    def ramp_to(q_arm, posture, gripper, settle, note="", max_vel=None):
        moving = {j for j, v in posture.items() if abs(v - now[j]) > 1e-6}
        ramps.append(sorted(moving))
        if "right_arm_joint4" in moving and moving != {"right_arm_joint4"}:
            return ("attached_object_left intersects right_gripper_link", 0, 0.0)  # elbow and shoulder together
        now.update(posture)
        return None

    sim = SimpleNamespace(
        locked_nominal={j: 0.0 for j in names[:4]} | {"torso_joint1": 1.0}, hands=lambda: {"gym_shoe_1": "left"},
        robot=SimpleNamespace(get_joint_positions=lambda: th.tensor([now[j] for j in names]),
                              arm_names=("left", "right"), arm_joint_names={"left": [], "right": names[:4]}),
        joint_index={n: i for i, n in enumerate(names)}, posture={j: now[j] for j in names[:4]},
        q_arm=lambda: np.zeros(2), last_gripper=1.0, ramp_to=ramp_to,
    )
    R1ProSim.restore_locked_arm(sim)
    assert ramps == [sorted(names[:4]), ["right_arm_joint4"], sorted(names[:3])]  # straight refused; elbow, then the rest
    assert all(abs(now[j]) < 1e-9 for j in names[:4])


def test_the_tuck_takes_the_idle_arm_to_its_own_planners_ready_posture_unless_it_holds_something():
    ramps = []
    right = [f"right_arm_joint{i}" for i in range(1, 8)]
    sim = SimpleNamespace(
        other_arm="right", hands=lambda: {}, posture={j: 0.0 for j in right}, q_arm=lambda: np.zeros(11),
        robot=SimpleNamespace(arm_joint_names={"right": right}), last_gripper=1.0,
        ramp_to=lambda q, posture, *a, **k: ramps.append(posture) or None,
    )
    assert R1ProSim.tuck_idle_arm(sim)
    assert ramps[0]["right_arm_joint4"] < -1.5 and set(ramps[0]) == set(right)  # the elbow bent; no torso joint
    sim.hands = lambda: {"honey_jar_1": "right"}
    assert not R1ProSim.tuck_idle_arm(sim) and len(ramps) == 1


@pytest.mark.parametrize("refusal", ["right_gripper_finger_link2 intersects cabinet",
                                     "base_link intersects right_gripper_finger_link1",  # the model's link order
                                     "left_arm_link6 intersects right_gripper_finger_link1",
                                     "left_gripper_link intersects cabinet"])
def test_the_idle_arm_is_tucked_when_it_rides_the_torso_into_the_handle_approach(monkeypatch, refusal):
    """store_honey 302/303 (2026-09-23): 'handle approach rejected: right_gripper_finger_link2 intersects
    bottom_cabinet_slgzfc_0' -- the idle right hand, hanging, rode the torso's lean into the cabinet and the episode
    ended on that one refusal. The robot's own pairs are named base, left, right: the idle hand into the base or the
    working forearm is its to fix too. A refusal by the working arm's own hand is not."""
    joint = dict(name="j", kind="prismatic", lower=0.0, upper=0.3, position=0.0, axis=[1, 0, 0], origin=[0, 0, 0],
                 link="drawer", closed=0.0)
    state = [joint]
    monkeypatch.setattr("omnigibson.tiptop.r1pro.openable_joints", lambda obj: [state[-1]])
    ramps, tucks = [], []

    def ramp_to(q_arm, posture, gripper, settle, note="", **kwargs):
        ramps.append(note)
        if note.startswith("approach") and not tucks:
            return (refusal, 0, 0.0, 300)
        return None

    grasp = dict(joint=joint, jaws=[np.array([0.0, 0.0, 1.0])], kind="bar", tips=[0.0, 0.0, 0.8], nudges=0, travel=0.24)
    plan = dict(solutions=[[0.1], [0.2], [0.3]], reached=1, why="", grasp_pose=np.eye(4), approach=np.array([-1.0, 0.0, 0.0]))
    ik = SimpleNamespace(fk=lambda q, frame: (np.zeros(3), np.array([0.0, 0.0, 0.0, 1.0])), solve=lambda *a, **k: None)
    sim = SimpleNamespace(
        arm="left", other_arm="right", OPEN=1.0, CLOSE=-1.0, hands=lambda: {}, posture={"right_arm_joint1": 0.0},
        q_home=None, planned_joints=[],
        scene_object=lambda name: SimpleNamespace(name="cab"), joint_index={"left_arm_joint1": 0},
        robot=SimpleNamespace(eef_links={"left": SimpleNamespace(get_position_orientation=lambda: (th.zeros(3), None))},
                              get_joint_positions=lambda: th.zeros(1)),
        container_grasps=lambda obj, joints, fraction, hand_world, height=None: [grasp],
        arm_ik=lambda arm, frame=None, with_torso=False, fingers=None: ik, ik_joint_names=lambda arm, with_torso=False: ["left_arm_joint1"],
        scene_aabbs=lambda: [], solve_pull=lambda ik, g, jaw, seed, **kwargs: (plan, ""),
        reach_plan=lambda *a, **kwargs: [], planned_standoff=lambda *a: True, _targets_from=lambda joints_of, q: q,
        container_body=lambda obj, link: None,
        ramp_to=ramp_to, tuck_idle_arm=lambda: tucks.append(1) or True, grasp_contacts=R1ProSim.grasp_contacts,
        close_on=lambda *a, **kwargs: ([0.2], True), follow_pull=lambda *a: state.append(dict(joint, position=0.25)) or (1, ""),
        hold=lambda n, g: None, arm_hits_scene=lambda *a, **kwargs: False,
    )
    sim._drive_joint = MethodType(R1ProSim._drive_joint, sim)
    result = R1ProSim.open_container(sim, "left", "cab", stand=False)
    if "right_" in refusal:
        assert result["opened"] and tucks == [1] and ramps.count("approach the handle of cab") == 2
    else:
        assert not result["opened"] and tucks == [] and "handle approach rejected" in result["why"]


def _shelves(tops, x=(0.7, 1.1), y=(-0.4, 0.4), thick=0.02):
    """A case of boards (top faces at ``tops``) with a back panel, as one trimesh in the world frame."""
    parts = []
    for z in tops:
        board = trimesh.creation.box([x[1] - x[0], y[1] - y[0], thick])
        board.apply_translation([(x[0] + x[1]) / 2, (y[0] + y[1]) / 2, z - thick / 2])
        parts.append(board)
    back = trimesh.creation.box([0.02, y[1] - y[0], max(tops)])
    back.apply_translation([x[1] - 0.01, (y[0] + y[1]) / 2, max(tops) / 2])
    return trimesh.util.concatenate(parts + [back])


def _region_sim(mesh, base=(0.0, 0.0), objects=None):
    sim = SimpleNamespace(
        base_pose=lambda: (th.tensor([*base, 0.0]), th.tensor([0.0, 0.0, 0.0, 1.0])),
        collision_mesh_world=lambda obj: mesh,
        scene_object=lambda name: (objects or {})[name],
        scene_aabbs=lambda: [(SimpleNamespace(category="floors"), np.array([-9.0, -9.0, -0.1]), np.array([9.0, 9.0, 0.0]))],
        objects={}, seen_boxes={},  # nothing seen by a capture yet
    )
    for name in ("to_base", "region_box", "shelf_of", "rest_region", "footprint_region", "floor_surface", "own_box",
                 "item_height", "seen_aabb", "stamp_region", "knife_region", "heat_region", "attach_target", "push_face",
                 "upright_rotation", "inside_region", "aim_target"):
        setattr(sim, name, MethodType(getattr(R1ProSim, name), sim))
    return sim


def _seen(name, at, lo, hi, sim):
    """A movable the captures saw as the box (lo, hi) in its own frame, standing at ``at`` unturned."""
    pose = (th.tensor(at), th.tensor([0.0, 0.0, 0.0, 1.0]))
    obj = SimpleNamespace(fixed_base=False, get_position_orientation=lambda: pose)
    sim.objects[name.split(".")[0] + "_1"] = obj
    sim.seen_boxes[name.split(".")[0] + "_1"] = (np.array(lo), np.array(hi))
    return obj


def test_a_stamp_box_covers_the_densest_cluster_the_tool_and_its_margin_can_take_and_no_more():
    """E-stamp: an adjacency remover deletes every particle inside its link's world AABB grown by 2 cm
    (particle_modifier.py:524-531). The region is the tool centres that put the cluster inside that box, grown by
    the tool's half extents (the planner keeps every sphere of the tool inside a surface), at the cluster's own
    height, holding the yaw the fit was made at. The vacuum removes inside its projection slab instead, offset from
    its body and hanging 1-21 mm under it: it hovers, halfway into that band."""
    objects = {}
    sim = _region_sim(None, objects=objects)
    brush = ("scrub_brush.n.01_1", [5.0, 5.0, 1.2], [-0.078, -0.024, -0.012], [0.078, 0.024, 0.012])  # hsejyi, scaled
    objects["scrub_brush.n.01_1"] = _seen(*brush, sim)
    objects["shoe.n.01_1"] = None
    patch = np.array([[1.00, 2.00, 0.31], [1.10, 2.00, 0.31], [1.05, 2.05, 0.31], [1.02, 2.03, 0.31], [1.08, 2.01, 0.312]])  # fmt: skip
    far = np.array([[1.50, 2.00, 0.31], [1.55, 2.02, 0.31], [1.52, 2.04, 0.31]])  # a second, smaller cluster
    region = sim.stamp_region("scrub_brush.n.01_1", "shoe.n.01_1", np.vstack([far, patch]))
    assert region["pose"][:2] == pytest.approx([1.05, 2.025]) and region["pose"][2] + 0.01 == pytest.approx(0.31)
    # tool centres from x 1.002 to 1.098 and y 2.006 to 2.044 put the patch inside the 0.196 x 0.088 reach: the
    # region is that box plus the brush's own half extents, and a millimetre more would miss a particle
    assert region["dims"] == pytest.approx([0.252, 0.086, 0.02], abs=1e-6)
    assert region["rotation"] == np.eye(3).tolist() and region["yaw_tolerance"] == pytest.approx(np.radians(5))
    assert sim.stamp_region("scrub_brush.n.01_1", "shoe.n.01_1", np.zeros((0, 3))) is None
    objects["vacuum.n.04_1"] = _seen("vacuum.n.04_1", [0.0, 0.0, 0.3], [-0.5, -0.15, -0.3], [0.5, 0.15, 0.3], sim)
    slab = (np.array([-0.0006, -0.151, -0.3214]), np.array([0.1186, 0.151, -0.3011]))  # bdmsbr's, in its own frame
    dust = np.array([[0.00, 0.0, 0.0], [0.10, 0.0, 0.0], [0.05, 0.25, 0.0], [0.02, 0.1, 0.0], [0.9, 0.9, 0.0]])
    region = sim.stamp_region("vacuum.n.04_1", "floor.n.01_1", dust, projection=slab)
    assert region["pose"][2] + 0.01 == pytest.approx(0.01125, abs=1e-4)  # hovering: the slab straddles the dust
    assert region["pose"][:2] == pytest.approx([-0.009, 0.125], abs=1e-6)  # the body sits back from the slab
    assert region["dims"][:2] == pytest.approx([1.0192, 0.352], abs=1e-6)  # the slab's reach, no margin, plus the body


def test_a_knife_is_set_down_across_the_foods_top_along_its_own_length():
    """E-knife: the slicer fires on any knife-link contact while armed (slicer_active.py:72-135), so the cut is a
    placement on the food's top: a strip the knife's length longer than the food one way and its width wider the
    other, the knife turned along it. A square of the knife's length, yaw free, let a third of the placements the
    planner accepted lie entirely beside the food (knife_region_miss.out)."""
    from omnigibson.tiptop.r1pro import STAMP_YAW_TOL

    objects = {}
    sim = _region_sim(None, objects=objects)
    objects["onion.n.01_1"] = _seen("onion.n.01_1", [1.0, 0.5, 0.8], [-0.05, -0.05, -0.05], [0.05, 0.05, 0.05], sim)
    knife = _seen("knife.n.01_1", [0.0, 0.0, 1.0], [-0.12, -0.01, -0.005], [0.12, 0.01, 0.005], sim)
    objects["knife.n.01_1"] = knife
    region = sim.knife_region("knife.n.01_1", "onion.n.01_1")
    assert region["pose"][:2] == pytest.approx([1.0, 0.5]) and region["pose"][2] + 0.01 == pytest.approx(0.85)
    assert region["dims"] == pytest.approx([0.34, 0.12, 0.02])  # along x as the knife lies; a centre inside crosses
    assert np.allclose(region["rotation"], np.eye(3)) and region["yaw_tolerance"] == STAMP_YAW_TOL
    yaw60 = th.tensor([0.0, 0.0, np.sin(np.pi / 6), np.cos(np.pi / 6)])  # the knife held 60 deg off x: nearer y
    knife.get_position_orientation = lambda: (th.tensor([0.0, 0.0, 1.0]), yaw60)
    region = sim.knife_region("knife.n.01_1", "onion.n.01_1")
    assert region["dims"] == pytest.approx([0.12, 0.34, 0.02])
    long_axis = np.array(region["rotation"]) @ [np.cos(np.pi / 3), np.sin(np.pi / 3), 0.0]
    assert np.abs(long_axis) == pytest.approx([0.0, 1.0, 0.0], abs=1e-9), "turned the short way onto the strip"
    sim.seen_boxes.pop("knife_1")
    assert sim.knife_region("knife.n.01_1", "onion.n.01_1") is None


def test_a_heat_box_keeps_the_item_inside_the_heat_links_sphere_on_the_cooking_surface():
    """E-heat: any overlap with the burner's 0.2 m sphere heats the whole object (heat_source_or_sink.py:253-266).
    The box holds the item's centre inside that sphere at the surface's height, on the map's cooking surface under
    the link (the grate), not on the back panel that rises within the same radius, nor the drip tray under the grate."""
    cooktop = trimesh.creation.box([0.6, 0.6, 0.02]).apply_translation([2.0, 1.0, 0.89])
    tray = trimesh.creation.box([0.2, 0.2, 0.01]).apply_translation([2.0, 1.15, 0.855])  # under the grate's bars
    panel = trimesh.creation.box([0.6, 0.05, 0.4]).apply_translation([2.0, 1.3, 1.1])
    cooktop = trimesh.util.concatenate([cooktop, tray])
    objects = {"stove.n.01_1": SimpleNamespace(fixed_base=True)}
    sim = _region_sim(trimesh.util.concatenate([cooktop, panel]), objects=objects)
    objects["frying_pan.n.01_1"] = _seen("frying_pan.n.01_1", [0.0, 0.0, 0.0], [-0.14, -0.14, 0.0], [0.14, 0.14, 0.05], sim)  # fmt: skip
    link = (np.array([2.0, 1.15, 0.905]), 0.2)
    region = sim.heat_region("frying_pan.n.01_1", "stove.n.01_1", link)
    assert region["pose"][2] + 0.01 == pytest.approx(0.90) and region["pose"][:2] == pytest.approx([2.0, 1.15])
    centres = np.array(region["dims"][:2]) / 2 - 0.14  # how far the pan's centre may go from the link
    assert np.hypot(*centres) <= 0.2 and centres.min() > 0.13  # every corner inside the disk, most of it used
    assert sim.heat_region("frying_pan.n.01_1", "stove.n.01_1", None) is None  # an oven heats inside: no near region


def test_the_push_face_of_a_book_lying_under_a_shelf_board_points_into_the_compartment_with_the_travel_to_its_edge():
    """N-push (spec S05): a flat book on a roofed board of the map's bookcase is pushed on the face turned away from
    the robot, at mid-height, until its near edge hangs PUSH_OVERHANG past the board's front edge, where a pinch can
    close on it. A book already hanging over, one standing on edge, one on the open top board or on a movable shelf
    gets no push; from the far side of the case the other face is pushed."""
    from omnigibson.tiptop.r1pro import PUSH_OVERHANG, PUSH_RADIUS

    case = _shelves([0.44, 0.72, 1.0])  # boards x 0.7..1.1, the open front toward the robot at x 0.7
    bookcase = SimpleNamespace(fixed_base=True, category="bookcase")
    objects = {}
    sim = _region_sim(case, objects=objects)
    sim.scene_aabbs = lambda: [(bookcase, np.array([0.69, -0.41, 0.0]), np.array([1.11, 0.41, 1.0]))]
    objects["comic_book.n.01_3"] = _seen("comic_book.n.01_3", [0.90, 0.1, 0.735], [-0.14, -0.1, -0.015], [0.14, 0.1, 0.015], sim)  # fmt: skip
    position, normal, radius, depth = sim.push_face("comic_book.n.01_3")
    assert normal == pytest.approx([1.0, 0.0, 0.0]) and radius == PUSH_RADIUS
    assert position == pytest.approx([1.04, 0.1, 0.735])  # its far face, mid-height
    assert depth == pytest.approx(0.76 - 0.7 + PUSH_OVERHANG, abs=2e-3)  # near edge 0.76 -> 0.67: 3 cm past the front at 0.7
    far = _region_sim(case, base=(2.0, 0.0), objects=objects)  # standing behind the case: the other face, the other way
    far.scene_aabbs, far.objects, far.seen_boxes = sim.scene_aabbs, sim.objects, sim.seen_boxes
    position, normal, _, depth = far.push_face("comic_book.n.01_3")
    assert normal == pytest.approx([-1.0, 0.0, 0.0]) and position[0] == pytest.approx(0.76)
    assert depth == pytest.approx(1.08 - 1.04 + PUSH_OVERHANG, abs=2e-3)  # the board ends at the back panel's face, not its own end
    sim.seen_boxes["comic_book_1"] = (np.array([-0.25, -0.1, -0.015]), np.array([0.03, 0.1, 0.015]))  # hangs 5 cm out
    assert sim.push_face("comic_book.n.01_3") is None
    sim.seen_boxes["comic_book_1"] = (np.array([-0.14, -0.1, -0.015]), np.array([0.14, 0.1, 0.015]))
    on_edge = (th.tensor([0.90, 0.1, 0.82]), th.tensor([np.sin(np.pi / 4), 0.0, 0.0, np.cos(np.pi / 4)]))  # rolled 90 deg
    objects["comic_book.n.01_3"].get_position_orientation = lambda: on_edge
    assert sim.push_face("comic_book.n.01_3") is None
    objects["comic_book.n.01_3"].get_position_orientation = lambda: (th.tensor([0.90, 0.1, 1.015]), th.tensor([0.0, 0.0, 0.0, 1.0]))
    assert sim.push_face("comic_book.n.01_3") is None  # the top board: a pinch from above takes it
    objects["comic_book.n.01_3"].get_position_orientation = lambda: (th.tensor([0.90, 0.1, 0.735]), th.tensor([0.0, 0.0, 0.0, 1.0]))
    bookcase.fixed_base = False
    assert sim.push_face("comic_book.n.01_3") is None  # a movable case's boards are not the map's to read


def test_a_flat_book_under_a_shelf_board_is_pushed_out_before_its_pick_round_and_a_pour_tilts_after_its_round():
    """N-push: Episode.pick slides the item out first when push_face says so, once (it hangs over the edge after);
    N-rotate: Episode.pour is the keep-hold round over the target, then the wrist tilt."""
    from omnigibson.tiptop.bench import Episode
    from b1k.bridge.strategies import atom

    rounds, tilts = [], []
    ep, _ = _pick_ep(attached_first=None, target=SimpleNamespace(name="comic_book_3"), grasping_mode="assisted")
    faces = iter([("face", "normal", 0.01, 0.09), None])
    ep.sim.push_face = lambda bddl: next(faces)
    ep.sim.side_entry = lambda item, container: False
    ep.rounds = 2
    ep.plan_and_execute = lambda atoms, floor=False: rounds.append(atoms)
    ep.achieve = lambda atoms: rounds.append(atoms) or True
    assert ep.pick("comic_book.n.01_3") is False
    hold = atom("holding", "comic_book.n.01_3")
    assert rounds == [[atom("push", "comic_book.n.01_3")], [hold], [hold]]
    ep.sim.tilt_wrist = lambda arm: tilts.append(arm) or True
    ep.sim.video_caption = None
    ep.pour = MethodType(Episode.pour, ep)
    assert ep.pour("grated_cheese.n.01_1", "pizza_dough.n.01_1") is True
    assert rounds[-1] == [atom("pour", "grated_cheese.n.01_1", "pizza_dough.n.01_1")] and tilts == ["left"]
    ep.achieve = lambda atoms: False
    assert ep.pour("grated_cheese.n.01_1", "pizza_dough.n.01_1") is False and tilts == ["left"]


def test_a_rose_longer_than_the_vase_is_wide_is_stood_on_end_yaw_free_and_a_nail_target_hangs_the_alarm_vertical():
    """E-6dof (spec S21): upright_rotation turns the seen box's longest axis vertical the shorter way round;
    inside_region carries it, yaw free, only for an item longer than the vessel's narrower side (roses by flat drop
    2/5). attach_target carries roll and pitch too: the fire alarm's male frame onto the wall nail's, turned 90 deg,
    its bottom set where the disc hangs."""
    from omnigibson.tiptop.r1pro import ATTACH_LIFT, ATTACH_TOL
    import omnigibson.utils.transform_utils as T

    objects = {}
    sim = _region_sim(None, objects=objects)
    yaw30 = th.tensor([0.0, 0.0, np.sin(np.pi / 12), np.cos(np.pi / 12)])
    rose = SimpleNamespace(fixed_base=False, get_position_orientation=lambda: (th.tensor([1.0, 0.5, 0.75]), yaw30))
    objects["rose.n.01_1"] = sim.objects["rose_1"] = rose
    sim.seen_boxes["rose_1"] = (np.array([-0.15, -0.03, -0.02]), np.array([0.15, 0.03, 0.02]))
    R = np.array(sim.upright_rotation("rose.n.01_1"))
    assert R @ T.quat2mat(yaw30).numpy() @ [1.0, 0.0, 0.0] == pytest.approx([0.0, 0.0, 1.0], abs=1e-9)
    assert np.allclose(R @ R.T, np.eye(3)) and np.linalg.det(R) == pytest.approx(1.0)
    sim.inside_rect = lambda item, container: (np.array([2.0, 1.0]), np.array([0.06, 0.06]), 0.554, 0.723)  # a 12 cm vase
    region = sim.inside_region("rose.n.01_1", "vase.n.01_1")
    assert region["rotation"] == R.tolist() and region["yaw_tolerance"] is None
    assert region["pose"][:2] == pytest.approx([2.0, 1.0]) and region["dims"][:2] == pytest.approx([0.12, 0.12])
    sim.seen_boxes["rose_1"] = (np.array([-0.05, -0.03, -0.02]), np.array([0.05, 0.03, 0.02]))  # a short item lies
    assert "rotation" not in sim.inside_region("rose.n.01_1", "vase.n.01_1")
    alarm = SimpleNamespace(fixed_base=False, get_position_orientation=lambda: (th.tensor([0.6, 0.0, 0.019]), th.tensor([0.0, 0.0, 0.0, 1.0])))  # fmt: skip
    objects["fire_alarm.n.02_1"] = sim.objects["fire_alarm_1"] = alarm
    sim.seen_boxes["fire_alarm_1"] = (np.array([-0.05, -0.05, -0.019]), np.array([0.05, 0.05, 0.019]))  # a disc, flat
    male = np.eye(4)
    male[:3, 3] = [0.6, 0.0, 0.0]  # on the disc's back
    female = trimesh.transformations.rotation_matrix(np.pi / 2, [0.0, 1.0, 0.0])  # the nail: z turned onto x
    female[:3, 3] = [1.0, 0.3, 1.71]
    region = sim.attach_target("fire_alarm.n.02_1", "wall_nail.n.01_1", (male, female))
    assert np.allclose(region["rotation"], female[:3, :3])
    assert region["pose"][2] + 0.01 == pytest.approx(1.71 - 0.05 + ATTACH_LIFT)  # hanging vertical: 5 cm below the nail
    assert region["dims"][:2] == pytest.approx([2 * (0.019 + ATTACH_TOL / np.sqrt(2)), 2 * (0.05 + ATTACH_TOL / np.sqrt(2))])


def test_fixture_for_tracks_the_nearest_doorless_heat_source_so_a_goal_atom_can_name_it(monkeypatch):
    """W-ipress / E-heat: the burner is outside the BDDL scope in the stove-route tasks (spec 6.1.6). The nearest
    fixture with the taxonomy's ability is tracked under its scene name, one with no door first (a microwave or oven
    door gates its heat), and toggled_on(<it>) then translates like any task object's."""
    from omnigibson.tiptop import articulation
    from omnigibson.tiptop.bench import Episode

    def fixture(name, category, at, door=False):
        centre = th.tensor([*at, 0.9])
        return SimpleNamespace(name=name, category=category, fixed_base=True, aabb_center=centre, door=door)

    microwave = fixture("microwave_hjjxmi_0", "microwave", (1.0, 0.0), door=True)
    stove = fixture("stove_ykretu_0", "stove", (3.0, 0.0))
    fridge = fixture("fridge_dszchb_0", "fridge", (0.5, 0.0), door=True)
    table = SimpleNamespace(name="table_7", category="breakfast_table", fixed_base=False)  # movable: never a fixture
    monkeypatch.setattr(articulation, "openable_joints", lambda obj: [{"name": "j_door"}] if obj.door else [])
    objects = {o.name: o for o in (microwave, stove, fridge)}
    sim = SimpleNamespace(
        env=SimpleNamespace(scene=SimpleNamespace(objects=[table, microwave, stove, fridge])),
        base_pose=lambda: (th.zeros(3), th.tensor([0.0, 0.0, 0.0, 1.0])),
        objects={}, bddl_names={"frying_pan_1": "frying_pan.n.01_1"}, scene_object=lambda name: objects[name],
    )  # fmt: skip
    sim.track, sim.tiptop_goal = MethodType(R1ProSim.track, sim), MethodType(R1ProSim.tiptop_goal, sim)
    ep = SimpleNamespace(sim=sim, boxes=lambda *names: {})
    ep.fixture_for = MethodType(Episode.fixture_for, ep)
    assert ep.fixture_for("heatSource", near="frying_pan.n.01_1") == "stove_ykretu_0"  # 3 m off beats 1 m: no door
    assert ep.fixture_for("coldSource") == "fridge_dszchb_0" and ep.fixture_for("particleRemover") is None
    labels, atoms = sim.tiptop_goal([{"predicate": "toggled_on", "args": ["stove_ykretu_0"]}], category_level=False)
    assert atoms == [{"predicate": "pressed", "args": ["stove_ykretu_0_button"]}] and "stove_ykretu_0" in labels
    assert sim.objects["stove_ykretu_0"] is stove and sim.bddl_names["frying_pan_1"] == "frying_pan.n.01_1"


def _aabb(lo, hi, fixed_base=True):
    return SimpleNamespace(aabb=(th.tensor(lo, dtype=th.float32), th.tensor(hi, dtype=th.float32)),
                           fixed_base=fixed_base, get_position_orientation=lambda: (th.zeros(3), th.tensor([0.0, 0.0, 0.0, 1.0])))  # fmt: skip


def test_a_case_of_shelves_offers_one_reachable_compartment_not_its_bottom_board():
    """A bookcase has ONE fillable volume spanning every compartment, and inside_region aimed at its lowest z: 32 of
    32 bookcase regions went to the bottom shelf (IK 0/256) and the fallback put 16 of 18 items on the lid. The
    boards come from the physical mesh; the compartment is the one nearest a comfortable place height with
    headroom for the item under the next board, and within reach of the base."""
    from omnigibson.tiptop.r1pro import boards

    case = _shelves([0.12, 0.44, 0.73, 1.10, 1.50])
    found = boards(case.vertices, case.faces)
    assert [round(z, 2) for z, *_ in found] == [0.12, 0.44, 0.73, 1.10, 1.50]
    assert [round(c, 2) for *_, c in found[:-1]] == [0.42, 0.71, 1.08, 1.48] and found[-1][3] == float("inf")
    fillable = (np.array([0.7, -0.4, 0.126]), np.array([1.1, 0.4, 1.42]))
    sim = _region_sim(case)
    z, lo, hi, ceiling = sim.shelf_of(None, 0.04, fillable)
    assert (z, ceiling) == pytest.approx((0.73, 1.08)) and lo.tolist() == [0.7, -0.4] and hi.tolist() == [1.1, 0.4]
    assert sim.shelf_of(None, 0.31, fillable)[0] == pytest.approx(0.73)  # the only compartment 0.34 m clear
    assert sim.shelf_of(None, 0.5, fillable) is None  # nothing has room for it
    assert sim.shelf_of(None, 0.04, (fillable[0], np.array([1.1, 0.4, 0.6])))[0] == pytest.approx(0.44)
    # 2.2 m from the near edge nothing is in reach: still the best board, the stance search moves the robot to it
    assert _region_sim(case, base=(-1.5, 0.0)).shelf_of(None, 0.04, fillable)[0] == pytest.approx(0.73)


def test_a_board_above_the_arms_reach_is_never_chosen_even_when_it_is_the_only_one_in_reach():
    """putting_shoes_on_rack i0 2026-09-24: standing 1.06 m from the hallstand's bench (0.55 m), the only board within
    BOARD_REACH was its 2.37 m top, and 6 rounds IK-failed on it. The bench is sent from there too."""
    bench = trimesh.creation.box([0.55, 1.5, 0.02])
    bench.apply_translation([1.335, 0.0, 0.54])  # its near edge 1.06 m ahead
    top = trimesh.creation.box([0.58, 1.7, 0.02])
    top.apply_translation([0.99, 0.0, 2.36])  # 0.7 m ahead, in reach
    sim = _region_sim(trimesh.util.concatenate([bench, top]))
    assert sim.shelf_of(None, 0.12)[0] == pytest.approx(0.55)
    assert _region_sim(top).shelf_of(None, 0.12) is None  # the top alone offers nothing


def test_a_touching_goal_gets_a_reachable_board_of_the_fixture_not_its_hull_top():
    """touching(shoe, hallstand) was planned on the hull's top: the stepped stand's highest visible shelf, IK 0/256
    (4 of 5 rounds); the one shoe that landed rests on its bench at 0.55 m. The board nearest 0.9 m is a shelf
    that starts 1.02 m ahead, out of reach from here, so the bench is sent; from closer, that shelf."""
    bench = trimesh.creation.box([0.67, 1.5, 0.02])
    bench.apply_translation([1.065, 0.0, 0.54])
    shelf = trimesh.creation.box([0.36, 1.2, 0.02])
    shelf.apply_translation([1.20, 0.0, 0.96])
    stand = trimesh.util.concatenate([bench, shelf])
    objects = {
        "gym_shoe.n.01_2": _aabb([0.0, 0.0, 0.0], [0.12, 0.27, 0.12]),
        "hallstand.n.01_1": SimpleNamespace(fixed_base=True),
        "shoe_rack.n.01_1": SimpleNamespace(fixed_base=False),  # movable: not the scanned map, its mesh is not read
    }
    region = _region_sim(stand, objects=objects).rest_region("gym_shoe.n.01_2", "hallstand.n.01_1")
    assert region["pose"][2] + region["dims"][2] / 2 == pytest.approx(0.55)  # the bench's top, base frame
    assert region["dims"][:2] == pytest.approx([0.67, 1.5])
    closer = _region_sim(stand, base=(0.3, 0.0), objects=objects).rest_region("gym_shoe.n.01_2", "hallstand.n.01_1")
    assert closer["pose"][0] + 0.3 == pytest.approx(1.20) and closer["pose"][2] + 0.01 == pytest.approx(0.97)
    assert _region_sim(stand, base=(0.3, 0.0), objects=objects).rest_region("gym_shoe.n.01_2", "hallstand.n.01_1")
    assert _region_sim(stand, objects=objects).rest_region("gym_shoe.n.01_2", "shoe_rack.n.01_1") is None
    seen = _region_sim(stand, objects=objects)  # the item's height comes from the capture's points once it has them
    seen.objects, seen.seen_boxes = {"shoe_2": objects["gym_shoe.n.01_2"]}, {"shoe_2": (np.zeros(3), np.array([0.1, 0.2, 0.5]))}
    assert seen.item_height("gym_shoe.n.01_2") == pytest.approx(0.5)
    assert _region_sim(stand, objects=objects).item_height("gym_shoe.n.01_2") == 0.0  # unseen: never its AABB


def test_a_named_table_and_a_fixtures_footprint_become_the_planners_support_plane():
    """ontop(x, table_1) went out as on(x, "table") with no surface, and the plane RANSAC fitted was the floor in 4
    of 5 executed rounds. under(x, sink) is the floor inside the sink's footprint."""
    objects = {
        "table.n.02_1": _aabb([24.41, 22.11, 0.0], [24.89, 22.89, 0.347]),
        "sink.n.01_1": _aabb([2.5, 2.5, 0.35], [3.0, 3.7, 1.07]),
    }
    sim = _region_sim(None, base=(24.0, 22.0), objects=objects)
    top = sim.footprint_region("table.n.02_1")
    assert top["pose"][:2] == pytest.approx([0.65, 0.5]) and top["pose"][2] + 0.01 == pytest.approx(0.347)
    assert top["dims"] == pytest.approx([0.48, 0.78, 0.02], abs=1e-4)
    slab = sim.floor_surface(within="sink.n.01_1")
    assert slab["pose"][:3] == pytest.approx([-21.25, -18.9, -0.01], abs=1e-4)
    assert slab["dims"] == pytest.approx([0.5, 1.2, 0.02], abs=1e-4)
    assert sim.floor_surface()["pose"][2] == pytest.approx(-0.01)  # the plain slab is unchanged
    # the dispatch: which surface rides under the planner's support label, and a fixture's board under its own
    sim = SimpleNamespace(
        floor_surface=lambda within=None: {"floor": within},
        footprint_region=lambda table: None if table == "table.n.02_4" else {"top": table},  # _4: never seen
        rest_region=lambda item, fixture: {"board": fixture}, inside_region=lambda item, container: None,
        label_of=lambda bddl: bddl.split(".")[0] + "_1", region_refused=set(),
        scene_object=lambda name: SimpleNamespace(fixed_base=name != "table.n.02_3"),  # _3 is a movable table
    )
    regions = lambda *atoms: R1ProSim.inside_regions(sim, [{"predicate": p, "args": list(a)} for p, *a in atoms])
    assert regions(("under", "mousetrap.n.01_4", "sink.n.01_1")) == {"table": {"floor": "sink.n.01_1"}}
    assert regions(("ontop", "book.n.02_1", "table.n.02_1")) == {"table": {"top": "table.n.02_1"}}
    assert regions(("ontop", "book.n.02_1", "table.n.02_1"), ("ontop", "book.n.02_2", "table.n.02_2")) == {}
    assert regions(("ontop", "book.n.02_1", "table.n.02_3")) == {"table": {"top": "table.n.02_3"}}  # from its own box
    assert regions(("ontop", "book.n.02_1", "table.n.02_4")) == {}
    assert regions(("ontop", "x.n.01_1", "floor.n.01_1"), ("under", "y.n.01_1", "sink.n.01_1")) == {"table": {"floor": None}}
    assert regions(("ontop", "easter_egg.n.01_1", "lawn.n.01_1")) == {"table": {"floor": None}}  # a lawn is ground
    assert regions(("touching", "gym_shoe.n.01_2", "hallstand.n.01_1")) == {"hallstand_1": {"board": "hallstand.n.01_1"}}


def test_a_lawn_is_the_floor_under_the_robot():
    """hiding_Easter_eggs wants the eggs ontop lawn.n.01_1 and has no floor at all: the lawn's top is the slab."""
    from omnigibson.tiptop.bench import Episode

    sim = _region_sim(None, base=(3.0, 4.0))
    lawn = SimpleNamespace(category="lawn")
    sim.scene_aabbs = lambda: [(lawn, np.array([-9.0, -9.0, -0.1]), np.array([9.0, 9.0, 0.05]))]
    assert sim.floor_surface()["pose"][2] + 0.01 == pytest.approx(0.05)
    assert Episode.is_floor(None, "lawn.n.01_1") and Episode.is_floor(None, "floor.n.01_2")
    assert not Episode.is_floor(None, "table.n.02_1")
    sim.env = SimpleNamespace(task=SimpleNamespace(object_scope={"easter_egg.n.01_1": None, "lawn.n.01_1": None}))
    assert R1ProSim.floor_name(sim) == "lawn.n.01_1"  # a scope with no floor at all raised in Episode.__init__
    # chopping_wood, clean_your_rusty_garden_tools and stacking_wood stand the agent on a driveway, with no floor
    sim.env.task.object_scope = {"log.n.01_1": None, "driveway.n.01_1": None, "agent.n.01_1": None}
    assert R1ProSim.floor_name(sim) == "driveway.n.01_1" and Episode.is_floor(None, "driveway.n.01_1")


def test_a_movable_tables_slab_is_the_world_box_of_the_points_that_saw_it():
    """putting_up_Christmas_decorations_inside names a table that is not fixed: its AABB is not the map's to read,
    so its top is the world box of its own seen box; nothing until a capture has seen it."""
    yaw90 = th.tensor([0.0, 0.0, np.sin(np.pi / 4), np.cos(np.pi / 4)])
    table = SimpleNamespace(fixed_base=False, get_position_orientation=lambda: (th.tensor([2.0, 1.0, 0.0]), yaw90))
    sim = _region_sim(None, objects={"table.n.02_3": table})
    assert sim.footprint_region("table.n.02_3") is None
    sim.objects = {"table_3": table}
    sim.seen_boxes = {"table_3": (np.array([-0.4, -0.25, 0.0]), np.array([0.4, 0.25, 0.75]))}
    slab = sim.footprint_region("table.n.02_3")
    assert slab["pose"][:3] == pytest.approx([2.0, 1.0, 0.74], abs=1e-6)
    assert slab["dims"] == pytest.approx([0.5, 0.8, 0.02])


def _button_world(mesh, marker, extent, pos=(0.0, 0.0, 0.0), quat=(0.0, 0.0, 0.0, 1.0)):
    """button_world over a fake toggle object: ``mesh`` and ``marker`` in the object's frame, posed at (pos, quat)."""
    from omnigibson.object_states import ToggledOn
    import omnigibson.utils.transform_utils as T

    pos, quat = th.tensor(pos), th.tensor(quat)
    rot = T.quat2mat(quat).numpy().astype(np.float64)
    world = mesh.copy()
    world.apply_transform(T.pose2mat((pos, quat)).numpy())
    marker_w = th.tensor(pos.numpy() + rot @ np.asarray(marker, dtype=np.float64), dtype=th.float32)
    identity = th.tensor([0.0, 0.0, 0.0, 1.0])
    link = SimpleNamespace(get_position_orientation=lambda: (marker_w, identity), scale=th.ones(3))
    state = SimpleNamespace(link=link, visual_marker=SimpleNamespace(extent=th.full((3,), extent)), scale=th.ones(3))
    obj = SimpleNamespace(states={ToggledOn: state}, get_position_orientation=lambda: (pos, quat))
    sim = SimpleNamespace(scene_object=lambda bddl: obj, collision_mesh_world=lambda o: world)
    return R1ProSim.button_world(sim, "x.n.01_1")


def test_a_button_sits_on_the_body_face_the_press_is_braced_on_and_its_position_is_on_that_surface():
    """The visual mesh carries the toggle marker: a 46 mm sphere at the switch's face pulled the box face to the
    wall side of every wall switch and the lighter's button to its underside (marker_in_box.out). Off the
    physical mesh, the switch's face is +x; the lighter's button is 8.4 mm from its end and 8.9 mm from its top,
    and of the faces within the marker's radius the top is the one a press is braced on. The washer's marker
    centre floats 11.2 mm proud of the body (washer_gap.out): the position is projected onto the face."""
    switch = trimesh.creation.box([0.02, 0.08, 0.12])
    pos, normal, radius = _button_world(switch, [0.0098, 0.0, -0.008], 0.046, pos=(2.0, 3.0, 1.2),
                                        quat=(0.0, 0.0, np.sin(np.pi / 4), np.cos(np.pi / 4)))  # on a wall, facing +y
    assert normal == pytest.approx([0.0, 1.0, 0.0], abs=1e-6) and radius == pytest.approx(0.046)
    assert pos == pytest.approx([2.0, 3.01, 1.192], abs=1e-6)  # projected onto the face, 0.2 mm out
    lighter = trimesh.creation.box([0.06, 0.07, 0.018])
    pos, normal, _ = _button_world(lighter, [0.018, 0.0266, 0.0], 0.0209)  # lying flat: the top face
    assert normal == pytest.approx([0.0, 0.0, 1.0], abs=1e-6) and pos == pytest.approx([0.018, 0.0266, 0.009], abs=1e-6)
    stood = (np.sin(np.pi / 4), 0.0, 0.0, np.cos(np.pi / 4))  # stood on its end: its +y face is now up
    pos, normal, _ = _button_world(lighter, [0.018, 0.0266, 0.0], 0.0209, quat=stood)
    assert normal == pytest.approx([0.0, 0.0, 1.0], abs=1e-6) and pos == pytest.approx([0.018, 0.0, 0.035], abs=1e-6)
    washer = trimesh.creation.box([0.6, 0.6, 0.85])
    pos, normal, _ = _button_world(washer, [0.3112, 0.0, 0.33], 0.046)
    assert normal == pytest.approx([1.0, 0.0, 0.0], abs=1e-6) and pos == pytest.approx([0.3, 0.0, 0.33], abs=1e-6)


def test_the_widened_retry_is_told_what_the_first_search_refused():
    """bringing_in_wood 301 (2026-09-23): the 1.1 m retry re-tested the same 8 kerb-refused candidates as the 0.9 m
    search, whose refusals lived in place_robot_for's local list; 152 of 377 failed searches repeated themselves."""
    from omnigibson.tiptop.bench import REACH_FAR, Episode

    calls = []

    def place_robot_for(*names, reach=0.9, avoid=(), refused=None):
        calls.append((reach, list(avoid), list(refused)))
        if reach < REACH_FAR:
            refused.append((-6.23, -3.12, np.pi / 2, "paver_sekqqq_0"))
            raise RuntimeError("no collision-free base destination")
        return {"x": -7.03, "y": -2.32, "yaw": 0.0}

    sim = SimpleNamespace(place_robot_for=place_robot_for, hold=lambda n, g: None, last_gripper=None, n_steps=0,
                          settled_level=lambda x, y: (True, ""), robot=None)
    ep = SimpleNamespace(sim=sim, stood={}, records=[], args=SimpleNamespace(settle_steps=1), last_level=None,
                         fallen=lambda: False, opened_at={})
    ep.stand_for = MethodType(Episode.stand_for, ep)
    assert ep.stand_for("plywood.n.01_1")["x"] == -7.03
    assert calls == [(0.9, [], []), (REACH_FAR, [], [(-6.23, -3.12, np.pi / 2, "paver_sekqqq_0")])]
    ep.stand_for("plywood.n.01_1")  # the next search avoids the stance taken and starts with nothing refused
    assert calls[2] == (0.9, [(-7.03, -2.32)], [])


def test_side_entry_is_a_roof_within_the_hand_stack_over_the_item_and_an_open_top_or_a_movable_has_none():
    """spec S13: the wrist tops out 20.9 cm above the fingertips (enclosed/hand_stack.out), so a board 18 cm above a
    10 cm item stops a top-down hand (35 of 40 bookcase place rounds had the hand straight down). A movable
    container's mesh is not ours to read: it is never asked."""
    case = _shelves([0.44, 0.72])  # the next board's underside 0.70: 16 cm over a 10 cm item on the 0.44 board
    fixed = SimpleNamespace(fixed_base=True)
    sim = SimpleNamespace(scene_object=lambda name: fixed, collision_mesh_world=lambda obj: case, item_height=lambda name: 0.10,
                          inside_rect=lambda item, container: (np.array([0.9, 0.0]), np.array([0.2, 0.4]), 0.44, 0.70))  # fmt: skip
    sim.side_entry = MethodType(R1ProSim.side_entry, sim)
    assert sim.side_entry("book_1", "bookcase_1") is True
    sim.item_height = lambda name: 0.01  # 25 cm clear over a flat sheet: the hand and its margin fit
    assert sim.side_entry("book_1", "bookcase_1") is False
    sim.item_height = lambda name: 0.04  # 22 cm: the stack fits by a centimetre, which no planner margin allows
    assert sim.side_entry("book_1", "bookcase_1") is True
    walls = [trimesh.creation.box([0.02, 0.8, 0.3]).apply_translation([x, 0.0, 0.59]) for x in (0.69, 1.11)]
    walls += [trimesh.creation.box([0.44, 0.02, 0.3]).apply_translation([0.9, y, 0.59]) for y in (-0.41, 0.41)]
    sim.collision_mesh_world = lambda obj: trimesh.util.concatenate(walls + [_shelves([0.44])])  # an open-top bin
    sim.item_height = lambda name: 0.10
    assert sim.side_entry("book_1", "bin_1") is False
    sim.scene_object = lambda name: SimpleNamespace(fixed_base=False)
    sim.collision_mesh_world = lambda obj: pytest.fail("a movable's mesh must not be read")
    assert sim.side_entry("book_1", "box_1") is False


def test_a_two_column_fillable_gives_a_rectangle_in_the_bay_nearest_the_opened_door():
    """fridge petcxr's one fillable volume is two columns with the AABB's centre between them, so every centred
    rectangle was refused and the fridge got no region (spec 6.2). Its columns start 26 cm apart (side_entry_check.out),
    so the bay's floor is its own, probed down from the bay, not the volume's bottom: that is the other column's."""
    def accepts(points):
        x, z = points.numpy()[:, 0], points.numpy()[:, 2]
        return th.tensor(((np.abs(x - 0.2) < 0.2) & (z >= 0.0)) | ((np.abs(x - 1.0) < 0.2) & (z >= 0.26)))

    link = SimpleNamespace(check_points_in_volume=accepts)
    lo, hi = np.array([0.0, -0.3, 0.0]), np.array([1.2, 0.3, 0.5])
    centre, half, floor = R1ProSim.bay(None, link, lo, hi, near=[1.1, 0.0])
    assert centre[0] == pytest.approx(1.0, abs=0.05) and half[0] == pytest.approx(0.2, abs=0.05)
    assert centre[1] == pytest.approx(0.0, abs=1e-6) and half[1] == pytest.approx(0.3, abs=0.05)
    assert floor == pytest.approx(0.26, abs=0.011), "the right column's own floor, not the volume's bottom"
    centre, half, floor = R1ProSim.bay(None, link, lo, hi, near=[0.1, 0.0])
    assert centre[0] == pytest.approx(0.2, abs=0.05) and floor == pytest.approx(0.0, abs=0.011)
    assert R1ProSim.bay(None, SimpleNamespace(check_points_in_volume=lambda p: th.zeros(len(p), dtype=th.bool)),
                        lo, hi)[0][0] == pytest.approx(0.6)  # nothing accepted: the AABB's own, the caller shrinks it


def test_the_stance_a_container_was_opened_from_is_stood_at_again_before_any_search():
    from omnigibson.tiptop.bench import Episode

    calls = []
    sim = SimpleNamespace(
        place_robot=lambda x, y, yaw, note="": calls.append(("place", x, y, yaw)),
        place_robot_for=lambda *names, **kw: calls.append(("search", names)) or {"x": 9.0, "y": 9.0, "yaw": 0.0},
        look_at=lambda *names: calls.append(("look", names)),
        hold=lambda n, g: None, last_gripper=None, n_steps=0, settled_level=lambda x, y: (True, ""), robot=None,
        video_caption="", arm="left",
    )  # fmt: skip
    ep = SimpleNamespace(sim=sim, stood={}, records=[], args=SimpleNamespace(settle_steps=1), last_level=None,
                         fallen=lambda: False, opened_at={"fridge.n.01_1": (1.0, 2.0, 0.5)})  # fmt: skip
    ep.stand_for = MethodType(Episode.stand_for, ep)
    assert ep.stand_for("fridge.n.01_1") == {"x": 1.0, "y": 2.0, "yaw": 0.5}
    # place_robot clears the look target: set again here, or the captures aim at the holding hand, not the bay
    assert calls == [("place", 1.0, 2.0, 0.5), ("look", ("fridge.n.01_1",))]
    ep.stand_for("fridge.n.01_1")  # a second call for the same object stands somewhere else, as ever
    assert calls[-1] == ("search", ("fridge.n.01_1",))
    ep.stand_for("fridge.n.01_1", "jar.n.01_1")  # a pair is the pair's own search
    assert calls[-1] == ("search", ("fridge.n.01_1", "jar.n.01_1"))
    # the stance is recorded by open_up itself, from what open_container reports; a failed open records nothing
    ep.opened_at, ep.spec = {}, None
    sim.open_container = lambda arm, name, **kw: {"opened": True, "stance": (3.0, 4.0, 0.25)}
    ep.open_up = MethodType(Episode.open_up, ep)
    assert ep.open_up("fridge.n.01_1") is True and ep.opened_at == {"fridge.n.01_1": (3.0, 4.0, 0.25)}
    assert ep.stand_for("fridge.n.01_1") == {"x": 3.0, "y": 4.0, "yaw": 0.25} and calls[-2] == ("place", 3.0, 4.0, 0.25)
    sim.open_container = lambda arm, name, **kw: {"opened": False, "why": "no grasp"}
    assert ep.open_up("fridge.n.01_1") is False and ep.opened_at == {"fridge.n.01_1": (3.0, 4.0, 0.25)}


def test_closing_pushes_every_open_joint_to_its_closed_end_and_a_roofed_pick_is_marked_for_the_side_grasp(monkeypatch):
    from omnigibson.tiptop import articulation
    from omnigibson.tiptop.bench import Episode

    joints = [dict(name="j_door", lower=0.0, upper=1.6, position=1.2, closed=0.0),
              dict(name="j_lid", lower=-1.0, upper=0.0, position=-0.02, closed=0.0),  # hung the other way: shut
              dict(name="j_drawer", lower=0.0, upper=0.4, position=0.3, closed=0.0)]  # fmt: skip
    monkeypatch.setattr(articulation, "openable_joints", lambda obj: joints)
    pushes = []
    sim = SimpleNamespace(arm="left", n_steps=0, scene_object=lambda name: "fridge",
                          push_joint=lambda arm, name, j, target: pushes.append((j["name"], target)) or {"reached": True, "why": ""})  # fmt: skip
    ep = SimpleNamespace(sim=sim, records=[], spec=None, is_shut=lambda name: True)
    ep.open_up = MethodType(Episode.open_up, ep)
    assert ep.open_up("fridge.n.01_1", fraction=0.0) is True
    assert pushes == [("j_door", 0.0), ("j_drawer", 0.0)] and [r["close"] for r in ep.records] == ["fridge.n.01_1"] * 2

    marks = []
    ep, _ = _pick_ep(attached_first=None, target=SimpleNamespace(name="book_1"), grasping_mode="assisted")
    ep.sim.side_entry = lambda item, container: container == "bookcase.n.01_1"
    ep.support_of, ep.is_floor = lambda b: "bookcase.n.01_1", lambda name: False
    ep.plan_and_execute = lambda atoms, floor=False: marks.append(set(ep.sim.side_grasp))
    ep.pick("book.n.02_1", into="box.n.01_1")  # out of a shelf
    assert marks == [{"book.n.02_1"}] and ep.sim.side_grasp == set(), "marked for that round only"
    ep.pick("book.n.02_1", into="bookcase.n.01_1")  # into one
    ep.support_of = lambda b: "table.n.02_1"
    ep.pick("book.n.02_1")
    assert marks[1:] == [{"book.n.02_1"}, set()]


def test_push_face_leaves_what_a_top_down_pinch_takes_and_gives_up_without_an_edge():
    """Review of tier 4 (push-face-false-positives): sweep4 replayed, 9 of 16 pushes were on the round's own pick
    target -- a 4.9 cm die on a bed (a 0.44 m stroke that would tip it off the edge), a stapler, a banana, ice cubes
    and a tray 2.26 m toward the corner of a bar's box when the edge ray missed, a board game 1.2 m down a bed. A
    push is for a flat item wider than the jaw both ways, in a compartment (the books sit 25-28 cm under the next
    board, more than the hand stack: a roof test would refuse the very case), at most PUSH_MAX_DEPTH from an edge
    found on the board itself."""
    from omnigibson.tiptop.r1pro import PUSH_MAX_DEPTH

    bookcase = SimpleNamespace(fixed_base=True, category="bookcase")
    objects = {}
    sim = _region_sim(_shelves([0.44, 0.72, 1.0]), objects=objects)
    sim.scene_aabbs = lambda: [(bookcase, np.array([0.69, -0.41, 0.0]), np.array([1.11, 0.41, 1.0]))]
    die = _seen("die.n.01_1", [0.90, 0.1, 0.744], [-0.0245, -0.0245, -0.024], [0.0245, 0.0245, 0.024], sim)
    objects["die.n.01_1"] = die
    assert sim.push_face("die.n.01_1") is None, "a die is flat by its box and in a compartment, and a pinch takes it"
    sim.seen_boxes["die_1"] = (np.array([-0.14, -0.1, -0.015]), np.array([0.14, 0.1, 0.015]))  # a book's box
    assert sim.push_face("die.n.01_1") is not None
    deep = _region_sim(_shelves([0.44, 0.72, 1.0], x=(0.7, 2.0)), objects=objects)  # a bed: 1.3 m deep
    deep.scene_aabbs = lambda: [(bookcase, np.array([0.69, -0.41, 0.0]), np.array([2.01, 0.41, 1.0]))]
    deep.objects, deep.seen_boxes = sim.objects, sim.seen_boxes
    die.get_position_orientation = lambda: (th.tensor([0.7 + PUSH_MAX_DEPTH + 0.3, 0.1, 0.735]), th.tensor([0.0, 0.0, 0.0, 1.0]))
    assert deep.push_face("die.n.01_1") is None, "a stroke longer than a compartment is deep"
    # otwukr's shelves are thinner than the 5 mm a ray from inside the board started under: it missed every one and
    # the board's box corner stood in (which happened to be right for a case square to the robot)
    thin = _region_sim(_shelves([0.44, 0.72, 1.0], thick=0.004), objects=objects)
    thin.scene_aabbs, thin.objects, thin.seen_boxes = sim.scene_aabbs, sim.objects, sim.seen_boxes
    die.get_position_orientation = lambda: (th.tensor([0.90, 0.1, 0.735]), th.tensor([0.0, 0.0, 0.0, 1.0]))
    assert thin.push_face("die.n.01_1")[3] == pytest.approx(sim.push_face("die.n.01_1")[3], abs=2e-3)


def test_a_push_hint_carries_the_strokes_depth_to_the_planner():
    """button_hints puts push_face's depth on the <item>_button hint (T5); without it cuTAMP strokes its config's
    1.5 cm (tiptop_sim_r1pro.yml push_depth) and the book never reaches the edge."""
    from omnigibson.tiptop.r1pro import PUSH_OVERHANG, PUSH_RADIUS

    bookcase = SimpleNamespace(fixed_base=True, category="bookcase")
    objects = {}
    sim = _region_sim(_shelves([0.44, 0.72, 1.0]), objects=objects)
    sim.scene_aabbs = lambda: [(bookcase, np.array([0.69, -0.41, 0.0]), np.array([1.11, 0.41, 1.0]))]
    objects["comic_book.n.01_3"] = _seen("comic_book.n.01_3", [0.90, 0.1, 0.735], [-0.14, -0.1, -0.015], [0.14, 0.1, 0.015], sim)  # fmt: skip
    sim.label_of = lambda bddl: "comic_book_3"
    sim.button_hints = MethodType(R1ProSim.button_hints, sim)
    hint = sim.button_hints([{"predicate": "push", "args": ["comic_book.n.01_3"]}])["comic_book_3_button"]
    assert hint["depth"] == pytest.approx(0.76 - 0.7 + PUSH_OVERHANG) and hint["radius"] == PUSH_RADIUS
    assert hint["position"] == pytest.approx([1.04, 0.1, 0.735]) and hint["normal"] == pytest.approx([1.0, 0.0, 0.0])


def test_a_level_load_rides_through_the_teleport_unfolded(monkeypatch):
    """E-level (review level-carry-tipped-by-travel-fold): the fold drives every arm joint to zero, which tips a load
    held level at the ready posture 60 deg on the way (fold_tilt.out), and the unfold tips it back. Carrying level
    the arm stays as it is; the landing check covers it where it is."""
    monkeypatch.setattr("omnigibson.tiptop.r1pro.gm", SimpleNamespace(HEADLESS=True))
    sim, checks, unfolds = _placement_sim(blocked_beyond=2.0)
    folds = []
    sim.fold_for_travel = lambda: folds.append(1) or [1.0, 1.0]
    sim.level_held = lambda arm: {"plate_1"}
    sim.stance_ready = None
    R1ProSim.place_robot(sim, 1.0, 0.0, 0.0)
    assert folds == [] and unfolds == [] and checks == [None] and sim.stance_ready is None
    sim.level_held = lambda arm: set()
    R1ProSim.place_robot(sim, 1.0, 0.0, 0.0)
    assert folds == [1] and unfolds == [[1.0, 1.0]]
    held = SimpleNamespace(level={"plate_1", "tray_1"}, hands=lambda: {"plate_1": "left", "tray_1": "right", "cup_1": "left"})
    assert R1ProSim.level_held(held, "left") == {"plate_1"} and R1ProSim.level_held(held, "right") == {"tray_1"}
    assert not R1ProSim.level_held(SimpleNamespace(level=set()), "left")


def test_a_flat_item_that_fits_the_compartment_lies_and_one_that_fits_neither_way_stands():
    """Review (upright-rule-fires-on-flat-items): the rule compared the longest extent with the narrower side, so a
    22 x 18 x 3 cm puzzle that fits the 25 x 18.8 cm toy box lying (and did, sweep4 r04) was stood 22 cm tall in an
    11.6 cm box, and a 29 cm board game whose centre on end stands 3 cm over the box's walls. Stood on end only when
    the footprint's diagonal is longer than the rectangle's (it fits lying at no angle) and its centre on end is
    still under the ceiling (where the scorer's Inside looks for it)."""
    objects = {}
    sim = _region_sim(None, objects=objects)
    objects["jigsaw_puzzle.n.01_1"] = _seen("jigsaw_puzzle.n.01_1", [1.0, 0.5, 0.8], [-0.11, -0.09, -0.015], [0.11, 0.09, 0.015], sim)  # fmt: skip
    sim.inside_rect = lambda item, container: (np.array([2.0, 1.0]), np.array([0.125, 0.094]), 0.797, 0.913)
    assert "rotation" not in sim.inside_region("jigsaw_puzzle.n.01_1", "toy_box.n.01_1")
    sim.inside_rect = lambda item, container: (np.array([2.0, 1.0]), np.array([0.094, 0.125]), 0.797, 0.913)  # turned
    assert "rotation" not in sim.inside_region("jigsaw_puzzle.n.01_1", "toy_box.n.01_1")
    sim.seen_boxes["jigsaw_puzzle_1"] = (np.array([-0.15, -0.095, -0.015]), np.array([0.15, 0.095, 0.015]))  # 30 x 19 cm
    assert "rotation" not in sim.inside_region("jigsaw_puzzle.n.01_1", "toy_box.n.01_1"), "on end its centre is over the walls"
    sim.inside_rect = lambda item, container: (np.array([2.0, 1.0]), np.array([0.094, 0.125]), 0.797, 0.96)  # a deeper box
    assert "rotation" in sim.inside_region("jigsaw_puzzle.n.01_1", "toy_box.n.01_1")
    sim.inside_rect = lambda item, container: (np.array([2.0, 1.0]), np.array([0.0545, 0.056]), 0.6, 0.75)  # a basket's
    sim.seen_boxes["jigsaw_puzzle_1"] = (np.array([-0.062, -0.0375, -0.01]), np.array([0.062, 0.0375, 0.01]))  # a cheese slice
    assert "rotation" not in sim.inside_region("jigsaw_puzzle.n.01_1", "toy_box.n.01_1"), "12.4 x 7.5 cm lies in 10.9 x 11.2 across"


def test_the_intent_regions_are_routed_by_predicate_and_filed_where_the_wire_names_the_plane():
    """T8: stamp, cut, heat, aim and attached each take their hint from the oracle and land under the target's
    label, or under the planner's support label for aim and for a floor target (protocol.tiptop_goal names the
    plane there); without an oracle none is made. And aim_target's yaw turns the nozzle's spray onto the target."""
    from b1k.bridge.protocol import PLANNER_SUPPORT

    made = []
    sim = SimpleNamespace(
        label_of=lambda bddl: bddl.split(".")[0] + "_1", region_refused=set(),
        scene_object=lambda name: SimpleNamespace(fixed_base=True),
        stamp_region=lambda tool, target, particles, projection=None: made.append(("stamp", particles, projection)) or {"r": "stamp"},
        knife_region=lambda knife, food: {"r": "cut"},
        heat_region=lambda item, source, link: made.append(("heat", link)) or {"r": "heat"},
        aim_target=lambda tool, target, nozzle: made.append(("aim", nozzle)) or {"r": "aim"},
        attach_target=lambda child, parent, frames: made.append(("attach", frames)) or {"r": "attach"},
    )  # fmt: skip
    oracle = SimpleNamespace(particles=lambda t: "P", projection_box=lambda i: "SLAB", heat_link=lambda s: "HEAT",
                             nozzle=lambda t: "NOZ", attach_frames=lambda c, p: "FR")  # fmt: skip
    regions = lambda p, *a, **kw: R1ProSim.inside_regions(sim, [{"predicate": p, "args": list(a)}], **kw)
    assert regions("stamp", "vacuum.n.04_1", "floor.n.01_1", oracle=oracle) == {PLANNER_SUPPORT: {"r": "stamp"}}
    assert regions("stamp", "scrub_brush.n.01_1", "shoe.n.01_1", oracle=oracle) == {"shoe_1": {"r": "stamp"}}
    assert regions("cut", "knife.n.01_1", "onion.n.01_1", oracle=oracle) == {"onion_1": {"r": "cut"}}
    assert regions("heat", "pan.n.01_1", "stove.n.01_1", oracle=oracle) == {"stove_1": {"r": "heat"}}
    assert regions("aim", "atomizer.n.01_1", "plant.n.01_1", oracle=oracle) == {PLANNER_SUPPORT: {"r": "aim"}}
    assert regions("attached", "alarm.n.01_1", "nail.n.01_1", oracle=oracle) == {"nail_1": {"r": "attach"}}
    assert made == [("stamp", "P", "SLAB"), ("stamp", "P", "SLAB"), ("heat", "HEAT"), ("aim", "NOZ"), ("attach", "FR")]
    assert regions("aim", "atomizer.n.01_1", "plant.n.01_1") == {} and regions("stamp", "a.n.01_1", "b.n.01_1") == {}
    objects = {}
    real = _region_sim(None, objects=objects)
    objects["plant.n.01_1"] = _seen("plant.n.01_1", [1.0, 0.5, 0.3], [-0.1, -0.1, -0.3], [0.1, 0.1, 0.3], real)
    objects["atomizer.n.01_1"] = _seen("atomizer.n.01_1", [0.0, 0.0, 0.1], [-0.03, -0.03, -0.1], [0.03, 0.03, 0.1], real)
    frame = trimesh.transformations.rotation_matrix(np.pi / 2, [0.0, 1.0, 0.0])  # the nozzle's -z now looks along +x
    region = real.aim_target("atomizer.n.01_1", "plant.n.01_1", (frame, 0.4))
    to_target = np.array([1.0, 0.5]) / np.hypot(1.0, 0.5)
    assert (np.array(region["rotation"]) @ -frame[:3, 2])[:2] == pytest.approx(to_target, abs=1e-9), "spray at the plant"
    assert np.hypot(*region["pose"][:2]) < np.hypot(1.0, 0.5), "stood on the robot's side of it"


def test_the_bay_is_the_one_behind_the_door_furthest_from_its_own_closed_end(monkeypatch):
    """T12 (E-region petcxr): inside_rect hands bay() the opened door's centre, and the door that is most open is
    measured from its own closed end (a door hung the other way rests at its upper limit shut). With the AABB centre
    between petcxr's two columns the tie went to the lower level: the column behind the shut door."""
    from omnigibson.tiptop import articulation

    def accepts(points):
        x, z = points.numpy()[:, 0], points.numpy()[:, 2]
        return th.tensor(((np.abs(x - 0.2) < 0.2) & (z >= 0.0)) | ((np.abs(x - 1.0) < 0.2) & (z >= 0.26)))

    lo, hi = th.tensor([0.0, -0.3, 0.0]), th.tensor([1.2, 0.3, 0.5])
    fill = SimpleNamespace(name="fill", is_meta_link=True, meta_link_type="fillable", visual_aabb=(lo, hi),
                           visual_aabb_center=(lo + hi) / 2, visual_aabb_extent=hi - lo, check_points_in_volume=accepts)  # fmt: skip
    doors = {n: SimpleNamespace(aabb=(th.tensor([x, -0.3, 0.0]), th.tensor([x + 0.05, 0.3, 0.5]))) for n, x in (("left", -0.05), ("right", 1.2))}  # fmt: skip
    fridge = SimpleNamespace(fixed_base=True, links={"fill": fill, **doors})
    joints = [dict(name="j_left", kind="revolute", axis=[0, 0, 1], origin=[0, 0, 0], lower=0.0, upper=1.6, position=1.5, closed=1.6, link="left"),
              dict(name="j_right", kind="revolute", axis=[0, 0, 1], origin=[0, 0, 0], lower=0.0, upper=1.6, position=1.2, closed=0.0, link="right")]  # fmt: skip
    monkeypatch.setattr(articulation, "openable_joints", lambda obj: joints)
    sim = SimpleNamespace(scene_object=lambda name: fridge, item_height=lambda name: 0.1)
    sim.inside_rect, sim.bay = MethodType(R1ProSim.inside_rect, sim), MethodType(R1ProSim.bay, sim)
    centre, half, floor, ceiling = sim.inside_rect("jar.n.01_1", "fridge.n.01_1")
    assert centre[0] == pytest.approx(1.0, abs=0.05) and floor == pytest.approx(0.26, abs=0.011) and ceiling == 0.5
    assert 0.1 < half[0] < 0.2 and half[1] > 0.2  # the run of accepted points about it, shrunk to the corners accepted
    joints[0]["position"] = 0.3  # the left door 1.3 rad off its own closed end: the most open, its column
    assert sim.inside_rect("jar.n.01_1", "fridge.n.01_1")[0][0] == pytest.approx(0.2, abs=0.05)


def test_region_rotations_are_the_base_frames_on_a_turned_robot():
    """T13: upright_rotation and attach_target hand the planner a rotation in the BASE frame (its object frame is
    base-aligned at the capture). Every region test's robot faced world +x, where the frames coincide, so dropping
    the conversion passed; on a robot yawed 90 deg the unconverted rose stays lying across the vase."""
    import omnigibson.utils.transform_utils as T

    objects = {}
    sim = _region_sim(None, objects=objects)
    yawed = th.tensor([0.0, 0.0, np.sin(np.pi / 4), np.cos(np.pi / 4)])  # the base turned 90 deg
    sim.base_pose = lambda: (th.tensor([0.0, 0.0, 0.0]), yawed)
    base = T.quat2mat(yawed).numpy()
    yaw30 = th.tensor([0.0, 0.0, np.sin(np.pi / 12), np.cos(np.pi / 12)])
    objects["rose.n.01_1"] = sim.objects["rose_1"] = SimpleNamespace(fixed_base=False, get_position_orientation=lambda: (th.tensor([1.0, 0.5, 0.75]), yaw30))  # fmt: skip
    sim.seen_boxes["rose_1"] = (np.array([-0.15, -0.03, -0.02]), np.array([0.15, 0.03, 0.02]))
    R_base = np.array(sim.upright_rotation("rose.n.01_1"))
    long_axis_base = base.T @ T.quat2mat(yaw30).numpy() @ [1.0, 0.0, 0.0]
    assert R_base @ long_axis_base == pytest.approx([0.0, 0.0, 1.0], abs=1e-6)
    alarm = SimpleNamespace(fixed_base=False, get_position_orientation=lambda: (th.tensor([0.6, 0.0, 0.019]), th.tensor([0.0, 0.0, 0.0, 1.0])))  # fmt: skip
    objects["fire_alarm.n.02_1"] = sim.objects["fire_alarm_1"] = alarm
    sim.seen_boxes["fire_alarm_1"] = (np.array([-0.05, -0.05, -0.019]), np.array([0.05, 0.05, 0.019]))
    male, female = np.eye(4), trimesh.transformations.rotation_matrix(np.pi / 2, [0.0, 1.0, 0.0])
    male[:3, 3], female[:3, 3] = [0.6, 0.0, 0.0], [1.0, 0.3, 1.71]
    region = sim.attach_target("fire_alarm.n.02_1", "wall_nail.n.01_1", (male, female))
    assert np.allclose(base @ np.array(region["rotation"]) @ base.T, female[:3, :3]), "the nail's frame, in the base's"


def test_the_pour_tilts_the_wrist_the_way_its_limit_allows_and_a_stopped_ramp_is_no_pour():
    """T16 (N-rotate): joint7 within 90 deg of its upper limit tips negative; a ramp the preflight or the block
    detector stopped reports False, and the wrist is still turned back."""
    from omnigibson.tiptop.r1pro import POUR_HOLD_STEPS, POUR_TILT

    ramps = []
    joint = SimpleNamespace(upper_limit=2.0, lower_limit=-2.0)
    sim = SimpleNamespace(planned_joints=["left_arm_joint1", "left_arm_joint7"], q_arm=lambda: [0.3, 1.0], posture={},
                          last_gripper=-1.0, robot=SimpleNamespace(joints={"left_arm_joint7": joint}),
                          ramp_to=lambda q, posture, gripper, settle, note="", **kw: ramps.append((list(q), settle)) or None)  # fmt: skip
    assert R1ProSim.tilt_wrist(sim, "left") is True
    assert ramps == [([0.3, 1.0 - POUR_TILT], POUR_HOLD_STEPS), ([0.3, 1.0], ramps[1][1])]  # up would hit the limit
    sim.q_arm = lambda: [0.3, 0.0]
    assert R1ProSim.tilt_wrist(sim, "left") and ramps[-2][0] == [0.3, POUR_TILT]
    sim.ramp_to = lambda q, *a, **kw: ramps.append(list(q)) or ("left_arm_link7 intersects table", 0, 0.0)
    assert R1ProSim.tilt_wrist(sim, "left") is False and ramps[-1] == [0.3, 0.0]
    assert R1ProSim.tilt_wrist(SimpleNamespace(planned_joints=["torso_joint1"]), "left") is False
