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
        robot=SimpleNamespace(get_joint_positions=lambda: th.tensor([0.4, 0.0, 1.2])),
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

    source = inspect.getsource(R1ProSim.open_container)
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
        held_objects={}, _fold_blocked=False, planned_joints=["j1", "j2"],
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


def test_return_to_ready_ramps_then_asks_the_planner_for_the_ready_configuration():
    from omnigibson.tiptop.r1pro import R1ProSim

    asked = []

    def sim_with(ramp_result, held=None):
        return SimpleNamespace(
            q_home=[1.0, 2.0], q_arm=lambda: [0.0, 0.0], posture={}, last_gripper=0.0, arm="left",
            robot=SimpleNamespace(_ag_obj_in_hand={"left": held}),
            ramp_to=lambda *args, **kwargs: ramp_result,
            planned_approach=lambda where, quat, note, goal_q=None: asked.append(goal_q) or True,
        )

    assert R1ProSim.return_to_ready(sim_with(None)) and asked == []
    assert R1ProSim.return_to_ready(sim_with(("left_gripper_link intersects booth", 0, 0.0)))
    assert asked == [[1.0, 2.0]]  # the planner is asked for the ready CONFIGURATION, not a pose
    # holding something: the move request cannot carry it (it would be a room static the fingers start in)
    assert not R1ProSim.return_to_ready(sim_with(("attached_object_left intersects table", 0, 0.0), held=object()))
    assert asked == [[1.0, 2.0]]
    home = SimpleNamespace(q_home=[1.0, 2.0], q_arm=lambda: [1.0, 2.0])
    assert R1ProSim.return_to_ready(home)  # already there: nothing moves


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


def _pick_ep(attached_first, target, releases=True):
    from omnigibson.tiptop.bench import Episode

    calls = []

    def hold(n, gripper=None):
        calls.append(("hold", gripper))
        if releases:
            sim.robot._ag_obj_in_hand["left"] = None

    sim = SimpleNamespace(
        arm="left", n_steps=0, OPEN=1.0, held_objects={},
        robot=SimpleNamespace(_ag_obj_in_hand={"left": attached_first}),
        press_grasp=lambda arm, bddl, spare=(): False, scene_object=lambda name: target, hold=hold,
        return_to_ready=lambda note, allowed_contacts=None: calls.append(("ready", note, allowed_contacts)) or True,
        retreat_contacts=lambda bddl=None: {"floors_1": {"left_gripper_link"}, "target": {"left_gripper_link"}},
    )
    ep = SimpleNamespace(
        sim=sim, rounds=1, records=[], planners={"left": ("planner", {})},
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
        _motion_obstacles=lambda held, allowed, base=None: [("floors_1", None), ("desk_1", None), ("paver_1", None)],
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
