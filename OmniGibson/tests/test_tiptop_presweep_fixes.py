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
        robot=SimpleNamespace(_ag_obj_in_hand={"left": attached_first}), jaw_spans=lambda bddl: True,
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


def test_an_object_wider_than_the_jaw_every_way_round_skips_the_planner_round():
    """assembling_gift_baskets 2026-09-23: every 10 cm pillar candle cost a full planner holding() round (0 of 9
    could succeed, 673 s over 3 episodes) before the press took it."""
    candle = SimpleNamespace(name="pillar_candle_88")
    planned = []
    for spans in (False, True):
        ep, calls = _pick_ep(attached_first=None, target=candle)
        ep.sim.press_grasp = lambda arm, bddl, spare=(): True
        ep.sim.jaw_spans = lambda bddl: spans
        ep.plan_and_execute = lambda atoms, floor=False: planned.append(atoms)
        assert ep.pick("candle.n.01_1") is True
        assert planned == ([] if not spans else [[{"predicate": "holding", "args": ["candle.n.01_1"]}]])


def test_the_jaw_test_reads_the_objects_own_level_axes():
    from omnigibson.tiptop.r1pro import JAW_GAP

    def sim_with(lo, hi):
        return SimpleNamespace(
            scene_object=lambda name: name,
            own_box=lambda obj: (np.zeros(3), np.eye(3), np.array(lo), np.array(hi)),
        )

    assert not R1ProSim.jaw_spans(sim_with([-0.05, -0.05, 0.0], [0.05, 0.05, 0.11]), "candle")  # 10 cm every way round
    assert R1ProSim.jaw_spans(sim_with([-0.14, -0.045, 0.0], [0.14, 0.045, 0.12]), "shoe")  # 9 cm across, 12 cm tall
    assert R1ProSim.jaw_spans(sim_with([-0.225, -0.21, -0.015], [0.225, 0.21, 0.015]), "tile")  # thin: a side grasp spans it
    # a plate lying flat, 17 x 18 cm and 2.6 cm thick: the planner held it by the rim (putting_dirty_dishes, sweep3)
    assert R1ProSim.jaw_spans(sim_with([-0.085, -0.09, 0.0], [0.085, 0.09, 0.026]), "plate")
    assert R1ProSim.jaw_spans(SimpleNamespace(scene_object=lambda n: n, own_box=lambda obj: None), "unseen")
    assert JAW_GAP < 0.1


def test_a_lift_refused_by_the_robots_own_body_lets_go_where_the_object_rests():
    """laying_tile 301/302/303 (2026-09-23): the post-grasp lift was refused at sample 0 against base_link, the tile
    stayed in the hand, and no motion ever left that start: 0 env steps after the grasp, TransferBlocked."""
    from omnigibson.tiptop.run import DROP_STEPS

    def sim_with(stopped):
        holds = []
        tile = SimpleNamespace(name="ceramic_tile_186", aabb=(th.zeros(3), th.ones(3)))
        sim = SimpleNamespace(
            robot=SimpleNamespace(get_joint_positions=lambda: th.zeros(1)), joint_index={"j": 0},
            scene_aabbs=lambda: [], retreat_contacts=lambda: {}, objects={"tile_1": tile}, held_objects={},
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
                 link="drawer")
    state = [joint]
    monkeypatch.setattr("omnigibson.tiptop.r1pro.openable_joints", lambda obj: [state[-1]])
    ramps, tucks = [], []

    def ramp_to(q_arm, posture, gripper, settle, note="", **kwargs):
        ramps.append(note)
        if note.startswith("approach") and not tucks:
            return (refusal, 0, 0.0, 300)
        return None

    grasp = dict(joint=joint, jaws=[np.array([0.0, 0.0, 1.0])], kind="bar", tips=[0.0, 0.0, 0.8], nudges=0, travel=0.24)
    plan = dict(solutions=[[0.1], [0.2], [0.3]], reached=1, why="", grasp_pose=np.eye(4), lead=np.array([-1.0, 0.0, 0.0]))
    ik = SimpleNamespace(fk=lambda q, frame: (np.zeros(3), np.array([0.0, 0.0, 0.0, 1.0])), solve=lambda *a, **k: None)
    sim = SimpleNamespace(
        arm="left", other_arm="right", OPEN=1.0, posture={"right_arm_joint1": 0.0}, q_home=None, planned_joints=[],
        scene_object=lambda name: SimpleNamespace(name="cab"), joint_index={"left_arm_joint1": 0},
        robot=SimpleNamespace(eef_links={"left": SimpleNamespace(get_position_orientation=lambda: (th.zeros(3), None))},
                              get_joint_positions=lambda: th.zeros(1)),
        container_grasps=lambda obj, joints, fraction, hand_world, height=None: [grasp],
        arm_ik=lambda arm, frame=None, with_torso=False: ik, ik_joint_names=lambda arm, with_torso=False: ["left_arm_joint1"],
        scene_aabbs=lambda: [], solve_pull=lambda ik, g, jaw, seed, **kwargs: (plan, ""),
        reach_plan=lambda *a, **kwargs: [], planned_standoff=lambda *a: True, _targets_from=lambda joints_of, q: q,
        container_body=lambda obj, link: None,
        ramp_to=ramp_to, tuck_idle_arm=lambda: tucks.append(1) or True, grasp_contacts=R1ProSim.grasp_contacts,
        close_on=lambda *a, **kwargs: ([0.2], True), follow_pull=lambda *a: state.append(dict(joint, position=0.25)) or (1, ""),
        hold=lambda n, g: None, arm_hits_scene=lambda *a, **kwargs: False,
    )
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
                 "item_height"):
        setattr(sim, name, MethodType(getattr(R1ProSim, name), sim))
    return sim


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
    assert _region_sim(case, base=(-1.5, 0.0)).shelf_of(None, 0.04, fillable) is None  # 2.2 m from the near edge


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
        floor_surface=lambda within=None: {"floor": within}, footprint_region=lambda table: {"top": table},
        rest_region=lambda item, fixture: {"board": fixture}, inside_region=lambda item, container: None,
        label_of=lambda bddl: bddl.split(".")[0] + "_1", region_refused=set(),
        scene_object=lambda name: SimpleNamespace(fixed_base=name != "table.n.02_3"),  # _3 is a movable table
    )
    regions = lambda *atoms: R1ProSim.inside_regions(sim, [{"predicate": p, "args": list(a)} for p, *a in atoms])
    assert regions(("under", "mousetrap.n.01_4", "sink.n.01_1")) == {"table": {"floor": "sink.n.01_1"}}
    assert regions(("ontop", "book.n.02_1", "table.n.02_1")) == {"table": {"top": "table.n.02_1"}}
    assert regions(("ontop", "book.n.02_1", "table.n.02_1"), ("ontop", "book.n.02_2", "table.n.02_2")) == {}
    assert regions(("ontop", "book.n.02_1", "table.n.02_3")) == {}  # a movable table's box is not the map's to read
    assert regions(("ontop", "x.n.01_1", "floor.n.01_1"), ("under", "y.n.01_1", "sink.n.01_1")) == {"table": {"floor": None}}
    assert regions(("touching", "gym_shoe.n.01_2", "hallstand.n.01_1")) == {"hallstand_1": {"board": "hallstand.n.01_1"}}


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
                         fallen=lambda: False)
    ep.stand_for = MethodType(Episode.stand_for, ep)
    assert ep.stand_for("plywood.n.01_1")["x"] == -7.03
    assert calls == [(0.9, [], []), (REACH_FAR, [], [(-6.23, -3.12, np.pi / 2, "paver_sekqqq_0")])]
    ep.stand_for("plywood.n.01_1")  # the next search avoids the stance taken and starts with nothing refused
    assert calls[2] == (0.9, [(-7.03, -2.32)], [])
