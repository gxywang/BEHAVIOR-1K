"""Following a joint with the gripper: the geometry of opening a drawer or a door. No simulator.

Opening is the largest single unlock left in the challenge set -- expanding all 100 goals puts the current
vocabulary's ceiling at a mean q_score of 0.415 and open/not-open takes it to 0.636 -- and all of the geometry can
be checked here, which is where a mistake in it should be found.
"""

import numpy as np
import pytest

from omnigibson.tiptop.articulation import (
    OPEN_FRACTION_REACH,
    OPEN_FRACTION_SCORED,
    follow_joint,
    is_open,
    joint_transform,
    opening_travel,
    pose_matrix,
    rotation_about,
)

HANDLE = pose_matrix([0.5, 0.0, 0.9], [0.0, 0.0, 0.0, 1.0])


def test_a_drawer_handle_slides_straight_out_along_the_joint_axis():
    poses = follow_joint(HANDLE, "prismatic", [1.0, 0.0, 0.0], [0, 0, 0], travel=0.30, steps=5)
    assert len(poses) == 5
    assert np.allclose(poses[0], HANDLE), "the path starts where the hand must take hold"
    assert np.allclose(poses[-1][:3, 3], [0.80, 0.0, 0.9])
    assert all(np.allclose(p[:3, :3], np.eye(3)) for p in poses), "a drawer does not turn the hand"
    steps = np.diff([p[0, 3] for p in poses])
    assert np.allclose(steps, steps[0]), "evenly spaced along the slide"


def test_a_door_handle_swings_on_an_arc_and_turns_the_hand_with_it():
    hinge = [0.5, -0.4, 0.9]  # the handle is 0.4 m from the hinge, on a vertical axis
    poses = follow_joint(HANDLE, "revolute", [0.0, 0.0, 1.0], hinge, travel=np.pi / 2, steps=9)
    radius = [float(np.linalg.norm(p[:3, 3] - np.asarray(hinge))) for p in poses]
    assert np.allclose(radius, radius[0], atol=1e-9), "the handle stays on its arc"
    assert radius[0] == pytest.approx(0.4, abs=1e-9)
    turned = poses[-1][:3, :3] @ np.array([1.0, 0.0, 0.0])
    assert np.allclose(turned, [0.0, 1.0, 0.0], atol=1e-9), "the hand turned with the door, a quarter turn"


def test_the_hand_must_turn_on_a_door_or_it_would_have_to_slide_around_the_handle():
    hinge = [0.5, -0.4, 0.9]
    poses = follow_joint(HANDLE, "revolute", [0.0, 0.0, 1.0], hinge, travel=np.pi / 3, steps=4)
    for p in poses[1:]:
        assert not np.allclose(p[:3, :3], np.eye(3))


def test_an_unopenable_joint_is_refused_rather_than_guessed_at():
    with pytest.raises(ValueError, match="does not open"):
        joint_transform("fixed", [1, 0, 0], [0, 0, 0], 0.1)


def test_rotation_about_a_zero_axis_is_the_identity_not_a_crash():
    assert np.allclose(rotation_about([0.0, 0.0, 0.0], 1.0), np.eye(3))


def test_travel_goes_away_from_the_closed_limit_so_either_hinging_works():
    # a drawer closed at its lower limit opens by going up the range
    assert opening_travel("prismatic", 0.0, 0.4, position=0.0, fraction=0.5, closed=0.0) == pytest.approx(0.2)
    # a door closed at its UPPER limit opens by going down it
    assert opening_travel("revolute", -1.6, 0.0, position=0.0, fraction=0.5, closed=0.0) == pytest.approx(-0.8)


def test_opening_a_closed_drawer_by_a_fraction_means_that_fraction_of_its_range():
    # store_honey's cabinet: four drawers, range [0, 0.39], all resting closed. 80% open is 0.312 m of travel --
    # the earlier version answered 0.078, being the distance to a target 80% of the way from the FAR limit
    for fraction in (0.80, OPEN_FRACTION_SCORED):
        travel = opening_travel("prismatic", 0.0, 0.39, position=0.0, fraction=fraction, closed=0.0)
        assert travel == pytest.approx(0.312, abs=1e-6)


def test_a_partly_open_joint_travels_only_the_rest_of_the_way():
    assert opening_travel("prismatic", 0.0, 0.4, position=0.1, fraction=0.5, closed=0.0) == pytest.approx(0.1)


def test_travel_never_asks_for_more_than_the_joint_has():
    far = opening_travel("prismatic", 0.0, 0.4, position=0.0, fraction=2.0, closed=0.0)
    assert far == pytest.approx(0.4), "clamped to the limit, not 0.8 m into the cabinet"


def test_a_joint_with_no_range_asks_for_no_travel():
    assert opening_travel("prismatic", 0.3, 0.3, position=0.3, fraction=0.8, closed=0.3) == 0.0


def test_both_open_strokes_clear_the_state_threshold_and_stiction():
    """Both fractions ask for a long pull, and the reason is measured rather than assumed.

    This test used to assert the opposite -- that scoring an `open` atom was a much shorter, cheaper stroke than
    reaching inside. Trying it in the simulator refuted that: asked for 8% of store_honey's drawer range (31 mm
    spread over ten steps, 3 mm a step) the drawer moved 7 mm and stayed shut, while 80% (31 mm a step) tracked
    the hand one-to-one to 13 cm, where the ARM ran out of reach. Short steps do not break the joint's stiction.
    """
    assert OPEN_FRACTION_SCORED > 0.05, "OmniGibson flips Open at 5% of the range; clear it"
    assert OPEN_FRACTION_REACH > 0.05
    assert OPEN_FRACTION_SCORED >= 0.5, "a short stroke was measured not to move the drawer at all"
    # 10% of a range is the most that can be asked before the per-step move is under a centimetre on a 0.39 m
    # drawer split into OPEN_PATH_STEPS steps, which is the regime that failed.
    assert OPEN_FRACTION_SCORED * 0.39 / 10 > 0.01, "per-step move must be over a centimetre to break stiction"


def test_is_open_matches_omnigibsons_five_percent_rule_from_the_closed_end():
    assert not is_open(0.0, 0.4, position=0.01, closed=0.0)  # 2.5% out: still closed
    assert is_open(0.0, 0.4, position=0.03, closed=0.0)  # 7.5% out: open
    assert not is_open(0.0, 0.4, position=0.39, closed=0.4), "a joint resting at its closed limit is closed"


def test_a_lid_resting_at_its_open_limit_reads_open_and_closes_toward_its_closed_end():
    """A laptop or a car trunk rests at its OPEN limit; OmniGibson's Open state counts from the closed end (the lower
    limit, or the upper one where the metadata lists the joint with direction -1)."""
    assert is_open(0.0, 2.4, position=2.4, closed=0.0)  # the laptop at its open limit
    assert not is_open(0.0, 2.4, position=2.4, closed=2.4)  # a door hung the other way, resting shut
    assert opening_travel("revolute", 0.0, 2.4, position=2.4, fraction=0.0, closed=0.0) == pytest.approx(-2.4)
    assert opening_travel("revolute", 0.0, 2.4, position=0.1, fraction=0.5, closed=0.0) == pytest.approx(1.1)
    assert opening_travel("revolute", -1.6, 0.0, position=-0.2, fraction=0.0, closed=0.0) == pytest.approx(0.2)


def test_joint_frame_turns_the_axis_letter_by_the_joints_own_rotation_and_puts_the_hinge_at_its_own_origin():
    """USD puts the joint frame at parent * (localPos0, localRot0) and the axis letter is in THAT frame. Read as
    R_parent @ letter through the parent's origin, 38 of 39 challenge hinges led the hand off by more than 30 deg
    (skill_gap joint_reader_check.out)."""
    from omnigibson.tiptop.articulation import joint_frame

    quarter = (np.cos(np.pi / 4), 0.0, 0.0, np.sin(np.pi / 4))  # localRot0: a quarter turn about z, wxyz
    axis, origin = joint_frame([1.0, 2.0, 0.0], np.eye(3), [0.3, 0.0, 0.5], quarter, [1.0, 0.0, 0.0])
    assert np.allclose(axis, [0.0, 1.0, 0.0], atol=1e-9), "the X letter, turned 90 deg by localRot0"
    assert np.allclose(origin, [1.3, 2.0, 0.5])
    parent = rotation_about([0.0, 0.0, 1.0], np.pi / 2)  # and the parent link's own pose composes on top
    axis, origin = joint_frame([0.0, 0.0, 0.0], parent, [0.3, 0.0, 0.5], quarter, [1.0, 0.0, 0.0])
    assert np.allclose(axis, [-1.0, 0.0, 0.0], atol=1e-9) and np.allclose(origin, [0.0, 0.3, 0.5], atol=1e-9)


def test_a_door_open_80_deg_is_pushed_on_the_face_that_trails_its_motion_whichever_way_it_goes():
    """push_joint (F-close): the closed hand comes in along the link's motion onto the face that trails it, a door's
    outer face to shut it and its inner face to push it wider. Nothing is taken hold of."""
    from types import MethodType, SimpleNamespace

    import trimesh

    from omnigibson.tiptop.r1pro import R1ProSim

    # a 2 cm panel hinged on a vertical axis through the origin, along +y when shut, swung 80 deg toward -x
    panel = trimesh.creation.box([0.02, 0.5, 1.0])
    panel.apply_translation([0.0, 0.25, 0.5])
    panel.apply_transform(trimesh.transformations.rotation_matrix(np.radians(80), [0.0, 0.0, 1.0]))
    j = dict(name="j_door", kind="revolute", axis=np.array([0.0, 0.0, 1.0]), origin=np.zeros(3), lower=0.0,
             upper=1.6, position=np.radians(80), closed=0.0, link="door")  # fmt: skip
    obj = SimpleNamespace(links={"door": SimpleNamespace(name="door")})
    sim = SimpleNamespace(link_trimesh_world=lambda link: panel)
    sim.surface_point = MethodType(R1ProSim.surface_point, sim)
    hand = np.array([0.0, 0.0, 0.9])
    for target in (0.0, 1.5):  # shut it; push it wider
        grasps = R1ProSim.push_grasps(sim, obj, j, target, hand)
        assert grasps and all(g["kind"] == "push" and g["press"] == 0.0 and g["nudges"] == 0 for g in grasps)
        motion = np.array([np.cos(np.radians(80)), np.sin(np.radians(80)), 0.0]) * (1.0 if target < j["position"] else -1.0)
        for g in grasps:
            assert np.allclose(g["into"], motion, atol=1e-6) and np.allclose(g["lead"], -motion, atol=1e-6)
            assert (g["tips"] - panel.centroid) @ motion == pytest.approx(-0.01, abs=1e-3), "on the trailing face"
        assert grasps[0]["tips"][2] == pytest.approx(0.9, abs=0.15), "nearest the hand's own height first"


# --------------------------------------------------------------- the handle heuristic
class _Link:
    """A link with just the box the handle heuristic reads."""

    def __init__(self, lo, hi):
        import torch as th

        self.aabb = (th.tensor(lo, dtype=th.float32), th.tensor(hi, dtype=th.float32))


def test_the_handle_sits_on_the_face_that_leads_when_the_drawer_opens():
    from omnigibson.tiptop.articulation import handle_point

    drawer = _Link([0.0, -0.25, 0.5], [0.5, 0.25, 0.65])  # 0.5 m deep, opening along +x
    point = handle_point(drawer, [1.0, 0.0, 0.0], opening_sign=+1.0)
    assert point[0] == pytest.approx(0.5), "the drawer front, not its middle"
    assert point[1] == pytest.approx(0.0) and point[2] == pytest.approx(0.575)


def test_the_handle_follows_the_direction_the_joint_actually_opens():
    from omnigibson.tiptop.articulation import handle_point

    drawer = _Link([0.0, -0.25, 0.5], [0.5, 0.25, 0.65])
    assert handle_point(drawer, [1.0, 0.0, 0.0], opening_sign=-1.0)[0] == pytest.approx(0.0)


def test_a_degenerate_axis_falls_back_to_the_middle_of_the_link():
    from omnigibson.tiptop.articulation import handle_point

    link = _Link([0.0, 0.0, 0.0], [1.0, 1.0, 1.0])
    assert np.allclose(handle_point(link, [0.0, 0.0, 0.0], 1.0), [0.5, 0.5, 0.5])


# --------------------------------------------------------------- grasp orientations for a handle
def test_the_hands_own_orientation_is_offered_first():
    from omnigibson.tiptop.articulation import grasp_orientations

    cur = np.eye(3)
    out = grasp_orientations(cur, [1.0, 0.0, 0.0])
    assert np.allclose(out[0], cur), "what the hand is already holding costs nothing to try"
    assert len(out) > 1, "and it is rarely the one that works"


def test_every_offered_orientation_is_a_rotation():
    from omnigibson.tiptop.articulation import grasp_orientations

    cur = np.array([[0.0, -1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 1.0]])
    for rot in grasp_orientations(cur, [0.0, 1.0, 0.3]):
        assert np.allclose(rot @ rot.T, np.eye(3), atol=1e-9), "orthonormal"
        assert float(np.linalg.det(rot)) == pytest.approx(1.0, abs=1e-9), "right handed, not a reflection"


def test_some_offered_orientation_faces_the_way_the_drawer_comes_out():
    from omnigibson.tiptop.articulation import grasp_orientations

    pull = np.array([1.0, 0.0, 0.0])  # the drawer comes toward +x, so a jaw axis should point back along -x
    out = grasp_orientations(np.eye(3), pull)
    assert any(any(np.allclose(rot[:, i], -pull, atol=1e-6) for i in range(3)) for rot in out), (
        "at least one candidate turns an axis of the hand to face the drawer"
    )


def test_a_degenerate_pull_direction_just_offers_what_the_hand_has():
    from omnigibson.tiptop.articulation import grasp_orientations

    assert len(grasp_orientations(np.eye(3), [0.0, 0.0, 0.0])) == 1


def test_the_edge_grip_aims_at_the_top_of_the_panel_and_the_face_grip_at_its_middle():
    """No BEHAVIOR asset marks a handle: store_honey's cabinet is four flat drawer fronts tagged only "openable".

    A parallel jaw has nothing to close on at the middle of a flat face, but it can come down over the panel's top
    edge and close across its thickness, so both are offered and the caller tries the edge first.
    """
    from omnigibson.tiptop.articulation import handle_point

    drawer = _Link([0.0, -0.25, 0.50], [0.5, 0.25, 0.65])  # 15 cm tall front, opening along +x
    face = handle_point(drawer, [1.0, 0.0, 0.0], opening_sign=+1.0, grip="face")
    edge = handle_point(drawer, [1.0, 0.0, 0.0], opening_sign=+1.0, grip="edge")
    assert face[0] == pytest.approx(0.5) and edge[0] == pytest.approx(0.5), "both on the leading face"
    assert face[2] == pytest.approx(0.575), "the face grip aims at the middle of the panel"
    assert edge[2] == pytest.approx(0.635), "the edge grip aims just below its top, where a jaw can pinch"
    assert edge[2] < 0.65, "and inside the panel, not in the air above it"


# --------------------------------------------------------------- the handle, read off the link's own mesh
def _box(lo, hi, n=4):
    """Vertices on the six faces of a box, ``n`` per edge, like a coarse mesh of a panel or a bar."""
    lo, hi = np.asarray(lo, float), np.asarray(hi, float)
    xs, ys, zs = (np.linspace(lo[i], hi[i], n) for i in range(3))
    pts = []
    for x in xs:
        for y in ys:
            pts += [[x, y, lo[2]], [x, y, hi[2]]]
    for x in xs:
        for z in zs:
            pts += [[x, lo[1], z], [x, hi[1], z]]
    for y in ys:
        for z in zs:
            pts += [[lo[0], y, z], [hi[0], y, z]]
    return np.asarray(pts, float)


def _drawer_with_rail():
    """store_honey's drawer as measured (bottom_cabinet/slgzfc, asset scale): a 0.426 deep box whose front panel
    spans 0.428 x 0.133, with a rail 0.376 wide x 0.008 tall x 0.010 deep standing 0.024 m proud, near the top."""
    box = _box([-0.13, -0.214, 0.136], [0.273, 0.214, 0.269], n=6)
    rail = _box([0.287, -0.188, 0.232], [0.297, 0.188, 0.240], n=5)
    return np.concatenate([box, rail])


def test_a_drawer_rail_is_read_as_a_bar_with_a_vertical_jaw():
    """The survey that preceded this called store_honey's drawer front "a 9 mm knife-edge lip, nothing a jaw can
    close around". Sliced into 4 mm layers, the front-most centimetre of that link is a 0.376 x 0.008 strip and the
    full 0.428 x 0.133 face lies 2.4 cm behind it: a rail. The densest layer -- what the survey took as the
    panel -- is the rail's own back face."""
    from omnigibson.tiptop.articulation import handle_on

    h = handle_on(_drawer_with_rail(), lead=[1.0, 0.0, 0.0])
    assert h["kind"] == "bar"
    assert h["proud"] == pytest.approx(0.024, abs=0.003)
    assert h["panel"] == pytest.approx(0.273, abs=0.003), "the panel is the first full-face layer, not the densest"
    assert np.allclose(np.abs(h["jaw"]), [0.0, 0.0, 1.0], atol=1e-6), "a horizontal rail is closed on from above and below"
    assert h["span"] == pytest.approx(0.376, abs=0.005) and np.allclose(np.abs(h["along"]), [0.0, 1.0, 0.0], atol=1e-6)
    assert h["point"][2] == pytest.approx(0.236, abs=0.003), "the grasp is on the rail, not the middle of the face"
    assert 0.287 <= h["point"][0] <= 0.297


def test_a_vertical_bar_on_a_door_is_read_as_a_bar_with_a_horizontal_jaw():
    """fridge/petcxr's right door: a bar 0.035 wide x 0.986 tall standing about 6 cm out from a 0.56 x 1.98 panel."""
    from omnigibson.tiptop.articulation import handle_on

    door = _box([0.40, -0.28, -1.0], [0.44, 0.28, 0.98], n=8)
    bar = _box([0.47, 0.20, -0.25], [0.50, 0.235, 0.736], n=6)
    h = handle_on(np.concatenate([door, bar]), lead=[1.0, 0.0, 0.0])
    assert h["kind"] == "bar"
    assert np.allclose(np.abs(h["jaw"]), [0.0, 1.0, 0.0], atol=1e-6), "closed on from the sides"
    assert np.allclose(np.abs(h["along"]), [0.0, 0.0, 1.0], atol=1e-6) and h["span"] == pytest.approx(0.986, abs=0.01)
    assert h["proud"] == pytest.approx(0.06, abs=0.005)


def test_a_flat_panel_offers_its_face_and_nothing_to_close_around():
    from omnigibson.tiptop.articulation import handle_on

    h = handle_on(_box([0.0, -0.2, 0.0], [0.4, 0.2, 0.15], n=6), lead=[1.0, 0.0, 0.0])
    assert h["kind"] == "flat" and h["jaw"] is None
    assert h["point"][0] == pytest.approx(0.4, abs=0.003), "on the front face"
    assert h["point"][1] == pytest.approx(0.0, abs=0.01) and h["point"][2] == pytest.approx(0.075, abs=0.01)


def test_a_shallow_step_is_a_lip_to_press_on_not_a_bar():
    from omnigibson.tiptop.articulation import handle_on

    panel = _box([0.0, -0.2, 0.0], [0.4, 0.2, 0.15], n=6)
    step = _box([0.40, -0.19, 0.12], [0.41, 0.19, 0.13], n=5)  # 1.0 cm proud: too shallow for the pads
    h = handle_on(np.concatenate([panel, step]), lead=[1.0, 0.0, 0.0])
    assert h["kind"] == "lip" and h["jaw"] is None
    assert h["point"][0] == pytest.approx(0.4, abs=0.003), "the press goes on the panel"


def test_a_trim_strip_beside_a_bar_does_not_widen_the_bar():
    """fridge/dszchb: a full-height trim strip 1 cm in front of the panel next to a bar 4.4 cm out. Read over the
    whole proud set they span the door; read over its front half, only the bar is left."""
    from omnigibson.tiptop.articulation import handle_on

    door = _box([0.30, -0.30, -0.6], [0.32, 0.30, 0.8], n=8)
    trim = _box([0.32, 0.27, -0.6], [0.33, 0.30, 0.8], n=6)  # 1 cm proud, the far edge, full height
    bar = _box([0.35, -0.27, 0.0], [0.364, -0.255, 0.24], n=5)  # 4.4 cm proud, the near edge
    h = handle_on(np.concatenate([door, trim, bar]), lead=[1.0, 0.0, 0.0])
    assert h["kind"] == "bar"
    assert h["extent"][0] < 0.03, "the bar's width, not the door's"
    assert h["span"] == pytest.approx(0.24, abs=0.01)


def test_leading_direction_of_a_drawer_is_its_slide_and_of_a_door_its_first_swing():
    from omnigibson.tiptop.articulation import leading_direction

    assert np.allclose(leading_direction("prismatic", [0.0, 1.0, 0.0], [0, 0, 0], [], travel=-0.3), [0.0, -1.0, 0.0])
    # a door hinged on a vertical axis at the origin, its panel along +y: opening (+z rotation) swings it to -x
    door = _box([-0.01, 0.0, 0.0], [0.01, 0.5, 1.0], n=5)
    lead = leading_direction("revolute", [0.0, 0.0, 1.0], [0.0, 0.0, 0.0], door, travel=+1.5)
    assert np.allclose(lead, [-1.0, 0.0, 0.0], atol=1e-6)
    assert np.allclose(leading_direction("revolute", [0.0, 0.0, 1.0], [0.0, 0.0, 0.0], door, travel=-1.5), [1.0, 0.0, 0.0])


def test_a_drawer_the_opening_stance_cannot_reach_is_pulled_from_the_looking_stance():
    """open_container picks its stance because the PULL solves from it, and never checks the arm can get to
    where the pull starts.

    store_honey is the case: "15 of 15 pull waypoints solve (80% of the range)" and then "left_arm_joint4 stopped
    following on the way to the standoff (leg 1 of 1, hand 40.1 cm short)" -- 4 attempts over two builds, every
    one of them. The generic looking stance that this replaced opened the same drawer 3 times out of 3, pulling
    13.0, 12.9 and 30.1 cm (2026-09-13/14). So a standoff failure falls back to standing the old way.
    """
    from omnigibson.tiptop.bench import Episode

    class Spec:
        opens = {}

    class Sim:
        arm = "left"
        n_steps = 0
        video_caption = ""

        def __init__(self, second):
            self.second, self.calls, self.stood = second, [], 0

        def open_container(self, arm, name, fraction=None, joint=None, height=None, stand=True):
            self.calls.append(stand)
            if stand:
                return {"opened": False, "why": "left_arm_joint4 stopped following on the way to the standoff "
                                                "(leg 1 of 1, hand 40.1 cm short)"}
            return self.second

        def return_to_ready(self, note="", allowed_contacts=None):
            return True

    class Ep:
        open_up = Episode.open_up

        def __init__(self, second):
            self.sim, self.spec, self.records = Sim(second), Spec(), []

        def stand_for(self, *names):
            self.sim.stood += 1
            return {}

    # the fallback rescues it
    ep = Ep({"opened": True, "position": 0.30})
    assert ep.open_up("cabinet.n.01_1") is True
    assert ep.sim.calls == [True, False], "try the pull-solving stance first, then the looking one"
    assert ep.sim.stood == 1, "the fallback has to stand again before it pulls"
    assert len(ep.records) == 2 and ep.records[1]["after_standoff_failed"], "both attempts are recorded, once each"

    # the fallback fails too: one record each, no double entry, and the verdict is False
    ep = Ep({"opened": False, "why": "no grasp on cabinet.n.01_1 solves from here"})
    assert ep.open_up("cabinet.n.01_1") is False
    assert len(ep.records) == 2, "each attempt recorded exactly once"

    # a failure that is NOT the standoff does not trigger a second pull
    class SimOther(Sim):
        def open_container(self, arm, name, fraction=None, joint=None, height=None, stand=True):
            self.calls.append(stand)
            return {"opened": False, "why": "no joint of cabinet.n.01_1 has a handle this arm can reach"}

    ep = Ep(None)
    ep.sim = SimOther(None)
    assert ep.open_up("cabinet.n.01_1") is False
    assert ep.sim.calls == [True], "only a standoff failure is worth standing again for"


def test_a_handle_approach_rejected_after_the_standoff_reach_falls_back_to_the_looking_stance_from_ready():
    """store_honey 301, twice (manip1g and manip2, 2026-09-25): the straight reach was refused, the planner's
    "(any configuration)" path reached the standoff in another IK branch, and the straight approach solved from the
    first branch was refused -- "handle approach rejected: left_realsense_link intersects torso_link2". That was
    final (q 0.0 at 240 steps), while the run whose planned path was itself rejected got the looking-stance
    fallback and opened the drawer (manip1, 2026-09-24). Same failure -- the arm cannot get from the opening stance
    to the hold -- same fallback. And the fallback stands from the ready posture: the executed reach left the torso
    bent (head camera 1.21 m), and the retry's search from there rejected all 5948 candidates as out of frame."""
    from omnigibson.tiptop.bench import Episode

    calls = []

    class Sim:
        arm, n_steps, video_caption = "left", 0, ""

        def open_container(self, arm, name, fraction=None, joint=None, height=None, stand=True):
            calls.append(("open", stand))
            if stand:
                return {"opened": False, "why": "handle approach rejected: left_realsense_link intersects torso_link2"}
            return {"opened": True, "position": 0.31}

        def return_to_ready(self, note="", allowed_contacts=None):
            return calls.append(("ready",)) or True

    class Ep:
        open_up = Episode.open_up

        def __init__(self):
            self.sim, self.spec, self.records = Sim(), None, []

        def stand_for(self, *names):
            calls.append(("stand", names))
            return {}

    ep = Ep()
    assert ep.open_up("cabinet.n.01_1") is True
    assert calls == [("open", True), ("ready",), ("stand", ("cabinet.n.01_1",)), ("open", False)]
    assert len(ep.records) == 2 and ep.records[1]["after_standoff_failed"]


def test_a_pressed_grasp_that_worked_is_entered_in_the_hand_record():
    """press_grasp physically takes a flat object -- close_on presses until the assist reports it holds -- but
    nothing wrote the robot's OWN hand record, which note_hands writes only after a PLANNER round. So holding()
    said False, Episode.pick threw the success away, and three tasks lost every flat-object pick (2026-09-15).

    The record is written by note_hands' rule: the knowledge source's localization, and the fingers only when
    nothing can localize it. Not from the simulator's grasp assist, which stays diagnostic (scene.check_hands).
    """
    import numpy as np

    from omnigibson.tiptop.bench import Episode

    class Sim:
        arm = "left"

        OPEN = 1.0

        def __init__(self, finger, hand=(1.0, 0.0, 0.8)):
            self.held_objects, self.finger, self.hand = {}, finger, np.array(hand)
            self.opened = []

        def hold(self, n, gripper=None):
            self.opened.append((n, gripper))

        tracked_label = staticmethod(lambda n: n.replace(".n.01_", "_"))

        def eef_pose_base(self, arm):
            m = np.eye(4)
            m[:3, 3] = self.hand
            return m

        base_to_world = staticmethod(lambda p: np.asarray(p, float))

        def grasp_sensed(self, arm):
            return self.finger > 0.006

        def hands(self):
            return dict(self.held_objects)

    class Knowledge:
        def __init__(self, center):
            self.center = center

        def localize(self, *names):
            if self.center is None:
                raise KeyError(names[0])
            c = np.asarray(self.center, float)
            return {n: {"center": c, "lo": c - 0.03, "hi": c + 0.03} for n in names}

    class Ep:
        note_pressed_grasp = Episode.note_pressed_grasp

        def __init__(self, sim, know):
            self.sim, self.knowledge = sim, know

    # localized at the hand: recorded, whatever the fingers say (sticky closes them through the object)
    ep = Ep(Sim(finger=0.0), Knowledge([1.02, 0.0, 0.78]))
    assert ep.note_pressed_grasp("book.n.01_1") is True
    assert ep.sim.hands() == {"book_1": "left"}, "a pressed grasp that worked must reach holding()"

    # localized far from the hand, fingers on SOMETHING: not recorded, and the hand is opened rather than left
    # shut on whatever else was under it -- the same hazard the planner path has (8c8ae6484)
    ep = Ep(Sim(finger=0.04), Knowledge([3.0, 0.0, 0.1]))
    assert ep.note_pressed_grasp("book.n.01_1") is False
    assert ep.sim.hands() == {}
    assert ep.sim.opened, "a hand shut on the wrong thing must be opened"

    # fingers on nothing: nothing to drop, so do not spend the steps
    ep = Ep(Sim(finger=0.0), Knowledge([3.0, 0.0, 0.1]))
    assert ep.note_pressed_grasp("book.n.01_1") is False
    assert ep.sim.opened == [], "an empty hand needs no opening"

    # nothing can localize it: the fingers decide, exactly as note_hands falls back
    ep = Ep(Sim(finger=0.04), Knowledge(None))
    assert ep.note_pressed_grasp("book.n.01_1") is True
    ep = Ep(Sim(finger=0.0), Knowledge(None))
    assert ep.note_pressed_grasp("book.n.01_1") is False



def test_the_pressed_grasp_no_longer_refuses_a_thick_object_out_of_hand():
    """It used to return without moving for anything thicker than FLAT_THICKNESS, 366 times across the corpus --
    pillows 48, a tissue dispenser 21, soda cans 37, the fax machine 14.

    By the time press_grasp runs the PLANNER HAS ALREADY FAILED on that object, so there is nothing else left to
    try; and sticky grasping needs no flatness at all, since OmniGibson skips both the antipodal raycast and the
    two-finger requirement in that mode (2026-09-15).
    """
    import inspect

    from omnigibson.tiptop.r1pro import R1ProSim

    body = inspect.getsource(R1ProSim.press_grasp).split('"""')[-1]
    assert "FLAT_THICKNESS" in body, "the thickness is still worth reporting"
    early = body[: body.index("top =")] if "top =" in body else body
    assert "return False" not in early, "but it must not refuse the attempt before the arm moves"


def test_the_pressed_grasp_tries_the_front_when_above_is_a_shelf():
    """A book standing in a shelf cannot be reached from above -- above it is the next shelf.

    boxing_books_up_for_storage refused every attempt with "the way down to it sweeps through
    bookcase_otwukr_2", and once the collision check reads the real mesh that refusal is CORRECT: the arm really
    would pass through the shelf. So the press tries a horizontal approach onto the face nearest the robot, which
    is how a person takes a book off a shelf. Sticky grasping needs one finger in contact, not a jaw around the
    whole width (2026-09-15).
    """
    import inspect

    from omnigibson.tiptop.r1pro import R1ProSim

    body = inspect.getsource(R1ProSim.press_grasp).split('"""')[-1]
    assert "from above" in body and "from the front" in body, "both ways in have to be tried"
    assert body.index("from above") < body.index("from the front"), "above first: it is right for anything lying flat"
    assert "into_dir" in body, "the press direction has to follow the approach, not stay hardcoded down"
    assert "close_on(" in body and "down," not in body.split("close_on(")[1][:80], (
        "close_on must press along the approach it actually took"
    )


def test_the_press_chooses_its_way_in_the_way_the_drawer_pull_does():
    """A pose being reachable is not the same as the arm being able to get to it.

    sorting_books_on_shelf: the front approach SOLVED, and the straight joint-space line to it swept
    bookcase_otwukr_1 -- a different bookcase from the one holding the book. reach_plan offers five ways in
    (straight, via the ready posture, torso first, arm first, elbow first) and the drawer pull has used it since
    2026-09-14.

    It is used here WITHOUT its fallback. reach_plan returns its least-sweeping plan rather than refusing, which
    is right for a drawer; sorting_books_on_shelf is a stacking-order task where an arm swung through a bookcase
    knocks the order about, so a dirty plan is still refused (2026-09-15).
    """
    import inspect

    from omnigibson.tiptop.r1pro import R1ProSim

    body = inspect.getsource(R1ProSim.press_grasp).split('"""')[-1]
    assert "reach_plan(" in body, "five ways in, not one straight line"
    after = body[body.index("reach_plan(") :]
    assert "path_hits_scene" in after, "and the chosen plan is still checked"
    assert "continue" in after[: after.index("approaching the sticky precontact")], "a dirty plan is still refused"
    assert "_targets_from" in after, "every leg of the chosen plan is ramped, not just the last"


def test_pressed_grasp_keeps_supports_and_stops_after_a_rejected_approach():
    from types import SimpleNamespace

    import torch as th
    import trimesh

    from omnigibson.tiptop.r1pro import R1ProSim

    book = SimpleNamespace(name="book", aabb=(th.tensor([0.5, 0.0, 0.7]), th.tensor([0.65, 0.1, 0.72])))
    calls = []

    def reach(*args, **kwargs):
        calls.append(("reach", kwargs))
        return [[0.2]]

    def ramp(*args, **kwargs):
        calls.append(("ramp", kwargs, args))
        if len(calls) == 1:  # closing the hand where it is, before the approach
            return None
        return ("right_realsense_link intersects bookcase", 0, 0.0)

    sim = SimpleNamespace(
        scene_object=lambda name: book,
        to_base=lambda pos, quat: (pos, quat),
        base_pose=lambda: (th.zeros(3), th.tensor([0.0, 0.0, 0.0, 1.0])),
        collision_mesh_world=lambda obj: trimesh.creation.box([0.15, 0.1, 0.02], transform=np.array(
            [[1, 0, 0, 0.575], [0, 1, 0, 0.05], [0, 0, 1, 0.71], [0, 0, 0, 1]])),
        arm_ik=lambda *args, **kwargs: object(),
        ik_joint_names=lambda *args, **kwargs: ["left_arm_joint1"],
        robot=SimpleNamespace(get_joint_positions=lambda: th.zeros(1)),
        joint_index={"left_arm_joint1": 0},
        scene_aabbs=lambda: [],
        _press_solution=lambda *args, **kwargs: [0.2],
        grasp_target=lambda *args, **kwargs: (np.zeros(3), np.eye(3)),
        reach_plan=reach,
        path_hits_scene=lambda *args, **kwargs: [],
        _targets_from=lambda joints, target: target,
        ramp_to=ramp,
        grasp_contacts=R1ProSim.grasp_contacts,
        posture={},
        OPEN=1.0,
        CLOSE=-1.0,
        q_arm=lambda: np.zeros(1),
    )
    # No close_on method: a failed approach must return without pressing deeper.
    assert R1ProSim.press_grasp(sim, "left", "book", spare=("bookcase",)) is False
    assert calls[0][0] == "ramp" and calls[0][2][2] == sim.CLOSE  # closed first, in free space
    assert calls[1][1]["exclude"] == {"book"}
    assert calls[2][2][2] == sim.CLOSE  # the transit travels closed
    assert not calls[2][1].get("allowed_contacts")  # and keeps the target collidable


# --------------------------------------------------------------- container_grasps and the pull (F-revopen)
def _door(top_heavy=False):
    """A 2 cm panel hinged on a vertical axis through the origin, shut along +y, z 0..1, as the link's world mesh.
    ``top_heavy`` packs vertices into its front layer near the top the way a bevelled door's mesh does (gjeoer's
    vertex mean sits 26 cm above mid-height, edge_on_assets.out)."""
    import trimesh

    panel = trimesh.creation.box([0.02, 0.5, 1.0]).apply_translation([0.0, 0.25, 0.5])
    if top_heavy:
        strip = trimesh.creation.box([0.008, 0.4, 0.1]).apply_translation([-0.006, 0.25, 0.9])  # inside the front layer
        for _ in range(3):
            strip = strip.subdivide()
        panel = trimesh.util.concatenate([panel, strip])
    return panel


def _door_sim(mesh, grasping_mode):
    from types import MethodType, SimpleNamespace

    import torch as th

    from omnigibson.tiptop.r1pro import R1ProSim

    lo, hi = mesh.bounds
    link = SimpleNamespace(name="door", aabb=(th.tensor(lo, dtype=th.float32), th.tensor(hi, dtype=th.float32)))
    sim = SimpleNamespace(link_trimesh_world=lambda link, collision_only=False: mesh, robot=SimpleNamespace(grasping_mode=grasping_mode),
                          edge_roofed=lambda obj, moving, point, direction: None)  # fmt: skip
    sim.surface_point = MethodType(R1ProSim.surface_point, sim)
    return sim, SimpleNamespace(name="cabinet", links={"door": link})


def test_the_grips_offered_on_a_door_come_from_its_edge_not_its_vertex_mean_and_weld_under_assisted():
    """T6 / ART-1 / ART-2: under assisted grasping only the edge grip is offered on a flat door (a pressed face has
    no second finger and no ray between the pads); it pinches EDGE_INSET below the link's TOP vertex, the jaw
    EDGE_THICKNESS behind the face, coming DOWN over the panel (approach up) rather than in along the lead through
    it. handle_on's face_centre is a vertex mean: 26 cm above mid-height on gjeoer's door it put the old edge grip
    in the air. A door's travel is clipped to DOOR_TRAVEL_MAX; a lid asks for LID_FRACTION and, since a lid short
    of balance falls shut, is a plan only when the whole pull solves (min_fraction)."""
    from omnigibson.tiptop.r1pro import (
        DOOR_TRAVEL_MAX, EDGE_INSET, EDGE_THICKNESS, LID_FRACTION, OPEN_MIN_FRACTION, R1ProSim,
    )  # fmt: skip

    door = dict(name="j_door", kind="revolute", axis=[0.0, 0.0, 1.0], origin=[0.0, 0.0, 0.0], lower=0.0, upper=2.4,
                position=0.0, closed=0.0, link="door")  # fmt: skip
    hand = np.array([0.0, 0.25, 0.9])
    for top_heavy in (False, True):
        sim, obj = _door_sim(_door(top_heavy), "assisted")
        grasps = R1ProSim.container_grasps(sim, obj, [door], 0.9, hand)
        assert [g["kind"] for g in grasps] == ["edge"]
        edge = grasps[0]
        assert edge["tips"][2] == pytest.approx(1.0 - EDGE_INSET, abs=1e-3), "below the TOP vertex, not the mean"
        assert abs(edge["tips"][0]) < EDGE_THICKNESS and 0.05 < edge["tips"][1] < 0.45, "the jaw inside the panel"
        assert np.allclose(edge["into"], [0.0, 0.0, -1.0]) and np.allclose(edge["approach"], [0.0, 0.0, 1.0])
        assert np.allclose(edge["lead"], [-1.0, 0.0, 0.0], atol=0.05) and np.allclose(edge["jaws"], [edge["lead"], -edge["lead"]])
        assert edge["travel"] == pytest.approx(DOOR_TRAVEL_MAX) and edge["min_fraction"] == OPEN_MIN_FRACTION
        # the push column too: on the door, centred on its extent, whatever the mesh's vertex mean
        pushes = R1ProSim.push_grasps(sim, obj, door, 1.5, hand)
        heights = [g["tips"][2] for g in pushes]
        assert all(0.0 <= z <= 1.0 for z in heights) and np.mean(heights) == pytest.approx(0.5, abs=0.02)
    sim, obj = _door_sim(_door(), "sticky")
    grasps = R1ProSim.container_grasps(sim, obj, [door], 0.9, hand)
    assert [g["kind"] for g in grasps][-1] == "edge" and {g["kind"] for g in grasps[:-1]} == {"flat"}
    assert all(g["tips"][0] == pytest.approx(-0.01, abs=1e-3) and 0.0 < g["tips"][2] < 1.0 for g in grasps[:-1])
    assert all("approach" not in g and g["min_fraction"] == OPEN_MIN_FRACTION for g in grasps[:-1])
    lid = dict(door, name="j_lid", axis=[0.0, 1.0, 0.0])  # a horizontal hinge
    (grasp,) = R1ProSim.container_grasps(_door_sim(_door(), "assisted")[0], obj, [lid], 0.5, hand)
    assert grasp["travel"] == pytest.approx(LID_FRACTION * 2.4) and grasp["min_fraction"] == LID_FRACTION


def test_the_standoff_lies_along_the_grasps_approach_and_a_lid_is_pulled_whole_or_not_at_all():
    """ART-1 / ART-4: solve_pull puts the standoff OPEN_APPROACH out along the grasp's approach (up for the edge
    grip; its lead otherwise) and hands the executed motions that direction; a partial pull is kept from
    min_fraction of the range, which for a lid is the whole pull (10 of 15 waypoints of the car trunk's pull, 57%,
    was kept and released below its 78% balance)."""
    from types import SimpleNamespace

    from omnigibson.tiptop.r1pro import LID_FRACTION, OPEN_APPROACH, OPEN_MIN_FRACTION, R1ProSim

    asked, solves = [], [99]

    def solve(pos, quat, seed=None, tolerance_pos=0.0, tolerance_rad=0.0):
        if len(asked) >= solves[0]:
            return None
        asked.append(np.array(pos, dtype=float))
        return [0.0]

    pull = [pose_matrix([0.5 - 0.02 * i, 0.0, 0.9], [0.0, 0.0, 0.0, 1.0]) for i in range(16)]
    sim = SimpleNamespace(
        _grasp_pose_base=lambda arm, g, jaw, base_pose=None: (pull[0][:3, 3], pull[0][:3, :3]),
        _pull_poses=lambda g, start, base_pose=None: (pull, None, None), container_body=lambda obj, link: None,
        arm_hits_scene=lambda *a, **k: [], body_hits=lambda *a, **k: False, ramp_refusal=lambda *a, **k: "",
    )  # fmt: skip
    drawer = dict(kind="prismatic", lower=0.0, upper=0.4, position=0.0)
    edge = dict(joint=drawer, travel=0.3, kind="edge", lead=[-1.0, 0.0, 0.0], approach=[0.0, 0.0, 1.0], min_fraction=OPEN_MIN_FRACTION)  # fmt: skip
    solve_pull = lambda g: R1ProSim.solve_pull(sim, SimpleNamespace(solve=solve), g, [0, 0, 1], [0.0], base_pose=(0.0, 0.0, 0.0), aabbs=[])  # fmt: skip
    plan, _ = solve_pull(edge)
    assert asked[0] == pytest.approx([0.5, 0.0, 0.9 + OPEN_APPROACH]) and plan["approach"] == pytest.approx([0.0, 0.0, 1.0])
    asked.clear()
    plan, _ = solve_pull({k: v for k, v in edge.items() if k != "approach"} | {"kind": "bar"})
    assert asked[0] == pytest.approx([0.5 - OPEN_APPROACH, 0.0, 0.9]) and plan["approach"] == pytest.approx([-1.0, 0.0, 0.0])
    trunk = dict(kind="revolute", lower=0.0, upper=2.53, position=0.0)
    lid = dict(edge, joint=trunk, travel=LID_FRACTION * 2.53, min_fraction=LID_FRACTION)
    solves[0] = 2 + 10  # the standoff, the grasp and 10 of the 15 waypoints
    asked.clear()
    plan, why = solve_pull(lid)
    assert plan is None and "57% of the range" in why
    asked.clear()
    assert solve_pull(dict(lid, min_fraction=OPEN_MIN_FRACTION))[0]["reached"] == 10  # a door: kept, pushed on after
    solves[0] = 99
    asked.clear()
    assert solve_pull(lid)[0]["reached"] == 15


def test_a_pull_that_falls_open_past_its_target_is_not_pushed_back(monkeypatch):
    """ART-5: the push continuation is for a pull that stopped SHORT (a door's arc past any fixed stance). A lid
    released past balance falls open to its limit, 15% of its range beyond the 0.85 target, and was pushed back."""
    from types import SimpleNamespace

    import torch as th

    from omnigibson.tiptop.r1pro import LID_FRACTION, R1ProSim

    lid = dict(name="j_lid", kind="revolute", axis=[0.0, 1.0, 0.0], origin=[0.0, 0.0, 0.0], lower=0.0, upper=2.967,
               position=0.0, closed=0.0, link="lid")  # fmt: skip
    state = [lid]
    monkeypatch.setattr("omnigibson.tiptop.r1pro.openable_joints", lambda obj: [state[-1]])
    chosen = dict(joint=lid, travel=LID_FRACTION * 2.967, kind="edge")
    pushes = []
    sim = SimpleNamespace(
        scene_object=lambda name: SimpleNamespace(name="toolbox"), OPEN=1.0,
        robot=SimpleNamespace(eef_links={"left": SimpleNamespace(get_position_orientation=lambda: (th.zeros(3), None))}),
        container_grasps=lambda *a, **k: [chosen],
        _drive_joint=lambda *a, **k: {"grasp": chosen, "plan": {"reached": 15}, "waypoints": 15, "why": "", "held": True, "stance": (1.0, 2.0, 0.0)},
        push_joint=lambda arm, name, j, target: pushes.append(round(target, 3)) or {"reached": True, "position": target, "why": ""},
    )  # fmt: skip
    state.append(dict(lid, position=2.967))  # fell open to the limit after the release
    assert R1ProSim.open_container(sim, "left", "toolbox")["opened"] and pushes == []
    state.append(dict(lid, position=1.0))  # stopped short at 34%: pushed the rest of the way
    assert R1ProSim.open_container(sim, "left", "toolbox")["opened"] and pushes == [round(LID_FRACTION * 2.967, 3)]


def test_a_push_close_blocks_the_grasp_assist_around_the_whole_motion():
    """ART-3: under sticky grasping the assist welds what one closed finger touches for 0.3 s -- the door the hand
    pushes shut -- and nothing opens the hand after; the retreat then dragged the door back open. The push runs
    with the arm's grasp search blocked, and unblocked whatever ends it."""
    from types import SimpleNamespace

    import torch as th

    from omnigibson.tiptop.r1pro import R1ProSim

    calls = []
    joint = dict(name="j_door", lower=0.0, upper=1.6, position=1.2, closed=0.0, link="door")
    sim = SimpleNamespace(
        scene_object=lambda name: SimpleNamespace(name="fridge"),
        robot=SimpleNamespace(eef_links={"left": SimpleNamespace(get_position_orientation=lambda: (th.zeros(3), None))}),
        push_grasps=lambda obj, j, target, hand: [dict(joint=j)],
        block_grasping=lambda arm: calls.append(("block", arm)), unblock_grasping=lambda: calls.append("unblock"),
        _drive_joint=lambda *a, **k: calls.append("push") or {"why": "no stance"},
    )  # fmt: skip
    assert R1ProSim.push_joint(sim, "left", "fridge", joint, 0.0)["reached"] is False
    assert calls == [("block", "left"), "push", "unblock"]
    sim._drive_joint = lambda *a, **k: (_ for _ in ()).throw(RuntimeError("stance rejected"))
    with pytest.raises(RuntimeError):
        R1ProSim.push_joint(sim, "left", "fridge", joint, 0.0)
    assert calls[-1] == "unblock"
    sim.push_grasps = lambda *a: []
    assert R1ProSim.push_joint(sim, "left", "fridge", joint, 0.0)["reached"] is False and calls[-1] == "unblock"


def test_a_handle_beside_a_flush_neighbour_is_judged_by_the_hands_own_clearance_not_the_limbs():
    """freeze_fruit 301 (manip2): fridge dszchb's bar sits at its door's free edge and bottom_cabinet_fancyy_0,
    2 m tall, stands flush beside it and 4 cm proud of the door. At the grasp the outer finger's origin is 3.5 cm
    from the cabinet and the gripper 6.6 cm, and ARM_RADIUS refused 117 of 120 stances ("the arm at the grasp
    would be in bottom_cabinet_fancyy_0"). solve_pull now judges the hand's links as points within
    HAND_BODY_CLEARANCE, as body_hits judges them at the container; the limb keeps its radius (offline
    reproduction on the decrypted assets: a stance in 3 tries, 10 of 15 waypoints, 2026-09-25)."""
    from types import MethodType, SimpleNamespace

    import trimesh

    from omnigibson.tiptop.r1pro import HAND_BODY_CLEARANCE, HAND_LINKS, R1ProSim

    class Body:
        name = "neighbour"

    cabinet = trimesh.creation.box([1.0, 0.6, 2.0]).apply_translation([1.0, 0.0, 1.0])  # its face at x = 0.5
    points = {f"left_arm_link{i}": [0.1 + 0.04 * i, 0.3 - 0.1 * i, 1.4 - 0.05 * i] for i in range(1, 8)}  # the limb clear
    points.update({f"left_{s}": [0.434, 0.0, 1.1] for s in HAND_LINKS if "arm_link" not in s})  # 6.6 cm, as measured
    points["left_gripper_finger_link1"] = [0.465, 0.05, 1.05]  # the outer finger 3.5 cm from the cabinet's face
    ik = SimpleNamespace(fk=lambda q, name: (np.asarray(points[name], float), None))
    sim = SimpleNamespace(objects={}, hands=lambda: {}, send_room=False, base_to_world=lambda p: np.asarray(p, float),
                          robot=SimpleNamespace(arm_link_names={"left": [f"left_arm_link{i}" for i in range(1, 8)]}),
                          collision_mesh_world=lambda obj: cabinet)  # fmt: skip
    for name in ("_link_points", "_to_world", "arm_points", "held_points"):
        setattr(sim, name, MethodType(getattr(R1ProSim, name), sim))
    aabbs = [(Body(), cabinet.bounds[0], cabinet.bounds[1])]
    assert R1ProSim.arm_hits_scene(sim, "left", ik, [0.0], aabbs=aabbs) == ["neighbour"], "the limb's radius refuses it"
    assert R1ProSim.arm_hits_scene(sim, "left", ik, [0.0], aabbs=aabbs, hand_clearance=HAND_BODY_CLEARANCE) == []
    points["left_gripper_finger_link1"] = [0.495, 0.05, 1.05]  # 5 mm off the face: in it by any measure
    assert R1ProSim.arm_hits_scene(sim, "left", ik, [0.0], aabbs=aabbs, hand_clearance=HAND_BODY_CLEARANCE) == ["neighbour"]
    points["left_gripper_finger_link1"] = [0.465, 0.05, 1.05]
    points["left_arm_link6"] = [0.46, 0.2, 1.2]  # the wrist 4 cm off the face: the limb's radius still stands
    assert R1ProSim.arm_hits_scene(sim, "left", ik, [0.0], aabbs=aabbs, hand_clearance=HAND_BODY_CLEARANCE) == ["neighbour"]
    # and solve_pull asks for exactly that at every pose of the motion
    seen = []
    pull = [pose_matrix([0.5 - 0.02 * i, 0.0, 0.9], [0.0, 0.0, 0.0, 1.0]) for i in range(16)]
    spy = SimpleNamespace(
        _grasp_pose_base=lambda arm, g, jaw, base_pose=None: (pull[0][:3, 3], pull[0][:3, :3]),
        _pull_poses=lambda g, start, base_pose=None: (pull, None, None), container_body=lambda obj, link: None,
        arm_hits_scene=lambda *a, **k: seen.append(k.get("hand_clearance")) or [], body_hits=lambda *a, **k: False,
        ramp_refusal=lambda *a, **k: "",
    )  # fmt: skip
    grasp = dict(joint=dict(kind="prismatic", lower=0.0, upper=0.4, position=0.0), travel=0.3, kind="bar", lead=[-1.0, 0.0, 0.0])
    plan, _ = R1ProSim.solve_pull(spy, SimpleNamespace(solve=lambda *a, **k: [0.0]), grasp, [0, 0, 1], [0.0], base_pose=(0.0, 0.0, 0.0), aabbs=[])
    assert plan["reached"] == 15 and seen == [HAND_BODY_CLEARANCE] * 17


def test_a_panel_edge_under_a_rail_or_the_top_of_a_dome_is_nothing_to_pinch():
    """storing_food / wash_a_baseball_cap 301 (manip2): under assisted grasping a flat door offers only the edge
    grip, and every attempt spent 120 stance tries (2 min) learning it cannot be taken. rkgjer's and gjeoer's doors
    top out under the cabinet's own rail (3.6 cm of air; the wrist sits 7.8 cm above the tips) and then the
    countertop; ynwamu's porthole is a dome 20 cm deep whose "edge" point hangs in the air in front of it.
    container_grasps refuses both before any stance is tried; a free-standing panel keeps its edge grip."""
    from types import MethodType, SimpleNamespace

    import trimesh

    from omnigibson.tiptop.r1pro import R1ProSim

    door = dict(name="j_door", kind="revolute", axis=[0.0, 0.0, 1.0], origin=[0.0, 0.0, 0.0], lower=0.0, upper=2.4,
                position=0.0, closed=0.0, link="door")  # fmt: skip
    hand = np.array([0.0, 0.25, 0.9])

    def grips(panel, body=None, scene=()):
        sim, obj = _door_sim(panel, "assisted")
        sim.container_body = lambda obj, moving: body
        sim.scene_aabbs = lambda: [(o, o.mesh.bounds[0], o.mesh.bounds[1]) for o in scene]
        sim.collision_mesh_world = lambda o: o.mesh
        sim.edge_roofed = MethodType(R1ProSim.edge_roofed, sim)
        return [g["kind"] for g in R1ProSim.container_grasps(sim, obj, [door], 0.9, hand)]

    assert grips(_door()) == ["edge"], "a free-standing panel is pinched from above"
    rail = trimesh.creation.box([0.04, 0.6, 0.036]).apply_translation([0.0, 0.25, 1.018])  # the cabinet's rail over the door
    assert grips(_door(), body=rail) == []
    slab = trimesh.creation.box([0.6, 0.6, 0.036]).apply_translation([0.0, 0.25, 1.05])
    countertop = SimpleNamespace(name="countertop", category="countertop", mesh=slab)
    assert grips(_door(), scene=[countertop]) == []
    dome = trimesh.creation.box([0.2, 0.5, 1.0]).apply_translation([0.09, 0.25, 0.5])  # its leading face where the panel's was
    assert grips(dome) == [], "20 cm of material behind the point is not a panel between the fingers"


def test_openable_joints_reads_the_joint_frame_and_the_closed_end_off_a_live_object(monkeypatch):
    """T7 (F-reader): the world axis and hinge come from the parent link's pose composed with the joint's own
    localPos0 (times the instance scale) and localRot0 (OmniGibson's xyzw); the closed end is the upper limit for a
    joint the Open metadata lists with direction -1 (laptop_nvulcs, trash_can_ifzxzj), so resting there reads shut."""
    from types import SimpleNamespace

    import torch as th

    from omnigibson.tiptop.articulation import openable_joints
    from omnigibson.utils.constants import JointType

    quarter_xyzw = th.tensor([0.0, 0.0, np.sin(np.pi / 4), np.cos(np.pi / 4)])  # localRot0: a quarter turn about z
    joint = SimpleNamespace(joint_type=JointType.JOINT_REVOLUTE, lower_limit=0.0, upper_limit=2.4, axis="X",
                            body0="/World/laptop/base_link", body1="/World/laptop/lid",
                            local_position_0=th.tensor([0.3, 0.0, 0.5]), local_orientation_0=quarter_xyzw,
                            get_state=lambda: (th.tensor([2.4]), None, None))  # fmt: skip
    links = {name: SimpleNamespace(get_position_orientation=lambda: (th.tensor([1.0, 2.0, 0.0]), th.tensor([0.0, 0.0, 0.0, 1.0])))
             for name in ("base_link", "lid")}  # fmt: skip

    class Triples(list):  # the direction-annotated metadata: (joint_id, joint_name, direction), read by .items()
        def items(self):
            return list(self)

    laptop = SimpleNamespace(name="laptop", joints={"j_lid": joint}, links=links, scale=(2.0, 2.0, 2.0),
                             metadata={"openable_joint_ids": Triples([(0, "j_lid", -1)])})  # fmt: skip
    (j,) = openable_joints(laptop)
    assert np.allclose(j["axis"], [0.0, 1.0, 0.0], atol=1e-6), "the X letter turned by localRot0"
    assert np.allclose(j["origin"], [1.6, 2.0, 1.0]), "the parent's origin plus localPos0 at the instance's scale"
    assert j["closed"] == 2.4 and j["link"] == "lid" and j["kind"] == "revolute" and j["position"] == pytest.approx(2.4)
    assert not is_open(j["lower"], j["upper"], j["position"], closed=j["closed"]), "resting at its closed end"
    laptop.metadata = {}  # no direction metadata: the lower limit is the closed end
    assert openable_joints(laptop)[0]["closed"] == 0.0


def _bar_door_sim(grasping_mode="assisted"):
    """fridge/petcxr's right door as in test_a_vertical_bar_on_a_door_is_read_as_a_bar_with_a_horizontal_jaw, hinged
    on z through its far edge, with container_grasps' own helpers."""
    import trimesh

    door = trimesh.util.concatenate([trimesh.creation.box(bounds=[[0.40, -0.28, -1.0], [0.44, 0.28, 0.98]]).subdivide(),
                                     trimesh.creation.box(bounds=[[0.47, 0.20, -0.25], [0.50, 0.235, 0.736]]).subdivide()])  # fmt: skip
    sim, obj = _door_sim(door, grasping_mode)
    joint = dict(name="j_door", kind="revolute", axis=[0.0, 0.0, -1.0], origin=[0.42, -0.28, 0.0], lower=0.0,
                 upper=2.0, position=0.0, closed=0.0, link="door")  # fmt: skip
    return sim, obj, joint


def test_a_bar_is_approached_with_the_jaw_open_to_the_bar_and_the_ik_judges_the_hand_at_that_opening(monkeypatch):
    """storing_food 301 (manip2, 2026-09-25): fancyy's lower door has its bar under the wall oven, and the fingers,
    fully open 6.3 cm to each side, put the upper one in the oven on the approach from both stances tried
    ("left_gripper_finger_link1 intersects oven_ffitak_0"). A bar grasp now carries jaw_open, half the bar's width
    plus BAR_JAW_ROOM (finger joint metres); the IK the stance is solved with holds the fingers there, and the
    gripper command that puts them there is linear over the finger joint's range, as the smooth controller reads it."""
    from types import SimpleNamespace

    from omnigibson.tiptop.r1pro import BAR_JAW_ROOM, R1ProSim

    sim, obj, joint = _bar_door_sim()
    grasps = R1ProSim.container_grasps(sim, obj, [joint], 0.8, np.array([0.5, 0.2, 0.3]))
    assert grasps and {g["kind"] for g in grasps} == {"bar"}
    assert all(g["jaw_open"] == pytest.approx(0.035 / 2 + BAR_JAW_ROOM, abs=2e-3) for g in grasps), "the bar is 3.5 cm wide"
    built = []
    monkeypatch.setattr("omnigibson.tiptop.r1pro.ArmIK", lambda urdf, joints, fixed, frame: built.append(fixed) or fixed)
    names = ["left_arm_joint1", "left_gripper_finger_joint1", "left_gripper_finger_joint2", "right_gripper_finger_joint1"]
    fingers = {"left": names[1:3], "right": names[3:]}
    joints = {n: SimpleNamespace(lower_limit=0.0, upper_limit=0.05) for n in names[1:]}
    arm = SimpleNamespace(robot=SimpleNamespace(get_joint_positions=lambda: np.full(4, 0.05), urdf_path="r1pro.urdf",
                                                finger_joint_names=fingers, joints=joints, arm_joint_names={"left": names[:1]}),
                          joint_index={n: i for i, n in enumerate(names)}, urdf_joints=set(names), OPEN=1.0,
                          ik_joint_names=lambda arm, with_torso=False: names[:1])  # fmt: skip
    R1ProSim.arm_ik(arm, "left", frame="left_gripper_link", with_torso=True, fingers=0.0275)
    assert built[-1] == {names[1]: 0.0275, names[2]: 0.0275, names[3]: 0.05}, "the working hand only"
    R1ProSim.arm_ik(arm, "left", frame="left_gripper_link")
    assert built[-1][names[1]] == 0.05, "as the fingers stand, without an opening"
    assert R1ProSim.jaw_command(arm, "left", 0.0275) == pytest.approx(0.1)
    assert R1ProSim.jaw_command(arm, "left", None) == 1.0 and R1ProSim.jaw_command(arm, "left", 0.2) == 1.0


def test_the_finger_opening_is_what_the_ramps_preflight_judges_a_hand_beside_a_neighbour_by():
    """ramp_refusal asks the ramps' own collision model (the r1pro cuRobo spheres, the simulator's disabled pairs)
    about a motion before the robot stands where it will run, with the joints the IK does not solve held where the
    IK holds them: a block where the fully open finger would be refuses the hand at the stance's IK built fully open,
    and is clear of the same pose at a bar's jaw_open. The base pose is the stance's, not where the robot is."""
    from pathlib import Path
    from types import MethodType, SimpleNamespace

    import torch as th
    import trimesh
    import yaml

    from omnigibson.tiptop.r1pro import R1ProSim

    r1pro = Path(__file__).resolve().parents[2] / "datasets/omnigibson-robot-assets/models/r1pro"
    urdf = r1pro / "urdf/r1pro.urdf"
    names = [f"torso_joint{i}" for i in range(1, 5)] + [f"{s}_arm_joint{i}" for i in range(1, 8) for s in ("left", "right")]
    names += [f"{s}_gripper_finger_joint{i}" for s in ("left", "right") for i in (1, 2)]
    q = th.zeros(len(names))
    home = dict(zip([f"torso_joint{i}" for i in range(1, 5)] + [f"left_arm_joint{i}" for i in range(1, 8)],
                    [1.025, -1.45, -0.47, 0.0, -1.6312, 0.2636, -1.812, -1.4576, -0.0508, -0.3727, -1.3193]))  # fmt: skip
    for i, n in enumerate(names):
        q[i] = home.get(n, 0.05 if "finger" in n else 0.0)
    block = SimpleNamespace(name="oven", category="oven")
    sim = SimpleNamespace(
        robot=SimpleNamespace(urdf_path=str(urdf), get_joint_positions=lambda: q,
                              disabled_collision_pairs=yaml.safe_load((r1pro / "r1pro.yaml").read_text())["disabled_collision_pairs"],
                              finger_joint_names={s: [f"{s}_gripper_finger_joint1", f"{s}_gripper_finger_joint2"] for s in ("left", "right")},
                              trunk_joint_names=names[:4], arm_joint_names={"left": [f"left_arm_joint{i}" for i in range(1, 8)]}),
        joint_index={n: i for i, n in enumerate(names)}, urdf_joints=set(names), objects={}, hands=lambda: {},
        base_pose=lambda: (th.tensor([5.0, 5.0, 0.0]), th.tensor([0.0, 0.0, 0.0, 1.0])),  # far from the stance
        grasp_contacts=R1ProSim.grasp_contacts, scene_aabbs=lambda: [(block, *block.mesh.bounds)],
        collision_mesh_world=lambda obj: obj.mesh,
    )  # fmt: skip
    for name in ("_motion_collision_model", "_motion_obstacles", "arm_ik", "ik_joint_names", "ramp_refusal"):
        setattr(sim, name, MethodType(getattr(R1ProSim, name), sim))
    model = sim._motion_collision_model()
    stance = (1.0, 2.0, np.pi / 2)
    world = np.eye(4)
    world[:3, :3] = [[0.0, -1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 1.0]]
    world[:3, 3] = [1.0, 2.0, 0.0]
    finger = model.links == "left_gripper_finger_link1"
    at = model.centres([float(v) for v in q], model.links[finger], model.local_centres[finger])[0] @ world[:3, :3].T + world[:3, 3]
    block.mesh = trimesh.creation.box([0.012, 0.012, 0.012]).apply_translation(at)  # where the open finger's base is
    arm = [float(q[names.index(n)]) for n in sim.ik_joint_names("left", with_torso=True)]
    assert sim.ramp_refusal(sim.arm_ik("left", "left_gripper_link", True), arm, arm, stance) == \
        "left_gripper_finger_link1 intersects oven"  # fmt: skip
    assert sim.ramp_refusal(sim.arm_ik("left", "left_gripper_link", True, fingers=0.0175), arm, arm, stance) == ""
    assert sim.ramp_refusal(sim.arm_ik("left", "left_gripper_link", True), arm, arm) == "", "at (5, 5) it is far off"


def test_solve_pull_stops_where_the_ramps_preflight_would_refuse_the_motion():
    """fancyy (storing_food 301, manip2): the stance search took stances whose approach the ramps then refused for
    the idle right hand in the cabinet, the head in it, the elbow through the torso -- the whole robot, which
    solve_pull's arm polyline never saw. Every motion of the plan now goes through ramp_refusal: the approach with the
    container whole (the hand allowed onto it), each pull step with its moving link left out; the pull is cut
    where it is refused, like a waypoint without IK."""
    from types import SimpleNamespace

    from omnigibson.tiptop.r1pro import R1ProSim

    pull = [pose_matrix([0.5 - 0.02 * i, 0.0, 0.9], [0.0, 0.0, 0.0, 1.0]) for i in range(16)]
    asked = []

    def refusal(ik, q_from, q_to, base_pose=None, arm="left", obj=None, moving=None):
        asked.append((round(q_from[0], 1), round(q_to[0], 1), moving))
        return "right_gripper_link intersects cabinet" if round(q_to[0], 1) == 1.2 else ""

    solves = iter(np.arange(0.0, 1.7, 0.1))
    sim = SimpleNamespace(
        _grasp_pose_base=lambda arm, g, jaw, base_pose=None: (pull[0][:3, 3], pull[0][:3, :3]),
        _pull_poses=lambda g, start, base_pose=None: (pull, None, None), container_body=lambda obj, link: None,
        arm_hits_scene=lambda *a, **k: [], body_hits=lambda *a, **k: False, ramp_refusal=refusal,
    )  # fmt: skip
    grasp = dict(joint=dict(kind="prismatic", lower=0.0, upper=0.4, position=0.0, link="drawer"), travel=0.3, kind="bar",
                 lead=[-1.0, 0.0, 0.0])  # fmt: skip
    ik = SimpleNamespace(solve=lambda *a, **k: [float(next(solves))])
    plan, why = R1ProSim.solve_pull(sim, ik, grasp, [0, 0, 1], [0.0], base_pose=(0.0, 0.0, 0.0), aabbs=[],
                                    obj=SimpleNamespace(name="cabinet"))  # fmt: skip
    assert asked[0] == (0.0, 0.1, None), "standoff to grasp: the container whole"
    assert all(moving == "drawer" for _, _, moving in asked[1:]), "each pull step without the drawer that moves"
    assert plan["reached"] == 10 and why == "right_gripper_link intersects cabinet at pull waypoint 11"


def test_the_stance_search_takes_a_plan_that_pulls_half_the_way_at_once_and_a_shorter_one_only_at_the_end():
    """store_honey's slgzfc drawer (offline, 2026-09-25): with the ramps' preflight in solve_pull the first plan
    stopped at 5 of 15 waypoints (the idle hand into the base as the torso leans back) and the fifth pulled all 15."""
    from types import SimpleNamespace

    import torch as th

    from omnigibson.tiptop.r1pro import R1ProSim

    def search(reached, limit=120):
        plans = iter(reached)

        def solve_pull(*a, **k):
            n = next(plans, None)
            return (None, "no inverse kinematics for the grasp") if n is None else (dict(reached=n, fraction=n / 15, why=""), "")

        sim = SimpleNamespace(
            scene_aabbs=lambda: [], arm_ik=lambda *a, **k: None, ik_joint_names=lambda *a, **k: [], q_home=[],
            planned_joints=[], _footprint_free=lambda *a, **k: (True, "free", None), solve_pull=solve_pull,
            base_pose=lambda: (th.zeros(3), None),
        )  # fmt: skip
        grasp = dict(joint=dict(name="j"), lead=[-1.0, 0.0, 0.0], tips=[1.0, 0.0, 0.8], jaws=["up"])
        return R1ProSim.stance_for_grasp(sim, SimpleNamespace(name="cabinet"), [grasp], limit=limit)

    assert search([None, None, 5, None, 15])[0] == search([None, None, None, None, 15])[0], "5 of 15 passed over"
    assert search([None, 8])[0] == search([None, 8, 15])[0], "8 of 15 is half the way: taken at once"
    short = search([None, 3, None, 5, 4], limit=6)
    assert short[0] == search([None, None, None, 5], limit=6)[0], "only short plans: the furthest, once the tries are spent"
    assert search([], limit=6) == (None, None, None)


def _drive_sim(grasp, plan, again=None, branch=None):
    """_drive_joint's collaborators as fakes: a one-joint arm whose stance, reach, hold and pull all succeed and
    record what they were asked. ``branch``: where the reach leaves the arm (the planner's any-configuration path);
    ``again``: what solve_pull answers when asked from there."""
    from types import MethodType, SimpleNamespace

    import torch as th

    from omnigibson.tiptop.r1pro import R1ProSim

    q = th.tensor([plan["solutions"][0][0]])
    log = []

    def solve_pull(ik, g, jaw, seed, **kwargs):
        log.append(("solve", round(float(seed[0]), 2), ik))
        return (plan, "") if again is None or seed[0] != branch else again

    def planned_standoff(arm, ik, joints_of, q_standoff, name, gripper=None):
        log.append(("planned", gripper))
        if branch is not None:
            q[0] = branch
        return True

    ik = SimpleNamespace(fk=lambda q, frame: (np.zeros(3), np.array([0.0, 0.0, 0.0, 1.0])), solve=lambda *a, **k: [0.9])
    finger = SimpleNamespace(lower_limit=0.0, upper_limit=0.05)
    sim = SimpleNamespace(
        arm="left", other_arm="right", OPEN=1.0, CLOSE=-1.0, hands=lambda: {}, posture={}, q_home=None, planned_joints=[],
        joint_index={"left_arm_joint1": 0}, scene_aabbs=lambda: [], container_body=lambda obj, link: None,
        robot=SimpleNamespace(get_joint_positions=lambda: q, finger_joint_names={"left": ["f1", "f2"]}, joints={"f1": finger}),
        arm_ik=lambda arm, frame=None, with_torso=False, fingers=None: log.append(("ik", fingers)) or ik,
        ik_joint_names=lambda arm, with_torso=False: ["left_arm_joint1"],
        stance_for_grasp=lambda obj, grasps, arm="left", with_torso=True: ((1.0, 2.0, 0.0), grasp, grasp["jaws"][0]),
        place_robot=lambda *a, **k: None, hold=lambda n, g: log.append(("hold", g)), solve_pull=solve_pull,
        reach_plan=lambda *a, **k: [], planned_standoff=planned_standoff, _targets_from=lambda joints_of, q: list(q),
        ramp_to=lambda q_arm, posture, gripper, settle, note="", **k: log.append((note.split(" of ")[0], list(q_arm), gripper)),
        grasp_contacts=R1ProSim.grasp_contacts, tuck_idle_arm=lambda: False,
        close_on=lambda *a, **k: ([0.2], True), follow_pull=lambda *a: (1, ""), arm_hits_scene=lambda *a, **k: [],
    )  # fmt: skip
    sim.jaw_command = MethodType(R1ProSim.jaw_command, sim)
    return sim, log


def test_the_hand_comes_in_at_the_bars_jaw_and_lets_go_fully_open(monkeypatch):
    """The bar's jaw_open is what the fingers are commanded to from the stance on -- the reach, the planner's
    reach (which plans with the fingers as they are, and whose approach's preflight would otherwise sweep them
    from fully open over the whole path) and the approach -- so the ramps judge the hand the search judged. The
    hold is let go fully open: the assist reads any command short of OPEN as a grasp and never releases, and an
    early return must not leave that command behind for the next motion (open_container resets it)."""
    from types import SimpleNamespace

    import torch as th

    from omnigibson.tiptop.r1pro import R1ProSim

    grasp = dict(joint=dict(name="j_door", link="door", position=0.0), kind="bar", tips=[1.0, 0.0, 1.0], jaws=[[0, 1, 0]],
                 nudges=1, jaw_open=0.0275)  # fmt: skip
    plan = dict(solutions=[[0.1], [0.2], [0.3]], reached=1, why="", grasp_pose=np.eye(4), approach=np.array([-1.0, 0.0, 0.0]))
    sim, log = _drive_sim(grasp, plan)
    run = R1ProSim._drive_joint(sim, "left", SimpleNamespace(name="fridge"), [grasp], "fridge")
    jaw = 0.1  # -1 + 2 * 0.0275 / 0.05
    assert run["plan"] is plan and ("ik", 0.0275) in log and ("ik", None) not in log
    commands = [entry[-1] for entry in log if entry[0] in ("hold", "planned", "approach the handle")]
    assert commands[:3] == [pytest.approx(jaw)] * 3, "stance hold, planner's reach, approach: all at the jaw"
    assert ("hold", 1.0) in log and log[-1] == ("back off from fridge", [0.9], 1.0), "let go and back off fully open"
    push = dict(grasp, jaw_open=None)
    sim, log = _drive_sim(push, plan)
    R1ProSim._drive_joint(sim, "left", SimpleNamespace(name="fridge"), [push], "fridge", take_hold=False)
    assert {entry[-1] for entry in log if entry[0] in ("hold", "planned", "approach the handle", "back off from fridge")} == {-1.0}
    left = SimpleNamespace(last_gripper=jaw, OPEN=1.0, scene_object=lambda name: SimpleNamespace(name=name),
                           robot=SimpleNamespace(eef_links={"left": SimpleNamespace(get_position_orientation=lambda: (th.zeros(3), None))}),
                           container_grasps=lambda *a, **k: [grasp], _drive_joint=lambda *a, **k: {"why": "handle approach rejected"})  # fmt: skip
    monkeypatch.setattr("omnigibson.tiptop.r1pro.openable_joints", lambda obj: [grasp["joint"]])
    assert R1ProSim.open_container(left, "left", "fridge")["opened"] is False and left.last_gripper == 1.0


def test_the_approach_is_solved_again_from_where_the_reach_left_the_arm():
    """storing_food 301 (manip2): the straight reach was refused, the planner's "(any configuration)" path put the
    arm at the standoff in another IK branch, and the straight approach from there to the plan's grasp swept the
    elbow through the torso ("handle approach rejected: left_arm_link6 intersects torso_link4"). The grasp and pull
    are solved again from the measured posture and approached from there; the plan stands when that fails, and
    nothing is re-solved when the reach ended where the plan starts."""
    from types import SimpleNamespace

    from omnigibson.tiptop.r1pro import R1ProSim

    grasp = dict(joint=dict(name="j_door", link="door", position=0.0), kind="bar", tips=[1.0, 0.0, 1.0], jaws=[[0, 1, 0]],
                 nudges=1, jaw_open=0.0275)  # fmt: skip
    plan = dict(solutions=[[0.1], [0.2], [0.3]], reached=1, why="", grasp_pose=np.eye(4), approach=np.array([-1.0, 0.0, 0.0]))
    again = dict(plan, solutions=[[1.5], [1.6], [1.7]])
    sim, log = _drive_sim(grasp, plan, again=(again, ""), branch=1.5)
    run = R1ProSim._drive_joint(sim, "left", SimpleNamespace(name="fridge"), [grasp], "fridge")
    assert [e[1] for e in log if e[0] == "solve"] == [0.1, 1.5], "the stance's plan, then from the branch"
    assert run["plan"] is again and next(e[1] for e in log if e[0] == "approach the handle") == [1.6]
    sim, log = _drive_sim(grasp, plan, again=(None, "no inverse kinematics for the grasp"), branch=1.5)
    run = R1ProSim._drive_joint(sim, "left", SimpleNamespace(name="fridge"), [grasp], "fridge")
    assert run["plan"] is plan and next(e[1] for e in log if e[0] == "approach the handle") == [0.2], "the plan stands"
    sim, log = _drive_sim(grasp, plan)
    R1ProSim._drive_joint(sim, "left", SimpleNamespace(name="fridge"), [grasp], "fridge")
    assert [e[1] for e in log if e[0] == "solve"] == [0.1], "at the plan's standoff: nothing to solve again"
